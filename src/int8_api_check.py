"""Choose the INT8 dynamic-quantization API for this setup (torch 2.14, CPU, onednn).

Candidates (PyTorch docs: torch.ao.quantization is deprecated, migrate eager
quantize_dynamic to torchao's quantize_):
  A  torch.ao.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
  B  torchao.quantization.quantize_(model, Int8DynamicActivationInt8WeightConfig(
         reduce_range=torchao.utils.should_reduce_range(cpu)))   eager, no torch.compile
Both: int8 weights, activations quantized dynamically at run time, all nn.Linear
layers (including the classifier head).

Checked per candidate: runs?, layers converted, single-pass time, size on disk, agreement with
FP32 on 200 fixed images, and a short latency probe. The probe is only used to
choose the API; it is not a reported result (those come from latency.py).
Output: results/int8_api_check.json
"""
import json
import time
import warnings

import numpy as np
import timm
import torch
import torchao
from torch import nn

from bench import size_on_disk_mb, time_block
from class_mapping import load_mapping
from common import RESULTS_DIR, setup
from data import CachedImagenetteVal

MODEL = "deit_tiny_patch16_224"
N_IMAGES = 200
# A candidate whose eager single-image pass is this many times slower than FP32
# is recorded as too slow and not evaluated further.
TOO_SLOW = 10


def fp32_model():
    return timm.create_model(MODEL, pretrained=True).eval()


def quant_a(model):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        warnings.simplefilter("ignore", category=DeprecationWarning)
        return torch.ao.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)


def quant_b(model):
    from torchao.quantization import Int8DynamicActivationInt8WeightConfig, quantize_
    from torchao.utils import should_reduce_range
    quantize_(model, Int8DynamicActivationInt8WeightConfig(
        reduce_range=should_reduce_range(torch.device("cpu"))))
    return model


def count_converted(model) -> dict:
    types = {}
    for m in model.modules():
        if isinstance(m, nn.Linear) or "Linear" in type(m).__name__:
            name = f"{type(m).__module__}.{type(m).__name__}"
            if isinstance(m, nn.Linear) and hasattr(m, "weight"):
                name += f" (weight: {type(m.weight).__name__})"
            types[name] = types.get(name, 0) + 1
    return types


@torch.inference_mode()
def single_pass_ms(model, x1, n: int = 3) -> list[float]:
    """n single-image forward passes after one untimed pass (quick sanity timing)."""
    model(x1)
    out = []
    for _ in range(n):
        t0 = time.perf_counter()
        model(x1)
        out.append((time.perf_counter() - t0) * 1000)
    return out


@torch.inference_mode()
def logits(model, x):
    return torch.cat([model(x[i:i + 25]) for i in range(0, len(x), 25)])


def main() -> None:
    setup()
    from torchao.utils import _cpu_is_vnni_supported, should_reduce_range
    report = {
        "torch": torch.__version__, "torchao": torchao.__version__,
        "quantized_engine": torch.backends.quantized.engine,
        "cpu_vnni_supported (torchao check)": bool(_cpu_is_vnni_supported()),
        "reduce_range_used_for_B": bool(should_reduce_range(torch.device("cpu"))),
        "n_images": N_IMAGES,
    }
    base = fp32_model()
    cfg = timm.data.resolve_data_config({}, model=base)
    ds = CachedImagenetteVal(cfg)
    idx = np.linspace(0, len(ds) - 1, N_IMAGES).astype(int)
    x = torch.stack([ds[i][0] for i in idx])
    labels = torch.tensor([ds[i][1] for i in idx])
    in_idx = torch.tensor([ds[i][2] for i in idx])
    sub = torch.tensor([r["imagenet_index"] for r in load_mapping()])
    y_fp32 = logits(base, x)

    def summarize(y):
        return {
            "max_abs_logit_diff_vs_fp32": (y - y_fp32).abs().max().item(),
            "mean_abs_logit_diff_vs_fp32": (y - y_fp32).abs().mean().item(),
            "top1_agreement_vs_fp32_unrestricted": (y.argmax(1) == y_fp32.argmax(1)).float().mean().item(),
            "top1_agreement_vs_fp32_restricted": (y[:, sub].argmax(1) == y_fp32[:, sub].argmax(1)).float().mean().item(),
            "top1_restricted_on_subset": (y[:, sub].argmax(1) == labels).float().mean().item(),
            "top1_unrestricted_on_subset": (y.argmax(1) == in_idx).float().mean().item(),
        }

    x1 = x[:1]
    fp32_single = single_pass_ms(base, x1)
    report["fp32"] = {**summarize(y_fp32), "size_on_disk_mb": size_on_disk_mb(base),
                      "single_pass_ms": fp32_single}
    candidates = {"A_torch_ao_quantize_dynamic": quant_a, "B_torchao_quantize_": quant_b}
    models = {"fp32": base}
    for name, fn in candidates.items():
        entry = {}
        try:
            t0 = time.perf_counter()
            q = fn(fp32_model())
            entry["quantize_seconds"] = time.perf_counter() - t0
            entry["modules"] = count_converted(q)
            entry["single_pass_ms"] = single_pass_ms(q, x1)
            if min(entry["single_pass_ms"]) > TOO_SLOW * max(fp32_single):
                entry["runs"] = True
                entry["skipped"] = (f"eager forward too slow (> {TOO_SLOW}x FP32); "
                                    f"agreement and latency probe not run")
                report[name] = entry
                print(f"{name}: skipped, single passes {entry['single_pass_ms']} ms", flush=True)
                continue
            entry.update(summarize(logits(q, x)))
            try:
                entry["size_on_disk_mb"] = size_on_disk_mb(q)
            except Exception as e:  # noqa: BLE001
                entry["size_on_disk_mb"] = f"failed: {type(e).__name__}: {e}"
            entry["runs"] = True
            models[name] = q
        except Exception as e:  # noqa: BLE001
            entry = {"runs": False, "error": f"{type(e).__name__}: {e}"}
        report[name] = entry

    # Latency probe (choice only): 30 s heat-up, then 2 rounds of 1 block each.
    with torch.inference_mode():
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < 30:
            base(x1)
    probe = {k: [] for k in models}
    for _ in range(2):
        for k, m in models.items():
            probe[k].append(time_block(m, x1)["latency_median_ms"])
    report["latency_probe_block_medians_ms (choice only, not a result)"] = probe

    out = RESULTS_DIR / "int8_api_check.json"
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
