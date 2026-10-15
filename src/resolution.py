"""Experiment 3: resolution scaling on KITTI camera images (no labels, no accuracy).

Images: KITTI raw drive 2011_09_26_drive_0001_sync, left colour camera (image_02),
108 frames of 1242x375 px. The frames are consecutive (10 Hz), so they are highly
correlated; they are not independent samples.

Resolutions keep KITTI's aspect ratio: height H from HEIGHTS, width = H * 1242/375
rounded to the nearest multiple of 16 (the patch size). 368x1216 is the closest to
the original frame (slight downscale); larger sizes are upscaled and only measure
how cost grows with the number of tokens. Preprocessing: the whole frame is resized
to (H, W) with the model's timm interpolation (bicubic, antialias), no crop, then
the model's mean/std normalization. The model uses timm's dynamic_img_size=True:
the 14x14 position embedding is interpolated (bicubic) to the patch grid on every
forward pass, so that cost is included in the latency.

Methods per resolution (see models.py):
  fp32          timm, fused scaled_dot_product_attention
  int8          INT8 dynamic quantization of nn.Linear, attention as fp32
  tome_r0       ToMe code with no merging: explicit softmax(QK^T)V attention. Control
                that separates the attention-kernel effect from merging.
  tome_p8       ToMe with r scaled to the input, r = round(8 * patches / 196): the same
                fraction of tokens per block as r = 8 at 224 (no top-1 loss on Imagenette).
  tome_p8_int8  tome_p8, then INT8

Limits (CLAUDE.md; checked before every configuration, skips are recorded, not hidden):
  memory: estimated largest single tensor > 1.5 GB, or estimated peak > 70% of the
          RAM available at that moment -> "skipped: estimated memory too high".
  time:   one forward pass > TIME_LIMIT_S (or extrapolated quadratically from the
          previous size to > 2x that, then not even tried) -> "skipped: time limit".
          Once a method is skipped, all larger sizes of that method are skipped.

Subcommands (run in this order; memory and agreement only run what latency ran):
  latency    heat-up, then interleaved FP32 reference blocks (rule B drift check,
             see NOTES.md) and one adaptive block per configuration.
  memory     peak memory, each configuration saved/verified in one fresh process and
             measured in another (memory_worker.py).
  agreement  top-1 agreement (1000 ImageNet classes) with FP32 at the same
             resolution over AGREEMENT_FRAMES. Only a stability proxy, not accuracy.

Outputs in results/resolution/: latency.csv, memory.csv, agreement.csv, raw/ (per run),
predictions/ (per frame).
"""
import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import psutil
import torch
from PIL import Image
from timm.data import ImageNetInfo, str_to_interp_mode
from torchvision import transforms

import models
from accuracy import upsert
from bench import (DRIFT_THRESHOLD, drift_check_rolling, now, run_memory_worker,
                   save_and_verify, size_on_disk_mb, system_state, time_block, token_counts)
from common import DATA_DIR, NUM_THREADS, RESULTS_DIR, setup

MODEL = "deit_tiny_patch16_224"
FRAMES_DIR = DATA_DIR / "kitti" / "2011_09_26" / "2011_09_26_drive_0001_sync" / "image_02" / "data"
KITTI_W, KITTI_H = 1242, 375
PATCH = 16
HEIGHTS = [64, 128, 192, 256, 368, 512, 736, 1024]
METHODS = ["fp32", "int8", "tome_r0", "tome_p8", "tome_p8_int8"]
BASELINE = "fp32"
REFERENCE_HEIGHT = 128  # FP32 reference blocks for the drift check run at this size
LATENCY_FRAME = 0       # one fixed frame for latency and memory
AGREEMENT_FRAMES = list(range(0, 108, 4))  # 27 frames, fixed

# Timing protocol. Default 10 warm-up + 50 timed runs; reduced only when one pass is
# so slow that this would exceed the budgets below (numbers used are recorded).
WARMUP_MAX, WARMUP_MIN, WARMUP_BUDGET_S = 10, 2, 20.0
RUNS_MAX, RUNS_MIN, TIMED_BUDGET_S = 50, 5, 60.0
HEATUP_SECONDS = 180.0
TIME_LIMIT_S = 30.0          # single forward pass
AGREEMENT_BUDGET_S = 600.0   # per configuration and resolution

# Memory estimate (bytes are float32). Explicit attention keeps the scores, their
# softmax and (ToMe proportional attention) the scores + log(size) alive at once:
# assume 3 copies of heads x N x N. Fused SDPA does not build N x N; its largest
# tensor is the MLP hidden activation N x 4*dim, assume 4 of those.
MAX_TENSOR_MB = 1.5 * 1024
RAM_FRACTION = 0.7
RUNTIME_MB = 350.0  # Python + torch + timm + loaded model (memory.csv: ~331 MB at 224)
HEADS, DIM = 3, 192
MB = 1024**2

OUT_DIR = RESULTS_DIR / "resolution"


def sizes() -> list[tuple[int, int]]:
    return [(h, PATCH * math.floor(h * KITTI_W / KITTI_H / PATCH + 0.5)) for h in HEIGHTS]


def n_patches(h: int, w: int) -> int:
    return (h // PATCH) * (w // PATCH)


def frame_paths() -> list:
    paths = sorted(FRAMES_DIR.glob("*.png"))
    if len(paths) != 108:
        raise RuntimeError(f"expected 108 KITTI frames in {FRAMES_DIR}, found {len(paths)}")
    return paths


def load_frame(path, h: int, w: int, data_config: dict) -> torch.Tensor:
    """Whole frame resized to (h, w), no crop; model's interpolation and mean/std."""
    t = transforms.Compose([
        transforms.Resize((h, w), interpolation=str_to_interp_mode(data_config["interpolation"]),
                          antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(data_config["mean"], data_config["std"]),
    ])
    with Image.open(path) as im:
        return t(im.convert("RGB")).unsqueeze(0)


def build(method: str, h: int, w: int):
    return models.build(MODEL, method, dynamic_img_size=True, n_patches=n_patches(h, w))


def explicit_attention(method: str) -> bool:
    return models.parse_method(method)["tome_r"] is not None


def memory_estimate(method: str, h: int, w: int) -> dict:
    """Estimated memory of one configuration, and whether it may run now."""
    n = n_patches(h, w) + 1  # tokens entering the first block (ToMe merges after attention)
    if explicit_attention(method):
        largest = HEADS * n * n * 4 / MB
        extra = 3 * largest
    else:
        largest = n * 4 * DIM * 4 / MB
        extra = 4 * largest
    available = psutil.virtual_memory().available / MB
    total = RUNTIME_MB + extra
    reason = None
    if largest > MAX_TENSOR_MB:
        reason = (f"skipped: estimated memory too high (largest tensor {largest:.0f} MB "
                  f"> {MAX_TENSOR_MB:.0f} MB)")
    elif total > RAM_FRACTION * available:
        reason = (f"skipped: estimated memory too high (peak {total:.0f} MB > "
                  f"{RAM_FRACTION:.0%} of available RAM {available:.0f} MB)")
    return {"est_largest_tensor_mb": largest, "est_peak_mb": total,
            "available_ram_mb_before": available, "skip_reason": reason}


def base_row(method: str, h: int, w: int) -> dict:
    return {"model": MODEL, "method": method, "height": h, "width": w,
            "input_size": f"{h}x{w}", "n_patches": n_patches(h, w),
            "tome_r_per_block": getattr_r(method, h, w)}


def getattr_r(method: str, h: int, w: int):
    spec = models.parse_method(method)
    if spec["tome_r"] is None:
        return None
    return models.scaled_r(spec["tome_r"], n_patches(h, w)) if spec["tome_scaled"] else spec["tome_r"]


# ---------------------------------------------------------------- latency
@torch.inference_mode()
def adaptive_block(model, x, first_pass_s: float) -> dict:
    """10 warm-up + 50 timed runs, fewer only if one pass is too slow for the budgets.
    The first (already done) pass counts as one warm-up run."""
    warmup = max(WARMUP_MIN, min(WARMUP_MAX, int(WARMUP_BUDGET_S / first_pass_s)))
    runs = max(RUNS_MIN, min(RUNS_MAX, int(TIMED_BUDGET_S / first_pass_s)))
    r = time_block(model, x, warmup=warmup - 1, runs=runs)
    r["warmup"] = warmup
    return r


@torch.inference_mode()
def timed_first_pass(model, x) -> float:
    t0 = time.perf_counter()
    model(x)
    return time.perf_counter() - t0


def cmd_latency(args) -> None:
    setup()
    paths = frame_paths()
    state_before = system_state()
    run_id = state_before["timestamp"].replace(":", "-")
    print(f"run {run_id}  idle CPU {state_before['idle_cpu_percent']:.1f}%  "
          f"AC power {state_before['on_ac_power']}  "
          f"available RAM {psutil.virtual_memory().available / MB:.0f} MB")

    ref_h, ref_w = REFERENCE_HEIGHT, dict(sizes())[REFERENCE_HEIGHT]
    ref_model, cfg = build(BASELINE, ref_h, ref_w)
    ref_x = load_frame(paths[LATENCY_FRAME], ref_h, ref_w, cfg)

    t0, n = time.perf_counter(), 0
    heat_start = now()
    with torch.inference_mode():
        while time.perf_counter() - t0 < args.heatup_seconds:
            ref_model(ref_x)
            n += 1
    heatup = {"start_time": heat_start, "end_time": now(), "seconds": args.heatup_seconds,
              "forward_passes": n, "input_size": f"{ref_h}x{ref_w}"}
    print(f"heat-up: {n} untimed passes at {ref_h}x{ref_w} in {args.heatup_seconds:.0f} s")

    blocks, runs, rows = [], [], []

    def reference_block():
        r = time_block(ref_model, ref_x)
        times = r.pop("times_ms")
        i = len(blocks)
        blocks.append({"run_id": run_id, "block": i, "kind": "reference", "method": BASELINE,
                       "input_size": f"{ref_h}x{ref_w}", **r})
        runs.extend({"run_id": run_id, "block": i, "iter": k, "latency_ms": t}
                    for k, t in enumerate(times))
        print(f"block {i:3d} reference {ref_h}x{ref_w} median {r['latency_median_ms']:8.2f} ms")

    reference_block()
    last_first_pass = {}  # method -> (n_tokens, seconds) at the previous size
    stopped = {}          # method -> reason; larger sizes are skipped too
    for h, w in [s for s in sizes() if s[0] in args.heights]:
        x = load_frame(paths[LATENCY_FRAME], h, w, cfg)
        n_tok = n_patches(h, w) + 1
        for method in args.methods:
            row = {**base_row(method, h, w), "run_id": run_id, "time": now(), "status": "ok"}
            est = memory_estimate(method, h, w)
            row.update(est)
            reason = stopped.get(method) and f"skipped: a smaller size was already skipped ({stopped[method]})"
            reason = reason or est["skip_reason"]
            if reason is None and method in last_first_pass:
                n_prev, t_prev = last_first_pass[method]
                t_est = t_prev * (n_tok / n_prev) ** 2
                row["est_first_pass_s"] = t_est
                if t_est > 2 * TIME_LIMIT_S:
                    reason = (f"skipped: time limit (estimated pass {t_est:.0f} s "
                              f"> {2 * TIME_LIMIT_S:.0f} s, not tried)")
            if reason is None:
                model, _ = build(method, h, w)
                t1 = timed_first_pass(model, x)
                row["first_pass_s"] = t1
                last_first_pass[method] = (n_tok, t1)
                if t1 > TIME_LIMIT_S:
                    reason = f"skipped: time limit (first pass {t1:.1f} s > {TIME_LIMIT_S:.0f} s)"
                else:
                    reference_block()
                    r = adaptive_block(model, x, t1)
                    times = r.pop("times_ms")
                    i = len(blocks)
                    blocks.append({"run_id": run_id, "block": i, "kind": "config", "method": method,
                                   "input_size": f"{h}x{w}", **r})
                    runs.extend({"run_id": run_id, "block": i, "iter": k, "latency_ms": t}
                                for k, t in enumerate(times))
                    tc = token_counts(model, x)
                    row.update({
                        "block": i, "warmup": r["warmup"], "runs": r["runs"],
                        "latency_median_ms": r["latency_median_ms"],
                        "latency_p95_ms": r["latency_p95_ms"],
                        "latency_min_ms": r["latency_min_ms"],
                        "latency_max_ms": r["latency_max_ms"],
                        "throughput_img_per_s": r["throughput_img_per_s"],
                        **tc,
                        "tokens_in_expected": n_tok,
                        "tokens_verified": tc["tokens_in"] == n_tok,
                        "size_on_disk_mb": size_on_disk_mb(model),
                    })
                    print(f"block {i:3d} {method:13s} {h}x{w} tokens {tc['tokens_in']:5d} -> "
                          f"{tc['tokens_out']:5d}  median {r['latency_median_ms']:9.2f} ms  "
                          f"({r['warmup']} warm-up, {r['runs']} runs)")
                del model
            if reason is not None:
                row["status"] = reason
                stopped.setdefault(method, reason)
                print(f"          {method:13s} {h}x{w} {reason}")
            rows.append(row)
    reference_block()
    del ref_model

    blocks_df, runs_df = pd.DataFrame(blocks), pd.DataFrame(runs)
    ref = blocks_df[blocks_df.kind == "reference"].latency_median_ms.tolist()
    drift = drift_check_rolling(ref)
    for row in rows:
        row.update({"torch_threads": torch.get_num_threads(),
                    "idle_cpu_percent_before": state_before["idle_cpu_percent"],
                    "reference_drift_spread": drift["relative_spread"],
                    "reference_raw_spread": drift["raw_spread_old_rule"],
                    "n_ignored_reference_spikes": drift["n_ignored_spikes"]})
    meta = {"run_id": run_id, "model": MODEL, "methods": args.methods, "heights": args.heights,
            "frame": paths[LATENCY_FRAME].name, "reference_input_size": f"{ref_h}x{ref_w}",
            "system_state_before": state_before, "system_state_after": system_state(),
            "heatup": heatup, "drift": drift, "end_time": now()}

    raw = args.out / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    blocks_df.to_csv(raw / f"{run_id}_latency_blocks.csv", index=False)
    runs_df.to_csv(raw / f"{run_id}_latency_runs.csv", index=False)
    pd.DataFrame(rows).to_csv(raw / f"{run_id}_latency_summary.csv", index=False)
    (raw / f"{run_id}_latency_meta.json").write_text(json.dumps(meta, indent=2))

    print(f"\nreference block medians (ms): {[round(m, 2) for m in ref]}")
    print(f"rolling-median spread {drift['relative_spread']:.1%} (threshold {DRIFT_THRESHOLD:.0%}); "
          f"raw spread {drift['raw_spread_old_rule']:.1%}; ignored spikes {drift['n_ignored_spikes']}")
    if drift["drift_detected"]:
        print("DRIFT DETECTED: summary NOT written to latency.csv. Raw logs kept.")
    else:
        for row in rows:
            upsert(args.out / "latency.csv", row, ["model", "method", "input_size"])
        print(f"No drift. Summary written to {args.out / 'latency.csv'}")


def cmd_merge(args) -> None:
    """Post-hoc decision A (2026-10-15, see NOTES.md): both latency runs failed rule B
    (reference noise ~15% over 30 min, no trend). They are accepted together: per
    configuration the median of the run medians (with 2 runs: their mean), both runs'
    values, and the run-to-run difference. p95 is over the pooled timed runs.
    A configuration counts as run (status ok) if it ran in at least one run;
    n_runs_ok and the per-run status say which."""
    raw = args.out / "raw"
    runs = []
    for run_id in args.run_ids:
        s = pd.read_csv(raw / f"{run_id}_latency_summary.csv")
        t = pd.read_csv(raw / f"{run_id}_latency_runs.csv")
        meta = json.loads((raw / f"{run_id}_latency_meta.json").read_text())
        runs.append((run_id, s, t, meta["drift"]))
    rows = []
    for _, first in runs[0][1].iterrows():
        key = (first.method, first.input_size)
        per_run = [(rid, s[(s.method == key[0]) & (s.input_size == key[1])].iloc[0], t, d)
                   for rid, s, t, d in runs]
        row = {c: first[c] for c in ["model", "method", "height", "width", "input_size",
                                     "n_patches", "tome_r_per_block"]}
        ok = [(rid, r, t) for rid, r, t, _ in per_run if r.status == "ok"]
        for k, (rid, r, _, d) in enumerate(per_run, 1):
            row[f"run{k}_id"] = rid
            row[f"run{k}_status"] = r.status
            row[f"run{k}_reference_drift_spread"] = d["relative_spread"]
            for c in ["latency_median_ms", "latency_p95_ms", "warmup", "runs", "first_pass_s"]:
                row[f"run{k}_{c}"] = r.get(c)
        row["n_runs_ok"] = len(ok)
        row["status"] = "ok" if ok else per_run[0][1].status
        if ok:
            tokens = {(r.tokens_in, r.tokens_out, r.tokens_mean_per_block) for _, r, _ in ok}
            assert len(tokens) == 1, f"token counts differ between runs for {key}: {tokens}"
            medians = [r.latency_median_ms for _, r, _ in ok]
            pooled = pd.concat([t[t.block == r.block].latency_ms for _, r, t in ok])
            median = float(pd.Series(medians).median())
            row.update({
                "latency_median_ms": median,
                "latency_p95_ms": float(pooled.quantile(0.95)),
                "n_timed_runs_total": len(pooled),
                "throughput_img_per_s": 1000.0 / median,
                "run_to_run_diff": (max(medians) - min(medians)) / min(medians) if len(ok) > 1 else None,
                "tokens_in": ok[0][1].tokens_in, "tokens_out": ok[0][1].tokens_out,
                "tokens_mean_per_block": ok[0][1].tokens_mean_per_block,
                "tokens_verified": all(bool(r.tokens_verified) for _, r, _ in ok),
                "size_on_disk_mb": ok[0][1].size_on_disk_mb,
            })
        row.update({"est_largest_tensor_mb": first.est_largest_tensor_mb, "est_peak_mb": first.est_peak_mb,
                    "torch_threads": first.torch_threads,
                    "acceptance": "post-hoc decision A: two runs, both failed rule B (see NOTES.md)"})
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(args.out / "latency.csv", index=False)
    print(df[["method", "input_size", "n_runs_ok", "run1_latency_median_ms", "run2_latency_median_ms",
              "latency_median_ms", "run_to_run_diff", "status"]].to_string())
    d = df.run_to_run_diff.dropna()
    print(f"run-to-run diff over {len(d)} configs: median {d.median():.1%}, max {d.max():.1%}")
    for rid, _, _, drift in runs:
        m = drift["baseline_block_medians_ms"]
        print(f"{rid}: reference blocks {len(m)}, min {min(m):.2f}, max {max(m):.2f} ms, "
              f"rolling spread {drift['relative_spread']:.1%}, raw spread {drift['raw_spread_old_rule']:.1%}")
    print(f"Wrote {args.out / 'latency.csv'}")


# ---------------------------------------------------------------- memory
def cmd_memory_prepare(args) -> None:
    """Child process: build one configuration, save it and verify the reload."""
    setup()
    h, w = args.size
    model, cfg = build(args.method, h, w)
    x = load_frame(frame_paths()[LATENCY_FRAME], h, w, cfg)
    model_file, input_file = save_and_verify(model, x, f"kitti_{MODEL}_{args.method}_{h}x{w}")
    print("RESULT " + json.dumps({"model_file": str(model_file), "input_file": str(input_file)}))


def latency_status(out, method: str, h: int, w: int):
    """Status of this configuration in latency.csv (None if not measured)."""
    path = out / "latency.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    hit = df[(df.method == method) & (df.input_size == f"{h}x{w}")]
    return None if hit.empty else hit.iloc[0].to_dict()


def cmd_memory(args) -> None:
    setup()
    state = system_state(idle_seconds=1.0)
    run_id = state["timestamp"].replace(":", "-")
    rows = []
    for h, w in [s for s in sizes() if s[0] in args.heights]:
        for method in args.methods:
            row = {**base_row(method, h, w), "run_id": run_id, "time": now(), "status": "ok",
                   "measurement": "fresh process loads saved model", "memory_inference_runs": args.runs}
            lat = latency_status(args.out, method, h, w)
            est = memory_estimate(method, h, w)
            row.update(est)
            if lat is None:
                row["status"] = "skipped: no latency result for this configuration"
            elif lat["status"] != "ok":
                row["status"] = f"not run ({lat['status']})"
            elif est["skip_reason"]:
                row["status"] = est["skip_reason"]
            else:
                prep = subprocess.run(
                    [sys.executable, "-u", __file__, "memory-prepare", "--method", method,
                     "--size", str(h), str(w)],
                    capture_output=True, text=True, timeout=900)
                if prep.returncode != 0:
                    raise RuntimeError(f"memory-prepare failed:\n{prep.stderr[-2000:]}")
                files = json.loads(next(l for l in prep.stdout.splitlines()
                                        if l.startswith("RESULT "))[len("RESULT "):])
                r = run_memory_worker(Path(files["model_file"]), Path(files["input_file"]),
                                      NUM_THREADS, timeout_s=900, runs=args.runs)
                row.update(r)
                row["measured_minus_estimated_peak_mb"] = r["peak_total_mb"] - est["est_peak_mb"]
            rows.append(row)
            msg = (f"peak {row['peak_total_mb']:8.2f} MB (+inference {row['inference_increment_mb']:7.2f}; "
                   f"estimate {est['est_peak_mb']:.0f})" if row["status"] == "ok" else row["status"])
            print(f"{method:13s} {h}x{w}  {msg}")

    raw = args.out / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(raw / f"{run_id}_memory.csv", index=False)
    for row in rows:
        upsert(args.out / "memory.csv", row, ["model", "method", "input_size"])
    print(f"Wrote {args.out / 'memory.csv'}")


# ---------------------------------------------------------------- agreement
@torch.inference_mode()
def cmd_agreement(args) -> None:
    setup()
    paths = frame_paths()
    frames = [paths[i] for i in AGREEMENT_FRAMES]
    info = ImageNetInfo()
    run_id = now().replace(":", "-")
    rows = []
    pred_dir = args.out / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    for h, w in [s for s in sizes() if s[0] in args.heights]:
        preds = {}
        for method in [BASELINE] + [m for m in args.methods if m != BASELINE]:
            row = {**base_row(method, h, w), "run_id": run_id, "time": now(), "status": "ok",
                   "n_frames": len(frames), "frames": "every 4th frame, 0-104"}
            lat = latency_status(args.out, method, h, w)
            est = memory_estimate(method, h, w)
            if lat is None:
                row["status"] = "skipped: no latency result for this configuration"
            elif lat["status"] != "ok":
                row["status"] = f"not run ({lat['status']})"
            elif len(frames) * lat["latency_median_ms"] / 1000 > AGREEMENT_BUDGET_S:
                row["status"] = (f"skipped: time limit (estimated "
                                 f"{len(frames) * lat['latency_median_ms'] / 1000:.0f} s > "
                                 f"{AGREEMENT_BUDGET_S:.0f} s)")
            elif est["skip_reason"]:
                row["status"] = est["skip_reason"]
            elif method != BASELINE and BASELINE not in preds:
                row["status"] = "skipped: no FP32 predictions at this resolution"
            if row["status"] != "ok":
                rows.append(row)
                print(f"{method:13s} {h}x{w}  {row['status']}")
                continue
            t0 = time.perf_counter()
            model, cfg = build(method, h, w)
            top1 = [int(model(load_frame(f, h, w, cfg)).argmax(-1)) for f in frames]
            del model
            preds[method] = top1
            pd.DataFrame({"frame": [f.name for f in frames], "top1_index": top1,
                          "top1_name": [info.index_to_description(i) for i in top1]}) \
                .to_csv(pred_dir / f"{h}x{w}_{method}.csv", index=False)
            ref = preds[BASELINE]
            n_agree = sum(a == b for a, b in zip(top1, ref))
            row.update({"n_agree_with_fp32": n_agree, "agreement_with_fp32": n_agree / len(frames),
                        "n_distinct_top1": len(set(top1)), "eval_seconds": time.perf_counter() - t0})
            rows.append(row)
            print(f"{method:13s} {h}x{w}  agreement {n_agree}/{len(frames)}  "
                  f"({row['eval_seconds']:.0f} s)")

    raw = args.out / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(raw / f"{run_id}_agreement.csv", index=False)
    for row in rows:
        upsert(args.out / "agreement.csv", row, ["model", "method", "input_size"])
    print(f"Wrote {args.out / 'agreement.csv'}")


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ["latency", "memory", "agreement"]:
        s = sub.add_parser(name)
        s.add_argument("--heights", nargs="+", type=int, default=HEIGHTS)
        s.add_argument("--methods", nargs="+", default=METHODS)
        s.add_argument("--out", type=Path, default=OUT_DIR)
    sub.choices["latency"].add_argument("--heatup-seconds", type=float, default=HEATUP_SECONDS)
    sub.choices["memory"].add_argument("--runs", type=int, default=3,
                                       help="inference passes in the memory worker")
    s = sub.add_parser("merge")
    s.add_argument("--run-ids", nargs="+", required=True)
    s.add_argument("--out", type=Path, default=OUT_DIR)
    s = sub.add_parser("memory-prepare")
    s.add_argument("--method", required=True)
    s.add_argument("--size", nargs=2, type=int, required=True)
    args = p.parse_args()
    {"latency": cmd_latency, "merge": cmd_merge, "memory": cmd_memory, "memory-prepare": cmd_memory_prepare,
     "agreement": cmd_agreement}[args.cmd](args)


if __name__ == "__main__":
    main()
