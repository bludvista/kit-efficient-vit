"""Check whether the official ToMe code (facebookresearch/ToMe) works with the
installed timm version, for deit_tiny_patch16_224.

ToMe was tested with timm 0.4.12; its timm patch replaces timm's Block and
Attention forward methods with copies of the 0.4.12 versions. This script tests:
  1. the patch imports and applies;
  2. with r = 0 the patched model gives the same logits as unpatched timm
     (with timm's fused SDPA attention and with its explicit attention path);
  3. the number of tokens after each block follows the paper's rule
     n_{l+1} = n_l - min(r, (n_l - 1) // 2)   (class token is never merged);
  4. one merge step equals an independent loop implementation written from the
     paper (Bolya et al., ICLR 2023, Sec. 3): alternate tokens into sets A and B,
     cosine similarity of keys, class token protected, merge the r most similar
     A->B edges, size-weighted average.
Results: results/tome_compat_check.json
"""
import json
import subprocess
import sys

import timm
import torch
import torch.nn.functional as F

from common import DATA_DIR, RESULTS_DIR, setup
from data import ImagenetteVal

TOME_DIR = DATA_DIR / "third_party" / "ToMe"
sys.path.insert(0, str(TOME_DIR))
import tome  # noqa: E402  (official code, pinned commit recorded below)
from tome.merge import bipartite_soft_matching, merge_wavg  # noqa: E402

MODEL = "deit_tiny_patch16_224"
R_VALUES = [0, 4, 8, 12, 16, 20, 24, 32]


def expected_tokens(n: int, r: int, depth: int) -> list[int]:
    counts = []
    for _ in range(depth):
        n = n - min(r, (n - 1) // 2)
        counts.append(n)
    return counts


def tokens_per_block(model, x) -> list[int]:
    counts = []
    hooks = [b.register_forward_hook(lambda m, i, o: counts.append(o.shape[1])) for b in model.blocks]
    with torch.inference_mode():
        model(x)
    for h in hooks:
        h.remove()
    return counts


def build_patched(fused: bool):
    model = timm.create_model(MODEL, pretrained=True).eval()
    for blk in model.blocks:
        blk.attn.fused_attn = fused
    return model


def reference_merge(metric, x, size, r):
    """Loop implementation of one ToMe merge step, written from the paper text.
    Batch size 1. Token 0 is the class token and is never merged."""
    metric, x, size = metric[0], x[0], size[0]
    a_idx = list(range(0, x.shape[0], 2))   # set A: even positions (includes class token)
    b_idx = list(range(1, x.shape[0], 2))   # set B: odd positions
    m = F.normalize(metric, dim=-1)
    best = {}
    for i in a_idx:
        if i == 0:
            continue  # class token protected
        sims = torch.stack([m[i] @ m[j] for j in b_idx])
        best[i] = (sims.max().item(), b_idx[int(sims.argmax())])
    merged_a = sorted(best, key=lambda i: best[i][0], reverse=True)[:r]
    # size-weighted average into the destination B tokens
    acc = {j: (x[j] * size[j], size[j].clone()) for j in b_idx}
    for i in merged_a:
        j = best[i][1]
        acc[j] = (acc[j][0] + x[i] * size[i], acc[j][1] + size[i])
    kept_a = [i for i in a_idx if i not in merged_a]
    out_x = [x[i] for i in kept_a] + [acc[j][0] / acc[j][1] for j in b_idx]
    out_s = [size[i] for i in kept_a] + [acc[j][1] for j in b_idx]
    return torch.stack(out_x)[None], torch.stack(out_s)[None]


def main() -> None:
    setup()
    report = {
        "timm_version": timm.__version__,
        "torch_version": torch.__version__,
        "tome_commit": subprocess.run(["git", "-C", str(TOME_DIR), "rev-parse", "HEAD"],
                                      capture_output=True, text=True).stdout.strip(),
        "model": MODEL,
    }

    # Modules the 0.4.12-era ToMe forward ignores must be inactive in this model.
    ref = timm.create_model(MODEL, pretrained=True).eval()
    blk = ref.blocks[0]
    ignored = {
        "ls1/ls2 are Identity": isinstance(blk.ls1, torch.nn.Identity) and isinstance(blk.ls2, torch.nn.Identity),
        "q_norm/k_norm are Identity": isinstance(blk.attn.q_norm, torch.nn.Identity) and isinstance(blk.attn.k_norm, torch.nn.Identity),
        "attn.norm is Identity": isinstance(blk.attn.norm, torch.nn.Identity),
        "attn.gate is None": blk.attn.gate is None,
        "baseline uses fused SDPA attention": bool(blk.attn.fused_attn),
    }
    report["modules_ignored_by_tome"] = ignored

    data_config = timm.data.resolve_data_config({}, model=ref)
    ds = ImagenetteVal(data_config)
    x = torch.stack([ds[i * 390][0] for i in range(8)])  # 8 images, spread over classes

    # Test 2: r = 0 equivalence, with both attention paths of unpatched timm.
    equivalence = {}
    for fused in (True, False):
        base = build_patched(fused)
        with torch.inference_mode():
            y_base = base(x)
        tome.patch.timm(base)
        base.r = 0
        with torch.inference_mode():
            y_tome = base(x)
        diff = (y_base - y_tome).abs().max().item()
        equivalence[f"timm_fused_attn={fused}"] = {
            "max_abs_logit_diff": diff,
            "same_top1_all_images": bool((y_base.argmax(1) == y_tome.argmax(1)).all()),
            "allclose_atol_1e-4": bool(torch.allclose(y_base, y_tome, atol=1e-4)),
        }
    report["r0_equivalence"] = equivalence

    # Test 3: tokens per block vs the paper's rule.
    model = build_patched(True)
    tome.patch.timm(model)
    n0 = model.patch_embed.num_patches + 1
    counts = {}
    for r in R_VALUES:
        model.r = r
        got = tokens_per_block(model, x[:1])
        exp = expected_tokens(n0, r, len(model.blocks))
        counts[str(r)] = {"tokens_after_each_block": got, "expected": exp, "match": got == exp}
    report["tokens_in"] = n0
    report["token_counts"] = counts

    # Test 4: one merge step, official vs reference loop, random inputs.
    torch.manual_seed(0)
    merge_tests = []
    for n, r in [(11, 3), (197, 8), (197, 16), (50, 24)]:
        metric = torch.randn(1, n, 16)
        xx = torch.randn(1, n, 8)
        size = torch.randint(1, 4, (1, n, 1)).float()
        merge, _ = bipartite_soft_matching(metric, r, class_token=True)
        x_off, s_off = merge_wavg(merge, xx, size)
        r_eff = min(r, (n - 1) // 2)
        x_ref, s_ref = reference_merge(metric, xx, size, r_eff)
        merge_tests.append({
            "tokens": n, "r": r, "r_effective": r_eff,
            "shapes_equal": list(x_off.shape) == list(x_ref.shape),
            "max_abs_diff_x": (x_off - x_ref).abs().max().item() if x_off.shape == x_ref.shape else None,
            "sizes_equal": bool(torch.equal(s_off, s_ref)) if s_off.shape == s_ref.shape else False,
        })
    report["merge_step_vs_reference"] = merge_tests

    out = RESULTS_DIR / "tome_compat_check.json"
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
