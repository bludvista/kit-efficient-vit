# Running notes: decisions, failures, irregularities

All numbers here are copied from files in `results/`; the file is named for each.

## Measurement conditions
- Windows power mode: overlay GUID `961cc777-…`, confirmed by the user as
  "Best power efficiency". The user confirmed it was not changed during the
  project so far; every timing run records it (`*_meta.json`). On AC power.
- 4 threads (= physical cores). Batch size 1.
- Latency excludes image loading and preprocessing (one fixed preprocessed image).

## Phase 1 FP32 latency (93.31 ms) is invalid, replaced
- Run `latency_raw/2026-10-14T14-02-58_*`: FP32 median 93.31 ms, idle CPU before 20.8%.
- The user reports a browser (5 tabs) and VS Code were open around that time.
- All later runs give about 60-62 ms for the same configuration. The cause of the
  difference is not proven; the run is kept in `latency_raw/` but not used.

## Rejected run: drift
- Run `latency_raw/2026-10-14T14-50-57_*`: baseline block medians spread 12.5%
  (> 10% threshold), so it was NOT written to `latency.csv`. Its numbers are not used.

## Stability test (`stability/`, `src/stability.py`)
- 3 sessions x 10 FP32 blocks, 10 min apart, Edge closed, VS Code open.
- In every session block 0 was faster (55.48, 50.93, 52.75 ms) than the rest:
  a cool CPU is faster for the first seconds.
- Blocks 1-9 spread within session: 6.1%, 21.6%, 7.3%. Session 2 drifted upward.
- Median of blocks 1-9 per session: 61.80, 61.15, 60.72 ms (1.8% between sessions).
- Decision taken from this, before the next run: 60 s untimed heat-up before the
  first timed block; keep the 10% threshold on (warm) baseline blocks.

## ToMe
- Official facebookresearch/ToMe, commit af95e4b, unchanged; tested with timm 0.4.12
  by its authors, we use timm 1.0.30. Checks in `tome_compat_check.json` pass
  (r=0 equals timm; token counts follow the paper's rule; merge step equals a loop
  reference implementation written from the paper).
- ToMe ignores LayerScale / QK-norm / attn norm / gate: all inactive in DeiT-Tiny
  (checked), but the patch would be wrong for models that use them.
- ToMe uses explicit attention; timm FP32 uses fused SDPA. `tome_r0` is reported
  separately to show the kernel difference.
- With constant r, the 50% cap per layer starts to bind at r=16 (last layer).
- Each ToMe configuration has one 50-run timed block per latency run; the FP32
  baseline has one block between every pair, so 8 blocks / 400 runs.

## Memory
- At 224 px, batch 1, peak memory is reached while loading weights; inference adds
  0 MB on top (`inference_increment_mb`). So peak memory does not distinguish the
  methods at this input size.

## Data
- Imagenette-160 val, 3925 images, full set (no subset). Images upscaled to 224 by
  timm's eval transform. A uint8 cache of the resized/cropped images is used for
  accuracy runs; verified bit-identical to the original transform (max diff 0.0
  over 100 images) and FP32 predictions identical for all 3925 images.
