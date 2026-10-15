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

## Rejected run: Phase 3 latency, sustained-load drift
- Run `latency_raw/2026-10-15T14-28-48_*`: 15 configurations, 3 blocks each,
  85 timed blocks from 14:30:00 to 14:36:31, after a 60 s heat-up. Edge closed.
- Baseline block medians: 62.20 ms at the start, rising to 73.87 ms after 33 s and
  staying at 74.03-82.57 ms for the rest of the run, plus three isolated spikes
  (87.90, 94.81, 112.18 ms). Spread 80.3% > 10%, so NOT written to `latency.csv`.
- Meaning: under continuous load for more than about 1.5-2 min this laptop settles
  at a slower level (~77 ms for FP32). All accepted latencies so far (Phase 2 run
  `2026-10-14T15-38-31`, ~2 min of timing) were measured before reaching that level.
- See "Phase 3 latency run" below: it replaces all earlier latencies.

## Rule B (sustained load), from 2026-10-15 on
- Defined AFTER the rejected run `2026-10-15T14-28-48`, having seen its data
  (including its three spikes). It applies only to runs made after it; that run
  stays rejected, and earlier accepted/rejected decisions are not revisited.
- Heat-up: 180 s untimed FP32 (was 60 s), to reach the sustained-load level.
- Drift rule: spread of the rolling median of 3 consecutive baseline blocks must be
  <= 10%. Isolated single-block spikes therefore do not fail a run, but a lasting shift does.
- Nothing is hidden: every run records the number of ignored baseline spikes (blocks
  > 10% from the median of themselves and their neighbours) and lists them, the
  raw spread under the old rule, and per configuration the number of its blocks
  > 10% from its own median. p95 latency is computed over all timed runs, spikes included.
- Reported latency per configuration = median of its 3 block medians; the min/max
  block medians and the spread between blocks are reported with it.

## Phase 3 latency run (accepted, rule B): replaces all earlier latencies
- Run `latency_raw/2026-10-15T14-45-58_*`, 180 s heat-up, 15 configurations x 3 blocks,
  43 baseline blocks. Chrome and Edge closed, VS Code open. Rolling-median spread
  7.3% (<= 10%), raw spread 48.1%; 1 ignored baseline spike (block 26 of the
  baseline sequence, 111.72 ms).
- The Phase 2 single-block latencies (run `2026-10-14T15-38-31`, short burst, before
  the sustained-load level) are replaced by this run in `latency.csv` for FP32 and
  all ToMe r. The old run stays in `latency_raw/` for reference only.
- Config blocks > 10% from their own median (kept in median/p95, counted in
  `n_block_outliers`): tome_r8 (one block 91.31 ms), tome_r20_int8 (one block 78.88 ms).
- FP32 baseline here 79.79 ms vs 62.29 ms in the Phase 2 burst run (ratio 1.281).
- Measured, not explained: ToMe + INT8 is slower than ToMe alone at every r, while
  INT8 alone is faster than FP32. A plausible but UNTESTED explanation is the fixed
  cost of quantizing activations in every linear call, which matters more as ToMe
  shrinks the matrices. Do not state it as a finding unless it is measured.

## Memory of INT8 configurations is not a deployment number
- `peak_total_mb` for int8 (366.56) and tome_r*_int8 (380.85-384.42) is higher than
  FP32 (358.92) because the memory worker loads the FP32 model and then quantizes it
  in the same process, so the peak includes the FP32 copy. It does not show the
  memory needed to run an already-quantized model. The ~15 MB extra of
  ToMe + INT8 over INT8 is not explained.

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

## INT8: choice of API (`int8_api_check.json`, `src/int8_api_check.py`)
- PyTorch docs (2.14): `torch.ao.quantization` is deprecated (its own warning says
  "will be removed in 2.10", but it is still present in torch 2.14.1); eager
  `quantize_dynamic` users should migrate to torchao's `quantize_`.
- Candidates tested on this laptop:
  A `torch.ao.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)`,
  engine `onednn` (the only engine in this torch build);
  B torchao 0.18.0 `quantize_(model, Int8DynamicActivationInt8WeightConfig(reduce_range=...))`,
  eager (no `torch.compile`). torchao's check reports no VNNI on this CPU, so B used
  `reduce_range=True`.
- B ran but took 1859-1958 ms per single image vs 54-64 ms for FP32, i.e. far over
  the 10x "too slow" limit, so it was not evaluated further. Making it fast would
  need `torch.compile` (on Windows also a C++ compiler), which would compare a
  compiled model against eager FP32. Not pursued.
- A converts all 49 linear layers (12 blocks x 4 + classifier head); 6.15 MB on disk
  vs 21.85 MB; on 200 fixed images top-1 agreement with FP32 is 99.5% (10 classes)
  and 92.0% (1000 classes).
- Choice: A. Reason: it is the only candidate that runs at a usable speed in eager
  mode on this CPU. Drawback: deprecated API; works in the pinned torch 2.14.1.
- The latency probe inside the check (FP32 78.59 / 73.22 ms, A 75.11 / 71.23 ms) was a
  short pilot used only for choosing (30 s heat-up, 2 blocks). It is NOT a reported
  result; reported INT8 latencies come from `latency.py`.
- A first run of the check was stopped by me after ~10 min with no output: it was
  evaluating B on 200 images at ~1.9 s each. The script was changed to time single
  passes first and skip a candidate that is > 10x slower than FP32.

## Memory
- At 224 px, batch 1, peak memory is reached while loading weights; inference adds
  0 MB on top (`inference_increment_mb`). So peak memory does not distinguish the
  methods at this input size.

## Data
- Imagenette-160 val, 3925 images, full set (no subset). Images upscaled to 224 by
  timm's eval transform. A uint8 cache of the resized/cropped images is used for
  accuracy runs; verified bit-identical to the original transform (max diff 0.0
  over 100 images) and FP32 predictions identical for all 3925 images.
