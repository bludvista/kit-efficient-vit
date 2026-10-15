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

## Memory measurement: old method superseded (2026-10-15)
- Old method (Phase 1-3 runs, `latency_raw/*_summary.csv`): the fresh process built
  the configuration itself from the pretrained FP32 checkpoint. For INT8 the peak
  therefore included the FP32 copy made before quantization: int8 366.56 MB and
  tome_r*_int8 380.85-384.42 MB, higher than FP32 358.92 MB. Those numbers do not
  show the memory needed to run an INT8 model and are no longer used. Their columns
  were removed from `latency.csv`; they remain in `latency_raw/` for reference.
- New method (`src/memory.py`, results in `memory.csv`): each configuration is built,
  saved with torch.save (whole module), checked to reload to identical outputs, and
  a fresh process (`memory_worker.py`) only loads and runs it. All 15 saved models
  reproduced their outputs exactly.
- To make ToMe models savable, models.py registers ToMe's class (created by ToMe's
  own make_tome_class, code unchanged) under a fixed module name. Checked: identical
  logits to the original ToMe patch on 25 images for tome_r8 and tome_r16_int8, and
  `tome_compat_check.py` still passes.
- Results, run `memory_raw/2026-10-15T15-20-30.csv`, 224 px, batch 1:
  FP32 339.64 MB, INT8 336.69 MB; ToMe 341.69-347.25 MB; ToMe + INT8 354.81-359.02 MB.
  About 305-307 MB of every value is the Python/torch/timm runtime after imports.
- With the new method inference does add memory at 224 px: FP32 +8.38 MB, ToMe
  +10.27 to +15.80 MB, INT8 +10.56 MB, ToMe + INT8 +29.39 to +32.70 MB. So at this
  input size ToMe and especially ToMe + INT8 need MORE peak memory than FP32, not
  less (measured; the cause is not measured).
- INT8 load increment is 19.74 MB vs FP32 25.13 MB (-5.4 MB), although the saved
  file is 6.19 vs 21.92 MB (-15.7 MB). Not explained.
- One measurement per configuration; run-to-run variability of memory was not
  measured. Only data point: FP32 338.89 MB in an aborted earlier attempt of the
  same method (15:17, ToMe failed to load, nothing written) vs 339.64 MB here.

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

## Memory (old method, superseded; see "Memory measurement" above)
- With the old method, peak memory was reached while loading the pretrained weights
  and inference added 0 MB on top. That conclusion does not hold for the new method.

## DeiT-Small: skipped (decided by the user, 2026-10-15)
- Reason: time limits on this hardware. One full Imagenette accuracy pass of
  DeiT-Tiny takes 224 s (FP32, `accuracy.csv` eval_seconds). DeiT-Small has twice the
  embedding width (384 vs 192) and 22,050,664 vs 5,717,416 parameters (both
  checked in timm 1.0.30) and is expected to take several times longer (not
  measured), so its 15 configurations would far exceed the 30-minute limit, and the
  time is better spent on the KITTI resolution scaling.
  No DeiT-Small number was measured, so none is reported.

## Data
- Imagenette-160 val, 3925 images, full set (no subset). Images upscaled to 224 by
  timm's eval transform. A uint8 cache of the resized/cropped images is used for
  accuracy runs; verified bit-identical to the original transform (max diff 0.0
  over 100 images) and FP32 predictions identical for all 3925 images.

## Phase 4: resolution scaling on KITTI (`src/resolution.py`, `results/resolution/`)
- Data: KITTI raw drive `2011_09_26_drive_0001_sync`, left colour camera `image_02`,
  108 frames of 1242x375 px (put in `data/kitti/` by the user). Consecutive frames
  at 10 Hz, so highly correlated. Latency and memory use frame 0000000000.
- Sizes keep KITTI's aspect ratio: H in {64, 128, 192, 256, 368, 512, 736, 1024},
  W = H * 1242/375 rounded to a multiple of 16. 368x1216 is closest to the original
  (slight downscale); 512 and up are UPSCALED and only show how cost grows with tokens.
- Whole frame resized (bicubic, antialias), no crop, model mean/std. timm
  `dynamic_img_size=True`; the 14x14 position embedding is interpolated (bicubic) on
  every forward pass. Checked: outputs at 224x224 identical to the normal model; the
  normal model rejects 128x416; tokens entering block 0 equal patches + 1 at every
  size that ran (`tokens_verified` in latency.csv).
- Constant r = 8 does not transfer across sizes (in a pilot it merged 53 tokens down
  to 2, and 1749 only to 1653). So `tome_p8` uses r = round(8 * patches / 196), the
  same fraction per block as r = 8 at 224 (no top-1 loss on Imagenette). `tome_r0`
  (ToMe code, no merging, explicit attention) is the kernel control at every size.
- Timing protocol: 10 warm-up + 50 timed runs, reduced only when a pass is too slow
  (warm-up budget 20 s, min 2; timed budget 60 s, min 5); numbers used are in
  latency.csv per run. One block per configuration per run (not 3 as in Phase 3).
  Drift check: FP32 reference blocks at 128x416 before every configuration.
- Limits (decided before the runs): memory = largest tensor > 1.5 GB or estimated
  peak > 70% of available RAM (estimate: 350 MB runtime + 3 x heads*N^2*4 B for
  explicit attention, 4 x N*768*4 B for SDPA); time = one pass > 30 s.
- Breaking points (both runs agree except FP32 at 13569 tokens):
  * ToMe variants (explicit attention): skipped at 736x2432 (6993 tokens), estimated
    peak 2029 MB > 70% of available RAM (1384 MB in run 1, 1517 MB in run 2);
    largest tensor estimate 559.64 MB, so the RAM rule, not the 1.5 GB rule, stopped them.
  * FP32 at 1024x3392 (13569 tokens): ran in run 1 (27.52 s median), skipped in run 2
    (first pass 30.2 s > 30 s). It sits exactly at the time limit.
  * INT8 at 1024x3392 ran in both runs (27.02 s and 29.25 s).
  * The 1.5 GB single-tensor rule would stop explicit attention at 13569 tokens
    (2107 MB estimate); SDPA was not stopped by memory at any size tried.
  * Free RAM at the start of the runs was only 877-880 MB (of 5.9 GB total).

### Post-hoc decision A (2026-10-15, user): two latency runs accepted together
- Both latency runs FAILED rule B, so neither alone was written to latency.csv:
  * run `2026-10-15T17-23-30`: 36 reference blocks, 54.67-68.56 ms, rolling-median
    spread 16.1%, raw spread 25.4%, 2 ignored spikes. Browsers closed, VS Code open.
  * run `2026-10-15T21-17-16`: 35 reference blocks, 53.03-69.40 ms, rolling-median
    spread 17.8%, raw spread 30.9%, 1 ignored spike. Browsers and VS Code closed.
  * The reference wandered up and down with no steady trend; over ~32-min runs this
    laptop's noise is larger than the 10% rule B threshold. Cause not measured.
- Decision taken AFTER seeing both runs (so it is post-hoc), by the user: accept both
  runs together instead of a third identical run or a new protocol. Rule B is not
  changed for any other experiment.
- `latency.csv` (made by `resolution.py merge`): latency_median_ms = median of the run
  medians (= their mean for 2 runs), both runs' values in run1_*/run2_*, p95 over the
  pooled timed runs, `run_to_run_diff` = (max - min) / min of the run medians.
- Run-to-run difference over the 33 configurations that ran in both runs: median
  7.6%, max 21.6% (int8 at 192x640). FP32 at 1024x3392 has one run only (n_runs_ok = 1).
- Consequence for interpretation: differences between methods at the same resolution
  that are smaller than about the reference noise (16-18%) or the run-to-run
  difference of those configurations are NOT resolved by these measurements. This
  includes most INT8 vs FP32 and tome_p8 vs FP32 comparisons. The growth of latency
  with tokens (FP32 22.44 ms at 53 tokens; INT8 28.14 s at 13569 tokens) is far larger than the noise.

### Phase 4 peak memory (`results/resolution/memory.csv`)
- Method as in Phase 3 (fresh process loads the saved model), plus: building/saving/
  verifying also runs in its own process (`memory-prepare`), so no two copies of a
  large configuration are alive at once. 3 inference passes in the worker. One
  measurement per configuration; run-to-run variability not measured.
- Inference increment, FP32 (SDPA): 6.95 MB at 53 tokens -> 140.37 MB at 13569 tokens.
  tome_r0 (explicit attention, no merging): 292.05 MB at 3393 tokens vs FP32 45.52 MB.
- tome_p8 and tome_p8_int8 peak HIGHER than tome_r0 at every size (e.g. 368x1216:
  462.10 and 466.50 vs 436.62 MB). Merging does not lower peak memory here; block 0
  attends over all tokens before any merge, so the N x N peak of block 0 remains
  (this mechanism is the reason by construction of ToMe; the extra MB above tome_r0
  are measured, not explained).
- The skip estimate is not an upper bound: it was below the measurement for some
  ToMe configurations (tome_p8_int8: 387.20 vs 375 MB at 256x848, 466.50 vs 455 MB
  at 368x1216) and above it at 512x1696 (745 vs 632.40-685.85 MB). The skipped
  6993-token ToMe configurations were estimated at 2029 MB; they were not run, so
  their real peak is unknown.

### Phase 4 top-1 agreement (`results/resolution/agreement.csv`, `predictions/`)
- STABILITY PROXY ONLY, not accuracy: KITTI frames have no ImageNet labels. Top-1 over
  all 1000 ImageNet classes, compared with FP32 at the same resolution, on 27 frames
  (every 4th frame, 0-104) of one 11-s drive; the frames are highly correlated.
- tome_r0 agrees 27/27 at every size it ran (explicit attention = SDPA in top-1).
- Skips: the ToMe variants at 6993+ tokens (memory, as in latency); FP32 and INT8 at
  13569 tokens (estimated 743 / 760 s > 600 s budget); tome_p8 and tome_p8_int8 at
  512x1696: skipped by the RAM rule (available RAM 883 MB in the first agreement run,
  825 MB in a fresh-process retry of just these configurations, `agreement_log_run2.txt`).
  The retry recomputed FP32 at 512x1696 (27/27, same predictions file overwritten).
  The latency and memory runs did run these two configurations, when more RAM was free.

## Phase 5: attention scaling, synthetic (`src/attention.py`, `results/attention/`)
- Random float32 Q, K, V, batch 1, 3 heads, head dim 64 (DeiT-Tiny's shape),
  N = 256 ... 65536 (doubling). naive = softmax(QK^T/sqrt(d))V written out;
  sdpa = F.scaled_dot_product_attention, no mask. CPU only; memory-efficient attention
  kernels mainly target GPUs, so published GPU results need not carry over.
- `check.json`: outputs agree (max abs diff <= 2.98e-7 at N = 256, 1024, 2048). The
  profiler shows sdpa runs `aten::_scaled_dot_product_flash_attention_for_cpu` here.
- Timing run `2026-10-15T22-44-39`: FAILED rule B (reference sdpa N=1024 rolling-median
  spread 20.2%, raw 38.0%; the reference got faster during the run, ~18.6 -> ~14.9 ms).
  Decided BEFORE the run (written in attention.py): report it with its noise, no rerun.
  `drift_detected` is in every row of timing.csv. At each N sdpa was timed right after
  naive; with the machine getting faster over the run, this order favours sdpa slightly.
- Breaking points: naive skipped from N = 8192 (estimated peak 1810 MB > 70% of
  available RAM 859 MB; single tensor 768 MB, under 1.5 GB). sdpa ran to N = 32768
  (12086.211 ms median); N = 65536 hit the time limit (first call 48.6 s > 30 s).
- Memory (fresh process per point, one call): naive N = 4096 was skipped in the memory
  step (estimate 646 MB > 70% of 874 MB available) although its timing ran earlier,
  when more RAM was free; so naive memory only up to N = 2048.
- The memory estimate assumed a 250 MB runtime; the measured peak before the call is
  about 335-340 MB (the worker imports the project modules incl. pandas/timm). So the
  estimates were too low by about that difference; the skip decisions were made with
  the too-low estimate.
