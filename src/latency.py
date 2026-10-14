"""Latency, throughput, peak memory and size on disk, batch size 1.

Thermal-drift control: timed blocks of the configurations are interleaved with
FP32 baseline blocks (B, C1, B, C2, ..., B; at least MIN_BASELINE_BLOCKS of B).
If baseline block medians spread by more than bench.DRIFT_THRESHOLD, the run
is flagged and the summary CSV is NOT updated; only raw logs are kept.

Outputs (run_id = start timestamp):
  results/latency_raw/<run_id>_runs.csv    every timed forward pass
  results/latency_raw/<run_id>_blocks.csv  one row per block
  results/latency_raw/<run_id>_meta.json   system state, drift check
  results/latency.csv                      summary per (model, method), only if no drift
"""
import argparse
import json

import pandas as pd
import torch

import models
from accuracy import upsert
from bench import (DRIFT_THRESHOLD, drift_check, measure_memory, now,
                   size_on_disk_mb, system_state, time_block)
from common import RESULTS_DIR, setup
from data import timing_input

RAW_DIR = RESULTS_DIR / "latency_raw"
SUMMARY_CSV = RESULTS_DIR / "latency.csv"
BASELINE = "fp32"
MIN_BASELINE_BLOCKS = 3


def schedule(methods: list[str], min_baseline: int) -> list[str]:
    configs = [m for m in methods if m != BASELINE]
    order = [BASELINE]
    for m in configs:
        order += [m, BASELINE]
    while order.count(BASELINE) < min_baseline:
        order.append(BASELINE)
    return order


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="deit_tiny_patch16_224")
    p.add_argument("--methods", nargs="+", default=[BASELINE])
    p.add_argument("--min-baseline-blocks", type=int, default=MIN_BASELINE_BLOCKS)
    args = p.parse_args()
    setup()

    state_before = system_state()
    run_id = state_before["timestamp"].replace(":", "-")
    print(f"run {run_id}  idle CPU {state_before['idle_cpu_percent']:.1f}%  "
          f"AC power {state_before['on_ac_power']}")

    built = {m: models.build(args.model, m) for m in set(args.methods) | {BASELINE}}
    x = timing_input(built[BASELINE][1])

    blocks, runs = [], []
    for i, method in enumerate(schedule(args.methods, args.min_baseline_blocks)):
        r = time_block(built[method][0], x)
        times = r.pop("times_ms")
        blocks.append({"run_id": run_id, "block": i, "model": args.model, "method": method, **r})
        runs += [{"run_id": run_id, "block": i, "method": method, "iter": k, "latency_ms": t}
                 for k, t in enumerate(times)]
        print(f"block {i:2d} {method:12s} median {r['latency_median_ms']:7.2f} ms  "
              f"p95 {r['latency_p95_ms']:7.2f} ms  ({r['start_time']})")

    blocks_df = pd.DataFrame(blocks)
    runs_df = pd.DataFrame(runs)
    drift = drift_check(blocks_df[blocks_df.method == BASELINE].latency_median_ms.tolist())

    summary = []
    for method in args.methods:
        t = runs_df[runs_df.method == method].latency_ms
        b = blocks_df[blocks_df.method == method]
        median = float(t.median())
        summary.append({
            "model": args.model,
            "method": method,
            "run_id": run_id,
            "start_time": b.start_time.min(),
            "end_time": b.end_time.max(),
            "n_blocks": len(b),
            "n_timed_runs": len(t),
            "warmup_per_block": int(b.warmup.iloc[0]),
            "latency_median_ms": median,
            "latency_p95_ms": float(t.quantile(0.95)),
            "throughput_img_per_s": 1000.0 / median,
            "size_on_disk_mb": size_on_disk_mb(built[method][0]),
            "torch_threads": torch.get_num_threads(),
            "idle_cpu_percent_before": state_before["idle_cpu_percent"],
            "baseline_drift_spread": drift["relative_spread"],
        })
    del built

    # Peak memory: each configuration in its own fresh process, after timing.
    for row in summary:
        row.update(measure_memory(args.model, row["method"]))

    meta = {"run_id": run_id, "model": args.model, "methods": args.methods,
            "system_state_before": state_before, "system_state_after": system_state(),
            "drift": drift, "end_time": now()}

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    blocks_df.to_csv(RAW_DIR / f"{run_id}_blocks.csv", index=False)
    runs_df.to_csv(RAW_DIR / f"{run_id}_runs.csv", index=False)
    pd.DataFrame(summary).to_csv(RAW_DIR / f"{run_id}_summary.csv", index=False)
    (RAW_DIR / f"{run_id}_meta.json").write_text(json.dumps(meta, indent=2))

    print(f"\nbaseline block medians (ms): {[round(m, 2) for m in drift['baseline_block_medians_ms']]}")
    print(f"relative spread {drift['relative_spread']:.1%} (threshold {DRIFT_THRESHOLD:.0%})")
    if drift["drift_detected"]:
        print("DRIFT DETECTED: summary NOT written to results/latency.csv. Raw logs kept.")
    else:
        for row in summary:
            upsert(SUMMARY_CSV, row, ["model", "method"])
        print(f"No drift. Summary written to {SUMMARY_CSV}")
    print(pd.DataFrame(summary).drop(columns=["run_id", "start_time", "end_time"]).T.to_string())


if __name__ == "__main__":
    main()
