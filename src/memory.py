"""Peak memory per configuration, batch size 1, each in a fresh process.

Each configuration is built here (e.g. FP32 -> ToMe patch -> INT8), saved with
torch.save, checked to reload to identical outputs, and then loaded and run by
memory_worker.py in a new Python process. So the measured process never holds
an FP32 copy of a quantized model (the earlier method did; see NOTES.md).

Outputs (run_id = start timestamp):
  results/memory_raw/<run_id>.csv   all rows of this run
  results/memory.csv                one row per (model, method, input_size)
"""
import argparse

import pandas as pd
import torch

import models
from accuracy import upsert
from bench import measure_memory, now, system_state
from common import NUM_THREADS, RESULTS_DIR, setup
from data import timing_input

RAW_DIR = RESULTS_DIR / "memory_raw"
SUMMARY_CSV = RESULTS_DIR / "memory.csv"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="deit_tiny_patch16_224")
    p.add_argument("--methods", nargs="+", default=["fp32"])
    args = p.parse_args()
    setup()

    state = system_state(idle_seconds=1.0)
    run_id = state["timestamp"].replace(":", "-")
    rows = []
    for method in args.methods:
        model, cfg = models.build(args.model, method)
        x = timing_input(cfg)
        tag = f"{args.model}_{method}_{'x'.join(map(str, x.shape[2:]))}"
        r = measure_memory(model, x, tag, NUM_THREADS)
        del model
        row = {"model": args.model, "method": method,
               "input_size": "x".join(map(str, x.shape[2:])), "run_id": run_id,
               "time": now(), "measurement": "fresh process loads saved model", **r}
        rows.append(row)
        print(f"{method:14s} peak {r['peak_total_mb']:7.2f} MB  (import {r['peak_after_import_mb']:.2f}, "
              f"+load {r['load_increment_mb']:.2f}, +inference {r['inference_increment_mb']:.2f}; "
              f"file {r['saved_model_file_mb']:.2f} MB)")

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(RAW_DIR / f"{run_id}.csv", index=False)
    for row in rows:
        upsert(SUMMARY_CSV, row, ["model", "method", "input_size"])
    print(f"Wrote {SUMMARY_CSV}")


if __name__ == "__main__":
    main()
