"""Measure peak memory of one configuration. Run in a fresh process (see bench.measure_memory).

On Windows, psutil's peak_wset is the peak working set over the whole process
lifetime. It cannot be reset, so we read it at three points:
  1. after imports            (Python + torch + timm runtime)
  2. after the model is built (adds weight loading)
  3. after inference          (adds activations)
All values include the Python/torch runtime; increments are also reported.
"""
import argparse
import json

import psutil
import torch

from common import setup

MB = 1024**2


def peak_mb() -> float:
    return psutil.Process().memory_info().peak_wset / MB


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--method", required=True)
    p.add_argument("--runs", type=int, default=20)
    args = p.parse_args()

    setup()
    import models
    from data import timing_input

    after_import = peak_mb()
    model, data_config = models.build(args.model, args.method)
    x = timing_input(data_config)
    after_load = peak_mb()
    with torch.inference_mode():
        for _ in range(args.runs):
            model(x)
    after_inference = peak_mb()

    result = {
        "peak_after_import_mb": after_import,
        "peak_after_load_mb": after_load,
        "peak_total_mb": after_inference,
        "load_increment_mb": after_load - after_import,
        "inference_increment_mb": after_inference - after_load,
        "memory_inference_runs": args.runs,
    }
    print("RESULT " + json.dumps(result))


if __name__ == "__main__":
    main()
