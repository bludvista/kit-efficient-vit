"""Measure peak memory of one saved configuration. Run in a fresh process (see bench.measure_memory).

The configuration was built in another process and saved with torch.save
(whole model, e.g. already quantized). Here we only load it, so no FP32 copy
of a quantized model is ever created in this process.

On Windows, psutil's peak_wset is the peak working set over the whole process
lifetime. It cannot be reset, so we read it at three points:
  1. after imports            (Python + torch + timm runtime)
  2. after loading the model  (adds the saved model file and its tensors)
  3. after inference          (adds activations)
All values include the Python/torch runtime; increments are also reported.
"""
import argparse
import json

import psutil
import torch

MB = 1024**2


def peak_mb() -> float:
    return psutil.Process().memory_info().peak_wset / MB


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model-file", required=True)
    p.add_argument("--input-file", required=True)
    p.add_argument("--runs", type=int, default=20)
    p.add_argument("--threads", type=int, required=True)
    args = p.parse_args()

    torch.manual_seed(0)
    torch.set_num_threads(args.threads)
    import timm  # noqa: F401  (classes needed to unpickle the model)
    import models
    models.register_tome_class()  # needed to unpickle ToMe models, see models.py

    after_import = peak_mb()
    model = torch.load(args.model_file, weights_only=False).eval()
    x = torch.load(args.input_file)
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
