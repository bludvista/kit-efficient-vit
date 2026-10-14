"""Experiment 1: top-1 accuracy on the Imagenette validation set.

Two numbers per configuration, both with the pre-trained ImageNet-1k head:
- restricted top-1: argmax over the 10 Imagenette class logits only
- unrestricted top-1: argmax over all 1000 ImageNet logits

Imagenette-160 images (shortest side 160 px) are upscaled by timm's eval
transform (resize shortest side to 224 / crop_pct, then centre crop 224).
Batch size here does not affect accuracy; latency is measured separately.
"""
import argparse
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

import models
from bench import now
from class_mapping import load_mapping
from common import RESULTS_DIR, SEED, setup
from data import CachedImagenetteVal

ACC_CSV = RESULTS_DIR / "accuracy.csv"
PRED_DIR = RESULTS_DIR / "predictions"


def upsert(csv_path, row: dict, keys: list[str]) -> None:
    """Replace the row with the same key values, or append it."""
    new = pd.DataFrame([row])
    if csv_path.exists():
        old = pd.read_csv(csv_path)
        mask = np.logical_and.reduce([old[k] == row[k] for k in keys])
        new = pd.concat([old[~mask], new], ignore_index=True)
    new.to_csv(csv_path, index=False)


@torch.inference_mode()
def evaluate(model_name: str, method: str, subset: int | None, batch_size: int) -> dict:
    model, data_config = models.build(model_name, method)
    ds = CachedImagenetteVal(data_config)  # identical to ImagenetteVal, see data.verify_cache
    indices = None
    if subset is not None and subset < len(ds):
        rng = np.random.default_rng(SEED)
        indices = sorted(rng.choice(len(ds), size=subset, replace=False).tolist())
        ds = Subset(ds, indices)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    imagenette_idx = torch.tensor([r["imagenet_index"] for r in load_mapping()])

    records = []
    start, t0 = now(), time.perf_counter()
    for x, label, imagenet_index, names in loader:
        logits = model(x)
        pred_restricted = logits[:, imagenette_idx].argmax(dim=1)
        pred_unrestricted = logits.argmax(dim=1)
        for i in range(len(label)):
            records.append({
                "file": names[i],
                "label": int(label[i]),
                "imagenet_index": int(imagenet_index[i]),
                "pred_restricted": int(pred_restricted[i]),
                "pred_unrestricted": int(pred_unrestricted[i]),
            })
    elapsed = time.perf_counter() - t0

    preds = pd.DataFrame(records)
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    preds.to_csv(PRED_DIR / f"{model_name}_{method}.csv", index=False)
    return {
        "model": model_name,
        "method": method,
        "n_images": len(preds),
        "subset": "full" if indices is None else f"random seed {SEED}",
        "top1_restricted": float((preds.pred_restricted == preds.label).mean()),
        "top1_unrestricted": float((preds.pred_unrestricted == preds.imagenet_index).mean()),
        "correct_restricted": int((preds.pred_restricted == preds.label).sum()),
        "correct_unrestricted": int((preds.pred_unrestricted == preds.imagenet_index).sum()),
        "input_size": "x".join(map(str, data_config["input_size"][1:])),
        "interpolation": data_config["interpolation"],
        "crop_pct": data_config["crop_pct"],
        "source_resolution": "160 (upscaled)",
        "eval_batch_size": batch_size,
        "eval_seconds": elapsed,
        "start_time": start,
        "end_time": now(),
        "torch_threads": torch.get_num_threads(),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="deit_tiny_patch16_224")
    p.add_argument("--method", default="fp32")
    p.add_argument("--subset", type=int, default=None, help="fixed random subset size")
    p.add_argument("--batch-size", type=int, default=32)
    args = p.parse_args()
    setup()
    row = evaluate(args.model, args.method, args.subset, args.batch_size)
    upsert(ACC_CSV, row, ["model", "method"])
    for k, v in row.items():
        print(f"{k:22s} {v}")


if __name__ == "__main__":
    main()
