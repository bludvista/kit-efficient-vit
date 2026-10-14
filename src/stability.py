"""Timing-stability test of this laptop: FP32 baseline only, no other configs.

One session = N consecutive baseline blocks (10 warm-up + 50 timed runs each).
Run several sessions some minutes apart to see the spread of block medians
within a session and between sessions. Used to choose the drift rule BEFORE
further latency measurements; nothing here goes into results/latency.csv.

Outputs: results/stability/<session_id>_blocks.csv, _runs.csv, _meta.json
"""
import argparse
import json

import pandas as pd

import models
from bench import system_state, time_block
from common import RESULTS_DIR, setup
from data import timing_input

OUT_DIR = RESULTS_DIR / "stability"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="deit_tiny_patch16_224")
    p.add_argument("--blocks", type=int, default=10)
    p.add_argument("--label", default="")
    args = p.parse_args()
    setup()

    before = system_state()
    session_id = before["timestamp"].replace(":", "-")
    model, cfg = models.build(args.model, "fp32")
    x = timing_input(cfg)

    blocks, runs = [], []
    for i in range(args.blocks):
        r = time_block(model, x)
        times = r.pop("times_ms")
        blocks.append({"session_id": session_id, "label": args.label, "block": i, **r})
        runs += [{"session_id": session_id, "block": i, "iter": k, "latency_ms": t}
                 for k, t in enumerate(times)]
    after = system_state()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(blocks)
    df.to_csv(OUT_DIR / f"{session_id}_blocks.csv", index=False)
    pd.DataFrame(runs).to_csv(OUT_DIR / f"{session_id}_runs.csv", index=False)
    med = df.latency_median_ms
    meta = {"session_id": session_id, "label": args.label, "model": args.model,
            "system_state_before": before, "system_state_after": after,
            "block_medians_ms": med.tolist(),
            "session_median_ms": float(pd.DataFrame(runs).latency_ms.median()),
            "relative_spread": float((med.max() - med.min()) / med.min())}
    (OUT_DIR / f"{session_id}_meta.json").write_text(json.dumps(meta, indent=2))

    print(f"session {session_id} ({args.label})  idle CPU before {before['idle_cpu_percent']:.1f}%  "
          f"apps {before['running_apps']}")
    print("block medians (ms):", [round(m, 2) for m in med])
    print(f"session median {meta['session_median_ms']:.2f} ms, "
          f"within-session spread {meta['relative_spread']:.1%}")


if __name__ == "__main__":
    main()
