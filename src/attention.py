"""Experiment 4: attention scaling with random inputs (synthetic).

naive: softmax(Q K^T / sqrt(d)) V written out, builds the N x N score matrix.
sdpa:  torch.nn.functional.scaled_dot_product_attention (no mask, no dropout).
Shape: batch 1, HEADS = 3 heads, head dim 64 (= DeiT-Tiny), float32, sequence
length N from SEQ_LENS. Memory-efficient attention kernels mainly target GPUs; these
are CPU results only. The SDPA kernel actually used on this CPU is recorded by the
check subcommand (torch.profiler op names).

Limits (decided before running, same as resolution.py): largest tensor > 1.5 GB or
estimated peak > 70% of available RAM -> "skipped: estimated memory too high"; one
call > 30 s -> "skipped: time limit"; then larger N of that method are skipped too.
Drift: rule B unchanged (see NOTES.md). Decided before the run: if it fails, the run
is reported with its measured noise and not repeated.

Subcommands:
  check   naive vs sdpa outputs agree; which SDPA kernel runs
  timing  heat-up, sdpa reference blocks at REF_N between configurations
  memory  peak memory of one call, each (method, N) in a fresh process
Outputs in results/attention/.
"""
import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import psutil
import torch
import torch.nn.functional as F

from accuracy import upsert
from bench import DRIFT_THRESHOLD, drift_check_rolling, now, system_state, time_block
from common import NUM_THREADS, RESULTS_DIR, setup

HEADS, HEAD_DIM = 3, 64
SEQ_LENS = [256 * 2**i for i in range(9)]  # 256 ... 65536
METHODS = ["naive", "sdpa"]
REF_N, REF_METHOD = 1024, "sdpa"
WARMUP_MAX, WARMUP_MIN, WARMUP_BUDGET_S = 10, 2, 20.0
RUNS_MAX, RUNS_MIN, TIMED_BUDGET_S = 50, 5, 60.0
HEATUP_SECONDS = 180.0
TIME_LIMIT_S = 30.0
MAX_TENSOR_MB = 1.5 * 1024
RAM_FRACTION = 0.7
RUNTIME_MB = 250.0  # Python + torch, no model
MB = 1024**2
OUT_DIR = RESULTS_DIR / "attention"


def naive(q, k, v):
    attn = (q @ k.transpose(-2, -1)) / math.sqrt(q.shape[-1])
    return attn.softmax(dim=-1) @ v


def sdpa(q, k, v):
    return F.scaled_dot_product_attention(q, k, v)


FUNCS = {"naive": naive, "sdpa": sdpa}


def inputs(n: int):
    g = torch.Generator().manual_seed(0)
    return tuple(torch.randn(1, HEADS, n, HEAD_DIM, generator=g) for _ in range(3))


def memory_estimate(method: str, n: int) -> dict:
    qkv_out = 4 * HEADS * n * HEAD_DIM * 4 / MB
    if method == "naive":
        largest = HEADS * n * n * 4 / MB
        extra = qkv_out + 2 * largest  # scores and their softmax alive together
    else:
        largest = qkv_out / 4
        extra = 2 * qkv_out
    available = psutil.virtual_memory().available / MB
    total = RUNTIME_MB + extra
    reason = None
    if largest > MAX_TENSOR_MB:
        reason = (f"skipped: estimated memory too high (largest tensor {largest:.0f} MB "
                  f"> {MAX_TENSOR_MB:.0f} MB)")
    elif total > RAM_FRACTION * available:
        reason = (f"skipped: estimated memory too high (peak {total:.0f} MB > "
                  f"{RAM_FRACTION:.0%} of available RAM {available:.0f} MB)")
    return {"est_largest_tensor_mb": largest, "est_peak_mb": total,
            "available_ram_mb_before": available, "skip_reason": reason}


class Call:
    """Adapter so bench.time_block can time f(q, k, v) as model(x)."""

    def __init__(self, f):
        self.f = f

    def __call__(self, qkv):
        return self.f(*qkv)


@torch.inference_mode()
def cmd_check(args) -> None:
    setup()
    out = {"time": now(), "torch": torch.__version__, "checks": []}
    for n in [256, 1024, 2048]:
        q, k, v = inputs(n)
        a, b = naive(q, k, v), sdpa(q, k, v)
        out["checks"].append({"n": n, "max_abs_diff": float((a - b).abs().max()),
                              "allclose_atol1e-5": bool(torch.allclose(a, b, atol=1e-5))})
    q, k, v = inputs(1024)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
        sdpa(q, k, v)
    out["sdpa_cpu_ops"] = sorted({e.key for e in prof.key_averages()})
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "check.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


@torch.inference_mode()
def cmd_timing(args) -> None:
    setup()
    state_before = system_state()
    run_id = state_before["timestamp"].replace(":", "-")
    print(f"run {run_id}  idle CPU {state_before['idle_cpu_percent']:.1f}%  "
          f"AC power {state_before['on_ac_power']}")
    ref_call, ref_x = Call(FUNCS[REF_METHOD]), inputs(REF_N)
    t0, n_pass, heat_start = time.perf_counter(), 0, now()
    while time.perf_counter() - t0 < args.heatup_seconds:
        ref_call(ref_x)
        n_pass += 1
    heatup = {"start_time": heat_start, "end_time": now(), "seconds": args.heatup_seconds,
              "forward_passes": n_pass}
    print(f"heat-up: {n_pass} untimed calls in {args.heatup_seconds:.0f} s")

    blocks, runs, rows = [], [], []

    def record(kind, method, n, r):
        times = r.pop("times_ms")
        i = len(blocks)
        blocks.append({"run_id": run_id, "block": i, "kind": kind, "method": method, "seq_len": n, **r})
        runs.extend({"run_id": run_id, "block": i, "iter": k, "latency_ms": t} for k, t in enumerate(times))
        return i

    def reference_block():
        r = time_block(ref_call, ref_x)
        i = record("reference", REF_METHOD, REF_N, r)
        print(f"block {i:3d} reference {REF_METHOD} N={REF_N} median {r['latency_median_ms']:8.3f} ms")

    reference_block()
    stopped = {}
    for n in args.seq_lens:
        for method in args.methods:
            row = {"method": method, "seq_len": n, "heads": HEADS, "head_dim": HEAD_DIM,
                   "run_id": run_id, "time": now(), "status": "ok"}
            est = memory_estimate(method, n)
            row.update(est)
            reason = stopped.get(method) and f"skipped: a smaller N was already skipped ({stopped[method]})"
            reason = reason or est["skip_reason"]
            if reason is None:
                call, x = Call(FUNCS[method]), inputs(n)
                t = time.perf_counter()
                call(x)
                t1 = time.perf_counter() - t
                row["first_call_s"] = t1
                if t1 > TIME_LIMIT_S:
                    reason = f"skipped: time limit (first call {t1:.1f} s > {TIME_LIMIT_S:.0f} s)"
                else:
                    reference_block()
                    warmup = max(WARMUP_MIN, min(WARMUP_MAX, int(WARMUP_BUDGET_S / t1)))
                    n_runs = max(RUNS_MIN, min(RUNS_MAX, int(TIMED_BUDGET_S / t1)))
                    r = time_block(call, x, warmup=warmup - 1, runs=n_runs)
                    row.update({"warmup": warmup, "runs": n_runs,
                                "latency_median_ms": r["latency_median_ms"],
                                "latency_p95_ms": r["latency_p95_ms"],
                                "latency_min_ms": r["latency_min_ms"],
                                "latency_max_ms": r["latency_max_ms"]})
                    i = record("config", method, n, r)
                    row["block"] = i
                    print(f"block {i:3d} {method:5s} N={n:6d} median {row['latency_median_ms']:10.3f} ms "
                          f"({warmup} warm-up, {n_runs} runs)")
                del x
            if reason is not None:
                row["status"] = reason
                stopped.setdefault(method, reason)
                print(f"          {method:5s} N={n:6d} {reason}")
            rows.append(row)
    reference_block()

    blocks_df, runs_df = pd.DataFrame(blocks), pd.DataFrame(runs)
    drift = drift_check_rolling(blocks_df[blocks_df.kind == "reference"].latency_median_ms.tolist())
    for row in rows:
        row.update({"torch_threads": torch.get_num_threads(),
                    "idle_cpu_percent_before": state_before["idle_cpu_percent"],
                    "reference_drift_spread": drift["relative_spread"],
                    "reference_raw_spread": drift["raw_spread_old_rule"],
                    "drift_detected": drift["drift_detected"]})
    meta = {"run_id": run_id, "system_state_before": state_before, "system_state_after": system_state(),
            "heatup": heatup, "drift": drift, "reference": f"{REF_METHOD} N={REF_N}", "end_time": now()}
    raw = args.out / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    blocks_df.to_csv(raw / f"{run_id}_timing_blocks.csv", index=False)
    runs_df.to_csv(raw / f"{run_id}_timing_runs.csv", index=False)
    (raw / f"{run_id}_timing_meta.json").write_text(json.dumps(meta, indent=2))
    # Decided before the run: written even if rule B fails; drift_detected is in every row.
    pd.DataFrame(rows).to_csv(args.out / "timing.csv", index=False)
    print(f"\nreference rolling-median spread {drift['relative_spread']:.1%} "
          f"(threshold {DRIFT_THRESHOLD:.0%}); raw {drift['raw_spread_old_rule']:.1%}; "
          f"drift detected: {drift['drift_detected']}")
    print(f"Wrote {args.out / 'timing.csv'}")


def peak_mb() -> float:
    return psutil.Process().memory_info().peak_wset / MB


@torch.inference_mode()
def cmd_worker(args) -> None:
    """Fresh process: peak working set before and after one attention call."""
    torch.manual_seed(0)
    torch.set_num_threads(args.threads)
    q, k, v = inputs(args.seq_len)
    before = peak_mb()
    FUNCS[args.method](q, k, v)
    after = peak_mb()
    print("RESULT " + json.dumps({"peak_after_inputs_mb": before, "peak_total_mb": after,
                                  "call_increment_mb": after - before}))


def cmd_memory(args) -> None:
    timing = pd.read_csv(args.out / "timing.csv")
    run_id = now().replace(":", "-")
    rows = []
    for n in args.seq_lens:
        for method in args.methods:
            row = {"method": method, "seq_len": n, "heads": HEADS, "head_dim": HEAD_DIM,
                   "run_id": run_id, "time": now(), "status": "ok",
                   "measurement": "fresh process, one call"}
            t = timing[(timing.method == method) & (timing.seq_len == n)]
            est = memory_estimate(method, n)
            row.update(est)
            if t.empty or t.iloc[0].status != "ok":
                row["status"] = "not run (" + ("no timing result" if t.empty else t.iloc[0].status) + ")"
            elif est["skip_reason"]:
                row["status"] = est["skip_reason"]
            else:
                out = subprocess.run([sys.executable, "-u", __file__, "worker", "--method", method,
                                      "--seq-len", str(n), "--threads", str(NUM_THREADS)],
                                     capture_output=True, text=True, timeout=600)
                if out.returncode != 0:
                    raise RuntimeError(out.stderr[-2000:])
                line = next(l for l in out.stdout.splitlines() if l.startswith("RESULT "))
                row.update(json.loads(line[len("RESULT "):]))
            rows.append(row)
            msg = (f"peak {row['peak_total_mb']:8.2f} MB (+call {row['call_increment_mb']:8.2f})"
                   if row["status"] == "ok" else row["status"])
            print(f"{method:5s} N={n:6d}  {msg}")
    raw = args.out / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(raw / f"{run_id}_memory.csv", index=False)
    for row in rows:
        upsert(args.out / "memory.csv", row, ["method", "seq_len"])
    print(f"Wrote {args.out / 'memory.csv'}")


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ["check", "timing", "memory"]:
        s = sub.add_parser(name)
        s.add_argument("--seq-lens", nargs="+", type=int, default=SEQ_LENS)
        s.add_argument("--methods", nargs="+", default=METHODS)
        s.add_argument("--out", type=Path, default=OUT_DIR)
    sub.choices["timing"].add_argument("--heatup-seconds", type=float, default=HEATUP_SECONDS)
    s = sub.add_parser("worker")
    s.add_argument("--method", required=True)
    s.add_argument("--seq-len", type=int, required=True)
    s.add_argument("--threads", type=int, required=True)
    args = p.parse_args()
    {"check": cmd_check, "timing": cmd_timing, "memory": cmd_memory, "worker": cmd_worker}[args.cmd](args)


if __name__ == "__main__":
    main()
