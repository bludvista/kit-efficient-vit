"""Measurement helpers shared by all experiments.

- time_block: warm-up runs, then timed runs, batch size 1.
- drift_check: compares repeated baseline blocks to detect thermal slowdown.
- size_on_disk_mb: size of the saved state_dict.
- measure_memory: runs memory_worker.py in a fresh process and returns its result.
"""
import datetime
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import psutil
import torch

WARMUP = 10
RUNS = 50
# Maximum allowed relative spread of baseline-block medians, (max - min) / min.
DRIFT_THRESHOLD = 0.10


def now() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


# Names confirmed by the user in Windows Settings for GUIDs seen on this laptop.
POWER_OVERLAY_NAMES = {
    "961cc777-2547-4f9d-8174-7d86181b8a7a":
        "Best power efficiency (confirmed by user in Windows Settings, 2026-10-14)",
}


def power_mode() -> dict:
    """Windows power plan and power-mode overlay (raw GUIDs from the registry)."""
    state = {}
    try:
        import winreg
        key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                             r"SYSTEM\CurrentControlSet\Control\Power\User\PowerSchemes")
        for name in ["ActivePowerScheme", "ActiveOverlayAcPowerScheme", "ActiveOverlayDcPowerScheme"]:
            try:
                state[name] = winreg.QueryValueEx(key, name)[0]
            except FileNotFoundError:
                state[name] = None
    except (ImportError, OSError):
        pass
    state["overlay_ac_name"] = POWER_OVERLAY_NAMES.get(
        state.get("ActiveOverlayAcPowerScheme"), "unknown (not confirmed)")
    battery = psutil.sensors_battery()
    state["on_ac_power"] = None if battery is None else bool(battery.power_plugged)
    return state


def system_state(idle_seconds: float = 3.0) -> dict:
    """Snapshot taken right before a timing run: power settings and idle CPU load."""
    return {
        "timestamp": now(),
        "idle_cpu_percent": psutil.cpu_percent(interval=idle_seconds),
        **power_mode(),
    }


@torch.inference_mode()
def time_block(model, x, warmup: int = WARMUP, runs: int = RUNS) -> dict:
    """Time `runs` forward passes after `warmup` untimed ones. Times in ms."""
    start = now()
    for _ in range(warmup):
        model(x)
    times = np.empty(runs)
    for i in range(runs):
        t0 = time.perf_counter()
        model(x)
        times[i] = (time.perf_counter() - t0) * 1000.0
    median = float(np.median(times))
    return {
        "start_time": start,
        "end_time": now(),
        "warmup": warmup,
        "runs": runs,
        "latency_median_ms": median,
        "latency_p95_ms": float(np.percentile(times, 95)),
        "latency_mean_ms": float(times.mean()),
        "latency_min_ms": float(times.min()),
        "latency_max_ms": float(times.max()),
        "throughput_img_per_s": 1000.0 / median,
        "times_ms": times.tolist(),
    }


def drift_check(baseline_medians: list[float], threshold: float = DRIFT_THRESHOLD) -> dict:
    lo, hi = min(baseline_medians), max(baseline_medians)
    spread = (hi - lo) / lo
    return {
        "baseline_block_medians_ms": baseline_medians,
        "relative_spread": spread,
        "threshold": threshold,
        "drift_detected": spread > threshold,
    }


def size_on_disk_mb(model) -> float:
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "model.pt"
        torch.save(model.state_dict(), path)
        return path.stat().st_size / 1024**2


def measure_memory(model_name: str, method: str, timeout_s: int = 600) -> dict:
    """Peak memory of one configuration, measured in a fresh Python process."""
    worker = Path(__file__).with_name("memory_worker.py")
    out = subprocess.run(
        [sys.executable, "-u", str(worker), "--model", model_name, "--method", method],
        capture_output=True, text=True, timeout=timeout_s,
        env={**os.environ, "PYTHONWARNINGS": "ignore"},
    )
    if out.returncode != 0:
        raise RuntimeError(f"memory worker failed:\n{out.stderr[-2000:]}")
    # The worker prints exactly one JSON line prefixed with RESULT.
    line = next(l for l in out.stdout.splitlines() if l.startswith("RESULT "))
    return json.loads(line[len("RESULT "):])
