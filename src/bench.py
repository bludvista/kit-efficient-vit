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
import pandas as pd
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


WATCHED_APPS = ["chrome", "msedge", "firefox", "brave", "opera", "code", "onedrive"]


def running_apps() -> dict:
    """Number of running processes per watched app (browsers, VS Code, OneDrive)."""
    counts = {app: 0 for app in WATCHED_APPS}
    for p in psutil.process_iter(["name"]):
        name = (p.info["name"] or "").lower().removesuffix(".exe")
        if name in counts:
            counts[name] += 1
    return counts


def system_state(idle_seconds: float = 3.0) -> dict:
    """Snapshot taken right before a timing run: power settings, idle CPU load
    and which background apps are running."""
    return {
        "timestamp": now(),
        "idle_cpu_percent": psutil.cpu_percent(interval=idle_seconds),
        **power_mode(),
        "running_apps": running_apps(),
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


@torch.inference_mode()
def token_counts(model, x) -> dict:
    """Tokens entering the first block and leaving each block, via forward hooks."""
    counts, inputs = [], []
    hooks = [b.register_forward_hook(lambda m, i, o: counts.append(o.shape[1])) for b in model.blocks]
    hooks.append(model.blocks[0].register_forward_pre_hook(lambda m, i: inputs.append(i[0].shape[1])))
    model(x)
    for h in hooks:
        h.remove()
    return {
        "tokens_in": inputs[0],
        "tokens_out": counts[-1],
        "tokens_mean_per_block": float(np.mean(counts)),
    }


def drift_check_rolling(baseline_medians: list[float], window: int = 3,
                        threshold: float = DRIFT_THRESHOLD) -> dict:
    """Rule B (defined 2026-10-15 after the rejected run 2026-10-15T14-28-48; used
    only for later runs). Drift = spread of the rolling median over `window`
    consecutive baseline blocks > threshold. A single slow (or fast) block does not
    move a rolling median of 3, so isolated spikes do not fail the run, but they
    are counted: a block is an ignored spike if it differs by more than
    `threshold` from the centred median of itself and its neighbours."""
    m = pd.Series(baseline_medians, dtype=float)
    rolling = m.rolling(window).median().dropna()
    spread = float((rolling.max() - rolling.min()) / rolling.min())
    local = m.rolling(window, center=True, min_periods=2).median()
    dev = (m - local).abs() / local
    spikes = [{"baseline_index": int(i), "median_ms": float(m[i]),
               "local_median_ms": float(local[i]), "deviation": float(dev[i])}
              for i in m.index[dev > threshold]]
    raw = drift_check(baseline_medians, threshold)
    return {
        "rule": f"rolling median of {window} consecutive baseline blocks, spread <= {threshold:.0%}",
        "baseline_block_medians_ms": list(map(float, baseline_medians)),
        "rolling_medians_ms": rolling.tolist(),
        "relative_spread": spread,
        "threshold": threshold,
        "drift_detected": spread > threshold,
        "ignored_spikes": spikes,
        "n_ignored_spikes": len(spikes),
        "raw_spread_old_rule": raw["relative_spread"],
    }


def size_on_disk_mb(model) -> float:
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "model.pt"
        torch.save(model.state_dict(), path)
        return path.stat().st_size / 1024**2


MEMORY_CACHE = Path(__file__).resolve().parent.parent / "data" / "cache" / "memory"


@torch.inference_mode()
def save_and_verify(model, x, tag: str) -> tuple[Path, Path]:
    """Save the model (whole module, e.g. already quantized) and the input, and
    check that the saved model reloads to identical outputs."""
    MEMORY_CACHE.mkdir(parents=True, exist_ok=True)
    model_file, input_file = MEMORY_CACHE / f"{tag}.pt", MEMORY_CACHE / f"{tag}_input.pt"
    torch.save(model, model_file)
    torch.save(x, input_file)
    reloaded = torch.load(model_file, weights_only=False).eval()
    identical = bool(torch.equal(model(x), reloaded(x)))
    del reloaded
    if not identical:
        raise RuntimeError(f"saved model {tag} does not reproduce the outputs")
    return model_file, input_file


def measure_memory(model, x, tag: str, num_threads: int, timeout_s: int = 600,
                   runs: int = 20) -> dict:
    """Peak memory of an already-built configuration, measured in a fresh process:
    save_and_verify here, then memory_worker.py loads and runs it in a new process."""
    model_file, input_file = save_and_verify(model, x, tag)
    return run_memory_worker(model_file, input_file, num_threads, timeout_s, runs)


def run_memory_worker(model_file: Path, input_file: Path, num_threads: int,
                      timeout_s: int = 600, runs: int = 20) -> dict:
    """Run memory_worker.py on a model saved (and verified) by save_and_verify."""
    worker = Path(__file__).with_name("memory_worker.py")
    out = subprocess.run(
        [sys.executable, "-u", str(worker), "--model-file", str(model_file),
         "--input-file", str(input_file), "--threads", str(num_threads), "--runs", str(runs)],
        capture_output=True, text=True, timeout=timeout_s,
        cwd=Path(__file__).parent, env={**os.environ, "PYTHONWARNINGS": "ignore"},
    )
    if out.returncode != 0:
        raise RuntimeError(f"memory worker failed:\n{out.stderr[-2000:]}")
    # The worker prints exactly one JSON line prefixed with RESULT.
    line = next(l for l in out.stdout.splitlines() if l.startswith("RESULT "))
    return {
        **json.loads(line[len("RESULT "):]),
        "saved_model_file_mb": model_file.stat().st_size / 1024**2,
        "saved_model_reproduces_outputs": True,  # save_and_verify raises otherwise
    }
