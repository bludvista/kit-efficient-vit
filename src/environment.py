"""Record the software and hardware environment to results/environment.json."""
import datetime
import json
import platform
import subprocess
import sys

import psutil
import torch

from bench import power_mode
from common import NUM_THREADS, RESULTS_DIR, setup


def cpu_model() -> str:
    # platform.processor() on Windows only gives the family string, so ask WMI.
    if platform.system() == "Windows":
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "(Get-CimInstance Win32_Processor).Name"],
            capture_output=True, text=True,
        )
        return out.stdout.strip()
    return platform.processor()


def version(module_name: str):
    try:
        module = __import__(module_name)
        return module.__version__
    except ImportError:
        return None


def main() -> None:
    setup()
    env = {
        "date": datetime.datetime.now().isoformat(timespec="seconds"),
        "cpu_model": cpu_model(),
        "cpu_physical_cores": psutil.cpu_count(logical=False),
        "cpu_logical_cores": psutil.cpu_count(logical=True),
        "ram_total_gb": round(psutil.virtual_memory().total / 1024**3, 2),
        "os": platform.platform(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torchvision": version("torchvision"),
        "timm": version("timm"),
        "onnxruntime": version("onnxruntime"),
        "numpy": version("numpy"),
        "torch_num_threads": torch.get_num_threads(),
        "configured_num_threads": NUM_THREADS,
        "torch_quantized_engines": torch.backends.quantized.supported_engines,
        "torch_cpu_capability": torch.backends.cpu.get_cpu_capability(),
        "power": power_mode(),
        "torch_parallel_info": torch.__config__.parallel_info(),
    }
    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / "environment.json"
    path.write_text(json.dumps(env, indent=2))
    print(json.dumps({k: v for k, v in env.items() if k != "torch_parallel_info"}, indent=2))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
