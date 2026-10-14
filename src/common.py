"""Shared settings used by every experiment: paths, seeds, thread count."""
import os
import random
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
PLOTS_DIR = ROOT / "plots"

# The laptop CPU (Ryzen 5 3500U) has 4 physical cores / 8 logical threads.
# We fix intra-op threads to the physical core count for every measurement.
NUM_THREADS = 4
SEED = 0


def setup(seed: int = SEED, num_threads: int = NUM_THREADS) -> None:
    """Fix seeds and thread count. Call at the start of every script."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(num_threads)
    os.environ["OMP_NUM_THREADS"] = str(num_threads)
