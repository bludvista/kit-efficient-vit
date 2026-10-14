"""Build a model configuration by name.

Methods:
  fp32        timm model, unchanged (uses timm's fused SDPA attention)
  tome_r<R>   official ToMe patch (facebookresearch/ToMe, pinned commit) with
              constant r = R tokens merged per block. ToMe's attention is the
              explicit softmax(QK^T)V version, also for R = 0.
Compatibility with the installed timm is tested in tome_compat_check.py.
"""
import re
import subprocess
import sys

import timm
import torch
from timm.data import resolve_data_config

from common import DATA_DIR

TOME_DIR = DATA_DIR / "third_party" / "ToMe"
TOME_REPO = "https://github.com/facebookresearch/ToMe.git"
TOME_COMMIT = "af95e4b1befa172dadccd8c81e223b10090f9579"


def _import_tome():
    if not TOME_DIR.exists():
        subprocess.run(["git", "clone", "-q", TOME_REPO, str(TOME_DIR)], check=True)
        subprocess.run(["git", "-C", str(TOME_DIR), "checkout", "-q", TOME_COMMIT], check=True)
    head = subprocess.run(["git", "-C", str(TOME_DIR), "rev-parse", "HEAD"],
                          capture_output=True, text=True).stdout.strip()
    if head != TOME_COMMIT:
        raise RuntimeError(f"ToMe checkout is at {head}, expected {TOME_COMMIT}")
    if str(TOME_DIR) not in sys.path:
        sys.path.insert(0, str(TOME_DIR))
    import tome
    return tome


def parse_method(method: str) -> dict:
    m = re.fullmatch(r"fp32|tome_r(\d+)", method)
    if m is None:
        raise ValueError(f"Unknown method {method!r}")
    return {"tome_r": None if m.group(1) is None else int(m.group(1))}


def build(model_name: str, method: str = "fp32") -> tuple[torch.nn.Module, dict]:
    """Return (model in eval mode, timm data config)."""
    spec = parse_method(method)
    model = timm.create_model(model_name, pretrained=True).eval()
    data_config = resolve_data_config({}, model=model)
    if spec["tome_r"] is not None:
        tome = _import_tome()
        tome.patch.timm(model)  # prop_attn=True (default) for off-the-shelf models
        model.r = spec["tome_r"]
    return model, data_config
