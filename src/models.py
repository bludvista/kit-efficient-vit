"""Build a model configuration by name.

Methods:
  fp32        timm model, unchanged (uses timm's fused SDPA attention)
  tome_r<R>   official ToMe patch (facebookresearch/ToMe, pinned commit) with
              constant r = R tokens merged per block. ToMe's attention is the
              explicit softmax(QK^T)V version, also for R = 0.
  int8        INT8 dynamic quantization of all nn.Linear layers with
              torch.ao.quantization.quantize_dynamic (qint8 weights, activations
              quantized at run time; engine onednn). Chosen in int8_api_check.py.
  tome_r<R>_int8  ToMe patch first, then the same INT8 quantization.
Compatibility with the installed timm is tested in tome_compat_check.py.
"""
import re
import subprocess
import sys
import warnings

import timm
import torch
from torch import nn
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


def register_tome_class():
    """ToMe builds its VisionTransformer subclass inside a function, which pickle
    cannot save. Create that class once with ToMe's own make_tome_class (code
    unchanged) and register it here as models.ToMeVisionTransformer, so a patched
    model can be saved with torch.save and loaded in another process (call this
    before torch.load there)."""
    from timm.models.vision_transformer import VisionTransformer
    _import_tome()
    from tome.patch.timm import make_tome_class
    module = sys.modules[__name__]
    if not hasattr(module, "ToMeVisionTransformer"):
        cls = make_tome_class(VisionTransformer)
        cls.__module__, cls.__qualname__ = __name__, "ToMeVisionTransformer"
        module.ToMeVisionTransformer = cls
    return module.ToMeVisionTransformer


def _use_registered_tome_class(model) -> None:
    from timm.models.vision_transformer import VisionTransformer
    # make_tome_class(model.__class__) == make_tome_class(VisionTransformer) here
    assert type(model).__mro__[1] is VisionTransformer, type(model).__mro__
    model.__class__ = register_tome_class()


def parse_method(method: str) -> dict:
    m = re.fullmatch(r"(?:fp32|int8)|tome_r(\d+)(_int8)?", method)
    if m is None:
        raise ValueError(f"Unknown method {method!r}")
    return {"tome_r": None if m.group(1) is None else int(m.group(1)),
            "int8": method == "int8" or m.group(2) is not None}


def quantize_int8(model: torch.nn.Module) -> torch.nn.Module:
    """INT8 dynamic quantization of all nn.Linear layers (in place)."""
    with warnings.catch_warnings():
        # torch.ao.quantization is deprecated in favour of torchao; see NOTES.md.
        warnings.simplefilter("ignore", category=FutureWarning)
        warnings.simplefilter("ignore", category=DeprecationWarning)
        return torch.ao.quantization.quantize_dynamic(
            model, {nn.Linear}, dtype=torch.qint8, inplace=True)


def build(model_name: str, method: str = "fp32") -> tuple[torch.nn.Module, dict]:
    """Return (model in eval mode, timm data config)."""
    spec = parse_method(method)
    model = timm.create_model(model_name, pretrained=True).eval()
    data_config = resolve_data_config({}, model=model)
    if spec["tome_r"] is not None:
        tome = _import_tome()
        tome.patch.timm(model)  # prop_attn=True (default) for off-the-shelf models
        _use_registered_tome_class(model)
        model.r = spec["tome_r"]
    if spec["int8"]:
        model = quantize_int8(model)
    return model, data_config
