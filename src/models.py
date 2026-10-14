"""Build a model configuration by name.

A configuration is (model name, method). Phase 1 only has the FP32 baseline;
token merging and INT8 are added in later phases.
"""
import timm
import torch
from timm.data import resolve_data_config

METHODS = ["fp32"]


def build(model_name: str, method: str = "fp32") -> tuple[torch.nn.Module, dict]:
    """Return (model in eval mode, timm data config)."""
    if method not in METHODS:
        raise ValueError(f"Unknown method {method!r}; known: {METHODS}")
    model = timm.create_model(model_name, pretrained=True).eval()
    data_config = resolve_data_config({}, model=model)
    return model, data_config
