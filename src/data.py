"""Imagenette validation data with the model's own timm preprocessing."""
import torch
from PIL import Image
from timm.data import create_transform
from torch.utils.data import Dataset

from class_mapping import load_mapping
from common import DATA_DIR

VAL_DIR = DATA_DIR / "imagenette2-160" / "val"


class ImagenetteVal(Dataset):
    """Returns (image tensor, imagenette label 0-9, ImageNet index, file name)."""

    def __init__(self, data_config: dict, indices: list[int] | None = None):
        self.transform = create_transform(**data_config, is_training=False)
        self.samples = []
        for row in load_mapping():
            for f in sorted((VAL_DIR / row["wnid"]).glob("*.JPEG")):
                self.samples.append((f, row["imagenette_label"], row["imagenet_index"]))
        if indices is not None:
            self.samples = [self.samples[i] for i in indices]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label, imagenet_index = self.samples[i]
        with Image.open(path) as im:
            x = self.transform(im.convert("RGB"))
        return x, label, imagenet_index, path.name


def timing_input(data_config: dict) -> torch.Tensor:
    """A fixed real image (first validation image), batch size 1, for timing."""
    x, *_ = ImagenetteVal(data_config)[0]
    return x.unsqueeze(0)
