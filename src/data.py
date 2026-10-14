"""Imagenette validation data with the model's own timm preprocessing."""
import numpy as np
import torch
from PIL import Image
from timm.data import create_transform
from torch.utils.data import Dataset
from torchvision import transforms

from class_mapping import load_mapping
from common import DATA_DIR

VAL_DIR = DATA_DIR / "imagenette2-160" / "val"
CACHE_DIR = DATA_DIR / "cache"


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


class CachedImagenetteVal(Dataset):
    """Same samples and outputs as ImagenetteVal, but the resize + centre crop is
    done once and stored as uint8 on disk. Only ToTensor's /255 and Normalize
    are applied per item, which reproduces the timm transform exactly
    (checked by verify_cache)."""

    def __init__(self, data_config: dict):
        self.base = ImagenetteVal(data_config)
        ops = self.base.transform.transforms
        # timm eval transform: Resize, CenterCrop, ToTensor (MaybeToTensor), Normalize
        self.geometric = transforms.Compose(ops[:-2])
        self.normalize = ops[-1]
        h, w = data_config["input_size"][1:]
        key = f"{h}x{w}_{data_config['interpolation']}_crop{data_config['crop_pct']}"
        path = CACHE_DIR / f"imagenette_val_{key}.npy"
        if not path.exists():
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            arr = np.lib.format.open_memmap(path, mode="w+", dtype=np.uint8,
                                            shape=(len(self.base), h, w, 3))
            for i, (f, *_) in enumerate(self.base.samples):
                with Image.open(f) as im:
                    arr[i] = np.asarray(self.geometric(im.convert("RGB")))
            arr.flush()
            del arr
        self.images = np.load(path, mmap_mode="r")

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        path, label, imagenet_index = self.base.samples[i]
        x = torch.from_numpy(np.array(self.images[i])).permute(2, 0, 1).float().div(255)
        return self.normalize(x), label, imagenet_index, path.name


def verify_cache(data_config: dict, n: int = 50) -> float:
    """Max abs difference between cached and original transform over n images."""
    cached, orig = CachedImagenetteVal(data_config), ImagenetteVal(data_config)
    idx = np.linspace(0, len(orig) - 1, n).astype(int)
    return max((cached[i][0] - orig[i][0]).abs().max().item() for i in idx)


def timing_input(data_config: dict) -> torch.Tensor:
    """A fixed real image (first validation image), batch size 1, for timing."""
    x, *_ = ImagenetteVal(data_config)[0]
    return x.unsqueeze(0)
