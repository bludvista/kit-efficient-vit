"""Download Imagenette (160 px version) and check the validation split.

Source: https://github.com/fastai/imagenette (imagenette2-160.tgz).
The 160 px version has the shortest side resized to 160 px.
"""
import tarfile
import urllib.request
from collections import Counter

from PIL import Image

from common import DATA_DIR

URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz"
ARCHIVE = DATA_DIR / "downloads" / "imagenette2-160.tgz"
TARGET = DATA_DIR / "imagenette2-160"


def download() -> None:
    if TARGET.exists():
        print(f"Already present: {TARGET}")
        return
    ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
    if not ARCHIVE.exists():
        print(f"Downloading {URL}")
        urllib.request.urlretrieve(URL, ARCHIVE)
    print(f"Extracting to {DATA_DIR}")
    with tarfile.open(ARCHIVE) as tar:
        tar.extractall(DATA_DIR, filter="data")


def check() -> None:
    val = TARGET / "val"
    counts = Counter()
    sizes = Counter()
    for class_dir in sorted(val.iterdir()):
        files = sorted(class_dir.glob("*.JPEG"))
        counts[class_dir.name] = len(files)
        for f in files[:20]:  # sample a few per class for size statistics
            with Image.open(f) as im:
                sizes[min(im.size)] += 1
    print("Validation images per class (folder = WordNet synset ID):")
    for wnid, n in counts.items():
        print(f"  {wnid}: {n}")
    print(f"Classes: {len(counts)}, total images: {sum(counts.values())}")
    print(f"Shortest side in sampled images (px -> count): {dict(sizes)}")
    kitti = DATA_DIR / "kitti"
    n_kitti = len(list(kitti.glob("*.png"))) if kitti.exists() else 0
    print(f"KITTI folder present: {kitti.exists()}, PNG images: {n_kitti}")


if __name__ == "__main__":
    download()
    check()
