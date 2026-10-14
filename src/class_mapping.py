"""Build and verify the Imagenette -> ImageNet-1k class index mapping.

The mapping is never written by hand. Two independent ImageNet-1k index lists
are compared over all 1000 indices; only if they agree completely do we look up
the 10 Imagenette synset IDs (taken from the folder names).

Source A: timm's bundled list (timm/data/_info/imagenet_synsets.txt).
Source B: Keras' imagenet_class_index.json (Google TensorFlow storage).
"""
import csv
import json
import urllib.request

from timm.data import ImageNetInfo

from common import DATA_DIR, RESULTS_DIR

KERAS_URL = "https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json"
KERAS_FILE = DATA_DIR / "downloads" / "imagenet_class_index.json"
VAL_DIR = DATA_DIR / "imagenette2-160" / "val"
MAPPING_CSV = RESULTS_DIR / "imagenette_class_mapping.csv"


def load_keras_synsets() -> list[str]:
    if not KERAS_FILE.exists():
        KERAS_FILE.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(KERAS_URL, KERAS_FILE)
    index = json.loads(KERAS_FILE.read_text())
    return [index[str(i)][0] for i in range(len(index))]


def build_mapping() -> list[dict]:
    info = ImageNetInfo()
    timm_synsets = info.label_names()
    keras_synsets = load_keras_synsets()

    assert len(timm_synsets) == 1000, len(timm_synsets)
    assert len(keras_synsets) == 1000, len(keras_synsets)
    mismatches = [i for i in range(1000) if timm_synsets[i] != keras_synsets[i]]
    assert not mismatches, f"Index lists disagree at {mismatches[:10]}"
    print("Sources A (timm) and B (Keras) agree on all 1000 indices.")

    wnids = sorted(d.name for d in VAL_DIR.iterdir() if d.is_dir())
    assert len(wnids) == 10, wnids
    rows = []
    for imagenette_label, wnid in enumerate(wnids):
        idx = timm_synsets.index(wnid)
        rows.append({
            "imagenette_label": imagenette_label,
            "wnid": wnid,
            "imagenet_index": idx,
            "description": info.index_to_description(idx),
        })
    return rows


def load_mapping() -> list[dict]:
    """Read the verified mapping written by this script."""
    with open(MAPPING_CSV, newline="") as f:
        return [
            {**r, "imagenette_label": int(r["imagenette_label"]),
             "imagenet_index": int(r["imagenet_index"])}
            for r in csv.DictReader(f)
        ]


if __name__ == "__main__":
    rows = build_mapping()
    with open(MAPPING_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for r in rows:
        print(f"  {r['imagenette_label']}  {r['wnid']}  -> ImageNet {r['imagenet_index']:4d}  {r['description']}")
    print(f"Wrote {MAPPING_CSV}")
