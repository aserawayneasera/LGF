"""Original prepared datasets, strict file checks, and content provenance."""
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from core_bridge import load_core
from utils import sha256, signature

CLASSES = {"person", "bicycle", "car", "motorcycle", "bus", "truck"}


class CheckedOriginalDataset(load_core().COCODataset):
    def __getitem__(self, index):
        info = self.coco.loadImgs(self.ids[index])[0]
        path = Path(self.img_folder) / info["file_name"]
        if load_core().cv2.imread(str(path)) is None:
            raise OSError(f"Unable to decode dataset image: {path}")
        return super().__getitem__(index)


def dataset_paths(dataset, root):
    root = Path(root)
    if dataset in {"coco_nw", "coco_weather"}:
        tag = "non_weather" if dataset == "coco_nw" else "weather"
        folder = root / "COCO" / ("non_weather-mini" if dataset == "coco_nw" else "weather-mini")
        return {"train_images": folder / f"images/train2017_{tag}-2400k6c",
                "train_annotations": folder / f"annotations/mini_train2017_{tag}-2400k6c.json",
                "val_images": folder / f"images/val2017_{tag}-500k6c",
                "val_annotations": folder / f"annotations/mini_val2017_{tag}-500k6c.json"}
    if dataset == "acdc":
        folder = root / "ACDC-1/ACDC-1-NEW"
        return {"train_images": folder / "images/train", "train_annotations": folder / "annotations/mini_train.json",
                "val_images": folder / "images/val", "val_annotations": folder / "annotations/mini_val.json"}
    raise ValueError(dataset)


def inspect_split(image_root, annotations):
    image_root, annotations = Path(image_root), Path(annotations)
    doc = json.loads(annotations.read_text())
    categories = sorted(doc["categories"], key=lambda c: c["id"])
    if {c["name"] for c in categories} != CLASSES or len(categories) != 6:
        raise ValueError("Expected exactly the original six classes")
    images = doc["images"]
    ids = [im["id"] for im in images]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("Empty image list or duplicate image IDs")
    ann_ids = [a["id"] for a in doc["annotations"]]
    if len(set(ann_ids)) != len(ann_ids):
        raise ValueError("Duplicate annotation IDs")
    valid_ids, valid_cats = set(ids), {c["id"] for c in categories}
    for a in doc["annotations"]:
        if a["image_id"] not in valid_ids or a["category_id"] not in valid_cats:
            raise ValueError("Annotation references an unknown image/category")
        if len(a["bbox"]) != 4 or not np.isfinite(a["bbox"]).all():
            raise ValueError("Invalid bounding box")
    records = []
    for im in sorted(images, key=lambda i: i["id"]):
        p = image_root / im["file_name"]
        if not p.is_file():
            raise FileNotFoundError(p)
        records.append({"image_id": im["id"], "file_name": im["file_name"],
                        "sha256": sha256(p)})
    return {"image_count": len(images), "annotation_count": len(doc["annotations"]),
            "categories": [{"id": c["id"], "name": c["name"]} for c in categories],
            "annotations_sha256": sha256(annotations), "image_records": records,
            "content_sha256": signature({"annotations": sha256(annotations), "images": records})}


def audit(dataset, root, *, val_only=False):
    paths = dataset_paths(dataset, root)
    val = inspect_split(paths["val_images"], paths["val_annotations"])
    result = {"dataset": dataset, "paths": {k: str(v) for k, v in paths.items()}, "val": val,
              "lineage_limit": "Byte hashes detect identical files, not all transformed or related source images."}
    if not val_only:
        train = inspect_split(paths["train_images"], paths["train_annotations"])
        if train["categories"] != val["categories"]:
            raise ValueError("Train/validation category IDs or names differ")
        overlap = {r["sha256"] for r in train["image_records"]} & {r["sha256"] for r in val["image_records"]}
        if overlap:
            raise ValueError(f"Train/validation contain {len(overlap)} identical image file(s)")
        result["train"] = train
        result["train_val_identical_bytes"] = 0
    return result


def _worker_seed(worker_id):
    core = load_core()
    core.worker_init_fn(worker_id)
    info = torch.utils.data.get_worker_info()
    transform = getattr(info.dataset, "transforms", None)
    if hasattr(transform, "set_random_seed"):
        transform.set_random_seed(torch.initial_seed() % 2**32)


def make_loaders(audit_result, batch_size, workers, prefetch, seed, *, val_only=False):
    core = load_core()
    core.set_seed(seed)
    # The original reader substitutes black images after an I/O failure. The new
    # wrapper raises instead, so storage errors never become training samples.
    def dataset(split):
        p = audit_result["paths"]
        ds = CheckedOriginalDataset(p[f"{split}_images"], p[f"{split}_annotations"],
                                    transforms=core.get_transform(split == "train"), train=split == "train")
        if hasattr(ds.transforms, "set_random_seed"):
            ds.transforms.set_random_seed(seed)
        return ds
    generator = torch.Generator().manual_seed(seed)
    common = dict(batch_size=batch_size, collate_fn=core.collate_fn, generator=generator,
                  pin_memory=torch.cuda.is_available(), num_workers=workers, worker_init_fn=_worker_seed)
    if workers:
        common.update(persistent_workers=True, prefetch_factor=prefetch)
    val = dataset("val")
    val_loader = DataLoader(val, shuffle=False, drop_last=False, **common)
    train_loader = None
    if not val_only:
        train = dataset("train")
        train_loader = DataLoader(train, shuffle=True, drop_last=True, **common)
    return train_loader, val_loader, generator
