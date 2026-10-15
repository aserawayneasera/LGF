#!/usr/bin/env python3
"""Real ACDC fog/rain/snow/night LGF gate maps, using the saved ACDC checkpoint."""

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import paper1_gate_visualization_conditions_v2 as common


CONDITIONS = ("fog", "rain", "snow", "night")


def get_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--package-root", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--condition-index", type=Path, required=True,
                   help="ACDC image_conditions.csv from the weather-condition audit")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--level", choices=("C3", "C4", "C5"), default="C3")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--selection-seed", type=int, default=42)
    p.add_argument("--per-condition", type=int, default=2)
    p.add_argument("--plan-only", action="store_true")
    return p.parse_args()


def select_images(args, ann_file, image_root):
    images, annotations = common.read_annotations(ann_file)
    if not args.condition_index.is_file():
        raise FileNotFoundError(args.condition_index)
    rows = {}
    with args.condition_index.open(newline="", encoding="utf-8-sig") as file:
        for row in csv.DictReader(file):
            image_id = int(row["image_id"])
            if image_id in rows:
                raise ValueError(f"Duplicate ACDC audit image ID {image_id}")
            rows[image_id] = row
    if args.per_condition <= 0:
        raise ValueError("--per-condition must be positive")
    eligible = defaultdict(list)
    coverage = defaultdict(int)
    for image_id, image in images.items():
        audit = rows.get(image_id)
        if audit is None:
            raise ValueError(f"ACDC annotation image ID {image_id} is missing from the audit")
        if image["file_name"] != audit["file_name"]:
            raise ValueError(f"ACDC audit filename differs for image ID {image_id}")
        condition = audit["condition"].lower()
        if condition not in CONDITIONS:
            raise ValueError(f"Unexpected ACDC condition {condition!r} for image ID {image_id}")
        # Restrict supplementary figures to the source dataset's validation frames.
        if audit["official_split"].lower() != "val":
            continue
        coverage[condition] += 1
        small = common.small_anns(annotations[image_id])
        if small:
            eligible[condition].append((image_id, image, audit, small))
    print("Original ACDC validation frames:",
          {condition: coverage[condition] for condition in CONDITIONS}, flush=True)
    print("With at least one small GT object:",
          {condition: len(eligible[condition]) for condition in CONDITIONS}, flush=True)
    selected = []
    for index, condition in enumerate(CONDITIONS):
        group = sorted(eligible[condition], key=lambda item: item[0])
        random.Random(args.selection_seed + index).shuffle(group)
        if len(group) < args.per_condition:
            raise ValueError(f"Only {len(group)} eligible {condition} images; requested {args.per_condition}")
        for image_id, info, audit, small in group[:args.per_condition]:
            file = image_root / info["file_name"]
            if not file.is_file():
                raise FileNotFoundError(file)
            with Image.open(file) as image:
                image.verify()
            target = sorted(small, key=lambda ann: (-common.area(ann), str(ann.get("id", ""))))[0]
            selected.append({"id": image_id, "condition": condition, "image": info,
                             "audit": audit, "target": target, "annotations": annotations[image_id],
                             "path": file})
            print(f"{condition:5s} image_id={image_id} {file}", flush=True)
    return selected, coverage, {condition: len(eligible[condition]) for condition in CONDITIONS}


def load_acdc_model(args):
    import torch
    if not (args.package_root / "core_bridge.py").is_file():
        raise FileNotFoundError(args.package_root / "core_bridge.py")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; choose --device cpu or run on the GPU server")
    sys.path.insert(0, str(args.package_root.resolve()))
    from core_bridge import build_model

    try:
        ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    except TypeError:
        ck = torch.load(args.checkpoint, map_location="cpu")
    if ck.get("method") != "lgf_original" or ck.get("dataset") != "acdc":
        raise ValueError(f"Expected ACDC lgf_original checkpoint; got {ck.get('dataset')}/{ck.get('method')}")
    if ck.get("level") != args.level:
        raise ValueError(f"Checkpoint level {ck.get('level')} differs from {args.level}")
    model = build_model("lgf_original", 6, args.level, pretrained=False,
                        original_config=ck.get("config") or None)
    state = ck.get("ema_state_dict")
    if state is None:
        raise ValueError("Checkpoint has no EMA weights")
    if state and all(key.startswith("module.") for key in state):
        state = {key[7:]: value for key, value in state.items()}
    model.load_state_dict(state, strict=True)
    model.to(torch.device(args.device)).float().eval()
    blocks = [(name, module) for name, module in model.named_modules()
              if getattr(module, "gating_type", None) == "gated_spatial"
              and hasattr(module, "gate_norm")]
    if len(blocks) != 1:
        raise RuntimeError(f"Expected one LGF gate block, found {len(blocks)}")
    name, block = blocks[0]
    if tuple(block.branches) != ("local", "global"):
        raise ValueError(f"Unexpected branch order: {block.branches}")
    # The shared inference function imports torch lazily in its own module.
    common.torch = torch
    common.F = torch.nn.functional
    print(f"Loaded ACDC EMA checkpoint, gate block {name}", flush=True)
    return model, block, name, ck


def render_one(entry, image, local, global_gate, target, output, *, zoom=False):
    width, height = image.size
    crop = (common.crop_for_target(target["bbox"], width, height)
            if zoom else (0, 0, width, height))
    x1, y1, x2, y2 = crop
    arrays = [image.crop(crop), local[y1:y2, x1:x2], global_gate[y1:y2, x1:x2]]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), constrained_layout=True)
    color_mappable = None
    for col, (ax, title) in enumerate(zip(axes, ("ACDC image + small GT", "Local gate", "Global gate"))):
        if col == 0:
            ax.imshow(arrays[col])
        else:
            color_mappable = ax.imshow(arrays[col], cmap="viridis", vmin=0, vmax=1)
        common.draw_boxes(ax, entry["annotations"], target, (x1, y1))
        ax.set_xlim(0, x2 - x1)
        ax.set_ylim(y2 - y1, 0)
        ax.set_title(title)
        ax.axis("off")
    fig.colorbar(color_mappable, ax=axes[1:].tolist(), shrink=0.67,
                 label="Mean sigmoid gate across channels (0 to 1)")
    fig.suptitle(f"ACDC {entry['condition']} | image {entry['id']} | original split: val" +
                 (" | small-object crop" if zoom else ""))
    fig.savefig(output, dpi=175, bbox_inches="tight")
    plt.close(fig)


def main():
    args = get_args()
    acdc = args.data_root / "ACDC-1/ACDC-1-NEW"
    ann_file = acdc / "annotations/mini_val.json"
    image_root = acdc / "images/val"
    if not image_root.is_dir():
        raise FileNotFoundError(image_root)
    selected, coverage, eligible_counts = select_images(args, ann_file, image_root)
    if args.plan_only:
        print("PLAN ONLY: no model inference", flush=True)
        return
    args.output.mkdir(parents=True, exist_ok=True)
    figures = args.output / "figures"
    figures.mkdir(exist_ok=True)
    model, block, block_name, ck = load_acdc_model(args)
    stats = []
    manifest_rows = []
    for index, entry in enumerate(selected, 1):
        with Image.open(entry["path"]) as source:
            image = source.convert("RGB")
        local, global_gate = common.inference_gates(model, block, image, args.device)
        stem = f"{entry['condition']}_{entry['id']:04d}"
        render_one(entry, image, local, global_gate, entry["target"], figures / f"{stem}_full.png")
        render_one(entry, image, local, global_gate, entry["target"], figures / f"{stem}_small_crop.png", zoom=True)
        stats.append({"condition": entry["condition"], "image_id": entry["id"],
                      "file_name": entry["image"]["file_name"],
                      "local_target_mean": common.box_mean(local, entry["target"]["bbox"]),
                      "global_target_mean": common.box_mean(global_gate, entry["target"]["bbox"]),
                      "local_image_mean": float(np.mean(local)),
                      "global_image_mean": float(np.mean(global_gate))})
        manifest_rows.append({"condition": entry["condition"], "image_id": entry["id"],
                              "file_name": entry["image"]["file_name"],
                              "official_split": entry["audit"]["official_split"],
                              "condition_evidence": entry["audit"]["evidence"],
                              "target_annotation_id": entry["target"].get("id"),
                              "target_bbox": entry["target"]["bbox"]})
        print(f"[{index}/{len(selected)}] {entry['condition']} image {entry['id']}", flush=True)
    common.save_csv(args.output / "target_gate_stats.csv",
                    ["condition", "image_id", "file_name", "local_target_mean", "global_target_mean",
                     "local_image_mean", "global_image_mean"], stats)
    metadata = {"checkpoint": str(args.checkpoint.resolve()), "method": ck.get("method"),
                "dataset": ck.get("dataset"), "seed": ck.get("seed"), "level": ck.get("level"),
                "checkpoint_selection": args.checkpoint.name, "gate_block": block_name,
                "one_checkpoint_used_for_all_images": True, "training_performed": False,
                "weights_modified": False, "official_validation_only": True,
                "conditions": CONDITIONS, "official_val_image_counts": dict(coverage),
                "eligible_small_object_image_counts": eligible_counts,
                "selection_seed": args.selection_seed, "per_condition": args.per_condition,
                "image_rows": manifest_rows,
                "gate_value": "Channel mean of local/global sigmoid weights, before gamma scaling",
                "small_definition": "Non-crowd COCO annotation area < 1024 px^2",
                "figure_color_scale": [0, 1],
                "caveat": "Conditions comprise different real ACDC scenes; these images are not paired weather transformations."}
    (args.output / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"DONE: {figures}")


if __name__ == "__main__":
    main()
