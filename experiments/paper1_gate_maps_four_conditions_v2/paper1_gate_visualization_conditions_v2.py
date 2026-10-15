#!/usr/bin/env python3
"""Paired clear/fog/rain/snow LGF gate maps from one saved Paper 1 checkpoint.

Uses COCO-NW for clear images and COCO-weather for the three synthetic weather
versions of the SAME source scenes. Runs the same LGF checkpoint for every row.
The audited condition index supplies sibling images when the validation JSON
contains annotations for only one weather condition.
No model weights are changed and no training is performed.
"""

import argparse
import csv
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
from matplotlib.patches import Rectangle
from PIL import Image


CONDITIONS = ("clear", "fog", "rain", "snow")
ALIASES = {"clear": "clear", "fog": "fog", "foggy": "fog",
           "rain": "rain", "rainy": "rain", "snow": "snow", "snowy": "snow"}


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True,
                   help="COCO-weather lgf_original best_ap_small.pth; used for all four conditions")
    p.add_argument("--package-root", type=Path, required=True,
                   help="Directory containing core_bridge.py and vendor/original_core.py")
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--clear-ann", type=Path, default=None)
    p.add_argument("--weather-ann", type=Path, default=None,
                   help="Weather validation JSON; a single-condition JSON works with --weather-index")
    p.add_argument("--weather-index", type=Path, default=None,
                   help="Audited image_conditions.csv mapping scene stems to fog/rain/snow filenames and IDs")
    p.add_argument("--clear-images", type=Path, default=None)
    p.add_argument("--weather-images", type=Path, default=None)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--level", choices=("C3", "C4", "C5"), default="C3")
    p.add_argument("--num-scenes", type=int, default=4)
    p.add_argument("--seed", type=int, default=42,
                   help="Deterministic scene selection, not the model training seed")
    p.add_argument("--scene-key", action="append", default=[],
                   help="Optional source filename stem; repeat to select exact scenes")
    p.add_argument("--plan-only", action="store_true",
                   help="Check pairing, labels, boxes and files without loading the model")
    return p.parse_args()


def paths(args):
    nw = args.data_root / "COCO/non_weather-mini"
    weather = args.data_root / "COCO/weather-mini"
    return {
        "clear_ann": args.clear_ann or nw / "annotations/mini_val2017_non_weather-500k6c.json",
        "weather_ann": args.weather_ann or weather / "annotations/mini_val2017_weather-500k6c.json",
        "clear_images": args.clear_images or nw / "images/val2017_non_weather-500k6c",
        "weather_images": args.weather_images or weather / "images/val2017_weather-500k6c",
    }


def read_annotations(path):
    if not path.is_file():
        raise FileNotFoundError(path)
    doc = json.loads(path.read_text(encoding="utf-8"))
    images = {}
    anns = defaultdict(list)
    for info in doc["images"]:
        image_id = int(info["id"])
        if image_id in images:
            raise ValueError(f"Duplicate image ID {image_id} in {path}")
        images[image_id] = info
    for ann in doc["annotations"]:
        image_id = int(ann["image_id"])
        if image_id not in images:
            raise ValueError(f"Unknown annotation image ID {image_id} in {path}")
        if not int(ann.get("iscrowd", 0)):
            anns[image_id].append(ann)
    return images, anns


def scene_key(info):
    stem = Path(info["file_name"]).stem.lower()
    # The audited weather set uses 000000001584_{fog,rain,snow}.jpg.
    numbered = re.match(r"^(\d{12})(?:_|$)", stem)
    if numbered:
        return numbered.group(1)
    return re.sub(r"_(clear|fog|rain|snow)$", "", stem)


def weather_condition(info):
    name = Path(info["file_name"]).stem.lower()
    suffix = re.search(r"_(fog|rain|snow)$", name)
    from_name = suffix.group(1) if suffix else None
    field = info.get("weather", info.get("condition"))
    from_field = ALIASES.get(str(field).strip().lower()) if field is not None else None
    if field is not None and from_field is None:
        raise ValueError(f"Unknown weather metadata {field!r} for {info['file_name']}")
    if from_name and from_field and from_name != from_field:
        raise ValueError(f"Conflicting weather label for {info['file_name']}")
    condition = from_field or from_name
    if condition not in CONDITIONS[1:]:
        raise ValueError(f"Cannot establish fog/rain/snow for {info['file_name']}")
    return condition


def index_scenes(clear_images, weather_images):
    records = {c: {} for c in CONDITIONS}
    for info in clear_images.values():
        key = scene_key(info)
        if key in records["clear"]:
            raise ValueError(f"Duplicate clear scene {key}")
        records["clear"][key] = info
    for info in weather_images.values():
        condition = weather_condition(info)
        key = scene_key(info)
        if key in records[condition]:
            raise ValueError(f"Duplicate {condition} scene {key}")
        records[condition][key] = info
    return records


def complete_weather_from_index(records, weather_anns, index_path, weather_root):
    """Reuse annotated GT for paired appearance-only weather versions.

    The image IDs and filenames come from the uploaded weather-condition audit.
    Boxes are inherited from an annotated sibling of the same source scene.
    This is for visualization only; no synthetic GT is used for evaluation.
    """
    added = {condition: 0 for condition in CONDITIONS[1:]}
    if index_path is None:
        return added, []
    if not index_path.is_file():
        raise FileNotFoundError(index_path)
    audited = {}
    with index_path.open(newline="", encoding="utf-8-sig") as file:
        for row in csv.DictReader(file):
            condition = row["condition"].strip().lower()
            if condition not in CONDITIONS[1:]:
                continue
            key = scene_key(row)
            pair = (key, condition)
            if pair in audited:
                raise ValueError(f"Duplicate audited weather scene {pair}")
            audited[pair] = row
    if not audited:
        raise ValueError(f"No fog/rain/snow records in {index_path}")
    missing_files = []
    keys = sorted(set().union(*(set(records[c]) for c in CONDITIONS[1:])))
    for key in keys:
        annotated_source = next((records[c][key] for c in CONDITIONS[1:]
                                 if key in records[c] and
                                 "_gt_source_image_id" not in records[c][key]), None)
        if annotated_source is None:
            continue
        source_id = int(annotated_source["id"])
        for condition in CONDITIONS[1:]:
            audited_row = audited.get((key, condition))
            if key in records[condition]:
                if audited_row and Path(audited_row["file_name"]).name != Path(records[condition][key]["file_name"]).name:
                    raise ValueError(f"Weather audit and annotations disagree for {key} {condition}")
                continue
            if audited_row is None:
                continue
            file_name = audited_row["file_name"]
            image_path = weather_root / file_name
            if not image_path.is_file():
                missing_files.append(str(image_path))
                continue
            image_id = int(audited_row["image_id"])
            if image_id in weather_anns:
                raise ValueError(f"Audited image ID collision: {image_id}")
            records[condition][key] = {
                "id": image_id, "file_name": file_name, "weather": condition,
                "_gt_source_image_id": source_id,
            }
            weather_anns[image_id] = [
                {**ann, "image_id": image_id,
                 "id": f"inherited_from_{source_id}_ann_{ann.get('id', 'unknown')}"}
                for ann in weather_anns[source_id]
            ]
            added[condition] += 1
    print("Additional weather images with inherited sibling GT:", added, flush=True)
    if missing_files:
        print(f"Weather images missing from disk: {len(missing_files)}; example: {missing_files[0]}", flush=True)
    return added, missing_files


def area(ann):
    return float(ann.get("area", ann["bbox"][2] * ann["bbox"][3]))


def small_anns(anns):
    return [a for a in anns if 0 < area(a) < 32 ** 2 and
            float(a["bbox"][2]) > 0 and float(a["bbox"][3]) > 0]


def bbox_iou(box_a, box_b):
    ax, ay, aw, ah = map(float, box_a)
    bx, by, bw, bh = map(float, box_b)
    ix = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0.0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def matched_target(key, records, clear_anns, weather_anns):
    """Choose a small GT box shared across all four versions of a scene."""
    clear = records["clear"][key]
    candidates = sorted(small_anns(clear_anns[int(clear["id"])]),
                        key=lambda a: (-area(a), str(a.get("id", ""))))
    for target in candidates:
        matches = {"clear": target}
        for condition in CONDITIONS[1:]:
            info = records[condition][key]
            options = [a for a in small_anns(weather_anns[int(info["id"])])
                       if a["category_id"] == target["category_id"]]
            if not options:
                break
            best = max(options, key=lambda a: bbox_iou(target["bbox"], a["bbox"]))
            if bbox_iou(target["bbox"], best["bbox"]) < 0.90:
                break
            matches[condition] = best
        if len(matches) == 4:
            return matches
    return None


def select_scenes(args, path_map):
    clear_images, clear_anns = read_annotations(path_map["clear_ann"])
    weather_images, weather_anns = read_annotations(path_map["weather_ann"])
    records = index_scenes(clear_images, weather_images)
    derived_counts, missing_weather_files = complete_weather_from_index(
        records, weather_anns, args.weather_index, path_map["weather_images"])
    counts = {condition: len(records[condition]) for condition in CONDITIONS}
    print("Available condition counts:", counts, flush=True)
    shared = sorted(set.intersection(*(set(records[c]) for c in CONDITIONS)))
    if not shared:
        raise ValueError("No paired clear/fog/rain/snow scenes. Check the weather-index mapping and weather image files. "
                         f"Counts: {counts}; missing weather files: {len(missing_weather_files)}")
    eligible = {}
    for key in shared:
        matches = matched_target(key, records, clear_anns, weather_anns)
        if matches is not None:
            eligible[key] = matches
    if not eligible:
        raise ValueError("No paired scenes have a matching small-object GT box across all four conditions.")
    if args.scene_key:
        chosen = list(dict.fromkeys(args.scene_key))
        missing = [key for key in chosen if key not in eligible]
        if missing:
            raise ValueError(f"Requested scene(s) lack a four-condition small GT match: {missing}")
    else:
        if args.num_scenes <= 0:
            raise ValueError("--num-scenes must be positive")
        if args.num_scenes > len(eligible):
            raise ValueError(f"Requested {args.num_scenes} scenes; only {len(eligible)} are eligible")
        chosen = sorted(eligible)
        random.Random(args.seed).shuffle(chosen)
        chosen = chosen[:args.num_scenes]
    for key in chosen:
        sizes = set()
        for condition in CONDITIONS:
            info = records[condition][key]
            root = path_map["clear_images" if condition == "clear" else "weather_images"]
            image_file = root / info["file_name"]
            if not image_file.is_file():
                raise FileNotFoundError(image_file)
            with Image.open(image_file) as image:
                sizes.add(image.size)
            print(f"{key} {condition:5s} image_id={info['id']} {image_file}", flush=True)
        if len(sizes) != 1:
            raise ValueError(f"Paired scene {key} has differing image dimensions: {sizes}")
    print(f"Paired scenes: {len(shared)}; with matching small GT: {len(eligible)}; selected: {len(chosen)}")
    return chosen, records, eligible, clear_anns, weather_anns, counts, len(shared), derived_counts


def load_model(args):
    if not (args.package_root / "core_bridge.py").is_file():
        raise FileNotFoundError(args.package_root / "core_bridge.py")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use --device cpu or run on the GPU server")
    sys.path.insert(0, str(args.package_root.resolve()))
    from core_bridge import build_model

    try:
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    except TypeError:  # older torch versions lack weights_only
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
    if checkpoint.get("method") != "lgf_original":
        raise ValueError(f"Expected lgf_original checkpoint; got {checkpoint.get('method')!r}")
    if checkpoint.get("dataset") != "coco_weather":
        raise ValueError(f"Use one COCO-weather LGF checkpoint across all four scenes, got {checkpoint.get('dataset')!r}")
    if checkpoint.get("level") != args.level:
        raise ValueError(f"Checkpoint level={checkpoint.get('level')}, requested {args.level}")
    model = build_model("lgf_original", 6, args.level, pretrained=False,
                        original_config=checkpoint.get("config") or None)
    state = checkpoint.get("ema_state_dict")
    if state is None:
        raise ValueError("Checkpoint has no ema_state_dict")
    if state and all(key.startswith("module.") for key in state):
        state = {key[7:]: value for key, value in state.items()}
    model.load_state_dict(state, strict=True)
    model.to(torch.device(args.device)).float().eval()
    blocks = [(name, block) for name, block in model.named_modules()
              if getattr(block, "gating_type", None) == "gated_spatial"
              and hasattr(block, "gate_norm")]
    if len(blocks) != 1:
        raise RuntimeError(f"Expected exactly one gated_spatial LGF block; found {len(blocks)}")
    name, block = blocks[0]
    if tuple(block.branches) != ("local", "global"):
        raise ValueError(f"Unexpected LGF branch order {block.branches}")
    print(f"Loaded EMA model: {args.checkpoint}; gate block: {name}", flush=True)
    return model, block, name, checkpoint


def image_tensor(image, device):
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous().to(device)


def resize_gate(gate, padded_hw, resized_hw, original_hw):
    x = F.interpolate(gate[None, None].float(), size=padded_hw,
                      mode="bilinear", align_corners=False)
    x = x[..., :resized_hw[0], :resized_hw[1]]
    x = F.interpolate(x, size=original_hw, mode="bilinear", align_corners=False)
    return x[0, 0].numpy()


def inference_gates(model, block, image, device):
    captured = []

    def on_gate_norm(_module, _inputs, logits):
        # The LGF forward method uses sigmoid(clamp(gate_norm(logits))/tau).
        tau = F.softplus(block.temperature.float()) + 1e-3
        weights = torch.sigmoid(logits.float().clamp(-15, 15) / tau)
        n, two_c, h, w = weights.shape
        if n != 1 or two_c % 2:
            raise RuntimeError(f"Unexpected LGF gate shape {tuple(weights.shape)}")
        captured.append(weights.reshape(n, 2, two_c // 2, h, w)
                        .mean(dim=2).detach().cpu())

    handle = block.gate_norm.register_forward_hook(on_gate_norm)
    try:
        with torch.inference_mode():
            transformed, _ = model.transform([image_tensor(image, device)], None)
            model.backbone(transformed.tensors)
    finally:
        handle.remove()
    if len(captured) != 1:
        raise RuntimeError(f"Gate ran {len(captured)} times; expected once")
    maps = captured[0][0]
    padded_hw = transformed.tensors.shape[-2:]
    resized_hw = transformed.image_sizes[0]
    original_hw = (image.height, image.width)
    return (resize_gate(maps[0], padded_hw, resized_hw, original_hw),
            resize_gate(maps[1], padded_hw, resized_hw, original_hw))


def box_mean(gate, bbox):
    x, y, w, h = map(float, bbox)
    ih, iw = gate.shape
    x1, y1 = max(0, int(np.floor(x))), max(0, int(np.floor(y)))
    x2, y2 = min(iw, int(np.ceil(x + w))), min(ih, int(np.ceil(y + h)))
    return float(gate[y1:y2, x1:x2].mean()) if x2 > x1 and y2 > y1 else None


def crop_for_target(bbox, width, height):
    x, y, w, h = map(float, bbox)
    span = max(128, int(round(6 * max(w, h))))
    span_x, span_y = min(span, width), min(span, height)
    x1 = min(max(0, int(x + w / 2 - span_x / 2)), width - span_x)
    y1 = min(max(0, int(y + h / 2 - span_y / 2)), height - span_y)
    return x1, y1, x1 + span_x, y1 + span_y


def draw_boxes(ax, anns, selected, offset=(0, 0)):
    for ann in small_anns(anns):
        x, y, w, h = map(float, ann["bbox"])
        rect = Rectangle((x - offset[0], y - offset[1]), w, h,
                         fill=False, edgecolor="white",
                         linewidth=2.2 if ann is selected else 1.0)
        rect.set_path_effects([pe.Stroke(linewidth=4.0 if ann is selected else 2.4,
                                         foreground="black"), pe.Normal()])
        ax.add_patch(rect)


def save_panel(scene, rows, figure_path, *, zoom=False):
    fig, axes = plt.subplots(4, 3, figsize=(13, 14), constrained_layout=True)
    names = ("Input + small GT", "Local gate", "Global gate")
    for col, title in enumerate(names):
        axes[0, col].set_title(title, fontsize=13)
    target = rows["clear"]["selected"]["bbox"]
    width, height = rows["clear"]["image"].size
    crop = crop_for_target(target, width, height) if zoom else (0, 0, width, height)
    x1, y1, x2, y2 = crop
    color_mappable = None
    for row_i, condition in enumerate(CONDITIONS):
        item = rows[condition]
        arrays = [item["image"].crop(crop),
                  item["local"][y1:y2, x1:x2],
                  item["global"][y1:y2, x1:x2]]
        for col, ax in enumerate(axes[row_i]):
            if col == 0:
                ax.imshow(arrays[col])
            else:
                color_mappable = ax.imshow(arrays[col], cmap="viridis", vmin=0, vmax=1)
            draw_boxes(ax, item["anns"], item["selected"], (x1, y1))
            ax.set_xlim(0, x2 - x1)
            ax.set_ylim(y2 - y1, 0)
            ax.axis("off")
        axes[row_i, 0].text(-0.04, 0.5, condition.title(), transform=axes[row_i, 0].transAxes,
                            rotation=90, va="center", ha="right", fontsize=12)
    fig.colorbar(color_mappable, ax=axes[:, 1:].ravel().tolist(), shrink=0.67,
                 label="Mean sigmoid gate across channels (0 to 1)")
    fig.suptitle(f"Source scene {scene} | one COCO-weather LGF checkpoint" +
                 (" | matched small-object crop" if zoom else ""), fontsize=13)
    fig.savefig(figure_path, dpi=175, bbox_inches="tight")
    plt.close(fig)


def save_csv(path, fieldnames, rows):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = arguments()
    path_map = paths(args)
    for name in ("clear_images", "weather_images"):
        if not path_map[name].is_dir():
            raise FileNotFoundError(path_map[name])
    selected, records, targets, clear_anns, weather_anns, counts, shared_count, derived_counts = select_scenes(args, path_map)
    if args.plan_only:
        print("PLAN ONLY: paired scenes and image files checked; no model inference.")
        return
    global torch, F
    import torch
    import torch.nn.functional as F
    args.output.mkdir(parents=True, exist_ok=True)
    figure_dir = args.output / "figures"
    figure_dir.mkdir(exist_ok=True)
    model, block, block_name, ck = load_model(args)
    gate_rows = []
    selection_rows = []
    for idx, scene in enumerate(selected, start=1):
        rows = {}
        sizes = set()
        for condition in CONDITIONS:
            info = records[condition][scene]
            ann_list = (clear_anns if condition == "clear" else weather_anns)[int(info["id"])]
            root = path_map["clear_images" if condition == "clear" else "weather_images"]
            with Image.open(root / info["file_name"]) as image_file:
                image = image_file.convert("RGB")
            sizes.add(image.size)
            local, global_gate = inference_gates(model, block, image, args.device)
            rows[condition] = {"image": image, "local": local, "global": global_gate,
                               "anns": ann_list, "selected": targets[scene][condition]}
            target_box = targets[scene][condition]["bbox"]
            gate_rows.append({
                "scene": scene, "condition": condition, "dataset": "coco_nw" if condition == "clear" else "coco_weather",
                "image_id": info["id"], "file_name": info["file_name"],
                "gt_source_image_id": info.get("_gt_source_image_id", info["id"]),
                "target_annotation_id": targets[scene][condition].get("id", ""),
                "small_gt_count": len(small_anns(ann_list)),
                "local_target_mean": box_mean(local, target_box),
                "global_target_mean": box_mean(global_gate, target_box),
                "local_image_mean": float(local.mean()),
                "global_image_mean": float(global_gate.mean()),
            })
            selection_rows.append({"scene": scene, "condition": condition, "image_id": info["id"],
                                   "file_name": info["file_name"],
                                   "gt_source_image_id": info.get("_gt_source_image_id", info["id"]),
                                   "selected_small_bbox": list(target_box)})
        if len(sizes) != 1:
            raise ValueError(f"Image sizes differ across paired conditions for scene {scene}: {sizes}")
        save_panel(scene, rows, figure_dir / f"{idx:02d}_{scene}_full.png")
        save_panel(scene, rows, figure_dir / f"{idx:02d}_{scene}_small_object_crop.png", zoom=True)
        print(f"[{idx}/{len(selected)}] saved paired full and crop figures for {scene}", flush=True)
    save_csv(args.output / "target_gate_stats.csv",
             ["scene", "condition", "dataset", "image_id", "file_name", "gt_source_image_id", "target_annotation_id",
              "small_gt_count", "local_target_mean", "global_target_mean", "local_image_mean", "global_image_mean"],
             gate_rows)
    manifest = {
        "checkpoint": str(args.checkpoint.resolve()), "checkpoint_method": ck.get("method"),
        "checkpoint_dataset": ck.get("dataset"), "checkpoint_seed": ck.get("seed"),
        "checkpoint_level": ck.get("level"), "checkpoint_selection": args.checkpoint.name,
        "model_weights_modified": False, "model_training_performed": False,
        "same_checkpoint_used_for_every_condition": True, "gate_block": block_name,
        "gate_value": "Channel mean of the sigmoid weights applied to local and global branches, before gamma residual scaling",
        "spatial_alignment": "Bilinear resize from C3 to padded input, crop padding, resize to original image for display",
        "source_scenes_shared_across_conditions": shared_count,
        "condition_image_counts": counts,
        "inherited_gt_counts": derived_counts,
        "weather_condition_index": str(args.weather_index.resolve()) if args.weather_index else None,
        "inherited_gt_note": "Missing weather annotations use boxes from an annotated sibling scene; synthetic weather transforms change appearance but not geometry; this script performs visualization only",
        "selection_seed": args.seed, "eligible_scenes_with_matching_small_gt": len(targets),
        "selection": selection_rows,
        "annotation_paths": {k: str(path_map[k]) for k in ("clear_ann", "weather_ann")},
        "image_roots": {k: str(path_map[k]) for k in ("clear_images", "weather_images")},
        "small_definition": "Non-crowd COCO annotation area < 1024 px^2",
        "figure_color_scale": [0, 1],
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"DONE: {figure_dir}")
    print(f"Stats: {args.output / 'target_gate_stats.csv'}")


if __name__ == "__main__":
    main()
