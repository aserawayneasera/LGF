#!/usr/bin/env python3
"""CPU analysis of saved Paper 1 detections. No training or checkpoint loading."""
from __future__ import annotations

import argparse
from collections import defaultdict
import contextlib
import copy
import csv
import hashlib
import importlib.metadata
import io
import json
import math
from pathlib import Path
import random
import statistics
import sys

VERSION = "1.0.1"
SIZES = ("all", "small", "medium", "large")
CLASSES = {"person", "bicycle", "car", "motorcycle", "bus", "truck"}
LABELS = {"baseline": "RetinaNet", "lgf_original": "RetinaNet + LGF",
          "ugs_reimpl": "UGS reimplementation", "yolo11m": "YOLO11m",
          "rtdetr-l": "RT-DETR-L", "se": "RetinaNet + SE", "cbam": "RetinaNet + CBAM"}
EVENTS = ("lgf_recovery", "retinanet_recovery", "shared_miss", "lgf_only_fp")


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def size_bucket(area):
    # Exclusive diagnostic partition. Official COCO ranges also include their
    # shared endpoints; those results are saved separately.
    return "small" if area < 32 ** 2 else "medium" if area < 96 ** 2 else "large"


def box_iou(a, b):
    x = max(0., min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    y = max(0., min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    union = a[2] * a[3] + b[2] * b[3] - x * y
    return x * y / union if union > 0 else 0.


def remapped(path, mappings):
    value = str(path)
    for old, new in mappings:
        if value == old or value.startswith(old.rstrip("/") + "/"):
            return new.rstrip("/") + value[len(old.rstrip("/")):]
    return value


def resolve_file(value, base, mappings, expected_hash=None, alternatives=()):
    candidates = []
    if value:
        p = Path(remapped(value, mappings)).expanduser()
        candidates.append(p if p.is_absolute() else base / p)
    candidates.extend(Path(p) for p in alternatives)
    found = []
    for p in dict.fromkeys(candidates):
        if p.is_file() and (not expected_hash or digest(p) == expected_hash):
            found.append(p.resolve())
    if not found:
        raise ValueError(f"Missing file or SHA-256 mismatch: {value or list(alternatives)}")
    return found[0]


def document(path):
    doc = read_json(path)
    if "annotations" not in doc:
        raise ValueError(f"Ground-truth boxes required. Image-info or hidden-label test JSON is insufficient: {path}")
    if not doc.get("images") or not doc.get("categories"):
        raise ValueError(f"Empty ground-truth image/category list: {path}")
    if {c["name"] for c in doc["categories"]} != CLASSES or len(doc["categories"]) != 6:
        raise ValueError("Expected the Paper 1 six-class annotations")
    image_ids = [im["id"] for im in doc["images"]]
    category_ids = [c["id"] for c in doc["categories"]]
    ann_ids = [a["id"] for a in doc["annotations"]]
    if any(len(x) != len(set(x)) for x in [image_ids, category_ids, ann_ids]):
        raise ValueError("Duplicate ground-truth identifiers")
    if any(not isinstance(x, int) or x < 0 for x in image_ids + category_ids + ann_ids):
        raise ValueError("Ground-truth identifiers must be nonnegative integers")
    image_set, category_set = set(image_ids), set(category_ids)
    for im in doc["images"]:
        if im.get("width", 0) <= 0 or im.get("height", 0) <= 0:
            raise ValueError("Ground-truth image dimensions required")
    for a in doc["annotations"]:
        if a["image_id"] not in image_set or a["category_id"] not in category_set:
            raise ValueError("Unknown image/category in ground truth")
        if len(a["bbox"]) != 4 or not all(math.isfinite(float(v)) for v in a["bbox"]) or min(a["bbox"][2:]) < 0:
            raise ValueError("Invalid ground-truth box")
        if "area" not in a or not math.isfinite(float(a["area"])) or a["area"] < 0:
            raise ValueError("Finite original annotation area required for size analysis")
    return doc


def normalize_predictions(payload, doc):
    if not isinstance(payload, list):
        raise ValueError("Prediction file must contain an array")
    images = {im["id"]: im for im in doc["images"]}
    by_name = {im["file_name"]: im for im in doc["images"]}
    cats = {c["name"]: c["id"] for c in doc["categories"]}
    category_ids = set(cats.values())
    out = []
    if payload and "detections" in payload[0]:
        seen = set()
        for row in payload:
            if row.get("file_name") not in by_name or row["file_name"] in seen:
                raise ValueError("Filename prediction export does not match ground-truth images")
            seen.add(row["file_name"])
            image = by_name[row["file_name"]]
            # Filename exports sometimes use internal image IDs. Map by filename,
            # never by the internal IDs, to the supplied labelled annotations.
            for d in row["detections"]:
                if d.get("category_name") not in cats:
                    raise ValueError("Unknown class name in filename predictions")
                out.append({"image_id": image["id"], "category_id": cats[d["category_name"]],
                            "bbox": d["bbox"], "score": d["score"]})
        if seen != set(by_name):
            raise ValueError("Incomplete filename prediction export")
    else:
        out = [{k: d[k] for k in ["image_id", "category_id", "bbox", "score"]} for d in payload]
    for d in out:
        if d["image_id"] not in images or d["category_id"] not in category_ids:
            raise ValueError("Prediction image/category IDs do not match ground truth")
        if len(d["bbox"]) != 4 or not all(math.isfinite(float(v)) for v in [*d["bbox"], d["score"]]):
            raise ValueError("Nonfinite/invalid detection")
        if min(d["bbox"][2:]) < 0 or not 0 <= d["score"] <= 1:
            raise ValueError("Invalid box width/height or score")
    return out


def auto_entries(args):
    entries = []
    for dataset in args.datasets:
        for method in args.methods:
            root = args.recent_root if method in {"yolo11m", "rtdetr-l"} else (
                args.se_cbam_root or args.ugs_root) if method in {"se", "cbam"} else args.ugs_root
            for seed in args.seeds:
                run = root / dataset / method / f"seed_{seed}"
                entries.append({"dataset": dataset, "method": method, "seed": seed,
                                "report": str(run / f"{args.selection}.metrics.json"),
                                "run_directory": str(run)})
    return entries, {}


def load_inputs(args):
    mappings = [tuple(s.split("=", 1)) for s in args.remap]
    if any(not a or not b for a, b in mappings):
        raise ValueError("--remap requires OLD=NEW")
    if args.manifest:
        config = read_json(args.manifest)
        entries, data_config = config["runs"], config.get("datasets", {})
        base = args.manifest.resolve().parent
    else:
        entries, data_config = auto_entries(args)
        base = Path.cwd()
    runs, errors, identities = [], [], set()
    for entry in entries:
        dataset, method, seed = entry["dataset"], entry["method"], int(entry["seed"])
        identity = (dataset, method, seed)
        if identity in identities:
            raise ValueError(f"Duplicate dataset/method/seed: {identity}")
        identities.add(identity)
        try:
            report_path = resolve_file(entry["report"], base, mappings)
            report = read_json(report_path)
            if report.get("scientific_result") is False or report.get("synthetic"):
                raise ValueError("Software smoke/partial results are not accepted")
            if report.get("selection") != args.selection:
                raise ValueError("Checkpoint selection differs from --selection")
            aliases = {"original_none": "baseline", "original_lgf": "lgf_original",
                       "original_se": "se", "original_cbam": "cbam"}
            actual_method = report.get("method", report.get("model"))
            if aliases.get(actual_method, actual_method) != method:
                raise ValueError("Method identity differs from report")
            if report.get("dataset") != dataset or report.get("seed") != seed:
                raise ValueError("Dataset/seed identity differs from report")
            protocol = report.get("evaluation_protocol", {})
            if protocol.get("evaluation_split", "val") != "val":
                raise ValueError("This package uses selected validation reports; use labelled validation predictions")
            floor = protocol.get("score_thresh", protocol.get("score_floor"))
            if floor is None:
                raise ValueError("Prediction score floor absent from evaluation protocol")
            if min(args.confidences + [args.primary_confidence]) < float(floor) - 1e-8:
                raise ValueError(f"Requested score below saved export floor {floor}; fresh inference would be needed")
            epoch = report.get("epoch")
            alternatives = [report_path.parent / "predictions.json"]
            if isinstance(epoch, int):
                alternatives += [report_path.parent / folder / f"epoch_{epoch:03d}" / "predictions.json"
                                 for folder in ["evaluation", "shared_evaluation", "shared_evaluations"]]
            expected_pred = report.get("predictions_sha256", report.get("prediction_sha256"))
            if not expected_pred:
                raise ValueError("Saved prediction SHA-256 required")
            pred_path = resolve_file(report.get("predictions_file", report.get("prediction_file")),
                                     report_path.parent, mappings, expected_pred, alternatives)
            opts = {**data_config.get(dataset, {}), **entry}
            audit = None
            audit_path = None
            if opts.get("audit"):
                audit_path = resolve_file(opts["audit"], base, mappings)
                audit = read_json(audit_path)
            else:
                candidates = [report_path.parent / "data_audit.json"]
                if entry.get("run_directory"):
                    candidates.append(Path(entry["run_directory"]) / "data_audit.json")
                candidates.append(args.recent_root.parent / "prepared" / dataset / "conversion.json")
                for candidate in candidates:
                    candidate = Path(remapped(candidate, mappings))
                    if candidate.is_file():
                        value = read_json(candidate)
                        value = value.get("source", value)
                        if value.get("dataset") == dataset and "val" in value:
                            audit, audit_path = value, candidate.resolve()
                            break
            if not audit:
                raise ValueError("Saved data audit required. Supply datasets[NAME].audit in --manifest")
            audit = audit.get("source", audit)
            expected_content = report.get("validation_content_sha256", report.get("evaluation_content_sha256"))
            if expected_content != audit["val"].get("content_sha256"):
                raise ValueError("Prediction report and data audit content hashes differ")
            annotations = resolve_file(opts.get("annotations", audit["paths"]["val_annotations"]),
                                       base, mappings, audit["val"]["annotations_sha256"])
            images = Path(remapped(opts.get("images", audit["paths"]["val_images"]), mappings))
            if not images.is_absolute():
                images = base / images
            doc = document(annotations)
            preds = normalize_predictions(read_json(pred_path), doc)
            count = report.get("metrics", {}).get("detection_count")
            if count is not None and count != len(preds):
                raise ValueError("Prediction count differs from saved detection_count")
            expected_images = {r["image_id"]: r for r in audit["val"]["image_records"]}
            if set(expected_images) != {im["id"] for im in doc["images"]}:
                raise ValueError("Annotation image set differs from audit")
            if any(expected_images[im["id"]]["file_name"] != im["file_name"] for im in doc["images"]):
                raise ValueError("Annotation filenames differ from audit")
            canonical_content = signature({"annotations": digest(annotations), "images": [
                {k: r[k] for k in ["image_id", "file_name", "sha256"]}
                for r in sorted(expected_images.values(), key=lambda r: r["image_id"])]})
            runs.append({"dataset": dataset, "method": method, "seed": seed,
                         "report_path": str(report_path), "report_sha256": digest(report_path),
                         "prediction_path": str(pred_path), "prediction_sha256": expected_pred,
                         "annotations_path": str(annotations), "annotations_sha256": digest(annotations),
                         "images_path": str(images.resolve()), "audit_path": str(audit_path),
                         "content_identity": canonical_content, "checkpoint_sha256": report.get("checkpoint_sha256"),
                         "epoch": epoch, "selection": args.selection, "evaluation_protocol": protocol,
                         "saved_metrics": report.get("metrics", {}), "exported_detection_count": len(preds),
                         "document": doc, "expected_images": expected_images})
        except (ValueError, OSError, KeyError, TypeError) as exc:
            errors.append(f"{dataset}/{method}/seed_{seed}: {exc}")
    if errors:
        raise ValueError("Input checks failed. No analysis started:\n" + "\n".join(errors))
    if not runs:
        raise ValueError("No selected reports found")
    by_dataset = defaultdict(list)
    for run in runs:
        by_dataset[run["dataset"]].append(run)
    for dataset, group in by_dataset.items():
        if len({r["content_identity"] for r in group}) != 1:
            raise ValueError(f"Annotation/image content differs across {dataset} runs")
        seed_sets = [{r["seed"] for r in group if r["method"] == m} for m in {r["method"] for r in group}]
        if any(s != seed_sets[0] for s in seed_sets):
            raise ValueError(f"Method seed sets differ in {dataset}")
    return sorted(runs, key=lambda r: (r["dataset"], r["seed"], r["method"]))


def coco_match(doc, preds, ious, max_dets):
    import numpy as np
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    normalized = copy.deepcopy(doc)
    # Zero is COCOeval's unmatched sentinel. Preserve original IDs for auditing.
    for index, ann in enumerate(normalized["annotations"], 1):
        ann["source_annotation_id"] = ann["id"]
        ann["id"] = index
    normalized.setdefault("info", {})
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO()
        gt.dataset = normalized
        gt.createIndex()
        if preds:
            dt = gt.loadRes(copy.deepcopy(preds))
        else:
            dt = COCO()
            dt.dataset = {"images": normalized["images"], "categories": normalized["categories"], "annotations": []}
            dt.createIndex()
        ev = COCOeval(gt, dt, "bbox")
        ev.params.imgIds = sorted(gt.imgs)
        ev.params.catIds = sorted(gt.cats)
        ev.params.iouThrs = np.array(ious)
        ev.params.maxDets = [1, 10, max_dets]
        ev.evaluate()
    return ev, gt, dt


def operating_point(ev, gt, dt, threshold, iou_index, area="all"):
    import numpy as np
    area_index = ev.params.areaRngLbl.index(area)
    target_range = ev.params.areaRng[area_index]
    valid_gt, matched, detections = set(), set(), []
    for item in ev.evalImgs:
        if item is None or list(item["aRng"]) != list(target_range):
            continue
        valid_gt.update(gid for gid, ignored in zip(item["gtIds"], item["gtIgnore"]) if not ignored)
        keep = np.asarray(item["dtScores"]) >= threshold
        for j, did in enumerate(item["dtIds"]):
            if not keep[j]:
                continue
            gid = int(item["dtMatches"][iou_index, j])
            ignored = bool(item["dtIgnore"][iou_index, j])
            state = "IGN" if ignored else "TP" if gid else "FP"
            if state == "TP":
                matched.add(gid)
            det = dt.anns[did]
            detections.append({"prediction_index": did - 1, "image_id": det["image_id"],
                               "category_id": det["category_id"], "bbox": det["bbox"],
                               "score": det["score"], "outcome": state, "matched_gt_id": gid or None,
                               "size": size_bucket(gt.anns[gid]["area"] if state == "TP" else det["area"])})
    objects_by_image, detections_by_image = defaultdict(set), defaultdict(list)
    for gid in valid_gt:
        objects_by_image[gt.anns[gid]["image_id"]].add(gid)
    for detection in detections:
        detections_by_image[detection["image_id"]].append(detection)
    return {"valid_gt": valid_gt, "matched": matched, "missed": valid_gt - matched,
            "detections": detections, "gt": gt.anns, "objects_by_image": objects_by_image,
            "detections_by_image": detections_by_image}


def metrics(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    return {"precision": precision, "recall": recall,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
            "miss_rate": fn / (tp + fn) if tp + fn else None}


def counts(point, size="all", image_id=None):
    def gt_keep(gid):
        a = point["gt"][gid]
        return (size == "all" or size_bucket(a["area"]) == size) and (image_id is None or a["image_id"] == image_id)
    valid = point["valid_gt"] if image_id is None else point["objects_by_image"][image_id]
    boxes = point["detections"] if image_id is None else point["detections_by_image"][image_id]
    objects = {g for g in valid if gt_keep(g)}
    hits = objects & point["matched"]
    detections = [d for d in boxes if (size == "all" or d["size"] == size)
                  and (image_id is None or d["image_id"] == image_id)]
    tp, fn = len(hits), len(objects - hits)
    fp = sum(d["outcome"] == "FP" for d in detections)
    ignored = sum(d["outcome"] == "IGN" for d in detections)
    assert tp == sum(d["outcome"] == "TP" for d in detections)
    return {"ground_truth": len(objects), "tp": tp, "fp": fp, "fn": fn,
            "ignored_predictions": ignored, "evaluated_detections": tp + fp, **metrics(tp, fp, fn)}


def csv_write(path, rows):
    if not rows:
        Path(path).write_text("")
        return
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def seed_summary(rows, keys, fields):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in keys)].append(row)
    output = []
    for key, group in sorted(groups.items()):
        row = dict(zip(keys, key))
        row.update(n_seeds=len(group), seeds=",".join(str(r["seed"]) for r in sorted(group, key=lambda r:r["seed"])))
        for field in fields:
            vals = [r[field] for r in group if r[field] is not None]
            row[field + "_n_defined"] = len(vals)
            row[field + "_mean"] = statistics.mean(vals) if vals else None
            row[field + "_sd"] = statistics.stdev(vals) if len(vals) > 1 else None
        output.append(row)
    return output


def fp_pairs(left, right, overlap):
    # Match LGF FPs to RetinaNet FPs, same class, one-to-one. A TP counterpart
    # does not absorb an extra duplicate FP.
    available = {i for i, d in enumerate(left)}
    pairs, right_only = [], []
    for r in sorted(right, key=lambda d: (-d["score"], d["prediction_index"])):
        choices = [(box_iou(left[i]["bbox"], r["bbox"]), i) for i in available
                   if left[i]["category_id"] == r["category_id"]]
        best = max(choices, default=(0., -1))
        if best[0] >= overlap:
            pairs.append((left[best[1]], r, best[0]))
            available.remove(best[1])
        else:
            right_only.append(r)
    return pairs, [left[i] for i in sorted(available)], right_only


def paired_analysis(dataset, seed, doc, baseline, lgf, args):
    if baseline["valid_gt"] != lgf["valid_gt"]:
        raise ValueError("Paired detector ground-truth sets differ")
    images = {im["id"]: im for im in doc["images"]}
    categories = {c["id"]: c["name"] for c in doc["categories"]}
    objects, false_positives, candidates = [], [], []
    for gid in sorted(baseline["valid_gt"]):
        a = baseline["gt"][gid]
        b, l = gid in baseline["matched"], gid in lgf["matched"]
        event = "shared_hit" if b and l else "lgf_recovery" if l else "retinanet_recovery" if b else "shared_miss"
        row = {"dataset": dataset, "seed": seed, "image_id": a["image_id"],
               "file_name": images[a["image_id"]]["file_name"], "annotation_id": a["source_annotation_id"],
               "category": categories[a["category_id"]], "size": size_bucket(a["area"]),
               "baseline_tp": int(b), "lgf_tp": int(l), "event": event}
        objects.append(row)
        if event in EVENTS:
            candidates.append({**row, "focus_bbox": a["bbox"], "focus_gt_id": gid, "focus_prediction_index": None})
    bfp, lfp = defaultdict(list), defaultdict(list)
    for point, target in [(baseline, bfp), (lgf, lfp)]:
        for d in point["detections"]:
            if d["outcome"] == "FP":
                target[d["image_id"]].append(d)
    for image_id in sorted(images):
        pairs, b_only, l_only = fp_pairs(bfp[image_id], lfp[image_id], args.fp_pair_iou)
        counterpart = {r["prediction_index"]: (b["prediction_index"], overlap) for b, r, overlap in pairs}
        for event, dets in [("shared_fp", [r for _, r, _ in pairs]),
                            ("retinanet_only_fp", b_only), ("lgf_only_fp", l_only)]:
            for d in dets:
                row = {"dataset": dataset, "seed": seed, "image_id": image_id,
                       "file_name": images[image_id]["file_name"], "category": categories[d["category_id"]],
                       "size": size_bucket(d["bbox"][2] * d["bbox"][3]), "event": event,
                       "method": "baseline" if event == "retinanet_only_fp" else "lgf_original",
                       "prediction_index": d["prediction_index"], "score": d["score"],
                       "counterpart_prediction_index": counterpart.get(d["prediction_index"], (None, None))[0] if event == "shared_fp" else None,
                       "counterpart_box_iou": counterpart.get(d["prediction_index"], (None, None))[1] if event == "shared_fp" else None}
                false_positives.append(row)
                if event == "lgf_only_fp":
                    candidates.append({**row, "focus_bbox": d["bbox"], "focus_gt_id": None,
                                       "focus_prediction_index": d["prediction_index"]})
    summaries = []
    for size in SIZES:
        subset = [r for r in objects if size == "all" or r["size"] == size]
        errors = [r for r in false_positives if size == "all" or r["size"] == size]
        summaries.append({"dataset": dataset, "seed": seed, "size": size,
                          "confidence": args.primary_confidence, "iou": args.primary_iou,
                          "ground_truth": len(subset),
                          **{event: sum(r["event"] == event for r in subset)
                             for event in ["shared_hit", "lgf_recovery", "retinanet_recovery", "shared_miss"]},
                          **{event: sum(r["event"] == event for r in errors)
                             for event in ["shared_fp", "lgf_only_fp", "retinanet_only_fp"]}})
    return objects, false_positives, candidates, summaries


def select_examples(candidates, args, dataset):
    rng = random.Random(str(args.example_random_seed) + ":" + dataset)
    chosen, inventory = [], []
    for event in EVENTS:
        rows = [r for r in candidates if r["event"] == event and r["seed"] == args.example_seed
                and (args.example_size == "any" or r["size"] == args.example_size)]
        rng.shuffle(rows)
        selected, used = [], set()
        for row in rows:
            if row["image_id"] not in used:
                selected.append(row)
                used.add(row["image_id"])
            if len(selected) == args.examples_per_category:
                break
        chosen.extend(selected)
        inventory.append({"dataset": dataset, "seed": args.example_seed, "category": event,
                          "size_filter": args.example_size, "eligible_objects_or_boxes": len(rows),
                          "eligible_images": len({r["image_id"] for r in rows}), "selected_images": len(selected)})
    return chosen, inventory


def draw_figure(row, doc, points, source_run, destination, args):
    from PIL import Image, ImageDraw, ImageFont
    import PIL
    categories = {c["id"]: c["name"] for c in doc["categories"]}
    im = next(im for im in doc["images"] if im["id"] == row["image_id"])
    root = Path(source_run["images_path"]).resolve()
    image_path = (root / im["file_name"]).resolve()
    if not image_path.is_relative_to(root):
        raise ValueError("Image filename escapes image directory")
    expected = source_run["expected_images"][row["image_id"]]["sha256"]
    if digest(image_path) != expected:
        raise ValueError(f"Example image bytes differ from saved validation audit: {image_path}")
    frame = Image.open(image_path).convert("RGB")
    if frame.size != (im["width"], im["height"]):
        raise ValueError("Example image dimensions differ from annotation")
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 14)
        title_font = ImageFont.truetype("DejaVuSans.ttf", 20)
        label_font = ImageFont.truetype("DejaVuSans.ttf", 11)
    except OSError:
        font = title_font = label_font = ImageFont.load_default()
    colors = {"TP": "#00d46a", "FP": "#ff3b4d", "FN": "#ffb000", "IGN": "#aaaaaa", "GT": "#28c9ff"}
    models = ["baseline", "lgf_original"]
    if args.render_all_models:
        models += sorted(m for m in points if m not in models)
    panels = [("Ground truth", None)] + [(LABELS.get(m, m), m) for m in models]
    cols = 3
    cell_w, scene_h, zoom_h = 560, 340, 260
    groups = math.ceil(len(panels) / cols)
    group_h = scene_h + zoom_h + 52
    canvas = Image.new("RGB", (cols * cell_w, 112 + groups * group_h + 46), "#ffffff")
    draw = ImageDraw.Draw(canvas)
    title = row["event"].replace("_", " ")
    draw.text((14, 10), f"{row['dataset']} | seed {row['seed']} | {title}", fill="black", font=title_font)
    draw.text((14, 40), f"{row['file_name']} | {row['category']} | {row['size']}", fill="black", font=font)
    draw.text((14, 64), f"score >= {args.primary_confidence:g}, correct class and IoU >= {args.primary_iou:g}; lower panels: target crop", fill="black", font=font)
    x, y, w, h = row["focus_bbox"]
    cx, cy = x + w / 2, y + h / 2
    crop_w, crop_h = max(96., w * 3), max(96., h * 3)
    crop = (max(0, int(cx - crop_w / 2)), max(0, int(cy - crop_h / 2)),
            min(frame.width, int(math.ceil(cx + crop_w / 2))), min(frame.height, int(math.ceil(cy + crop_h / 2))))
    if crop[2] <= crop[0] or crop[3] <= crop[1]:
        raise ValueError("Selected focus box has no visible image intersection")
    baseline = points["baseline"]
    gt_boxes = [a for gid, a in baseline["gt"].items() if a["image_id"] == row["image_id"]]

    def panel(method, viewport, position, height):
        vx, vy, ex, ey = viewport
        region = frame.crop(viewport)
        scale = min((cell_w - 12) / region.width, (height - 8) / region.height)
        resized = region.resize((max(1, round(region.width * scale)), max(1, round(region.height * scale))), Image.Resampling.LANCZOS)
        px = position[0] + (cell_w - resized.width) // 2
        py = position[1] + (height - resized.height) // 2
        canvas.paste(resized, (px, py))

        def box(bbox, state, label, focus=False):
            bx, by, bw, bh = bbox
            if bx + bw <= vx or by + bh <= vy or bx >= ex or by >= ey:
                return
            rect = (px + max(0, bx - vx) * scale, py + max(0, by - vy) * scale,
                    px + min(ex - vx, bx + bw - vx) * scale, py + min(ey - vy, by + bh - vy) * scale)
            draw.rectangle(rect, outline=colors[state], width=4 if focus else 2)
            if focus:
                outer = (max(px, rect[0] - 3), max(py, rect[1] - 3),
                         min(px + resized.width, rect[2] + 3), min(py + resized.height, rect[3] + 3))
                draw.rectangle(outer, outline="#ffffff", width=1)
            text_pos = (rect[0], max(py, rect[1] - 13))
            bounds = draw.textbbox(text_pos, label, font=label_font)
            draw.rectangle(bounds, fill="#111111")
            draw.text(text_pos, label, fill=colors[state], font=label_font)
        if method is None:
            for ann in gt_boxes:
                state = "IGN" if ann.get("iscrowd") else "GT"
                box(ann["bbox"], state, categories[ann["category_id"]], ann["id"] == row["focus_gt_id"])
        else:
            point = points[method]
            for d in point["detections_by_image"][row["image_id"]]:
                focus = d["matched_gt_id"] == row["focus_gt_id"] if row["focus_gt_id"] else (
                    method == "lgf_original" and d["prediction_index"] == row["focus_prediction_index"])
                box(d["bbox"], d["outcome"], f"{categories[d['category_id']]} {d['score']:.2f}", focus)
            for gid in point["missed"]:
                ann = point["gt"][gid]
                if ann["image_id"] == row["image_id"]:
                    box(ann["bbox"], "FN", "MISS " + categories[ann["category_id"]], gid == row["focus_gt_id"])

    for i, (label, method) in enumerate(panels):
        px, py = (i % cols) * cell_w, 106 + (i // cols) * group_h
        draw.text((px + 12, py), label, fill="black", font=title_font)
        panel(method, (0, 0, frame.width, frame.height), (px, py + 30), scene_h)
        panel(method, crop, (px, py + 34 + scene_h), zoom_h)
    draw.text((14, canvas.height - 31), "Cyan: GT   Green: TP   Red: FP   Orange: missed GT   Grey: ignored/crowd. White outline: selected target.", fill="black", font=font)
    canvas.save(destination)
    return {"figure": str(destination), "image_sha256": expected, "crop_xyxy": list(crop), "pillow_version": PIL.__version__}


def analyze(args, runs):
    import numpy as np
    import pycocotools
    if not args.no_render:
        import PIL
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output directory is nonempty. Choose a new --output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    if not args.no_render:
        (output / "figures").mkdir(exist_ok=True)
    ious = sorted(set(args.ious + [args.primary_iou]))
    thresholds = sorted(set(args.confidences + [args.primary_confidence]))
    table, coco_table, image_table, paired_table = [], [], [], []
    paired_objects, paired_fps, candidate_rows, example_inventory, examples = [], [], [], [], []
    provenance = [{k: v for k, v in run.items() if k not in {"document", "predictions", "expected_images"}} for run in runs]
    groups = defaultdict(list)
    for run in runs:
        groups[(run["dataset"], run["seed"])].append(run)
    with (output / "detection_outcomes.jsonl").open("w") as detection_log, (output / "missed_objects.jsonl").open("w") as miss_log:
        for (dataset, seed), group in sorted(groups.items()):
            points, group_runs = {}, {r["method"]: r for r in group}
            for run in group:
                method = run["method"]
                print(f"ANALYZE {dataset}/{method}/seed_{seed}: {run['exported_detection_count']} exported boxes", flush=True)
                if (digest(run["prediction_path"]) != run["prediction_sha256"]
                        or digest(run["report_path"]) != run["report_sha256"]
                        or digest(run["annotations_path"]) != run["annotations_sha256"]):
                    raise ValueError("Selected prediction/report/annotations changed after input checks")
                predictions = normalize_predictions(read_json(run["prediction_path"]), run["document"])
                ev, gt, dt = coco_match(run["document"], predictions, ious, args.max_dets)
                identity = {"dataset": dataset, "method": method, "seed": seed}
                for ti, iou in enumerate(ious):
                    for confidence in thresholds:
                        point = operating_point(ev, gt, dt, confidence, ti)
                        for size in SIZES:
                            row = {**identity, "size": size, "confidence": confidence, "iou": iou,
                                   "image_count": len(run["document"]["images"]), **counts(point, size)}
                            row["fp_per_image"] = row["fp"] / row["image_count"]
                            table.append(row)
                            official = point if size == "all" else operating_point(ev, gt, dt, confidence, ti, size)
                            # The official per-area matcher already applies COCO's
                            # area ignore rules. Do not partition its result again.
                            cr = counts(official)
                            coco_table.append({**identity, "coco_area": size, "confidence": confidence,
                                               "iou": iou, **cr})
                        if confidence == args.primary_confidence and iou == args.primary_iou:
                            points[method] = point
                            images = {im["id"]: im for im in run["document"]["images"]}
                            cats = {c["id"]: c["name"] for c in run["document"]["categories"]}
                            for d in point["detections"]:
                                row = {**identity, **d, "category": cats[d["category_id"]],
                                       "file_name": images[d["image_id"]]["file_name"], "confidence": confidence, "iou": iou}
                                if d["matched_gt_id"]:
                                    row["matched_annotation_id"] = point["gt"][d["matched_gt_id"]]["source_annotation_id"]
                                detection_log.write(json.dumps(row, allow_nan=False) + "\n")
                            for gid in sorted(point["missed"]):
                                ann = point["gt"][gid]
                                miss_log.write(json.dumps({**identity, "image_id": ann["image_id"],
                                    "annotation_id": ann["source_annotation_id"], "category": cats[ann["category_id"]],
                                    "bbox": ann["bbox"], "area": ann["area"], "size": size_bucket(ann["area"]),
                                    "file_name": images[ann["image_id"]]["file_name"], "confidence": confidence, "iou": iou}) + "\n")
                            for im in images.values():
                                for size in SIZES:
                                    image_table.append({**identity, "image_id": im["id"], "file_name": im["file_name"],
                                        "condition": im.get("condition", im.get("weather", "unknown")),
                                        "size": size, "confidence": confidence, "iou": iou, **counts(point, size, im["id"])})
            if {"baseline", "lgf_original"}.issubset(points):
                doc = group[0]["document"]
                objects, fps, candidates, paired = paired_analysis(dataset, seed, doc, points["baseline"], points["lgf_original"], args)
                paired_objects.extend(objects)
                paired_fps.extend(fps)
                candidate_rows.extend(candidates)
                paired_table.extend(paired)
                if seed == args.example_seed:
                    selected, inventory = select_examples(candidates, args, dataset)
                    example_inventory.extend(inventory)
                    for index, row in enumerate(selected, 1):
                        result = dict(row)
                        if not args.no_render:
                            destination = output / "figures" / f"{dataset}_s{seed}_{row['event']}_{index:02d}.png"
                            result.update(draw_figure(row, doc, points, group_runs["baseline"], destination, args))
                        examples.append(result)
    csv_write(output / "counts_per_seed.csv", table)
    csv_write(output / "counts_seed_summary.csv", seed_summary(table, ["dataset", "method", "size", "confidence", "iou"],
              ["ground_truth", "tp", "fp", "fn", "precision", "recall", "f1", "miss_rate", "fp_per_image"]))
    csv_write(output / "coco_area_counts_per_seed.csv", coco_table)
    csv_write(output / "image_counts.csv", image_table)
    csv_write(output / "paired_objects.csv", paired_objects)
    csv_write(output / "paired_false_positives.csv", paired_fps)
    csv_write(output / "paired_counts_per_seed.csv", paired_table)
    if paired_table:
        csv_write(output / "paired_counts_seed_summary.csv", seed_summary(paired_table, ["dataset", "size", "confidence", "iou"],
                  ["shared_hit", "lgf_recovery", "retinanet_recovery", "shared_miss", "shared_fp", "lgf_only_fp", "retinanet_only_fp"]))
    csv_write(output / "example_category_inventory.csv", example_inventory)
    write_json(output / "all_example_candidates.json", candidate_rows)
    write_json(output / "selected_examples.json", examples)
    csv_write(output / "figure_review_template.csv", [{"dataset": e["dataset"], "seed": e["seed"],
        "event": e["event"], "file_name": e["file_name"], "category": e["category"], "size": e["size"],
        "figure": e.get("figure", ""), "observed_failure_or_recovery": "", "visible_occlusion_or_weather": "",
        "annotation_review": "", "caption_text": "", "include_in_manuscript_or_supplement": ""} for e in examples])
    source_metrics = []
    for run in runs:
        saved = run["saved_metrics"]
        source_metrics.append({"dataset": run["dataset"], "method": run["method"], "seed": run["seed"],
            "selection": run["selection"], "epoch": run["epoch"], "split": "validation", "AP_scale": "0-1",
            **{k: v for k, v in saved.items() if k != "detection_count"}, "exported_detection_count": run["exported_detection_count"]})
    csv_write(output / "source_metrics_per_seed.csv", source_metrics)
    manifest = {"version": VERSION, "analysis": "selected-validation-checkpoint diagnostics", "training_performed": False,
                "selection": args.selection, "confidence_thresholds": thresholds, "iou_thresholds": ious,
                "primary_confidence": args.primary_confidence, "primary_iou": args.primary_iou,
                "max_dets_per_image_per_category_in_COCOeval": args.max_dets,
                "fp_pair_iou": args.fp_pair_iou, "example_seed": args.example_seed,
                "example_random_seed": args.example_random_seed, "examples_per_category": args.examples_per_category,
                "example_size": args.example_size, "size_area_source": "GT: annotation area; FP/ignored: predicted bbox width*height",
                "diagnostic_partition": "small <1024; medium >=1024 and <9216; large >=9216; original coordinates",
                "official_coco_area_counts": "Native COCO area matcher, including shared size endpoints and area ignore rules",
                "tn": None, "tn_reason": "No fixed finite background detection units",
                "evaluated_area_semantics": "TP/FN use ground-truth annotation area; FP uses its own predicted box area. The exclusive diagnostic partition adds to all.",
                "score_interpretation": "Common thresholds are descriptive operating points; native scores are not cross-model calibrated",
                "sample_sd_interpretation": "Variation across seeds, not a confidence interval or significance test",
                "qualitative_interpretation": "Cases selected reproducibly within outcome categories; inspect all candidates and images before publication",
                "extra_fp_definition": "LGF FP without a one-to-one same-class RetinaNet FP counterpart at the declared box IoU",
                "script_sha256": digest(__file__), "numpy_version": np.__version__,
                "pycocotools_version": importlib.metadata.version("pycocotools"), "runs": provenance}
    write_json(output / "analysis_manifest.json", manifest)
    print(f"COMPLETE: {output}", flush=True)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ugs-root", type=Path, default=Path("/nas.dbms/asera/paper1_original/ugs_comparison/runs"))
    p.add_argument("--recent-root", type=Path, default=Path("/nas.dbms/asera/paper1_original/reviewer34/runs"))
    p.add_argument("--se-cbam-root", type=Path)
    p.add_argument("--datasets", nargs="+", default=["coco_nw", "coco_weather", "acdc"])
    p.add_argument("--methods", nargs="+", default=["baseline", "lgf_original", "ugs_reimpl", "yolo11m", "rtdetr-l"])
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 1337, 2025])
    p.add_argument("--selection", choices=["best_map", "best_ap_small"], default="best_map")
    p.add_argument("--manifest", type=Path, help="Explicit selected report/audit paths; see README")
    p.add_argument("--remap", action="append", default=["/tmp/asera=/nas.dbms/asera"], help="OLD=NEW prefix remap; recorded in resolved source paths")
    p.add_argument("--confidences", nargs="+", type=float, default=[0.05, 0.25, 0.50])
    p.add_argument("--ious", nargs="+", type=float, default=[0.50, 0.75])
    p.add_argument("--primary-confidence", type=float, default=0.25)
    p.add_argument("--primary-iou", type=float, default=0.50)
    p.add_argument("--max-dets", type=int, default=100)
    p.add_argument("--fp-pair-iou", type=float, default=0.50)
    p.add_argument("--example-seed", type=int, default=42)
    p.add_argument("--example-random-seed", type=int, default=20260930)
    p.add_argument("--example-size", choices=["small", "medium", "large", "any"], default="small")
    p.add_argument("--examples-per-category", type=int, default=2)
    p.add_argument("--render-all-models", action="store_true")
    p.add_argument("--no-render", action="store_true", help="Count/log detections without loading image pixels")
    p.add_argument("--check", action="store_true", help="Validate inputs, hashes, class mapping and selection; no analysis output")
    p.add_argument("--output", type=Path, default=Path("/nas.dbms/asera/paper1_original/balanced_failure_analysis_v1"))
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        values = args.confidences + args.ious + [args.primary_confidence, args.primary_iou, args.fp_pair_iou]
        if not all(math.isfinite(v) and 0 <= v <= 1 for v in values):
            raise ValueError("Confidence/IoU values must be finite and within [0,1]")
        if min(args.ious + [args.primary_iou, args.fp_pair_iou]) <= 0 or args.max_dets < 10 or args.examples_per_category < 1:
            raise ValueError("Positive IoU, max-dets >=10 and examples-per-category >=1 required")
        runs = load_inputs(args)
        print(f"INPUT CHECK PASSED: {len(runs)} selected reports", flush=True)
        if args.check:
            for run in runs:
                print(f"{run['dataset']}/{run['method']}/seed_{run['seed']} epoch={run['epoch']} boxes={run['exported_detection_count']} {run['selection']}")
            return 0
        analyze(args, runs)
        return 0
    except (ValueError, OSError, KeyError, ImportError, TypeError) as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
