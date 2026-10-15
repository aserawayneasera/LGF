"""Capture native original-evaluator detections without changing its AP calculation."""
import json
from pathlib import Path
import tempfile
import time

import torch
from core_bridge import load_core
from utils import plain, sha256, signature, write_json


def model_protocol(model):
    return {"min_size": list(model.transform.min_size), "max_size": model.transform.max_size,
            "mean": list(model.transform.image_mean), "std": list(model.transform.image_std),
            "score_thresh": model.score_thresh, "nms_thresh": model.nms_thresh,
            "topk_candidates": model.topk_candidates, "detections_per_img": model.detections_per_img,
            "anchor_sizes": model.anchor_generator.sizes,
            "anchor_ratios": model.anchor_generator.aspect_ratios,
            "iou_thresholds": [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95],
            "max_dets": [1, 10, 100], "inference_precision": "FP32", "weights": "EMA"}


def evaluate(model, loader, annotations, device, output_dir, audit_result, metadata):
    core = load_core()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    detections = []
    def capture(_module, inputs, outputs):
        # Called after native RetinaNet postprocessing, in original-image coordinates.
        images, = inputs
        image_ids = capture.image_ids
        for image_id, out in zip(image_ids, outputs):
            # Use the identical NumPy subtraction/float conversion as the
            # original evaluator, including float32 rounding of width/height.
            boxes = out["boxes"].detach().cpu().numpy()
            scores = out["scores"].detach().cpu().numpy()
            labels = out["labels"].detach().cpu().numpy()
            for box, score, label in zip(boxes, scores, labels):
                x1, y1, x2, y2 = box
                detections.append({"image_id": image_id,
                    "category_id": loader.dataset.label_to_cat_id[int(label)],
                    "bbox": [float(x1), float(y1), float(x2-x1), float(y2-y1)], "score": float(score)})
    class IdentifiedLoader:
        dataset = loader.dataset
        def __iter__(self):
            for images, targets in loader:
                capture.image_ids = [int(t["image_id"].item()) for t in targets]
                yield images, targets
        def __len__(self):
            return len(loader)
    # COCO.loadRes in some pycocotools releases requires info. Add that metadata
    # to a temporary copy only; boxes, categories and evaluation images are intact.
    doc = json.loads(Path(annotations).read_text())
    doc.setdefault("info", {"description": "Paper 1 original evaluation"})
    model.eval()
    handle = model.register_forward_hook(capture)
    start = time.perf_counter()
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", dir=output_dir) as f:
            json.dump(doc, f)
            f.flush()
            with torch.no_grad():
                metrics = core.coco_evaluation(model, IdentifiedLoader(), f.name, device)
    finally:
        handle.remove()
    elapsed = time.perf_counter() - start
    pred_path = output_dir / "predictions.json"
    write_json(pred_path, detections)
    if metrics["detection_count"] != len(detections):
        raise RuntimeError("Captured prediction count differs from the original evaluator")
    protocol = model_protocol(model)
    common = {"val_content_sha256": audit_result["val"]["content_sha256"],
              "categories": audit_result["val"]["categories"], "protocol": protocol}
    result = {**metadata, "metrics": plain(metrics), "evaluation_protocol": protocol,
              "evaluation_signature": signature(common), "dataset": audit_result["dataset"],
              "validation_content_sha256": common["val_content_sha256"],
              "predictions_file": str(pred_path), "predictions_sha256": sha256(pred_path),
              "evaluation_seconds": elapsed,
              "evaluation_status": "validation; not an independent held-out test"}
    write_json(output_dir / "metrics.json", result)
    return result
