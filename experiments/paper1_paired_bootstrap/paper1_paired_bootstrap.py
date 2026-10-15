#!/usr/bin/env python3
"""Paired hierarchical bootstrap for COCO APsmall from saved detections.

This script does not train a detector and does not run detector inference. It
reuses COCO-format predictions already saved by the Paper 1 runners and
recomputes COCO APsmall on bootstrap-resampled validation images.

Example (run once per dataset):

    python paper1_paired_bootstrap.py \
      --annotations /path/to/mini_val2017_non_weather-500k6c.json \
      --pair 42   /path/baseline/seed_42/best_ap_small.metrics.json \
                  /path/lgf/seed_42/best_ap_small.metrics.json \
      --pair 1337 /path/baseline/seed_1337/best_ap_small.metrics.json \
                  /path/lgf/seed_1337/best_ap_small.metrics.json \
      --pair 2025 /path/baseline/seed_2025/best_ap_small.metrics.json \
                  /path/lgf/seed_2025/best_ap_small.metrics.json \
      --replicates 10000 --workers 8 \
      --output bootstrap_coco_nw

Each report path may instead be a direct predictions.json path. Report files
are preferable because the script can verify the seed, checkpoint-selection
rule, prediction hash, and the reported full-data APsmall.

Method: within every replicate, validation images are sampled with replacement
and training-seed pairs are sampled with replacement. Baseline and proposed
detections always receive the same image and seed draws. The reported interval
is the percentile 95% confidence interval for the mean paired APsmall change.

Compatibility: --iou-backend auto probes pycocotools' compiled bbox IoU helper.
If the probe raises the known NumPy ndarray-conversion TypeError, the script
uses a NumPy bbox IoU implementation, including COCO crowd handling. COCOeval
still performs matching and metric accumulation. Use --iou-backend numpy to
select this implementation explicitly. This option affects bbox evaluation
only; it does not replace mask or keypoint evaluation.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
from importlib.metadata import PackageNotFoundError, version as package_version
import io
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import sys
from types import MethodType
from typing import Any

import numpy as np


VERSION = "1.0.1"
IOU_THRESHOLDS = np.linspace(0.50, 0.95, 10)
RECALL_THRESHOLDS = np.linspace(0.00, 1.00, 101)
MAX_DETECTIONS = 100
_BOOTSTRAP_STATE: dict[str, Any] | None = None


def bbox_iou_numpy(
    detection_boxes: list[list[float]],
    ground_truth_boxes: list[list[float]],
    iscrowd: list[int],
) -> np.ndarray:
    """COCO xywh overlap, with intersection/detection-area for crowd GT.

    The standard definition is implemented by bbIou in COCO's maskApi.c:
    https://github.com/cocodataset/cocoapi/blob/master/common/maskApi.c
    No pixel-inclusive +1, coordinate clipping, or denominator epsilon is used.
    """
    detections = np.asarray(detection_boxes, dtype=np.float64).reshape(-1, 4)
    ground_truth = np.asarray(ground_truth_boxes, dtype=np.float64).reshape(-1, 4)
    crowd = np.asarray(iscrowd, dtype=bool).reshape(-1)
    if crowd.size != len(ground_truth):
        raise ValueError("Each ground-truth box must have one iscrowd flag")
    for boxes in (detections, ground_truth):
        if not np.all(np.isfinite(boxes)) or np.any(boxes[:, 2:] < 0):
            raise ValueError("IoU requires finite xywh boxes with nonnegative widths and heights")
    overlaps = np.zeros((len(detections), len(ground_truth)), dtype=np.float64)
    if overlaps.size == 0:
        return overlaps

    lower = np.maximum(detections[:, None, :2], ground_truth[None, :, :2])
    upper = np.minimum(
        (detections[:, :2] + detections[:, 2:])[:, None, :],
        (ground_truth[:, :2] + ground_truth[:, 2:])[None, :, :],
    )
    extent = np.maximum(upper - lower, 0.0)
    intersection = extent[:, :, 0] * extent[:, :, 1]
    detection_area = detections[:, 2] * detections[:, 3]
    ground_truth_area = ground_truth[:, 2] * ground_truth[:, 3]
    union = detection_area[:, None] + ground_truth_area[None, :] - intersection
    denominator = np.where(crowd[None, :], detection_area[:, None], union)
    # Zero-area or disjoint boxes have zero overlap, including crowd boxes.
    np.divide(intersection, denominator, out=overlaps, where=intersection > 0)
    return overlaps


def compute_bbox_iou_numpy(evaluator: Any, image_id: int, category_id: int) -> Any:
    """Replace only one COCOeval instance's bbox overlap computation."""
    params = evaluator.params
    if params.iouType != "bbox":
        raise ValueError("The NumPy IoU backend supports bounding boxes only")
    categories = [category_id] if params.useCats else params.catIds
    ground_truth = [
        annotation for category in categories
        for annotation in evaluator._gts[image_id, category]
    ]
    detections = [
        annotation for category in categories
        for annotation in evaluator._dts[image_id, category]
    ]
    if not ground_truth or not detections:
        return []
    order = np.argsort([-annotation["score"] for annotation in detections], kind="mergesort")
    selected = [detections[index] for index in order[: params.maxDets[-1]]]
    return bbox_iou_numpy(
        [annotation["bbox"] for annotation in selected],
        [annotation["bbox"] for annotation in ground_truth],
        [int(annotation["iscrowd"]) for annotation in ground_truth],
    )


def select_iou_backend(requested: str) -> str:
    """Probe valid boxes so an environment error is separate from user data."""
    if requested == "numpy":
        return "numpy"
    if requested not in ("auto", "pycocotools"):
        raise ValueError(f"Unknown IoU backend: {requested}")
    try:
        from pycocotools import mask as mask_utils
    except ImportError as exc:
        raise RuntimeError("Install pycocotools first: python -m pip install pycocotools") from exc
    try:
        observed = mask_utils.iou(
            [[0.0, 0.0, 2.0, 2.0], [5.0, 5.0, 2.0, 2.0]],
            [[0.0, 0.0, 2.0, 2.0], [0.0, 0.0, 4.0, 4.0]],
            [0, 1],
        )
    except TypeError as exc:
        if "Cannot convert numpy.ndarray to numpy.ndarray" not in str(exc):
            raise
        if requested == "pycocotools":
            raise RuntimeError(
                "pycocotools' compiled bbox IoU helper failed on valid probe boxes. "
                "Rerun with --iou-backend numpy or repair the Python environment."
            ) from exc
        print(
            f"Compiled bbox IoU probe failed ({exc}); using NumPy bbox IoU.",
            file=sys.stderr,
            flush=True,
        )
        return "numpy"
    if not np.array_equal(observed, np.array([[1.0, 1.0], [0.0, 0.0]])):
        raise RuntimeError("pycocotools returned unexpected bbox IoU values on valid probe boxes")
    return "pycocotools"


def installed_version(package: str) -> str:
    try:
        return package_version(package)
    except PackageNotFoundError:
        return "unknown (package metadata unavailable)"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, path)


def validate_predictions(predictions: Any, image_ids: set[int], category_ids: set[int], source: Path) -> None:
    if not isinstance(predictions, list):
        raise ValueError(f"{source}: predictions must be a JSON list")
    required = {"image_id", "category_id", "bbox", "score"}
    for index, detection in enumerate(predictions):
        if not isinstance(detection, dict) or not required.issubset(detection):
            raise ValueError(f"{source}: detection {index} lacks {sorted(required)}")
        if detection["image_id"] not in image_ids:
            raise ValueError(f"{source}: detection {index} has an unknown image_id")
        if detection["category_id"] not in category_ids:
            raise ValueError(f"{source}: detection {index} has an unknown category_id")
        box = detection["bbox"]
        if not isinstance(box, list) or len(box) != 4:
            raise ValueError(f"{source}: detection {index} has an invalid bbox")
        values = [*box, detection["score"]]
        if not all(isinstance(value, (int, float)) and math.isfinite(float(value)) for value in values):
            raise ValueError(f"{source}: detection {index} contains a nonfinite value")
        if float(box[2]) < 0 or float(box[3]) < 0:
            raise ValueError(f"{source}: detection {index} has a negative box width or height")


def expected_prediction_hash(report: dict[str, Any]) -> str | None:
    for key in ("prediction_sha256", "predictions_sha256"):
        value = report.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def locate_predictions(report_path: Path, report: dict[str, Any]) -> Path:
    expected_hash = expected_prediction_hash(report)
    named_candidates: list[Path] = []
    for key in ("prediction_file", "predictions_file"):
        value = report.get(key)
        if isinstance(value, str) and value:
            raw = Path(value).expanduser()
            named_candidates.extend([raw, report_path.parent / raw, report_path.parent / raw.name])
    named_candidates.append(report_path.parent / "predictions.json")

    checked: set[Path] = set()
    for candidate in named_candidates:
        candidate = candidate.resolve()
        if candidate in checked or not candidate.is_file():
            continue
        checked.add(candidate)
        if expected_hash is None or sha256(candidate) == expected_hash:
            return candidate

    if expected_hash:
        # Absolute paths embedded in reports can become stale after a run folder
        # is copied. Recover only a byte-identical prediction file.
        for candidate in sorted(report_path.parent.rglob("predictions.json")):
            candidate = candidate.resolve()
            if candidate not in checked and sha256(candidate) == expected_hash:
                return candidate
        raise FileNotFoundError(
            f"{report_path}: could not locate predictions.json with SHA-256 {expected_hash}"
        )
    raise FileNotFoundError(
        f"{report_path}: no prediction path/hash was recorded; pass predictions.json directly"
    )


def load_prediction_source(
    source: Path,
    expected_seed: str,
    image_ids: set[int],
    category_ids: set[int],
) -> tuple[list[dict[str, Any]], dict[str, Any], Path]:
    source = source.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    document = read_json(source)
    if isinstance(document, list):
        report: dict[str, Any] = {}
        prediction_path = source
        predictions = document
    elif isinstance(document, dict):
        report = document
        prediction_path = locate_predictions(source, report)
        prediction_hash = expected_prediction_hash(report)
        if prediction_hash and sha256(prediction_path) != prediction_hash:
            raise ValueError(f"{prediction_path}: SHA-256 differs from {source}")
        predictions = read_json(prediction_path)
        report_seed = report.get("seed")
        if report_seed is not None and str(report_seed) != expected_seed:
            raise ValueError(
                f"{source}: report seed {report_seed!r} differs from --pair seed {expected_seed!r}"
            )
    else:
        raise ValueError(f"{source}: expected a metrics-report object or prediction list")
    validate_predictions(predictions, image_ids, category_ids, prediction_path)
    return predictions, report, prediction_path


def prepare_coco_evaluation(
    annotation_document: dict[str, Any],
    predictions: list[dict[str, Any]],
    iou_backend: str = "pycocotools",
) -> tuple[list[list[dict[str, Any] | None]], float]:
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError as exc:
        raise RuntimeError("Install pycocotools first: python -m pip install pycocotools") from exc

    # COCO.loadRes requires an info member in some releases. The source
    # annotation file is never modified.
    coco_document = dict(annotation_document)
    coco_document.setdefault("info", {})
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        ground_truth = COCO()
        ground_truth.dataset = coco_document
        ground_truth.createIndex()
        if predictions:
            detections = ground_truth.loadRes(predictions)
        else:
            detections = COCO()
            detections.dataset = {
                "images": list(coco_document["images"]),
                "categories": list(coco_document["categories"]),
                "annotations": [],
            }
            detections.createIndex()
        evaluator = COCOeval(ground_truth, detections, "bbox")
        if iou_backend == "numpy":
            evaluator.computeIoU = MethodType(compute_bbox_iou_numpy, evaluator)
        elif iou_backend != "pycocotools":
            raise ValueError(f"Unknown resolved IoU backend: {iou_backend}")
        evaluator.params.imgIds = sorted(image["id"] for image in coco_document["images"])
        evaluator.params.catIds = sorted(category["id"] for category in coco_document["categories"])
        evaluator.params.iouThrs = IOU_THRESHOLDS.copy()
        evaluator.params.recThrs = RECALL_THRESHOLDS.copy()
        evaluator.params.maxDets = [1, 10, MAX_DETECTIONS]
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()

    params = evaluator._paramsEval
    if params is None:
        raise RuntimeError("COCO evaluation did not preserve its evaluation parameters")
    try:
        small_index = list(params.areaRngLbl).index("small")
    except ValueError as exc:
        raise RuntimeError("COCO evaluator has no small-object area range") from exc
    image_count = len(params.imgIds)
    area_count = len(params.areaRng)
    grid: list[list[dict[str, Any] | None]] = []
    for category_index in range(len(params.catIds)):
        start = category_index * area_count * image_count + small_index * image_count
        grid.append(list(evaluator.evalImgs[start : start + image_count]))
    return grid, float(evaluator.stats[3])


def ap_small(grid: list[list[dict[str, Any] | None]], sampled_images: np.ndarray) -> float:
    """Reproduce COCOeval accumulation for AP, area=small, maxDets=100."""
    total_precision = 0.0
    precision_cells = 0
    for category_entries in grid:
        entries = [category_entries[int(index)] for index in sampled_images]
        entries = [entry for entry in entries if entry is not None]
        if not entries:
            continue
        scores = np.concatenate([entry["dtScores"][:MAX_DETECTIONS] for entry in entries])
        order = np.argsort(-scores, kind="mergesort")
        matches = np.concatenate(
            [entry["dtMatches"][:, :MAX_DETECTIONS] for entry in entries], axis=1
        )[:, order]
        detection_ignored = np.concatenate(
            [entry["dtIgnore"][:, :MAX_DETECTIONS] for entry in entries], axis=1
        )[:, order]
        ground_truth_ignored = np.concatenate([entry["gtIgnore"] for entry in entries])
        positive_count = int(np.count_nonzero(ground_truth_ignored == 0))
        if positive_count == 0:
            continue

        true_positives = np.logical_and(matches, np.logical_not(detection_ignored))
        false_positives = np.logical_and(np.logical_not(matches), np.logical_not(detection_ignored))
        tp_sum = np.cumsum(true_positives, axis=1, dtype=float)
        fp_sum = np.cumsum(false_positives, axis=1, dtype=float)
        for tp, fp in zip(tp_sum, fp_sum):
            recall = tp / positive_count
            precision = tp / (tp + fp + np.spacing(1))
            if precision.size:
                precision = np.maximum.accumulate(precision[::-1])[::-1]
                positions = np.searchsorted(recall, RECALL_THRESHOLDS, side="left")
                valid = positions < precision.size
                interpolated = np.zeros(RECALL_THRESHOLDS.size, dtype=float)
                interpolated[valid] = precision[positions[valid]]
                total_precision += float(interpolated.sum())
            precision_cells += RECALL_THRESHOLDS.size
    if precision_cells == 0:
        return float("nan")
    return total_precision / precision_cells


def reported_ap_small(report: dict[str, Any]) -> float | None:
    metrics = report.get("metrics")
    if not isinstance(metrics, dict):
        return None
    for key in ("AP_small", "APsmall"):
        value = metrics.get(key)
        if isinstance(value, (int, float)) and math.isfinite(float(value)) and float(value) >= 0:
            return float(value)
    return None


def one_bootstrap(replicate_index: int) -> float:
    if _BOOTSTRAP_STATE is None:
        raise RuntimeError("Bootstrap worker state is unavailable")
    state = _BOOTSTRAP_STATE
    rng = np.random.default_rng(np.random.SeedSequence([state["random_seed"], replicate_index]))
    sampled_images = rng.integers(0, state["image_count"], size=state["image_count"])
    sampled_seed_indices = rng.integers(0, len(state["pairs"]), size=len(state["pairs"]))
    counts = np.bincount(sampled_seed_indices, minlength=len(state["pairs"]))
    baseline_sum = 0.0
    proposed_sum = 0.0
    for pair_index in np.flatnonzero(counts):
        weight = int(counts[pair_index])
        pair = state["pairs"][int(pair_index)]
        baseline_sum += weight * ap_small(pair["baseline_grid"], sampled_images)
        proposed_sum += weight * ap_small(pair["proposed_grid"], sampled_images)
    divisor = len(state["pairs"])
    return proposed_sum / divisor - baseline_sum / divisor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Paired image-and-seed bootstrap CI for a COCO APsmall difference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--annotations", type=Path, required=True, help="COCO validation annotations")
    parser.add_argument(
        "--pair",
        nargs=3,
        action="append",
        metavar=("SEED", "BASELINE_REPORT_OR_PREDICTIONS", "PROPOSED_REPORT_OR_PREDICTIONS"),
        required=True,
        help="A matched training-seed pair; repeat for every seed",
    )
    parser.add_argument("--baseline-label", default="baseline")
    parser.add_argument("--proposed-label", default="LGF")
    parser.add_argument("--replicates", type=int, default=10000)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--random-seed", type=int, default=20260919)
    parser.add_argument(
        "--iou-backend",
        choices=("auto", "pycocotools", "numpy"),
        default="auto",
        help="BBox IoU implementation; auto falls back for the known NumPy conversion error",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Parallel processes; 0 chooses up to eight processes",
    )
    parser.add_argument("--output", type=Path, required=True, help="New output directory")
    return parser.parse_args()


def main() -> int:
    global _BOOTSTRAP_STATE
    args = parse_args()
    if args.replicates < 100:
        raise ValueError("Use at least 100 bootstrap replicates; 10000 is recommended for the paper")
    if not 0 < args.confidence < 1:
        raise ValueError("--confidence must be between 0 and 1")
    if args.workers < 0:
        raise ValueError("--workers cannot be negative")
    if args.output.exists():
        raise FileExistsError(f"Output already exists: {args.output}; choose a new directory")

    iou_backend = select_iou_backend(args.iou_backend)
    print(f"BBox IoU backend: {iou_backend}", flush=True)

    annotations_path = args.annotations.expanduser().resolve()
    annotation_document = read_json(annotations_path)
    for key in ("images", "annotations", "categories"):
        if not isinstance(annotation_document.get(key), list):
            raise ValueError(f"{annotations_path}: missing COCO {key!r} list")
    image_id_list = sorted(image["id"] for image in annotation_document["images"])
    category_id_list = sorted(category["id"] for category in annotation_document["categories"])
    if len(image_id_list) != len(set(image_id_list)) or not image_id_list:
        raise ValueError("Annotation image IDs must be unique and nonempty")
    if len(category_id_list) != len(set(category_id_list)) or not category_id_list:
        raise ValueError("Annotation category IDs must be unique and nonempty")
    annotation_ids = [annotation.get("id") for annotation in annotation_document["annotations"]]
    if any(value is None for value in annotation_ids) or len(annotation_ids) != len(set(annotation_ids)):
        raise ValueError("Annotation IDs must be present and unique")
    if any("area" not in annotation or "bbox" not in annotation for annotation in annotation_document["annotations"]):
        raise ValueError("Every ground-truth annotation must contain area and bbox")

    pair_arguments = args.pair
    seeds = [str(item[0]) for item in pair_arguments]
    if len(seeds) != len(set(seeds)):
        raise ValueError("Each --pair seed must be unique")
    image_ids = set(image_id_list)
    category_ids = set(category_id_list)
    prepared_pairs: list[dict[str, Any]] = []
    input_records: list[dict[str, Any]] = []
    observed_baseline: list[float] = []
    observed_proposed: list[float] = []
    selections: list[str] = []
    datasets: list[str] = []

    for seed, baseline_source, proposed_source in pair_arguments:
        print(f"Preparing seed {seed} ({args.baseline_label})", flush=True)
        baseline_predictions, baseline_report, baseline_path = load_prediction_source(
            Path(baseline_source), str(seed), image_ids, category_ids
        )
        baseline_grid, baseline_ap = prepare_coco_evaluation(
            annotation_document, baseline_predictions, iou_backend
        )
        print(f"Preparing seed {seed} ({args.proposed_label})", flush=True)
        proposed_predictions, proposed_report, proposed_path = load_prediction_source(
            Path(proposed_source), str(seed), image_ids, category_ids
        )
        proposed_grid, proposed_ap = prepare_coco_evaluation(
            annotation_document, proposed_predictions, iou_backend
        )

        full_indices = np.arange(len(image_id_list), dtype=int)
        custom_baseline_ap = ap_small(baseline_grid, full_indices)
        custom_proposed_ap = ap_small(proposed_grid, full_indices)
        if not math.isclose(custom_baseline_ap, baseline_ap, rel_tol=0, abs_tol=1e-12):
            raise RuntimeError("Internal APsmall accumulation differs from pycocotools for baseline")
        if not math.isclose(custom_proposed_ap, proposed_ap, rel_tol=0, abs_tol=1e-12):
            raise RuntimeError("Internal APsmall accumulation differs from pycocotools for proposed method")
        for label, computed, report in (
            (args.baseline_label, baseline_ap, baseline_report),
            (args.proposed_label, proposed_ap, proposed_report),
        ):
            recorded = reported_ap_small(report)
            if recorded is not None and not math.isclose(recorded, computed, rel_tol=0, abs_tol=1e-6):
                raise ValueError(
                    f"Seed {seed} {label}: saved APsmall={recorded:.9f}, but supplied annotations "
                    f"and predictions give {computed:.9f}"
                )
            selection = report.get("selection")
            if isinstance(selection, str) and selection:
                selections.append(selection)
            dataset = report.get("dataset")
            if isinstance(dataset, str) and dataset:
                datasets.append(dataset)

        prepared_pairs.append(
            {
                "seed": str(seed),
                "baseline_grid": baseline_grid,
                "proposed_grid": proposed_grid,
                "baseline_ap_small": baseline_ap,
                "proposed_ap_small": proposed_ap,
            }
        )
        observed_baseline.append(baseline_ap)
        observed_proposed.append(proposed_ap)
        input_records.append(
            {
                "seed": str(seed),
                "baseline_source": str(Path(baseline_source).expanduser().resolve()),
                "baseline_predictions": str(baseline_path),
                "baseline_predictions_sha256": sha256(baseline_path),
                "baseline_ap_small": baseline_ap,
                "proposed_source": str(Path(proposed_source).expanduser().resolve()),
                "proposed_predictions": str(proposed_path),
                "proposed_predictions_sha256": sha256(proposed_path),
                "proposed_ap_small": proposed_ap,
                "delta_ap_small": proposed_ap - baseline_ap,
            }
        )

    unique_selections = sorted(set(selections))
    if len(unique_selections) > 1:
        raise ValueError(f"Reports use different checkpoint-selection rules: {unique_selections}")
    unique_datasets = sorted(set(datasets))
    if len(unique_datasets) > 1:
        raise ValueError(f"Reports identify different datasets: {unique_datasets}")

    observed_baseline_mean = float(np.mean(observed_baseline))
    observed_proposed_mean = float(np.mean(observed_proposed))
    observed_delta = observed_proposed_mean - observed_baseline_mean
    per_seed_deltas = np.asarray(observed_proposed) - np.asarray(observed_baseline)
    baseline_seed_sd = (
        float(np.std(observed_baseline, ddof=1)) if len(observed_baseline) > 1 else None
    )
    proposed_seed_sd = (
        float(np.std(observed_proposed, ddof=1)) if len(observed_proposed) > 1 else None
    )
    delta_seed_sd = float(np.std(per_seed_deltas, ddof=1)) if len(per_seed_deltas) > 1 else None
    _BOOTSTRAP_STATE = {
        "pairs": prepared_pairs,
        "image_count": len(image_id_list),
        "random_seed": args.random_seed,
    }

    workers = args.workers or min(8, os.cpu_count() or 1)
    if workers > 1 and "fork" not in mp.get_all_start_methods():
        print("Fork-based multiprocessing is unavailable; using one worker", file=sys.stderr)
        workers = 1
    print(
        f"Bootstrap: {args.replicates} replicates, {len(prepared_pairs)} seed pairs, "
        f"{len(image_id_list)} images, {workers} worker(s)",
        flush=True,
    )
    if workers == 1:
        iterator = map(one_bootstrap, range(args.replicates))
        deltas = np.fromiter(iterator, dtype=float, count=args.replicates)
    else:
        context = mp.get_context("fork")
        chunksize = max(1, args.replicates // (workers * 20))
        with context.Pool(processes=workers) as pool:
            iterator = pool.imap(one_bootstrap, range(args.replicates), chunksize=chunksize)
            deltas = np.fromiter(iterator, dtype=float, count=args.replicates)
    if not np.all(np.isfinite(deltas)):
        raise RuntimeError("At least one bootstrap replicate produced a nonfinite APsmall change")

    alpha = 1.0 - args.confidence
    lower, upper = np.quantile(deltas, [alpha / 2.0, 1.0 - alpha / 2.0])
    probability_positive = float(np.mean(deltas > 0))
    probability_nonpositive = float(np.mean(deltas <= 0))
    ci_excludes_zero = bool(lower > 0 or upper < 0)
    warnings: list[str] = []
    if len(prepared_pairs) < 3:
        warnings.append("Fewer than three seed pairs; training-seed uncertainty is weakly characterized.")
    if not selections:
        warnings.append("No report supplied a checkpoint-selection label; verify identical selection rules manually.")
    if len(selections) != 2 * len(prepared_pairs):
        warnings.append("Some inputs lacked selection metadata; verify all checkpoints used one selection rule.")

    args.output.mkdir(parents=True)
    summary = {
        "script_version": VERSION,
        "bbox_iou_backend_requested": args.iou_backend,
        "bbox_iou_backend": iou_backend,
        "runtime": {
            "python": sys.version.split()[0],
            "python_executable": sys.executable,
            "numpy": np.__version__,
            "pycocotools": installed_version("pycocotools"),
        },
        "method": "paired hierarchical percentile bootstrap over validation images and training-seed pairs",
        "metric": "COCO APsmall@[IoU=0.50:0.95|maxDets=100]",
        "evaluation_scope": "validation predictions; this is not an independent held-out test estimate",
        "baseline_label": args.baseline_label,
        "proposed_label": args.proposed_label,
        "dataset": unique_datasets[0] if len(unique_datasets) == 1 else None,
        "selection": unique_selections[0] if len(unique_selections) == 1 else None,
        "annotation_file": str(annotations_path),
        "annotation_sha256": sha256(annotations_path),
        "image_count": len(image_id_list),
        "category_count": len(category_id_list),
        "seeds": seeds,
        "seed_pair_count": len(prepared_pairs),
        "replicates": args.replicates,
        "random_seed": args.random_seed,
        "confidence_level": args.confidence,
        "observed": {
            "baseline_mean_ap_small": observed_baseline_mean,
            "baseline_ap_small_sample_sd_across_seeds": baseline_seed_sd,
            "proposed_mean_ap_small": observed_proposed_mean,
            "proposed_ap_small_sample_sd_across_seeds": proposed_seed_sd,
            "delta_ap_small": observed_delta,
            "delta_ap_small_points": 100.0 * observed_delta,
            "delta_ap_small_sample_sd_across_seed_pairs": delta_seed_sd,
            "delta_ap_small_points_sample_sd_across_seed_pairs": (
                100.0 * delta_seed_sd if delta_seed_sd is not None else None
            ),
        },
        "bootstrap": {
            "delta_ap_small_percentile_ci": [float(lower), float(upper)],
            "delta_ap_small_points_percentile_ci": [100.0 * float(lower), 100.0 * float(upper)],
            "ci_excludes_zero": ci_excludes_zero,
            "fraction_delta_gt_zero": probability_positive,
            "fraction_delta_le_zero": probability_nonpositive,
            "note": "Bootstrap sign fractions are descriptive and are not labelled as a p-value.",
        },
        "per_seed": input_records,
        "warnings": warnings,
    }
    write_json(args.output / "paired_bootstrap_summary.json", summary)
    with (args.output / "paired_bootstrap_replicates.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["replicate", "delta_ap_small", "delta_ap_small_points"])
        writer.writerows((index, value, 100.0 * value) for index, value in enumerate(deltas))
    with (args.output / "paired_bootstrap_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "baseline_label",
                "proposed_label",
                "n_images",
                "n_seed_pairs",
                "replicates",
                "baseline_mean_ap_small",
                "baseline_seed_sd_ap_small",
                "proposed_mean_ap_small",
                "proposed_seed_sd_ap_small",
                "delta_ap_small",
                "delta_ap_small_points",
                "delta_seed_sd_ap_small",
                "delta_seed_sd_points",
                "ci_level",
                "ci_lower_ap_small",
                "ci_upper_ap_small",
                "ci_lower_points",
                "ci_upper_points",
                "ci_excludes_zero",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "baseline_label": args.baseline_label,
                "proposed_label": args.proposed_label,
                "n_images": len(image_id_list),
                "n_seed_pairs": len(prepared_pairs),
                "replicates": args.replicates,
                "baseline_mean_ap_small": observed_baseline_mean,
                "baseline_seed_sd_ap_small": baseline_seed_sd,
                "proposed_mean_ap_small": observed_proposed_mean,
                "proposed_seed_sd_ap_small": proposed_seed_sd,
                "delta_ap_small": observed_delta,
                "delta_ap_small_points": 100.0 * observed_delta,
                "delta_seed_sd_ap_small": delta_seed_sd,
                "delta_seed_sd_points": 100.0 * delta_seed_sd if delta_seed_sd is not None else None,
                "ci_level": args.confidence,
                "ci_lower_ap_small": float(lower),
                "ci_upper_ap_small": float(upper),
                "ci_lower_points": 100.0 * float(lower),
                "ci_upper_points": 100.0 * float(upper),
                "ci_excludes_zero": ci_excludes_zero,
            }
        )

    seed_sd_text = ""
    if baseline_seed_sd is not None and proposed_seed_sd is not None:
        seed_sd_text = (
            f" (sample SD: {args.baseline_label}={baseline_seed_sd:.6f}, "
            f"{args.proposed_label}={proposed_seed_sd:.6f})"
        )
    print(
        f"Observed APsmall: {args.baseline_label}={observed_baseline_mean:.6f}, "
        f"{args.proposed_label}={observed_proposed_mean:.6f}{seed_sd_text}",
        flush=True,
    )
    print(
        f"Paired change: {observed_delta:+.6f} ({100.0 * observed_delta:+.3f} AP points)",
        flush=True,
    )
    print(
        f"{100.0 * args.confidence:.1f}% percentile CI: [{lower:+.6f}, {upper:+.6f}] "
        f"AP = [{100.0 * lower:+.3f}, {100.0 * upper:+.3f}] points",
        flush=True,
    )
    print(f"CI excludes zero: {ci_excludes_zero}", flush=True)
    print(f"Wrote {args.output.resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, FileExistsError, ValueError, RuntimeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)
