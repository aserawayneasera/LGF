"""Synthetic checks for the recovered bootstrap, never research measurements."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import paper1_paired_bootstrap as bootstrap


def toy_annotations():
    return {
        "images": [{"id": 1, "width": 128, "height": 128},
                   {"id": 2, "width": 128, "height": 128}],
        "categories": [{"id": 1, "name": "person"}],
        "annotations": [{"id": 1, "image_id": 1, "category_id": 1,
                         "bbox": [10, 10, 16, 16], "area": 256, "iscrowd": 0},
                        {"id": 2, "image_id": 2, "category_id": 1,
                         "bbox": [20, 20, 16, 16], "area": 256, "iscrowd": 0}],
    }


def toy_predictions():
    return [{"image_id": ann["image_id"], "category_id": ann["category_id"],
             "bbox": ann["bbox"], "score": .9}
            for ann in toy_annotations()["annotations"]]


class InputAndIoUTests(unittest.TestCase):
    def test_crowd_uses_detection_area_and_empty_shapes(self):
        observed = bootstrap.bbox_iou_numpy([[0, 0, 2, 2]],
                                            [[0, 0, 4, 4], [0, 0, 4, 4]], [0, 1])
        np.testing.assert_array_equal(observed, [[.25, 1.]])
        self.assertEqual(bootstrap.bbox_iou_numpy([], [], []).shape, (0, 0))
        self.assertEqual(bootstrap.bbox_iou_numpy([], [[0, 0, 1, 1]], [0]).shape, (0, 1))

    def test_iou_rejects_negative_nonfinite_and_wrong_crowd_length(self):
        for box in [[0, 0, -1, 2], [0, 0, 2, float("nan")]]:
            with self.assertRaises(ValueError):
                bootstrap.bbox_iou_numpy([box], [[0, 0, 1, 1]], [0])
        with self.assertRaises(ValueError):
            bootstrap.bbox_iou_numpy([], [[0, 0, 1, 1]], [])

    def test_source_verifies_hash_and_seed_after_relocation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            predictions = root / "predictions.json"
            predictions.write_text(json.dumps(toy_predictions()))
            report = root / "best_map.metrics.json"
            report.write_text(json.dumps({"seed": 42,
                "prediction_file": "/missing/old/runs/predictions.json",
                "prediction_sha256": bootstrap.sha256(predictions)}))
            loaded, metadata, actual_path = bootstrap.load_prediction_source(
                report, "42", {1, 2}, {1})
            self.assertEqual(loaded, toy_predictions())
            self.assertEqual(actual_path, predictions)
            self.assertEqual(metadata["seed"], 42)
            with self.assertRaises(ValueError):
                bootstrap.load_prediction_source(report, "1337", {1, 2}, {1})
            predictions.write_text("[]")
            with self.assertRaises(FileNotFoundError):
                bootstrap.load_prediction_source(report, "42", {1, 2}, {1})

    def test_predictions_reject_unmatched_image_and_category(self):
        prediction = toy_predictions()[0]
        for changed in [dict(prediction, image_id=3), dict(prediction, category_id=2),
                        dict(prediction, bbox=[0, 0, -1, 1]),
                        dict(prediction, score=float("inf"))]:
            with self.assertRaises(ValueError):
                bootstrap.validate_predictions([changed], {1, 2}, {1}, Path("fixture.json"))

    def test_bootstrap_uses_same_image_draw_for_every_pair(self):
        baseline_a, proposed_a, baseline_b, proposed_b = object(), object(), object(), object()
        pairs = [{"baseline_grid": baseline_a, "proposed_grid": proposed_a},
                 {"baseline_grid": baseline_b, "proposed_grid": proposed_b}]
        calls = []
        def fake_ap(grid, images):
            calls.append(images.copy())
            return .2 if grid is baseline_a or grid is baseline_b else .3
        previous = bootstrap._BOOTSTRAP_STATE
        try:
            bootstrap._BOOTSTRAP_STATE = {"pairs": pairs, "image_count": 6, "random_seed": 123}
            with patch.object(bootstrap, "ap_small", side_effect=fake_ap):
                first = bootstrap.one_bootstrap(7)
                first_calls = list(calls)
                calls.clear()
                second = bootstrap.one_bootstrap(7)
            self.assertAlmostEqual(first, .1)
            self.assertEqual(first, second)
            for draw in first_calls + calls:
                np.testing.assert_array_equal(draw, first_calls[0])
        finally:
            bootstrap._BOOTSTRAP_STATE = previous


@unittest.skipUnless(importlib.util.find_spec("pycocotools"), "pycocotools is required")
class CocoAccumulationTests(unittest.TestCase):
    def test_accumulation_agrees_with_official_evaluator(self):
        predictions = toy_predictions()[:1]
        for backend in ("pycocotools", "numpy"):
            grid, official = bootstrap.prepare_coco_evaluation(toy_annotations(), predictions, backend)
            self.assertAlmostEqual(bootstrap.ap_small(grid, np.arange(2)), official, places=12)

    def test_duplicate_image_draw_keeps_duplicate_gt_and_detections(self):
        grid, _ = bootstrap.prepare_coco_evaluation(toy_annotations(), toy_predictions()[:1], "numpy")
        self.assertAlmostEqual(bootstrap.ap_small(grid, np.array([0, 0])), 1.)
        self.assertEqual(bootstrap.ap_small(grid, np.array([1, 1])), 0.)

    def test_identical_models_have_zero_delta_on_all_draws(self):
        grid, _ = bootstrap.prepare_coco_evaluation(toy_annotations(), toy_predictions()[:1], "numpy")
        previous = bootstrap._BOOTSTRAP_STATE
        try:
            bootstrap._BOOTSTRAP_STATE = {"pairs": [
                {"baseline_grid": grid, "proposed_grid": grid} for _ in range(3)],
                "image_count": 2, "random_seed": 123}
            self.assertTrue(all(bootstrap.one_bootstrap(i) == 0. for i in range(100)))
        finally:
            bootstrap._BOOTSTRAP_STATE = previous


if __name__ == "__main__":
    unittest.main()
