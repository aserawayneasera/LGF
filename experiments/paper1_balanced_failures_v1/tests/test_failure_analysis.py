"""Software checks on synthetic labels, never research results."""
import copy
import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("failure_analysis", ROOT / "paper1_failure_analysis.py")
fa = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fa)


def toy_doc():
    return {"images": [{"id": 1, "file_name": "one.png", "width": 320, "height": 240},
                       {"id": 2, "file_name": "empty.png", "width": 320, "height": 240}],
            "categories": [{"id": i + 1, "name": name} for i, name in enumerate(sorted(fa.CLASSES))],
            "annotations": [{"id": 0, "image_id": 1, "category_id": 1, "bbox": [10, 10, 16, 16], "area": 256, "iscrowd": 0},
                            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [80, 20, 40, 40], "area": 1600, "iscrowd": 0},
                            {"id": 2, "image_id": 1, "category_id": 1, "bbox": [140, 90, 100, 100], "area": 10000, "iscrowd": 0},
                            {"id": 3, "image_id": 1, "category_id": 1, "bbox": [260, 10, 40, 40], "area": 1600, "iscrowd": 1}]}


def prediction(box, score=.9, category=1, image=1):
    return {"image_id": image, "category_id": category, "bbox": box, "score": score}


def evaluate(doc, preds, threshold=.25, iou=.5, area="all"):
    ev, gt, dt = fa.coco_match(doc, preds, [iou], 100)
    return fa.operating_point(ev, gt, dt, threshold, 0, area)


class MatchingTests(unittest.TestCase):
    def test_zero_gt_id_duplicate_wrong_class_crowd_and_empty_image(self):
        doc = toy_doc()
        preds = [prediction([10, 10, 16, 16]), prediction([10, 10, 16, 16], .8),
                 prediction([80, 20, 40, 40], category=2), prediction([140, 90, 100, 100], .1),
                 prediction([260, 10, 20, 20]), prediction([260, 10, 20, 20], .8),
                 prediction([5, 5, 10, 10], image=2)]
        point = evaluate(doc, preds)
        row = fa.counts(point)
        self.assertEqual((row["ground_truth"], row["tp"], row["fp"], row["fn"], row["ignored_predictions"]), (3, 1, 3, 2, 2))
        self.assertEqual(fa.counts(point, image_id=2)["fp"], 1)
        self.assertEqual(point["gt"][1]["source_annotation_id"], 0)
        self.assertEqual(doc["annotations"][0]["id"], 0)

    def test_empty_detections_keep_all_fn(self):
        point = evaluate(toy_doc(), [])
        row = fa.counts(point)
        self.assertEqual((row["tp"], row["fp"], row["fn"]), (0, 0, 3))
        self.assertIsNone(row["precision"])
        self.assertEqual(row["recall"], 0)

    def test_exclusive_partition_endpoints_and_coco_overlap(self):
        doc = toy_doc()
        doc["annotations"] = [
            {"id": 0, "image_id": 1, "category_id": 1, "bbox": [0, 0, 32, 32], "area": 1024, "iscrowd": 0},
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [50, 50, 96, 96], "area": 9216, "iscrowd": 0}]
        preds = [prediction(a["bbox"]) for a in doc["annotations"]]
        point = evaluate(doc, preds)
        self.assertEqual([fa.counts(point, s)["tp"] for s in fa.SIZES], [2, 0, 1, 1])
        self.assertEqual(fa.counts(evaluate(doc, preds, area="small"))["tp"], 1)
        self.assertEqual(fa.counts(evaluate(doc, preds, area="medium"))["tp"], 2)

    def test_tp_size_uses_annotation_area_and_fp_uses_predicted_box(self):
        doc = toy_doc()
        doc["annotations"] = [doc["annotations"][1]]
        # Annotation area is a preserved mask area, different from bbox area.
        doc["annotations"][0]["area"] = 200
        point = evaluate(doc, [prediction([80, 20, 40, 40]), prediction([0, 0, 40, 40])])
        self.assertEqual(fa.counts(point, "small")["tp"], 1)
        self.assertEqual(fa.counts(point, "medium")["fp"], 1)

    def test_sweep_equals_fresh_coco_matching_at_each_threshold(self):
        doc = toy_doc()
        preds = [prediction([10, 10, 16, 16], .3), prediction([10, 10, 16, 16], .2),
                 prediction([80, 20, 40, 40], .7), prediction([140, 90, 90, 90], .9),
                 prediction([260, 10, 20, 20], .1), prediction([0, 150, 40, 40], .6)]
        ev, gt, dt = fa.coco_match(doc, preds, [.5, .75], 100)
        for ti, iou in enumerate([.5, .75]):
            for threshold in [.05, .25, .5]:
                filtered = [p for p in preds if p["score"] >= threshold]
                for area in fa.SIZES:
                    point = fa.operating_point(ev, gt, dt, threshold, ti, area)
                    reference = evaluate(doc, filtered, threshold, iou, area)
                    self.assertEqual(fa.counts(point), fa.counts(reference))

    def test_iou_changes_localization_outcome(self):
        doc = toy_doc()
        doc["annotations"] = [doc["annotations"][0]]
        pred = prediction([12, 12, 16, 16])
        self.assertEqual(fa.counts(evaluate(doc, [pred], iou=.5))["tp"], 1)
        self.assertEqual(fa.counts(evaluate(doc, [pred], iou=.75))["fn"], 1)

    def test_fp_pairing_is_same_class_and_one_to_one(self):
        a = {**prediction([0, 0, 16, 16]), "prediction_index": 0}
        b = {**a, "prediction_index": 1, "score": .8}
        c = {**a, "prediction_index": 2, "category_id": 2}
        pairs, base_only, lgf_only = fa.fp_pairs([a], [a, b, c], .5)
        self.assertEqual((len(pairs), len(base_only), len(lgf_only)), (1, 0, 2))

    def test_pair_transitions_cover_every_object(self):
        doc = toy_doc()
        doc["annotations"] = [a for a in doc["annotations"] if not a["iscrowd"]]
        extra = copy.deepcopy(doc["annotations"][0])
        extra.update(id=3, bbox=[250, 180, 16, 16])
        doc["annotations"].append(extra)
        base = evaluate(doc, [prediction(doc["annotations"][0]["bbox"]), prediction(doc["annotations"][1]["bbox"])])
        lgf = evaluate(doc, [prediction(doc["annotations"][0]["bbox"]), prediction(doc["annotations"][2]["bbox"]), prediction([260, 30, 16, 16])])
        args = SimpleNamespace(primary_confidence=.25, primary_iou=.5, fp_pair_iou=.5)
        objects, fps, candidates, rows = fa.paired_analysis("toy", 42, doc, base, lgf, args)
        self.assertEqual({r["event"] for r in objects}, {"shared_hit", "lgf_recovery", "retinanet_recovery", "shared_miss"})
        self.assertEqual(rows[0]["lgf_only_fp"], 1)
        self.assertEqual(len(objects), 4)

    def test_filename_export_maps_internal_ids_to_labelled_ids(self):
        doc = toy_doc()
        name = doc["categories"][0]["name"]
        rows = [{"image_id": 99, "file_name": "one.png", "detections": [{"category_name": name, "bbox": [0, 0, 10, 10], "score": .9}]},
                {"image_id": 100, "file_name": "empty.png", "detections": []}]
        self.assertEqual(fa.normalize_predictions(rows, doc)[0]["image_id"], 1)
        with self.assertRaises(ValueError):
            fa.normalize_predictions(rows[:1], doc)


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        from PIL import Image
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.images = self.root / "images"
        self.images.mkdir()
        self.doc = toy_doc()
        self.doc["annotations"] = [a for a in self.doc["annotations"] if not a["iscrowd"]]
        extra = copy.deepcopy(self.doc["annotations"][0])
        extra.update(id=3, bbox=[250, 180, 16, 16])
        self.doc["annotations"].append(extra)
        for image in self.doc["images"]:
            Image.new("RGB", (320, 240), "#455563").save(self.images / image["file_name"])
        self.annotations = self.root / "ground_truth.json"
        fa.write_json(self.annotations, self.doc)
        records = [{"image_id": im["id"], "file_name": im["file_name"], "sha256": fa.digest(self.images / im["file_name"])} for im in self.doc["images"]]
        content = fa.signature({"annotations": fa.digest(self.annotations), "images": records})
        self.audit = self.root / "audit.json"
        fa.write_json(self.audit, {"dataset": "toy", "paths": {"val_annotations": str(self.annotations), "val_images": str(self.images)},
                  "val": {"content_sha256": content, "annotations_sha256": fa.digest(self.annotations), "image_records": records}})
        entries = []
        for method in ["baseline", "lgf_original", "yolo11m"]:
            run = self.root / method
            run.mkdir()
            indices = [0, 1] if method == "baseline" else [0, 2]
            preds = [prediction(self.doc["annotations"][i]["bbox"]) for i in indices]
            if method != "baseline":
                preds.append(prediction([260, 30, 16, 16]))
            pred = run / "predictions.json"
            fa.write_json(pred, preds)
            report = run / "best_map.metrics.json"
            fa.write_json(report, {"dataset": "toy", "method": method, "seed": 42, "selection": "best_map", "epoch": 70,
                "prediction_file": str(pred), "prediction_sha256": fa.digest(pred), "evaluation_content_sha256": content,
                "evaluation_protocol": {"score_thresh": .05, "evaluation_split": "val"}, "metrics": {"detection_count": len(preds), "mAP": .2}})
            entries.append({"dataset": "toy", "method": method, "seed": 42, "report": str(report)})
        self.manifest = self.root / "inputs.json"
        fa.write_json(self.manifest, {"datasets": {"toy": {"audit": str(self.audit)}}, "runs": entries})

    def test_complete_output_and_all_model_figure(self):
        output = self.root / "analysis"
        code = fa.main(["--manifest", str(self.manifest), "--output", str(output), "--example-size", "any", "--render-all-models"])
        self.assertEqual(code, 0)
        with (output / "counts_per_seed.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 3 * 4 * 3 * 2)
        examples = fa.read_json(output / "selected_examples.json")
        self.assertEqual({r["event"] for r in examples}, set(fa.EVENTS))
        self.assertTrue(all(Path(r["figure"]).is_file() for r in examples))
        self.assertIsNone(fa.read_json(output / "analysis_manifest.json")["tn"])
        self.assertEqual(len((output / "detection_outcomes.jsonl").read_text().splitlines()), 8)

    def test_modified_prediction_is_rejected_before_output(self):
        path = self.root / "baseline" / "predictions.json"
        path.write_text("[]")
        output = self.root / "bad_output"
        self.assertEqual(fa.main(["--manifest", str(self.manifest), "--output", str(output)]), 1)
        self.assertFalse(output.exists())

    def test_below_export_floor_is_rejected(self):
        self.assertEqual(fa.main(["--manifest", str(self.manifest), "--confidences", "0.01", "--check"]), 1)

    def test_hidden_test_image_info_is_rejected(self):
        path = self.root / "image_info.json"
        doc = copy.deepcopy(self.doc)
        del doc["annotations"]
        fa.write_json(path, doc)
        with self.assertRaisesRegex(ValueError, "Ground-truth"):
            fa.document(path)

    def test_auto_ugs_and_recent_prediction_discovery(self):
        root = self.root / "ugs" / "toy" / "baseline" / "seed_42"
        root.mkdir(parents=True)
        report = fa.read_json(self.root / "baseline" / "best_map.metrics.json")
        report["prediction_file"] = "/tmp/asera/old/missing/predictions.json"
        epoch = root / "evaluation" / "epoch_070"
        epoch.mkdir(parents=True)
        (epoch / "predictions.json").write_bytes((self.root / "baseline" / "predictions.json").read_bytes())
        fa.write_json(root / "best_map.metrics.json", report)
        fa.write_json(root / "data_audit.json", fa.read_json(self.audit))
        recent = self.root / "recent" / "runs" / "toy" / "yolo11m" / "seed_42"
        recent.mkdir(parents=True)
        fa.write_json(recent / "best_map.metrics.json", fa.read_json(self.root / "yolo11m" / "best_map.metrics.json"))
        prepared = self.root / "recent" / "prepared" / "toy"
        prepared.mkdir(parents=True)
        fa.write_json(prepared / "conversion.json", {"source": fa.read_json(self.audit)})
        self.assertEqual(fa.main(["--ugs-root", str(self.root / "ugs"), "--recent-root", str(self.root / "recent" / "runs"),
                                  "--datasets", "toy", "--methods", "baseline", "yolo11m", "--seeds", "42", "--check"]), 0)

    def test_plural_archived_evaluations_require_selected_epoch_and_hash(self):
        recent_root = self.root / "recent" / "runs"
        run = recent_root / "toy" / "yolo11m" / "seed_42"
        run.mkdir(parents=True)
        report = fa.read_json(self.root / "yolo11m" / "best_map.metrics.json")
        report["epoch"] = 55
        report["prediction_file"] = "/tmp/asera/archive/shared_evaluation/epoch_055/predictions.json"
        prediction_bytes = (self.root / "yolo11m" / "predictions.json").read_bytes()
        selected = run / "shared_evaluations" / "epoch_055" / "predictions.json"
        selected.parent.mkdir(parents=True)
        selected.write_bytes(prediction_bytes)
        # Another epoch contains an identical export. It must not be used if the
        # exact selected epoch is missing or fails its saved prediction hash.
        other = run / "shared_evaluations" / "epoch_054" / "predictions.json"
        other.parent.mkdir(parents=True)
        other.write_bytes(prediction_bytes)
        fa.write_json(run / "best_map.metrics.json", report)
        prepared = self.root / "recent" / "prepared" / "toy"
        prepared.mkdir(parents=True)
        fa.write_json(prepared / "conversion.json", {"source": fa.read_json(self.audit)})
        argv = ["--recent-root", str(recent_root), "--datasets", "toy", "--methods", "yolo11m", "--seeds", "42", "--check"]
        runs = fa.load_inputs(fa.parser().parse_args(argv))
        self.assertEqual(runs[0]["prediction_path"], str(selected.resolve()))
        self.assertEqual(runs[0]["epoch"], 55)
        self.assertEqual(runs[0]["prediction_sha256"], fa.digest(selected))
        selected.write_text("[]")
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            fa.load_inputs(fa.parser().parse_args(argv))
        selected.unlink()
        with self.assertRaisesRegex(ValueError, "Missing file"):
            fa.load_inputs(fa.parser().parse_args(argv))


if __name__ == "__main__":
    unittest.main()
