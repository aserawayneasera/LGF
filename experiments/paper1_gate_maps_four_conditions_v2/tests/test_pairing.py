"""Synthetic paired-scene integrity checks without trained weights or torch."""
import copy
import csv
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import paper1_gate_visualization_conditions_v2 as maps


class PairingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.clear = self.root / "clear"
        self.weather = self.root / "weather"
        self.clear.mkdir()
        self.weather.mkdir()
        self.key = "000000000001"
        clear_image = {"id": 1, "file_name": f"{self.key}.jpg"}
        fog_image = {"id": 101, "file_name": f"{self.key}_fog.jpg", "weather": "fog"}
        annotation = {"id": 1, "image_id": 1, "category_id": 1,
                      "bbox": [5, 5, 10, 10], "area": 100, "iscrowd": 0}
        self.clear_document = {"images": [clear_image], "annotations": [annotation]}
        self.weather_document = {"images": [fog_image],
                                 "annotations": [dict(annotation, id=101, image_id=101)]}
        Image.new("RGB", (64, 64)).save(self.clear / clear_image["file_name"])
        rows = []
        for index, condition in enumerate(("fog", "rain", "snow"), start=1):
            filename = f"{self.key}_{condition}.jpg"
            Image.new("RGB", (64, 64)).save(self.weather / filename)
            rows.append({"image_id": 100 + index, "file_name": filename, "condition": condition})
        self.index = self.root / "index.csv"
        with self.index.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["image_id", "file_name", "condition"])
            writer.writeheader()
            writer.writerows(rows)
        self.path_map = {"clear_ann": self.root / "clear.json",
                         "weather_ann": self.root / "weather.json",
                         "clear_images": self.clear, "weather_images": self.weather}
        self.args = SimpleNamespace(weather_index=self.index, num_scenes=1, seed=42, scene_key=[])
        self.write_annotations()

    def write_annotations(self):
        self.path_map["clear_ann"].write_text(json.dumps(self.clear_document))
        self.path_map["weather_ann"].write_text(json.dumps(self.weather_document))

    def test_fog_only_annotations_inherit_identical_boxes_for_display(self):
        before = copy.deepcopy(self.weather_document)
        selected, records, targets, _, annotations, counts, shared, derived = maps.select_scenes(self.args, self.path_map)
        self.assertEqual(selected, [self.key])
        self.assertEqual(shared, 1)
        self.assertEqual(derived, {"fog": 0, "rain": 1, "snow": 1})
        self.assertEqual(counts, {name: 1 for name in maps.CONDITIONS})
        for condition in ("rain", "snow"):
            info = records[condition][self.key]
            self.assertEqual(info["_gt_source_image_id"], 101)
            self.assertEqual(targets[self.key][condition]["bbox"], [5, 5, 10, 10])
            self.assertEqual(annotations[info["id"]][0]["category_id"], 1)
        self.assertEqual(json.loads(self.path_map["weather_ann"].read_text()), before)

    def test_differing_image_dimensions_are_rejected(self):
        Image.new("RGB", (65, 64)).save(self.weather / f"{self.key}_snow.jpg")
        with self.assertRaisesRegex(ValueError, "differing image dimensions"):
            maps.select_scenes(self.args, self.path_map)

    def test_small_gt_mismatch_is_rejected(self):
        self.weather_document["annotations"][0]["bbox"] = [30, 30, 10, 10]
        self.write_annotations()
        with self.assertRaisesRegex(ValueError, "matching small-object GT"):
            maps.select_scenes(self.args, self.path_map)

    def test_conflicting_condition_label_is_rejected(self):
        self.weather_document["images"][0]["weather"] = "rain"
        self.write_annotations()
        with self.assertRaisesRegex(ValueError, "Conflicting weather label"):
            maps.select_scenes(self.args, self.path_map)

    def test_missing_sibling_image_prevents_four_condition_selection(self):
        (self.weather / f"{self.key}_snow.jpg").unlink()
        with self.assertRaisesRegex(ValueError, "No paired clear/fog/rain/snow scenes"):
            maps.select_scenes(self.args, self.path_map)


if __name__ == "__main__":
    unittest.main()
