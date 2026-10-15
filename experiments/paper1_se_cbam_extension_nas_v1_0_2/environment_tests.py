"""Regression checks for the reference environment comparison (no GPU required)."""
import copy
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import run_se_cbam as extension
import train_se_cbam_grid as grid


class EnvironmentChecks(unittest.TestCase):
    def setUp(self):
        self.recorded = {
            "python": "3.11.9", "packages": {"torch": "2.10.0", "torchvision": "0.25.0",
                "albumentations": "2.0.0", "opencv-python-headless": None},
            "cuda_available": True, "torch_cuda": "12.8", "gpu": "GPU A"}

    def test_identical_runtime_and_changed_gpu(self):
        actual = {**self.recorded, "gpu": "GPU B"}
        self.assertEqual(extension.environment_differences(self.recorded, self.recorded), {})
        self.assertEqual(extension.environment_differences(self.recorded, actual), {})
        self.assertEqual(extension.gpu_difference(self.recorded, actual),
                         {"reference": "GPU A", "current": "GPU B"})

    def test_package_python_and_cuda_changes_are_detailed_and_blocking(self):
        actual = copy.deepcopy(self.recorded)
        actual["python"] = "3.11.10"
        actual["packages"]["torch"] = "2.10.1"
        actual["torch_cuda"] = "12.9"
        differences = extension.environment_differences(self.recorded, actual)
        self.assertEqual(set(differences), {"python", "packages.torch", "torch_cuda"})
        self.assertEqual(differences["packages.torch"],
                         {"reference": "2.10.0", "current": "2.10.1"})

    def test_missing_package_field_is_reported(self):
        actual = copy.deepcopy(self.recorded)
        del actual["packages"]["opencv-python-headless"]
        self.assertEqual(extension.environment_differences(self.recorded, actual),
                         {"packages.opencv-python-headless":
                          {"reference": None, "current": "<missing>"}})

    def test_reference_validation_accepts_gpu_name_only(self):
        actual = {**self.recorded, "gpu": "GPU B"}
        runner = SimpleNamespace(VERSION="1.0.0", environment=lambda: actual,
                                 _se_cbam_original_source_hashes=lambda: {"run.py": "ok"})
        args = SimpleNamespace(dataset="coco_nw", seed=42, level="C3", epochs=80,
                               batch_size=4, accum_steps=4, num_workers=8,
                               prefetch_factor=4, base_lr=0.005, amp=False)
        reference = {"version": "1.0.0", "method": "lgf_original", "dataset": "coco_nw",
                     "seed": 42, "level_label_preserved": "C3", **extension.FIXED_PROTOCOL,
                     **{key: getattr(args, key) for key in extension.REFERENCE_FIELDS},
                     "environment": self.recorded, "source_hashes": {"run.py": "ok"}}
        with mock.patch("builtins.print"):
            extension.validate_reference_settings(args, reference, runner, SimpleNamespace(type="cuda"))
        actual["packages"] = {**actual["packages"], "torch": "2.10.1"}
        with self.assertRaisesRegex(ValueError, "packages.torch"):
            extension.validate_reference_settings(args, reference, runner, SimpleNamespace(type="cuda"))

    def test_resume_grid_accepts_gpu_name_only(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            args = SimpleNamespace(run_root=root, data_root=root / "data", log_root=root / "logs",
                                   device="cuda", threads=4)
            reference = {"version": "1.0.0", "method": "lgf_original", "dataset": "coco_nw",
                         "seed": 42, "level_label_preserved": "C3", **extension.FIXED_PROTOCOL,
                         **dict.fromkeys(extension.REFERENCE_FIELDS, 4), "environment": self.recorded,
                         "model_protocol": {}, "training_content_sha256": "t", "validation_content_sha256": "v",
                         "source_hashes": {}}
            run_dir = root / "coco_nw" / "se" / "seed_42"
            run_dir.mkdir(parents=True)
            prior = {**reference, "method": "se", "environment": {**self.recorded, "gpu": "GPU B"},
                     "source_hashes": extension.extension_hashes()}
            (run_dir / "protocol.json").write_text(json.dumps(prior))
            (run_dir / "last.pth").write_bytes(b"test-only-placeholder")
            with mock.patch.object(grid, "find_reference", return_value=(Path("/fixture/ref"), reference)):
                status, command, _, _ = grid.make_job(args, Path("/fixture/package"), "se", "coco_nw", 42, {})
                self.assertEqual(status, "RESUME")
                self.assertIn("--resume", command)
                prior["environment"]["packages"] = {**self.recorded["packages"], "torch": "2.10.1"}
                (run_dir / "protocol.json").write_text(json.dumps(prior))
                with self.assertRaisesRegex(ValueError, "packages.torch"):
                    grid.make_job(args, Path("/fixture/package"), "se", "coco_nw", 42, {})

    def test_grid_preflight_checks_runtime_before_any_training(self):
        with tempfile.TemporaryDirectory() as temp:
            reference = Path(temp) / "protocol.json"
            reference.write_text(json.dumps({"environment": self.recorded}))
            job = ("SKIP COMPLETED", ["python", "unused.py"], Path(temp) / "log", reference)
            current = {**self.recorded, "gpu": "GPU B"}
            runner = SimpleNamespace(environment=lambda: current)
            argv = ["grid.py", "--methods", "se", "--datasets", "coco_nw", "--seeds", "42"]
            with mock.patch.object(sys, "argv", argv), \
                 mock.patch.object(grid, "resolve_package", return_value=Path(temp)), \
                 mock.patch.object(grid, "original_hashes", return_value={}), \
                 mock.patch.object(grid, "make_job", return_value=job), \
                 mock.patch.object(grid, "load_runner", return_value=(runner, None)), \
                 mock.patch.object(grid, "execute") as execute, \
                 contextlib.redirect_stdout(io.StringIO()) as output:
                grid.main()
                self.assertIn("GPU MODEL DIFFERS", output.getvalue())
                self.assertIn("ENVIRONMENT PREFLIGHT PASS", output.getvalue())
                execute.assert_not_called()
                current["python"] = "3.11.10"
                with self.assertRaisesRegex(ValueError, '"python"'):
                    grid.main()
                execute.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
