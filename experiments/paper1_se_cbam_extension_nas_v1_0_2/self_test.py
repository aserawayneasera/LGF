#!/usr/bin/env python3
"""Software checks only, not research accuracy measurements.

Default: real installed dependencies, CPU, no pretrained download, no dataset access.
--offline-import-stubs is reserved for an offline test environment missing dataset
libraries. Such stubs raise when used and never enter the delivered training path.
The loop test deliberately uses a toy detector/loader/evaluator to test transaction
and resume behavior. Real-model tests separately exercise RetinaNet and its blocks.
"""
from __future__ import annotations
import argparse
import copy
import gc
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest import mock

import run_se_cbam as extension
import train_se_cbam_grid as grid

OPTIONS = None
REPORT = {"scientific_result": False, "real_dataset_training_tested": False,
          "COCO_metric_scoring_tested": False, "offline_import_stubs": [], "checks": []}


def missing_import_stubs():
    def unavailable(*args, **kwargs):
        raise RuntimeError("Dataset/logging dependency is unavailable in this offline test harness.")
    targets = [("albumentations", "ToTensorV2"), ("pycocotools", "COCO"), ("tensorboard", "SummaryWriter")]
    for package, name in targets:
        if importlib.util.find_spec(package) is not None:
            continue
        REPORT["offline_import_stubs"].append(package)
        if package == "albumentations":
            module = ModuleType(package); module.__path__ = []
            sub = ModuleType(package + ".pytorch"); sub.ToTensorV2 = unavailable
            sys.modules[package] = module; sys.modules[sub.__name__] = sub
        elif package == "pycocotools":
            module = ModuleType(package); module.__path__ = []; sys.modules[package] = module
            for suffix, attr in (("coco", "COCO"), ("cocoeval", "COCOeval")):
                sub = ModuleType(package + "." + suffix); setattr(sub, attr, unavailable)
                sys.modules[sub.__name__] = sub
        else:
            sub = ModuleType("torch.utils.tensorboard"); sub.SummaryWriter = unavailable
            sys.modules[sub.__name__] = sub


def setup_runtime():
    import torch
    torch.set_num_threads(4)
    package = extension.resolve_package(OPTIONS.package)
    before = grid.original_hashes(package)
    if OPTIONS.offline_import_stubs:
        missing_import_stubs()
    runner, bridge = extension.load_runner(package)
    core = bridge.load_core()
    core.args.img_size = 64  # Architecture probe only, within this test process.
    from ugs import UGSConfig
    return package, before, runner, bridge, core, UGSConfig()


class Checks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.package, cls.before, cls.runner, cls.bridge, cls.core, cls.config = setup_runtime()

    def tearDown(self):
        gc.collect()

    def test_01_training_functions_are_original(self):
        for name in ("run_train", "phase_optimizer", "training_step", "step_optimizer", "run_evaluate", "smoke"):
            function = getattr(self.runner, name)
            self.assertEqual(Path(function.__code__.co_filename).resolve(), self.package / "run.py")
        self.assertIn("se", self.runner.METHODS)
        self.assertIn("cbam", self.runner.METHODS)
        self.assertEqual(grid.original_hashes(self.package), self.before)

    def test_02_cli_and_labels(self):
        for method in extension.METHODS:
            with mock.patch.object(sys, "argv", ["run.py", "train", "--method", method]):
                args = self.runner.arguments()
            self.assertEqual(args.method, method)
            self.assertIn(method.upper(), self.runner.method_label(method, self.config))
            self.assertEqual(args.epochs, 80)
            self.assertEqual(args.batch_size * args.accum_steps, 16)

    def test_03_wrong_block_rejected(self):
        with self.assertRaises(ValueError):
            self.bridge.build_model("se", 6, pretrained=False, smoke_size=64,
                                    original_config=self.core.CONFIG_MAP["baseline"])

    def test_04_checkpoint_metadata_and_rejection(self):
        import torch
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "last.pth"
            value = {"method": "se", "config": copy.deepcopy(self.core.CONFIG_MAP["se"]),
                     "model_state_dict": {"example": torch.ones(2)}}
            self.runner.save_checkpoint(path, value)
            ck = torch.load(path, weights_only=False)
            self.assertEqual(ck["se_cbam_extension"]["sources"], extension.extension_hashes())
            self.assertNotIn("se_cbam_extension", value)  # No in-place mutation.
            with self.assertRaises(ValueError):
                self.runner.save_checkpoint(path, {"method": "cbam", "config": {"block_type": "none"}})

    def reference(self):
        import torch
        args = SimpleNamespace(dataset="coco_nw", method="se", seed=42, level="C3", epochs=80,
            batch_size=4, accum_steps=4, num_workers=8, prefetch_factor=4, base_lr=0.005, amp=False)
        ref = {"version": self.runner.VERSION, "method": "lgf_original", "dataset": "coco_nw",
               "seed": 42, "level_label_preserved": "C3", **extension.FIXED_PROTOCOL,
               **{k: getattr(args, k) for k in extension.REFERENCE_FIELDS},
               "environment": self.runner.environment(), "source_hashes": self.before,
               "model_protocol": {}, "training_content_sha256": "train-fixture", "validation_content_sha256": "val-fixture"}
        ref["protocol_sha256"] = self.runner.signature(ref)
        return args, ref, torch.device("cpu")

    def test_05_reference_matching_and_mismatch(self):
        args, ref, device = self.reference()
        extension.validate_reference_settings(args, ref, self.runner, device)
        for key, value in (("epochs", 79), ("batch_size", 8), ("seed", 2025), ("level", "C4")):
            wrong = copy.copy(args); setattr(wrong, key, value)
            with self.assertRaises(ValueError):
                extension.validate_reference_settings(wrong, ref, self.runner, device)
        for key in ("source_hashes", "environment"):
            wrong = copy.deepcopy(ref); wrong[key] = {}
            with self.assertRaises(ValueError):
                extension.validate_reference_settings(args, wrong, self.runner, device)

    def test_06_reference_data_mismatch_rejected(self):
        args, ref, device = self.reference()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "protocol.json"; path.write_text(json.dumps(ref))
            original_audit = self.runner.audit
            original_model_protocol = self.runner.model_protocol
            self.runner.audit = lambda *a, **k: {"train": {"content_sha256": "WRONG"}, "val": {"content_sha256": "val-fixture"}}
            try:
                extension.bind_reference(args, path, self.runner, device)
                with self.assertRaises(ValueError):
                    self.runner.audit("coco_nw", tmp)
            finally:
                self.runner.audit = original_audit
                self.runner.model_protocol = original_model_protocol

    def test_07_grid_inherits_settings_and_resume(self):
        _, ref, _ = self.reference()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = SimpleNamespace(reference_root=root / "ref", reference_method="lgf_original",
                run_root=root / "runs", data_root=root / "data", log_root=root / "logs", device="cpu", threads=4)
            rp = args.reference_root / "coco_nw/lgf_original/seed_42/protocol.json"
            rp.parent.mkdir(parents=True); rp.write_text(json.dumps(ref))
            status, command, _, _ = grid.make_job(args, self.package, "se", "coco_nw", 42, self.before)
            self.assertEqual(status, "TRAIN")
            for flag, value in (("--epochs", "80"), ("--accum-steps", "4"), ("--base-lr", "0.005")):
                self.assertEqual(command[command.index(flag) + 1], value)
            self.assertIn("--no-amp", command)
            dest = args.run_root / "coco_nw/se/seed_42"; dest.mkdir(parents=True)
            prior = {**ref, "method": "se", "source_hashes": {**self.before, **extension.extension_hashes()}}
            (dest / "protocol.json").write_text(json.dumps(prior)); (dest / "last.pth").touch()
            status, command, _, _ = grid.make_job(args, self.package, "se", "coco_nw", 42, self.before)
            self.assertEqual(status, "RESUME"); self.assertIn("--resume", command)

    def test_08_legacy_builder_passthrough(self):
        import torch
        for method in ("baseline", "lgf_original", "ugs_reimpl"):
            with self.subTest(method=method):
                torch.manual_seed(42)
                expected = self.bridge._se_cbam_native_build(method, 6, "C3", self.config, pretrained=False, smoke_size=64)
                expected_state = {k: v.detach().clone() for k, v in expected.state_dict().items()}
                del expected; gc.collect()
                torch.manual_seed(42)
                actual = self.bridge.build_model(method, 6, "C3", self.config, pretrained=False, smoke_size=64)
                self.assertEqual(set(actual.state_dict()), set(expected_state))
                self.assertTrue(all(torch.equal(v, expected_state[k]) for k, v in actual.state_dict().items()))
                del actual, expected_state; gc.collect()

    def test_09_original_attention_state_equality(self):
        import torch
        for method in extension.METHODS:
            with self.subTest(method=method):
                torch.manual_seed(42)
                expected = self.bridge._se_cbam_native_build(method, 6, "C3", self.config, pretrained=False,
                    smoke_size=64, original_config=copy.deepcopy(self.core.CONFIG_MAP[method]))
                expected_state = {k: v.detach().clone() for k, v in expected.state_dict().items()}
                del expected; gc.collect()
                torch.manual_seed(42)
                actual = self.bridge.build_model(method, 6, "C3", self.config, pretrained=False, smoke_size=64)
                self.assertTrue(all(torch.equal(v, expected_state[k]) for k, v in actual.state_dict().items()))
                block = next(iter(actual.backbone.block_fpn_in.values()))
                REPORT.setdefault("attention_blocks", {})[method] = {
                    "class": type(block).__name__, "parameters": sum(p.numel() for p in block.parameters()),
                    "insertion_keys": list(actual.backbone.block_fpn_in)}
                if method == "cbam":
                    self.assertEqual(block.convs.kernel_size, (5, 5)); self.assertEqual(block.beta, 20.0)
                del actual, expected_state; gc.collect()

    def test_10_real_detector_gradients_both_phases(self):
        import torch
        device = torch.device(OPTIONS.device)
        for method in extension.METHODS:
            with self.subTest(method=method):
                self.core.set_seed(42)
                model = self.bridge.build_model(method, 6, "C3", self.config, pretrained=False, smoke_size=64).to(device)
                images = [torch.rand(3, 64, 64), torch.rand(3, 64, 64)]
                targets = [{"boxes": torch.tensor([[10., 10., 26., 28.]]), "labels": torch.tensor([i])} for i in range(2)]
                scaler = torch.amp.GradScaler(device.type, enabled=False)
                for phase in ("heads", "all"):
                    optimizer, _ = self.runner.phase_optimizer(model, phase, 1e-4)
                    self.assertTrue(all(p.requires_grad for p in model.backbone.block_fpn_in.parameters()))
                    self.assertEqual(any(p.requires_grad for p in model.backbone.body.parameters()), phase == "all")
                    model.train()
                    losses, total = self.runner.training_step(model, images, targets, device, False, 1)
                    total.backward()
                    grads = [p.grad for p in model.backbone.block_fpn_in.parameters()]
                    self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in grads))
                    self.assertGreater(sum(float(g.abs().sum()) for g in grads), 0)
                    self.runner.step_optimizer(model, optimizer, scaler, False)
                    REPORT.setdefault("real_losses", []).append({"method": method, "phase": phase,
                        "losses": {k: float(v.detach()) for k, v in losses.items()}})
                del model, losses, total, optimizer, grads; gc.collect()

    def test_11_shared_test_adapter_strict_reload(self):
        if not OPTIONS.eval_dir:
            self.skipTest("Pass --eval-dir to check the existing shared exporter adapter.")
        import torch
        from evaluate_test_se_cbam import load_exporter
        exporter = load_exporter(self.package, Path(OPTIONS.eval_dir).resolve())
        for method in extension.METHODS:
            with self.subTest(method=method), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                self.core.set_seed(42)
                model = self.bridge.build_model(method, 6, "C3", self.config, pretrained=False, smoke_size=64).eval()
                with torch.no_grad():
                    model.head.classification_head.cls_logits.bias.fill_(2.)
                images = [torch.rand(3, 64, 40)]
                with torch.no_grad(): expected = model(images)
                ck = {"package_version": "1.0.0", "method": method, "dataset": "coco_nw", "seed": 42,
                      "level": "C3", "completed_epochs": 5, "ema_state_dict": model.state_dict(),
                      "config": copy.deepcopy(self.core.CONFIG_MAP[method]), "ugs_config": self.config.__dict__}
                path = root / "best_map.pth"; self.runner.save_checkpoint(path, ck)
                del ck, model; gc.collect()
                args = SimpleNamespace(method=method, train_dataset="coco_nw", seed=42, level="C3", device="cpu",
                    threads=4, batch_size=1, lgf_package=self.package, recent_package=root, runtime_root=root,
                    checkpoint=path)
                restored, device, meta, count, infer = exporter.load_model(args, [])
                self.assertEqual((restored.transform.min_size, restored.transform.max_size), ((800,), 1333))
                self.assertEqual(meta["model"], method); self.assertEqual(meta["epoch"], 5)
                # Numeric equality at the test-only size; production transform was checked above.
                restored.transform.min_size = (64,); restored.transform.max_size = 64
                actual = infer(images)
                self.assertGreater(len(actual[0]["boxes"]), 0)
                for key in ("boxes", "scores", "labels"):
                    torch.testing.assert_close(actual[0][key], expected[0][key], rtol=0, atol=0)
                protocol = {"schema": "paper1-test-export-1.0", "method": method, "adapter_sources": {}}
                first = exporter.signature(protocol)
                self.assertIn("se_cbam_extension/run_se_cbam.py", protocol["adapter_sources"])
                self.assertEqual(first, exporter.signature(protocol))
                del restored, infer, actual, expected; gc.collect()

    def test_12_native_training_loop_resume_and_selection(self):
        """Toy data/model/scoring isolate transaction handling, not accuracy."""
        import torch
        from torch import nn
        runner = self.runner
        class Toy(nn.Module):
            def __init__(self):
                super().__init__(); self.backbone = nn.Linear(3, 3); self.head = nn.Linear(3, 1)
            def forward(self, images, targets):
                values = torch.stack(images)
                prediction = self.head(self.backbone(values))
                return {"classification": (prediction - 1).square().mean(), "bbox_regression": prediction.abs().mean() * .01}
        class Loader:
            dataset = SimpleNamespace(transforms=None)
            def __len__(self): return 2
            def __iter__(self):
                for _ in range(2): yield [torch.rand(3), torch.rand(3)], [{}, {}]
        class Interrupt(Exception): pass
        def loaders(record, batch, workers, prefetch, seed):
            self.core.set_seed(seed)
            return Loader(), Loader(), torch.Generator().manual_seed(seed)
        def build(method, *a, **kw):
            self.core.CURRENT_CONFIG = copy.deepcopy(self.core.CONFIG_MAP[method]); return Toy()
        record = {"dataset": "coco_nw", "train": {"image_count": 4, "content_sha256": "toy-train"},
                  "val": {"image_count": 4, "content_sha256": "toy-val"}, "paths": {"val_annotations": "toy"}}
        def evaluate(model, loader, annotations, device, output, audit, metadata):
            # Different best epochs exercise both selection branches.
            metrics = {"mAP": metadata["epoch"] / 100, "AP_small": 1 - metadata["epoch"] / 100}
            return {**metadata, "metrics": metrics, "dataset": "coco_nw", "evaluation_signature": "toy"}
        device = torch.device("cpu")
        with tempfile.TemporaryDirectory() as tmp, mock.patch.multiple(runner,
                make_loaders=loaders, build_model=build, audit=lambda *a, **k: copy.deepcopy(record),
                model_protocol=lambda m: {"test_fixture": "toy"}, evaluate=evaluate):
            base = dict(dataset="coco_nw", method="se", seed=42, level="C3", data_root=tmp,
                epochs=6, batch_size=2, accum_steps=2, num_workers=0, prefetch_factor=4,
                base_lr=.005, amp=False, resume=None)
            args = SimpleNamespace(**base, run_root=str(Path(tmp) / "continuous"))
            runner.run_train(args, self.core, self.config, device)
            first = Path(args.run_root) / "coco_nw/se/seed_42"
            expected = torch.load(first / "last.pth", weights_only=False)
            args.run_root = str(Path(tmp) / "resumed")
            save = runner.save_checkpoint
            def stop_after_commit(path, state):
                save(path, state)
                if Path(path).name == "last.pth" and state["completed_epochs"] == 3:
                    raise Interrupt("Simulated interruption after a committed epoch")
            with mock.patch.object(runner, "save_checkpoint", stop_after_commit):
                with self.assertRaises(Interrupt): runner.run_train(args, self.core, self.config, device)
            second = Path(args.run_root) / "coco_nw/se/seed_42"
            args.resume = second / "last.pth"
            runner.run_train(args, self.core, self.config, device)
            actual = torch.load(second / "last.pth", weights_only=False)
            for name in ("model_state_dict", "ema_state_dict"):
                for key in expected[name]: torch.testing.assert_close(expected[name][key], actual[name][key], rtol=0, atol=0)
            self.assertEqual(actual["phase"], "all")
            self.assertEqual(torch.load(second / "best_map.pth", weights_only=False)["completed_epochs"], 6)
            self.assertEqual(torch.load(second / "best_ap_small.pth", weights_only=False)["completed_epochs"], 5)
            self.assertEqual(len((second / "history.jsonl").read_text().splitlines()), 6)

    def test_13_original_files_unchanged(self):
        self.assertEqual(grid.original_hashes(self.package), self.before)


class Results(unittest.TextTestResult):
    def addSuccess(self, test):
        super().addSuccess(test); REPORT["checks"].append({"name": test._testMethodName, "status": "passed"})
    def addFailure(self, test, err):
        super().addFailure(test, err); REPORT["checks"].append({"name": test._testMethodName, "status": "failed"})
    def addError(self, test, err):
        super().addError(test, err); REPORT["checks"].append({"name": str(test), "status": "error"})
    def addSkip(self, test, reason):
        super().addSkip(test, reason); REPORT["checks"].append({"name": test._testMethodName, "status": "skipped", "reason": reason})


def main():
    global OPTIONS
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--package", type=Path, required=True)
    p.add_argument("--eval-dir", type=Path)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--suite", choices=("unit", "models", "loop", "all"), default="all")
    p.add_argument("--offline-import-stubs", action="store_true")
    p.add_argument("--output", type=Path)
    OPTIONS = p.parse_args()
    numbers = {"unit": {1,2,3,4,5,6,7,13}, "models": {8,9,10,11,13}, "loop": {12,13}, "all": set(range(1,14))}[OPTIONS.suite]
    names = [name for name in unittest.defaultTestLoader.getTestCaseNames(Checks) if int(name.split("_")[1]) in numbers]
    suite = unittest.TestSuite(Checks(name) for name in names)
    result = unittest.TextTestRunner(verbosity=2, resultclass=Results).run(suite)
    REPORT.update(status="passed" if result.wasSuccessful() else "failed", tests_run=result.testsRun,
                  skipped=len(result.skipped), device=OPTIONS.device, suite=OPTIONS.suite)
    if OPTIONS.output:
        OPTIONS.output.parent.mkdir(parents=True, exist_ok=True)
        OPTIONS.output.write_text(json.dumps(REPORT, indent=2) + "\n")
    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    main()
