#!/usr/bin/env python3
"""SE/CBAM extension for Wayne's existing UGS comparison runner.

Keep this directory OUTSIDE paper1_ugs_comparison_v1. No existing source is edited.
Examples:
  python run_se_cbam.py --package /path/paper1_ugs_comparison_v1 check
  python run_se_cbam.py --package /path/paper1_ugs_comparison_v1 smoke --method se --synthetic --device cpu
  python run_se_cbam.py --package /path/paper1_ugs_comparison_v1 train --method cbam --dataset coco_nw --seed 42 --device cuda

All native run.py arguments are forwarded. Optional --reference-protocol checks
recorded settings against an existing baseline/LGF protocol.json before training.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import sys
from typing import Any

VERSION = "1.0.2"
METHODS = ("se", "cbam")
HERE = Path(__file__).resolve().parent
PROJECT_ROOT = Path("/nas.dbms/asera")
DEFAULT_DATA_ROOT = PROJECT_ROOT / "paper1_original/DATASET"
DEFAULT_RUN_ROOT = PROJECT_ROOT / "paper1_original/ugs_comparison/runs"
DEFAULT_RUNTIME_ROOT = PROJECT_ROOT / "paper1_original/test_runtime"


def configure_storage():
    """Set cache storage before importing the unchanged native package."""
    old_root = "/tmp/asera"
    value = os.environ.get("TORCH_HOME", "")
    if not value:
        value = str(PROJECT_ROOT / "paper1_original/torch_cache")
    elif value == old_root or value.startswith(old_root + "/"):
        value = str(PROJECT_ROOT) + value[len(old_root):]
    os.environ["TORCH_HOME"] = value


def with_storage_defaults(argv, defaults):
    """Supply NAS defaults without modifying native parsers or explicit flags."""
    result = list(argv)
    for flag, value in defaults.items():
        if not any(arg == flag or arg.startswith(flag + "=") for arg in result):
            result.extend([flag, str(value)])
    return result

REFERENCE_FIELDS = (
    "epochs", "batch_size", "accum_steps", "num_workers", "prefetch_factor",
    "base_lr", "amp",
)
FIXED_PROTOCOL = {
    "optimizer": "SGD(momentum=0.9,weight_decay=1e-4)",
    "warmup_epochs": 2, "unfreeze_at_epoch_index": 5,
    "scheduler_milestones_after_reset": [40, 60],
    "scheduler_reset_at_unfreeze": True, "ema_decay": 0.9999, "grad_clip": 10.0,
    "pretraining": "RetinaNet_ResNet50_FPN_Weights.COCO_V1",
    "augmentation_seed_policy": "explicit seed per run/worker",
    "selection": "save both best validation APsmall and best validation mAP; evaluate every 5 epochs",
}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def environment_differences(reference: Any, current: Any) -> dict[str, Any]:
    """Compare the recorded runtime; a GPU model name is audit metadata."""
    if not isinstance(reference, dict) or not isinstance(current, dict):
        return {"environment": {"reference": reference, "current": current}}
    differences = {}
    missing = object()
    show = lambda value: "<missing>" if value is missing else value
    for key in sorted((reference.keys() | current.keys()) - {"gpu"}):
        before, after = reference.get(key, missing), current.get(key, missing)
        if key == "packages" and isinstance(before, dict) and isinstance(after, dict):
            for package in sorted(before.keys() | after.keys()):
                old_version, new_version = before.get(package, missing), after.get(package, missing)
                if old_version != new_version:
                    differences[f"packages.{package}"] = {
                        "reference": show(old_version), "current": show(new_version)}
        elif before != after:
            differences[key] = {"reference": show(before), "current": show(after)}
    return differences


def gpu_difference(reference: Any, current: Any) -> dict[str, Any] | None:
    if isinstance(reference, dict) and isinstance(current, dict) and reference.get("gpu") != current.get("gpu"):
        return {"reference": reference.get("gpu"), "current": current.get("gpu")}
    return None


def extension_hashes() -> dict[str, str]:
    return {"se_cbam_extension/run_se_cbam.py": digest(Path(__file__))}


def resolve_package(explicit: str | Path | None = None) -> Path:
    value = explicit or os.environ.get("P1_UGS_PACKAGE")
    roots = [Path(value).expanduser()] if value else [
        Path.cwd(),
        Path("/nas.dbms/asera/NEW/paper1_all_codes_with_test_evaluation"),
        Path("/nas.dbms/asera/NEW/paper1_recovery_1_0_1/paper1_all_codes_with_test_evaluation"),
        Path("/nas.dbms/asera/NEW"),
    ]
    required = ("run.py", "core_bridge.py", "data.py", "evaluation.py", "utils.py",
                "ugs.py", "configs/ugs_reimpl.json", "vendor/original_core.py", "vendor/source_identity.json")
    for root in roots:
        for candidate in (root, root / "paper1_ugs_comparison_v1"):
            if all((candidate / name).is_file() for name in required):
                package = candidate.resolve()
                if HERE == package or package in HERE.parents:
                    raise ValueError("Keep the extension OUTSIDE the original package. Its recursive source hashes protect existing resumes.")
                return package
    raise FileNotFoundError("UGS package not found. Pass --package /absolute/path/to/paper1_ugs_comparison_v1.")


def import_at(name: str, path: Path):
    path = path.resolve()
    existing = sys.modules.get(name)
    if existing is not None:
        if Path(getattr(existing, "__file__", "")).resolve() != path:
            raise RuntimeError(f"Python module {name!r} already came from another package. Start a fresh Python process.")
        return existing
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def install_bridge(bridge):
    """Select the archived block, rather than allowing the old baseline fallback."""
    if hasattr(bridge, "_se_cbam_native_build"):
        return bridge
    native = bridge.build_model
    if "original_config" not in inspect.signature(native).parameters:
        raise RuntimeError("Unsupported core_bridge.py: original_config is required.")

    def build_model(method, num_classes, level="C3", ugs_config=None, *,
                    pretrained=True, smoke_size=None, original_config=None):
        if method not in METHODS:
            return native(method, num_classes, level, ugs_config, pretrained=pretrained,
                          smoke_size=smoke_size, original_config=original_config)
        core = bridge.load_core()
        cfg = copy.deepcopy(original_config if original_config is not None else core.CONFIG_MAP[method])
        if cfg.get("block_type") != method:
            raise ValueError(f"Requested {method}, but original_config selects {cfg.get('block_type')!r}.")
        model = native(method, num_classes, level, ugs_config, pretrained=pretrained,
                       smoke_size=smoke_size, original_config=cfg)
        expected = core.SEBlock if method == "se" else core.CBAMBlock
        blocks = getattr(model.backbone, "block_fpn_in", {})
        if len(blocks) != 1 or not all(isinstance(block, expected) for block in blocks.values()):
            raise RuntimeError(f"{method}: expected exactly one archived {expected.__name__} at the existing insertion point.")
        if core.CURRENT_CONFIG.get("block_type") != method:
            raise RuntimeError("Model construction lost the selected attention configuration.")
        return model

    bridge._se_cbam_native_build = native
    bridge.build_model = build_model
    return bridge


def load_bridge(package: Path):
    configure_storage()
    package = package.resolve()
    sys.path.insert(0, str(package))
    return install_bridge(import_at("core_bridge", package / "core_bridge.py"))


def load_runner(package: Path):
    bridge = load_bridge(package)
    runner = import_at("paper1_se_cbam_native_runner", package / "run.py")
    if getattr(runner, "_se_cbam_installed", False):
        return runner, bridge
    if runner.VERSION != "1.0.0":
        raise RuntimeError(f"Unsupported UGS runner version {runner.VERSION!r}. Expected the delivered 1.0.0 runner.")
    for name in ("arguments", "configure", "phase_optimizer", "training_step", "run_train",
                 "run_evaluate", "smoke", "summarize", "source_hashes", "save_checkpoint"):
        if not callable(getattr(runner, name, None)):
            raise RuntimeError(f"Incompatible runner: missing {name}.")
    native_label = runner.method_label
    native_hashes = runner.source_hashes
    native_save = runner.save_checkpoint

    def method_label(method, config):
        return {"se": "RetinaNet + SE (original implementation)",
                "cbam": "RetinaNet + CBAM (original implementation)"}.get(method) or native_label(method, config)

    def source_hashes():
        return {**native_hashes(), **extension_hashes()}

    def save_checkpoint(path, value):
        if isinstance(value, dict) and value.get("method") in METHODS:
            if value.get("config", {}).get("block_type") != value["method"]:
                raise ValueError("Refusing to save a checkpoint whose method and attention block differ.")
            value = {**value, "se_cbam_extension": {"version": VERSION, "sources": extension_hashes()}}
        return native_save(path, value)

    runner.METHODS = list(dict.fromkeys([*runner.METHODS, *METHODS]))
    runner.build_model = bridge.build_model
    runner.method_label = method_label
    runner.source_hashes = source_hashes
    runner.save_checkpoint = save_checkpoint
    runner._se_cbam_original_source_hashes = native_hashes
    runner._se_cbam_installed = True
    return runner, bridge


def validate_reference_settings(args, reference: dict, runner, device):
    if reference.get("method") not in {"baseline", "lgf_original"}:
        raise ValueError("Reference must be a baseline or lgf_original UGS-runner protocol.json.")
    expected = {"version": runner.VERSION, "dataset": args.dataset, "seed": args.seed,
                "level_label_preserved": args.level, **FIXED_PROTOCOL}
    expected.update({key: getattr(args, key) for key in REFERENCE_FIELDS})
    expected["amp"] = bool(args.amp and device.type == "cuda")
    mismatches = {key: {"reference": reference.get(key), "requested": value}
                  for key, value in expected.items() if reference.get(key) != value}
    current_environment = runner.environment()
    environment_mismatch = environment_differences(reference.get("environment"), current_environment)
    if environment_mismatch:
        mismatches["environment"] = environment_mismatch
    if reference.get("source_hashes") != runner._se_cbam_original_source_hashes():
        mismatches["source_hashes"] = "Original package source differs from the reference run."
    if mismatches:
        raise ValueError("Reference protocol mismatch:\n" + json.dumps(mismatches, indent=2))
    hardware = gpu_difference(reference["environment"], current_environment)
    if hardware:
        print("GPU MODEL DIFFERS FROM REFERENCE " + json.dumps(hardware, sort_keys=True), flush=True)


def bind_reference(args, path: Path, runner, device):
    reference = read_json(path)
    unsigned = {k: v for k, v in reference.items() if k not in {"protocol_sha256", "method_label"}}
    if reference.get("protocol_sha256") != runner.signature(unsigned):
        raise ValueError("Reference protocol.json has an invalid protocol_sha256.")
    validate_reference_settings(args, reference, runner, device)
    native_audit, native_model_protocol = runner.audit, runner.model_protocol

    def audit(dataset, root, **kwargs):
        record = native_audit(dataset, root, **kwargs)
        for split, key in (("train", "training_content_sha256"), ("val", "validation_content_sha256")):
            if split not in record or record[split]["content_sha256"] != reference.get(key):
                raise ValueError(f"{split} data differ from the reference LGF/baseline run.")
        record["se_cbam_reference"] = {"path": str(path.resolve()), "sha256": digest(path),
                                        "reference_method": reference["method"]}
        return record

    def model_protocol(model):
        result = native_model_protocol(model)
        # JSON-normalize tuples produced by anchor sizes/ratios.
        if json.loads(json.dumps(result)) != reference.get("model_protocol"):
            raise ValueError("Model preprocessing, anchors, or evaluation settings differ from the reference.")
        return result

    runner.audit, runner.model_protocol = audit, model_protocol
    print(f"REFERENCE SETTINGS PASS {path}", flush=True)


def validate_evaluation_checkpoint(args, runner):
    if args.checkpoint is None:
        raise ValueError("--checkpoint is required for evaluate.")
    import torch
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if ck.get("config", {}).get("block_type") != args.method:
        raise ValueError("Checkpoint attention configuration differs from --method.")
    if not args.original_checkpoint:
        identity = (ck.get("method"), ck.get("dataset"), ck.get("seed"), ck.get("level"))
        if identity != (args.method, args.dataset, args.seed, args.level):
            raise ValueError("Checkpoint method/dataset/seed/level differs from the requested evaluation.")
        if ck.get("package_version") != runner.VERSION:
            raise ValueError("Unsupported checkpoint package version.")
    del ck


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv == ["--help"] or argv == ["-h"]:
        print(__doc__)
        print("Actions: check, audit, smoke, train, evaluate, summarize.\nUse ACTION --help for native arguments.")
        return
    extra = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    extra.add_argument("--package", type=Path)
    extra.add_argument("--reference-protocol", type=Path)
    ext, forwarded = extra.parse_known_args(argv)
    package = resolve_package(ext.package)
    runner, bridge = load_runner(package)
    if not any(arg == "--method" or arg.startswith("--method=") for arg in forwarded):
        forwarded.extend(["--method", "se"])
    forwarded = with_storage_defaults(forwarded, {"--data-root": DEFAULT_DATA_ROOT, "--run-root": DEFAULT_RUN_ROOT})
    saved_argv = sys.argv
    try:
        sys.argv = [str(package / "run.py"), *forwarded]
        args = runner.arguments()
    finally:
        sys.argv = saved_argv
    if args.method not in METHODS:
        raise ValueError("This extension trains SE/CBAM only. Keep using the unchanged run.py for baseline/LGF/UGS.")
    if ext.reference_protocol and args.action != "train":
        raise ValueError("--reference-protocol applies to train only.")
    print(f"SE/CBAM EXTENSION {VERSION} | unchanged package: {package}", flush=True)
    if args.action == "summarize":
        runner.summarize(args)
        return
    core, config, device = runner.configure(args)
    if ext.reference_protocol:
        bind_reference(args, ext.reference_protocol, runner, device)
    if args.action == "check":
        print(json.dumps(runner.environment(), indent=2))
        print("ENVIRONMENT AND EXTENSION CONTRACT PASS")
    elif args.action == "audit":
        record = runner.audit(args.dataset, args.data_root)
        output = args.output or Path(args.run_root) / "audits" / (args.dataset + ".json")
        runner.write_json(output, record)
        print(f"DATA AUDIT PASS {output}")
    elif args.action == "smoke":
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
        runner.smoke(args, core, config, device)
    elif args.action == "train":
        runner.run_train(args, core, config, device)
    elif args.action == "evaluate":
        validate_evaluation_checkpoint(args, runner)
        runner.run_evaluate(args, config, device)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
