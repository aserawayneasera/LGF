#!/usr/bin/env python3
"""Use the existing shared test exporter with new SE/CBAM runner checkpoints.

This is a separate entry point. Existing evaluate_test.py, models.py, manifests,
checkpoint files, scoring logic, and other model exports are not edited.

Use all existing evaluate_test.py arguments, plus optional --eval-dir and --package.
Example:
  python evaluate_test_se_cbam.py --package /path/paper1_ugs_comparison_v1 \
    export --method se --train-dataset coco_nw --seed 42 --level C3 \
    --manifest /path/coco_nw_test.json --checkpoint /path/best_map.pth \
    --selection best_map --output /path/new_se_export --device 0
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
from run_se_cbam import (METHODS, DEFAULT_DATA_ROOT, DEFAULT_RUNTIME_ROOT, digest,
                         extension_hashes, import_at, load_bridge, resolve_package, with_storage_defaults)


def load_exporter(package: Path, eval_dir: Path):
    load_bridge(package)
    sys.path.insert(0, str(eval_dir))
    # Reject accidental imports from another evaluation package in long-lived processes.
    for name in ("test_data", "models"):
        import_at(name, eval_dir / (name + ".py"))
    module = import_at("paper1_se_cbam_test_exporter", eval_dir / "evaluate_test.py")
    if getattr(module, "_se_cbam_sources_installed", False):
        return module
    native_signature = module.signature
    native_load = module.load_model
    hashes = {**extension_hashes(), "se_cbam_extension/evaluate_test_se_cbam.py": digest(Path(__file__))}

    def signature(value):
        if isinstance(value, dict) and value.get("schema") == "paper1-test-export-1.0" and value.get("method") in METHODS:
            # Include the extension in the existing export integrity/resume signature.
            value.setdefault("adapter_sources", {}).update(hashes)
        return native_signature(value)

    def load_model(args, names):
        if args.method in METHODS:
            import torch
            ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
            if ck.get("config", {}).get("block_type") != args.method:
                raise ValueError("SE/CBAM checkpoint configuration differs from --method.")
            del ck
        return native_load(args, names)

    module.signature, module.load_model = signature, load_model
    module._se_cbam_sources_installed = True
    return module


def main():
    argv = sys.argv[1:]
    if not argv or argv in (["-h"], ["--help"]):
        print(__doc__)
        print("Actions: export, score, compare, package-acdc. Use ACTION --help for native arguments.")
        return
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--package", type=Path)
    parser.add_argument("--eval-dir", type=Path)
    extra, forwarded = parser.parse_known_args(argv)
    if forwarded and forwarded[0] == "export":
        forwarded = with_storage_defaults(forwarded, {"--data-root": DEFAULT_DATA_ROOT, "--runtime-root": DEFAULT_RUNTIME_ROOT})
    package = resolve_package(extra.package)
    eval_dir = (extra.eval_dir or package.parent / "paper1_test_evaluation_v1").expanduser().resolve()
    for name in ("evaluate_test.py", "models.py", "test_data.py"):
        if not (eval_dir / name).is_file():
            raise FileNotFoundError(f"Missing {eval_dir / name}. Set --eval-dir to the shared test-evaluation package.")
    module = load_exporter(package, eval_dir)
    previous = sys.argv
    try:
        sys.argv = [str(eval_dir / "evaluate_test.py"), *forwarded]
        args = module.arguments()
    finally:
        sys.argv = previous
    if args.action == "export":
        if args.method not in METHODS:
            raise ValueError("Use the original exporter for baseline/LGF/UGS/YOLO/RT-DETR.")
        if any(v == "--lgf-package" or v.startswith("--lgf-package=") for v in forwarded):
            if Path(args.lgf_package).resolve() != package:
                raise ValueError("--lgf-package and --package refer to different UGS packages.")
        args.lgf_package = package
    {"export": module.export, "score": module.score, "compare": module.compare,
     "package-acdc": module.package_acdc}[args.action](args)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
