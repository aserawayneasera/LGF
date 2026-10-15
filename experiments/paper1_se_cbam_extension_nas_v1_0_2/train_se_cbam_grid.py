#!/usr/bin/env python3
"""Run SE/CBAM sequentially with settings read from existing LGF/baseline runs.

--check prints the complete plan without training, writing files, or using a GPU.
The default grid is 2 methods x 3 datasets x 3 seeds. Completed runs are skipped;
interrupted runs resume from last.pth with the original strict protocol checks.
"""
from __future__ import annotations
import argparse
import datetime
import json
from pathlib import Path
import shlex
import subprocess
import sys

from run_se_cbam import (FIXED_PROTOCOL, HERE, METHODS, REFERENCE_FIELDS, digest,
                         environment_differences, extension_hashes, gpu_difference,
                         load_runner, read_json, resolve_package)

DEFAULT_RUNS = Path("/nas.dbms/asera/paper1_original/ugs_comparison/runs")
DATASETS = ("coco_nw", "coco_weather", "acdc")


def original_hashes(package):
    return {str(p.relative_to(package)): digest(p) for p in sorted(package.rglob("*.py"))
            if "__pycache__" not in p.parts}


def find_reference(args, dataset, seed):
    roots = [args.reference_root] if args.reference_root else [
        args.run_root, DEFAULT_RUNS]
    matches = list(dict.fromkeys((root / dataset / args.reference_method / f"seed_{seed}" / "protocol.json").resolve()
                               for root in roots
                               if (root / dataset / args.reference_method / f"seed_{seed}" / "protocol.json").is_file()))
    if not matches:
        raise FileNotFoundError(f"Missing {args.reference_method}/{dataset}/seed_{seed}/protocol.json. "
                                "Set --reference-root to the existing runs directory, or --reference-method baseline.")
    if len({read_json(p).get("protocol_sha256") for p in matches}) > 1:
        raise ValueError(f"Different reference runs exist for {dataset}, seed {seed}. Select one --reference-root.")
    path = matches[0]
    ref = read_json(path)
    if (ref.get("method"), ref.get("dataset"), ref.get("seed")) != (args.reference_method, dataset, seed):
        raise ValueError(f"Reference identity differs from its directory: {path}")
    if ref.get("version") != "1.0.0":
        raise ValueError(f"Unsupported reference runner version: {path}")
    for key in (*REFERENCE_FIELDS, "level_label_preserved", "model_protocol", "training_content_sha256",
                "validation_content_sha256", "environment", "source_hashes", "protocol_sha256"):
        if key not in ref:
            raise ValueError(f"Reference is missing {key}: {path}")
    for key, expected in FIXED_PROTOCOL.items():
        if ref.get(key) != expected:
            raise ValueError(f"Reference {key} does not match the delivered runner: {path}")
    unsigned = {k: v for k, v in ref.items() if k not in {"method_label", "protocol_sha256"}}
    import hashlib
    signature = hashlib.sha256(json.dumps(unsigned, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    if signature != ref["protocol_sha256"]:
        raise ValueError(f"Invalid reference protocol_sha256: {path}")
    return path, ref


def make_job(args, package, method, dataset, seed, hashes):
    reference_path, ref = find_reference(args, dataset, seed)
    if ref["source_hashes"] != hashes:
        raise ValueError(f"Current UGS source files differ from reference {reference_path}. "
                         "Use the original package used for those LGF/baseline runs.")
    run_dir = args.run_root / dataset / method / f"seed_{seed}"
    command = [sys.executable, "-u", str(HERE / "run_se_cbam.py"), "--package", str(package),
               "--reference-protocol", str(reference_path), "train", "--method", method,
               "--dataset", dataset, "--seed", str(seed), "--level", ref["level_label_preserved"],
               "--data-root", str(args.data_root), "--run-root", str(args.run_root),
               "--device", args.device, "--threads", str(args.threads)]
    for key in REFERENCE_FIELDS:
        if key != "amp":
            command += ["--" + key.replace("_", "-"), str(ref[key])]
    command += ["--amp" if ref["amp"] else "--no-amp"]
    status = "TRAIN"
    if run_dir.exists():
        protocol_path = run_dir / "protocol.json"
        if not protocol_path.is_file():
            raise ValueError(f"Existing run has no protocol.json: {run_dir}. No files were overwritten.")
        prior = read_json(protocol_path)
        if (prior.get("dataset"), prior.get("method"), prior.get("seed")) != (dataset, method, seed):
            raise ValueError(f"Existing run has another identity: {run_dir}")
        compare_keys = (*REFERENCE_FIELDS, *FIXED_PROTOCOL, "version", "level_label_preserved",
                        "training_content_sha256", "validation_content_sha256", "model_protocol")
        if any(prior.get(k) != ref.get(k) for k in compare_keys):
            raise ValueError(f"Existing SE/CBAM run settings differ from the selected reference: {run_dir}")
        differences = environment_differences(ref.get("environment"), prior.get("environment"))
        if differences:
            raise ValueError(f"Existing SE/CBAM run environment differs from the selected reference: {run_dir}\n"
                             + json.dumps(differences, indent=2))
        if prior.get("source_hashes") != {**hashes, **extension_hashes()}:
            raise ValueError(f"Existing SE/CBAM source fingerprint differs: {run_dir}")
        completed = run_dir / "completed.json"
        if completed.exists():
            done = read_json(completed)
            if done.get("protocol_sha256") != prior.get("protocol_sha256") or done.get("completed_epochs") != ref["epochs"]:
                raise ValueError(f"Completion metadata mismatch: {run_dir}")
            for selection in ("best_map", "best_ap_small"):
                if not (run_dir / f"{selection}.pth").is_file() or not (run_dir / f"{selection}.metrics.json").is_file():
                    raise ValueError(f"Completed run is missing selected checkpoint files: {run_dir}")
            status = "SKIP COMPLETED"
        elif (run_dir / "last.pth").is_file():
            command += ["--resume", str(run_dir / "last.pth")]
            status = "RESUME"
        else:
            raise ValueError(f"Existing incomplete run has no last.pth: {run_dir}. No files were overwritten.")
    return status, command, args.log_root / f"{method}_{dataset}_seed{seed}.log", reference_path


def execute(command, log):
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", buffering=1) as handle:
        handle.write("\nSTART " + datetime.datetime.now(datetime.timezone.utc).isoformat() + "\n")
        handle.write(shlex.join(command) + "\n")
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, bufsize=1)
        try:
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="", flush=True)
                handle.write(line)
            result = process.wait()
        except KeyboardInterrupt:
            process.terminate()
            process.wait()
            raise
        if result:
            raise RuntimeError(f"Training stopped with exit code {result}. See {log}. Rerun the same grid to resume committed checkpoints.")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--package", type=Path)
    p.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    p.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 1337, 2025])
    p.add_argument("--data-root", type=Path, default=Path("/nas.dbms/asera/paper1_original/DATASET"))
    p.add_argument("--run-root", type=Path, default=DEFAULT_RUNS)
    p.add_argument("--reference-root", type=Path)
    p.add_argument("--reference-method", choices=("lgf_original", "baseline"), default="lgf_original")
    p.add_argument("--log-root", type=Path)
    p.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--check", action="store_true")
    args = p.parse_args()
    for key in ("methods", "datasets", "seeds"):
        values = getattr(args, key)
        if len(values) != len(set(values)):
            p.error(f"--{key} contains duplicates")
    if args.threads < 1:
        p.error("--threads must be positive")
    args.run_root = args.run_root.expanduser().resolve()
    args.data_root = args.data_root.expanduser().resolve()
    args.log_root = (args.log_root or args.run_root.parent / "logs").expanduser().resolve()
    if args.reference_root:
        args.reference_root = args.reference_root.expanduser().resolve()
    package = resolve_package(args.package)
    hashes = original_hashes(package)
    # Validate the entire grid before starting the first job.
    jobs = [make_job(args, package, method, dataset, seed, hashes)
            for dataset in args.datasets for method in args.methods for seed in args.seeds]
    for status, command, log, reference in jobs:
        print(f"\n{status} | reference: {reference}\n{shlex.join(command)}\nLOG {log}")
    print(f"\n{len(jobs)} jobs. Sequential execution, one process at a time.", flush=True)
    if args.check:
        print("PLAN CHECK PASS. No training or dataset-content/GPU checks were performed.")
        return
    # Check every reference with the runtime that will launch the children.
    # Do this before starting the first of 18 jobs.
    runner, _ = load_runner(package)
    current_environment = runner.environment()
    hardware_pairs = set()
    for reference in sorted({reference for status, command, log, reference in jobs}):
        recorded_environment = read_json(reference).get("environment")
        differences = environment_differences(recorded_environment, current_environment)
        if differences:
            raise ValueError(f"Runtime differs from reference {reference}:\n"
                             + json.dumps(differences, indent=2))
        hardware = gpu_difference(recorded_environment, current_environment)
        if hardware:
            hardware_pairs.add((hardware["reference"], hardware["current"]))
    for before, after in sorted(hardware_pairs, key=lambda pair: str(pair)):
        print("GPU MODEL DIFFERS FROM REFERENCE " + json.dumps({"reference": before, "current": after}), flush=True)
    print("ENVIRONMENT PREFLIGHT PASS", flush=True)
    for status, command, log, reference in jobs:
        if status != "SKIP COMPLETED":
            execute(command, log)
    print("SE/CBAM GRID COMPLETE", flush=True)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
