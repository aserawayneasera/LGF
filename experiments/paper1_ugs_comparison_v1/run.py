#!/usr/bin/env python3
"""UGS comparison runner. Original LGF notebook files are never edited."""
import argparse
import contextlib
import copy
import csv
from dataclasses import asdict
import datetime
import gc
import json
import os
from pathlib import Path
import random
import re
import sys
import time

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("TORCH_HOME", "/tmp/asera/paper1_original/torch_cache")

import numpy as np
import torch
from tqdm import tqdm

from core_bridge import ROOT, build_model, load_core
from data import audit, make_loaders
from evaluation import evaluate, model_protocol
from ugs import UGSConfig
from utils import environment, plain, require_supported_torch, save_checkpoint, sha256, signature, source_hashes, write_json

METHODS = ["ugs_reimpl", "baseline", "lgf_original"]
VERSION = "1.0.0"


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    for action in ["check", "audit", "smoke", "train", "evaluate", "summarize"]:
        p = sub.add_parser(action)
        p.add_argument("--dataset", choices=["coco_nw", "coco_weather", "acdc"], default="coco_nw")
        p.add_argument("--data-root", default="/tmp/asera/paper1_original/DATASET")
        p.add_argument("--run-root", default="/tmp/asera/paper1_original/ugs_comparison/runs")
        p.add_argument("--method", choices=METHODS, default="ugs_reimpl")
        p.add_argument("--seed", type=int, default=42)
        p.add_argument("--level", choices=["C3", "C4", "C5"], default="C3")
        p.add_argument("--ugs-config", default=str(ROOT / "configs/ugs_reimpl.json"))
        p.add_argument("--device", choices=["cuda", "cpu"], default="cuda" if torch.cuda.is_available() else "cpu")
        p.add_argument("--batch-size", type=int, default=4)
        p.add_argument("--accum-steps", type=int, default=4)
        p.add_argument("--num-workers", type=int, default=8)
        p.add_argument("--prefetch-factor", type=int, default=4)
        p.add_argument("--threads", type=int, default=4)
        p.add_argument("--epochs", type=int, default=80)
        p.add_argument("--base-lr", type=float, default=0.005)
        p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=False)
        p.add_argument("--synthetic", action="store_true", help="Smoke only; no real data or pretrained download")
        p.add_argument("--resume", type=Path, help="Resume last.pth at an epoch boundary")
        p.add_argument("--checkpoint", type=Path)
        p.add_argument("--original-checkpoint", action="store_true")
        p.add_argument("--selection-note", default="selection not independently verified")
        p.add_argument("--output", type=Path)
        p.add_argument("--selection", choices=["best_ap_small", "best_map"], default="best_ap_small")
    return parser.parse_args()


def configure(args):
    require_supported_torch()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if min(args.epochs, args.batch_size, args.accum_steps, args.threads, args.prefetch_factor) < 1 or args.num_workers < 0:
        raise ValueError("Invalid training/loader argument")
    if not np.isfinite(args.base_lr) or args.base_lr <= 0:
        raise ValueError("Learning rate must be finite and positive")
    if os.environ.get("MAX_TRAIN_IMAGES"):
        raise RuntimeError("Unset MAX_TRAIN_IMAGES; the original reader uses it to change the training set")
    if args.synthetic and args.action != "smoke":
        raise ValueError("Synthetic data are only allowed for the smoke action")
    torch.set_num_threads(args.threads)
    core = load_core()
    core.set_seed(args.seed)
    core.args.seed = args.seed
    core.args.insert_level = args.level
    config = UGSConfig(**json.loads(Path(args.ugs_config).read_text()))
    return core, config, torch.device(args.device)


def method_label(method, config):
    return config.label if method == "ugs_reimpl" else {
        "baseline": "RetinaNet baseline (original model)",
        "lgf_original": "LGF (original model implementation)"}[method]


def phase_optimizer(model, phase, lr):
    for p in model.parameters():
        p.requires_grad_(phase == "all")
    if phase == "heads":
        for p in model.head.parameters():
            p.requires_grad_(True)
        if hasattr(model.backbone, "block_fpn_in"):
            for p in model.backbone.block_fpn_in.parameters():
                p.requires_grad_(True)
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad],
                                lr=lr, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40, 60], gamma=0.1)
    return optimizer, scheduler


def rng_state(generator, dataset):
    state = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
             "generator": generator.get_state(), "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}
    transform = getattr(dataset, "transforms", None)
    if transform is not None:
        state["augmentation_generators"] = [(copy.deepcopy(t.random_generator.bit_generator.state), t.py_random.getstate())
                                            for t in [transform] + list(transform.transforms)]
    return state


def restore_rng(state, generator, dataset):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    generator.set_state(state["generator"])
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
    transform = getattr(dataset, "transforms", None)
    if transform is not None and "augmentation_generators" in state:
        for t, (n, p) in zip([transform] + list(transform.transforms), state["augmentation_generators"]):
            t.random_generator.bit_generator.state = n
            t.py_random.setstate(p)


def step_optimizer(model, optimizer, scaler, amp):
    if amp:
        scaler.unscale_(optimizer)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0, error_if_nonfinite=True)
    if amp:
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return float(norm)


def training_step(model, images, targets, device, amp, accum):
    images = [im.to(device) for im in images]
    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
        losses = model(images, targets)
        total = sum(losses.values()) / accum
    if not torch.isfinite(total):
        raise FloatingPointError(f"Nonfinite loss: { {k:float(v.detach()) for k,v in losses.items()} }")
    return losses, total


def run_train(args, core, config, device):
    run_dir = Path(args.run_root) / args.dataset / args.method / f"seed_{args.seed}"
    if run_dir.exists() and not args.resume:
        raise FileExistsError(f"Run already exists: {run_dir}. Use --resume .../last.pth or a new --run-root.")
    if args.resume and args.resume.resolve().parent != run_dir.resolve():
        raise ValueError("Resume checkpoint must belong to this dataset/method/seed run directory")
    record = audit(args.dataset, args.data_root)
    run_dir.mkdir(parents=True, exist_ok=True)
    # A lock prevents two workers or tmux sessions from writing the same run.
    import fcntl
    lock = open(run_dir / "run.lock", "a+")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError("This run is already active") from exc
    train_loader, val_loader, generator = make_loaders(
        record, args.batch_size, args.num_workers, args.prefetch_factor, args.seed)
    if not len(train_loader) or len(train_loader) % args.accum_steps:
        raise ValueError("Training batches must be nonzero and divisible by accum-steps. "
                         "This avoids silently changing the original accumulation behavior.")
    amp = args.amp and device.type == "cuda"
    model = build_model(args.method, 6, args.level, config, pretrained=not bool(args.resume)).to(device)
    original_config = copy.deepcopy(core.CURRENT_CONFIG)
    env = environment()
    protocol = {"version": VERSION, "dataset": args.dataset, "method": args.method, "seed": args.seed,
        "level_label_preserved": args.level, "epochs": args.epochs, "batch_size": args.batch_size,
        "accum_steps": args.accum_steps, "num_workers": args.num_workers, "prefetch_factor": args.prefetch_factor,
        "base_lr": args.base_lr, "amp": amp, "optimizer": "SGD(momentum=0.9,weight_decay=1e-4)",
        "warmup_epochs": 2, "unfreeze_at_epoch_index": 5, "scheduler_milestones_after_reset": [40, 60],
        "scheduler_reset_at_unfreeze": True, "ema_decay": 0.9999, "grad_clip": 10.0,
        "pretraining": "RetinaNet_ResNet50_FPN_Weights.COCO_V1",
        "training_content_sha256": record["train"]["content_sha256"],
        "validation_content_sha256": record["val"]["content_sha256"],
        "model_protocol": model_protocol(model), "environment": env,
        "ugs_config": asdict(config) if args.method == "ugs_reimpl" else None,
        "source_hashes": source_hashes(), "augmentation_seed_policy": "explicit seed per run/worker",
        "selection": "save both best validation APsmall and best validation mAP; evaluate every 5 epochs"}
    protocol_id = signature(protocol)
    phase = "heads"
    optimizer, scheduler = phase_optimizer(model, phase, args.base_lr)
    ema = core.ModelEMA(model, decay=0.9999, device=device)
    scaler = torch.amp.GradScaler(device.type, enabled=amp)
    best = {"best_ap_small": -float("inf"), "best_map": -float("inf")}
    start_epoch = 0
    if args.resume:
        ck = torch.load(args.resume, map_location="cpu", weights_only=False)
        if ck["protocol_sha256"] != protocol_id:
            raise ValueError("Resume protocol differs from the saved run (data/config/environment/source)")
        model.load_state_dict(ck["model_state_dict"], strict=True)
        ema.ema.load_state_dict(ck["ema_state_dict"], strict=True)
        phase = ck["phase"]
        optimizer, scheduler = phase_optimizer(model, phase, args.base_lr)
        optimizer.load_state_dict(ck["optimizer_state_dict"])
        scheduler.load_state_dict(ck["scheduler_state_dict"])
        scaler.load_state_dict(ck["scaler_state_dict"])
        best, start_epoch = ck["best"], ck["completed_epochs"]
        history_path = run_dir / "history.jsonl"
        history = []
        if history_path.exists():
            for line in history_path.read_text().splitlines():
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if item["epoch"] <= start_epoch:
                    history.append(item)
        if not any(item["epoch"] == start_epoch for item in history) and ck.get("epoch_entry"):
            history.append(ck["epoch_entry"])
        temporary = history_path.with_suffix(".tmp")
        temporary.write_text("".join(json.dumps(item)+"\n" for item in history))
        os.replace(temporary, history_path)
        # last.pth commits the epoch before selected copies are published. Repair
        # a best-checkpoint copy if interruption occurred between those writes.
        for key in ck.get("improved_selection_keys", []):
            cp, mp = run_dir/(key+".pth"), run_dir/(key+".metrics.json")
            intact = False
            if cp.is_file() and mp.is_file():
                info = json.loads(mp.read_text())
                intact = info["epoch"] == start_epoch and info["checkpoint_sha256"] == sha256(cp)
            if not intact:
                save_checkpoint(cp, ck)
                write_json(mp, {**ck["epoch_result"], "selection": key,
                               "checkpoint": str(cp), "checkpoint_sha256": sha256(cp)})
        restore_rng(ck["rng"], generator, train_loader.dataset)
        with open(run_dir / "resumptions.jsonl", "a") as f:
            f.write(json.dumps({"checkpoint": str(args.resume), "completed_epochs": start_epoch,
                "time_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "limit": "Epoch-boundary state restored; multiworker augmentation continuation is not bitwise guaranteed."}) + "\n")
    write_json(run_dir / "data_audit.json", record)
    write_json(run_dir / "protocol.json", {**protocol, "protocol_sha256": protocol_id,
                                           "method_label": method_label(args.method, config)})
    print(f"RUN {method_label(args.method, config)} | {args.dataset} seed={args.seed}", flush=True)
    print(f"DATA train={record['train']['image_count']} val={record['val']['image_count']} | amp={amp}", flush=True)
    print(f"OUTPUT {run_dir}", flush=True)
    optimizer.zero_grad(set_to_none=True)
    try:
        for epoch in range(start_epoch, args.epochs):
            if phase == "heads" and epoch == 5:
                phase = "all"
                optimizer, scheduler = phase_optimizer(model, phase, args.base_lr)
            # Preserve the original notebook's actual warm-up and scheduler reset.
            if epoch < 2:
                for group in optimizer.param_groups:
                    group["lr"] *= min(1., (epoch + 1) / 2)
            model.train()
            started = time.perf_counter()
            totals, diagnostics, batches = {}, {}, 0
            prog = tqdm(train_loader, desc=f"{args.method} {args.dataset} s{args.seed} epoch {epoch+1}/{args.epochs}")
            for bi, (images, targets) in enumerate(prog):
                losses, total = training_step(model, images, targets, device, amp, args.accum_steps)
                if amp:
                    scaler.scale(total).backward()
                else:
                    total.backward()
                if (bi + 1) % args.accum_steps == 0:
                    step_optimizer(model, optimizer, scaler, amp)
                    ema.update(model)
                batches += 1
                for k, v in losses.items():
                    totals[k] = totals.get(k, 0.0) + float(v.detach())
                for k, v in getattr(model.head, "last_stats", {}).items():
                    if k == "max_perturbation_l2":
                        diagnostics[k] = max(diagnostics.get(k, 0.0), v)
                    elif k in {"positive_anchors", "regression_coordinates", "clipped_coordinates", "nonzero_perturbation_directions"}:
                        diagnostics[k] = diagnostics.get(k, 0) + v
                prog.set_postfix(loss=f"{float(total.detach())*args.accum_steps:.4f}", lr=optimizer.param_groups[0]["lr"])
            train_seconds = time.perf_counter() - started
            scheduler.step()
            entry = {"epoch": epoch + 1, "training_seconds": train_seconds,
                     "losses": {k: v / batches for k, v in totals.items()}, "diagnostics": diagnostics,
                     "lr_after_scheduler": optimizer.param_groups[0]["lr"], "phase": phase}
            result = None
            if (epoch + 1) % 5 == 0 or epoch + 1 == args.epochs:
                result = evaluate(ema.ema, val_loader, record["paths"]["val_annotations"], device,
                    run_dir / "evaluation" / f"epoch_{epoch+1:03d}", record,
                    {"method": args.method, "method_label": method_label(args.method, config),
                     "seed": args.seed, "epoch": epoch + 1, "protocol_sha256": protocol_id,
                     "ugs_config": asdict(config) if args.method == "ugs_reimpl" else None})
                entry["metrics"] = result["metrics"]
            improved = []
            if result:
                for key, metric in [("best_ap_small", "AP_small"), ("best_map", "mAP")]:
                    value = result["metrics"][metric]
                    if value is not None and value >= 0 and value > best[key]:
                        best[key] = value
                        improved.append(key)
            state = {"package_version": VERSION, "method": args.method, "ugs_config": asdict(config),
                     "config": original_config, "level": args.level, "seed": args.seed, "dataset": args.dataset,
                     "completed_epochs": epoch + 1, "phase": phase, "protocol_sha256": protocol_id,
                     "protocol": protocol, "best": best, "model_state_dict": model.state_dict(),
                     "ema_state_dict": ema.ema.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                     "scheduler_state_dict": scheduler.state_dict(), "scaler_state_dict": scaler.state_dict(),
                     "rng": rng_state(generator, train_loader.dataset), "epoch_entry": entry,
                     "epoch_result": result, "improved_selection_keys": improved}
            save_checkpoint(run_dir / "last.pth", state)
            for key in improved:
                cp = run_dir / (key + ".pth")
                save_checkpoint(cp, state)
                write_json(run_dir / (key + ".metrics.json"), {**result, "selection": key,
                           "checkpoint": str(cp), "checkpoint_sha256": sha256(cp)})
            with open(run_dir / "history.jsonl", "a") as f:
                f.write(json.dumps(plain(entry), allow_nan=False) + "\n")
                f.flush()
            print(f"EPOCH PASS {epoch+1} | train_seconds={train_seconds:.1f} | checkpoint={run_dir/'last.pth'}", flush=True)
        write_json(run_dir / "completed.json", {"completed_epochs": args.epochs,
                   "method_label": method_label(args.method, config), "best": best,
                   "protocol_sha256": protocol_id})
        print("TRAINING COMPLETE", flush=True)
    finally:
        lock.close()


def smoke(args, core, config, device):
    core.set_seed(args.seed)
    amp = args.amp and device.type == "cuda"
    if args.synthetic:
        model = build_model(args.method, 6, args.level, config, pretrained=False, smoke_size=64).to(device)
        images = [torch.rand(3, 64, 64), torch.rand(3, 64, 64)]
        targets = [{"boxes": torch.tensor([[10., 10., 26., 28.]]), "labels": torch.tensor([i])}
                   for i in range(2)]
    else:
        rec = audit(args.dataset, args.data_root)
        train, _, _ = make_loaders(rec, args.batch_size, args.num_workers, args.prefetch_factor, args.seed)
        if not len(train):
            raise ValueError("No full training batches")
        images, targets = next(iter(train))
        model = build_model(args.method, 6, args.level, config).to(device)
    scaler = torch.amp.GradScaler(device.type, enabled=amp)
    report = {"synthetic": args.synthetic, "method_label": method_label(args.method, config),
              "environment": environment(), "stages": [], "scientific_result": False}
    # Exercise both the frozen-backbone warm-up and full-backbone training.
    for phase in ["heads", "all"]:
        opt, _ = phase_optimizer(model, phase, 1e-4)
        model.train()
        losses, total = training_step(model, images, targets, device, amp, 1)
        if amp:
            scaler.scale(total).backward()
        else:
            total.backward()
        norm = step_optimizer(model, opt, scaler, amp)
        stage = {"phase": phase, "losses": {k: float(v.detach()) for k,v in losses.items()},
                 "gradient_norm": norm, "ugs_diagnostics": copy.deepcopy(getattr(model.head, "last_stats", {}))}
        report["stages"].append(stage)
        print("SMOKE PHASE PASS", json.dumps(stage), flush=True)
    model.eval()
    if args.synthetic:
        # Random classification weights start below the normal score threshold.
        # Lower it only for this synthetic check to exercise nonempty decoding/NMS.
        model.score_thresh = 0.0
        report["smoke_only_score_thresh"] = 0.0
    with torch.no_grad():
        predictions = model([im.to(device) for im in images])
    if args.synthetic and any(len(p["boxes"]) == 0 for p in predictions):
        raise RuntimeError("Synthetic smoke must exercise nonempty box decoding")
    for pred in predictions:
        if not torch.isfinite(pred["boxes"]).all() or not torch.isfinite(pred["scores"]).all():
            raise FloatingPointError("Nonfinite decoded predictions")
    import tempfile
    with tempfile.TemporaryDirectory(prefix="ugs_smoke_", dir=args.output.parent if args.output else None) as tmp:
        path = Path(tmp) / "state.pth"
        save_checkpoint(path, model.state_dict())
        model.load_state_dict(torch.load(path, map_location=device, weights_only=True), strict=True)
        with torch.no_grad():
            restored = model([im.to(device) for im in images])
        for before, after in zip(predictions, restored):
            for key in ["boxes", "scores", "labels"]:
                torch.testing.assert_close(before[key], after[key], rtol=0, atol=0)
    report["status"] = "passed"
    report["prediction_counts"] = [len(p["boxes"]) for p in predictions]
    report["checkpoint_reload_identical"] = True
    if args.output:
        write_json(args.output, report)
    print("SMOKE PASS: training, UR gradients, inference, checkpoint reload. No scientific result.", flush=True)


def run_evaluate(args, config, device):
    if not args.checkpoint:
        raise ValueError("--checkpoint is required")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    record = audit(args.dataset, args.data_root, val_only=True)
    _, loader, _ = make_loaders(record, args.batch_size, args.num_workers, args.prefetch_factor, args.seed, val_only=True)
    if args.original_checkpoint:
        cfg = checkpoint.get("config")
        if not cfg:
            raise ValueError("Original checkpoint must include its config")
        if cfg.get("block_type") not in {"none", "lgf", "se", "cbam"}:
            raise ValueError("Unsupported original checkpoint configuration")
        model = build_model("baseline", 6, args.level, pretrained=False, original_config=cfg).to(device)
        label = "Original checkpoint: " + str(cfg.get("description", cfg))
        method, selected = "original_" + cfg["block_type"], args.selection_note
        epoch = checkpoint["epoch"] + 1 if "epoch" in checkpoint else None
        seed = args.seed
        name = str(checkpoint.get("experiment_name", ""))
        identity = re.fullmatch(r"(coco_nw|coco_weather|acdc)_(C[345])_.+_s(-?\d+)", name)
        if identity and (identity[1], identity[2], int(identity[3])) != (args.dataset, args.level, args.seed):
            raise ValueError("Dataset/level/seed arguments differ from original checkpoint experiment_name")
    else:
        if checkpoint.get("package_version") != VERSION:
            raise ValueError("Not a checkpoint from this package; use --original-checkpoint when appropriate")
        if checkpoint["dataset"] != args.dataset:
            raise ValueError("Checkpoint was trained on another dataset; cross-dataset transfer is a different experiment")
        config = UGSConfig(**checkpoint["ugs_config"])
        method = checkpoint["method"]
        model = build_model(method, 6, checkpoint["level"], config, pretrained=False).to(device)
        label = method_label(method, config)
        seed = checkpoint["seed"]
        selected, epoch = args.checkpoint.stem, checkpoint["completed_epochs"]
    if "ema_state_dict" not in checkpoint:
        raise ValueError("EMA weights are required to match the original evaluation protocol")
    state = checkpoint["ema_state_dict"]
    if all(k.startswith("module.") for k in state):
        state = {k[7:]: v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    output = args.output or (Path(args.run_root) / "checkpoint_evaluations" / args.dataset / args.checkpoint.stem)
    if output.exists():
        raise FileExistsError(f"Evaluation output exists: {output}; choose a new --output")
    result = evaluate(model, loader, record["paths"]["val_annotations"], device, output, record,
                      {"method": method, "method_label": label, "seed": seed, "epoch": epoch,
                       "ugs_config": asdict(config) if method == "ugs_reimpl" else None,
                       "protocol_sha256": checkpoint.get("protocol_sha256"),
                       "selection": selected, "checkpoint": str(args.checkpoint),
                       "checkpoint_sha256": sha256(args.checkpoint)})
    print(json.dumps(result["metrics"], indent=2), flush=True)
    print(f"EVALUATION PASS {output}", flush=True)


def summarize(args):
    root = Path(args.run_root)
    results = []
    for p in sorted(root.glob(f"*/*/seed_*/{args.selection}.metrics.json")):
        r = json.loads(p.read_text())
        completed = p.parent / "completed.json"
        if not completed.is_file():
            print(f"SKIP incomplete run: {p.parent}", flush=True)
            continue
        protocol = json.loads((p.parent / "protocol.json").read_text())
        r["epochs_planned"] = protocol["epochs"]
        r["training_protocol"] = {k:v for k,v in protocol.items() if k not in {
            "method", "method_label", "seed", "protocol_sha256"}}
        results.append(r)
    if not results:
        raise RuntimeError("No completed selected-checkpoint results found")
    groups = {}
    for r in results:
        groups.setdefault((r["dataset"], r["method_label"]), []).append(r)
    output = args.output or root / "summary"
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for (dataset, method), items in groups.items():
        if len({r["seed"] for r in items}) != len(items):
            raise ValueError("Duplicate seeds in summary")
        if len({r["evaluation_signature"] for r in items}) != 1:
            raise ValueError("Evaluation settings differ across seeds")
        if len({signature(r["training_protocol"]) for r in items}) != 1:
            raise ValueError("Training settings differ across seeds")
        row = {"dataset": dataset, "method": method, "selection": args.selection,
               "seeds": ",".join(str(r["seed"]) for r in items), "n": len(items),
               "three_seed_grid_complete": {r["seed"] for r in items} == {42,1337,2025}}
        for metric in ["AP_small", "mAP", "AP50", "AP75", "AP_medium", "AP_large"]:
            values = [r["metrics"][metric] for r in items]
            valid = all(v is not None and v >= 0 for v in values)
            row[metric + "_mean"] = float(np.mean(values)) if valid else None
            row[metric + "_std"] = float(np.std(values, ddof=1)) if valid and len(values)>1 else None
        rows.append(row)
    # Methods sharing a dataset must use the same evaluation protocol.
    for dataset in {r["dataset"] for r in results}:
        if len({r["evaluation_signature"] for r in results if r["dataset"] == dataset}) != 1:
            raise ValueError(f"Incompatible evaluation protocols for {dataset}")
    write_json(output / "summary.json", rows)
    with open(output / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(json.dumps(rows, indent=2), flush=True)
    print(f"SUMMARY PASS {output}", flush=True)


def main():
    args = arguments()
    if args.output and args.action == "smoke":
        args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.action == "summarize":
        summarize(args)
        return
    core, config, device = configure(args)
    if args.action == "check":
        print(json.dumps(environment(), indent=2))
        print("ENVIRONMENT PASS")
    elif args.action == "audit":
        result = audit(args.dataset, args.data_root)
        output = args.output or Path(args.run_root) / "audits" / (args.dataset + ".json")
        write_json(output, result)
        print(f"DATA AUDIT PASS train={result['train']['image_count']} val={result['val']['image_count']} {output}")
    elif args.action == "smoke":
        smoke(args, core, config, device)
    elif args.action == "train":
        run_train(args, core, config, device)
    elif args.action == "evaluate":
        run_evaluate(args, config, device)


if __name__ == "__main__":
    main()
