#!/usr/bin/env python3
"""Read-only checkpoint profiling for Wayne's archived RetinaNet/LGF models."""
import argparse
import contextlib
import csv
import datetime as dt
import gc
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import re
import statistics
import subprocess
import sys
import time

# Set these before the frozen source imports torch or initializes CUDA.
# Its debugging default CUDA_LAUNCH_BLOCKING=1 would otherwise distort profiling.
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "model_source"
STAGES = {"0": "C3", "1": "C4", "2": "C5"}
CHANNELS = {"0": 512, "1": 1024, "2": 2048}
DATASETS = ("coco_nw", "coco_weather", "acdc")
PLACEMENTS = ("baseline", "C3", "C4", "C5", "C3_C4")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical_hash(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


def summary(values):
    values = sorted(float(x) for x in values)
    if not values:
        return None

    def percentile(q):
        p = (len(values) - 1) * q
        lo, hi = math.floor(p), math.ceil(p)
        return values[lo] + (values[hi] - values[lo]) * (p - lo)

    return {"n": len(values), "mean": statistics.mean(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0.0,
            "median": statistics.median(values), "p90": percentile(0.90),
            "p95": percentile(0.95), "min": values[0], "max": values[-1]}


def discover(args):
    entries = []
    for dataset in args.datasets:
        for placement in args.placements:
            method = "baseline" if placement == "baseline" else "lgf_original"
            folder = "runs" if placement in ("baseline", "C3") else "runs_" + placement
            for seed in args.seeds:
                run = args.runs_base / folder / dataset / method / ("seed_" + str(seed))
                entries.append({"dataset": dataset, "placement": placement, "seed": seed,
                                "checkpoint": str(run / (args.selection + ".pth")),
                                "metrics": str(run / (args.selection + ".metrics.json"))})
    return entries


def load_entries(args):
    if args.manifest:
        doc = read_json(args.manifest)
        entries = doc["checkpoints"] if isinstance(doc, dict) else doc
        if not isinstance(entries, list):
            raise ValueError("Manifest must contain a checkpoints list")
        entries = [dict(e) for e in entries if e["dataset"] in args.datasets
                   and e["placement"] in args.placements and int(e["seed"]) in args.seeds]
    else:
        entries = discover(args)
    seen = set()
    for e in entries:
        if e["dataset"] not in DATASETS or e["placement"] not in PLACEMENTS:
            raise ValueError("Unsupported manifest dataset/placement")
        identity = (e["dataset"], e["placement"], int(e["seed"]))
        if identity in seen:
            raise ValueError("Duplicate manifest entry: " + str(identity))
        seen.add(identity)
        if not Path(e["checkpoint"]).is_absolute():
            raise ValueError("Checkpoint paths must be absolute")
    if not entries:
        raise ValueError("No checkpoint entries match the requested filters")
    return entries


def import_runtime():
    import torch
    import torchvision
    from fvcore.nn import FlopCountAnalysis
    if not torch.__version__.startswith("2.10.") or not torchvision.__version__.startswith("0.25."):
        raise RuntimeError("Use the training environment: torch 2.10.x / torchvision 0.25.x")
    sys.path.insert(0, str(SOURCE))
    import core_bridge
    return torch, torchvision, FlopCountAnalysis, core_bridge


def normalize_state(state):
    if not isinstance(state, dict) or not state:
        raise ValueError("Missing or empty EMA state dictionary")
    keys = list(state)
    prefixed = [k.startswith("module.") for k in keys]
    if any(prefixed) and not all(prefixed):
        raise ValueError("Mixed module. prefixes in state dictionary")
    return {k[7:] if all(prefixed) else k: v for k, v in state.items()}


def checkpoint_keys(state):
    return sorted({m.group(1) for k in state
                   for m in [re.match(r"^backbone\.block_fpn_in\.([^.]+)\.", k)] if m})


@contextlib.contextmanager
def saved_backbone_keys(core, keys):
    """Reconstruct the saved module schema, without translating its old label."""
    original = core.CustomBackboneBlockBeforeFPN

    class SavedBackbone(original):
        def __init__(self, backbone_with_fpn, selected_levels):
            super().__init__(backbone_with_fpn, selected_levels=list(keys))
            if list(self.block_fpn_in) != list(keys):
                raise ValueError("Could not reconstruct all saved feature blocks")

    core.CustomBackboneBlockBeforeFPN = SavedBackbone
    try:
        yield
    finally:
        core.CustomBackboneBlockBeforeFPN = original


def reconstruct(ck, state, bridge):
    import torch
    keys = checkpoint_keys(state)
    if any(k not in STAGES for k in keys):
        raise ValueError("Unknown backbone feature keys: " + str(keys))
    core = bridge.load_core()
    config = ck.get("config")
    if not isinstance(config, dict):
        config = dict(core.CONFIG_MAP["lgf_gated_spatial" if keys else "baseline"])
    if keys and (config.get("block_type"), config.get("gating_type"),
                 config.get("BRANCH_PRESET")) != ("lgf", "gated_spatial", "local_global"):
        raise ValueError("This profiler supports the supplied spatial-gated LGF architecture")
    if not keys:
        config = dict(core.CONFIG_MAP["baseline"])
    logits = state.get("head.classification_head.cls_logits.weight")
    if logits is None or int(logits.shape[0]) != 54:
        raise ValueError("Expected the six-class, nine-anchor RetinaNet head")
    with saved_backbone_keys(core, keys):
        model = bridge.build_model("lgf_original" if keys else "baseline", 6,
                                   level="C3", pretrained=False, original_config=config,
                                   smoke_size=64)
    # smoke_size only speeds up the constructor's anchor-shape probe.
    model.transform.min_size, model.transform.max_size = (800,), 1333
    model.load_state_dict(state, strict=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    details = []
    blocks = getattr(model.backbone, "block_fpn_in", {})
    mapping = model.backbone.body.return_layers
    reverse = {str(value): stage for stage, value in mapping.items()}
    for key in keys:
        block = blocks[key]
        channels = block.local[0].in_channels
        if channels != CHANNELS[key]:
            raise ValueError("Checkpoint channel count disagrees with the returned ResNet stage")
        block.enable_visualization(False)
        details.append({"feature_key": key, "semantic_level": STAGES[key],
                        "resnet_stage": reverse[key], "channels": channels,
                        "parameters": sum(p.numel() for p in block.parameters())})
    return model, details


def evaluation_settings(model):
    return {"min_size": list(model.transform.min_size), "max_size": model.transform.max_size,
            "mean": list(model.transform.image_mean), "std": list(model.transform.image_std),
            "score_thresh": model.score_thresh, "nms_thresh": model.nms_thresh,
            "topk_candidates": model.topk_candidates, "detections_per_img": model.detections_per_img,
            "anchor_sizes": [list(x) for x in model.anchor_generator.sizes],
            "anchor_ratios": [list(x) for x in model.anchor_generator.aspect_ratios],
            "inference_precision": "FP32", "weights": "EMA"}


def validate_protocol(model, ck, metrics):
    actual = evaluation_settings(model)
    expected = metrics.get("evaluation_protocol") or ck.get("protocol", {}).get("model_protocol")
    if expected:
        for key in actual:
            if key in expected and expected[key] != actual[key]:
                raise ValueError("Inference protocol mismatch for " + key)
    return actual


def counts(model, height, width, FlopCountAnalysis):
    """Neural graph only. NMS and detection decoding are included in timing."""
    import torch
    from torch import nn

    class DenseGraph(nn.Module):
        def __init__(self, detector):
            super().__init__()
            self.backbone, self.head = detector.backbone, detector.head

        def forward(self, tensor):
            features = list(self.backbone(tensor).values())
            heads = self.head(features)
            return heads["cls_logits"], heads["bbox_regression"]

    shapes, per_module, handles = {}, {}, []
    image = torch.zeros((3, height, width), dtype=torch.float32)
    with torch.inference_mode():
        transformed, _ = model.transform([image], None)
    graph = DenseGraph(model).eval()

    def hook(name):
        def count(module, inputs, output):
            if isinstance(module, nn.Conv2d):
                macs = output.numel() * (module.in_channels // module.groups) * math.prod(module.kernel_size)
            else:
                macs = output.numel() * module.in_features
            per_module[name] = per_module.get(name, 0) + int(macs)
        return count

    for name, module in graph.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            handles.append(module.register_forward_hook(hook(name)))
    blocks = getattr(model.backbone, "block_fpn_in", {})
    for key, block in blocks.items():
        def shape_hook(module, inputs, output, key=key):
            shapes[key] = list(inputs[0].shape)
        handles.append(block.register_forward_hook(shape_hook))
    try:
        with torch.inference_mode():
            graph(transformed.tensors)
    finally:
        for h in handles:
            h.remove()
    with torch.inference_mode():
        analysis = FlopCountAnalysis(graph, (transformed.tensors,))
        analysis.unsupported_ops_warnings(False).uncalled_modules_warnings(False).tracer_warnings("none")
        total = int(analysis.total())
        operations = {k: int(v) for k, v in analysis.by_operator().items()}
        unsupported = dict(analysis.unsupported_ops())
    macs = sum(per_module.values())
    fvcore_macs = sum(operations.get(k, 0) for k in ("conv", "linear"))
    if macs != fvcore_macs:
        raise ValueError("Hook convolution/linear MAC count disagrees with fvcore")
    lgf_macs = sum(v for k, v in per_module.items() if k.startswith("backbone.block_fpn_in."))
    params = sum(p.numel() for p in model.parameters())
    added = sum(p.numel() for b in blocks.values() for p in b.parameters())
    return {"params_total": params, "params_lgf": added, "params_without_lgf": params - added,
            "reference_input_chw": [3, height, width], "padded_input_nchw": list(transformed.tensors.shape),
            "lgf_input_shapes": shapes, "conv_linear_MACs": macs,
            "conv_linear_FLOPs_2_per_MAC": 2 * macs, "lgf_conv_MACs": lgf_macs,
            "fvcore_supported_FLOPs_FMA1": total, "fvcore_by_operator": operations,
            "fvcore_unsupported_ops": unsupported, "conv_linear_MACs_by_module": per_module,
            "flop_scope": "backbone + LGF + FPN + detection head; excludes image transform, anchors, decoding, NMS",
            "flop_convention": "fvcore: one fused multiply-add is one FLOP; separate conv/linear FLOPs use two per MAC"}


def dataset_paths(dataset, root):
    if dataset in ("coco_nw", "coco_weather"):
        tag = "non_weather" if dataset == "coco_nw" else "weather"
        folder = root / "COCO" / ("non_weather-mini" if dataset == "coco_nw" else "weather-mini")
        return (folder / ("images/val2017_" + tag + "-500k6c"),
                folder / ("annotations/mini_val2017_" + tag + "-500k6c.json"))
    folder = root / "ACDC-1/ACDC-1-NEW"
    return folder / "images/val", folder / "annotations/mini_val.json"


def prepare_samples(dataset, root, n, sample_seed):
    import torch
    import cv2
    cv2.setNumThreads(0)
    image_root, annotations = dataset_paths(dataset, root)
    doc = read_json(annotations)
    images = sorted(doc["images"], key=lambda x: x["id"])
    if len({x["id"] for x in images}) != len(images):
        raise ValueError("Duplicate validation image IDs")
    if n > len(images):
        raise ValueError("sample-images exceeds available validation images")
    rng = random.Random(sample_seed)
    selected = rng.sample(images, n)
    tensors, records = [], []
    for info in selected:
        path = image_root / info["file_name"]
        array = cv2.imread(str(path))
        if array is None:
            raise OSError("Unable to decode " + str(path))
        array = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous().float().div_(255.0)
        tensors.append(tensor)
        records.append({"image_id": info["id"], "file_name": info["file_name"],
                        "sha256": sha256(path), "chw": list(tensor.shape)})
    provenance = {"dataset": dataset, "annotations": str(annotations), "image_root": str(image_root),
                  "annotations_sha256": sha256(annotations), "sample_seed": sample_seed,
                  "image_count": len(tensors), "images": records}
    provenance["sample_sha256"] = canonical_hash(provenance)
    return tensors, provenance


def validate_samples(provenance, run):
    audit_path = Path(run) / "data_audit.json"
    if not audit_path.is_file():
        return "recorded sample hashes; run data_audit.json unavailable"
    audit = read_json(audit_path)["val"]
    if audit["annotations_sha256"] != provenance["annotations_sha256"]:
        raise ValueError("Validation annotations differ from the training run's audit")
    expected = {x["image_id"]: x for x in audit["image_records"]}
    for image in provenance["images"]:
        old = expected.get(image["image_id"])
        if not old or old["file_name"] != image["file_name"] or old["sha256"] != image["sha256"]:
            raise ValueError("Timing sample differs from the run's audited validation image")
    return "sample images and annotations match the run's data audit"


def timing(model, tensors, device, args):
    import torch
    results = []
    is_cuda = device.type == "cuda"

    def sync():
        if is_cuda:
            torch.cuda.synchronize(device)

    for batch_size in args.batches:
        if batch_size > len(tensors):
            raise ValueError("Batch size exceeds timing sample count")
        warm = [x.to(device) for x in tensors[:batch_size]]
        with torch.inference_mode():
            for _ in range(args.warmup):
                output = model(warm)
                del output
        sync()
        del warm
        if is_cuda:
            torch.cuda.reset_peak_memory_stats(device)
        samples, round_stats = [], []
        start_event = torch.cuda.Event(enable_timing=True) if is_cuda else None
        end_event = torch.cuda.Event(enable_timing=True) if is_cuda else None
        with torch.inference_mode():
            for repeat in range(args.repeats):
                indices = list(range(len(tensors)))
                random.Random(args.sample_seed + repeat).shuffle(indices)
                usable = len(indices) - len(indices) % batch_size
                round_values = []
                for begin in range(0, usable, batch_size):
                    chosen = indices[begin:begin + batch_size]
                    batch = [tensors[i].to(device) for i in chosen]
                    sync()  # Host-to-device copies finish before the timer starts.
                    if is_cuda:
                        start_event.record(torch.cuda.current_stream(device))
                    t0 = time.perf_counter()
                    output = model(batch)
                    if is_cuda:
                        end_event.record(torch.cuda.current_stream(device))
                    sync()
                    wall_ms = (time.perf_counter() - t0) * 1000.0
                    event_ms = start_event.elapsed_time(end_event) if is_cuda else None
                    samples.append({"repeat": repeat + 1, "sample_indices": chosen,
                                    "wall_batch_ms": wall_ms, "wall_per_image_ms": wall_ms / batch_size,
                                    "cuda_event_batch_ms": event_ms,
                                    "detection_counts": [len(x["boxes"]) for x in output]})
                    round_values.append(wall_ms / batch_size)
                    del output, batch
                round_stats.append(summary(round_values))
        per_image = summary([x["wall_per_image_ms"] for x in samples])
        results.append({"batch_size": batch_size,
                        "timing_kind": "batch-one detector latency" if batch_size == 1 else "amortized time per image",
                        "wall_per_image_ms": per_image,
                        "wall_batch_ms": summary([x["wall_batch_ms"] for x in samples]),
                        "cuda_event_batch_ms": summary([x["cuda_event_batch_ms"] for x in samples]) if is_cuda else None,
                        "round_wall_per_image_ms": round_stats,
                        "images_per_second": 1000.0 / per_image["mean"],
                        "peak_allocated_MiB": torch.cuda.max_memory_allocated(device) / 2**20 if is_cuda else None,
                        "peak_reserved_MiB": torch.cuda.max_memory_reserved(device) / 2**20 if is_cuda else None,
                        "warmup_iterations": args.warmup, "samples": samples,
                        "scope": "model(list_of_device_tensors): normalization, resize, padding, backbone, LGF, FPN, head, anchors, decoding, NMS, output rescaling",
                        "excluded": "disk reads, image decoding, CPU tensor preparation, host-to-device copies, COCO evaluation, serialization"})
        print("  batch", batch_size, "mean ms/image", round(per_image["mean"], 4), flush=True)
    return results


def gpu_snapshot():
    try:
        return subprocess.run(["nvidia-smi", "--query-gpu=index,name,driver_version,memory.used,utilization.gpu,temperature.gpu,power.draw",
                               "--format=csv,noheader"], capture_output=True, text=True, timeout=5).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def environment(torch, torchvision, device):
    packages = {}
    for name in ("fvcore", "numpy", "albumentations", "opencv-python-headless", "pycocotools"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {"python": sys.version, "torch": torch.__version__, "torchvision": torchvision.__version__,
            "packages": packages, "device": str(device), "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_tf32": torch.backends.cudnn.allow_tf32,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "CUDA_LAUNCH_BLOCKING": os.environ["CUDA_LAUNCH_BLOCKING"],
            "cpu_threads": torch.get_num_threads(), "nvidia_smi_at_start": gpu_snapshot()}


def flat_rows(reports):
    rows = []
    for r in reports:
        if r.get("status") != "ok":
            continue
        count = r["counts"]
        resolved = r["resolved_placement"]
        common = {"dataset": r["dataset"], "saved_run_label": r["placement"],
                  "resolved_placement": resolved, "seed": r["seed"],
                  "selection": r["selection"], "checkpoint": r["checkpoint"],
                  "params_M": count["params_total"] / 1e6,
                  "LGF_params_M": count["params_lgf"] / 1e6,
                  "fvcore_supported_GFLOPs_FMA1": count["fvcore_supported_FLOPs_FMA1"] / 1e9,
                  "conv_linear_GMACs": count["conv_linear_MACs"] / 1e9,
                  "conv_linear_GFLOPs_2_per_MAC": count["conv_linear_FLOPs_2_per_MAC"] / 1e9,
                  "validation_APsmall": r["validation_metrics"].get("AP_small"),
                  "validation_mAP": r["validation_metrics"].get("mAP"),
                  "reference_padded_shape": "x".join(map(str, count["padded_input_nchw"])),
                  "unprofiled_operators": json.dumps(count["fvcore_unsupported_ops"], sort_keys=True)}
        for t in r["timing"] or [None]:
            row = dict(common)
            row.update({"batch_size": t["batch_size"] if t else None,
                        "mean_ms_per_image": t["wall_per_image_ms"]["mean"] if t else None,
                        "median_ms_per_image": t["wall_per_image_ms"]["median"] if t else None,
                        "p95_ms_per_image": t["wall_per_image_ms"]["p95"] if t else None,
                        "timed_batches": t["wall_per_image_ms"]["n"] if t else 0,
                        "images_per_second": t["images_per_second"] if t else None,
                        "peak_allocated_MiB": t["peak_allocated_MiB"] if t else None})
            rows.append(row)
    # Use a profiled baseline from the same dataset, seed, input and timing batch.
    baselines = {(x["dataset"], x["seed"], x["batch_size"]): x for x in rows
                 if x["saved_run_label"] == "baseline" and x["resolved_placement"] == "none"}
    for row in rows:
        baseline = baselines.get((row["dataset"], row["seed"], row["batch_size"]))
        row["params_overhead_pct"] = None
        row["conv_MACs_overhead_pct"] = None
        row["time_overhead_pct"] = None
        if baseline:
            row["params_overhead_pct"] = 100 * (row["params_M"] / baseline["params_M"] - 1)
            row["conv_MACs_overhead_pct"] = 100 * (row["conv_linear_GMACs"] / baseline["conv_linear_GMACs"] - 1)
            if row["mean_ms_per_image"] is not None:
                row["time_overhead_pct"] = 100 * (row["mean_ms_per_image"] / baseline["mean_ms_per_image"] - 1)
    return rows


def save_tables(output, reports):
    rows = flat_rows(reports)
    if not rows:
        return
    with (output / "placement_profile_per_seed.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    groups = {}
    for row in rows:
        key = (row["dataset"], row["saved_run_label"], row["resolved_placement"], row["batch_size"])
        groups.setdefault(key, []).append(row)
    grouped = []
    for key, members in groups.items():
        result = dict(zip(("dataset", "saved_run_label", "resolved_placement", "batch_size"), key))
        result.update({"n_seeds": len(members), "seeds": " ".join(str(m["seed"]) for m in members)})
        for name in ("params_M", "LGF_params_M", "fvcore_supported_GFLOPs_FMA1",
                     "conv_linear_GMACs", "conv_linear_GFLOPs_2_per_MAC", "validation_APsmall",
                     "validation_mAP", "mean_ms_per_image", "p95_ms_per_image", "images_per_second",
                     "params_overhead_pct", "conv_MACs_overhead_pct", "time_overhead_pct"):
            vals = [x[name] for x in members if x[name] is not None]
            s = summary(vals)
            result[name + "_mean"] = s["mean"] if s else None
            result[name + "_std_across_seeds"] = s["std"] if s else None
        grouped.append(result)
    with (output / "placement_profile_mean_std.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(grouped[0]))
        w.writeheader()
        w.writerows(grouped)


def smoke(args, torch, FlopCountAnalysis, bridge):
    """Exercise strict checkpoint reconstruction using random weights, no training."""
    core = bridge.load_core()
    core.set_seed(123)
    outcome = {"status": "synthetic smoke test, not experimental results", "models": []}
    for keys in ([], ["0"], ["1"], ["2"], ["0", "1"]):
        cfg = dict(core.CONFIG_MAP["lgf_gated_spatial" if keys else "baseline"])
        with saved_backbone_keys(core, keys):
            initial = bridge.build_model("lgf_original" if keys else "baseline", 6,
                                         pretrained=False, original_config=cfg, smoke_size=64)
        state = initial.state_dict()
        model, details = reconstruct({"config": cfg}, state, bridge)
        del initial, state
        model.transform.min_size, model.transform.max_size = (64,), 64
        count = counts(model, 64, 64, FlopCountAnalysis)
        for block in details:
            c = block["channels"]
            assert block["parameters"] == 11 * c * c // 4 + 19 * c + 2
        # Exercise box decoding and NMS with untrained synthetic inputs.
        # Real checkpoint profiling retains the recorded score/top-k settings.
        model.score_thresh, model.topk_candidates = 0.0, 12
        small = [torch.rand(3, 64, 64), torch.rand(3, 64, 64)]
        small_args = argparse.Namespace(batches=[1, 2], warmup=1, repeats=1, sample_seed=123)
        measured = timing(model, small, torch.device("cpu"), small_args)
        assert all(any(n > 0 for n in sample["detection_counts"])
                   for t in measured for sample in t["samples"])
        outcome["models"].append({"keys": keys, "blocks": details, "counts": count,
                                  "smoke_timing_only": measured})
        del model
        gc.collect()
    write_json(args.output / "smoke_test.json", outcome)
    print("SMOKE PASS: strict saved-state reload, all four placements, counts, inference at batches 1 and 2.")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-base", type=Path, default=Path("/nas.dbms/asera/paper1_original/ugs_comparison"))
    parser.add_argument("--data-root", type=Path, default=Path("/nas.dbms/asera/paper1_original/DATASET"))
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--placements", nargs="+", choices=PLACEMENTS, default=list(PLACEMENTS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 1337, 2025])
    parser.add_argument("--selection", choices=["best_ap_small", "best_map"], default="best_ap_small")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batches", nargs="+", type=int, default=[1, 16])
    parser.add_argument("--sample-images", type=int, default=112)
    parser.add_argument("--sample-seed", type=int, default=731)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--reference-height", type=int, default=800)
    parser.add_argument("--reference-width", type=int, default=1333)
    parser.add_argument("--counts-only", action="store_true")
    parser.add_argument("--inventory-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    for name in ("sample_images", "warmup", "repeats", "reference_height", "reference_width"):
        if getattr(args, name) < 1:
            parser.error(name + " must be positive")
    if any(b < 1 for b in args.batches):
        parser.error("batch sizes must be positive")
    if len(set(args.batches)) != len(args.batches):
        parser.error("duplicate batch sizes")
    args.output = args.output or args.runs_base / ("placement_profile_v1_" + dt.datetime.now().strftime("%Y%m%d_%H%M%S"))
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Output directory must be new or empty; existing profiling results are preserved")
    args.output.mkdir(parents=True, exist_ok=True)
    entries = load_entries(args)
    inventory = [{**e, "checkpoint_exists": Path(e["checkpoint"]).is_file(),
                  "metrics_exist": bool(e.get("metrics") and Path(e["metrics"]).is_file())} for e in entries]
    write_json(args.output / "inventory.json", inventory)
    if args.inventory_only:
        for e in inventory:
            print("FOUND" if e["checkpoint_exists"] else "MISSING", e["checkpoint"])
        return 0
    torch, torchvision, FlopCountAnalysis, bridge = import_runtime()
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    device = torch.device(args.device)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("Supported devices are cpu and cuda")
    if device.type == "cuda" and not torch.cuda.is_available() and not args.smoke:
        raise RuntimeError("CUDA unavailable. This does not substitute CPU measurements for GPU latency.")
    if device.type == "cuda" and not args.smoke:
        torch.cuda.set_device(device)
    bridge.load_core().set_seed(args.sample_seed)
    if args.smoke:
        return smoke(args, torch, FlopCountAnalysis, bridge)
    env = environment(torch, torchvision, device)
    meta = {"created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            "environment": env,
            "source_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in sorted(SOURCE.rglob("*")) if p.is_file() and "__pycache__" not in str(p)},
            "profiler_sha256": sha256(__file__),
            "AP_status": "neighboring selected-checkpoint validation metrics; not new test evaluation or automatically the retained manuscript reference",
            "seed_std_status": "sample standard deviation across checkpoint seeds; not an uncertainty interval for wall-time samples"}
    write_json(args.output / "profiling_protocol.json", meta)
    reports, samples_cache, errors = [], {}, 0
    # Seeded model order reduces dependence on a fixed placement order.
    work = inventory[:]
    random.Random(args.sample_seed).shuffle(work)
    for i, entry in enumerate(work):
        print(f"[{i+1}/{len(work)}] {entry['dataset']} {entry['placement']} seed {entry['seed']}", flush=True)
        report = dict(entry)
        model = None
        ck, state = None, None
        if not entry["checkpoint_exists"]:
            report.update({"status": "missing_checkpoint", "reason": "not profiled, no replacement model is trained"})
        else:
            try:
                cp = Path(entry["checkpoint"])
                digest = sha256(cp)
                metrics = read_json(entry["metrics"]) if entry["metrics_exist"] else {}
                if metrics.get("checkpoint_sha256") and metrics["checkpoint_sha256"] != digest:
                    raise ValueError("Selected metrics reference a different checkpoint SHA256")
                ck = torch.load(cp, map_location="cpu", weights_only=False)
                if not isinstance(ck, dict) or "ema_state_dict" not in ck:
                    raise ValueError("An EMA checkpoint is required; model weights are not silently substituted")
                for key in ("dataset", "seed"):
                    if key in ck and ck[key] != entry[key]:
                        raise ValueError("Manifest differs from checkpoint " + key)
                for key in ("dataset", "seed"):
                    if key in metrics and metrics[key] != entry[key]:
                        raise ValueError("Manifest differs from selected metrics " + key)
                if metrics.get("selection") and metrics["selection"] != args.selection:
                    raise ValueError("Metrics selection differs from --selection")
                state = normalize_state(ck["ema_state_dict"])
                model, details = reconstruct(ck, state, bridge)
                settings = validate_protocol(model, ck, metrics)
                resolved = "_".join(x["semantic_level"] for x in details) or "none"
                report.update({"checkpoint_sha256": digest, "weights": "EMA", "selection": metrics.get("selection", args.selection),
                               "checkpoint_level_label": ck.get("level", ck.get("insert_level")),
                               "checkpoint_completed_epochs": ck.get("completed_epochs", ck.get("epoch")),
                               "selected_epoch": metrics.get("epoch"), "resolved_placement": resolved,
                               "saved_feature_blocks": details, "evaluation_settings": settings,
                               "validation_metrics": metrics.get("metrics", {}),
                               "accuracy_source": entry.get("metrics") if metrics else None,
                               "counts": counts(model, args.reference_height, args.reference_width, FlopCountAnalysis)})
                state, ck = None, None
                gc.collect()
                if args.counts_only:
                    report["timing"] = []
                else:
                    dataset = entry["dataset"]
                    if dataset not in samples_cache:
                        samples_cache[dataset] = prepare_samples(dataset, args.data_root, args.sample_images, args.sample_seed)
                        write_json(args.output / ("timing_images_" + dataset + ".json"), samples_cache[dataset][1])
                    tensors, provenance = samples_cache[dataset]
                    report["sample_data_validation"] = validate_samples(provenance, cp.parent)
                    report["timing_sample_sha256"] = provenance["sample_sha256"]
                    report["gpu_before_timing"] = gpu_snapshot()
                    model = model.to(device)
                    report["timing"] = timing(model, tensors, device, args)
                    report["gpu_after_timing"] = gpu_snapshot()
                report["status"] = "ok"
            except Exception as exc:
                errors += 1
                report.update({"status": "error", "error_type": type(exc).__name__, "error": str(exc)})
                print("  ERROR:", type(exc).__name__, str(exc), flush=True)
            finally:
                model, ck, state = None, None, None
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        reports.append(report)
        name = entry["dataset"] + "_" + entry["placement"] + "_seed" + str(entry["seed"]) + ".json"
        write_json(args.output / "per_checkpoint" / name, report)
        write_json(args.output / "all_profiles.json", reports)
        save_tables(args.output, reports)
    completed = sum(r.get("status") == "ok" for r in reports)
    final = {"profiled": completed, "errors": errors,
             "missing_checkpoints": sum(r.get("status") == "missing_checkpoint" for r in reports),
             "output": str(args.output), "environment": env}
    write_json(args.output / "completion.json", final)
    archive = Path(str(args.output) + ".zip")
    import zipfile
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(args.output.rglob("*")):
            if p.is_file():
                z.write(p, arcname=str(Path(args.output.name) / p.relative_to(args.output)))
    print(json.dumps(final, indent=2), flush=True)
    print("RESULTS ZIP:", archive, flush=True)
    return 1 if errors or not completed else 0


if __name__ == "__main__":
    raise SystemExit(main())
