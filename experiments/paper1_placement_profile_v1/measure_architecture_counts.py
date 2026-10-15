#!/usr/bin/env python3
"""Weight-independent reference counts for verified pre-FPN architectures."""
import argparse
import csv
import gc
from pathlib import Path
import profile_placements as p


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Use a new or empty output directory")
    args.output.mkdir(parents=True, exist_ok=True)
    torch, torchvision, FlopCountAnalysis, bridge = p.import_runtime()
    torch.set_num_threads(min(4, p.os.cpu_count() or 1))
    core = bridge.load_core()
    core.set_seed(731)
    report = {"status": "architecture reference only; no trained checkpoints or latency measurements",
              "definition": "Correct semantic pre-FPN C3=512, C4=1024, C5=2048 channels",
              "input_chw": [3, 800, 1333], "torch": torch.__version__, "torchvision": torchvision.__version__,
              "source_sha256": p.sha256(p.SOURCE / "vendor/original_core.py"), "models": []}
    rows = []
    for label, keys in (("baseline", []), ("C3", ["0"]), ("C4", ["1"]), ("C5", ["2"]), ("C3+C4", ["0", "1"])):
        cfg = dict(core.CONFIG_MAP["lgf_gated_spatial" if keys else "baseline"])
        with p.saved_backbone_keys(core, keys):
            model = bridge.build_model("lgf_original" if keys else "baseline", 6,
                                       pretrained=False, original_config=cfg, smoke_size=64)
        model.transform.min_size, model.transform.max_size = (800,), 1333
        model.eval()
        for param in model.parameters():
            param.requires_grad_(False)
        count = p.counts(model, 800, 1333, FlopCountAnalysis)
        report["models"].append({"architecture": label, "feature_keys": keys, "counts": count})
        rows.append({"architecture": label, "params_M": count["params_total"] / 1e6,
                     "LGF_params_M": count["params_lgf"] / 1e6,
                     "fvcore_supported_GFLOPs_FMA1": count["fvcore_supported_FLOPs_FMA1"] / 1e9,
                     "conv_linear_GMACs": count["conv_linear_MACs"] / 1e9,
                     "conv_linear_GFLOPs_2_per_MAC": count["conv_linear_FLOPs_2_per_MAC"] / 1e9,
                     "input_padded_NCHW": "x".join(map(str, count["padded_input_nchw"])),
                     "status": "architecture reference; not measured from the trained run"})
        print(label, rows[-1], flush=True)
        del model
        gc.collect()
    p.write_json(args.output / "architecture_reference_counts.json", report)
    with (args.output / "architecture_reference_counts.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
