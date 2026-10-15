#!/usr/bin/env python3
"""Combine matching UGS and original-LGF evaluation reports, without inventing results."""
import argparse
import csv
import json
from pathlib import Path
import statistics
from utils import write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reports", nargs="+", required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    reports = [json.loads(Path(f).read_text()) for f in args.reports]
    if len({r["dataset"] for r in reports}) != 1 or len({r["evaluation_signature"] for r in reports}) != 1:
        raise ValueError("Comparison requires one dataset and identical validation files/evaluation settings")
    selection = {r.get("selection") for r in reports}
    if len(selection) != 1 or next(iter(selection)) not in {"best_ap_small", "best_map"}:
        raise ValueError("Resolve and verify a shared best_ap_small or best_map selection rule first")
    for r in reports:
        if not r.get("checkpoint_sha256"):
            raise ValueError("Use selected-checkpoint reports with recorded checkpoint hashes")
    groups = {}
    per_run = []
    for r in reports:
        groups.setdefault(r["method_label"], []).append(r)
        per_run.append({"method": r["method_label"], "seed": r["seed"], "epoch": r["epoch"],
                        "checkpoint": r["checkpoint"], "selection": r["selection"],
                        **{k:r["metrics"][k] for k in ["AP_small","mAP","AP50","AP75","AP_medium","AP_large"]}})
    seed_sets = []
    summary = []
    for method, items in groups.items():
        if len({json.dumps(r.get("ugs_config"),sort_keys=True) for r in items}) != 1:
            raise ValueError(f"UGS configurations differ across seeds for {method}")
        if any(r.get("method") == "ugs_reimpl" and not r.get("ugs_config") for r in items):
            raise ValueError("UGS report is missing its configuration")
        seeds = [r["seed"] for r in items]
        if len(set(seeds)) != len(seeds):
            raise ValueError(f"Duplicate seed for {method}")
        seed_sets.append(set(seeds))
        row = {"method":method,"n":len(items),"seeds":",".join(map(str,sorted(seeds)))}
        for metric in ["AP_small","mAP","AP50","AP75","AP_medium","AP_large"]:
            values = [r["metrics"][metric] for r in items]
            valid = all(v is not None and v>=0 for v in values)
            row[metric+"_mean"] = statistics.mean(values) if valid else None
            row[metric+"_std"] = statistics.stdev(values) if valid and len(values)>1 else None
        summary.append(row)
    if len(groups) < 2 or any(s != seed_sets[0] for s in seed_sets):
        raise ValueError("Supply at least two methods with matching seed sets")
    args.output.mkdir(parents=True,exist_ok=True)
    for name, rows in [("per_run",per_run),("comparison",summary)]:
        write_json(args.output/(name+".json"),rows)
        with open(args.output/(name+".csv"),"w",newline="") as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    write_json(args.output/"comparison_scope.json",{
        "dataset":reports[0]["dataset"],"selection":next(iter(selection)),
        "evaluation_signature":reports[0]["evaluation_signature"],
        "three_seed_grid_complete":seed_sets[0]=={42,1337,2025},
        "training_equivalence":"Not established by matching evaluation alone. Check training provenance and METHOD_NOTES.md.",
        "ugs_status":"Independent reimplementation with disclosed assumptions; not author code."})
    print(json.dumps(summary,indent=2))


if __name__=="__main__":
    main()
