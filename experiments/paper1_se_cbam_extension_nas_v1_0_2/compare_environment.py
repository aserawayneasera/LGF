#!/usr/bin/env python3
"""Show the exact runtime differences from an LGF/baseline reference protocol."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from run_se_cbam import (environment_differences, gpu_difference, import_at,
                         read_json, resolve_package)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--package", type=Path, required=True)
    args = parser.parse_args()

    reference = read_json(args.reference)
    package = resolve_package(args.package)
    utils = import_at("paper1_se_cbam_reference_utils", package / "utils.py")
    unsigned = {key: value for key, value in reference.items()
                if key not in {"method_label", "protocol_sha256"}}
    if utils.signature(unsigned) != reference.get("protocol_sha256"):
        parser.error("Reference protocol_sha256 is invalid; leave the saved file unchanged.")
    current = utils.environment()
    recorded = reference.get("environment")
    differences = environment_differences(recorded, current)
    print(json.dumps({"reference": str(args.reference), "recorded": recorded,
                      "current": current, "blocking_differences": differences,
                      "gpu_name_difference": gpu_difference(recorded, current)}, indent=2))
    if differences:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
