# Paired hierarchical APsmall bootstrap

The recovered `paper1_paired_bootstrap(2).py` is included byte for byte under the
portable filename `paper1_paired_bootstrap.py`. Version: 1.0.1.

This program reads selected-checkpoint prediction exports, recomputes COCO
APsmall, and resamples images and matched training-seed pairs together. It does
not load detector weights or train a detector.

Install the CPU requirements, then run from the repository root:

```bash
python -m pip install -r experiments/paper1_paired_bootstrap/requirements.txt
python experiments/paper1_paired_bootstrap/paper1_paired_bootstrap.py --help
python -m unittest discover -s experiments/paper1_paired_bootstrap/tests -v
```

See [diagnostics.md](../../docs/diagnostics.md#paired-bootstrap) for full commands,
input provenance, checkpoint selection, units, and interpretation.

Input annotation IDs must be strictly positive. COCOeval uses ID zero as the
unmatched sentinel. The preserved historical bootstrap does not normalize zero
IDs. Keep the annotation document supporting the saved AP unchanged. When a
document contains ID zero, use a separately versioned, validated correction
rather than editing an existing run's files or describing the historical result
as reproduced.

The `tests/` fixtures are synthetic software checks. They are not paper results.
