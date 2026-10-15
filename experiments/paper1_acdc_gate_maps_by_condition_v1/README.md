> Repository integration: use [diagnostics.md](../../docs/diagnostics.md) for portable commands and input provenance. The instructions below describe the recovered historical server workflow. Scientific Python/model sources retain their original bytes. Shell launchers use repository-relative paths and documented environment overrides.

# ACDC LGF gate maps by real weather condition

The reviewer requested local/global gate maps for clear, fog, rain and snow.
The separate COCO bundle already addresses that exact request. ACDC has no
clear subset in this validation dataset; the audited ACDC conditions are fog,
rain, snow and night. This bundle produces optional supplementary ACDC figures.

Copy the four files `paper1_acdc_gate_visualization.py`,
`paper1_gate_visualization_conditions_v2.py`, `acdc_image_conditions.csv`, and
`run_gate_maps_acdc.sh` into `/nas.dbms/asera/`, then run:

```bash
bash /nas.dbms/asera/run_gate_maps_acdc.sh
```

Only original ACDC validation frames are eligible. The first pass checks
condition labels, small GT objects and image paths without loading a model.
The next pass uses one saved ACDC LGF EMA checkpoint for two images per
condition. Each image gets a full panel and a crop around one small object.
The local/global colour scale is fixed at 0 to 1. The scenes are different
real ACDC images, unlike the paired synthetic COCO scenes.

The output is:

`/nas.dbms/asera/paper1_original/gate_visualization/acdc_by_condition_seed42`

If your Python is located elsewhere, override `P1_UGS_PY` when invoking the
wrapper. ACDC figures are for interpretation; the previous control experiment
provides the performance evidence for the gate's contribution.
