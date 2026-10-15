> Repository integration: use [diagnostics.md](../../docs/diagnostics.md) for portable commands and input provenance. The instructions below describe the recovered historical server workflow. Scientific Python/model sources retain their original bytes. Shell launchers use repository-relative paths and documented environment overrides.

# Paired LGF gate figures for Reviewer 4, comment 10 (v2)

The earlier script failed because the default COCO-weather validation JSON
contains 500 fog images and no rain or snow records. This version uses the
uploaded weather evaluation's audited `image_conditions.csv` to identify the
matching rain and snow files and their image IDs. It reuses ground-truth boxes
from the fog version of each source scene for visualization. The weather
generation functions change image appearance without resizing or moving
objects. The script checks that the selected files exist, the same small box
is present in clear and fog annotations, and all four image dimensions match.

Copy all three supplied files to `/nas.dbms/asera/` and run:

```bash
bash /nas.dbms/asera/run_gate_maps_conditions_v2.sh
```

The wrapper first checks the pairing without loading a model. It then uses
the COCO-weather LGF seed-42 `best_ap_small.pth` EMA weights for clear, fog,
rain and snow, keeping the model fixed. Four source scenes are chosen using
selection seed 42. Each scene produces a full panel and a small-object crop.
The local and global maps share a 0-to-1 color scale. The manifest records
which weather boxes came from an annotated sibling scene.

Results go to:

`/nas.dbms/asera/paper1_original/gate_visualization/reviewer4_paired_seed42_v2`

`target_gate_stats.csv` provides descriptive gate means for the highlighted
small object. It does not test whether the gate improves detection accuracy.
Use the existing gate control experiment and computational analysis for that
separate reviewer request. ACDC figures are optional for comment 10.

If your Python environment is elsewhere, override it:

```bash
P1_UGS_PY=/path/to/your/working/python bash /nas.dbms/asera/run_gate_maps_conditions_v2.sh
```

If the rain or snow files are missing from the image directory, the first
check reports the missing file count and an example path. The script does not
create new weather images.
