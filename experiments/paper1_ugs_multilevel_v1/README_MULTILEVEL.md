# LGF Original: simultaneous C3+C4 and C3+C4+C5 pre-FPN training

This standalone package is derived from the archived `paper1_ugs_comparison_v1`.
The byte-exact `vendor/original_core.py`, the LGF block implementation, train/evaluation logic, and original data reader are preserved. `run.py` and `core_bridge.py` have narrowly scoped changes; `multilevel.py` implements verified multiple pre-FPN insertions. This is a **new architecture for fresh training**, not a change to existing checkpoints.

## Installation on the server

```bash
cd /nas.dbms/asera/NEW
unzip -q paper1_ugs_multilevel_v1.zip
cd paper1_ugs_multilevel_v1
sha256sum -c CHECKSUMS.sha256

P1_UGS_PY=/nas.dbms/asera/paper1_ugs_env/bin/python
"$P1_UGS_PY" -u run.py smoke --synthetic --device cuda --method lgf_original --level C3_C4
"$P1_UGS_PY" -u run.py smoke --synthetic --device cuda --method lgf_original --level C3_C4_C5
```

If the GPU is unavailable for the smoke test, use `--device cpu`.

## Run experiments

By default this runs 18 experiments: 2 placements x 3 datasets x 3 seeds.

```bash
bash train_lgf_multilevel.sh
```

For one placement only:

```bash
bash train_lgf_multilevel.sh C3_C4
bash train_lgf_multilevel.sh C3_C4_C5
```

The shell command is safe to rerun. It skips completed runs and resumes interrupted runs when `last.pth` exists, using the original resume protocol checks.

## File and result layout

```
/nas.dbms/asera/paper1_original/ugs_comparison/
  runs_C3_C4/{dataset}/lgf_original/seed_{seed}/
  runs_C3_C4_C5/{dataset}/lgf_original/seed_{seed}/
  console_logs_multilevel/
```

Each run writes `protocol.json`, `last.pth`, selected `best_ap_small.pth`, `best_map.pth`, `history.jsonl`, `evaluation/`, and `completed.json`.

## Verify location and sizes

Pre-FPN feature assignments are resolved directly from Torchvision ResNet50 `body.return_layers`:

| Semantic level | ResNet stage | Feature key | Channels |
| --- | --- | --- | --- |
| C3 | layer2 | 0 | 512 |
| C4 | layer3 | 1 | 1024 |
| C5 | layer4 | 2 | 2048 |

The model-building code raises a descriptive exception rather than continuing when these conditions do not hold. Each level has a **separate independently parameterized StableLGFBlock** with the original `gated_spatial` configuration. The original RetinaNet pretrained weights, head, transforms, and anchors are constructed through the original `core.build_model` factory.

## Important audit for earlier single-level checkpoints

The archived single-level wrapper resolves the symbolic levels incorrectly: its `semantic_to_actual` treats `actual_keys[0]` as `layer1`, although Torchvision's RetinaNet returns actual keys `0` (layer2), `1` (layer3), `2` (layer4). Thus the saved **single-level** labels might not describe the true insertion stage. The new multi-level code resolves `body.return_layers` correctly, without changing the legacy wrapper. Validate any old claims against the actual checkpoint keys before comparing placements:

```bash
"$P1_UGS_PY" audit_checkpoint_levels.py \
  /nas.dbms/asera/paper1_original/ugs_comparison/runs/coco_nw/lgf_original/seed_42/last.pth
```

This is a serious provenance finding and affects interpretation of previously reported single-level placement results. Prior checkpoints remain byte-for-byte unchanged.

## Checkpoint evaluation

Use this new runner for evaluation, since the earlier `paper1_test_evaluation_v1` loader only recognizes the original single-level values:

```bash
"$P1_UGS_PY" run.py evaluate \
  --dataset coco_nw --method lgf_original --level C3_C4 --device cuda \
  --checkpoint /nas.dbms/asera/paper1_original/ugs_comparison/runs_C3_C4/coco_nw/lgf_original/seed_42/best_ap_small.pth \
  --output /nas.dbms/asera/paper1_original/ugs_comparison/evaluate_C3_C4_coco_nw_seed42
```

The runner's `evaluate` action is validation evaluation. The separate official-test export tool has **not** been changed to support these new architectures.

## Parameter and reproducibility notes

The existing runner trains for 80 epochs with batch size 4, accumulation 4, SGD LR 0.005, EMA 0.9999, its original warm-up/unfreeze and checkpoint schedule, and validation AP/APS selection. Data from `/nas.dbms/asera/paper1_original/DATASET`. The wrapper stores exact source hashes in the run protocol, and interrupted runs require a matching checkpoint and unchanged source hashes. C3+C4 and C3+C4+C5 add separate LGF parameters, so FLOPs, memory and inference time will be higher than single-level C3.

The original `vendor/original_core.py` remains unchanged and integrity checks remain active.

Run the structural and gradient test suite in the server environment:

```bash
"$P1_UGS_PY" -m unittest discover -s tests -p 'test_multilevel.py' -v
```
