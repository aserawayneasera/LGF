# Validation, 15 September 2026

No real Paper 1 test AP results were produced in this workspace.

Verified:

- Nineteen workflow tests passed. The default discovery run lists 20 tests, with the opt-in real-model test skipped.
- The real-model test passed separately for all seven adapters: RetinaNet, original LGF, SE, CBAM, UGS reimplementation, YOLO11m, and RT-DETR-L.
- Real-model checks used the full architectures, synthetic images, FP32, and the actual shorter-side 800 / longer-side cap 1333 evaluation geometry. Each adapter returned 100 nonempty detections. Original-model/UGS predictions matched direct forward results exactly after checkpoint loading.
- Workflow tests covered official category/image-ID remapping, source-image overlap detection, archive traversal rejection, ACDC test-only extraction, download restart with byte ranges and ETags, checksum rejection, private-link redaction, and deterministic weather generation without changing ground-truth boxes.
- Export tests covered checkpoint freezing, interrupted inference restart, saved-result reuse, changed selection/prediction rejection, and exclusion of partial software exports from server submissions and research metrics.
- Hidden-label exports cannot generate local AP. Labelled tests exercised perfect/empty detections, six-class evaluation in a dataset containing another class, condition-specific scoring, annotation ID zero, and paired-seed comparison checks.
- The actual official COCO image-info ZIP downloaded through the new CLI. The archive hash and 20,288-image test-dev JSON hash match the pinned official bytes.
- The four original uploads, storage-only notebook copies, and all Python files in both previous model packages remained byte-for-byte unchanged. New test code resides in a sibling folder so the old training source-hash checks see the same files.
- All new Python files and notebook code cells parsed successfully.

Environment: Python 3.12.14, torch 2.10.0+cpu, torchvision 0.25.0+cpu, Ultralytics 8.3.253, NumPy 2.3.5, Albumentations 2.0.8, and pycocotools 2.0.11.

Evidence files are under `validation/`. Local CPU timings and synthetic outputs are software checks, not model performance or speed comparisons.

Not verified here: full COCO image-archive transfer, authenticated ACDC downloads, actual ACDC test image/category metadata, live server submission acceptance, server availability of six-class APsmall, GPU operation, and test-scene independence against the user's complete experiment history. The code supports official metadata supplied by the user and does not invent missing ACDC server IDs.

Reproduce from this toolkit directory:

```bash
python -m unittest discover -s tests -v
P1_REAL_MODEL_TESTS=1 python -m unittest discover -s tests -p test_models.py -v
python test_data.py download --dataset coco --metadata-only --test-root /tmp/asera/paper1_test_metadata_check
```

The real-model test requires both adjacent model packages and the pinned newer-detector environment. Tests construct temporary synthetic fixtures and do not read the user's actual test labels.
