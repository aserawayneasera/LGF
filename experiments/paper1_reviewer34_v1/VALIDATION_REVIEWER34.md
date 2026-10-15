# Software validation

Prepared 2026-09-14. No real Paper 1 training experiment or comparative AP result was produced here.

Environment: Python 3.12.14, torch 2.10.0+cpu, torchvision 0.25.0+cpu, Ultralytics 8.3.253, NumPy 2.3.5, pycocotools 2.0.11, and Albumentations 2.0.8. CUDA was unavailable.

## Verified

- Twenty-one automated tests passed. They cover COCO-to-YOLO category/box mapping, original small-box filtering and crowd handling, invalid input rejection, prepared-file integrity, COCO perfect/empty-detection cases, interpolation agreement with Torchvision, inverse coordinate mapping, selection/provenance guards, paired-seed reporting, checkpoint/prediction integrity, and exclusion of synthetic smoke results from research tables.
- The LGF adapter loaded an original-format spatial-gated LGF checkpoint and exported exactly the detections produced by the original model's direct forward pass. The test used nonempty predictions and synthetic images. Original model/source files were not edited.
- A full YOLO11m architecture completed one synthetic training epoch, common rectangular-input evaluation, selected-checkpoint creation, and identical prediction export after reload.
- A full RT-DETR-L architecture completed the same pipeline. Its smoke training size was 128 to provide enough proposals for the unchanged 300-query decoder. Rectangular evaluation used a 128/192 resize policy.
- A two-epoch YOLO11m test deliberately interrupted training after a native epoch checkpoint. Resume completed the remaining epoch and created the shared-COCO selected checkpoint.
- The official COCO-pretrained YOLO11m and RT-DETR-L assets loaded successfully with 80 source classes. Their SHA-256 values are recorded and enforced by the script. Training replaces the class head through the native six-class dataset configuration.
- The original four uploads, the earlier storage-only notebooks, and the UGS package were checked for unintended file changes.

Evidence is in `validation/`. The test logs contain synthetic metrics and local CPU timings only. Those values are not manuscript results or cross-model speed benchmarks.

## Limits

Full-size 1280 training, 800/1333 GPU evaluation, convergence on the real datasets, GPU memory use, and cross-model runtime performance were not tested here. The smoke tests use random initialization and generated images; pretrained loading was checked separately. No author-paper accuracy reproduction is claimed.

The comparison recipe needs a pilot convergence check. Model-specific training and preprocessing details must be disclosed. Scene-level train/validation/test independence is not proven by image-byte hashes. The default selected-checkpoint scores are validation scores.

## Reproduce

```bash
export P1_UGS_PACKAGE=/tmp/asera/NEW/paper1_ugs_comparison_v1
python -m unittest discover -s tests -v
python reviewer34_compare.py smoke --model yolo11m --device cpu
python reviewer34_compare.py smoke --model rtdetr-l --device cpu
```

The legacy adapter test is skipped if `P1_UGS_PACKAGE` is unavailable. Both model smoke tests remain runnable without the older package.
