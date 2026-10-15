# Repair and diagnostic steps

The uploaded log is from `hayabusa`, RTX 5080, Torch 2.10.0+cu128,
Ultralytics 8.3.253. The earlier empty-shell-variable issue has been resolved.

## What the log establishes

| Stage | Observed result |
| --- | --- |
| YOLO11m synthetic training | Completed two training batches, then segmentation fault before validation results |
| RT-DETR-L synthetic training | Completed one epoch, native validation, shared evaluation, and checkpoint selection |
| RT-DETR-L reload check | Failed exact prediction equality after training |
| Real COCO-NW preparation | Passed |

Neither smoke test passed in the uploaded log. The zero AP values come from one
randomly initialized epoch on generated images. They are not research results.

## Confirmed code issue and repair

In the pinned `ultralytics==8.3.253` source, `BaseTrainer._do_train()` calls
`unset_deterministic()` after training. This disables deterministic algorithms,
resets cuDNN deterministic mode, and removes `CUBLAS_WORKSPACE_CONFIG`.
The original script compared an evaluation inside training with an evaluation
after this reset. Their numerical execution settings were different.

Version 1.0.1 runs shared inference with deterministic algorithms enabled,
cuDNN deterministic mode enabled, cuDNN benchmarking disabled, and
`CUBLAS_WORKSPACE_CONFIG=:4096:8`. A context manager restores the caller's
settings after inference, including when an exception occurs. Training seeds,
architectures, losses, optimizer recipe, data conversion, checkpoint-selection
metrics, and confidence thresholds are unchanged. The evaluation protocol now
records the inference policy, so old and new report signatures are distinct.

Exact prediction equality remains required. The repair does not round scores,
accept an AP-only match, or raise tolerance. The smoke check also verifies
checkpoint hashes, evaluation signatures, and metric equality. If predictions
differ, `reload_comparison.json` records counts, row identities, and maximum
coordinate/score differences when row identities align.

The settings mismatch is established from source. Whether this repair resolves
your specific GPU reload failure requires a new run on that GPU. The old log
does not contain the two prediction files or their numerical differences.

## YOLO segmentation fault

The log does not identify the failing native operation. CUDA/NMS or another
native-library problem remains unconfirmed. There is no CUDA out-of-memory
message. An installation upgrade or batch-size change is not justified by this
log alone.

`diagnose_reviewer34.py` runs each probe in a separate Python process and records
return codes, signals, Python/module paths, and logs. It tests tensor arithmetic,
convolution forward/backward and evaluation, and nonempty Torchvision NMS on
CPU and the selected GPU. It then runs both full-model synthetic smoke tests.
Native stack reporting and synchronous CUDA launches are enabled only in these
diagnostic child processes. No dependency versions or global installations are
changed. NMS is tested, not replaced or bypassed.

Smoke evidence is kept under the printed diagnostic directory, including after
exceptions. A native crash can leave a child `status.json` at `running`; the
parent `summary.json` records the actual failed process and signal. Each run
produces `diagnostic_report.zip` with text logs and JSON evidence, excluding
large checkpoint files and images.

## Run on the server

Upload the updated ZIP to `/tmp/asera/NEW`. Extract into a new directory so any
existing interrupted training continues to retain its original runner:

```bash
unzip /tmp/asera/NEW/paper1_all_codes_with_test_evaluation.zip -d /tmp/asera/NEW/paper1_recovery_1_0_1
cd /tmp/asera/NEW/paper1_recovery_1_0_1/paper1_all_codes_with_test_evaluation/paper1_reviewer34_v1
export P34PY=/tmp/asera/paper1_reviewer34_env/bin/python
sha256sum -c CHECKSUMS.sha256
"$P34PY" -u diagnose_reviewer34.py --device 0
```

The default interpreter is the environment from your earlier setup. If that path
does not exist on `hayabusa`, use the interpreter which ran the uploaded log.
Do not switch blindly to the base Python or hide user-site packages. The log
shows Torch loaded from `/nas.dbms/asera/.local/lib/python3.10/site-packages`;
this path alone does not prove an incompatible installation.

Run in the same terminal/tmux session as the `export`. The last output line
beginning `REPORT ZIP` gives the report to upload if any stage fails.

After a successful diagnostic run, repeat the ordinary GPU smoke commands
without `CUDA_LAUNCH_BLOCKING` before launching full experiments:

```bash
"$P34PY" -u reviewer34_compare.py smoke --model yolo11m --device 0
"$P34PY" -u reviewer34_compare.py smoke --model rtdetr-l --device 0
```

Require `"status": "passed"` for both models. A line saying `TRAINING COMPLETE`
precedes the reload check and is not the final smoke pass.

Then use the existing README training commands. The already prepared COCO-NW
labels are compatible; their audit and checksum checks still run. Old interrupted
training is intentionally not resumed with the new source hash. Do not delete
protocol files or weaken resume checks. Keep an old interrupted run with its old
runner, or use a new work root for a new run.

## Validation performed here

- 29 automated tests passed, including the eight new reload/settings tests,
  original LGF adapter agreement, and the native training interruption/resume test.
- CPU arithmetic, convolution, and Torchvision NMS probes passed.
- Full YOLO11m and RT-DETR-L synthetic training/evaluation/reload smoke tests passed
  on CPU. Selected checkpoint bytes, exported predictions, and metrics matched.
- CUDA is unavailable here. No RTX 5080 smoke pass, real-dataset convergence,
  GPU memory claim, runtime benchmark, or full experiment completion is claimed.

Current evidence is in `validation/reload_fix_tests.log` and
`validation/reload_fix_cpu/`. Other validation files describe the earlier release.

Reference: [PyTorch reproducibility documentation](https://docs.pytorch.org/docs/2.10/notes/randomness.html).
The pinned Ultralytics wheel source, rather than a different release's online
documentation, was used to verify the teardown behavior.
