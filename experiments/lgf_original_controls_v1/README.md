> Repository integration: use [diagnostics.md](../../docs/diagnostics.md) for portable commands and input provenance. The instructions below describe the recovered historical server workflow. Scientific Python/model sources retain their original bytes. Shell launchers use repository-relative paths and documented environment overrides.

# LGF original gate controls, version 1.0.0

Evaluate saved `lgf_original` EMA checkpoints. No detector training is performed.
The bundled model source and evaluation helpers are unchanged copies from the
delivered UGS comparison package. You do not need to install Ultralytics or run
YOLO, RT-DETR, or UGS to perform these controls.

## Quick start on Atlantis

Copy `lgf_original_controls_v1.zip` into `/tmp/asera/NEW`, then:

```bash
unzip -n /tmp/asera/NEW/lgf_original_controls_v1.zip -d /tmp/asera/NEW
cd /tmp/asera/NEW/lgf_original_controls_v1
export P1_CTRL_PY=/nas.dbms/asera/micromamba/envs/CUDA_12.8_env/bin/python

"$P1_CTRL_PY" -u controls.py check --device cuda
"$P1_CTRL_PY" -u controls.py run --device cuda
```

The defaults evaluate all three datasets (`coco_nw`, `coco_weather`, `acdc`) and
all three training seeds (`42`, `1337`, `2025`). Eight controls times nine
checkpoints gives 72 validation evaluation passes. Jobs run sequentially on one
GPU. This is evaluation of Table S6 controls, not an official hidden-test-set
submission or a separately trained ungated-model comparison.

The `check` action checks the requested checkpoint inventory, validates dataset
files, and loads each EMA state strictly. `run` checks the inventory first and
checks numerical equivalence of the native gate calculation before evaluating.

Default checkpoint root:

```text
/tmp/asera/paper1_original/ugs_comparison/runs
```

Expected example checkpoint:

```text
/tmp/asera/paper1_original/ugs_comparison/runs/coco_nw/lgf_original/seed_42/best_ap_small.pth
```

Default dataset root:

```text
/tmp/asera/paper1_original/DATASET
```

Default output directory:

```text
/tmp/asera/paper1_original/lgf_controls_v1/paper
```

If your paths differ, pass both commands the same `--run-root` and `--data-root`.
Use the Python environment already running your LGF comparison experiments.
The package targets torch 2.10.x and torchvision 0.25.x and does not download
pretrained weights. Existing model checkpoints must be available.

## Which saved models work?

- `lgf_original` checkpoints produced by `paper1_ugs_comparison_v1/run.py`.
- The same LGF checkpoints passed to `reviewer34_compare.py` for evaluation.
  Reviewer34 uses the earlier LGF loader; it does not train a separate LGF model.
- Original notebook `BEST_...pth` spatial-gated LGF checkpoints with their config,
  experiment name, and EMA weights. Supply these explicitly and use `best_map`,
  since the inspected original notebook selected checkpoints by mAP.

The UGS comparison package stores multiple methods. A checkpoint for
`ugs_reimpl` itself is a different model and is rejected. Baseline, SE, CBAM,
YOLO, RT-DETR, spatially global gates, and ungated LGF are also rejected.
The original insertion-level mapping is preserved exactly.

The evaluator reads EMA weights only and uses strict state-dictionary loading.
It does not guess a different insertion level, omit mismatched tensors, fall
back to non-EMA weights, or use an arbitrary last checkpoint.

## One completed run first

If other models are still training, narrow the requested grid:

```bash
"$P1_CTRL_PY" -u controls.py check \
  --datasets coco_nw --seeds 42 --device cuda

"$P1_CTRL_PY" -u controls.py run \
  --datasets coco_nw --seeds 42 --device cuda \
  --output-root /tmp/asera/paper1_original/lgf_controls_v1/paper_coco_nw_seed42
```

For all datasets and all seeds, use the default full-grid output directory.
An output directory belongs to one exact grid and settings. Use a new directory
when changing the requested seeds, dataset, checkpoint bytes, suite, device,
batch size, source code, or other recorded settings.

To select by validation mAP instead, use `--selection best_map` consistently
and a separate output directory. Choose one selection rule before comparing
controls. Every control within a run uses the exact same selected checkpoint.

Explicit comparison-runner checkpoint:

```bash
"$P1_CTRL_PY" -u controls.py run \
  --checkpoint /tmp/asera/paper1_original/ugs_comparison/runs/coco_nw/lgf_original/seed_42/best_ap_small.pth \
  --datasets coco_nw --seeds 42 --selection best_ap_small \
  --device cuda \
  --output-root /tmp/asera/paper1_original/lgf_controls_v1/explicit_coco_nw_seed42
```

For an original notebook checkpoint, replace that path with the actual
`BEST_...pth` file, set the matching dataset/seed/level, and use
`--selection best_map`. A six-class spatial-gated LGF checkpoint with matching
metadata is required. Original notebook files lack some dataset provenance
stored by the newer runner; the result records this limitation explicitly.

## Control definitions: read before replacing Table S6

The recovered original notebook and the manuscript describe different controls.
This release labels them separately. Old Table S6 values are not guaranteed to
reproduce under the manuscript equations. Fresh results must be reported with
the exact suite used. Neither suite changes the trained model or its saved weights.

In the default `--suite paper`, every output is `X + gamma*Y`, retaining learned
gamma unchanged. `L` and `G` are the trained local and broadcast global branches.
The sigmoid gate weights are independent and do not necessarily sum to one.

| Control | Fusion Y |
| --- | --- |
| native | Original forward, `wL*L + wG*G` |
| uniform | `0.5*L + 0.5*G` |
| shuffle | Permute native gate maps jointly across batch times channels, then weighted fusion |
| swap | `wG*L + wL*G` |
| alpha_0.25 | `0.25*L + G` |
| alpha_0.5 | `0.5*L + G` |
| alpha_2.0 | `2.0*L + G` |
| no_gate | `L + G` |

Fixed-weight controls do not execute the gate head. The two branches and gamma
remain intact. Removing the entire module is a different intervention.

The shuffle uses seed 123 by default and applies the same permutation to both
gate tensors. It mixes channels and images within each inference batch, not
across the whole dataset. With batch size one, only channels are permuted.
Results depend on batch composition, so use the same batch size for every seed
and control. Default batch size is four, matching the earlier runner. A final
partial batch is retained. There is no random spatial permutation.

Uniform weights, sum fusion, and fixed alpha mixing also change feature-update
magnitude. These post-training controls measure sensitivity to the intervention;
they do not isolate spatial adaptivity alone or establish the performance of a
model trained without a gate.

### Historical notebook suite

Use `--suite notebook` and a separate output directory to repeat the recovered
notebook definitions across seeds:

```bash
"$P1_CTRL_PY" -u controls.py run \
  --suite notebook --device cuda \
  --output-root /tmp/asera/paper1_original/lgf_controls_v1/notebook
```

This suite includes nine controls: native, global mean, uniform, shuffle, swap,
three alpha values, and no_module. For non-native controls other than no_module,
the notebook rescales the weights by a per-image mean weight budget. Its alpha
controls scale the learned local gate, normalize the two gates, then apply that
budget rescaling. Its no_module output is X, removing the entire LGF update.
This is not no_gate fusion. The old script computed branches before returning X;
the new no_module implementation bypasses that unnecessary computation. Output
equivalence holds, but its execution time is not a reproduction of that old path.

Tests compare these numerical outputs against the extracted original functions.
Do not combine the two suites in one table without identifying the different
equations. For a revised table matching the manuscript equations, use `paper`.

## Outputs and interpretation

- `controls_per_seed.csv`: all metrics per checkpoint/control plus paired changes.
- `controls_mean_std.csv`: mean and sample standard deviation across training seeds.
- `summary_status.json`: reports whether every requested run finished.
- `experiment.json`: the requested grid, input checkpoint hashes, settings,
  source hashes, and environment.
- `<dataset>/seed_<seed>/checkpoint.json`: gate parameter cost, gamma, native
  parity evidence, and data-verification status.
- `<dataset>/seed_<seed>/<control>/predictions.json`: COCO predictions.
- `<dataset>/seed_<seed>/<control>/control_result.json`: metrics and provenance.

AP and mAP values use the 0-1 scale. Delta columns use AP points:

`delta = 100 * (control AP - native AP)`.

For example, a native value of 0.0866 and a no_gate value of 0.0548 give -3.18
points. A negative delta means the control performed worse than native LGF.

Standard deviations use ddof=1 across training seeds. A single seed has null
standard deviation. Missing metrics and unavailable size categories are not
converted to zero during aggregation. These standard deviations are not
statistical significance tests. No best-seed selection occurs in the summaries.

## Resume after interruption

Repeat the identical `run` command. Completed controls are reused after checking
their provenance and prediction hashes. An interrupted control is reevaluated.
The evaluator rejects concurrent writes to the same output directory.

To rebuild summaries without loading models:

```bash
"$P1_CTRL_PY" controls.py summarize \
  --output-root /tmp/asera/paper1_original/lgf_controls_v1/paper
```

A partial summary is marked partial. Original checkpoints and dataset files are
never rewritten. Do not evaluate a best checkpoint while training is replacing
that file. The evaluator checks for changes and stops if detected.

## Gate cost and optional timings

Gate parameter count is exact for the loaded model. The hypothetical parameter
count after deleting the gate is reported separately from the in-memory model,
which retains the head for reversible evaluation. The convolution cost estimate
uses the observed first-batch feature shape and two FLOPs per multiply-add.
It excludes normalization, sigmoid, and elementwise arithmetic and is not a
whole-detector FLOP count.

For optional timing alongside all controls, use `--benchmark` from the start:

```bash
"$P1_CTRL_PY" -u controls.py run --device cuda --benchmark \
  --output-root /tmp/asera/paper1_original/lgf_controls_v1/paper_with_timing
```

Timings use five warmup and twenty measured repeats of the first validation batch,
preloaded on the device, with CUDA synchronization. They include detector
postprocessing. Report the batch size and device. Amortized per-image batch time
is not single-image latency. The original code's CUDA environment settings are
preserved and recorded. Fixed paper controls skip gate computation, so no_gate
timing excludes the head. Run timing on an otherwise idle GPU.

## Local software checks

```bash
"$P1_CTRL_PY" -m unittest discover -s tests -p 'test_*.py' -v
```

See `VALIDATION.md` for checks completed during package preparation and their
limits. Real trained checkpoints and datasets were not available in this session.
The synthetic test images and weights do not provide scientific results.

If `check` reports a missing checkpoint, choose a completed dataset/seed or the
correct checkpoint root. If strict loading or provenance verification fails,
keep the full error message and diagnose the mismatch rather than weakening the
loader. If CPU/CUDA memory is insufficient, choose a smaller batch size and a
new output directory; preserve that batch size across every control and seed.
