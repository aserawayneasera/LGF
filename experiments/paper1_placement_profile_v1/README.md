# Placement parameters, FLOPs, and inference time

This separate profiling package reads existing six-class RetinaNet/LGF EMA checkpoints. It does not train, resume training, edit checkpoints, or modify the manuscript or response letter. The bundled model source comes from `paper1_ugs_multilevel_v1.zip`; `vendor/original_core.py` remains byte-identical, with SHA256 `64a662586fd3c5b71a850ca7db19079d13366dfabae7b764e02de1f49567ae95`.

## Run on your server

Use the Python environment used for the placement experiments. The required PyTorch versions are 2.10.x and torchvision 0.25.x. Install only the extra FLOP counter.

Place the ZIP in `/nas.dbms/asera/NEW`, then run:

```bash
cd /nas.dbms/asera/NEW
unzip -q paper1_placement_profile_v1.zip
cd paper1_placement_profile_v1
export P1_PROFILE_PY="$(command -v python)"
"$P1_PROFILE_PY" -m pip install -r requirements-profile.txt
bash run_profile.sh
```

Run profiling while the GPU is free of other training or inference workloads. GPU state is recorded before and after each timing measurement. No other process is stopped.

The default searches these locations:

| Saved run label | Run root below `/nas.dbms/asera/paper1_original/ugs_comparison` |
| --- | --- |
| RetinaNet baseline | `runs/{dataset}/baseline/seed_{seed}` |
| C3 | `runs/{dataset}/lgf_original/seed_{seed}` |
| C4 | `runs_C4/{dataset}/lgf_original/seed_{seed}` |
| C5 | `runs_C5/{dataset}/lgf_original/seed_{seed}` |
| C3+C4 | `runs_C3_C4/{dataset}/lgf_original/seed_{seed}` |

Datasets: `coco_nw`, `coco_weather`, `acdc`. Seeds: 42, 1337, 2025. Selection: `best_ap_small.pth`. Missing checkpoints are listed and skipped. The uploaded `levels.zip` contains no `.pth` weights and no ACDC C3+C4 results; this package reads any existing weights from the server and reports actual availability.

The default C3 and baseline paths point to checkpoints under the runner's `runs` folder. To profile different retained manuscript checkpoints, give their absolute paths in a manifest. The profiler never replaces manuscript accuracy values with newer run scores.

To list checkpoint locations before profiling:

```bash
bash run_profile.sh --inventory-only
```

To use a different base folder:

```bash
bash run_profile.sh --runs-base /absolute/path/to/ugs_comparison
```

To use specific saved checkpoints, copy and edit `manifest.example.json`:

```bash
bash run_profile.sh --manifest my_checkpoints.json
```

Each manifest entry needs `dataset`, `placement`, `seed`, and an absolute `checkpoint` path. `metrics` is optional. Supported placements are `baseline`, `C3`, `C4`, `C5`, and `C3_C4`. Add one entry per checkpoint. If selected metrics are provided, their checkpoint SHA256, dataset, seed, and selection must agree. If no selected metrics accompany a checkpoint, accuracy fields stay empty. Select `--selection best_map` for a manifest of best-mAP checkpoints.

## Measurements

- Parameters: all model parameters, LGF parameters alone, and the parameter count after subtracting LGF. FrozenBN buffers are not counted as parameters.
- Complexity reference: one RGB image, 800 x 1333, passed through the normal transform. The resulting padded tensor shape is recorded, usually 1 x 3 x 800 x 1344.
- FLOP estimate: backbone, LGF, FPN, and the complete detection head, counted by fvcore. Its convention is one fused multiply-add per FLOP. Supported operator totals and every unsupported operator are recorded.
- Independent check: forward hooks count all Conv2d/Linear multiply-accumulate operations and must agree with fvcore's convolution/linear subtotal. The CSV supplies both GMACs and convolution/linear GFLOPs using two operations per MAC.
- Full detector inference: FP32, original 800/1333 resizing, original score/NMS thresholds, EMA weights, TF32 disabled, deterministic cuDNN settings. The source's debugging setting `CUDA_LAUNCH_BLOCKING=1` is disabled for profiling. This change applies only to the profiling process.
- Batch one: true synchronized wall time for one detector call per image. Batch 16: synchronized batch wall time divided by 16, described as amortized time per image. Each reports mean, median, p95, and throughput.
- Timing scope: normalization, resizing, padding, backbone, LGF, FPN, head, anchor generation, box decoding, NMS, and output rescaling. Disk reads, decoding, CPU tensor preparation, host-to-device copies, COCO scoring, and serialization finish outside the timed region.
- Default timing sample: 112 deterministic validation images per dataset, 20 warm-up calls per batch size, and three passes over the sample. All placements and seeds use the same images, batch grouping, and sample order for a given dataset. Tail images are omitted for a batch size which does not divide the sample count.
- Peak allocated and reserved GPU memory are measured after warm-up. Counts and timings come from separate executions so FLOP tracing and hooks do not affect latency.

FLOP counts omit image preprocessing, anchor creation, decoding, and NMS. Those operations are included in detector timing. fvcore's unsupported-operator list makes its counting scope explicit. Do not label the separate two-per-MAC convolution total as a complete count of every scalar operation. The historical manuscript's FLOP numbers are not recalculated or silently replaced by this package.

## Saved architecture verification

The profiler reconstructs the block schema from the checkpoint tensor keys, then uses strict EMA state loading. It records the saved run label alongside the actual feature keys, channels, and ResNet stage. C3, C4, and C5 correspond to feature keys 0, 1, and 2, with 512, 1024, and 2048 channels. This records what the saved detector contains; it never moves its blocks or substitutes a newly initialized detector for a missing trained checkpoint. A checkpoint with no saved LGF block is reported as resolved placement `none`.

The full LGF block parameter formula for the supplied spatial gate is `11*C*C/4 + 19*C + 2`. The model's actual parameters are counted directly. The formula serves as a separate smoke-test check.

## Output

Each run writes a new timestamped directory below the run base and a ZIP of that directory. The final console line gives the exact results ZIP path. Send that ZIP for the accuracy-complexity analysis and the narrow manuscript/response-letter update.

Files include:

- `placement_profile_per_seed.csv`: parameters, defined FLOP/MAC estimates, selected validation APsmall/mAP, and latency per checkpoint and batch size.
- `placement_profile_mean_std.csv`: mean and sample standard deviation across the available checkpoint seeds. `n_seeds` shows incomplete groups.
- `per_checkpoint/*.json`: actual architecture, checkpoint hash, operator details, raw per-call timing, GPU state, and exact inference scope.
- `timing_images_*.json`: sampled image IDs, dimensions, and hashes.
- `profiling_protocol.json`: hardware/software, profiling settings, and model/profiler source hashes.
- `inventory.json`, `completion.json`, `all_profiles.json`: availability and outcomes, including any failed measurement.

Overhead percentages use a successfully profiled RetinaNet baseline with the same dataset, seed, and timing batch size. If that baseline is absent, overhead fields stay empty. Across-seed standard deviation does not constitute a confidence interval for an accuracy difference or a GPU latency estimate. Accuracy fields refer to the selected checkpoint's existing validation metrics, without a new accuracy evaluation.

## Smaller run, counts only, or checks

For a first GPU run:

```bash
bash run_profile.sh --datasets coco_nw --seeds 42
```

For parameters and complexity from the trained checkpoints, without loading validation images:

```bash
bash run_profile.sh --counts-only --device cpu
```

For a synthetic CPU check of strict checkpoint reconstruction, counts, and inference for all four placements plus RetinaNet:

```bash
bash run_profile.sh --smoke --device cpu
```

For weight-independent architecture reference counts:

```bash
"$P1_PROFILE_PY" measure_architecture_counts.py --output architecture_reference_new
```

`architecture_reference/` contains counts generated from fresh, verified semantic architectures using the archived model definitions. These are structural reference values, without checkpoint-specific accuracy or GPU latency. Verify saved feature blocks before connecting these reference counts to a named experimental run.

Use `--batches 1` if only batch-one latency is required. Use `--device cuda:1` to select another GPU. `--data-root` changes the dataset base. `--output` selects a new or empty output folder. The profiling process never writes inside any existing checkpoint run directory.

## Method sources

- PyTorch 2.10 CUDA timing and synchronization: https://docs.pytorch.org/docs/2.10/notes/cuda.html
- Torchvision 0.25 RetinaNet implementation: https://docs.pytorch.org/vision/0.25/_modules/torchvision/models/detection/retinanet.html
- fvcore counting convention and supported operators: https://github.com/facebookresearch/fvcore/blob/main/fvcore/nn/flop_count.py
