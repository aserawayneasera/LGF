# Smoke/reload repair 1.0.1, 15 September 2026

Read `RECOVERY_1_0_1.md` first if your log contains a YOLO segmentation fault or
`Checkpoint reload predictions differ`. This repair changes the comparison runner's
inference settings and diagnostics. Existing original LGF/UGS training files and the
test-evaluation toolkit retain their uploaded bytes. Prepared labels remain compatible.
Run the supplied diagnostic command on your GPU before starting full experiments.

# Official test evaluation update, 15 September 2026

Use the sibling `paper1_test_evaluation_v1` toolkit for official COCO/ACDC test downloads and prediction exports. It supports all seven supplied detector/control methods. The earlier test-evaluation addition left `reviewer34_compare.py` unchanged. The subsequent 1.0.1 repair above changes its source hash, so retain the old runner for any old interrupted training run.

Read `../paper1_test_evaluation_v1/README_TEST_EVALUATION.md`. The older `--split test --annotations ...` route below requires real labelled holdout annotations. Official hidden-label test image-info JSON is not ground truth. The new workflow creates server submissions without claiming local AP. The new sibling notebook `4.test_evaluate_all_methods.ipynb` provides a launcher.

---

# Paper 1: Reviewer 3 and Reviewer 4 comparison package

One runnable script, `reviewer34_compare.py`, supports two newer detector baselines:

| Option | Implementation | Purpose |
| --- | --- | --- |
| `yolo11m` | Ultralytics YOLO11m | First additional detector baseline |
| `rtdetr-l` | Ultralytics RT-DETR-L with HGNetv2 | Optional transformer-based baseline |

YOLO11m is the first experiment to run. One additional model directly addresses Reviewer 3's minimum request. RT-DETR-L broadens the comparison; the reviewers did not explicitly require this named variant. Both support Reviewer 4's request for stronger recent baselines beyond RetinaNet+SE/CBAM.

The script also evaluates your existing original LGF, RetinaNet baseline, and UGS checkpoints through the previous UGS package. The original notebooks, model code, and C3 mapping remain intact. This package does not train LGF inside YOLO or RT-DETR.

The YOLO11 baseline is standard YOLO11m. It is not a reproduction of Bayraktar and Bakirci's attention-augmented YOLO11 paper. Reviewer 4 requested discussion of that work. No custom attention module is invented here.

## Sources and implementation status

- [Reviewer decision letter, 14 September 2026](https://mail.google.com/mail/#all/1a09f0752fd6a47a).
- [Ultralytics YOLO11 documentation](https://docs.ultralytics.com/models/yolo11/).
- [Ultralytics RT-DETR documentation](https://docs.ultralytics.com/models/rtdetr/).
- [Pinned Ultralytics source, v8.3.253](https://github.com/ultralytics/ultralytics/tree/v8.3.253).
- Bayraktar and Bakirci, [Attention-Augmented YOLO11 for High-Precision Aircraft Detection in Synthetic Aperture Radar Imagery](https://doi.org/10.1109/DSPA64310.2025.10977903), DSPA 2025. Bibliographic identity verified through Crossref. This package does not implement the paper's modifications.
- [IEEE document 11271980](https://ieeexplore.ieee.org/document/11271980), separately named by Reviewer 4. The full bibliographic record was not resolved here; verify before adding a formal reference.

The architecture and pretrained-model loaders come from Ultralytics. The surrounding dataset conversion, checkpoint selection, common evaluation, and reporting are new code. The fine-tuning recipe below is an explicit comparison recipe, not a claimed reproduction of either detector's published benchmark training.

## Installation on atlantis

Copy `paper1_reviewer34_v1.zip` to `/tmp/asera/NEW`, then:

```bash
cd /tmp/asera/NEW
unzip paper1_reviewer34_v1.zip
cd paper1_reviewer34_v1
sha256sum -c CHECKSUMS.sha256

/nas.dbms/asera/micromamba/envs/CUDA_12.8_env/bin/python -m venv --system-site-packages /tmp/asera/paper1_reviewer34_env
export P34PY=/tmp/asera/paper1_reviewer34_env/bin/python
"$P34PY" -m pip install -r requirements-reviewer34.txt
"$P34PY" reviewer34_compare.py check --device 0
```

The separate environment reuses your installed CUDA torch 2.10.x and torchvision 0.25.x. Ultralytics is pinned to 8.3.253 for reproducibility. The official pretrained assets were downloaded and loaded locally, and the script checks their recorded SHA-256 values. Do not upgrade the package or replace the weights midway through a run or seed group.

`--device 0` means one GPU. For a CPU smoke check, use `--device cpu`. The script does not implement multi-GPU training. Full-size CPU training is not the intended workflow.

If you download only the Python file, the script has no other local code dependency for YOLO11m or RT-DETR-L. Install the five packages listed in `requirements-reviewer34.txt`. Original LGF/UGS evaluation additionally needs the earlier `paper1_ugs_comparison_v1` directory.

## Data and model storage

| Content | Default location |
| --- | --- |
| Original prepared six-class datasets | `/tmp/asera/paper1_original/DATASET` |
| Converted YOLO-format labels and image links | `/tmp/asera/paper1_original/reviewer34/prepared` |
| Downloaded pretrained detector weights | `/tmp/asera/paper1_original/reviewer34/pretrained` |
| Training runs and selected models | `/tmp/asera/paper1_original/reviewer34/runs` |
| Original LGF/UGS adapter package | `/tmp/asera/NEW/paper1_ugs_comparison_v1` |

Use the existing prepared data copied for the earlier package. Do not regenerate weather images or change splits for this comparison. Original COCO JSON files remain unchanged. Conversion writes separate YOLO labels and symbolic links to the same images.

Supported datasets are `coco_nw`, `coco_weather`, and `acdc`. The script preserves the sorted COCO category-ID mapping. It follows the original reader's exclusion of boxes with width or height at most one pixel. It retains crowd annotations as training targets, matching the original reader, while the COCO evaluator retains the original crowd/area metadata.

Preparation rejects missing/undecodable images, dimension mismatches, duplicate target rows, out-of-bounds training boxes, and identical train/validation image bytes. Image hashes do not prove that different weather versions belong to different underlying scenes. Reviewer 3 and Reviewer 4's scene-leakage concern needs source-image lineage evidence.

`/tmp` is temporary. Copy completed runs to your chosen durable destination before the server clears temporary storage. The script does not automatically mirror models to NAS.

## First experiment

Run the synthetic checks before using GPU time on a full experiment:

```bash
"$P34PY" reviewer34_compare.py smoke --model yolo11m --device 0
"$P34PY" reviewer34_compare.py smoke --model rtdetr-l --device 0

"$P34PY" reviewer34_compare.py prepare --dataset coco_nw --device 0
```

The smoke actions run the full architectures on generated images. YOLO11m uses a 64-pixel training size; RT-DETR-L uses 128 because its 300-query decoder needs enough encoder proposals. Their evaluation images are rectangular. Smoke evaluation lowers the confidence threshold solely to exercise nonempty decoding. Smoke outputs are never accepted by the comparison command as research results.

For the first full experiment, start a tmux session and run:

```bash
tmux new -s paper1_r34
```

Inside tmux:

```bash
cd /tmp/asera/NEW/paper1_reviewer34_v1
export P34PY=/tmp/asera/paper1_reviewer34_env/bin/python
mkdir -p /tmp/asera/paper1_original/reviewer34/logs
set -o pipefail

"$P34PY" -u reviewer34_compare.py train --model yolo11m --dataset coco_nw --seed 42 --device 0 \
  2>&1 | tee /tmp/asera/paper1_original/reviewer34/logs/yolo11m_coco_nw_seed42.log
```

Detach with Ctrl+B, then D. Reattach with `tmux attach -t paper1_r34`.

Default training settings:

| Setting | YOLO11m | RT-DETR-L |
| --- | --- | --- |
| Initial weights | COCO pretrained | COCO pretrained |
| Epochs | 80 | 80 |
| Training image size | 1280 | 1280 |
| Batch size / nominal accumulation target | 4 / 16 | 4 / 16 |
| Optimizer | SGD | AdamW |
| Initial learning rate | 0.005 | 0.0001 |
| Weight decay | 0.0001 | 0.0001 |
| Warmup | 2 epochs | 2 epochs |
| AMP | Disabled | Disabled |
| Early stopping | Disabled | Disabled |
| Newer-detector architecture | Unmodified | Unmodified |

Training uses each detector's native square-input pipeline. Horizontal flipping remains enabled. Mosaic, mixup, geometric scaling/translation, and HSV augmentation are disabled. Native arguments are recorded in `native/args.yaml`; active Albumentations defaults are printed in the training log. The pinned installation retains Blur, MedianBlur, ToGray, and CLAHE, each with probability 0.01. The recipe differs from LGF's aspect-preserving training transform, brightness/contrast augmentation, freezing schedule, and optimizer schedule. The nominal accumulation target does not imply an identical optimizer-step history during native warmup.

These settings need a convergence check on your data. Do not treat an undertrained or failed baseline as evidence of LGF superiority. Freeze the recipe before the final seed grid. If changes are needed after a pilot, use a new `--work-root` and keep the pilot outside the final table.

After seed 42 finishes and the learning curves look reasonable, run the remaining seeds:

```bash
for seed in 1337 2025; do
  "$P34PY" -u reviewer34_compare.py train --model yolo11m --dataset coco_nw --seed "$seed" --device 0 || break
done
```

For the optional transformer baseline, use `--model rtdetr-l`. For the other datasets, use `--dataset coco_weather` or `--dataset acdc`. Each model/dataset combination adds three complete training runs. One successful YOLO11m comparison is the first priority; do not start every combination before inspecting the first run.

If GPU memory is insufficient, reduce `--batch` while keeping the nominal accumulation target recorded. Keep `--eval-batch` consistent across compared methods. A changed training configuration needs a fresh work root for a clean seed group.

## Shared evaluation and checkpoint selection

All comparison metrics use the same original COCO annotations, category IDs, IoU thresholds 0.50:0.05:0.95, area definitions, and maximum detections `[1, 10, 100]`.

Evaluation preserves image aspect ratio with shorter side 800 and longer-side cap 1333, followed by stride-32 padding. The resize/interpolation implementation was checked against Torchvision. The original LGF evaluator retains its own original transform. Evaluation uses batch 4 by default, FP32, EMA weights, and the original 0.05 confidence threshold. The newer models receive already-resized tensors, bypassing their default inference image resizing. Their native input normalization and postprocessing remain model-specific. YOLO11m and RetinaNet use class-aware NMS; RT-DETR-L uses native query ranking without NMS.

The custom selection callback evaluates every five epochs and at the final epoch, then saves both best APsmall and best mAP. Ultralytics also writes its own auxiliary validation results and `native/weights/best.pt`; those use the native validation pipeline and are not the shared comparison scores.

Use these files for comparisons:

```text
.../runs/coco_nw/yolo11m/seed_42/best_map.pt
.../runs/coco_nw/yolo11m/seed_42/best_map.metrics.json
.../runs/coco_nw/yolo11m/seed_42/best_ap_small.pt
.../runs/coco_nw/yolo11m/seed_42/best_ap_small.metrics.json
```

Each run also records source/data hashes, initialization-weight hashes, native arguments, selected-epoch history, actual detections, and a completion marker. Candidate checkpoints that improved either shared selection metric are retained.

Resume an interrupted run with the same arguments plus `--resume`:

```bash
"$P34PY" -u reviewer34_compare.py train --model yolo11m --dataset coco_nw --seed 42 --device 0 --resume
```

Resume uses the native `last.pt` checkpoint and checks the recorded protocol. Native Ultralytics checkpoints serialize EMA/optimizer state using reduced precision; restart is not a bitwise continuation guarantee. A completed, stripped native checkpoint is not resumable. An interruption during shared evaluation requires recovery of that evaluation before continuation; the script blocks silently skipping the affected checkpoint-selection epoch.

## Evaluate your original LGF and compare

First identify a checkpoint with verified dataset, seed, level, and selection history. The original notebook saves best mAP, while its aggregation code selects peak APsmall. Those two epochs are not automatically the same.

For a verified original best-mAP LGF checkpoint, replace the placeholder path and run:

```bash
export P1_LGF_CKPT=/absolute/path/to/verified_original_lgf_seed42.pth

"$P34PY" reviewer34_compare.py evaluate \
  --model lgf_original --checkpoint "$P1_LGF_CKPT" \
  --dataset coco_nw --level C3 --seed 42 --device 0 \
  --selection best_map \
  --selection-provenance "Verified original best-mAP checkpoint for this dataset and seed" \
  --output /tmp/asera/paper1_original/reviewer34/evaluations/coco_nw/lgf_seed42
```

The declaration is not proof by itself. Check the original logs and checkpoint metadata first. Leave selection as `unverified` while provenance is unresolved; comparison rejects unverified selection. The adapter requires spatial-gated LGF and preserves the original C3 mapping.

Then compare the two seed-42 reports:

```bash
"$P34PY" reviewer34_compare.py compare \
  --reports \
  /tmp/asera/paper1_original/reviewer34/evaluations/coco_nw/lgf_seed42/metrics.json \
  /tmp/asera/paper1_original/reviewer34/runs/coco_nw/yolo11m/seed_42/best_map.metrics.json \
  --output /tmp/asera/paper1_original/reviewer34/comparisons/coco_nw_seed42
```

For the final three-seed table, pass all six report paths. For three methods, pass nine report paths. The command checks matching seeds, dataset/evaluation signatures, selection rules, checkpoint bytes, prediction bytes, and configuration consistency within each method. It reports mean and sample standard deviation on the 0–1 AP scale. Standard deviation is not a statistical significance test.

To evaluate earlier package checkpoints, use `--model baseline` or `--model ugs_reimpl` and their actual checkpoint/seed. They require the same earlier package path. Reports from the old runner should be regenerated through this evaluator before combining them with the newer detectors, so batch and evaluation provenance are explicit.

## Independent test evaluation

The default reports are validation results. Best-epoch selection uses validation. A separate test evaluation requires a fixed selected checkpoint plus both `--split test --annotations /path/to/test.json --images /path/to/test_images`.

The script rejects exact image-byte overlap with the supplied training/selection-validation splits. Scene-level independence requires dataset lineage checks, particularly for paired clear/weather images. Renaming a split to "test" does not establish independence. No new split is created automatically, and no existing training data are reassigned.

## Reviewer 3: complete request checklist

The original review contains seven major comments plus several presentation and language corrections. These are the requested actions and the scope of this package:

| Request | Action | Covered here? |
| --- | --- | --- |
| Reference 12 appears nonexistent | Verify the source and provide a genuine link or correct/remove the reference | No, reference audit required |
| Compare with at least one newer detector fine-tuned on the same data | Train YOLO11m; optionally add RT-DETR-L | Code provided; real experiments pending |
| Explain C3 and FPN before first use | Move/clarify terminology in the manuscript | No code remapping required |
| Replace "non-weather" terminology | Use clear versus adverse/challenging weather in the prose | Manuscript edit pending; existing file paths retained |
| Distinguish validation from an independent test set | Recover selection history and establish a valid evaluation protocol | Validation is labelled; separate test evaluation supported, independence not established |
| Clarify whether weather augmentation happened before or after splitting | Provide source-image lineage and generation order | Byte-overlap checks provided; scene lineage pending |
| Consider removing Section 5.3 | Decide with the coauthors whether to remove, narrow, or better explain the transfer study | Not an automatic deletion; Reviewer 4 asks for more transfer discussion |
| Explain applicability beyond RetinaNet | State required architectural interfaces and narrow untested generality claims | Manuscript argument pending |
| Repair the graphical abstract | Improve text size and readability | Figure edit pending |
| Clarify "small" means a small image region | Revise abstract and introduction | Manuscript edit pending |
| Avoid overloading H in the equations | Rename the hidden tensor or the dimension symbol | Manuscript edit pending |
| Expand AR and identify supplementary figures | Define Average Recall and mark Figures S1/S2 as supplementary | Manuscript edit pending |
| Fix Table 1 caption and width | Describe all metrics and fit the text area | Table edit pending |
| Reduce repeated numbers in prose | Discuss patterns rather than restating tables | Manuscript edit pending |
| Repair references | Citation spacing/capitalization; ref. 6 year; published versions of refs. 13/16; missing data in refs. 17/18/23 | Reference audit pending |
| Revise English throughout and use consistent "small object" wording | Address the detailed sentence corrections across Sections 1–7 | Language edit pending |

## Reviewer 4: all 16 numbered requests

| No. | Request | How to address it | Covered here? |
| --- | --- | --- | --- |
| 1 | Statistical support for modest APsmall gains | Paired confidence intervals/significance analysis with an appropriate unit of resampling | Seeds and detections recorded; inferential analysis pending |
| 2 | Stronger weather-generalization design | Clear/mixed-weather training followed by genuinely unseen conditions | Separate study or narrower claims; not supplied as a new experiment here |
| 3 | Explain higher transfer APsmall but lower overall mAP | Discuss the trade-off directly | Existing-results discussion pending |
| 4 | Stronger, newer baselines | YOLO11m and optional RT-DETR-L on the same prepared data | Code provided; real experiments pending |
| 5 | Prevent paired-scene leakage | Group clear/weather versions by source scene before splitting and document lineage | Exact-byte checks only; lineage evidence pending |
| 6 | Separate fog, rain, snow results | Evaluate retained weather-condition groups | Requires verified condition labels; not inferred from filenames |
| 7 | Discuss IEEE document 11271980 | Verify the reference and compare feature-resolution strategies | Bibliographic verification pending |
| 8 | Explain low absolute ACDC APsmall | Separate relative improvement from deployment reliability | Existing-results discussion pending |
| 9 | Quantify gating's value versus cost | Compare existing gate/no-gate ablations and their computation | Original ablations unchanged; cost analysis pending |
| 10 | Visualize local/global gates | Show verified clear/fog/rain/snow examples around small objects | Visualization pending |
| 11 | Empirical placement study | Evidence for placement choices, or a justified scope/claim revision | No original placement or C3 mapping change included |
| 12 | Support architecture-agnostic claims | Test LGF in another detector, or narrow the claim | Standard YOLO/RT-DETR comparisons do not establish this claim |
| 13 | Clearer accuracy–complexity analysis | Report parameter cost and a controlled latency/throughput benchmark | Parameter counts provided; matched runtime benchmark pending |
| 14 | Discuss attention-augmented YOLO11 for SAR aircraft | Cite and compare its mechanism/domain with LGF | Reference identified; special architecture not reimplemented |
| 15 | More qualitative failures and successes | Show baseline failures fixed by LGF, shared failures, and new false positives | Detections exported; figure selection pending |
| 16 | Narrow or strengthen robustness claims | Align claims with the transfer evidence, or add justified generalization tests | Manuscript/experimental decision pending |

Reviewer 3 questions the usefulness of Section 5.3, while Reviewer 4 finds the transfer results interesting and requests deeper analysis. These comments need a reasoned response rather than treating both as instructions for incompatible edits.

## Is the comparison fair?

Yes, as a comparison of complete detectors on your application, provided the data, evaluation, selection rules, convergence checks, and resource reporting are transparent.

Different architectures are appropriate for the reviewers' question: how does RetinaNet+LGF perform against newer alternatives on the same traffic-object data? The script aligns evaluation images, class mapping, resize geometry, batch size, precision, confidence cutoff, detection cap, and COCO scoring. It records model-specific normalization and postprocessing.

The training recipes are not identical. Native square training at 1280, the SGD/AdamW choice, augmentation details, full-network fine-tuning, and schedules differ from the original LGF run. Report those differences. Do not describe this as a strictly identical-training experiment. Verify useful convergence and allow a defensible tuning budget for each baseline before freezing the final protocol. Disclose the COCO pretraining source and any known pretraining/evaluation overlap.

Use two separate conclusions:

1. RetinaNet versus RetinaNet+LGF under matched training measures the contribution of your module.
2. RetinaNet+LGF versus YOLO11m/RT-DETR-L measures the performance of complete detector configurations.

A newer detector winning does not erase an LGF improvement within RetinaNet. An LGF configuration winning does not establish universal superiority or architecture independence. Adding standard YOLO or RT-DETR does not demonstrate that LGF works inside either architecture.

Report APsmall together with overall AP and the medium/large-object trade-offs. Include model parameter counts. Before making efficiency claims, benchmark all selected models on the same GPU, batch, precision, image distribution, timing scope, and warmup procedure. The script's evaluation wall time includes I/O and framework overhead, and the native console timings use a different pipeline; neither is a controlled cross-model latency benchmark.

The strongest economical next step is one verified YOLO11m comparison, followed by the other seeds. Use RT-DETR-L if time permits and the broader comparison adds evidence. Keep the supplied UGS adaptation labelled "UGS (our RetinaNet reimplementation)". Preserve the original submitted table as historical evidence and explain any new protocol/results separately.

日本語: 同じ画像、クラス、評価方法、モデル選択ルールで比べるなら、完成した検出モデル同士の比較として妥当です。ただし、学習方法や計算量の違いを明記してください。LGF自体の効果は、同じ条件のRetinaNetとRetinaNet＋LGFで確認します。YOLOやRT-DETRとの比較だけでは、LGFがどのモデルでも有効とは言えません。
