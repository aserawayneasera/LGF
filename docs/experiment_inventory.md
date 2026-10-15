# Experiment and protocol inventory

This is the PDF-derived inventory. Implementation details omitted from the PDFs are supplied by the recovered source packages in this release. Read [training](training_and_evaluation.md), [diagnostics](diagnostics.md), [weather and placement](weather_and_placement.md), and [provenance](reproducibility.md) for those source-level details. Statements below about unavailable information refer to the PDFs unless explicitly stated otherwise.

Source of truth: `LGF_Draft_8pages_with_supp-revised-marked.pdf` (25 pages, main article pp. 1–8, supplement pp. 9–25) and `1.PRL_response_letter_revised_LATEST_VERSION.pdf`. This inventory records what those documents state. Earlier memory values, older papers, and unverified implementation assumptions must not silently replace these values.

## Scope and provenance rules

- Paper ID: PRLETTERS-D-26-00171. Title: Local-Global Fusion for Small Object Detection.
- Six classes, in manuscript listing order: person, car, truck, bicycle, bus, motorcycle. Numeric category IDs and exact mapping are not supplied in the PDFs.
- All scored experiments use validation data. No independently scored official ACDC test or other test partition is reported. Test-prediction exports exist without scored reference metrics (Reviewer 3 major comment 5).
- Seed set for three-seed studies: 42, 1337, 2025.
- Retained original tables and later reviewer experiments are distinct result families. Preserve their identities, checkpoint rules, and annotation snapshots. Do not average or overwrite their outputs into one table.
- `C3` and `C4` are retained saved-run labels. Standard FPN C3 is ResNet layer2, stride 8 (method and captions). The manuscript explicitly preserves the original C3 name; actual historical routing is not recoverable from the PDFs. Record both run label and inspected runtime route/channel/stride in new exports.
- Main-text baseline is a customized RetinaNet setup, not stock torchvision defaults. CBAM is expressly CBAM-style, not a canonical reproduction.

## Experiment inventory and reviewer mapping

| Experiment family | Models / conditions | Checkpoint rule | Required outputs | Article/supplement location | Reviewer mapping |
|---|---|---|---|---|---|
| Matched-condition primary comparison | Baseline, one SE block, one CBAM-style block, LGF; unmodified COCO, synthetic weather COCO, ACDC; three seeds | Comparison selects best validation APsmall; legacy source saves best mAP, an explicitly documented discrepancy | Six AP metrics, six AR metrics, three-seed means and standard deviations | 4.1–4.3, 5.1–5.2; Tables 1–2; S3–S4 Tables S3–S4; S11 Table S13 | R3 major 5; R4 1, 8, 13, 16 |
| Separately trained gating alternatives | Baseline; LGF sum, two-scalar softmax, spatial sigmoid; COCO clear/synthetic; three seeds | Table says best checkpoint; retained comparison is APsmall selected | APsmall and mAP mean/std | 3.3, 5.4; S4 Table S5 | R4 9 |
| Clear-trained transfer | Baseline, CBAM-style, SE, LGF trained on unmodified COCO; evaluated on synthetic COCO versions without retraining | One selected clear-trained checkpoint per method; exact seed/selection metric not supplied here | All six AP metrics, one fixed checkpoint | 5.3; S5 Table S6 | R3 major 7; R4 2, 3, 16 |
| Frozen-checkpoint gate interventions | LGF, 3 datasets × 3 seeds × 8 interventions = 72 evaluations | Best validation APsmall; fixed learned branches and checkpoint | APsmall means/sample std; ideally six AP and provenance for every evaluation | 4.4, 5.5; S7 Table S7 | R4 9 |
| Recent detector validation | RetinaNet-based UGS reimplementation, YOLO11m, RT-DETR-L; three datasets × three seeds | Best APsmall for Table S10 | APsmall and mAP mean/sample std; per-seed metrics | 5.6; Table 3; S9 Table S10 | R1 recent literature; R3 major 2; R4 4, 8, 12 |
| Paired uncertainty for recent-detector rankings | LGF minus YOLO11m and LGF minus RT-DETR-L, three datasets | Selected validation checkpoints; later detector experiment | APsmall change in percentage points, 10,000 paired seed/image resamples, 95% percentile interval | 5.6; S9 Table S11 | R4 1, 4 |
| Condition-level AP | Baseline, LGF, UGS, YOLO11m, RT-DETR-L; synthetic fog/rain/snow, ACDC fog/night/rain/snow; three seeds | One best validation mAP checkpoint per detector/seed/dataset used in EVERY condition | Condition APsmall, mAP, pooled AP computed across all images first, then seed mean/sample std | 5.9 Table 7; S8 Tables S8–S9 | R3 major 6; R4 2, 5, 6, 16 |
| Scene leakage / weather pairing audit | Current train/validation annotations, 500 synthetic source scenes | Not applicable | No exact source-frame overlap; paired scene IDs across weather; counts and duplicate audit | 4.2; concluding S8 paragraph p.12 | R3 major 6; R4 5 |
| Matched gate visualizations | One synthetic-weather-trained seed-42 LGF best-APsmall checkpoint; four source scenes × clear/fog/rain/snow | One fixed checkpoint for all 16 inputs | Channel-mean local/global maps fixed scale 0–1, selected small GT boxes, object-level descriptive gate means | Discussion; S10 Figures S3–S6 and Table S12 | R4 10 |
| Saved-prediction outcome diagnostics | Baseline, LGF, UGS, YOLO11m, RT-DETR-L, all seeds | Best validation mAP; no retraining or new inference | GT, TP, FP, FN, evaluated detections, precision, recall, F1; overall/small/medium/large; mean/sample std | Discussion; S12.1 Tables S14–S16 | R4 15 |
| Balanced qualitative examples | Seed 42, same five models; shared image and target crop | Saved best-mAP predictions | LGF success/baseline miss; baseline success/LGF miss; shared miss; additional annotation-relative LGF FP | S12 Figures S7–S10 | R4 15 |
| Original single-insertion detector cost | Baseline, SE, CBAM-style, LGF | Original protocol | Params, FLOPs, batch16 throughput and amortized ms/image | 5.7 Table 4 | R4 9, 13 |
| C3/C4 placement comparison | Retained C3 primary references; C4 three seeds and three datasets | C4 synthetic selects original 500-fog-image subset, then pools 1500 weather images; selection rule for other C4 runs not specified | All six AP values for C4; C4–retained-C3 descriptive APsmall/mAP differences | 3.4, 5.8 Tables 5–6; S13 Table S17 | R4 11, 13 |
| Reference-image C3/C4 cost | Baseline, saved C3, saved C4 | Profile checkpoints independently of detector accuracy | Total params, convolution/linear GMACs at 3×800×1333; RTX5090 mentioned | Table 6; 5.8; S13 | R4 11, 13 |
| Independent test prediction export | Models/splits not specified in supplied PDFs | Exact setup unavailable | Predictions only, no reported test scores | Response R3 major 5 | R3 major 5 |

## Shared primary training setup supplied in the manuscript

- RetinaNet + ResNet-50 + FPN, COCO pretrained weights.
- Aspect-ratio preserving resize: shorter side 800 px, longer-side cap 1333 px.
- Replace the classification output for six target classes; bias uses prior probability 0.01, so bias is `-log((1-0.01)/0.01)`, not 0.01.
- Retain box-regression head and pretrained FrozenBatchNorm layers.
- LGF GroupNorm uses 32 groups.
- Five-level custom anchor sizes for P3–P7: `(32,48,64)`, `(64,96,128)`, `(128,192,256)`, `(256,384,512)`, `(384,512,640)`; aspect ratios 0.5, 1, 2.
- Focal loss; SGD momentum 0.9, weight decay 1e-4; LR 0.005.
- Two-epoch warmup; 80 epochs; LR drops at epochs 40 and 60. Drop factor is not explicitly given.
- Horizontal flips and brightness/contrast augmentation. Only synthetic-weather training receives weather transformations.
- No batch size, gradient accumulation, EMA, exact flip probability, brightness/contrast range, warmup start factor, focal alpha/gamma, pretrained weight enum, device/version, or deterministic setting is stated in current PDFs.
- YOLO11m and RT-DETR-L use native training setups and 1280×1280 inputs. Exact optimizer, batch, schedule, augmentations, output caps, pretraining artifacts, native package versions, and RT-DETR implementation are not supplied.
- UGS is the authors' RetinaNet-based reimplementation. No exact gradient stabilization formula, code, optimizer details, or hyperparameters appear in the PDFs.

## LGF formulas and specified design

For `X ∈ R^(C×H×W)`:

1. Local: `L = ReLU(GN32(Conv1x1(DWConv3x3(X))))`.
2. Global: `g = SiLU(GN32(Conv1x1(GAP(X))))`, broadcast `g` across H,W to `G`.
3. Default spatial gate bottleneck reduction r=4: `Z=ReLU(Conv_red1x1(X))`, `U=GN32(Conv_exp1x1(Z))`; split U into `U_local,U_global` of C channels each.
4. `tau=softplus(t)+1e-3`, learned scalar t; `w_local=sigmoid(U_local/tau)` and `w_global=sigmoid(U_global/tau)`. These are independent gates, no sum-to-one constraint.
5. `Y=w_local*L+w_global*G`; output `X+gamma*Y`, one scalar gamma per LGF block, initialized 0.1.
6. Softmax variant: two learnable scalar logits and shared positive temperature; global competitive branch weights `softmax([a_local,a_global]/tau)`.
7. Sum variant: `Y=L+G`.
8. Convolution biases, initialization, padding choices, gate t/logit initialization, exact route channel counts, and implementation naming/state_dict keys are not provided.
9. SE appears once at the same route. Its reduction ratio and activations are not specified.
10. CBAM-style appears once at the same route and uses smooth softmax pooling instead of literal max pooling and a 5×5 spatial kernel. Smooth-pooling formula/temperature and channel reduction details are not specified.

## Dataset construction and size records

- COCO train2017 and val2017 are separate sources. Select 2400 training and 500 validation source scenes within these partitions, then apply optional weather transforms. Split precedes weather generation.
- Each evaluated synthetic source scene has fog, rain, snow forms with shared annotations. Later condition evaluation uses 500 images per condition, 1500 total, from exactly 500 source scenes.
- The preparation notebook preserved weather-off output, not the exact historical weather-on invocation. Exact weather rendering parameters, package, RNG and historical training multiplicity are unresolved.
- ACDC uses supplied six-class mapping, custom aggregate training 2737 and validation 399. Validation counts: fog91, night94, rain112, snow102. Conditions do not share exactly matched scenes.
- S2 annotation overview: COCO train 11369 objects = small3556 + medium3725 + large4088; COCO val2514 = small816 + medium902 + large796. ACDC train24588 = small9058 + medium11384 + large4146; ACDC val2444 = small778 + medium1142 + large524.
- S1 image occurrences are per-class presence, so rows need not sum to image counts. COCO train person2183/car424/truck208/bicycle119/bus133/motorcycle106; val453/92/37/25/30/32. ACDC train1464/2679/833/610/423/449; val160/388/99/56/38/56.
- Outcome S14/S15 COCO GT snapshot differs: 2473 nonignored objects = small1078 + medium854 + large541. This differs from S2 object and size counts. Do not resolve by guessing.
- S12 states COCO-Weather diagnostic export has 1500 images, but Table S15 reports GT2473, identical to Table S14 clear GT. Source recovery resolves the actual export support: `balanced_failure_analysis_v1.zip/image_counts.csv` contains 500 unique weather images, all fog. Its manifest references the historical `mini_val2017_weather-500k6c.json`. Record the diagnostic support as 500 fog images when rerunning those saved exports. The pooled 1500-image condition AP is a separate workflow. See [reproducibility](reproducibility.md) for the manuscript/export mismatch.

## AP and AR conventions

- All displayed AP values use 0–1 except difference tables and bootstrap which use AP percentage points `100*(A-B)`.
- COCO-format bounding-box evaluator for the six categories. Standard AP includes AP[.50:.95], AP50, AP75, APsmall, APmedium, APlarge; AR includes maxDets1/10/100 and ARsmall/medium/large.
- Exact AP maxDets/score floor/model caps for saved outputs are not specified in PDFs. Use explicit recorded parameters in new outputs.
- Pooled AP is obtained by scoring all pooled images for each seed, then computing seed mean/std. Pooled AP is NOT mean condition AP.
- Where named, std is sample std across three seeds. Original summaries use rounded standard deviations. A rounded ±0.00 does not establish absence of seed variance.
- Selected validation checkpoint scores are not independent test performance. Best APsmall and best mAP exports must stay separate.

## Frozen checkpoint controls

- All branches and trained model parameters remain fixed. Only fusion weights change at inference.
- Eight settings: native learned maps; both maps 0.5; shuffled maps; swapped local/global maps; fixed alpha0.25; alpha0.5; alpha2.0; unscaled sum L+G.
- Shuffling granularity/RNG and fixed-alpha formula are not stated. Do not choose e.g. `(alpha,1-alpha)` versus `(alpha,1)` or blend formula without historical code.
- Unscaled sum changes residual update magnitude, so interpret separately from equal0.5/0.5.
- APsmall mean ± sample std: native clear0.2206±0.009/weather0.1847±0.002/acdc0.0839±0.004; uniform0.2140±0.013/0.1775±0.009/0.0617±0.003; shuffle0.2134±0.013/0.1779±0.009/0.0579±0.002; swap0.2160±0.013/0.1783±0.006/0.0633±0.004; alpha0.25 0.2139±0.012/0.1803±0.006/0.0592±0.002; alpha0.5 0.2102±0.016/0.1750±0.008/0.0599±0.001; alpha2 0.1598±0.006/0.1458±0.004/0.0243±0.009; sum0.1932±0.008/0.1692±0.010/0.0528±0.006.

## Gate visualization reproduction inputs

- Checkpoint: synthetic-weather-trained LGF, seed42, best APsmall.
- Scene IDs: 000000163117, 000000121242, 000000082812, 000000140840.
- One selected small annotated object per scene; exact annotation IDs/box coordinates not provided.
- Rows clear/fog/rain/snow, columns input/channel-mean local gate/channel-mean global gate; fixed zero-to-one color scale throughout.
- Table S12 means at four selected objects per condition: clear local0.4902/global0.4978; fog0.4901/0.4976; rain0.4906/0.4978; snow0.4912/0.4980. Do not substitute these descriptive values for causal interpretation or detection performance.

## Detection diagnostics and examples

- Reuse saved best-mAP predictions; no new training or inference.
- Confidence≥0.25, correct class, box IoU≥0.50.
- Score-ordered COCO bbox matching with ordinary objects one-to-one and COCO crowd/ignore behavior. At most100 detections per image per category plus saved detector-specific output caps.
- TP matched evaluated prediction; FP unmatched evaluated prediction; FN unmatched nonignored GT. Ignored predictions excluded. D=TP+FP, P=TP/D, R=TP/(TP+FN), F1=2PR/(P+R). TN is undefined in object detection without fixed finite background units.
- Exclusive diagnostic sizes: small annotation area<1024, medium1024≤area<9216, large≥9216, in original image coordinates. TP/FN use GT annotation area, FP uses predicted box w×h. Do not implement the diagnostic size partition by naïvely summing COCOeval independent area-range evaluations.
- S14 clear small TP/FP baseline437.3/112.0, LGF432.3/105.0; S15 synthetic baseline362.7/86.7, LGF384.7/96.3; S16 ACDC baseline268.0/257.3, LGF255.7/215.3.
- Figure S7: COCO-Weather image336658, filename000000336658_fog.jpg, small car annotation1066, area195.71; LGF TP score0.257, baseline miss.
- Figure S8: ACDC image120, small car annotation699, area814; baseline TP score0.343, LGF miss; YOLO11m and RT-DETR-L also match.
- Figure S9: COCO-Weather image387916, filename000000387916_fog.jpg, person annotation1135, annotation area696.96; baseline and LGF miss; RT-DETR-L matches.
- Figure S10: ACDC image23, LGF car prediction892 score0.267; annotation-relative FP over parked-vehicle cluster with valid detections nearby; no same-class baseline FP paired at box IoU≥0.50. This is an additional box, not proof of background hallucination or greater LGF aggregate FP total.

## Reference reported values

### Primary APsmall / mAP

| Dataset | Baseline | CBAM-style | SE | LGF |
|---|---|---|---|---|
| Clear COCO | 0.218/0.330 | 0.215/0.335 | 0.213/0.323 | 0.230/0.332 |
| Synthetic COCO | 0.177/0.280 | 0.167/0.287 | 0.166/0.281 | 0.189/0.283 |
| ACDC | 0.076/0.322 | 0.074/0.318 | 0.075/0.317 | 0.083/0.330 |

### Recent detector best-APsmall values

| Dataset | UGS APsmall/mAP | YOLO11m APsmall/mAP | RT-DETR-L APsmall/mAP |
|---|---|---|---|
| Clear | 0.1909/0.3227 | 0.1492/0.2360 | 0.1176/0.1886 |
| Synthetic | 0.1676/0.3041 | 0.1255/0.1984 | 0.0721/0.1545 |
| ACDC | 0.0800/0.3245 | 0.2644/0.4581 | 0.1769/0.3440 |

### Bootstrap intervals, percentage points

| Dataset | LGF–YOLO11m difference [95% interval] | LGF–RT-DETR-L difference [95% interval] |
|---|---|---|
| Clear | +7.142 [3.435,12.844] | +10.304 [5.396,15.546] |
| Synthetic | +5.918 [3.172,10.185] | +11.262 [7.556,14.942] |
| ACDC | −18.046 [−23.358,−13.464] | −9.300 [−14.735,−5.385] |

These intervals concern later detector rankings, NOT the retained primary baseline–LGF improvements. Matched primary gain has no paired uncertainty interval in the current submission.

### C4 all six metrics, seed mean/sample std

| Dataset | APsmall | APmedium | APlarge | AP50 | AP75 | mAP |
|---|---|---|---|---|---|---|
| Clear | 0.2257±0.004 | 0.3875±0.007 | 0.4148±0.006 | 0.6539±0.009 | 0.2725±0.002 | 0.3181±0.004 |
| Synthetic pooled1500 | 0.1788±0.002 | 0.3460±0.004 | 0.4067±0.006 | 0.5973±0.013 | 0.2543±0.012 | 0.2984±0.005 |
| ACDC | 0.0887±0.013 | 0.3304±0.009 | 0.5374±0.021 | 0.5545±0.009 | 0.3366±0.006 | 0.3233±0.008 |

C4–retained-C3 APsmall points: clear−0.43, synthetic−1.02, ACDC+0.57. mAP points: clear−1.39, synthetic+1.54, ACDC−0.67. Protocol differences preclude treating these as verified matched comparisons. No C2, C5, or multi-level accuracy conclusions are part of final submission.

### Cost protocols, keep separate

Original Table4: baseline32.25M/81.85GFLOPs/10.98ms amortized/91.05img/s; CBAM32.38/81.86/11.39/87.79; SE32.38/81.86/11.05/90.50; LGF35.16/84.81/11.64/85.93. Batch size16. LGF change+2.91M(+9.0%),+2.96GFLOPs(+3.6%),+6.0% amortized time. Input size, hardware, precision, FLOP convention and original warmup/repetitions are not established here.

Reference Table6: 3×800×1333 input, convolution/linear GMACs; baseline32.252M/214.988GMAC, saved C3 35.155M/222.735GMAC, saved C4 43.825M/222.719GMAC. C4 adds8.67M,24.66% versus C3 and35.88% versus baseline. RTX5090 is mentioned in Section5.7. Exact profiling implementation, preprocessing/padding, FP32 details, CUDA versions, and batch timing protocol require archived profile code.

Numerical parameter growth strongly warrants inspection of route/channel manifests before attempting an exact historical model reconstruction. Do not silently force standard C3/C4 routes to match reported run-label measurements.

## Unresolved facts required for exact historical release

1. Original GitHub repository/code and reviewer experiment archives; checkpoint/metric/export filenames and checksums.
2. Exact historical LGF insertion tensors (run label, module path, channel count, observed stride, transformation path).
3. Checkpoint-selection manifest linking each published table to actual checkpoint, seed, metric, validation snapshot and protocol. Legacy saver best-mAP versus selected best-APsmall is explicit, not an incidental naming detail.
4. Exact annotation snapshots/category map and reason S2/S14 object-size counts differ; whether S15 is per-condition normalized or uses a different snapshot.
5. Historical weather generation invocation, transforms, strengths, seeds and training image multiplicity.
6. UGS reimplementation equations and hyperparameters. PDFs alone are insufficient to call a new arbitrary gradient scaler the authors' UGS reimplementation.
7. Frozen alpha formula, shuffle axes and randomization behavior.
8. Canonical/native detector package versions, training config, output caps, score floors and NMS settings.
9. Bootstrap implementation: seed resampling nesting, image or source-scene clustering for synthetic pairs, RNG and exact checkpoint IDs.
10. Gate visualization selected annotation IDs, native gate hook point and object aggregation definition.
11. Profiling input pre/post transforms, padding, FLOP/MAC counting convention and exact warmup/repetition/device/precision.
12. Dataset/checkpoint distribution rights and author-selected license. Avoid imposing a new license on recovered upstream code without checking its existing license.

## Manuscript/response document discrepancies worth recording, not silently correcting

- Response R4 comment11 cites Supplementary Tables S17–S18 and full profile. Supplied manuscript includes Table S17 on p.25 and Table6 in main text; no TableS18 appears.
- Section5.7 sentence refers to “Supplementary Table6” although reference-image profile is main-text Table6.
- Several response locations call Table7 p.7 although extracted manuscript prints Table7 on p.6. These are documentation locations, not algorithm facts.
- Native controls APsmall differs from primary LGF APsmall despite both APsmall-selection descriptions, so archived checkpoint and protocol lineage is needed.
- Paper presents two COCO object-size histograms for separate snapshots. Do not assert one common annotation identity until data artifacts establish this.

## Release implementation recommendations

- A complete source-backed release should contain the historical detector/attention code plus separate reviewer experiment CLIs, not a single training script labeled as all experiments.
- Every run export should include resolved config, package versions, git commit, seed, dataset annotation SHA256, preprocessing, numeric category mapping, run label AND observed route, checkpoint SHA256, selection metric and evaluator settings.
- Preserve reported reference numbers in a labelled transcription file. New code outputs must never be prefilled with published scores.
- Separate example/demo pipelines from historical reproducibility status. If reconstructing any missing script from equations, mark it as a reconstruction with explicit unresolved parameters.
