# UGS reimplementation: method and comparison notes

Version 1.0.0. Prepared 2026-09-14 for the supplied Paper 1 RetinaNet/LGF code.

## Attribution and status

The method target is Sun et al., [Uncertainty-Aware Gradient Stabilization for Small Object Detection, ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/papers/Sun_Uncertainty-Aware_Gradient_Stabilization_for_Small_Object_Detection_ICCV_2025_paper.pdf). The accessible primary manuscript used for equations was [arXiv:2303.01803v2](https://arxiv.org/html/2303.01803v2); the [ICCV supplement](https://openaccess.thecvf.com/content/ICCV2025/supplemental/Sun_Uncertainty-Aware_Gradient_Stabilization_ICCV_2025_supplemental.pdf) was also inspected. No author-released UGS repository was verified through the paper, author website, or public repository searches on 2026-09-14. This is a bounded search result, not proof that no implementation exists.

The paper motivates distributional localization with uncertainty minimization and refinement. This package adapts those components to the user's horizontal Torchvision RetinaNet. Its mathematical and software choices below are independently specified. It does not reproduce the paper's MMDetection experiments or establish their numerical results. Use the label **“UGS (our RetinaNet reimplementation)”** in a comparison table.

## Implemented objective

The detector keeps native RetinaNet positive-anchor matching and focal classification loss. Four encoded box offsets are represented by categorical distributions. The expected offsets feed the original box decoder and postprocessing.

For each coordinate, the grid is generated from `n+1` uniformly spaced values on `[-alpha, alpha]`, transformed using the exponential nonuniform mapping associated with Eq. 9. `beta=0` is handled by its uniform-grid limit. The supplied configuration has `alpha=5`, `beta=1`, `n=10`, hence 11 support points per coordinate.

An encoded target between adjacent grid values `g_l` and `g_r` uses weights:

```text
w_l = (g_r - target) / (g_r - g_l)
w_r = (target - g_l) / (g_r - g_l)
```

These normalized targets preserve the expectation of every in-range offset. The printed interpolation expression associated with Eq. 8 appears inconsistent with normalized interpolation, particularly after nonuniform remapping. We explicitly choose the barycentric construction above. Out-of-range targets are clipped to the grid endpoints and counted in the training diagnostics.

Localization cross-entropy is `CE = -sum(q * log p)`. Uncertainty minimization uses categorical entropy `H = -sum(p * log p)`. Autograd differentiates entropy through softmax. The actual logit derivative is `-p_i * (log(p_i) + H)`; we do not manually substitute the incomplete logit derivative printed with Eq. 13.

For refinement, the code obtains the gradient of positive-anchor entropy with respect to each FPN feature tensor. Per image and FPN level it forms:

```text
epsilon = rho * grad_H / max(L2_norm(grad_H), 1e-12)
```

The direction is detached. A second localization pass computes entropy using `feature + epsilon`. The outer loss still differentiates through the feature and localization tower. This is a first-order adversarial refinement choice. Feature regularization is the mean-square magnitude of the unperturbed features on image/level pairs containing positives.

The implementation's total training objective is:

```text
native classification focal loss
+ localization CE
+ um_weight * positive-anchor entropy
+ ur_weight * (perturbed positive-anchor entropy
               + feature_reg_weight * feature mean-square penalty)
```

Localization CE and both entropy terms sum over the four coordinates, divide by the image's positive-anchor count, and average over images. Images with no positives contribute zero localization loss. Feature regularization sums eligible FPN-level means and averages over images. FPN order, anchor order, matching, and classification are inherited unchanged.

## Assumptions requiring disclosure

| Choice | Supplied implementation | Status |
| --- | --- | --- |
| Detector integration | Horizontal Torchvision RetinaNet, all P3–P7 regression features | Independent adaptation to the supplied detector |
| Grid | `alpha=5`, `beta=1`, 10 intervals | Explicit starting choice; no canonical author RetinaNet recipe verified |
| Target interpolation | Normalized weights on actual adjacent support points | Resolution of an apparent printed inconsistency |
| UM/UR sampling | Positive anchors only | Explicit reduction choice |
| UM weight | 0.1 | Unverified starting value |
| UR weight | 0.1 | Unverified starting value |
| Perturbation radius | 0.05 per image and FPN level | Unverified starting value and normalization scope |
| Feature regularizer weight | 0.1 inside UR | Explicit separation of the outer and inner coefficients |
| Minimax approximation | One normalized ascent direction, detached | First-order implementation choice |
| Feature regularizer | Mean-square features; sum eligible levels | Explicit reduction choice |
| Out-of-support targets | Clip and record counts | Explicit range-handling choice |

The paper does not give enough detail in the inspected sources to establish a unique recipe for all of these choices in this detector. Do not describe the supplied values as author defaults. A negative result under this recipe is not sufficient evidence that published UGS is ineffective. Obtain author code/settings if available, or disclose and justify the adaptation and any tuning budget in the revision.

Setting `ur_weight=0` disables UR; setting `um_weight=0` disables the direct UM term. Labels change accordingly. A CE-only ablation should not be presented as complete UGS.

## Relationship to the original LGF code

`vendor/original_core.py` is the exact source of zero-based cell 1 in the uploaded `2.train_evaluate_controls.ipynb`. Its SHA-256 is recorded in `vendor/source_identity.json` and checked on import. The notebook itself is untouched by this package. The archived cell's old training/launch functions are never invoked.

The new bridge calls the original `build_model`, uses the original six-class classification initialization, and preserves the original LGF block and level mapping. For UGS it starts from the original baseline: the regression convolution tower is copied, and the terminal four-offset convolution is replaced with a newly initialized distribution convolution. Classification, backbone/FPN, anchors, box encoding/decoding, and NMS follow the original model. The distribution head requires new terminal weights; it cannot reuse the four-output regression kernel directly.

The pretrained source is `RetinaNet_ResNet50_FPN_Weights.COCO_V1`. Offline checkpoint reconstruction reproduces the pretrained factory's FrozenBatchNorm schema. It does not download weights because the checkpoint supplies them. Integration was checked against [Torchvision 0.25 RetinaNet source](https://docs.pytorch.org/vision/0.25/_modules/torchvision/models/detection/retinanet.html).

### Separate runner: preserved choices and disclosed differences

Preserved choices include the prepared datasets and category mapping; original augmentation definitions; actual transform min/max size 800/1333; custom anchors; SGD parameters; effective batch size 16; first five epochs with the backbone frozen; optimizer/scheduler reset when unfreezing; EMA 0.9999; gradient clipping at 10; and native COCO evaluation every five epochs plus the final epoch.

The original warmup multiplies the current learning rate: the first epoch halves it, and the second multiplies it by one. It stays at half the base value until the epoch-index-5 optimizer reset. The MultiStepLR schedule also restarts at unfreezing. This runner preserves that actual behavior rather than replacing it with the prose description of the schedule.

Operational differences are confined to this separate runner:

- FP32 is the default. The original CUDA loop enables AMP. `--amp` is available, but CUDA and AMP were not validated locally. Choose the same precision for new controls and UGS, and disclose any difference from historical LGF training.
- Albumentations generators are explicitly seeded per run and worker. The original seeding fallback does not establish an identical augmentation sequence under Albumentations 2. This is not bitwise old-run reproduction.
- Missing or undecodable images raise errors instead of becoming substitute black images. Input hashing and category/duplicate checks are added. The wrapper's decode check also adds data-loading work.
- Both best-mAP and best-APsmall checkpoints are saved, along with epoch-boundary resumable state. The old checkpoint/aggregation selection mismatch is not silently carried into a comparison.
- The new loop writes JSON diagnostics and exported detections instead of using the old NAS loggers or running the old per-validation timing/complexity routines. No speed claim is produced by this package.
- Resume is guarded against changed data, configuration, source, or environment. Multiworker restart is reproducible at the saved-state level but not guaranteed to reproduce an uninterrupted augmentation sequence.

The source's original C3/C4 mapping is deliberately retained. No architecture, placement, or original notebook fix is included.

## What a defensible comparison needs

First check that the new method trains on the same prepared images and that the selected checkpoint corresponds to the stated rule. Use the same three seeds, validation files, evaluator, image sizes, and precision policy. Report APsmall plus overall and other size APs from actual detections; present the 0–1 or percentage scale consistently.

Existing LGF checkpoints can save training time when their selection and training provenance are recoverable. Otherwise, running the unchanged LGF model through this same new runner provides a cleaner control for its operational differences. The comparison utility checks evaluation compatibility, not historical training equivalence.

The supplied metrics are validation metrics and checkpoint selection uses that validation set. They are not untouched held-out test estimates. A separate test evaluation or a transparently described validation protocol is needed for stronger generalization claims. Do not paste UGS's published scores from different datasets/class sets into this six-class comparison.

Software tests establish tensor arithmetic, gradient flow, export correctness, and checkpoint recovery. They do not establish benchmark fidelity, convergence on the real datasets, or superiority over LGF.
