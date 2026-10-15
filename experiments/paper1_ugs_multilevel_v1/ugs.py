"""Independent UGS reimplementation for the original Torchvision RetinaNet.

Read METHOD_NOTES.md before using the results in a manuscript. This is not
author-provided code. Unspecified choices are explicit in UGSConfig.
"""
from dataclasses import asdict, dataclass
import copy
import math

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class UGSConfig:
    alpha: float = 5.0
    beta: float = 1.0
    intervals: int = 10
    um_weight: float = 0.1
    ur_weight: float = 0.1
    rho: float = 0.05
    feature_reg_weight: float = 0.1

    def __post_init__(self):
        for key, value in asdict(self).items():
            if not math.isfinite(value):
                raise ValueError(f"{key} must be finite")
        if self.alpha <= 0 or self.beta < 0 or self.alpha * self.beta > 80:
            raise ValueError("Require alpha>0, beta>=0, alpha*beta<=80")
        if not isinstance(self.intervals, int) or self.intervals < 2:
            raise ValueError("intervals must be an integer >=2")
        if min(self.um_weight, self.ur_weight, self.rho, self.feature_reg_weight) < 0:
            raise ValueError("Loss weights and rho must be nonnegative")

    @property
    def label(self):
        components = ["CE"]
        if self.um_weight > 0:
            components.append("UM")
        if self.ur_weight > 0:
            components.append("UR")
        return "UGS reimplementation (RetinaNet; " + "+".join(components) + ")"


def nonuniform_grid(config, *, dtype=torch.float32):
    """Equation 9, with a continuous uniform-grid limit for beta=0."""
    y = torch.linspace(-config.alpha, config.alpha, config.intervals + 1, dtype=torch.float64)
    if config.beta > 0:
        y = y.sign() * config.alpha * torch.expm1(config.beta * y.abs()) / math.expm1(
            config.alpha * config.beta
        )
    y[0], y[-1] = -config.alpha, config.alpha
    if config.intervals % 2 == 0:
        y[config.intervals // 2] = 0
    return y.to(dtype)


def two_hot(targets, grid):
    """Barycentric two-hot targets on the actual nonuniform support.

    The two weights sum to one and reconstruct the in-range target exactly.
    Outside-range values are clipped explicitly and counted by the caller.
    This resolves the inconsistent printed interpolation expression; see notes.
    """
    if not torch.isfinite(targets).all():
        raise FloatingPointError("Nonfinite box-regression targets")
    t = targets.clamp(grid[0], grid[-1])
    right = torch.searchsorted(grid, t.contiguous()).clamp(1, grid.numel() - 1)
    left = right - 1
    wr = (t - grid[left]) / (grid[right] - grid[left])
    return left, right, 1.0 - wr, wr


def distribution_terms(logits, targets, grid):
    """Return per-coordinate CE and entropy, with stable FP32 arithmetic."""
    logp = F.log_softmax(logits.float(), dim=-1)
    left, right, wl, wr = two_hot(targets.float(), grid.float())
    ce = -(wl * logp.gather(-1, left.unsqueeze(-1)).squeeze(-1)
           + wr * logp.gather(-1, right.unsqueeze(-1)).squeeze(-1))
    entropy = -(logp.exp() * logp).sum(-1)
    return ce, entropy


def expectation(logits, grid):
    return (F.softmax(logits.float(), dim=-1) * grid.float()).sum(-1)


def perturbation(gradient, rho):
    """Equation 15, per image and per FPN level; stop gradient through epsilon."""
    g = gradient.detach().float()
    if not torch.isfinite(g).all():
        raise FloatingPointError("Nonfinite uncertainty gradient")
    norm = torch.linalg.vector_norm(g.flatten(1), dim=1).reshape(-1, 1, 1, 1)
    return rho * g / norm.clamp_min(1e-12)


class DistributionRegressionHead(nn.Module):
    def __init__(self, original_regression_head, num_anchors, config):
        super().__init__()
        self.conv = copy.deepcopy(original_regression_head.conv)
        self.box_coder = copy.deepcopy(original_regression_head.box_coder)
        self.num_anchors = num_anchors
        self.num_bins = config.intervals + 1
        in_channels = original_regression_head.bbox_reg.in_channels
        self.bbox_distribution = nn.Conv2d(in_channels, num_anchors * 4 * self.num_bins, 3, padding=1)
        nn.init.normal_(self.bbox_distribution.weight, std=0.01)
        nn.init.zeros_(self.bbox_distribution.bias)
        self.register_buffer("grid", nonuniform_grid(config))

    def forward_level(self, feature):
        x = self.bbox_distribution(self.conv(feature))
        b, _, h, w = x.shape
        # Channel order A, coordinate, bin. Anchor order H,W,A matches Torchvision.
        return x.reshape(b, self.num_anchors, 4, self.num_bins, h, w).permute(
            0, 4, 5, 1, 2, 3
        ).reshape(b, h * w * self.num_anchors, 4, self.num_bins)

    def forward(self, features):
        return torch.cat([self.forward_level(x) for x in features], dim=1)


class UGSHead(nn.Module):
    """Use the native RetinaNet matching, classification, decoding and NMS.

    Only the localization head/objective changes. Transient features are consumed
    by compute_loss and never included in checkpoints or retained for inference.
    """
    def __init__(self, original_head, config):
        super().__init__()
        self.config = config
        self.classification_head = original_head.classification_head
        self.regression_head = DistributionRegressionHead(
            original_head.regression_head, self.classification_head.num_anchors, config
        )
        self._features = None
        self.last_stats = {}

    def forward(self, features):
        if self.training:
            # During the original head-only warm-up the backbone is frozen.
            # Leaf features allow the UR direction without unfreezing its weights.
            features = [x if x.requires_grad else x.detach().requires_grad_(True) for x in features]
            self._features = features
        else:
            self._features = None
        logits = self.regression_head(features)
        outputs = {"cls_logits": self.classification_head(features),
                   "bbox_regression": expectation(logits, self.regression_head.grid)}
        if self.training:
            outputs["ugs_logits"] = logits
        return outputs

    def compute_loss(self, targets, head_outputs, anchors, matched_idxs):
        features, self._features = self._features, None
        if features is None:
            raise RuntimeError("UGS training features are unavailable")
        classification = self.classification_head.compute_loss(targets, head_outputs, matched_idxs)
        logits = head_outputs["ugs_logits"]
        grid = self.regression_head.grid
        bsize = len(targets)
        zero = logits.float().sum() * 0.0
        ce, um = zero, zero
        positives = [torch.where(m >= 0)[0] for m in matched_idxs]
        positive_count = sum(p.numel() for p in positives)
        clipped = 0
        for b, ids in enumerate(positives):
            if ids.numel() == 0:
                continue
            encoded = self.regression_head.box_coder.encode_single(
                targets[b]["boxes"][matched_idxs[b][ids]], anchors[b][ids]
            ).float()
            clipped += int(((encoded < grid[0]) | (encoded > grid[-1])).sum().detach())
            ce_i, um_i = distribution_terms(logits[b, ids], encoded, grid)
            ce = ce + ce_i.sum() / ids.numel() / bsize
            um = um + um_i.sum() / ids.numel() / bsize

        ur_entropy, feature_reg = zero, zero
        max_norm, nonzero_directions = 0.0, 0
        if self.config.ur_weight > 0 and positive_count:
            # Scale this local gradient query to reduce FP16 underflow under AMP.
            # GradScaler's optimizer scaling is independent of this calculation.
            gradients = torch.autograd.grad(um * 1024.0, features, retain_graph=True,
                                            create_graph=False, allow_unused=True)
            offset = 0
            for feature, gradient in zip(features, gradients):
                n = feature.shape[-2] * feature.shape[-1] * self.regression_head.num_anchors
                if gradient is None:
                    gradient = torch.zeros_like(feature)
                epsilon = perturbation(gradient.float() / 1024.0, self.config.rho)
                norms = torch.linalg.vector_norm(epsilon.flatten(1), dim=1)
                max_norm = max(max_norm, float(norms.max().detach()))
                nonzero_directions += int((norms > 0).sum().detach())
                # The perturbation is detached; the outer loss retains gradients
                # through the feature and localization tower (first-order UR).
                shifted = self.regression_head.forward_level(feature + epsilon.to(feature.dtype))
                logp = F.log_softmax(shifted.float(), dim=-1)
                ent = -(logp.exp() * logp).sum(-1)
                for b, ids in enumerate(positives):
                    local = ids[(ids >= offset) & (ids < offset + n)] - offset
                    if local.numel():
                        ur_entropy = ur_entropy + ent[b, local].sum() / ids.numel() / bsize
                        # Explicit reduction choice: mean-square feature magnitude
                        # on levels/images with positive anchors (not an opaque sum).
                        feature_reg = feature_reg + feature[b].float().square().mean() / bsize
                offset += n
            if offset != logits.shape[1]:
                raise RuntimeError("FPN anchor order/length mismatch")

        self.last_stats = {
            "positive_anchors": positive_count,
            "regression_coordinates": positive_count * 4,
            "clipped_coordinates": clipped,
            "max_perturbation_l2": max_norm,
            "nonzero_perturbation_directions": nonzero_directions,
            "ce_unweighted": float(ce.detach()), "um_unweighted": float(um.detach()),
            "ur_entropy_unweighted": float(ur_entropy.detach()),
            "feature_reg_unweighted": float(feature_reg.detach()),
        }
        return {"classification": classification,
                "ugs_ce": ce,
                "ugs_um": self.config.um_weight * um,
                "ugs_ur": self.config.ur_weight * (
                    ur_entropy + self.config.feature_reg_weight * feature_reg)}


def attach_ugs(model, config):
    model.head = UGSHead(model.head, config)
    return model
