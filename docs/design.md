# Design

- LGF sits at C3 before FPN.
- Local branch: depthwise 3×3 then 1×1, GroupNorm, ReLU.
- Global branch: GAP, 1×1, GroupNorm, SiLU, broadcast to H×W.
- Gating: per-pixel and per-channel sigmoid weights. Output is a weighted sum of the branches. Residual adds back to the input.
- Choose sigmoid gating for best AP_small in ablations. Sum and softmax are close.

Why C3
- Small objects map best to P3 in FPN. C3 feeds P3. Early fusion helps the pyramid propagate better signals for small targets.

Complexity
- +9% params, +3.6% FLOPs, +6% latency at 640px vs baseline.
