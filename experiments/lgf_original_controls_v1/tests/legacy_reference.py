"""Verbatim reference functions from the original user notebook; test use only."""
def _native_spatial_weights(block, x):
    # Compute native gated_spatial weights wL, wG using the block's own layers
    h = F.relu(block.gate_reduce(x), inplace=True)
    logits = block.gate_expand(h)
    logits = block.gate_norm(logits)
    with torch.cuda.amp.autocast(enabled=False):
        logits32 = logits.float().clamp_(-15, 15)
        tau = F.softplus(block.temperature.float()) + 1e-3
        w = torch.sigmoid(logits32 / tau)
    N, twoC, H, W = w.shape
    C = twoC // 2
    w = w.to(dtype=x.dtype).view(N, 2, C, H, W)
    return w[:,0], w[:,1]  # wL, wG

def _patch_gate_override(
    model,
    override="native",
    *,
    gate_seed=None,
    gamma_mult=None,
    temp_mult=None,
    temp_set=None,
    alpha_local=None,            # α for the α‑sweep: scales local gates before renorm
    global_mean_mode="scalar"    # "scalar" = per-image scalar, "channel" = per-channel
):
    """
    override options:
      "native"               learned gate maps as-is
      "global_mean"          per-image scalar gate (paper); set global_mean_mode to "channel" to match old per-channel behavior
      "uniform"              wL=wG=0.5
      "shuffle"              shuffle gates across images AND channels
      "swap_lg"              swap learned local/global gate assignments
      "no_module"            remove module, return x
      "only_local"           wL=1, wG=0
      "only_global"          wL=0, wG=1
    Optional knobs:
      alpha_local            α-sweep; multiply local gates by α then renormalize with global
      temp_mult/temp_set     gate temperature tweaks
      gamma_mult             multiplicatively scale learned residual γ (do not overwrite)
    """
    device = next(model.parameters()).device
    rng = torch.Generator(device=device)
    if gate_seed is not None:
        rng.manual_seed(int(gate_seed))

    for m in model.modules():
        if m.__class__.__name__ != "StableLGFBlock":
            continue

        # keep learned gamma, allow multiplicative scaling only
        if gamma_mult is not None:
            with torch.no_grad():
                m.gamma.mul_(float(gamma_mult))

        # optional temperature probe
        if hasattr(m, "temperature"):
            with torch.no_grad():
                if temp_mult is not None:
                    m.temperature.mul_(float(temp_mult))
                if temp_set is not None:
                    m.temperature.fill_(float(temp_set))

        if override == "native" and alpha_local is None:
            continue

        if not hasattr(m, "_orig_forward"):
            m._orig_forward = m.forward

        def wrapped_forward(self, x, *a, **k):
            # must be the spatial gate for these controls to mean anything
            if getattr(self, "gating_type", None) != "gated_spatial":
                # fall back to original forward to avoid misleading results
                return self._orig_forward(x, *a, **k) if hasattr(self, "_orig_forward") else super(type(self), self).forward(x, *a, **k)

            # compute branches
            L = self.local(x) if getattr(self, "local", None) is not None else None
            G = None
            if getattr(self, "global_branch", None) is not None:
                g = self.global_branch(x)
                G = g.expand_as(L if L is not None else g)

            # kill the whole module if asked
            if override == "no_module":
                return x

            # degenerate cases
            if L is None or G is None:
                out_core = L if L is not None else G
                return x + self.gamma * out_core

            # native spatial gates
            wL_native, wG_native = _native_spatial_weights(self, x)  # assumes these sum to ~1

            # build wL, wG
            wL, wG = wL_native, wG_native

            if override == "only_local":
                wL = torch.ones_like(wL_native, 1)
                wG = torch.zeros_like(wG_native, 0)

            elif override == "only_global":
                wL = torch.zeros_like(wL_native, 0)
                wG = torch.ones_like(wG_native, 1)

            elif override == "uniform":
                wL = torch.full_like(wL_native, 0.5)
                wG = torch.full_like(wG_native, 0.5)

            elif override == "global_mean":
                if global_mean_mode == "scalar":
                    # per-image scalar (paper): mean over channels and spatial
                    wL_s = wL_native.mean(dim=(1,2,3), keepdim=True)
                    wG_s = wG_native.mean(dim=(1,2,3), keepdim=True)
                else:
                    # legacy per-channel: mean over spatial only
                    wL_s = wL_native.mean(dim=(2,3), keepdim=True)
                    wG_s = wG_native.mean(dim=(2,3), keepdim=True)
                wL = wL_s.expand_as(wL_native)
                wG = wG_s.expand_as(wG_native)

            elif override == "shuffle":
                # shuffle across images AND channels with a single permutation applied to both gates
                N, C, H, W = wL_native.shape
                if N*C > 1:
                    idx = torch.randperm(N*C, generator=rng, device=wL_native.device)
                    wL = wL_native.reshape(N*C, H, W)[idx].reshape(N, C, H, W)
                    wG = wG_native.reshape(N*C, H, W)[idx].reshape(N, C, H, W)

            elif override == "swap_lg":
                wL, wG = wG_native, wL_native

            # α‑sweep on top of whatever override we used
            if alpha_local is not None:
                wL = wL * float(alpha_local)
                denom = wL + wG + 1e-6
                wL = wL / denom
                wG = wG / denom

            # --- budget preservation: match sum of weights to native ---
            # compute native sum budget (choose per-image scalar; switch to per-channel if you prefer)
            s_native = (wL_native + wG_native).mean(dim=(1,2,3), keepdim=True)  # [N,1,1,1]
            s_override = (wL + wG).mean(dim=(1,2,3), keepdim=True) + 1e-6
            scale = s_native / s_override
            wL = wL * scale
            wG = wG * scale
            # --- end budget preservation ---

            out_core = wL * L + wG * G
            return x + self.gamma * out_core

        m.forward = types.MethodType(wrapped_forward, m)

