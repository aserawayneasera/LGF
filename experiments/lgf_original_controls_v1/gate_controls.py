"""Reversible inference-only interventions on the original StableLGFBlock.

paper: equations printed in Table S6's description.
notebook: the retrieved 2.train_evaluate_controls.ipynb implementation.
Neither suite changes parameters, temperature, residual gamma, or training code.
"""
import contextlib
import types

PAPER = ('native', 'uniform', 'shuffle', 'swap', 'alpha_0.25', 'alpha_0.5',
         'alpha_2.0', 'no_gate')
NOTEBOOK = ('native', 'global_mean', 'uniform', 'shuffle', 'swap',
            'alpha_0.25', 'alpha_0.5', 'alpha_2.0', 'no_module')


def names(suite):
    if suite not in ('paper', 'notebook'):
        raise ValueError('Unknown control suite')
    return PAPER if suite == 'paper' else NOTEBOOK


def definition(suite, control):
    if control not in names(suite):
        raise ValueError(f'{control} does not belong to {suite}')
    base = {
        'native': 'Original forward; Y = wL*L + wG*G',
        'uniform': 'Y = 0.5*L + 0.5*G',
        'shuffle': 'Joint permutation of batch*channels, same permutation for wL and wG; no spatial permutation',
        'swap': 'Y = wG*L + wL*G',
        'no_gate': 'Y = L + G; gate head is not executed',
        'no_module': 'Output = X; entire LGF block bypassed',
        'global_mean': 'Per-image means of each native gate over channels and spatial dimensions',
    }
    if control.startswith('alpha_'):
        alpha = float(control.split('_')[1])
        text = (f'Y = {alpha}*L + G; gate head is not executed' if suite == 'paper'
                else f'wL *= {alpha}; divide both by (wL+wG+1e-6)')
    else:
        text = base[control]
    if suite == 'notebook' and control not in ('native', 'no_module'):
        text += '; then scale both weights by mean(native sum)/(mean(control sum)+1e-6) per image'
    return text + '; preserve learned residual gamma'


def blocks(model):
    found = [(n, m) for n, m in model.named_modules()
             if type(m).__name__ == 'StableLGFBlock']
    if not found:
        raise ValueError('No original StableLGFBlock found; this evaluator accepts lgf_original only')
    for name, m in found:
        if (m.gating_type != 'gated_spatial' or tuple(m.branches) != ('local', 'global')
                or m.local is None or m.global_branch is None):
            raise ValueError(f'{name}: expected both branches and gated_spatial')
        for attr in ('gate_reduce', 'gate_expand', 'gate_norm', 'temperature', 'gamma'):
            if not hasattr(m, attr):
                raise ValueError(f'{name}: missing {attr}')
    return found


def native_weights(block, x):
    import torch
    from torch.nn import functional as F
    h = F.relu(block.gate_reduce(x), inplace=True)
    logits = block.gate_norm(block.gate_expand(h))
    # This runner uses FP32 inference, as the delivered evaluator does.
    logits32 = logits.float().clamp_(-15, 15)
    tau = F.softplus(block.temperature.float()) + 1e-3
    w = torch.sigmoid(logits32 / tau).to(dtype=x.dtype)
    n, twice_c, height, width = w.shape
    w = w.view(n, 2, twice_c // 2, height, width)
    return w[:, 0], w[:, 1]


def transformed_forward(block, x, control, suite, generator):
    import torch
    if control == 'no_module':
        return x
    local = block.local(x)
    global_ = block.global_branch(x).expand_as(x)
    if suite == 'paper' and control in ('uniform', 'no_gate'):
        out = 0.5 * local + 0.5 * global_ if control == 'uniform' else local + global_
        return x + block.gamma * out
    if suite == 'paper' and control.startswith('alpha_'):
        return x + block.gamma * (float(control.split('_')[1]) * local + global_)
    wl0, wg0 = native_weights(block, x)
    wl, wg = wl0, wg0
    if control == 'uniform':
        wl, wg = torch.full_like(wl, 0.5), torch.full_like(wg, 0.5)
    elif control == 'global_mean':
        wl = wl.mean(dim=(1, 2, 3), keepdim=True).expand_as(wl)
        wg = wg.mean(dim=(1, 2, 3), keepdim=True).expand_as(wg)
    elif control == 'swap':
        wl, wg = wg, wl
    elif control == 'shuffle':
        n, c, h, w = wl.shape
        permutation = torch.randperm(n*c, generator=generator, device=x.device)
        wl = wl.reshape(n*c, h, w)[permutation].reshape(n, c, h, w)
        wg = wg.reshape(n*c, h, w)[permutation].reshape(n, c, h, w)
    elif control.startswith('alpha_'):
        wl = wl * float(control.split('_')[1])
        denom = wl + wg + 1e-6
        wl, wg = wl / denom, wg / denom
    elif control not in ('native', 'native_reconstructed'):
        raise ValueError(control)
    if suite == 'notebook' and control not in ('native', 'native_reconstructed'):
        native_sum = (wl0 + wg0).mean(dim=(1, 2, 3), keepdim=True)
        control_sum = (wl + wg).mean(dim=(1, 2, 3), keepdim=True) + 1e-6
        scale = native_sum / control_sum
        wl, wg = wl * scale, wg * scale
    return x + block.gamma * (wl * local + wg * global_)


@contextlib.contextmanager
def apply_control(model, control, suite='paper', shuffle_seed=123, verify_native=False):
    """Restore forwards even on errors. Native uses the unmodified original forward."""
    import torch
    if control not in names(suite) and control != 'native_reconstructed':
        raise ValueError(control)
    found = blocks(model)
    generator = torch.Generator(device=next(model.parameters()).device)
    generator.manual_seed(shuffle_seed)
    restored = []
    evidence = {'verified_blocks': [], 'block_shapes': {}}
    try:
        for name, block in found:
            had_instance = 'forward' in block.__dict__
            old_instance = block.__dict__.get('forward')
            original = block.forward
            restored.append((block, had_instance, old_instance))
            def wrapped(self, x, _name=name, _original=original):
                evidence['block_shapes'][_name] = list(x.shape)
                if control == 'native':
                    out = _original(x)
                    if verify_native and _name not in evidence['verified_blocks']:
                        reconstructed = transformed_forward(self, x, 'native_reconstructed', 'paper', generator)
                        torch.testing.assert_close(out, reconstructed, rtol=0, atol=0)
                        evidence['verified_blocks'].append(_name)
                    return out
                return transformed_forward(self, x, control, suite, generator)
            block.forward = types.MethodType(wrapped, block)
        yield evidence
    finally:
        for block, had_instance, old_instance in restored:
            if had_instance:
                block.forward = old_instance
            else:
                delattr(block, 'forward')


def parameter_cost(model):
    total = sum(p.numel() for p in model.parameters())
    gate = 0
    detail = []
    for name, m in blocks(model):
        count = sum(p.numel() for attr in ('gate_reduce', 'gate_expand', 'gate_norm')
                    for p in getattr(m, attr).parameters()) + m.temperature.numel()
        gate += count
        detail.append({'block': name, 'channels': m.gate_reduce.in_channels,
                       'gate_parameters': count, 'learned_gamma': float(m.gamma.detach().cpu())})
    return {'total_parameters': total, 'gate_parameters': gate,
            'parameters_if_gate_head_deleted': total - gate, 'blocks': detail,
            'note': 'Weights remain in memory for reversible controls; no_gate skips execution. '
                    'The parameter difference describes deleting the unused gate head.'}
