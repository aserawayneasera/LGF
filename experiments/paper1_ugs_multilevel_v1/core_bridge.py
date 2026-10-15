"""Load Wayne's unmodified model/data definitions without running notebook cells."""
import contextlib
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

_CORE = None
ROOT = Path(__file__).resolve().parent


def load_core():
    global _CORE
    if _CORE is not None:
        return _CORE
    os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
    os.environ.setdefault("TORCH_HOME", "/nas.dbms/asera/paper1_original/torch_cache")
    path = ROOT / "vendor/original_core.py"
    identity = json.loads((ROOT / "vendor/source_identity.json").read_text())
    if hashlib.sha256(path.read_bytes()).hexdigest() != identity["source_sha256"]:
        raise RuntimeError("The archived original model source was modified")
    spec = importlib.util.spec_from_file_location("paper1_frozen_original", path)
    module = importlib.util.module_from_spec(spec)
    old_argv = sys.argv
    try:
        sys.argv = [str(path)]
        spec.loader.exec_module(module)
    finally:
        sys.argv = old_argv
    _CORE = module
    return module


@contextlib.contextmanager
def factory_without_download(core):
    """Reconstruct the pretrained model's FrozenBN layout without downloading weights."""
    import torchvision
    from torchvision.ops.misc import FrozenBatchNorm2d
    native = core.retinanet_resnet50_fpn
    def factory(*args, **kwargs):
        kwargs.pop("weights", None)
        kwargs.pop("weights_backbone", None)
        kwargs["trainable_backbone_layers"] = 5
        model = torchvision.models.detection.retinanet_resnet50_fpn(
            weights=None, weights_backbone=None, **kwargs
        )
        # A pretrained COCO_V1 factory uses FrozenBatchNorm2d. Preserve its state
        # schema when building offline for checkpoint loading or synthetic tests.
        def freeze_bn(parent):
            import torch
            for name, child in parent.named_children():
                if isinstance(child, torch.nn.BatchNorm2d):
                    frozen = FrozenBatchNorm2d(child.num_features, eps=0.0)
                    with torch.no_grad():
                        frozen.weight.copy_(child.weight)
                        frozen.bias.copy_(child.bias)
                        frozen.running_mean.copy_(child.running_mean)
                        frozen.running_var.copy_(child.running_var)
                    setattr(parent, name, frozen)
                else:
                    freeze_bn(child)
        freeze_bn(model.backbone.body)
        return model
    core.retinanet_resnet50_fpn = factory
    try:
        yield
    finally:
        core.retinanet_resnet50_fpn = native


def build_model(method, num_classes, level="C3", ugs_config=None, *, pretrained=True,
                smoke_size=None, original_config=None):
    from ugs import UGSConfig, attach_ugs
    core = load_core()
    if original_config is not None:
        core.CURRENT_CONFIG = dict(original_config)
    else:
        key = "lgf_gated_spatial" if method == "lgf_original" else "baseline"
        core.CURRENT_CONFIG = dict(core.CONFIG_MAP[key])
    old_probe = core.args.img_size
    try:
        if smoke_size:
            core.args.img_size = smoke_size
        if level in {"C3_C4", "C3_C4_C5"}:
            if method != "lgf_original" or original_config is not None:
                raise ValueError("Multi-level placement supports lgf_original training/checkpoints only")
            from multilevel import build_multilevel_lgf
            builder = lambda: build_multilevel_lgf(core, num_classes, level)
        else:
            builder = lambda: core.build_model(num_classes, level)
        with contextlib.nullcontext() if pretrained else factory_without_download(core):
            model = builder()
        core.convert_bn_to_gn(model, num_groups=32, convert_frozen=False)
    finally:
        core.args.img_size = old_probe
    if method == "ugs_reimpl":
        if core.CURRENT_CONFIG.get("block_type") != "none":
            raise ValueError("UGS comparison uses the baseline backbone, without LGF")
        attach_ugs(model, ugs_config or UGSConfig())
    if smoke_size:
        model.transform.min_size = (smoke_size,)
        model.transform.max_size = smoke_size
    return model
