"""Add genuine simultaneous pre-FPN C3+C4 or C3+C4+C5 LGF blocks.

Keep vendor/original_core.py byte-identical. The original model's constructor
and pretrained weights/head/anchor configuration remain in use; only its
CustomBackboneBlockBeforeFPN constructor is substituted for one model build.
"""
from contextlib import contextmanager

LEVEL_SETS = {
    "C3_C4": ("C3", "C4"),
    "C3_C4_C5": ("C3", "C4", "C5"),
}
CHANNELS = {"C3": 512, "C4": 1024, "C5": 2048}


@contextmanager
def scoped_backbone_constructor(core, levels):
    """Use body.return_layers, not the original index-based fallback.

    Torchvision RetinaNet/ResNet50 body returns feature keys 0, 1, 2 for
    ResNet stages layer2, layer3, layer4, respectively. Resolve from the
    live architecture, fail if this assumption stops holding.
    """
    original = core.CustomBackboneBlockBeforeFPN

    class MultiLevelBackboneBeforeFPN(original):
        def __init__(self, backbone_with_fpn, selected_levels):
            body = backbone_with_fpn.body
            return_layers = getattr(body, "return_layers", None)
            if not isinstance(return_layers, dict):
                raise RuntimeError("Backbone does not expose body.return_layers")
            if list(return_layers) != ["layer2", "layer3", "layer4"]:
                raise RuntimeError(f"Unexpected returned ResNet stages: {return_layers}")

            semantic = [core.LEVEL_MAP[level] for level in levels]
            keys = [return_layers[stage] for stage in semantic]
            if len(set(keys)) != len(levels):
                raise RuntimeError(f"Duplicate resolved FPN input keys: {keys}")

            # The original class accepts *actual feature keys* directly.
            # Passing [0,1] or [0,1,2] bypasses its inaccurate
            # semantic_to_actual enumeration for symbolic layer names.
            super().__init__(backbone_with_fpn, selected_levels=keys)
            if list(self.selected_actual) != keys or list(self.block_fpn_in) != keys:
                raise RuntimeError("LGF levels were not inserted as requested")
            for level, key in zip(levels, keys):
                block = self.block_fpn_in[str(key)]
                if not isinstance(block, core.StableLGFBlock):
                    raise RuntimeError(f"Feature {key} does not contain original LGF block")
                actual_channels = block.local[0].in_channels
                if actual_channels != CHANNELS[level]:
                    raise RuntimeError(
                        f"{level} must have {CHANNELS[level]} channels; "
                        f"resolved {key} has {actual_channels}"
                    )
            self.lgf_semantic_levels = tuple(levels)

    core.CustomBackboneBlockBeforeFPN = MultiLevelBackboneBeforeFPN
    try:
        yield
    finally:
        core.CustomBackboneBlockBeforeFPN = original


def build_multilevel_lgf(core, num_classes, level):
    """Build exactly the original spatial-gated LGF modules, once per level."""
    if level not in LEVEL_SETS:
        raise ValueError(f"Unsupported multi-level selection: {level}")
    if (core.CURRENT_CONFIG.get("block_type"),
        core.CURRENT_CONFIG.get("gating_type"),
        core.CURRENT_CONFIG.get("BRANCH_PRESET")) != ("lgf", "gated_spatial", "local_global"):
        raise ValueError("Multi-level model requires the original spatial-gated LGF configuration")
    levels = LEVEL_SETS[level]
    with scoped_backbone_constructor(core, levels):
        # C3 here is a placeholder to trigger the original backbone wrapper.
        # The patched constructor replaces it with all requested levels.
        model = core.build_model(num_classes, "C3")
    if tuple(model.backbone.lgf_semantic_levels) != levels:
        raise RuntimeError("Multi-level model verification failed")
    return model
