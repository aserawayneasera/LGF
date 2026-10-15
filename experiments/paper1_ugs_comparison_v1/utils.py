import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import tempfile


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def plain(value):
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        return plain(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def signature(value):
    return hashlib.sha256(json.dumps(plain(value), sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(plain(value), indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def save_checkpoint(path, value):
    import torch
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def environment():
    import torch, torchvision
    packages = {}
    for p in ["torch", "torchvision", "albumentations", "numpy", "pycocotools", "opencv-python-headless"]:
        try:
            packages[p] = importlib.metadata.version(p)
        except importlib.metadata.PackageNotFoundError:
            packages[p] = None
    return {"python": platform.python_version(), "packages": packages,
            "cuda_available": torch.cuda.is_available(), "torch_cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}


def require_supported_torch():
    import torch, torchvision
    if not torch.__version__.startswith("2.10.") or not torchvision.__version__.startswith("0.25."):
        raise RuntimeError("This package targets torch 2.10.x and torchvision 0.25.x. "
                           f"Found {torch.__version__}, {torchvision.__version__}.")


def source_hashes():
    root = Path(__file__).resolve().parent
    return {str(p.relative_to(root)): sha256(p) for p in sorted(root.rglob("*.py"))
            if "__pycache__" not in p.parts}
