# train_lgf.py
# One-file script: training, 3-seed grid launcher, and aggregation.
# Implements: 0, 1A, 1B, 1D, 2, 4
# Requires: torch, torchvision, pycocotools, albumentations, numpy, tqdm

import os
import csv
import gc
import json
import copy
import sys
import math
import time
import random
import argparse
import tempfile
import warnings
import subprocess
import datetime
from glob import glob
from collections import OrderedDict

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from albumentations.pytorch import ToTensorV2
import albumentations as A
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from torchvision.models.detection import (
    retinanet_resnet50_fpn,
    RetinaNet_ResNet50_FPN_Weights,
)
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.ops.misc import FrozenBatchNorm2d
from tqdm import tqdm
import matplotlib.pyplot as plt

# --------------------------- Env & determinism --------------------------------
os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")
os.environ.setdefault("TORCH_SHOW_CPP_STACKTRACES", "1")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
warnings.filterwarnings("ignore", category=UserWarning, module="albumentations")
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")

try:
    from torch.amp import autocast, GradScaler
    autocast_kwargs = dict(device_type="cuda", dtype=torch.float16) if device.type == "cuda" else dict(dtype=torch.bfloat16)
    scaler = GradScaler(device="cuda") if device.type == "cuda" else None
except Exception:
    from torch.cuda.amp import autocast, GradScaler
    autocast_kwargs = dict(dtype=torch.float16) if device.type == "cuda" else dict(dtype=torch.bfloat16)
    scaler = GradScaler() if device.type == "cuda" else None

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends, "cudnn") and hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = False
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass
    # Albumentations
    try:
        if hasattr(A, "set_seed"):
            A.set_seed(seed)
        else:
            from albumentations import random_utils
            random_utils.set_seed(seed)
    except Exception:
        pass

# ----------------------------- Args -------------------------------------------
def parse_args():
    p = argparse.ArgumentParser("lgf trainer with grid + aggregation")

    # What to train
    p.add_argument("--mode", type=str, default="baseline",
                   choices=["baseline","se","cbam","lgf_sum","lgf_softmax","lgf_gated","lgf_gated_spatial"])
    p.add_argument("--insert-level", type=str, default="C3", choices=["C3","C4","C5"])  # 1B
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--exp-name", type=str, default=None)

    # Dataset flags (1A)
    p.add_argument("--dataset", type=str, default="coco_nw",
                   choices=["coco_nw","coco_weather","acdc","custom"])
    p.add_argument("--train-img", type=str, default=None)
    p.add_argument("--train-ann", type=str, default=None)
    p.add_argument("--val-img", type=str, default=None)
    p.add_argument("--val-ann", type=str, default=None)

    # Performance knobs (1D)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--prefetch-factor", type=int, default=4)

    # Training schedule
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--accum-steps", type=int, default=4)
    p.add_argument("--base-lr", type=float, default=0.005)
    p.add_argument("--warmup-epochs", type=int, default=2)
    p.add_argument("--lr-milestones", type=int, nargs="*", default=[40,60])
    p.add_argument("--img-size", type=int, default=640)

    # Grid runner (2)
    p.add_argument("--run-grid", action="store_true")
    p.add_argument("--grid-seeds", type=int, nargs="*", default=[42,1337,2025])
    p.add_argument("--grid-datasets", type=str, nargs="*", default=None)
    p.add_argument("--grid-levels", type=str, nargs="*", default=["C3"])
    p.add_argument("--grid-modes", type=str, nargs="*", default=None)
    p.add_argument("--subprocess", action="store_true", help="force subprocess grid even in notebook")

    # Aggregation (4)
    p.add_argument("--aggregate", action="store_true")
    p.add_argument("--agg-datasets", type=str, nargs="*", default=None)
    p.add_argument("--agg-levels", type=str, nargs="*", default=["C3"])
    p.add_argument("--agg-modes", type=str, nargs="*", default=None)
    p.add_argument("--agg-seeds", type=int, nargs="*", default=[42,1337,2025])

    # Jupyter friendliness
    args, _ = p.parse_known_args()
    return args

args = parse_args()

# ------------------------- Experiment configs ---------------------------------
BASELINE_CONFIG = dict(BRANCH_PRESET="none", gating_type="none", block_type="none",
                       description="BASELINE: RetinaNet without custom blocks")

lgf_SUM_CONFIG      = dict(BRANCH_PRESET="local_global", gating_type="sum",          block_type="lgf", description="lgf: local+global, sum")
lgf_SOFTMAX_CONFIG  = dict(BRANCH_PRESET="local_global", gating_type="softmax",      block_type="lgf", description="lgf: local+global, softmax")
lgf_GATED_CONFIG    = dict(BRANCH_PRESET="local_global", gating_type="gated",        block_type="lgf", description="lgf: local+global, sigmoid gate")
lgf_GATED_SP_CONFIG = dict(BRANCH_PRESET="local_global", gating_type="gated_spatial",block_type="lgf", description="lgf: local+global, spatial gate")

CONFIG_MAP = {
    "baseline": BASELINE_CONFIG,
    "lgf_sum": lgf_SUM_CONFIG,
    "lgf_softmax": lgf_SOFTMAX_CONFIG,
    "lgf_gated": lgf_GATED_CONFIG,
    "lgf_gated_spatial": lgf_GATED_SP_CONFIG,
    "se": dict(BRANCH_PRESET="none", gating_type="none", block_type="se",   description="SE before FPN"),
    "cbam": dict(BRANCH_PRESET="none", gating_type="none", block_type="cbam", description="CBAM before FPN"),
}

CURRENT_CONFIG = CONFIG_MAP[args.mode]
LEVEL_MAP = {"C3":"layer2", "C4":"layer3", "C5":"layer4"}  # 1B

# ------------------------- Anchors & helpers ----------------------------------
def _probe_feature_names(backbone: nn.Module) -> list:
    with torch.no_grad():
        x = torch.zeros(1,3,args.img_size,args.img_size, device=next(backbone.parameters()).device)
        feats = backbone(x)
        if not isinstance(feats, OrderedDict):
            raise RuntimeError("Backbone must return OrderedDict of features")
        return list(feats.keys())

_SIZE_MAP = {
    "0": (32,48,64),
    "1": (64,96,128),
    "2": (128,192,256),
    "3": (256,384,512),
    "p6": (256,384,512),
    "pool": (384,512,640),
    "p7": (384,512,640),
}

FORCE_STOCK_ANCHORS = False
_STOCK_AG = None
def get_stock_anchor_generator():
    global _STOCK_AG
    if _STOCK_AG is None:
        _STOCK_AG = retinanet_resnet50_fpn(weights=RetinaNet_ResNet50_FPN_Weights.COCO_V1).anchor_generator
    return _STOCK_AG

def make_anchor_generator_for(backbone: nn.Module) -> AnchorGenerator:
    names = _probe_feature_names(backbone)
    try:
        sizes = tuple(_SIZE_MAP[n] for n in names)
    except KeyError as e:
        raise RuntimeError(f"No anchor size tuple for feature '{e.args[0]}'. Backbone keys={names}")
    ratios = ((0.5,1.0,2.0),) * len(sizes)
    return AnchorGenerator(sizes=sizes, aspect_ratios=ratios)

# ---------------------------- Blocks ------------------------------------------
class StableLGFBlock(nn.Module):
    def __init__(self, channels, branches=("local","global"), gating_type="sum", norm_groups=32, squeeze_ratio=16):
        super().__init__()
        self.branches = tuple(branches)
        self.gating_type = gating_type
        self.num_branches = len(self.branches)
        self.save_maps = False
        self.viz_cache = {}
        def GN(c): return nn.GroupNorm(norm_groups, c)

        # Local branch
        self.local = None
        if "local" in self.branches:
            self.local = nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
                nn.Conv2d(channels, channels, 1, bias=False),
                GN(channels), nn.ReLU(inplace=True)
            )

        # Global branch
        self.global_branch = None
        if "global" in self.branches:
            self.global_branch = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(channels, channels, 1, bias=False),
                GN(channels), nn.SiLU(inplace=True)
            )

        # Gates
        if self.num_branches > 1:
            if self.gating_type == "softmax":
                self.temperature = nn.Parameter(torch.tensor(1.0))
                self.branch_weights = nn.Parameter(torch.ones(self.num_branches))
            elif self.gating_type == "gated":
                hid = max(channels // squeeze_ratio, 4)
                self.temperature = nn.Parameter(torch.tensor(1.0))
                self.gate_mlp = nn.Sequential(
                    nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                    nn.Linear(channels, hid), nn.ReLU(inplace=True),
                    nn.Linear(hid, self.num_branches)
                )
                nn.init.zeros_(self.gate_mlp[-1].weight); nn.init.zeros_(self.gate_mlp[-1].bias)
            elif self.gating_type == "gated_spatial":
                r = 4
                self.temperature = nn.Parameter(torch.tensor(1.0))
                self.gate_reduce = nn.Conv2d(channels, channels//r, 1, bias=False)
                self.gate_expand = nn.Conv2d(channels//r, 2*channels, 1, bias=True)
                self.gate_norm   = nn.GroupNorm(num_groups=norm_groups, num_channels=2*channels)
                nn.init.zeros_(self.gate_expand.weight); nn.init.zeros_(self.gate_expand.bias)

        self.gamma = nn.Parameter(torch.tensor(0.1))

    def enable_visualization(self, enable=True):
        self.save_maps = bool(enable)
        if not enable:
            self.viz_cache = {}

    def _broadcast(self, s, like):
        if s.dim() == 1:
            s = s.view(-1,1,1,1)
        return s.expand_as(like)

    def forward(self, x):
        feats = []
        L = self.local(x) if self.local is not None else None
        if L is not None: feats.append(L)
        G = None
        if self.global_branch is not None:
            g = self.global_branch(x)
            G = g.expand_as(x)
            feats.append(G)

        if len(feats) == 1:
            out = feats[0]
        else:
            if self.gating_type == "sum":
                out = feats[0] + feats[1]
                wL = torch.full_like(L, 0.5); wG = torch.full_like(G, 0.5)
            elif self.gating_type == "softmax":
                tau = F.softplus(self.temperature) + 1e-3
                w = F.softmax(self.branch_weights / tau, dim=0)
                out = w[0]*feats[0] + w[1]*feats[1]
                wL = self._broadcast(w[0], L); wG = self._broadcast(w[1], G)
            elif self.gating_type == "gated":
                with torch.cuda.amp.autocast(enabled=False):
                    logits = self.gate_mlp(x.float())
                    tau = F.softplus(self.temperature.float()) + 1e-3
                    logits = logits.clamp_(-15,15)
                    w32 = torch.sigmoid(logits / tau)
                w = w32.to(dtype=x.dtype)
                wL = self._broadcast(w[:,0], L); wG = self._broadcast(w[:,1], G)
                out = wL*L + wG*G
            elif self.gating_type == "gated_spatial":
                h = F.relu(self.gate_reduce(x), inplace=True)
                logits = self.gate_expand(h)
                logits = self.gate_norm(logits)
                with torch.cuda.amp.autocast(enabled=False):
                    logits32 = logits.float().clamp_(-15,15)
                    tau = F.softplus(self.temperature.float()) + 1e-3
                    w = torch.sigmoid(logits32 / tau)
                w = w.to(dtype=x.dtype)
                N, twoC, H, W = w.shape
                C = twoC//2
                w = w.view(N,2,C,H,W)
                wL, wG = w[:,0], w[:,1]
                out = wL*L + wG*G
            else:
                out = feats[0] + feats[1]

        out = x + self.gamma*out
        if self.save_maps:
            self.viz_cache = {"L": (L.detach() if L is not None else None),
                              "G": (G.detach() if G is not None else None)}
        return out

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.fc  = nn.Sequential(
            nn.Linear(channels, channels//reduction, bias=False), nn.ReLU(inplace=True),
            nn.Linear(channels//reduction, channels, bias=False), nn.Sigmoid()
        )
    def forward(self, x):
        b,c,_,_ = x.shape
        y = self.fc(self.avg(x).view(b,c)).view(b,c,1,1)
        return x * y

# class CBAMBlock(nn.Module):
#     def __init__(self, channels, reduction=16, k=7):
#         super().__init__()
#         self.avg = nn.AdaptiveAvgPool2d(1)
#         self.maxp = nn.AdaptiveMaxPool2d(1)
#         self.fc = nn.Sequential(
#             nn.Conv2d(channels, channels//reduction, 1, bias=False), nn.ReLU(inplace=True),
#             nn.Conv2d(channels//reduction, channels, 1, bias=False)
#         )
#         self.sigc = nn.Sigmoid()
#         self.convs = nn.Conv2d(2,1,k,padding=k//2,bias=False)
#         self.sigs = nn.Sigmoid()
#     def forward(self, x):
#         ca = self.fc(self.avg(x)) + self.fc(self.maxp(x))
#         x = x * self.sigc(ca)
#         s = torch.cat([x.mean(1,True), x.amax(1,True)], dim=1)
#         return x * self.sigs(self.convs(s))

class CBAMBlock(nn.Module):
    """
    Determinism-safe CBAM.
    - Channel attention: avg pool + softmax-pooling over HxW (approximates max) -> shared MLP
    - Spatial attention: concat(mean over C, softmax-pooling over C) -> 7x7 conv
    The softmax pooling avoids adaptive_max_pool2d backward, which is non-deterministic on CUDA.
    """
    def __init__(self, channels, reduction=16, k=5, beta=20.0):
        super().__init__()
        self.beta = beta
        self.avg = nn.AdaptiveAvgPool2d(1)  # deterministic
        # shared MLP for channel attention
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.sigc = nn.Sigmoid()
        # spatial attention
        self.convs = nn.Conv2d(2, 1, k, padding=k // 2, bias=False)
        self.sigs = nn.Sigmoid()

    @torch.no_grad()
    def _normalize_stable(self, x, dim):
        # subtract max along 'dim' for numerical stability, keep graph by doing it outside no_grad in callers
        return x - x.max(dim=dim, keepdim=True).values

    def _softmax_pool_spatial(self, x):
        # softmax over H*W per channel, returns [B,C,1,1]
        B, C, H, W = x.shape
        x_flat = x.view(B, C, H * W)
        # stabilize
        x_norm = x_flat - x_flat.max(dim=2, keepdim=True).values
        w = F.softmax(self.beta * x_norm, dim=2)
        pooled = (w * x_flat).sum(dim=2)  # [B,C]
        return pooled.view(B, C, 1, 1)

    def _softmax_pool_channel(self, x):
        # softmax over C per spatial location, returns [B,1,H,W]
        B, C, H, W = x.shape
        x_hw_c = x.permute(0, 2, 3, 1)              # [B,H,W,C]
        x_norm = x_hw_c - x_hw_c.max(dim=3, keepdim=True).values
        w = F.softmax(self.beta * x_norm, dim=3)    # [B,H,W,C]
        pooled = (w * x_hw_c).sum(dim=3, keepdim=True)  # [B,H,W,1]
        return pooled.permute(0, 3, 1, 2)           # [B,1,H,W]

    def forward(self, x):
        # Channel attention
        avg_pool = self.avg(x)
        smx_pool = self._softmax_pool_spatial(x)
        ca = self.fc(avg_pool) + self.fc(smx_pool)
        x = x * self.sigc(ca)

        # Spatial attention
        s_mean = x.mean(dim=1, keepdim=True)
        s_softmax_max = self._softmax_pool_channel(x)
        s = torch.cat([s_mean, s_softmax_max], dim=1)
        sa = self.sigs(self.convs(s))
        return x * sa

# ----------------------------- Dataset (1A, 1D) -------------------------------
def select_dataset_by_name(name: str):
    # Defaults from your code for COCO Non-Weather; others can be overridden via flags.
    if name == "coco_nw":
        return (
            "/nas.dbms/asera/PROJECTS/DATASET/COCO/non_weather-mini/images/train2017_non_weather-2400k6c",
            "/nas.dbms/asera/PROJECTS/DATASET/COCO/non_weather-mini/annotations/mini_train2017_non_weather-2400k6c.json",
            "/nas.dbms/asera/PROJECTS/DATASET/COCO/non_weather-mini/images/val2017_non_weather-500k6c",
            "/nas.dbms/asera/PROJECTS/DATASET/COCO/non_weather-mini/annotations/mini_val2017_non_weather-500k6c.json"

            # "/root/COCO/images/train2017_non_weather-2400k6c",
            # "/root/COCO/annotations/mini_train2017_non_weather-2400k6c.json",
            # "/root/COCO/images/val2017_non_weather-500k6c",
            # "/root/COCO/annotations/mini_val2017_non_weather-500k6c.json"
        )
    elif name == "acdc":
        return (
            "/nas.dbms/asera/PROJECTS/DATASET/ACDC-1/ACDC-1-NEW/images/train",
            "/nas.dbms/asera/PROJECTS/DATASET/ACDC-1/ACDC-1-NEW/annotations/mini_train.json",
            "/nas.dbms/asera/PROJECTS/DATASET/ACDC-1/ACDC-1-NEW/images/val",
            "/nas.dbms/asera/PROJECTS/DATASET/ACDC-1/ACDC-1-NEW/annotations/mini_val.json"
        )
    elif name == "coco_weather":
        # Fill these to your synthetic-weather paths or pass --train-img/--train-ann/--val-img/--val-ann
        return (
            "/nas.dbms/asera/PROJECTS/DATASET/COCO/weather-mini/images/train2017_weather-2400k6c",
            "/nas.dbms/asera/PROJECTS/DATASET/COCO/weather-mini/annotations/mini_train2017_weather-2400k6c.json", 
            "/nas.dbms/asera/PROJECTS/DATASET/COCO/weather-mini/images/val2017_weather-500k6c",
            "/nas.dbms/asera/PROJECTS/DATASET/COCO/weather-mini/annotations/mini_val2017_weather-500k6c.json"
        
            # "/root/COCO/images/train2017_weather-2400k6c",
            # "/root/COCO/annotations/mini_train2017_weather-2400k6c.json",
            # "/root/COCO/images/val2017_weather-500k6c",
            # "/root/COCO/annotations/mini_val2017_weather-500k6c.json"

        )
        # raise RuntimeError("Set --train-img/--train-ann/--val-img/--val-ann for coco_weather.")
    else:  # custom
        raise RuntimeError("Use --train-img/--train-ann/--val-img/--val-ann for dataset=custom.")

def patch_annotations_once(ann_file):
    with open(ann_file,"r") as f: data = json.load(f)
    if "info" not in data:
        data["info"] = {"description":"Patched COCO dataset","version":"1.0"}
        with tempfile.NamedTemporaryFile("w+",suffix=".json",delete=False) as tmp:
            json.dump(data,tmp); tmp.flush()
            return tmp.name
    return ann_file

class COCODataset(torch.utils.data.Dataset):
    def __init__(self, img_folder, ann_file, transforms=None, train=False):
        self.img_folder = img_folder
        self.coco = COCO(ann_file)
        self.transforms = transforms
        self.train = train
        valid_cat_ids = sorted(self.coco.getCatIds())
        self.cat_id_to_label = {cid:i for i,cid in enumerate(valid_cat_ids)}
        self.label_to_cat_id = {v:k for k,v in self.cat_id_to_label.items()}
        # expose number of classes for model construction
        self.num_classes = len(valid_cat_ids)
        self.ids = sorted(self.coco.imgs.keys())
        # Optional ablation subset
        k_env = os.getenv("MAX_TRAIN_IMAGES")
        if self.train and k_env:
            k = min(int(k_env), len(self.ids))
            rng = np.random.default_rng(12345)
            self.ids = rng.choice(self.ids, size=k, replace=False).tolist()

    def __len__(self): return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        info = self.coco.loadImgs(img_id)[0]
        path = os.path.join(self.img_folder, info["file_name"])
        img = cv2.imread(path)
        if img is None:
            img = np.zeros((info["height"], info["width"], 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        boxes, labels = [], []
        for a in anns:
            x,y,w,h = a["bbox"]
            if w>1 and h>1:
                boxes.append([x,y,x+w,y+h])
                labels.append(self.cat_id_to_label[a["category_id"]])

        if self.transforms:
            t = self.transforms(image=img, bboxes=boxes, labels=labels)
            img, boxes, labels = t["image"], t["bboxes"], t["labels"]
        if img.dtype == torch.uint8:
            img = img.float()/255.0
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1,4)
        labels = torch.tensor([int(l) for l in labels], dtype=torch.int64)
        target = {"boxes": boxes, "labels": labels, "image_id": torch.tensor([img_id]),
                  "orig_size": torch.tensor([info["width"], info["height"]])}
        return img, target

def get_transform(train=True):
    if train:
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            ToTensorV2(),
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels'], min_area=1, min_visibility=0.1, check_each_transform=True))
    else:
        return A.Compose([ToTensorV2()],
            bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels'], min_area=1, min_visibility=0.1, check_each_transform=True))

def collate_fn(batch): return tuple(zip(*batch))

def build_datasets_and_loaders(seed, dataset_code, overrides):
    set_seed(seed)
    if overrides["train_img"] and overrides["train_ann"] and overrides["val_img"] and overrides["val_ann"]:
        tr_img, tr_ann, va_img, va_ann = overrides["train_img"], overrides["train_ann"], overrides["val_img"], overrides["val_ann"]
    else:
        tr_img, tr_ann, va_img, va_ann = select_dataset_by_name(dataset_code)
    va_ann = patch_annotations_once(va_ann)
    train_dataset = COCODataset(tr_img, tr_ann, transforms=get_transform(train=True),  train=True)
    val_dataset   = COCODataset(va_img, va_ann, transforms=get_transform(train=False), train=False)

    g = torch.Generator(); g.manual_seed(seed)
    common = dict(collate_fn=collate_fn, worker_init_fn=worker_init_fn, pin_memory=True, generator=g)
    if args.num_workers > 0:
        common.update(num_workers=args.num_workers, persistent_workers=True, prefetch_factor=args.prefetch_factor)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True,  **common)
    val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size, shuffle=False, drop_last=False, **common)
    return train_dataset, val_dataset, train_loader, val_loader, va_ann

def worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(int(worker_seed)); random.seed(int(worker_seed))
    try:
        if hasattr(A, "set_seed"):
            A.set_seed(int(worker_seed))
        else:
            from albumentations import random_utils
            random_utils.set_seed(int(worker_seed))
    except Exception:
        pass

# ------------------------------ Model -----------------------------------------
def convert_bn_to_gn(module, num_groups=32, convert_frozen=False):
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            setattr(module, name, nn.GroupNorm(num_groups=num_groups, num_channels=child.num_features))
        elif convert_frozen and isinstance(child, FrozenBatchNorm2d):
            setattr(module, name, nn.GroupNorm(num_groups=num_groups, num_channels=child.num_features))
        else:
            convert_bn_to_gn(child, num_groups=num_groups, convert_frozen=convert_frozen)
    return module

def get_block(channels, block_type, branches, gating_type):
    if block_type == "lgf":
        return StableLGFBlock(channels, branches=branches, gating_type=gating_type)
    if block_type == "se":
        return SEBlock(channels)
    if block_type == "cbam":
        return CBAMBlock(channels)
    return nn.Identity()

class CustomBackboneBlockBeforeFPN(nn.Module):
    def __init__(self, backbone_with_fpn, selected_levels):
        super().__init__()
        self.body = backbone_with_fpn.body
        self.fpn  = backbone_with_fpn.fpn

        with torch.no_grad():
            feats = self.body(torch.zeros(1,3,224,224))
            fpn_feats = self.fpn(feats)
        actual_keys = list(feats.keys())
        semantic_to_actual = {
            "layer1": actual_keys[0] if len(actual_keys)>0 else None,
            "layer2": actual_keys[1] if len(actual_keys)>1 else None,
            "layer3": actual_keys[2] if len(actual_keys)>2 else None,
            "layer4": actual_keys[3] if len(actual_keys)>3 else None,
        }
        wanted = []
        for lvl in selected_levels:
            if lvl in actual_keys: wanted.append(lvl)
            elif lvl in semantic_to_actual and semantic_to_actual[lvl] is not None: wanted.append(semantic_to_actual[lvl])
        self.selected_actual = wanted

        self.block_fpn_in = nn.ModuleDict()
        for k in self.selected_actual:
            C = feats[k].shape[1]
            self.block_fpn_in[str(k)] = get_block(C, CURRENT_CONFIG.get("block_type","none"),
                                                  ("local","global") if CURRENT_CONFIG["BRANCH_PRESET"]=="local_global" else (),
                                                  CURRENT_CONFIG["gating_type"])

        first_out = next(iter(fpn_feats.keys()))
        self.out_channels = fpn_feats[first_out].shape[1]

    def forward(self, x):
        feats = self.body(x)
        for k in self.selected_actual:
            feats[k] = self.block_fpn_in[str(k)](feats[k])
        return self.fpn(feats)

def build_model(num_classes, insert_level):
    pretrained = retinanet_resnet50_fpn(weights=RetinaNet_ResNet50_FPN_Weights.COCO_V1)
    convert_bn_to_gn(pretrained.backbone.body, num_groups=32, convert_frozen=False)

    sel_levels = [LEVEL_MAP[insert_level]] if CURRENT_CONFIG.get("block_type","none") != "none" else []
    if sel_levels:
        bb = CustomBackboneBlockBeforeFPN(pretrained.backbone, selected_levels=sel_levels)
    else:
        bb = pretrained.backbone

    ag = get_stock_anchor_generator() if FORCE_STOCK_ANCHORS else make_anchor_generator_for(bb)
    model = torchvision.models.detection.RetinaNet(
        backbone=bb,
        num_classes=num_classes,
        anchor_generator=ag,
        head=pretrained.head,
        transform=pretrained.transform,
        detections_per_img=100,
        nms_thresh=0.5,
        score_thresh=0.05,
    )

    # Update cls head for our class count
    in_ch = model.head.classification_head.cls_logits.in_channels
    n_anchors = model.head.classification_head.num_anchors
    model.head.classification_head.cls_logits = nn.Conv2d(in_ch, n_anchors*num_classes, kernel_size=3, padding=1)
    model.head.classification_head.num_classes = num_classes
    torch.nn.init.normal_(model.head.classification_head.cls_logits.weight, std=0.01)
    prior_prob = 0.01
    bias_value = -torch.log(torch.tensor((1.0 - prior_prob) / prior_prob))
    torch.nn.init.constant_(model.head.classification_head.cls_logits.bias, bias_value)
    return model

# ----------------------------- Eval & logging ---------------------------------
def coco_evaluation(model, data_loader, ann_file, device):
    coco_gt = COCO(ann_file)
    detections = []
    model.eval()
    with torch.no_grad():
        for images, targets in tqdm(data_loader, desc="Evaluating"):
            images = [im.to(device) for im in images]
            outputs = model(images)
            for i,out in enumerate(outputs):
                img_id = int(targets[i]["image_id"].item())
                boxes = out["boxes"].detach().cpu().numpy()
                scores = out["scores"].detach().cpu().numpy()
                labels = out["labels"].detach().cpu().numpy()
                for b,s,l in zip(boxes,scores,labels):
                    x1,y1,x2,y2 = b
                    cat = data_loader.dataset.label_to_cat_id[int(l)] if hasattr(data_loader.dataset,"label_to_cat_id") else int(l)
                    detections.append({"image_id":img_id,"category_id":cat,
                                       "bbox":[float(x1),float(y1),float(x2-x1),float(y2-y1)],
                                       "score":float(s)})
    if not detections:
        return {k:0.0 for k in ["mAP","AP50","AP75","AP_small","AP_medium","AP_large",
                                "AR1","AR10","AR100","AR_small","AR_medium","AR_large"]} | \
               {"AP_per_class":{}, "detailed_metrics":{}, "detection_count":0}

    coco_dt = coco_gt.loadRes(detections)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.params.iouThrs = np.array([0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95])
    coco_eval.params.maxDets = [1,10,100]
    coco_eval.evaluate(); coco_eval.accumulate(); coco_eval.summarize()
    stats = coco_eval.stats

    precisions = coco_eval.eval["precision"]
    K = precisions.shape[2]
    ap_per_class = []
    for k in range(K):
        p = precisions[:,:,k,0,2]
        p = p[p>-1]
        ap_per_class.append(np.mean(p) if p.size else float("nan"))
    names = [c["name"] for c in coco_gt.loadCats(coco_gt.getCatIds())]
    per_class = dict(zip(names, ap_per_class))

    recall = coco_eval.eval["recall"]
    ar_dets = []
    for mdi, md in enumerate([1,10,100]):
        r = recall[:,:,:,mdi]; r = r[r>-1]
        ar_dets.append(np.mean(r) if r.size else 0.0)
    # stats layout: [AP, AP50, AP75, APS, APM, APL, AR1, AR10, AR100, ARS, ARM, ARL]
    return {
        "mAP": stats[0], "AP50": stats[1], "AP75": stats[2],
        "AP_small": stats[3], "AP_medium": stats[4], "AP_large": stats[5],
        "AR1": stats[6], "AR10": stats[7], "AR100": stats[8],
        "AR_small": stats[9], "AR_medium": stats[10], "AR_large": stats[11],
        "AP_per_class": per_class,
        "detailed_metrics": {},
        "detection_count": len(detections),
    }

class ValidationLogger:
    def __init__(self, experiment_name):
        self.experiment = experiment_name
        self.log_dir = os.path.join("/nas.dbms/asera/validation_logs", experiment_name)
        os.makedirs(self.log_dir, exist_ok=True)
        self.csv_path = os.path.join(self.log_dir, "validation_results.csv")
        self._init_csv()
    def _init_csv(self):
        fields = ["epoch","timestamp","experiment","mAP","AP50","AP75","AP_small","AP_medium","AP_large",
                  "AR1","AR10","AR100","AR_small","AR_medium","AR_large","detection_count"]
        with open(self.csv_path,"w",newline="") as f:
            csv.DictWriter(f, fieldnames=fields).writeheader()
    def log(self, epoch, metrics, writer):
        fields = ["epoch","timestamp","experiment","mAP","AP50","AP75","AP_small","AP_medium","AP_large",
                  "AR1","AR10","AR100","AR_small","AR_medium","AR_large","detection_count"]
        row = {
            "epoch": epoch, "timestamp": datetime.datetime.now().isoformat(), "experiment": self.experiment,
            **{k: metrics[k] for k in fields if k in metrics}
        }
        with open(self.csv_path,"a",newline="") as f:
            csv.DictWriter(f, fieldnames=fields).writerow(row)
        # TB scalars
        for k,v in metrics.items():
            if isinstance(v,(int,float)):
                writer.add_scalar(f"{self.experiment}/Val/{k}", v, epoch)
        # JSON snapshot per epoch
        with open(os.path.join(self.log_dir, f"epoch_{epoch:03d}_results.json"), "w") as f:
            json.dump({"epoch":epoch,"timestamp":row["timestamp"],"experiment":self.experiment,"metrics":metrics}, f, indent=2)
        return metrics

# ------------------------------ Complexity/speed -------------------------------
def benchmark_inference(model, data_loader, device, max_images=200, warmup_batches=20):
    model.eval()
    timings = []; counted = 0
    with torch.no_grad():
        if device.type == "cuda":
            starter = torch.cuda.Event(enable_timing=True)
            ender = torch.cuda.Event(enable_timing=True)
            # warmup
            for bi,(images,_) in enumerate(data_loader):
                _ = model([im.to(device) for im in images])
                if bi+1 >= warmup_batches: break
            for (images,_) in data_loader:
                images = [im.to(device) for im in images]
                starter.record(); _ = model(images); ender.record()
                torch.cuda.synchronize()
                ms_per_img = starter.elapsed_time(ender)/max(len(images),1)
                timings.append(ms_per_img); counted += len(images)
                if counted >= max_images: break
            lat = float(np.mean(timings)) if timings else float("nan")
            return {"latency_ms_per_image": lat, "images_per_second": (1000.0/lat if lat>0 else float("nan"))}
        else:
            # CPU fallback
            for (images,_) in data_loader:
                images = [im.to(device) for im in images]
                t0 = time.perf_counter(); _ = model(images); dt = time.perf_counter()-t0
                timings.append(1000.0*dt/max(len(images),1)); counted += len(images)
                if counted >= max_images: break
            lat = float(np.mean(timings)) if timings else float("nan")
            return {"latency_ms_per_image": lat, "images_per_second": (1000.0/lat if lat>0 else float("nan"))}

def try_flops_params(model, image_size=(3, 800, 1333)):
    params_m = sum(p.numel() for p in model.parameters())/1e6
    try:
        from fvcore.nn import FlopCountAnalysis
        m = copy.deepcopy(model).to("cpu").eval()
        dummy = torch.zeros(1,*image_size)
        flops = FlopCountAnalysis(m, ([dummy],)).total()
        return {"params_M": float(params_m), "FLOPs_G": float(flops)/1e9}
    except Exception:
        try:
            from thop import profile
            m = copy.deepcopy(model).to("cpu").eval()
            dummy = torch.zeros(1,*image_size)
            macs,_ = profile(m, inputs=([dummy],), verbose=False)
            return {"params_M": float(params_m), "FLOPs_G": float(macs)/1e9}
        except Exception:
            return {"params_M": float(params_m), "FLOPs_G": None}

# ------------------------------ Train (best by mAP) ---------------------------
def train_one(experiment_name, train_dataset, val_dataset, train_loader, val_loader, patched_val_ann_file, insert_level):
    C = train_dataset.num_classes
    model = build_model(C, insert_level).to(device)

    # assert isolation: only the block we asked for
    has_lgf = any(isinstance(m, StableLGFBlock) for m in model.modules())
    has_cbam  = any(isinstance(m, CBAMBlock)       for m in model.modules())
    has_se    = any(isinstance(m, SEBlock)          for m in model.modules())
    bt = CURRENT_CONFIG.get("block_type","none")
    if bt == "lgf": assert has_lgf and not has_cbam and not has_se
    if bt == "cbam":  assert has_cbam  and not has_lgf and not has_se
    if bt == "se":    assert has_se    and not has_lgf and not has_cbam
    if bt == "none":  assert not (has_lgf or has_cbam or has_se)

    convert_bn_to_gn(model, num_groups=32, convert_frozen=False)

    # Warmup freeze: heads + pre-FPN blocks
    for p in model.parameters(): p.requires_grad = False
    for p in model.head.parameters(): p.requires_grad = True
    if hasattr(model.backbone, "block_fpn_in"):
        for p in model.backbone.block_fpn_in.parameters(): p.requires_grad = True

    ema = ModelEMA(model, decay=0.9999, device=device)
    val_logger = ValidationLogger(experiment_name)
    writer = SummaryWriter(f"/nas.dbms/asera/NEW-4.1.2/runs/{experiment_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(trainable, lr=args.base_lr, momentum=0.9, weight_decay=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_milestones, gamma=0.1)

    best_map = 0.0
    best_epoch = -1
    best_ckpt = None
    unfroze = False

    for epoch in range(args.epochs):
        if not unfroze and epoch == 5:
            for p in model.parameters(): p.requires_grad = True
            optimizer = torch.optim.SGD(model.parameters(), lr=args.base_lr, momentum=0.9, weight_decay=1e-4)
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_milestones, gamma=0.1)
            unfroze = True

        # warmup
        if epoch < args.warmup_epochs:
            lr_scale = min(1., float(epoch+1)/args.warmup_epochs)
            for g in optimizer.param_groups: g["lr"] = args.base_lr * lr_scale

        model.train()
        ep_loss = 0.0
        prog = tqdm(train_loader, desc=f"{experiment_name} - Epoch {epoch+1}/{args.epochs}")
        for bi,(images,targets) in enumerate(prog):
            images = [im.to(device) for im in images]
            targets = [{k:v.to(device) for k,v in t.items()} for t in targets]
            if device.type == "cuda":
                with autocast(**autocast_kwargs):
                    loss_dict = model(images, targets)
                    loss = sum(loss_dict.values())/args.accum_steps
                scaler.scale(loss).backward()
                if (bi+1) % args.accum_steps == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                    scaler.step(optimizer); scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    ema.update(model)
                ep_loss += loss.item()*args.accum_steps
                prog.set_postfix(loss=loss.item()*args.accum_steps, lr=optimizer.param_groups[0]["lr"])
            else:
                loss_dict = model(images, targets)
                loss = sum(loss_dict.values())/args.accum_steps
                loss.backward()
                if (bi+1) % args.accum_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    ema.update(model)
                ep_loss += loss.item()*args.accum_steps
                prog.set_postfix(loss=loss.item()*args.accum_steps, lr=optimizer.param_groups[0]["lr"])

        lr_scheduler.step()

        if (epoch+1) % 5 == 0 or epoch == args.epochs-1:
            eval_results = validate_model(ema.ema, epoch, writer, experiment_name, val_logger, val_loader, patched_val_ann_file)
            spd = benchmark_inference(ema.ema, val_loader, device)
            cmp = try_flops_params(ema.ema)
            print("[SPEED]", spd, "[COMPLEXITY]", cmp)
            if eval_results["mAP"] > best_map:  # best-by-mAP (fixes AP_small bias)
            # change to AP_small to match the paper
            # if eval_results["AP_small"] > best_map_small:
                best_map = eval_results["mAP"]
                # best_map_small = eval_results["AP_small"]
                best_epoch = epoch+1
                os.makedirs("/nas.dbms/asera/NEW", exist_ok=True)
                best_ckpt = os.path.join("/nas.dbms/asera/NEW",
                    f"BEST_{experiment_name}_epoch_{best_epoch}_map_{best_map:.4f}.pth")
                # best_ckpt_small = os.path.join("/nas.dbms/asera/NEW",
                #     f"BEST_{experiment_name}_epoch_{best_epoch}_ap_small_{best_map_small:.4f}.pth")
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "ema_state_dict": ema.ema.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scaler_state_dict": (scaler.state_dict() if scaler is not None else None),
                    "best_map": best_map,
                    # "best_map_small": best_map_small,
                    "eval_results": eval_results,
                    "experiment_name": experiment_name,
                    "config": CURRENT_CONFIG,
                }, best_ckpt)
                print(f"[SAVE] Best {experiment_name} by mAP: {best_map:.4f} @ epoch {best_epoch}")
                # print(f"[SAVE] Best {experiment_name} by AP_small: {best_map_small:.4f} @ epoch {best_epoch}")

        if (epoch+1) % 10 == 0:
            torch.cuda.empty_cache(); gc.collect()

    writer.close()
    return best_map, best_epoch, best_ckpt

def validate_model(model, epoch, writer, experiment_name, val_logger, val_loader, patched_val_ann_file):
    model.eval()
    eval_results = coco_evaluation(model, val_loader, patched_val_ann_file, device)
    # >>> ADD THIS BLOCK <<<
    try:
        from lgf_controls_toolkit import append_result
        os.makedirs("results", exist_ok=True)
        append_result("results/all_runs.csv", dict(
            exp=experiment_name,
            # if you added these as args, they’ll be picked up; else pass fixed strings
            gate_override=getattr(args, "gate_override", ""),   # ok if empty
            alpha_local=getattr(args, "alpha_local", ""),       # ok if empty
            insert_level=args.insert_level,
            seed=args.seed,
            mAP=eval_results["mAP"],
            AP_small=eval_results["AP_small"],
            AP50=eval_results["AP50"],
            AP75=eval_results["AP75"],
            AR100=eval_results["AR100"],
            dets=eval_results.get("detection_count", 0),
            ips=0.0,  # you log ips elsewhere; leave 0.0 here or wire in your speed dict
        ))
    except Exception as e:
        print(f"[WARN] append_result failed: {e}")
    # <<< END ADD >>>

    logged = val_logger.log(epoch, eval_results, writer)
    model.train()
    return logged

class ModelEMA:
    def __init__(self, model, decay=0.9999, device=None):
        self.ema = copy.deepcopy(model)
        self.ema.eval()
        self.decay = decay
        if device is not None: self.ema.to(device)
        for p in self.ema.parameters(): p.requires_grad_(False)
    def update(self, model):
        with torch.no_grad():
            msd = model.state_dict()
            for k,v in self.ema.state_dict().items():
                if not v.dtype.is_floating_point: continue
                src = msd.get(k); 
                if src is None: continue
                v.copy_(v * self.decay + src.to(v.device) * (1. - self.decay))

# ------------------------------- Aggregation (4) -------------------------------
def best_by_map(run_dir):
    files = sorted(glob(os.path.join(run_dir, "epoch_*_results.json")))
    best = None
    for f in files:
        d = json.load(open(f))
        m = d["metrics"]
        row = {"epoch": d["epoch"], **{k:m[k] for k in ["mAP","AP50","AP75","AP_small","AP_medium","AP_large"]}}
        if best is None or row["mAP"] > best["mAP"]:
            best = row
    return best

def best_by_apsmall(run_dir):
    files = sorted(glob(os.path.join(run_dir, "epoch_*_results.json")))
    best = None
    for f in files:
        d = json.load(open(f))
        m = d["metrics"]
        row = {"epoch": d["epoch"], **{k:m[k] for k in ["AP_small","AP_medium","AP_large","AP50","AP75","mAP","AR1","AR10","AR100","AR_small","AR_medium","AR_large"]}}
        if best is None or row["AP_small"] > best["AP_small"]:
            best = row
    return best

def aggregate_group(group_name, run_names):
    rows = []
    for r in run_names:
        path = os.path.join("/nas.dbms/asera/validation_logs", r)
        if not os.path.isdir(path):
            print(f"[SKIP] missing {path}")
            continue
        # rows.append(best_by_map(path))
        rows.append(best_by_apsmall(path))
    if not rows:
        print(f"[EMPTY] {group_name}")
        return
    def ms(key):
        vals = np.array([row[key] for row in rows], dtype=float)
        return vals.mean(), vals.std(ddof=1) if len(vals) > 1 else 0.0
    print("\n" + "="*70)
    print(group_name)
    for k in ["AP_small","AP_medium","AP_large","AP50","AP75","mAP","AR1","AR10","AR100","AR_small","AR_medium","AR_large"]:
        mu, sd = ms(k)
        print(f"{k}: {mu:.3f} ± {sd:.3f}  (epochs {[row['epoch'] for row in rows]})")

# ------------------------------- Grid runner (2) -------------------------------
def run_grid_subprocess(modes, datasets, levels, seeds):
    this = os.path.abspath(sys.argv[0])
    py = sys.executable
    for ds in datasets:
        for lvl in levels:
            for mode in modes:
                for s in seeds:
                    exp = f"{ds}_{lvl}_{mode}_s{s}"
                    cmd = [
                        py, "-u", this,
                        "--mode", mode,
                        "--dataset", ds,
                        "--insert-level", lvl,
                        "--seed", str(s),
                        "--exp-name", exp,
                        "--epochs", str(args.epochs),
                        "--batch-size", str(args.batch_size),
                        "--accum-steps", str(args.accum_steps),
                        "--num-workers", str(args.num_workers),
                        "--prefetch-factor", str(args.prefetch_factor),
                    ]
                    print("[RUN]", " ".join(cmd))
                    subprocess.run(cmd, check=True)

def run_grid_inprocess(modes, datasets, levels, seeds):
    # Clean in-process loop that rebuilds loaders for each seed/dataset (covers item 0)
    for ds in datasets:
        for lvl in levels:
            for mode in modes:
                global CURRENT_CONFIG
                CURRENT_CONFIG = CONFIG_MAP[mode]
                for s in seeds:
                    exp = f"{ds}_{lvl}_{mode}_s{s}"
                    overrides = dict(train_img=args.train_img, train_ann=args.train_ann,
                                     val_img=args.val_img, val_ann=args.val_ann)
                    train_dataset, val_dataset, train_loader, val_loader, patched_val = \
                        build_datasets_and_loaders(s, ds, overrides)
                    print(f"\n=== {exp} ===")
                    best_map, best_epoch, best_ckpt = train_one(exp, train_dataset, val_dataset,
                                                                train_loader, val_loader, patched_val, lvl)
                    print(f"[DONE] {exp}: best mAP {best_map:.4f} @ {best_epoch}; ckpt={best_ckpt}")

# ----------------------------------- main -------------------------------------
def main():
    print(f"Device: {device}")
    print(f"Mode={args.mode} Insert={args.insert_level} Dataset={args.dataset} Seed={args.seed}")

    # Grid
    if args.run_grid:
        modes = args.grid_modes or ["baseline","se","cbam","lgf_gated_spatial"]
        datasets = args.grid_datasets or ["coco_nw"]
        levels = args.grid_levels or ["C3"]
        seeds  = args.grid_seeds or [42,1337,2025]
        if args.subprocess and hasattr(sys, "argv") and sys.argv[0].endswith(".py"):
            run_grid_subprocess(modes, datasets, levels, seeds)
        else:
            run_grid_inprocess(modes, datasets, levels, seeds)
        return

    # Single run
    overrides = dict(train_img=args.train_img, train_ann=args.train_ann,
                     val_img=args.val_img, val_ann=args.val_ann)
    train_dataset, val_dataset, train_loader, val_loader, patched_val = \
        build_datasets_and_loaders(args.seed, args.dataset, overrides)

    global CURRENT_CONFIG
    exp = args.exp_name or f"{args.dataset}_{args.insert_level}_{args.mode}_s{args.seed}"
    print(f"Training on {len(train_dataset)} images; validating on {len(val_dataset)} images")
    print(f"Batch={args.batch_size}, Accum={args.accum_steps}, LR={args.base_lr}, Epochs={args.epochs}")
    print(f"Training on {len(train_dataset)} images; validating on {len(val_dataset)} images")
    print(f"Batch={args.batch-size if hasattr(args,'batch-size') else args.batch_size}, Accum={args.accum_steps}, LR={args.base_lr}, Epochs={args.epochs}")

    best_map, best_epoch, best_ckpt = train_one(exp, train_dataset, val_dataset,
                                                train_loader, val_loader, patched_val, args.insert_level)
    print(f"[RESULT] {exp}: best mAP {best_map:.4f} @ epoch {best_epoch}; ckpt={best_ckpt}")

    # Aggregate if asked
    if args.aggregate:
        agg_datasets = args.agg_datasets or [args.dataset]
        agg_levels   = args.agg_levels or [args.insert_level]
        agg_modes    = args.agg_modes  or [args.mode]
        agg_seeds    = args.agg_seeds
        for ds in agg_datasets:
            for lvl in agg_levels:
                for mode in agg_modes:
                    runs = [f"{ds}_{lvl}_{mode}_s{s}" for s in agg_seeds]
                    aggregate_group(f"Aggregate {ds} {lvl} {mode}", runs)

if __name__ == "__main__":
    main()
