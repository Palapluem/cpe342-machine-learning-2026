"""5-Fold SGKF Generation-2 Corruption-Hardened Trainer (Nightmare-V4 + De-Rotation + SigLIP) for Task 4.

Complements Phase 1/2 (CNXv3 + DINOv3L + EVA02L) with:
1. Vision-Language Contrastive ViTs (`vit_large_patch16_siglip_384`, `vit_so400m_patch14_siglip_384`)
   plus V4 upgrades for `dinov3l_v4` and `eva02l_v4` at 2:1 native aspect ratio.
2. Deterministic Single-Image Black-Border Frame De-Rotation (`detect_and_derotate_np`):
   100% pure geometric restoration (zero test fitting) estimating boundary edge slope on
   dark-border triangles (`[-45, +45]` deg) and straightening before inference.
3. Nightmare-V4 Universal Pre-Specified Augmentations:
   - Rotation `[-45, +45]` deg with black borders
   - RandomErasing / Cutout (1-2 blocks, 8-22% area)
   - Pixelation (downscale 0.35x-0.70x + bilinear/nearest upscale) & Gaussian Blur
   - Neon RGB gain tints, grayscale/saturation jitter, extreme under/overexposure, fog
   - Gaussian noise, JPEG artifacts, MixUp/CutMix (`p=0.30, alpha=0.2`), and Model EMA (`0.997`).
4. Clean De-Rotated 6-View TTA (`v6_derot`): `[Orig, HFlip, DeRot, DeRot+HFlip, GW(Orig), GW(DeRot)]`.

Usage:
    python train_t4_gen2_v4.py --data-dir /content/t4_data --out-dir /content/drive/MyDrive/t4_gen2_v4_outputs --models siglip_l16
"""
from __future__ import annotations

import argparse
import copy
import gc
import io
import math
import os
import random
import shutil
import time
import zipfile
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageFilter
from sklearn.metrics import f1_score, log_loss
from torch.utils.data import DataLoader, Dataset


CLASS_MAP: Dict[str, int] = {
    "7-11k": 0,
    "7-11K": 0,
    "fiveN": 1,
    "freefried": 2,
    "pube": 3,
    "roblock": 4,
}


@dataclass(frozen=True)
class ModelSpec:
    """Configuration for a Gen-2 V4 corruption-hardened Task 4 vision backbone."""

    key: str
    timm_name: str
    img_hw: Tuple[int, int]
    epochs: int
    base_lr: float
    backbone_lr_mult: float
    batch_a100: int
    batch_l4: int
    batch_t4: int
    eval_batch_t4: int
    grad_ckpt: bool
    use_img_size_arg: bool
    channels_last_ampere: bool
    default_freeze_vit_blocks: int


MODEL_SPECS: Dict[str, ModelSpec] = {
    "siglip_l16": ModelSpec(
        key="siglip_l16",
        timm_name="vit_large_patch16_siglip_384",
        img_hw=(192, 384),
        epochs=5,
        base_lr=2.0e-4,
        backbone_lr_mult=0.20,
        batch_a100=64,
        batch_l4=32,
        batch_t4=16,
        eval_batch_t4=48,
        grad_ckpt=True,
        use_img_size_arg=True,
        channels_last_ampere=False,
        default_freeze_vit_blocks=8,
    ),
    "siglip_so400m": ModelSpec(
        key="siglip_so400m",
        timm_name="vit_so400m_patch14_siglip_384",
        img_hw=(196, 392),
        epochs=4,
        base_lr=1.6e-4,
        backbone_lr_mult=0.18,
        batch_a100=48,
        batch_l4=24,
        batch_t4=12,
        eval_batch_t4=36,
        grad_ckpt=True,
        use_img_size_arg=True,
        channels_last_ampere=False,
        default_freeze_vit_blocks=10,
    ),
    "dinov3l_v4": ModelSpec(
        key="dinov3l_v4",
        timm_name="vit_large_patch16_dinov3.lvd1689m",
        img_hw=(192, 384),
        epochs=5,
        base_lr=2.0e-4,
        backbone_lr_mult=0.22,
        batch_a100=64,
        batch_l4=32,
        batch_t4=16,
        eval_batch_t4=48,
        grad_ckpt=True,
        use_img_size_arg=True,
        channels_last_ampere=False,
        default_freeze_vit_blocks=6,
    ),
    "eva02l_v4": ModelSpec(
        key="eva02l_v4",
        timm_name="eva02_large_patch14_448.mim_m38m_ft_in22k_in1k",
        img_hw=(196, 392),
        epochs=4,
        base_lr=2.0e-4,
        backbone_lr_mult=0.22,
        batch_a100=48,
        batch_l4=32,
        batch_t4=16,
        eval_batch_t4=48,
        grad_ckpt=True,
        use_img_size_arg=True,
        channels_last_ampere=False,
        default_freeze_vit_blocks=6,
    ),
}


def seed_everything(seed: int) -> None:
    """Set deterministic RNG seeds across stdlib, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_amp_dtype(device: torch.device) -> Tuple[Optional[torch.dtype], bool]:
    """Select native AMP precision; force float16 + GradScaler on sm_75 T4."""
    if device.type != "cuda":
        return None, False
    cc_major, _ = torch.cuda.get_device_capability(0)
    if cc_major >= 8 and torch.cuda.is_bf16_supported():
        return torch.bfloat16, False
    return torch.float16, True


def load_jpeg_bytes(folder: str, filenames: List[str]) -> List[bytes]:
    """Preload compressed JPEG byte buffers into host RAM."""
    buffers: List[bytes] = []
    for fn in filenames:
        with open(os.path.join(folder, fn), "rb") as f:
            buffers.append(f.read())
    return buffers


def _fit_robust_line_slope(xs: np.ndarray, ys: np.ndarray) -> Tuple[float, int]:
    """Robustly fit line slope dy/dx from longest contiguous/inlier edge points."""
    if len(xs) < 12:
        return 0.0, 0
    # Compute pairwise slopes across strided points and take median, then refine inliers
    step = max(1, len(xs) // 24)
    slopes: List[float] = []
    for i in range(0, len(xs), step):
        dx = xs - xs[i]
        dy = ys - ys[i]
        valid = np.abs(dx) >= max(8.0, 0.10 * (xs.max() - xs.min()))
        if valid.any():
            slopes.extend((dy[valid] / dx[valid]).tolist())
    if not slopes:
        return 0.0, 0
    s0 = float(np.median(slopes))
    resid = ys - s0 * xs
    c0 = float(np.median(resid))
    inliers = np.abs(resid - c0) <= 3.0
    if int(inliers.sum()) < 10:
        return 0.0, 0
    xi, yi = xs[inliers], ys[inliers]
    x_span = float(xi.max() - xi.min())
    if x_span < 12.0:
        return 0.0, 0
    xc = xi - xi.mean()
    yc = yi - yi.mean()
    slope = float(np.dot(xc, yc) / max(np.dot(xc, xc), 1e-6))
    return slope, int(inliers.sum())


def estimate_border_rotation_deg(arr_u8: np.ndarray, black_thr: int = 14) -> float:
    """Estimate frame rotation angle in [-45, +45] deg from dark corner wedges."""
    h, w, _ = arr_u8.shape
    lum = arr_u8.max(axis=2)
    ch, cw = max(2, int(h * 0.14)), max(2, int(w * 0.14))
    corners = np.concatenate(
        [
            lum[:ch, :cw].ravel(),
            lum[:ch, -cw:].ravel(),
            lum[-ch:, :cw].ravel(),
            lum[-ch:, -cw:].ravel(),
        ]
    )
    center = lum[int(h * 0.25) : int(h * 0.75), int(w * 0.25) : int(w * 0.75)]
    if float((corners < black_thr).mean()) < 0.08 or float((center >= black_thr + 6).mean()) < 0.25:
        return 0.0

    active = lum >= black_thr
    # Find uppermost and lowermost active row inside interior [3..h-4] for each column
    top_y = np.argmax(active, axis=0).astype(np.float64)
    has_active_col = active.any(axis=0)
    bot_y = (h - 1 - np.argmax(active[::-1, :], axis=0)).astype(np.float64)

    xs_all = np.arange(w, dtype=np.float64)
    m_top = has_active_col & (top_y >= 2.0) & (top_y <= 0.45 * h) & (xs_all >= 2.0) & (xs_all <= w - 3.0)
    m_bot = has_active_col & (bot_y <= h - 3.0) & (bot_y >= 0.55 * h) & (xs_all >= 2.0) & (xs_all <= w - 3.0)

    estimates: List[Tuple[float, int]] = []
    if int(m_top.sum()) >= 12:
        sl, nin = _fit_robust_line_slope(xs_all[m_top], top_y[m_top])
        if nin >= 10:
            estimates.append((sl, nin))
    if int(m_bot.sum()) >= 12:
        sl, nin = _fit_robust_line_slope(xs_all[m_bot], bot_y[m_bot])
        if nin >= 10:
            estimates.append((sl, nin))

    if not estimates:
        return 0.0
    # Weighted slope average across detected top/bottom edges
    total_w = sum(n for _, n in estimates)
    slope_mean = sum(s * n for s, n in estimates) / total_w
    # In image coordinates (x right, y down), PIL.rotate(+ang) is CCW, causing top edge dy/dx < 0.
    # Straightening back requires rotating by -theta_ccw = +arctan(dy/dx) in PIL degrees.
    ang_deg = float(np.degrees(np.arctan(slope_mean)))
    if abs(ang_deg) < 3.0 or abs(ang_deg) > 46.0:
        return 0.0
    return float(np.clip(ang_deg, -45.0, 45.0))


def detect_and_derotate_pil(pil_img: Image.Image) -> Tuple[Image.Image, float]:
    """Deterministically straighten a rotated frame with black borders (single-image, zero leakage)."""
    arr = np.asarray(pil_img, dtype=np.uint8)
    corr_deg = estimate_border_rotation_deg(arr)
    if abs(corr_deg) < 3.0:
        return pil_img, 0.0
    # PIL rotate by corr_deg cancels the black-border tilt
    derot = pil_img.rotate(corr_deg, resample=Image.BILINEAR, fillcolor=(0, 0, 0))
    return derot, corr_deg


def apply_nightmare_v4(pil_img: Image.Image, rng: np.random.Generator) -> np.ndarray:
    """Universal Nightmare-V4 pre-specified corruptions (no test statistics used)."""
    w, h = pil_img.size
    if rng.random() < 0.50:
        pil_img = pil_img.transpose(Image.FLIP_LEFT_RIGHT)
    if rng.random() < 0.60:
        ang = float(rng.uniform(-45.0, 45.0))
        pil_img = pil_img.rotate(ang, resample=Image.BILINEAR, fillcolor=(0, 0, 0))
    if rng.random() < 0.22:
        scale = float(rng.uniform(0.35, 0.70))
        sw, sh = max(16, int(w * scale)), max(8, int(h * scale))
        resample_down = Image.BILINEAR if rng.random() < 0.5 else Image.NEAREST
        pil_img = pil_img.resize((sw, sh), resample=resample_down).resize((w, h), resample=Image.BILINEAR)
    if rng.random() < 0.18:
        pil_img = pil_img.filter(ImageFilter.GaussianBlur(radius=float(rng.uniform(0.4, 1.5))))

    arr = np.asarray(pil_img, dtype=np.float32) * (1.0 / 255.0)

    # Color tint & saturation / grayscale shifts
    r_col = rng.random()
    if r_col < 0.14:
        gray = arr.mean(axis=2, keepdims=True)
        alpha_sat = float(rng.uniform(0.0, 0.35))
        arr = gray * (1.0 - alpha_sat) + arr * alpha_sat
    elif r_col < 0.65:
        gains = rng.uniform(0.28, 2.25, size=(1, 1, 3)).astype(np.float32)
        gains /= float(gains.mean())
        arr = np.clip(arr * gains, 0.0, 1.0)

    # Exposure & contrast regimes
    r_exp = rng.random()
    if r_exp < 0.22:
        arr = np.power(np.clip(arr, 1e-4, 1.0), float(rng.uniform(1.8, 3.2))) * float(rng.uniform(0.28, 0.70))
    elif r_exp < 0.42:
        arr = np.clip(arr * float(rng.uniform(1.25, 2.15)) + float(rng.uniform(0.22, 0.48)), 0.0, 1.0)
    elif r_exp < 0.56:
        arr = np.clip(0.5 + (arr - 0.5) * float(rng.uniform(0.28, 0.65)) + float(rng.uniform(-0.15, 0.15)), 0.0, 1.0)

    # RandomErasing / Cutout (1-2 blocks, 8-22% total area) simulating HUD popups or heavy occlusion
    if rng.random() < 0.30:
        n_blocks = 1 if rng.random() < 0.65 else 2
        for _ in range(n_blocks):
            area_frac = float(rng.uniform(0.06, 0.14))
            bh = max(4, int(h * math.sqrt(area_frac) * float(rng.uniform(0.6, 1.4))))
            bw = max(8, int((w * h * area_frac) / max(bh, 1)))
            bh, bw = min(bh, h - 2), min(bw, w - 2)
            y0 = int(rng.integers(0, max(1, h - bh)))
            x0 = int(rng.integers(0, max(1, w - bw)))
            fill_val = float(rng.uniform(0.0, 1.0)) if rng.random() < 0.4 else 0.0
            arr[y0 : y0 + bh, x0 : x0 + bw, :] = fill_val

    if rng.random() < 0.40:
        arr = arr + rng.normal(0.0, float(rng.uniform(0.015, 0.080)), size=arr.shape).astype(np.float32)
    arr_u8 = np.clip(arr * 255.0, 0.0, 255.0).astype(np.uint8)

    if rng.random() < 0.22:
        buf = io.BytesIO()
        Image.fromarray(arr_u8).save(buf, format="JPEG", quality=int(rng.integers(24, 75)))
        buf.seek(0)
        arr_u8 = np.asarray(Image.open(buf).convert("RGB"), dtype=np.uint8)

    return arr_u8


def interior_gw_stretch_np(arr_u8: np.ndarray) -> np.ndarray:
    """Normalize extreme color tints and exposures using central 60% interior statistics."""
    h, w, _ = arr_u8.shape
    h0, h1 = int(h * 0.2), int(h * 0.8)
    w0, w1 = int(w * 0.2), int(w * 0.8)
    b = arr_u8.astype(np.float32) * (1.0 / 255.0)
    patch = b[h0:h1:2, w0:w1:2, :]
    ch_m = np.clip(patch.mean(axis=(0, 1), keepdims=True), 0.025, 1.0)
    b_gw = np.clip(b * (ch_m.mean(axis=2, keepdims=True) / ch_m), 0.0, 1.0)
    p_sub = b_gw[h0:h1:3, w0:w1:3, :].ravel()
    lo, hi = np.quantile(p_sub, (0.02, 0.98))
    b_st = np.clip((b_gw - float(lo)) / max(float(hi - lo), 0.10), 0.0, 1.0)
    m_int = float(np.clip(b_st[h0:h1:3, w0:w1:3, :].mean(), 0.05, 0.95))
    gamma = float(np.clip(math.log(0.45) / math.log(m_int), 0.50, 1.80))
    return np.clip(np.power(b_st, gamma), 0.0, 1.0).astype(np.float32)


def interior_gw_stretch_torch(b: torch.Tensor) -> torch.Tensor:
    """Vectorized GPU Gray-World + Percentile Contrast Stretch matching interior_gw_stretch_np."""
    _, _, h, w = b.shape
    h0, h1 = int(h * 0.2), int(h * 0.8)
    w0, w1 = int(w * 0.2), int(w * 0.8)
    patch = b[:, :, h0:h1:2, w0:w1:2]
    ch_m = patch.mean(dim=(2, 3), keepdim=True).clamp(0.025, 1.0)
    b_gw = (b * (ch_m.mean(dim=1, keepdim=True) / ch_m)).clamp(0.0, 1.0)
    p_sub = b_gw[:, :, h0:h1:3, w0:w1:3].reshape(b.shape[0], -1)
    q = torch.tensor([0.02, 0.98], device=b.device, dtype=b.dtype)
    qs = torch.quantile(p_sub, q, dim=1)
    lo = qs[0].view(-1, 1, 1, 1)
    hi = qs[1].view(-1, 1, 1, 1)
    b_st = ((b_gw - lo) / (hi - lo).clamp(min=0.10)).clamp(0.0, 1.0)
    m_int = b_st[:, :, h0:h1:3, w0:w1:3].mean(dim=(1, 2, 3)).clamp(0.05, 0.95)
    gamma = (math.log(0.45) / torch.log(m_int)).clamp(0.50, 1.80).view(-1, 1, 1, 1)
    return b_st.pow(gamma).clamp(0.0, 1.0)


def make_deterministic_exthard_v2(
    jpeg_bytes_list: List[bytes], indices: np.ndarray, seed: int = 456
) -> List[np.ndarray]:
    """Generate deterministic ExtHard-V2 stress validation images (rot up to +-42 deg, pixelation, cutout)."""
    rng = np.random.default_rng(seed)
    out: List[np.ndarray] = []
    for k, idx in enumerate(indices):
        im = Image.open(io.BytesIO(jpeg_bytes_list[int(idx)])).convert("RGB")
        im = im.rotate(float(rng.uniform(-42.0, 42.0)), resample=Image.BILINEAR, fillcolor=(0, 0, 0))
        if k % 5 == 0:
            w, h = im.size
            sc = float(rng.uniform(0.40, 0.65))
            im = im.resize((max(16, int(w * sc)), max(8, int(h * sc))), Image.BILINEAR).resize((w, h), Image.BILINEAR)
        arr = np.asarray(im, dtype=np.float32) * (1.0 / 255.0)
        mode = k % 5
        if mode == 0:
            arr = np.power(np.clip(arr, 1e-4, 1.0), float(rng.uniform(1.8, 2.9))) * float(rng.uniform(0.32, 0.65))
        elif mode == 1:
            arr = np.clip(arr * float(rng.uniform(1.3, 2.05)) + float(rng.uniform(0.28, 0.48)), 0.0, 1.0)
        elif mode == 2:
            g = rng.uniform(0.32, 2.1, size=(1, 1, 3)).astype(np.float32)
            g /= float(g.max())
            arr = np.clip(np.power(arr, 0.85) * g * float(rng.uniform(0.8, 1.4)), 0.0, 1.0)
        elif mode == 3:
            gray = arr.mean(axis=2, keepdims=True)
            arr = np.clip(gray * 0.8 + arr * 0.2, 0.0, 1.0)
        else:
            arr = np.clip(0.5 + (arr - 0.5) * float(rng.uniform(0.35, 0.68)) + rng.uniform(-0.14, 0.14, (1, 1, 3)), 0.0, 1.0)
        arr = arr + rng.normal(0.0, float(rng.uniform(0.025, 0.070)), size=arr.shape).astype(np.float32)
        out.append(np.clip(arr * 255.0, 0.0, 255.0).astype(np.uint8))
    return out


class Task4TrainV4Dataset(Dataset):
    """Training dataset applying optional deterministic de-rotation on clean inputs + Nightmare-V4 + IntGW."""

    def __init__(
        self,
        jpeg_bytes: List[bytes],
        indices: np.ndarray,
        labels: np.ndarray,
        img_hw: Tuple[int, int],
        p_nightmare: float = 0.52,
        p_norm_view: float = 0.25,
        seed: int = 42,
    ) -> None:
        self.jpeg_bytes = jpeg_bytes
        self.indices = indices
        self.labels = labels
        self.img_hw = img_hw
        self.p_nightmare = p_nightmare
        self.p_norm_view = p_norm_view
        self.seed = seed
        self._rng: Optional[np.random.Generator] = None

    def _get_rng(self) -> np.random.Generator:
        if self._rng is None:
            info = torch.utils.data.get_worker_info()
            wid = info.id if info is not None else 0
            self._rng = np.random.default_rng(self.seed + wid * 7919)
        return self._rng

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, pos: int) -> Tuple[torch.Tensor, torch.Tensor]:
        rng = self._get_rng()
        idx = int(self.indices[pos])
        pil = Image.open(io.BytesIO(self.jpeg_bytes[idx])).convert("RGB")
        target_wh = (self.img_hw[1], self.img_hw[0])
        if pil.size != target_wh:
            pil = pil.resize(target_wh, resample=Image.BILINEAR)

        if rng.random() < 0.35:
            pil, _ = detect_and_derotate_pil(pil)

        if rng.random() < self.p_nightmare:
            arr_u8 = apply_nightmare_v4(pil, rng)
        else:
            if rng.random() < 0.50:
                pil = pil.transpose(Image.FLIP_LEFT_RIGHT)
            arr_u8 = np.asarray(pil, dtype=np.uint8)

        arr_f = interior_gw_stretch_np(arr_u8) if rng.random() < self.p_norm_view else (arr_u8.astype(np.float32) * (1.0 / 255.0))
        return torch.from_numpy(arr_f).permute(2, 0, 1).contiguous(), torch.tensor(int(self.labels[idx]), dtype=torch.long)


class Task4DeRotViewsDataset(Dataset):
    """Inference dataset yielding `[Orig, DeRotated]` spatial pairs for clean 6-view GPU TTA."""

    def __init__(
        self,
        source_items: List[bytes] | List[np.ndarray],
        indices: Optional[np.ndarray],
        img_hw: Tuple[int, int],
        is_bytes: bool = True,
    ) -> None:
        self.source_items = source_items
        self.indices = indices if indices is not None else np.arange(len(source_items))
        self.img_hw = img_hw
        self.is_bytes = is_bytes

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, pos: int) -> torch.Tensor:
        item = self.source_items[int(self.indices[pos])]
        pil = Image.open(io.BytesIO(item)).convert("RGB") if self.is_bytes else Image.fromarray(item)
        target_wh = (self.img_hw[1], self.img_hw[0])
        if pil.size != target_wh:
            pil = pil.resize(target_wh, resample=Image.BILINEAR)
        derot_pil, _ = detect_and_derotate_pil(pil)
        stacked = np.stack([np.asarray(pil, dtype=np.uint8), np.asarray(derot_pil, dtype=np.uint8)], axis=0)
        return torch.from_numpy(stacked).permute(0, 3, 1, 2).contiguous()


def apply_t4_freeze(model: nn.Module, spec: ModelSpec, freeze_vit_blocks: int) -> Tuple[int, int]:
    """Freeze only lowest ViT blocks (keep patch_embed/pos_embed/head/attn_pool trainable)."""
    total_params = sum(p.numel() for p in model.parameters())
    if freeze_vit_blocks > 0:
        for name, param in model.named_parameters():
            if name.startswith("blocks."):
                parts = name.split(".")
                if len(parts) > 1 and parts[1].isdigit() and int(parts[1]) < freeze_vit_blocks:
                    param.requires_grad_(False)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable_params, total_params


def build_model(
    spec: ModelSpec,
    pretrained: bool = True,
    freeze_vit_blocks: Optional[int] = None,
) -> Tuple[nn.Module, Tuple[float, ...], Tuple[float, ...], Tuple[int, int]]:
    """Instantiate timm model with exact 2:1 img_size, partial freeze, and RGB normalization."""
    fb = spec.default_freeze_vit_blocks if freeze_vit_blocks is None else freeze_vit_blocks
    kwargs: Dict[str, object] = {"pretrained": pretrained, "num_classes": 5}
    if spec.use_img_size_arg:
        kwargs["img_size"] = spec.img_hw
    model = timm.create_model(spec.timm_name, **kwargs)
    if spec.grad_ckpt and hasattr(model, "set_grad_checkpointing"):
        model.set_grad_checkpointing(True)
    param_counts = apply_t4_freeze(model, spec, freeze_vit_blocks=fb)
    cfg = timm.data.resolve_model_data_config(model)
    return model, tuple(cfg["mean"]), tuple(cfg["std"]), param_counts


def pick_batch_and_accum(spec: ModelSpec, device: torch.device, target_eff_bs: int = 48) -> Tuple[int, int, int]:
    """Select batch size and gradient accumulation for detected GPU VRAM."""
    if device.type != "cuda":
        return 8, 1, 16
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    if vram_gb >= 36.0:
        bs, eval_bs = spec.batch_a100, 96
    elif vram_gb >= 21.0:
        bs, eval_bs = spec.batch_l4, 48
    else:
        bs, eval_bs = spec.batch_t4, spec.eval_batch_t4
    return bs, max(1, math.ceil(target_eff_bs / bs)), eval_bs


def make_param_groups(model: nn.Module, base_lr: float, bb_mult: float, wd: float = 2e-2) -> List[Dict]:
    """Separate classifier/pool heads from backbone weights and exclude 1D norm/bias from decay."""
    hd_d, hd_n, bb_d, bb_n = [], [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_h = name.startswith("head") or name.startswith("fc") or name.startswith("attn_pool") or ".head." in name
        no_w = p.ndim <= 1 or name.endswith(".bias") or "norm" in name.lower()
        target = (hd_n if no_w else hd_d) if is_h else (bb_n if no_w else bb_d)
        target.append(p)
    groups: List[Dict] = []
    for lst, lr_v, wd_v in [
        (hd_d, base_lr, wd),
        (hd_n, base_lr, 0.0),
        (bb_d, base_lr * bb_mult, wd),
        (bb_n, base_lr * bb_mult, 0.0),
    ]:
        if lst:
            groups.append({"params": lst, "lr": lr_v, "weight_decay": wd_v})
    return groups


@torch.no_grad()
def predict_derot_6view_tta(
    model: nn.Module,
    source_items: List[bytes] | List[np.ndarray],
    indices: Optional[np.ndarray],
    spec: ModelSpec,
    mean_t: torch.Tensor,
    std_t: torch.Tensor,
    device: torch.device,
    amp_dtype: Optional[torch.dtype],
    batch_size: int = 48,
    num_workers: int = 2,
    is_bytes: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run clean De-Rotated 6-view GPU TTA: [Orig, HFlip, DeRot, DeRot+HFlip, GW(Orig), GW(DeRot)].

    Returns:
      - `orig2`: weighted `[Orig, HFlip, DeRot]` (`0.40, 0.25, 0.35`) compatible with `oof_orig2` slot
      - `w6`:    weighted 6-view probabilities (`0.22, 0.14, 0.24, 0.14, 0.13, 0.13`) compatible with `oof_w6` slot
    """
    ds = Task4DeRotViewsDataset(source_items, indices, spec.img_hw, is_bytes=is_bytes)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=(device.type == "cuda"))
    model.eval()
    probs_6_all: List[np.ndarray] = []

    for x_u8 in loader:
        b, _, c, h, w = x_u8.shape
        x_f = x_u8.to(device, non_blocking=True).float().mul_(1.0 / 255.0)
        v_orig = x_f[:, 0]
        v_derot = x_f[:, 1]
        v_orig_flip = torch.flip(v_orig, dims=[3])
        v_derot_flip = torch.flip(v_derot, dims=[3])
        v_orig_gw = interior_gw_stretch_torch(v_orig)
        v_derot_gw = interior_gw_stretch_torch(v_derot)

        views = torch.stack([v_orig, v_orig_flip, v_derot, v_derot_flip, v_orig_gw, v_derot_gw], dim=1)
        flat = views.view(b * 6, c, h, w).sub_(mean_t).div_(std_t).contiguous()
        if amp_dtype is not None:
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                logits = model(flat)
        else:
            logits = model(flat)
        pr = F.softmax(logits.float(), dim=1).view(b, 6, 5).cpu().numpy()
        probs_6_all.append(pr)

    all_6 = np.concatenate(probs_6_all, axis=0)
    w_o2 = np.array([0.40, 0.25, 0.35], dtype=np.float64)[None, :, None]
    orig2 = np.sum(all_6[:, [0, 1, 2], :] * w_o2, axis=1)
    orig2 /= np.clip(orig2.sum(axis=1, keepdims=True), 1e-8, None)

    w_v6 = np.array([0.22, 0.14, 0.24, 0.14, 0.13, 0.13], dtype=np.float64)[None, :, None]
    w6 = np.sum(all_6 * w_v6, axis=1)
    w6 /= np.clip(w6.sum(axis=1, keepdims=True), 1e-8, None)
    return orig2, w6


def apply_mixup_cutmix(
    xb: torch.Tensor,
    yb: torch.Tensor,
    num_classes: int = 5,
    p_apply: float = 0.30,
    alpha: float = 0.2,
    label_smoothing: float = 0.04,
) -> torch.Tensor:
    """Apply soft-label MixUp or CutMix in batch tensor space."""
    y_oh = F.one_hot(yb, num_classes=num_classes).float()
    y_soft = y_oh * (1.0 - label_smoothing) + label_smoothing / num_classes
    if random.random() >= p_apply or xb.size(0) < 2:
        return y_soft
    lam = float(np.random.beta(alpha, alpha))
    perm = torch.randperm(xb.size(0), device=xb.device)
    if random.random() < 0.50:
        xb.mul_(lam).add_(xb[perm], alpha=(1.0 - lam))
        return lam * y_soft + (1.0 - lam) * y_soft[perm]
    _, _, h, w = xb.shape
    cut_r = math.sqrt(max(1e-4, 1.0 - lam))
    cw, ch = max(1, int(w * cut_r)), max(1, int(h * cut_r))
    cx, cy = int(random.uniform(0, w)), int(random.uniform(0, h))
    x1, x2 = max(0, cx - cw // 2), min(w, cx + cw // 2)
    y1, y2 = max(0, cy - ch // 2), min(h, cy + ch // 2)
    xb[:, :, y1:y2, x1:x2] = xb[perm, :, y1:y2, x1:x2]
    lam_adj = 1.0 - float((x2 - x1) * (y2 - y1)) / float(w * h)
    return lam_adj * y_soft + (1.0 - lam_adj) * y_soft[perm]


def update_ema(ema_state: Dict[str, torch.Tensor], model: nn.Module, decay: float = 0.997) -> None:
    """Update exponential moving average parameter state on device."""
    with torch.no_grad():
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if name not in ema_state:
                ema_state[name] = p.detach().clone()
            else:
                ema_state[name].mul_(decay).add_(p.detach(), alpha=(1.0 - decay))


def apply_ema_weights(model: nn.Module, ema_state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Swap EMA weights into model for evaluation and return original state backup."""
    backup = {}
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name in ema_state:
                backup[name] = p.detach().clone()
                p.copy_(ema_state[name])
    return backup


def restore_weights(model: nn.Module, backup: Dict[str, torch.Tensor]) -> None:
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name in backup:
                p.copy_(backup[name])


def train_one_model(
    spec: ModelSpec,
    data_dir: Path,
    out_dir: Path,
    folds_arg: Optional[List[int]],
    epochs_override: Optional[int],
    freeze_override: Optional[int],
    num_workers: int,
    resume: bool,
) -> None:
    """Train all 5 SGKF folds of a Gen-2 V4 model with EMA, atomic checkpoints, and export bundle."""
    train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
    test_df = pd.read_csv(os.path.join(data_dir, "test.csv"))
    y = train_df["class"].map(CLASS_MAP).astype(int).to_numpy() if "class" in train_df.columns else train_df["label"].astype(int).to_numpy()
    tr_files = (train_df["file_name"] if "file_name" in train_df.columns else (train_df["id"].astype(str) + ".jpg")).astype(str).tolist()
    te_files = (test_df["file_name"] if "file_name" in test_df.columns else (test_df["id"].astype(str) + ".jpg")).astype(str).tolist()

    folds = np.load(os.path.join(data_dir, "folds_sgkf.npy")).astype(int)
    assert len(folds) == len(y)

    tr_bytes = load_jpeg_bytes(os.path.join(data_dir, "train"), tr_files)
    te_bytes = load_jpeg_bytes(os.path.join(data_dir, "test"), te_files)
    exthard_imgs = make_deterministic_exthard_v2(tr_bytes, np.arange(len(y)), seed=456)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype, use_scaler = select_amp_dtype(device)
    bs, accum, eval_bs = pick_batch_and_accum(spec, device, target_eff_bs=48)
    epochs = epochs_override if epochs_override is not None else spec.epochs
    fold_list = folds_arg if folds_arg is not None else list(range(5))

    m_dir = os.path.join(out_dir, f"folds_{spec.key}")
    os.makedirs(m_dir, exist_ok=True)
    print(f"\n[Gen-2 V4 {spec.key}] {spec.timm_name} | hw={spec.img_hw} | epochs={epochs} | bs={bs}x{accum} | amp={amp_dtype}")

    for k in fold_list:
        fold_npz = os.path.join(m_dir, f"fold{k}.npz")
        if resume and os.path.isfile(fold_npz):
            print(f"  [Skip] fold{k}.npz already completed")
            continue

        seed_everything(42 + k)
        tr_idx = np.where(folds != k)[0]
        va_idx = np.where(folds == k)[0]

        counts = np.bincount(y[tr_idx], minlength=5).astype(np.float32)
        cw = np.power(counts.sum() / (5.0 * counts), 0.35)
        cw_t = torch.tensor(cw / cw.mean(), device=device, dtype=torch.float32)

        model, mean_c, std_c, (n_tr_p, n_tot_p) = build_model(spec, pretrained=True, freeze_vit_blocks=freeze_override)
        model = model.to(device)
        mean_t = torch.tensor(mean_c, device=device, dtype=torch.float32).view(1, 3, 1, 1)
        std_t = torch.tensor(std_c, device=device, dtype=torch.float32).view(1, 3, 1, 1)

        train_ds = Task4TrainV4Dataset(tr_bytes, tr_idx, y, spec.img_hw, p_nightmare=0.52, p_norm_view=0.25, seed=100 + k)
        train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=num_workers, pin_memory=(device.type == "cuda"), drop_last=True)

        opt = torch.optim.AdamW(make_param_groups(model, spec.base_lr, spec.backbone_lr_mult))
        total_steps = max(1, len(train_loader) // accum) * epochs
        warmup = max(1, int(0.10 * total_steps))
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt,
            lambda st: (st + 1) / warmup if st < warmup else 0.5 * (1.0 + math.cos(math.pi * (st - warmup) / max(1, total_steps - warmup))),
        )
        scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
        ema_state: Dict[str, torch.Tensor] = {}

        ckpt_path = os.path.join(m_dir, f"fold{k}_ckpt.pt")
        start_epoch = 0
        if resume and os.path.isfile(ckpt_path):
            ck = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ck["model"])
            opt.load_state_dict(ck["opt"])
            sched.load_state_dict(ck["sched"])
            if use_scaler and "scaler" in ck:
                scaler.load_state_dict(ck["scaler"])
            ema_state = {ek: ev.to(device) for ek, ev in ck.get("ema", {}).items()}
            start_epoch = int(ck["epoch"]) + 1
            print(f"  Resuming {spec.key} fold {k} from epoch {start_epoch}/{epochs} (trainable={n_tr_p/1e6:.1f}M)")

        t0 = time.time()
        for ep in range(start_epoch, epochs):
            model.train()
            opt.zero_grad(set_to_none=True)
            running_loss, n_batches = 0.0, 0
            for step, (xb, yb) in enumerate(train_loader, 1):
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                y_soft = apply_mixup_cutmix(xb, yb, num_classes=5, p_apply=0.30, alpha=0.2, label_smoothing=0.04)
                xb = xb.sub_(mean_t).div_(std_t).contiguous()
                if amp_dtype is not None:
                    with torch.autocast(device_type="cuda", dtype=amp_dtype):
                        logits = model(xb)
                        log_pr = F.log_softmax(logits.float(), dim=1)
                        loss = -(y_soft * cw_t[None, :] * log_pr).sum(dim=1).mean() / accum
                else:
                    logits = model(xb)
                    log_pr = F.log_softmax(logits.float(), dim=1)
                    loss = -(y_soft * cw_t[None, :] * log_pr).sum(dim=1).mean() / accum

                if use_scaler:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                running_loss += float(loss.item()) * accum
                n_batches += 1

                if step % accum == 0 or step == len(train_loader):
                    if use_scaler:
                        scaler.unscale_(opt)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                        scaler.step(opt)
                        scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                        opt.step()
                    opt.zero_grad(set_to_none=True)
                    sched.step()
                    update_ema(ema_state, model, decay=0.997)

            tmp_ck = ckpt_path + ".tmp"
            torch.save(
                {
                    "epoch": ep,
                    "model": model.state_dict(),
                    "opt": opt.state_dict(),
                    "sched": sched.state_dict(),
                    "scaler": scaler.state_dict() if use_scaler else {},
                    "ema": {ek: ev.cpu() for ek, ev in ema_state.items()},
                },
                tmp_ck,
            )
            os.replace(tmp_ck, ckpt_path)
            print(f"  [{spec.key} F{k}] Ep {ep+1}/{epochs} | loss={running_loss/max(1,n_batches):.4f} | elapsed={time.time()-t0:.0f}s")

        del opt, sched
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

        backup = apply_ema_weights(model, ema_state)
        va_o2, va_w6 = predict_derot_6view_tta(model, tr_bytes, va_idx, spec, mean_t, std_t, device, amp_dtype, batch_size=eval_bs, num_workers=num_workers, is_bytes=True)
        _, ext_w6 = predict_derot_6view_tta(model, exthard_imgs, va_idx, spec, mean_t, std_t, device, amp_dtype, batch_size=eval_bs, num_workers=num_workers, is_bytes=False)
        te_o2, te_w6 = predict_derot_6view_tta(model, te_bytes, None, spec, mean_t, std_t, device, amp_dtype, batch_size=eval_bs, num_workers=num_workers, is_bytes=True)
        restore_weights(model, backup)

        f1_v = float(f1_score(y[va_idx], va_w6.argmax(1), average="macro"))
        f1_e = float(f1_score(y[va_idx], ext_w6.argmax(1), average="macro"))
        print(f"  [Done {spec.key} F{k}] Val6TTA F1={f1_v:.5f} | ExtHardV2 F1={f1_e:.5f} | Val LL={log_loss(y[va_idx], va_w6):.5f}")

        tmp_npz = fold_npz + ".tmp.npz"
        np.savez_compressed(tmp_npz, va_idx=va_idx, va_o2=va_o2, va_w6=va_w6, ext_w6=ext_w6, te_o2=te_o2, te_w6=te_w6)
        os.replace(tmp_npz, fold_npz)
        if os.path.isfile(ckpt_path):
            os.remove(ckpt_path)

        del model, scaler, ema_state, backup
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Merge if all 5 folds exist
    fold_files = [os.path.join(m_dir, f"fold{k}.npz") for k in range(5)]
    if not all(os.path.isfile(p) for p in fold_files):
        print(f"[{spec.key}] Partial folds finished; waiting for remaining folds before bundle merge.")
        return

    n_tr, n_te = len(y), len(te_files)
    oof_o2 = np.zeros((n_tr, 5), dtype=np.float32)
    oof_w6 = np.zeros((n_tr, 5), dtype=np.float32)
    ext_w6 = np.zeros((n_tr, 5), dtype=np.float32)
    te_o2 = np.zeros((n_te, 5), dtype=np.float32)
    te_w6 = np.zeros((n_te, 5), dtype=np.float32)
    for k, fp in enumerate(fold_files):
        d = np.load(fp)
        vi = d["va_idx"]
        oof_o2[vi], oof_w6[vi], ext_w6[vi] = d["va_o2"], d["va_w6"], d["ext_w6"]
        te_o2 += d["te_o2"] / 5.0
        te_w6 += d["te_w6"] / 5.0

    print(
        f"\n>>> [5-Fold Complete {spec.key}] Orig2 F1={f1_score(y, oof_o2.argmax(1), average='macro'):.6f} | "
        f"W6 F1={f1_score(y, oof_w6.argmax(1), average='macro'):.6f} | ExtHardV2 F1={f1_score(y, ext_w6.argmax(1), average='macro'):.6f} | "
        f"W6 LL={log_loss(y, oof_w6):.6f}"
    )

    out_bundle = os.path.join(out_dir, f"task4-{spec.key}-sgkf-hard-bundle.npz")
    np.savez_compressed(out_bundle, folds=folds, oof_orig2=oof_o2, oof_w6=oof_w6, ext_w6=ext_w6, te_orig2=te_o2, te_w6=te_w6)

    p_cols = [f"prob_{i}" for i in range(5)]
    tr_ids = train_df["id"].astype(str).tolist()
    te_ids = test_df["id"].astype(str).tolist()
    exported = [out_bundle]
    for tag, tr_arr, te_arr in [("orig2", oof_o2, te_o2), ("hard6", oof_w6, te_w6)]:
        o_p = os.path.join(out_dir, f"task4-{spec.key}-{tag}-oof.csv")
        t_p = os.path.join(out_dir, f"task4-{spec.key}-{tag}-test-predictions.csv")
        pd.DataFrame({"id": tr_ids, "file_name": tr_files, "label": y, "fold": folds, **{p_cols[c]: tr_arr[:, c] for c in range(5)}}).to_csv(o_p, index=False)
        pd.DataFrame({"id": te_ids, "file_name": te_files, **{p_cols[c]: te_arr[:, c] for c in range(5)}}).to_csv(t_p, index=False)
        exported.extend([o_p, t_p])

    zip_p = os.path.join(out_dir, f"t4_gen2_{spec.key}.zip")
    with zipfile.ZipFile(zip_p, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for fpath in exported:
            zf.write(fpath, arcname=os.path.basename(fpath))
    print(f"Packaged export bundle: {zip_p}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Task 4 Gen-2 V4 Corruption-Hardened Trainer")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--models", nargs="+", default=["siglip_l16"], choices=list(MODEL_SPECS.keys()))
    ap.add_argument("--folds", nargs="+", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--freeze-vit-blocks", type=int, default=None)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    for m_key in args.models:
        train_one_model(
            MODEL_SPECS[m_key],
            data_dir=args.data_dir,
            out_dir=args.out_dir,
            folds_arg=args.folds,
            epochs_override=args.epochs,
            freeze_override=args.freeze_vit_blocks,
            num_workers=args.num_workers,
            resume=args.resume,
        )


if __name__ == "__main__":
    main()
