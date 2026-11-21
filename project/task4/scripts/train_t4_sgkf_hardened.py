"""5-Fold SGKF Corruption-Hardened Trainer for Task 4 (2:1 Aspect Ratio, Pure-Train).

Targets the four visual corruption regimes observed in Task 4:
1. Rotation [-35, +35] deg with black corner borders
2. Extreme overexposure (white-blown highlights) & ultra-dark underexposure + fog
3. Neon RGB channel gain tints, HSV hue shifts, and color inversion/solarization
4. Shot/Gaussian noise, blur, and low-quality JPEG artifacts

Usage:
    python train_t4_sgkf_hardened.py --data-dir /content/t4_data --out-dir /content/drive/MyDrive/t4_hard_outputs --models cnxv3 eva02l dinov3l
"""
from __future__ import annotations

import argparse
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


@dataclass(frozen=True)
class ModelSpec:
    """Configuration for a corruption-hardened Task 4 vision backbone."""

    key: str
    timm_name: str
    img_hw: Tuple[int, int]
    epochs: int
    base_lr: float
    backbone_lr_mult: float
    batch_a100: int
    batch_l4: int
    batch_t4: int
    grad_ckpt: bool
    use_img_size_arg: bool


MODEL_SPECS: Dict[str, ModelSpec] = {
    "cnxv3": ModelSpec(
        key="cnxv3",
        timm_name="convnext_base.dinov3_lvd1689m",
        img_hw=(192, 384),
        epochs=7,
        base_lr=3e-4,
        backbone_lr_mult=0.25,
        batch_a100=96,
        batch_l4=64,
        batch_t4=32,
        grad_ckpt=False,
        use_img_size_arg=False,
    ),
    "cnxv2b": ModelSpec(
        key="cnxv2b",
        timm_name="convnextv2_base.fcmae_ft_in22k_in1k",
        img_hw=(192, 384),
        epochs=7,
        base_lr=3e-4,
        backbone_lr_mult=0.25,
        batch_a100=96,
        batch_l4=64,
        batch_t4=32,
        grad_ckpt=False,
        use_img_size_arg=False,
    ),
    "eva02l": ModelSpec(
        key="eva02l",
        timm_name="eva02_large_patch14_448.mim_m38m_ft_in22k_in1k",
        img_hw=(196, 392),
        epochs=6,
        base_lr=1.8e-4,
        backbone_lr_mult=0.20,
        batch_a100=48,
        batch_l4=24,
        batch_t4=12,
        grad_ckpt=True,
        use_img_size_arg=True,
    ),
    "dinov3l": ModelSpec(
        key="dinov3l",
        timm_name="vit_large_patch16_dinov3.lvd1689m",
        img_hw=(192, 384),
        epochs=6,
        base_lr=1.8e-4,
        backbone_lr_mult=0.20,
        batch_a100=64,
        batch_l4=32,
        batch_t4=16,
        grad_ckpt=True,
        use_img_size_arg=True,
    ),
}


def seed_everything(seed: int) -> None:
    """Set deterministic RNG seeds across stdlib, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_jpeg_bytes(folder: str, filenames: List[str]) -> List[bytes]:
    """Preload compressed JPEG byte buffers into host RAM to avoid disk I/O stalls."""
    buffers: List[bytes] = []
    for fn in filenames:
        with open(os.path.join(folder, fn), "rb") as f:
            buffers.append(f.read())
    return buffers


def apply_nightmare_aug(pil_img: Image.Image, rng: np.random.Generator) -> np.ndarray:
    """Apply realistic hard corruptions matching Task 4's visual shift regimes."""
    if rng.random() < 0.50:
        pil_img = pil_img.transpose(Image.FLIP_LEFT_RIGHT)
    if rng.random() < 0.60:
        ang = float(rng.uniform(-35.0, 35.0))
        pil_img = pil_img.rotate(ang, resample=Image.BILINEAR, fillcolor=(0, 0, 0))
    if rng.random() < 0.18:
        pil_img = pil_img.filter(ImageFilter.GaussianBlur(radius=float(rng.uniform(0.4, 1.4))))

    arr = np.asarray(pil_img, dtype=np.float32) / 255.0

    # Neon channel gain / color tint (without destructive 1 - x color inversion)
    if rng.random() < 0.55:
        gains = rng.uniform(0.30, 2.20, size=(1, 1, 3)).astype(np.float32)
        gains /= float(gains.mean())
        arr = np.clip(arr * gains, 0.0, 1.0)

    # Exposure & contrast corruption
    r_exp = rng.random()
    if r_exp < 0.22:
        # Ultra-dark
        arr = np.power(np.clip(arr, 1e-4, 1.0), float(rng.uniform(1.8, 3.0))) * float(rng.uniform(0.32, 0.70))
    elif r_exp < 0.44:
        # White overexposed blown highlights
        arr = np.clip(arr * float(rng.uniform(1.25, 2.10)) + float(rng.uniform(0.24, 0.48)), 0.0, 1.0)
    elif r_exp < 0.58:
        # Foggy / washed-out contrast compression
        arr = np.clip(0.5 + (arr - 0.5) * float(rng.uniform(0.30, 0.65)) + float(rng.uniform(-0.15, 0.15)), 0.0, 1.0)

    if rng.random() < 0.45:
        arr = arr + rng.normal(0.0, float(rng.uniform(0.015, 0.080)), size=arr.shape).astype(np.float32)
    arr = np.clip(arr * 255.0, 0.0, 255.0).astype(np.uint8)

    if rng.random() < 0.25:
        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, format="JPEG", quality=int(rng.integers(25, 72)))
        buf.seek(0)
        arr = np.asarray(Image.open(buf).convert("RGB"), dtype=np.uint8)

    return arr


def interior_gw_stretch_np(arr_u8: np.ndarray) -> np.ndarray:
    """Normalize extreme color tints & exposures using interior 60% statistics (avoids rotated black corners)."""
    h, w, _ = arr_u8.shape
    h0, h1 = int(h * 0.2), int(h * 0.8)
    w0, w1 = int(w * 0.2), int(w * 0.8)
    b = arr_u8.astype(np.float32) / 255.0
    patch = b[h0:h1, w0:w1, :]
    ch_m = np.clip(patch.mean(axis=(0, 1), keepdims=True), 0.025, 1.0)
    b_gw = np.clip(b * (ch_m.mean(axis=2, keepdims=True) / ch_m), 0.0, 1.0)
    p_sub = b_gw[h0:h1:2, w0:w1:2, :].reshape(-1)
    lo = float(np.quantile(p_sub, 0.02))
    hi = float(np.quantile(p_sub, 0.98))
    b_st = np.clip((b_gw - lo) / max(hi - lo, 0.10), 0.0, 1.0)
    m_int = float(np.clip(b_st[h0:h1:2, w0:w1:2, :].mean(), 0.05, 0.95))
    gamma = float(np.clip(math.log(0.45) / math.log(m_int), 0.50, 1.80))
    return np.clip(np.power(b_st, gamma), 0.0, 1.0).astype(np.float32)


def make_deterministic_exthard(jpeg_bytes_list: List[bytes], indices: np.ndarray, seed: int = 123) -> List[np.ndarray]:
    """Generate deterministic ExtremeHard validation images for clean fold-level stress testing."""
    rng = np.random.default_rng(seed)
    out: List[np.ndarray] = []
    for k, idx in enumerate(indices):
        im = Image.open(io.BytesIO(jpeg_bytes_list[int(idx)])).convert("RGB")
        ang = float(rng.uniform(-35.0, 35.0))
        im = im.rotate(ang, resample=Image.BILINEAR, fillcolor=(0, 0, 0))
        arr = np.asarray(im, dtype=np.float32) / 255.0
        mode = k % 4
        if mode == 0:
            arr = np.power(np.clip(arr, 1e-4, 1.0), float(rng.uniform(1.8, 2.8))) * float(rng.uniform(0.35, 0.65))
        elif mode == 1:
            arr = np.clip(arr * float(rng.uniform(1.3, 2.0)) + float(rng.uniform(0.30, 0.50)), 0.0, 1.0)
        elif mode == 2:
            g = rng.uniform(0.35, 2.0, size=(1, 1, 3)).astype(np.float32)
            g /= float(g.max())
            arr = np.clip(np.power(arr, 0.85) * g * float(rng.uniform(0.8, 1.4)), 0.0, 1.0)
        else:
            arr = np.clip(0.5 + (arr - 0.5) * float(rng.uniform(0.4, 0.7)) + rng.uniform(-0.15, 0.15, (1, 1, 3)), 0.0, 1.0)
        arr = arr + rng.normal(0.0, float(rng.uniform(0.03, 0.07)), size=arr.shape).astype(np.float32)
        out.append(np.clip(arr * 255.0, 0.0, 255.0).astype(np.uint8))
    return out


class Task4TrainDataset(Dataset):
    """Training dataset supporting both raw/light and nightmare corruption augmentations + optional Interior-GW view."""

    def __init__(
        self,
        jpeg_bytes: List[bytes],
        indices: np.ndarray,
        labels: np.ndarray,
        img_hw: Tuple[int, int],
        p_nightmare: float = 0.65,
        p_norm_view: float = 0.20,
    ) -> None:
        self.jpeg_bytes = jpeg_bytes
        self.indices = indices
        self.labels = labels
        self.img_hw = img_hw
        self.p_nightmare = p_nightmare
        self.p_norm_view = p_norm_view

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, pos: int) -> Tuple[torch.Tensor, torch.Tensor]:
        idx = int(self.indices[pos])
        pil = Image.open(io.BytesIO(self.jpeg_bytes[idx])).convert("RGB")
        pil = pil.resize((self.img_hw[1], self.img_hw[0]), resample=Image.BILINEAR)

        rng = np.random.default_rng()
        if rng.random() < self.p_nightmare:
            arr_u8 = apply_nightmare_aug(pil, rng)
        else:
            if rng.random() < 0.50:
                pil = pil.transpose(Image.FLIP_LEFT_RIGHT)
            arr_u8 = np.asarray(pil, dtype=np.uint8)

        if rng.random() < self.p_norm_view:
            arr_f = interior_gw_stretch_np(arr_u8)
        else:
            arr_f = arr_u8.astype(np.float32) / 255.0

        ten = torch.from_numpy(arr_f).permute(2, 0, 1).contiguous()
        return ten, torch.tensor(int(self.labels[idx]), dtype=torch.long)


class Task4EvalViewDataset(Dataset):
    """Deterministic multi-view evaluation dataset (CPU resizing/rotation keeps GPU pipeline saturated)."""

    def __init__(
        self,
        source_items: List[bytes] | List[np.ndarray],
        indices: Optional[np.ndarray],
        img_hw: Tuple[int, int],
        view: str,
        is_bytes: bool = True,
    ) -> None:
        self.source_items = source_items
        self.indices = indices if indices is not None else np.arange(len(source_items))
        self.img_hw = img_hw
        self.view = view
        self.is_bytes = is_bytes

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, pos: int) -> torch.Tensor:
        item = self.source_items[int(self.indices[pos])]
        pil = Image.open(io.BytesIO(item)).convert("RGB") if self.is_bytes else Image.fromarray(item)
        pil = pil.resize((self.img_hw[1], self.img_hw[0]), resample=Image.BILINEAR)

        if self.view == "flip":
            pil = pil.transpose(Image.FLIP_LEFT_RIGHT)
        elif self.view == "rot_p18_gw":
            pil = pil.rotate(18.0, resample=Image.BILINEAR, fillcolor=(0, 0, 0))
        elif self.view == "rot_m18_gw":
            pil = pil.rotate(-18.0, resample=Image.BILINEAR, fillcolor=(0, 0, 0))
        elif self.view == "gw_flip":
            pil = pil.transpose(Image.FLIP_LEFT_RIGHT)

        arr_u8 = np.asarray(pil, dtype=np.uint8)
        if self.view in ("gw", "gw_flip", "rot_p18_gw", "rot_m18_gw"):
            arr_f = interior_gw_stretch_np(arr_u8)
        else:
            arr_f = arr_u8.astype(np.float32) / 255.0
        return torch.from_numpy(arr_f).permute(2, 0, 1).contiguous()


def build_model(spec: ModelSpec, pretrained: bool = True) -> Tuple[nn.Module, Tuple[float, ...], Tuple[float, ...]]:
    """Instantiate timm model with exact aspect-ratio img_size and resolved RGB mean/std."""
    kwargs = {"pretrained": pretrained, "num_classes": 5}
    if spec.use_img_size_arg:
        kwargs["img_size"] = spec.img_hw
    model = timm.create_model(spec.timm_name, **kwargs)
    if spec.grad_ckpt and hasattr(model, "set_grad_checkpointing"):
        model.set_grad_checkpointing(True)
    cfg = timm.data.resolve_model_data_config(model)
    return model, tuple(cfg["mean"]), tuple(cfg["std"])


def pick_batch_and_accum(spec: ModelSpec, device: torch.device, target_eff_bs: int = 64) -> Tuple[int, int]:
    """Select safe batch size and gradient accumulation steps for the detected GPU VRAM."""
    if device.type != "cuda":
        return 8, 1
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    if vram_gb >= 36.0:
        bs = spec.batch_a100
    elif vram_gb >= 21.0:
        bs = spec.batch_l4
    else:
        bs = spec.batch_t4
    accum = max(1, math.ceil(target_eff_bs / bs))
    return bs, accum


def make_param_groups(model: nn.Module, base_lr: float, bb_mult: float, wd: float = 2e-2) -> List[Dict]:
    """Separate classifier head vs backbone weights and exclude 1D norm/bias tensors from weight decay."""
    head_decay, head_no, bb_decay, bb_no = [], [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_head = name.startswith("head") or name.startswith("fc") or ".head." in name
        no_wd = p.ndim <= 1 or name.endswith(".bias") or "norm" in name.lower()
        if is_head:
            (head_no if no_wd else head_decay).append(p)
        else:
            (bb_no if no_wd else bb_decay).append(p)
    groups = []
    if head_decay:
        groups.append({"params": head_decay, "lr": base_lr, "weight_decay": wd})
    if head_no:
        groups.append({"params": head_no, "lr": base_lr, "weight_decay": 0.0})
    if bb_decay:
        groups.append({"params": bb_decay, "lr": base_lr * bb_mult, "weight_decay": wd})
    if bb_no:
        groups.append({"params": bb_no, "lr": base_lr * bb_mult, "weight_decay": 0.0})
    return groups


def update_ema(ema_state: Dict[str, torch.Tensor], model: nn.Module, decay: float = 0.997) -> None:
    """Update exponential moving average shadow weights on GPU/device."""
    with torch.no_grad():
        for k, v in model.state_dict().items():
            if k in ema_state and v.dtype.is_floating_point:
                ema_state[k].mul_(decay).add_(v.detach(), alpha=1.0 - decay)
            elif k not in ema_state:
                ema_state[k] = v.detach().clone()


@torch.no_grad()
def predict_views(
    model: nn.Module,
    source_items: List[bytes] | List[np.ndarray],
    indices: Optional[np.ndarray],
    spec: ModelSpec,
    mean_t: torch.Tensor,
    std_t: torch.Tensor,
    views: List[str],
    bs: int,
    device: torch.device,
    amp_dtype: Optional[torch.dtype],
    is_bytes: bool = True,
    num_workers: int = 4,
) -> Dict[str, np.ndarray]:
    """Run multi-view inference cleanly with GPU autocast."""
    model.eval()
    n = len(indices) if indices is not None else len(source_items)
    preds: Dict[str, np.ndarray] = {}
    for v in views:
        ds = Task4EvalViewDataset(source_items, indices, spec.img_hw, v, is_bytes=is_bytes)
        dl = DataLoader(ds, batch_size=bs * 2, shuffle=False, num_workers=num_workers, pin_memory=(device.type == "cuda"))
        chunks = []
        for xb in dl:
            xb = xb.to(device, non_blocking=True)
            xb = (xb - mean_t) / std_t
            if amp_dtype is not None:
                with torch.amp.autocast(device.type, dtype=amp_dtype):
                    logits = model(xb)
            else:
                logits = model(xb)
            chunks.append(F.softmax(logits.float(), dim=1).cpu().numpy())
        preds[v] = np.concatenate(chunks, axis=0) if chunks else np.zeros((0, 5), dtype=np.float32)
        assert len(preds[v]) == n
    return preds


def combine_tta_views(vdict: Dict[str, np.ndarray], mode: str) -> np.ndarray:
    """Combine view probabilities in geometric log-space."""
    if mode == "orig2":
        keys, ws = ["orig", "flip"], [0.55, 0.45]
    elif mode == "weighted6":
        keys = ["orig", "flip", "gw", "gw_flip", "rot_p18_gw", "rot_m18_gw"]
        ws = [0.28, 0.20, 0.20, 0.12, 0.10, 0.10]
    else:
        raise ValueError(f"Unknown combine mode: {mode}")
    lp = sum(w * np.log(np.clip(vdict[k], 1e-7, 1.0)) for k, w in zip(keys, ws))
    lp -= lp.max(axis=1, keepdims=True)
    ep = np.exp(lp)
    return (ep / ep.sum(axis=1, keepdims=True)).astype(np.float32)


def save_csv_preds(
    out_path: str,
    df_meta: pd.DataFrame,
    probs: np.ndarray,
    folds_col: Optional[np.ndarray] = None,
) -> None:
    """Write predictions matching project/task4/predictions/model-level CSV schema."""
    out = pd.DataFrame({"id": df_meta["id"].values, "file_name": df_meta["file_name"].values})
    if "label" in df_meta.columns and df_meta["label"].notna().all():
        out["label"] = df_meta["label"].astype(int).values
    if folds_col is not None:
        out["fold"] = folds_col.astype(int)
    for c in range(5):
        out[f"prob_{c}"] = np.round(probs[:, c].astype(np.float64), 7)
    out.to_csv(out_path, index=False)


def train_one_model(
    spec: ModelSpec,
    tr_bytes: List[bytes],
    te_bytes: List[bytes],
    y: np.ndarray,
    folds: np.ndarray,
    tr_df: pd.DataFrame,
    te_df: pd.DataFrame,
    out_dir: str,
    device: torch.device,
    pretrained: bool,
    epochs_override: Optional[int],
    max_folds: int,
    num_workers: int,
    resume: bool,
    seed: int = 42,
) -> None:
    """Train all SGKF folds for one backbone with automatic checkpoint resume and dual-stream TTA."""
    os.makedirs(out_dir, exist_ok=True)
    cache_dir = os.path.join(out_dir, f"folds_{spec.key}")
    os.makedirs(cache_dir, exist_ok=True)

    amp_dtype: Optional[torch.dtype] = None
    use_scaler = False
    if device.type == "cuda":
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        use_scaler = amp_dtype == torch.float16

    bs, accum = pick_batch_and_accum(spec, device)
    epochs = epochs_override if epochs_override is not None else spec.epochs
    eff_workers = min(num_workers, max(1, os.cpu_count() or num_workers)) if num_workers > 0 else 0
    views = ["orig", "flip", "gw", "gw_flip", "rot_p18_gw", "rot_m18_gw"]
    print(f"[{spec.key}] Config: img_hw={spec.img_hw} | bs={bs} (accum={accum}) | epochs={epochs} | workers={eff_workers}")

    n_tr, n_te = len(tr_df), len(te_df)
    oof_orig2 = np.zeros((n_tr, 5), dtype=np.float32)
    oof_w6 = np.zeros((n_tr, 5), dtype=np.float32)
    ext_w6 = np.zeros((n_tr, 5), dtype=np.float32)
    te_orig2 = np.zeros((n_te, 5), dtype=np.float32)
    te_w6 = np.zeros((n_te, 5), dtype=np.float32)
    folds_done = 0

    for k in range(min(5, max_folds)):
        fold_npz = os.path.join(cache_dir, f"fold{k}.npz")
        va_idx = np.where(folds == k)[0]
        tr_idx = np.where(folds != k)[0]

        if resume and os.path.isfile(fold_npz):
            z = np.load(fold_npz)
            oof_orig2[va_idx] = z["va_orig2"]
            oof_w6[va_idx] = z["va_w6"]
            ext_w6[va_idx] = z["ext_w6"]
            te_orig2 += z["te_orig2"]
            te_w6 += z["te_w6"]
            folds_done += 1
            f_val = f1_score(y[va_idx], oof_w6[va_idx].argmax(1), average="macro")
            f_ext = f1_score(y[va_idx], ext_w6[va_idx].argmax(1), average="macro")
            print(f"[{spec.key}] Fold {k} resumed from {fold_npz} | ValF1={f_val:.5f} | ExtHardF1={f_ext:.5f}")
            continue

        t0 = time.time()
        seed_everything(seed + k * 101)
        model, mean_tup, std_tup = build_model(spec, pretrained=pretrained)
        model = model.to(device)
        mean_t = torch.tensor(mean_tup, device=device, dtype=torch.float32).view(1, 3, 1, 1)
        std_t = torch.tensor(std_tup, device=device, dtype=torch.float32).view(1, 3, 1, 1)

        # Pure-train class prior for mild logit adjustment during training (tau=0.20)
        pi_tr = np.bincount(y[tr_idx], minlength=5) / max(1, len(tr_idx))
        logit_adj = torch.tensor(0.20 * np.log(np.clip(pi_tr, 1e-4, 1.0)), device=device, dtype=torch.float32)

        tr_ds = Task4TrainDataset(tr_bytes, tr_idx, y, spec.img_hw, p_nightmare=0.50, p_norm_view=0.25)
        tr_dl = DataLoader(
            tr_ds,
            batch_size=bs,
            shuffle=True,
            drop_last=len(tr_ds) >= bs,
            num_workers=eff_workers,
            pin_memory=(device.type == "cuda"),
            persistent_workers=(eff_workers > 0),
        )

        opt = torch.optim.AdamW(make_param_groups(model, spec.base_lr, spec.backbone_lr_mult))
        steps_per_ep = max(1, math.ceil(len(tr_dl) / accum))
        total_steps = steps_per_ep * epochs
        warmup_steps = max(1, int(0.08 * total_steps))

        def _lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return float(step + 1) / float(warmup_steps)
            prog = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * prog))

        sched = torch.optim.lr_scheduler.LambdaLR(opt, _lr_lambda)
        scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
        ema_state: Dict[str, torch.Tensor] = {}

        for ep in range(epochs):
            model.train()
            opt.zero_grad(set_to_none=True)
            ep_loss, n_seen = 0.0, 0
            for step_i, (xb, yb) in enumerate(tr_dl):
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                xb = (xb - mean_t) / std_t

                with torch.amp.autocast(device.type, dtype=amp_dtype) if amp_dtype is not None else torch.enable_grad():
                    logits = model(xb)
                    loss = F.cross_entropy(logits.float() + logit_adj, yb, label_smoothing=0.03)
                    loss_scaled = loss / accum

                if use_scaler:
                    scaler.scale(loss_scaled).backward()
                else:
                    loss_scaled.backward()

                is_step = ((step_i + 1) % accum == 0) or ((step_i + 1) == len(tr_dl))
                if is_step:
                    if use_scaler:
                        scaler.unscale_(opt)
                    nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                    if use_scaler:
                        scaler.step(opt)
                        scaler.update()
                    else:
                        opt.step()
                    opt.zero_grad(set_to_none=True)
                    sched.step()
                    update_ema(ema_state, model, decay=0.996 if ep >= 1 else 0.98)

                ep_loss += float(loss.item()) * len(yb)
                n_seen += len(yb)

            # Quick orig2 clean check at epoch end
            val_quick = predict_views(model, tr_bytes, va_idx, spec, mean_t, std_t, ["orig"], bs, device, amp_dtype, True, eff_workers)["orig"]
            vf1 = f1_score(y[va_idx], val_quick.argmax(1), average="macro")
            print(f"[{spec.key}] Fold {k} Ep {ep+1}/{epochs} loss={ep_loss/max(1,n_seen):.4f} ValCleanF1={vf1:.5f} ({time.time()-t0:.1f}s)")

        # Load EMA shadow weights for final OOF, ExtHard, and Test inference
        model.load_state_dict(ema_state, strict=False)
        va_views = predict_views(model, tr_bytes, va_idx, spec, mean_t, std_t, views, bs, device, amp_dtype, True, eff_workers)
        ext_imgs = make_deterministic_exthard(tr_bytes, va_idx, seed=123 + k)
        ext_views = predict_views(model, ext_imgs, None, spec, mean_t, std_t, views, bs, device, amp_dtype, False, eff_workers)
        te_views = predict_views(model, te_bytes, None, spec, mean_t, std_t, views, bs, device, amp_dtype, True, eff_workers)

        va_o2 = combine_tta_views(va_views, "orig2")
        va_w6 = combine_tta_views(va_views, "weighted6")
        ex_w6 = combine_tta_views(ext_views, "weighted6")
        t_o2 = combine_tta_views(te_views, "orig2")
        t_w6 = combine_tta_views(te_views, "weighted6")

        np.savez_compressed(fold_npz, va_orig2=va_o2, va_w6=va_w6, ext_w6=ex_w6, te_orig2=t_o2, te_w6=t_w6)
        oof_orig2[va_idx] = va_o2
        oof_w6[va_idx] = va_w6
        ext_w6[va_idx] = ex_w6
        te_orig2 += t_o2
        te_w6 += t_w6
        folds_done += 1

        f_val = f1_score(y[va_idx], va_w6.argmax(1), average="macro")
        f_ext = f1_score(y[va_idx], ex_w6.argmax(1), average="macro")
        print(f"[{spec.key}] DONE Fold {k} in {time.time()-t0:.1f}s | Val6TTA F1={f_val:.5f} | ExtHard F1={f_ext:.5f}")
        del model, opt, sched, scaler, ema_state, tr_dl, tr_ds
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if folds_done == 5:
        te_orig2 /= 5.0
        te_w6 /= 5.0
        f_all = f1_score(y, oof_w6.argmax(1), average="macro")
        f_o2 = f1_score(y, oof_orig2.argmax(1), average="macro")
        f_ext = f1_score(y, ext_w6.argmax(1), average="macro")
        ll_all = log_loss(y, np.clip(oof_w6, 1e-7, 1.0), labels=[0, 1, 2, 3, 4])
        print(f"\n=== [{spec.key}] 5-FOLD SGKF COMPLETE | OOF_W6 F1={f_all:.5f} LL={ll_all:.5f} | Orig2 F1={f_o2:.5f} | ExtHard F1={f_ext:.5f} ===\n")
        save_csv_preds(os.path.join(out_dir, f"task4-{spec.key}-hard6-oof.csv"), tr_df, oof_w6, folds)
        save_csv_preds(os.path.join(out_dir, f"task4-{spec.key}-hard6-test-predictions.csv"), te_df, te_w6)
        save_csv_preds(os.path.join(out_dir, f"task4-{spec.key}-orig2-oof.csv"), tr_df, oof_orig2, folds)
        save_csv_preds(os.path.join(out_dir, f"task4-{spec.key}-orig2-test-predictions.csv"), te_df, te_orig2)
        bundle_path = os.path.join(out_dir, f"task4-{spec.key}-sgkf-hard-bundle.npz")
        np.savez_compressed(
            bundle_path,
            folds=folds,
            oof_orig2=oof_orig2,
            oof_w6=oof_w6,
            ext_w6=ext_w6,
            te_orig2=te_orig2,
            te_w6=te_w6,
        )
        # Immediately package per-model download zip in case session disconnects during later backbones
        model_files = [
            f"task4-{spec.key}-hard6-oof.csv",
            f"task4-{spec.key}-hard6-test-predictions.csv",
            f"task4-{spec.key}-orig2-oof.csv",
            f"task4-{spec.key}-orig2-test-predictions.csv",
            f"task4-{spec.key}-sgkf-hard-bundle.npz",
        ]
        mzip_out = os.path.join(out_dir, f"t4_hard_{spec.key}.zip")
        with zipfile.ZipFile(mzip_out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for fn in model_files:
                fp = os.path.join(out_dir, fn)
                if os.path.isfile(fp):
                    zf.write(fp, fn)
        if os.path.isdir("/content") and os.path.abspath(mzip_out) != f"/content/t4_hard_{spec.key}.zip":
            shutil.copy2(mzip_out, f"/content/t4_hard_{spec.key}.zip")
        print(f"[READY] Packaged 5-fold {spec.key} bundle -> {mzip_out}", flush=True)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="5-Fold SGKF Hardened Task 4 Trainer")
    ap.add_argument("--data-dir", required=True, help="Directory with train.csv, test.csv, folds_sgkf.npy, train/, test/")
    ap.add_argument("--out-dir", required=True, help="Output directory for fold checkpoints and CSV bundles")
    ap.add_argument("--models", nargs="+", default=["cnxv3", "eva02l", "dinov3l"], choices=list(MODEL_SPECS.keys()))
    ap.add_argument("--epochs", type=int, default=None, help="Override default epoch count")
    ap.add_argument("--max-folds", type=int, default=5, help="Train up to N folds (5 for full run)")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--no-pretrained", action="store_true", help="Used for offline smoke testing")
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--cpu", action="store_true", help="Force CPU device (e.g., smoke tests)")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if args.cpu:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"Device: {device} | Models: {args.models}")

    tr_df = pd.read_csv(os.path.join(args.data_dir, "train.csv"))
    te_df = pd.read_csv(os.path.join(args.data_dir, "test.csv"))
    cmap = {"7-11k": 0, "fiveN": 1, "freefried": 2, "pube": 3, "roblock": 4}
    y = tr_df["label"].astype(int).to_numpy() if "label" in tr_df.columns else tr_df["class"].map(cmap).astype(int).to_numpy()
    folds = np.load(os.path.join(args.data_dir, "folds_sgkf.npy")).astype(int)
    assert len(tr_df) == len(folds) == len(y), "Manifest and folds_sgkf length mismatch"

    print("Preloading compressed JPEG bytes into RAM...")
    t_load = time.time()
    tr_bytes = load_jpeg_bytes(os.path.join(args.data_dir, "train"), tr_df["file_name"].tolist())
    te_bytes = load_jpeg_bytes(os.path.join(args.data_dir, "test"), te_df["file_name"].tolist())
    print(f"Preloaded train={len(tr_bytes):,}, test={len(te_bytes):,} images in {time.time()-t_load:.1f}s")

    for mkey in args.models:
        train_one_model(
            spec=MODEL_SPECS[mkey],
            tr_bytes=tr_bytes,
            te_bytes=te_bytes,
            y=y,
            folds=folds,
            tr_df=tr_df,
            te_df=te_df,
            out_dir=args.out_dir,
            device=device,
            pretrained=not args.no_pretrained,
            epochs_override=args.epochs,
            max_folds=args.max_folds,
            num_workers=args.num_workers,
            resume=not args.no_resume,
        )


if __name__ == "__main__":
    main()
