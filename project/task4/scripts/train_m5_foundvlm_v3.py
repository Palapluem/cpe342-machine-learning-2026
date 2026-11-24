#!/usr/bin/env python3
"""Fast Cholesky/LU Multi-Lambda Ridge + Single-Backbone MPS Streaming FoundVLM-v3 (~90s):
- Precomputes L2-normalized views once per backbone, streams one backbone at a time.
- Multi-lambda Dual-Condition Ridge (lam in {15, 45, 120}) via fast np.linalg.solve (~0.3s/fold),
  using 2-fold inner CV subtraction inside tr_m for exact out-of-fold 90d Meta-features (zero eigh/qr).
- Trains single-backbone GPU-resident MPS Dual-Condition CosineResMLP + SymKL (~103 MB peak VRAM)
  for bi in {0: siglip2_so400m, 1: dinov3_vitl, 5: dinov3_cnxb} with immediate MPS cleanup per fold.
- Stage 2 trains Joint 6-Backbone Meta-ResMLP (90d), VLM-3Fam (45d), DINO-3Fam (45d) on MPS (<5s total),
  anchors onto FoundVLM-v2 strictly inside folds != k, runs 4-way shadow ablation, & installs winner.
"""
from __future__ import annotations

import os

os.environ["OMP_NUM_THREADS"] = "8"
os.environ["VECLIB_MAXIMUM_THREADS"] = "8"
os.environ["OPENBLAS_NUM_THREADS"] = "8"
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse
import gc
import json
import math
import shutil
import sys
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import minimize
from sklearn.metrics import f1_score, log_loss
from sklearn.model_selection import StratifiedKFold

SEED = 42
LAMBDAS = (15.0, 45.0, 120.0)
SCRATCH_DIR = Path(__file__).resolve().parent
M5_OUT_DIR = SCRATCH_DIR / "t4_m5_outputs"
CPE_T4_DIR = Path("/Users/wisitsuwannao/Documents/Workspace/cpe342-machine-learning-2026/project/task4")
ML_T4_DIR = Path("/Users/wisitsuwannao/Documents/Workspace/ml00x/task4")

CLASS_MAP: Dict[str, int] = {
    "7-11k": 0,
    "7-11K": 0,
    "fiveN": 1,
    "freefried": 2,
    "pube": 3,
    "roblock": 4,
}

BACKBONE_FILES: List[Tuple[str, Path]] = [
    ("siglip2_so400m", CPE_T4_DIR / "emb_v4_siglip2_so400m.npz"),
    ("dinov3_vitl", CPE_T4_DIR / "emb_v4_dinov3_vitl.npz"),
    ("dinov2_vitg14", CPE_T4_DIR / "emb_v4_dinov2_vitg14.npz"),
    ("clip_cnxl", M5_OUT_DIR / "emb_v4_clip_cnxl.npz"),
    ("clip_vith14", M5_OUT_DIR / "emb_v4_clip_vith14.npz"),
    ("dinov3_cnxb", M5_OUT_DIR / "emb_v4_dinov3_cnxb.npz"),
]


def _l2_norm_np(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(n, 1e-8, None)


def _power_cl2n_from_normed(x_normed: np.ndarray, mu: np.ndarray, beta: float = 0.75) -> np.ndarray:
    z = _l2_norm_np(x_normed - mu)
    if beta != 1.0:
        z = _l2_norm_np(np.sign(z) * (np.abs(z) ** beta))
    return z


def _softmax_np(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def unif_to_train_prior(p_unif: np.ndarray, pi_vec: np.ndarray) -> np.ndarray:
    p = np.clip(p_unif, 1e-7, 1.0) * pi_vec[None, :]
    return p / p.sum(axis=1, keepdims=True)


def train_to_unif_prior(p_train: np.ndarray, pi_vec: np.ndarray) -> np.ndarray:
    p = np.clip(p_train, 1e-7, 1.0) / np.clip(pi_vec[None, :], 1e-7, 1.0)
    return p / p.sum(axis=1, keepdims=True)


def geo_pool(ps: Sequence[np.ndarray], ws: Sequence[float]) -> np.ndarray:
    lp = sum(float(wi) * np.log(np.clip(pi, 1e-7, 1.0)) for wi, pi in zip(ws, ps) if float(wi) > 0.0)
    return _softmax_np(lp)


def fit_slsqp_simplex(p_list: Sequence[np.ndarray], y_lbl: np.ndarray, max_w: float = 0.85) -> np.ndarray:
    m = len(p_list)
    lps = [np.log(np.clip(p, 1e-7, 1.0)) for p in p_list]
    idx = np.arange(len(y_lbl))

    def obj(w: np.ndarray) -> float:
        w = np.maximum(w, 0.0)
        w /= w.sum() + 1e-12
        pr = _softmax_np(sum(wi * lp_i for wi, lp_i in zip(w, lps)))
        return float(-np.mean(np.log(np.clip(pr[idx, y_lbl], 1e-12, 1.0))))

    res = minimize(
        obj,
        np.ones(m) / m,
        bounds=[(0.0, float(max_w))] * m,
        constraints={"type": "eq", "fun": lambda w: float(np.sum(w)) - 1.0},
        method="SLSQP",
    )
    w = np.maximum(res.x, 0.0)
    return w / w.sum()


def fit_slsqp_hybrid(
    p_clean_list: Sequence[np.ndarray],
    p_ext_list: Sequence[np.ndarray],
    y_lbl: np.ndarray,
    w_clean_loss: float = 0.75,
    max_w: float = 0.85,
) -> np.ndarray:
    m = len(p_clean_list)
    lps_c = [np.log(np.clip(p, 1e-7, 1.0)) for p in p_clean_list]
    lps_e = [np.log(np.clip(p, 1e-7, 1.0)) for p in p_ext_list]
    idx = np.arange(len(y_lbl))

    def _nll(lps: Sequence[np.ndarray], w: np.ndarray) -> float:
        pr = _softmax_np(sum(wi * lp_i for wi, lp_i in zip(w, lps)))
        return float(-np.mean(np.log(np.clip(pr[idx, y_lbl], 1e-12, 1.0))))

    def obj(w: np.ndarray) -> float:
        w = np.maximum(w, 0.0)
        w /= w.sum() + 1e-12
        return w_clean_loss * _nll(lps_c, w) + (1.0 - w_clean_loss) * _nll(lps_e, w)

    res = minimize(
        obj,
        np.ones(m) / m,
        bounds=[(0.0, float(max_w))] * m,
        constraints={"type": "eq", "fun": lambda w: float(np.sum(w)) - 1.0},
        method="SLSQP",
    )
    w = np.maximum(res.x, 0.0)
    return w / w.sum()


class CosineResMLP(nn.Module):
    def __init__(self, in_dim: int, block_slices: Sequence[Tuple[int, int]], hidden: int = 384, drop: float = 0.45, block_drop: float = 0.15):
        super().__init__()
        self.block_slices = tuple(block_slices)
        self.block_drop = float(block_drop)
        self.fc1 = nn.Linear(in_dim, hidden)
        self.ln1 = nn.LayerNorm(hidden)
        self.drop = nn.Dropout(drop)
        self.res_fc = nn.Linear(hidden, hidden)
        self.ln2 = nn.LayerNorm(hidden)
        self.w = nn.Parameter(torch.randn(5, hidden) * 0.02)
        self.scale = nn.Parameter(torch.tensor(16.0))

    def _apply_block_drop(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or len(self.block_slices) <= 1 or self.block_drop <= 0.0:
            return x
        bsz = x.shape[0]
        n_blk = len(self.block_slices)
        keep = (torch.rand((bsz, n_blk), device=x.device) >= self.block_drop).float()
        none_kept = keep.sum(dim=1, keepdim=True) == 0
        if none_kept.any():
            rand_blk = F.one_hot(torch.randint(0, n_blk, (bsz,), device=x.device), num_classes=n_blk).float()
            keep = torch.where(none_kept, rand_blk, keep)
        scale = float(n_blk) / keep.sum(dim=1, keepdim=True)
        masks = [keep[:, bi : bi + 1].expand(-1, e - s) for bi, (s, e) in enumerate(self.block_slices)]
        return x * (torch.cat(masks, dim=1) * scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._apply_block_drop(x)
        h = F.gelu(self.ln1(self.fc1(x)))
        h = h + self.drop(F.gelu(self.ln2(self.res_fc(self.drop(h)))))
        z = F.normalize(h, p=2, dim=-1)
        w = F.normalize(self.w, p=2, dim=-1)
        return self.scale.clamp(4.0, 40.0) * (z @ w.T)


def train_dual_mlp_mps_single(
    X_c_np: np.ndarray,
    X_e_np: np.ndarray,
    y_tr: np.ndarray,
    eval_list: Sequence[np.ndarray],
    block_slices: Sequence[Tuple[int, int]],
    hidden: int,
    drop: float,
    block_drop: float,
    seeds: Sequence[int],
    epochs: int,
    device: torch.device,
    lr: float = 1.5e-3,
    weight_decay: float = 2.5e-2,
    batch_size: int = 512,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Single-backbone MPS-resident Dual-Condition CosineResMLP + SymKL (~103 MB peak VRAM)."""
    pi_k = np.bincount(y_tr, minlength=5).astype(np.float64) / len(y_tr)
    log_pi = torch.tensor(np.log(np.clip(pi_k, 1e-4, 1.0)), device=device, dtype=torch.float32)

    Xc_gpu = torch.from_numpy(np.ascontiguousarray(X_c_np.astype(np.float16)))
    Xe_gpu = torch.from_numpy(np.ascontiguousarray(X_e_np.astype(np.float16)))
    Xc_gpu = Xc_gpu.to(device)
    Xe_gpu = Xe_gpu.to(device)
    y_gpu = torch.from_numpy(y_tr.astype(np.int64)).to(device)

    n = len(y_tr)
    in_dim = X_c_np.shape[1]
    steps_per_ep = math.ceil(n / batch_size)
    p_va_c = np.zeros((len(eval_list[0]), 5), dtype=np.float64)
    p_va_e = np.zeros((len(eval_list[1]), 5), dtype=np.float64)
    p_te_c = np.zeros((len(eval_list[2]), 5), dtype=np.float64)
    t_kl = 1.5

    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = CosineResMLP(in_dim, block_slices, hidden=hidden, drop=drop, block_drop=block_drop).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs * steps_per_ep), eta_min=1e-5)

        for _ep in range(epochs):
            model.train()
            perm = torch.randperm(n, device=device)
            for s in range(0, n, batch_size):
                idx = perm[s : s + batch_size]
                xc = Xc_gpu[idx].float() + torch.randn((len(idx), in_dim), device=device) * 0.020
                xe = Xe_gpu[idx].float() + torch.randn((len(idx), in_dim), device=device) * 0.020
                yb_oh = F.one_hot(y_gpu[idx], num_classes=5).float() * 0.98 + (0.02 / 5.0)
                if np.random.random() < 0.50 and len(xc) > 1:
                    lam = float(np.random.beta(0.40, 0.40))
                    mp = torch.randperm(len(xc), device=device)
                    xc = lam * xc + (1.0 - lam) * xc[mp]
                    xe = lam * xe + (1.0 - lam) * xe[mp]
                    yb_oh = lam * yb_oh + (1.0 - lam) * yb_oh[mp]

                opt.zero_grad(set_to_none=True)
                post_c = model(xc) + log_pi.unsqueeze(0)
                post_e = model(xe) + log_pi.unsqueeze(0)
                ce_c = -(yb_oh * F.log_softmax(post_c, dim=1)).sum(dim=1).mean()
                ce_e = -(yb_oh * F.log_softmax(post_e, dim=1)).sum(dim=1).mean()

                pc = F.softmax(post_c / t_kl, dim=1)
                pe = F.softmax(post_e / t_kl, dim=1)
                lpc = F.log_softmax(post_c / t_kl, dim=1)
                lpe = F.log_softmax(post_e / t_kl, dim=1)
                sym_kl = 0.5 * ((pc * (lpc - lpe)).sum(dim=1).mean() + (pe * (lpe - lpc)).sum(dim=1).mean())

                loss = ce_c + 0.45 * ce_e + 0.25 * sym_kl
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                opt.step()
                sched.step()

        model.eval()
        outs = []
        with torch.no_grad():
            for Xe_np in eval_list:
                chunks = []
                for s in range(0, len(Xe_np), 2048):
                    xt = torch.from_numpy(np.ascontiguousarray(Xe_np[s : s + 2048].astype(np.float32, copy=False))).to(device)
                    chunks.append(F.softmax(model(xt), dim=1).cpu().numpy())
                outs.append(np.concatenate(chunks, axis=0))
        p_va_c += outs[0] / len(seeds)
        p_va_e += outs[1] / len(seeds)
        p_te_c += outs[2] / len(seeds)
        del model, opt, sched

    del Xc_gpu, Xe_gpu, y_gpu, log_pi
    if device.type == "mps":
        torch.mps.synchronize()
        torch.mps.empty_cache()
    return unif_to_train_prior(p_va_c, pi_k), unif_to_train_prior(p_va_e, pi_k), unif_to_train_prior(p_te_c, pi_k)


def fit_fast_dual_ridge(
    X_c: np.ndarray,
    X_e: np.ndarray,
    y_trk: np.ndarray,
    pi_k: np.ndarray,
    X_vc: np.ndarray,
    X_ve: np.ndarray,
    X_tc: np.ndarray,
    fold_seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fast np.linalg.solve multi-lambda ridge + 2-fold inner CV OOF meta-logits (zero eigh/qr)."""
    n_k, p_dim = X_c.shape
    Y_oh = np.zeros((n_k, 5), dtype=np.float32)
    Y_oh[np.arange(n_k), y_trk] = (1.0 / (5.0 * pi_k[y_trk])).astype(np.float32)

    G_full = (X_c.T @ X_c) + np.float32(0.45) * (X_e.T @ X_e)
    R_full = (X_c.T @ Y_oh) + np.float32(0.45) * (X_e.T @ Y_oh)
    eye = np.eye(p_dim, dtype=np.float32)

    def _solve_blocks(G_mat: np.ndarray, R_mat: np.ndarray) -> List[np.ndarray]:
        return [np.linalg.solve(G_mat + np.float32(lam) * eye, R_mat) for lam in LAMBDAS]

    Ws_full = _solve_blocks(G_full, R_full)

    def _eval_logits(X: np.ndarray, Ws: Sequence[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        parts = [X @ W for W in Ws]
        meta15 = np.concatenate(parts, axis=1).astype(np.float32)
        combo = (0.25 * parts[0] + 0.50 * parts[1] + 0.25 * parts[2]).astype(np.float64)
        return meta15, combo

    z_vc, l_vc = _eval_logits(X_vc, Ws_full)
    z_ve, l_ve = _eval_logits(X_ve, Ws_full)
    z_tc, l_tc = _eval_logits(X_tc, Ws_full)

    # Exact 2-fold inner OOF subtraction so Z_tr_c / Z_tr_e have zero in-sample ridge shrinkage gap
    skf2 = StratifiedKFold(n_splits=2, shuffle=True, random_state=fold_seed)
    z_tr_c = np.zeros((n_k, 15), dtype=np.float32)
    z_tr_e = np.zeros((n_k, 15), dtype=np.float32)
    l_tr_oof = np.zeros((n_k, 5), dtype=np.float64)
    for _, in_va in skf2.split(np.zeros(n_k), y_trk):
        Xc_v, Xe_v, Yv = X_c[in_va], X_e[in_va], Y_oh[in_va]
        G_sub = G_full - ((Xc_v.T @ Xc_v) + np.float32(0.45) * (Xe_v.T @ Xe_v))
        R_sub = R_full - ((Xc_v.T @ Yv) + np.float32(0.45) * (Xe_v.T @ Yv))
        Ws_in = _solve_blocks(G_sub, R_sub)
        z_tr_c[in_va], l_tr_oof[in_va] = _eval_logits(Xc_v, Ws_in)
        z_tr_e[in_va], _ = _eval_logits(Xe_v, Ws_in)

    log_pi = np.log(np.clip(pi_k, 1e-6, 1.0))[None, :]
    sw = 1.0 / (5.0 * pi_k[y_trk])
    sw /= sw.mean()
    idx = np.arange(n_k)

    def _oof_nll(t_arr: np.ndarray) -> float:
        pr = _softmax_np(l_tr_oof * float(t_arr[0]) + log_pi)
        return float(-np.mean(sw * np.log(np.clip(pr[idx, y_trk], 1e-12, 1.0))))

    inv_t = float(minimize(_oof_nll, [6.0], bounds=[(0.5, 25.0)], method="L-BFGS-B").x[0])
    p_vc = _softmax_np(l_vc * inv_t + log_pi)
    p_ve = _softmax_np(l_ve * inv_t + log_pi)
    p_tc = _softmax_np(l_tc * inv_t + log_pi)
    return p_vc, p_ve, p_tc, z_tr_c, z_tr_e, z_vc, z_ve, z_tc


def save_csv_preds(out_path: Path, df_meta: pd.DataFrame, probs: np.ndarray, folds_col=None) -> None:
    fnames = df_meta["file_name"] if "file_name" in df_meta.columns else (df_meta["id"].astype(str) + ".jpg")
    out = pd.DataFrame({"id": df_meta["id"].values, "file_name": fnames.values})
    if "label" in df_meta.columns and df_meta["label"].notna().all():
        out["label"] = df_meta["label"].astype(int).values
    if folds_col is not None:
        out["fold"] = folds_col.astype(int)
    for c in range(5):
        out[f"prob_{c}"] = np.round(probs[:, c].astype(np.float64), 7)
    out.to_csv(out_path, index=False)


def atomic_savez_compressed(path: Path, **arrays) -> None:
    tmp_path = path.with_suffix(f".tmp.{os.getpid()}.npz")
    np.savez_compressed(tmp_path, **arrays)
    os.replace(tmp_path, path)


def install_bundle_set(
    prefix: str,
    src_npz: Path,
    tr_df: pd.DataFrame,
    te_df: pd.DataFrame,
    folds: np.ndarray,
    zip_name: Optional[str] = None,
) -> None:
    with np.load(src_npz) as z:
        oof_w6 = z["oof_w6"].astype(np.float32)
        te_w6 = z["te_w6"].astype(np.float32)
        oof_o2 = z["oof_orig2"].astype(np.float32)
        te_o2 = z["te_orig2"].astype(np.float32)

    names = [
        f"task4-{prefix}-hard6-oof.csv",
        f"task4-{prefix}-hard6-test-predictions.csv",
        f"task4-{prefix}-orig2-oof.csv",
        f"task4-{prefix}-orig2-test-predictions.csv",
        f"task4-{prefix}-sgkf-hard-bundle.npz",
    ]
    save_csv_preds(M5_OUT_DIR / names[0], tr_df, oof_w6, folds)
    save_csv_preds(M5_OUT_DIR / names[1], te_df, te_w6)
    save_csv_preds(M5_OUT_DIR / names[2], tr_df, oof_o2, folds)
    save_csv_preds(M5_OUT_DIR / names[3], te_df, te_o2)
    shutil.copy2(src_npz, M5_OUT_DIR / names[4])

    for tgt_dir in (CPE_T4_DIR / "predictions/model-level", ML_T4_DIR / "predictions/model-level"):
        tgt_dir.mkdir(parents=True, exist_ok=True)
        for fn in names:
            shutil.copy2(M5_OUT_DIR / fn, tgt_dir / fn)

    if zip_name:
        zip_dir = CPE_T4_DIR / "t4_hard_bundles"
        zip_dir.mkdir(parents=True, exist_ok=True)
        zpath = zip_dir / zip_name
        with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for fn in names:
                zf.write(M5_OUT_DIR / fn, fn)
        print(f"Packaged -> {zpath}", flush=True)


def run_shadow_ablation(
    bak_v3: Path,
    sig_npz: Path,
    din_npz: Path,
) -> Tuple[str, Tuple[str, ...], Dict[str, Path], Dict[str, float]]:
    """Run 4-way outer fusion ablation vs active Run 27 (0.86680) and return winner."""
    sys.path.insert(0, str(SCRATCH_DIR))
    sys.path.insert(0, str(ML_T4_DIR / "scripts"))
    import install_and_fuse_clean_siglip as ics  # noqa: E402

    ml_orig = ML_T4_DIR / "predictions/model-level"
    tr_df = pd.read_csv(ML_T4_DIR / "data/train.csv")
    y = (
        tr_df["label"].astype(int).to_numpy()
        if "label" in tr_df.columns
        else tr_df["class"].map(CLASS_MAP).astype(int).to_numpy()
    )
    active_sub = pd.read_csv(ML_T4_DIR / "submit/index.csv")["task4"].to_numpy()
    b_sum, b_oof, b_ext, b_pred, corr_mask, pi_tr = ics.compute_fusion_arrays(ML_T4_DIR, exclude_keys=("siglip_l16",))
    assert np.array_equal(b_pred, active_sub), "Active Run 27 replay mismatch!"
    sw_all = 1.0 / pi_tr[y]
    idx_corr = np.where(corr_mask)[0]
    n = len(y)

    bm = b_sum["metrics"]
    b_comp = 0.40 * bm["clean_unif_oof_f1"] + 0.40 * bm["exthard6_unif_oof_f1"] + 0.20 * bm["corrupt_oof_f1"] - 0.05 * bm["oof_logloss"]
    print("\n=== 4-Candidate Shadow Ablation vs Active Run 27 (0.86680) ===", flush=True)
    print(f"Baseline Run 27: comp={b_comp:.6f} | metrics={json.dumps(bm)}", flush=True)

    candidates = [
        ("Cand1: FoundVLM-v3 (excl siglip_l16)", ("siglip_l16",), {}),
        ("Cand2: FoundVLM-v3 (keep siglip_l16)", (), {}),
        ("Cand3: FoundVLM-v3 + clean_siglip_m5 (excl siglip_l16)", ("siglip_l16",), {"clean_siglip": sig_npz}),
        (
            "Cand4: FoundVLM-v3 + clean_siglip_m5 + clean_dinov3l_m5 (excl siglip_l16)",
            ("siglip_l16",),
            {"clean_siglip": sig_npz, "clean_dinov3l": din_npz},
        ),
    ]

    best_tuple = None
    best_comp = -1e9
    for label, ex_keys, extra_map in candidates:
        sh_dir = Path(tempfile.mkdtemp(prefix="fv3_ablate_"))
        try:
            os.symlink(ML_T4_DIR / "data", sh_dir / "data")
            os.symlink(ML_T4_DIR / "models", sh_dir / "models")
            ml_sh = sh_dir / "predictions/model-level"
            ml_sh.mkdir(parents=True, exist_ok=True)
            for src in ml_orig.iterdir():
                os.symlink(src, ml_sh / src.name)
            dst_fv = ml_sh / "task4-foundvlm-sgkf-hard-bundle.npz"
            if dst_fv.exists() or dst_fv.is_symlink():
                dst_fv.unlink()
            os.symlink(bak_v3.resolve(), dst_fv)
            for prefix, bpath in extra_map.items():
                dst_ex = ml_sh / f"task4-{prefix}-sgkf-hard-bundle.npz"
                if dst_ex.exists() or dst_ex.is_symlink():
                    dst_ex.unlink()
                os.symlink(bpath.resolve(), dst_ex)

            s_new, oof_new, ext_new, pred_new, _, _ = ics.compute_fusion_arrays(sh_dir, exclude_keys=ex_keys)
            rng = np.random.default_rng(42)
            w_cln, w_cor, w_ext = 0, 0, 0
            p0c, p1c = b_oof.argmax(1), oof_new.argmax(1)
            p0e, p1e = b_ext.argmax(1), ext_new.argmax(1)
            for _ in range(500):
                idx = rng.integers(0, n, size=n)
                if f1_score(y[idx], p1c[idx], average="macro", sample_weight=sw_all[idx]) > f1_score(
                    y[idx], p0c[idx], average="macro", sample_weight=sw_all[idx]
                ):
                    w_cln += 1
                if f1_score(y[idx], p1e[idx], average="macro", sample_weight=sw_all[idx]) > f1_score(
                    y[idx], p0e[idx], average="macro", sample_weight=sw_all[idx]
                ):
                    w_ext += 1
                ic = rng.choice(idx_corr, size=len(idx_corr), replace=True)
                if f1_score(y[ic], p1c[ic], average="macro") > f1_score(y[ic], p0c[ic], average="macro"):
                    w_cor += 1

            nm = s_new["metrics"]
            comp = 0.40 * nm["clean_unif_oof_f1"] + 0.40 * nm["exthard6_unif_oof_f1"] + 0.20 * nm["corrupt_oof_f1"] - 0.05 * nm["oof_logloss"]
            d_rows = int((pred_new != active_sub).sum())
            print(f"\n--- {label} (comp={comp:.6f}, vsR27={comp-b_comp:+.6f}) ---", flush=True)
            print(f"  Discovered models : {s_new['discovered_fullres_models']}", flush=True)
            print(f"  Clean pool weights: {s_new['clean_pool_weights']}", flush=True)
            print(f"  ExtHard weights   : {s_new['exthard_weights']}", flush=True)
            print(f"  Shift gate        : {s_new['shift_gate']}", flush=True)
            print(f"  Metrics           : {nm}", flush=True)
            print(
                f"  Delta vs Run 27   : CleanUnifF1={nm['clean_unif_oof_f1'] - bm['clean_unif_oof_f1']:+.6f} "
                f"| CorrF1={nm['corrupt_oof_f1'] - bm['corrupt_oof_f1']:+.6f} "
                f"| ExtUnifF1={nm['exthard6_unif_oof_f1'] - bm['exthard6_unif_oof_f1']:+.6f} "
                f"| OOF_LL={nm['oof_logloss'] - bm['oof_logloss']:+.6f}",
                flush=True,
            )
            print(
                f"  500-Boot vs Run27 : CleanUnif={w_cln/500:.3f} | Corrupt={w_cor/500:.3f} | ExtHard6={w_ext/500:.3f} | Δrows={d_rows:,}",
                flush=True,
            )
            if comp > best_comp:
                best_comp = comp
                best_tuple = (label, ex_keys, extra_map, {"comp": comp, "base_comp": b_comp, **nm})
        finally:
            shutil.rmtree(sh_dir, ignore_errors=True)

    assert best_tuple is not None
    print(f"\n>>> WINNER: {best_tuple[0]} (comp delta={best_comp - b_comp:+.6f})", flush=True)
    return best_tuple


def main() -> None:
    ap = argparse.ArgumentParser(description="Fast Cholesky + Single-Backbone MPS Streaming FoundVLM-v3")
    ap.add_argument("--install", action="store_true", help="Install FoundVLM-v3 into both repos")
    ap.add_argument("--ablate-fusion", action="store_true", help="Run 4-candidate shadow ablation vs active Run 27 (0.86680)")
    ap.add_argument("--auto-install-best", action="store_true", help="Auto-install winning ablation config & update both repos")
    args = ap.parse_args()

    t0 = time.time()
    M5_OUT_DIR.mkdir(parents=True, exist_ok=True)
    v2_bak_path = M5_OUT_DIR / "task4-foundvlm-v2-backup.npz"
    assert v2_bak_path.is_file(), f"Missing v2 backup: {v2_bak_path}"

    tr_df = pd.read_csv(CPE_T4_DIR / "data/train.csv")
    te_df = pd.read_csv(CPE_T4_DIR / "data/test.csv")
    if "file_name" not in tr_df.columns:
        tr_df["file_name"] = tr_df["id"].astype(str) + ".jpg"
    if "file_name" not in te_df.columns:
        te_df["file_name"] = te_df["id"].astype(str) + ".jpg"
    n_tr, n_te = len(tr_df), len(te_df)
    assert n_tr == 31546 and n_te == 25889
    assert te_df["id"].iloc[0] == "ANS00001" and te_df["file_name"].iloc[0] == "sjqyfwc.jpg"

    y = (
        tr_df["label"].astype(int).to_numpy()
        if "label" in tr_df.columns
        else tr_df["class"].map(CLASS_MAP).astype(int).to_numpy()
    )
    folds = np.load(ML_T4_DIR / "predictions/model-level/task4-sgkf-e2e-bundle.npz")["folds"].astype(int)
    pi_all = np.bincount(y, minlength=5) / n_tr
    sw_all = 1.0 / pi_all[y]

    with np.load(v2_bak_path) as z_v2:
        assert np.array_equal(folds, z_v2["folds"].astype(int)), "Folds mismatch vs v2 backup"
        v2_oof_o2 = z_v2["oof_orig2"].astype(np.float64)
        v2_te_o2 = z_v2["te_orig2"].astype(np.float64)
        v2_oof_w6 = z_v2["oof_w6"].astype(np.float64)
        v2_ext_w6 = z_v2["ext_w6"].astype(np.float64)
        v2_te_w6 = z_v2["te_w6"].astype(np.float64)

    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    n_bb = len(BACKBONE_FILES)
    meta_dim = n_bb * 15
    fold_masks = [(folds != k, folds == k) for k in range(5)]

    Z_tr_c = [np.zeros((int(tr_m.sum()), meta_dim), dtype=np.float32) for tr_m, _ in fold_masks]
    Z_tr_e = [np.zeros((int(tr_m.sum()), meta_dim), dtype=np.float32) for tr_m, _ in fold_masks]
    Z_va_c = [np.zeros((int(va_m.sum()), meta_dim), dtype=np.float32) for _, va_m in fold_masks]
    Z_va_e = [np.zeros((int(va_m.sum()), meta_dim), dtype=np.float32) for _, va_m in fold_masks]
    Z_te_c = [np.zeros((n_te, meta_dim), dtype=np.float32) for _ in range(5)]

    pb_oof_c = [np.zeros((n_tr, 5), dtype=np.float64) for _ in range(n_bb)]
    pb_ext_c = [np.zeros((n_tr, 5), dtype=np.float64) for _ in range(n_bb)]
    pb_te_c = [np.zeros((n_te, 5), dtype=np.float64) for _ in range(n_bb)]

    mlp_specs_map = {
        0: (np.zeros((n_tr, 5)), np.zeros((n_tr, 5)), np.zeros((n_te, 5)), (404, 505)),
        1: (np.zeros((n_tr, 5)), np.zeros((n_tr, 5)), np.zeros((n_te, 5)), (606, 707)),
        5: (np.zeros((n_tr, 5)), np.zeros((n_tr, 5)), np.zeros((n_te, 5)), (101, 202)),
    }

    inv_s2 = np.float32(1.0 / math.sqrt(2.0))
    print(f"[FoundVLM-v3] Stage 1: Streaming {n_bb} backbones -> fast Multi-Lambda Solve + Single-BB MPS ({device})", flush=True)

    for bi, (bkey, fpath) in enumerate(BACKBONE_FILES):
        tb0 = time.time()
        assert fpath.is_file(), f"Missing embedding bundle: {fpath}"
        with np.load(fpath) as z:
            r2_tr_n = _l2_norm_np(z["tr_raw2"])
            a4_tr_n = _l2_norm_np(z["tr_all4"])
            r2_ex_n = _l2_norm_np(z["ext_raw2"])
            a4_ex_n = _l2_norm_np(z["ext_all4"])
            r2_te_n = _l2_norm_np(z["te_raw2"])
            a4_te_n = _l2_norm_np(z["te_all4"])

        d_raw = int(a4_tr_n.shape[1])
        c0, c1 = bi * 15, (bi + 1) * 15

        for k in range(5):
            tfk = time.time()
            tr_m, va_m = fold_masks[k]
            y_trk = y[tr_m]
            pi_k = np.bincount(y_trk, minlength=5).astype(np.float64) / len(y_trk)

            mu_r = r2_tr_n[tr_m].mean(axis=0, keepdims=True)
            mu_a = a4_tr_n[tr_m].mean(axis=0, keepdims=True)

            X_c = np.concatenate([_power_cl2n_from_normed(r2_tr_n[tr_m], mu_r), _power_cl2n_from_normed(a4_tr_n[tr_m], mu_a)], axis=1) * inv_s2
            X_e = np.concatenate([_power_cl2n_from_normed(r2_ex_n[tr_m], mu_r), _power_cl2n_from_normed(a4_ex_n[tr_m], mu_a)], axis=1) * inv_s2
            X_vc = np.concatenate([_power_cl2n_from_normed(r2_tr_n[va_m], mu_r), _power_cl2n_from_normed(a4_tr_n[va_m], mu_a)], axis=1) * inv_s2
            X_ve = np.concatenate([_power_cl2n_from_normed(r2_ex_n[va_m], mu_r), _power_cl2n_from_normed(a4_ex_n[va_m], mu_a)], axis=1) * inv_s2
            X_tc = np.concatenate([_power_cl2n_from_normed(r2_te_n, mu_r), _power_cl2n_from_normed(a4_te_n, mu_a)], axis=1) * inv_s2

            p_vc, p_ve, p_tc, zc_in, ze_in, z_vc, z_ve, z_tc = fit_fast_dual_ridge(
                X_c, X_e, y_trk, pi_k, X_vc, X_ve, X_tc, fold_seed=100 + bi * 11 + k
            )
            pb_oof_c[bi][va_m] = p_vc
            pb_ext_c[bi][va_m] = p_ve
            pb_te_c[bi] += p_tc / 5.0
            Z_tr_c[k][:, c0:c1] = zc_in
            Z_tr_e[k][:, c0:c1] = ze_in
            Z_va_c[k][:, c0:c1] = z_vc
            Z_va_e[k][:, c0:c1] = z_ve
            Z_te_c[k][:, c0:c1] = z_tc

            if bi in mlp_specs_map:
                po_arr, pe_arr, pt_arr, base_seeds = mlp_specs_map[bi]
                ov, oe, ot = train_dual_mlp_mps_single(
                    X_c, X_e, y_trk, [X_vc, X_ve, X_tc], [(0, d_raw), (d_raw, 2 * d_raw)],
                    hidden=384, drop=0.45, block_drop=0.10,
                    seeds=[base_seeds[0] + k * 17, base_seeds[1] + k * 17],
                    epochs=12, device=device, lr=1.5e-3,
                )
                po_arr[va_m], pe_arr[va_m] = ov, oe
                pt_arr += ot / 5.0

            del X_c, X_e, X_vc, X_ve, X_tc
            print(
                f"  [{bi+1}/{n_bb} {bkey:14s} F{k}] {time.time()-tfk:.1f}s | RidgeClnF1={f1_score(y[va_m], p_vc.argmax(1), average='macro'):.4f}",
                flush=True,
            )

        del r2_tr_n, a4_tr_n, r2_ex_n, a4_ex_n, r2_te_n, a4_te_n
        gc.collect()
        print(
            f"  ==> [{bi+1}/{n_bb} {bkey:14s}] total {time.time()-tb0:.1f}s | OOF_ClnF1={f1_score(y, pb_oof_c[bi].argmax(1), average='macro'):.5f} "
            f"| OOF_ExtF1={f1_score(y, pb_ext_c[bi].argmax(1), average='macro'):.5f}",
            flush=True,
        )

    # Stage 2: Fast MPS Meta-ResMLPs on 90d multi-lambda features (<5s total)
    slices_6 = [(i * 15, (i + 1) * 15) for i in range(n_bb)]
    slices_3 = [(i * 15, (i + 1) * 15) for i in range(3)]
    vlm_cols = np.concatenate([np.arange(bi * 15, (bi + 1) * 15) for bi in (0, 3, 4)])
    din_cols = np.concatenate([np.arange(bi * 15, (bi + 1) * 15) for bi in (1, 2, 5)])

    oof_b1_t, ext_b1_t, te_b1_t = np.zeros((n_tr, 5)), np.zeros((n_tr, 5)), np.zeros((n_te, 5))
    oof_b2_t, ext_b2_t, te_b2_t = np.zeros((n_tr, 5)), np.zeros((n_tr, 5)), np.zeros((n_te, 5))
    oof_b3_t, ext_b3_t, te_b3_t = np.zeros((n_tr, 5)), np.zeros((n_tr, 5)), np.zeros((n_te, 5))
    oof_b4_t, ext_b4_t, te_b4_t = np.zeros((n_tr, 5)), np.zeros((n_tr, 5)), np.zeros((n_te, 5))

    print(f"[FoundVLM-v3] Stage 2: Training Meta-ResMLPs on {meta_dim}d on {device}...", flush=True)
    for k in range(5):
        t_fk = time.time()
        tr_m, va_m = fold_masks[k]
        y_trk = y[tr_m]

        ov1, oe1, ot1 = train_dual_mlp_mps_single(
            Z_tr_c[k], Z_tr_e[k], y_trk, [Z_va_c[k], Z_va_e[k], Z_te_c[k]], slices_6,
            hidden=192, drop=0.35, block_drop=0.20,
            seeds=[42 + k * 37, 1024 + k * 37, 2025 + k * 37, 3141 + k * 37, 7777 + k * 37],
            epochs=18, device=device, lr=2.0e-3,
        )
        oof_b1_t[va_m], ext_b1_t[va_m] = ov1, oe1
        te_b1_t += ot1 / 5.0

        ov2, oe2, ot2 = train_dual_mlp_mps_single(
            Z_tr_c[k][:, vlm_cols], Z_tr_e[k][:, vlm_cols], y_trk,
            [Z_va_c[k][:, vlm_cols], Z_va_e[k][:, vlm_cols], Z_te_c[k][:, vlm_cols]], slices_3,
            hidden=128, drop=0.30, block_drop=0.15,
            seeds=[707 + k * 23, 808 + k * 23, 909 + k * 23], epochs=16, device=device, lr=2.0e-3,
        )
        oof_b2_t[va_m], ext_b2_t[va_m] = ov2, oe2
        te_b2_t += ot2 / 5.0

        ov3, oe3, ot3 = train_dual_mlp_mps_single(
            Z_tr_c[k][:, din_cols], Z_tr_e[k][:, din_cols], y_trk,
            [Z_va_c[k][:, din_cols], Z_va_e[k][:, din_cols], Z_te_c[k][:, din_cols]], slices_3,
            hidden=128, drop=0.30, block_drop=0.15,
            seeds=[1111 + k * 23, 2222 + k * 23, 3333 + k * 23], epochs=16, device=device, lr=2.0e-3,
        )
        oof_b3_t[va_m], ext_b3_t[va_m] = ov3, oe3
        te_b3_t += ot3 / 5.0

        w_sp = fit_slsqp_hybrid([p[tr_m] for p in pb_oof_c], [p[tr_m] for p in pb_ext_c], y_trk, w_clean_loss=0.75, max_w=0.60)
        oof_b4_t[va_m] = geo_pool([p[va_m] for p in pb_oof_c], w_sp)
        ext_b4_t[va_m] = geo_pool([p[va_m] for p in pb_ext_c], w_sp)
        te_b4_t += geo_pool(pb_te_c, w_sp) / 5.0

        print(
            f"  [Stage2 Fold {k}] {time.time()-t_fk:.1f}s | B1_6BB={f1_score(y[va_m], ov1.argmax(1), average='macro'):.4f} "
            f"B2_VLM={f1_score(y[va_m], ov2.argmax(1), average='macro'):.4f} "
            f"B3_DINO={f1_score(y[va_m], ov3.argmax(1), average='macro'):.4f} "
            f"B4_Ridge6={f1_score(y[va_m], oof_b4_t[va_m].argmax(1), average='macro'):.4f}",
            flush=True,
        )

    del Z_tr_c, Z_tr_e, Z_va_c, Z_va_e, Z_te_c
    gc.collect()

    oof_sig_full, ext_sig_full, te_sig_full, _ = mlp_specs_map[0]
    oof_din_full, ext_din_full, te_din_full, _ = mlp_specs_map[1]
    oof_cnxb_full, ext_cnxb_full, te_cnxb_full, _ = mlp_specs_map[5]

    for hname, po, pe in [
        ("v2_anchor_w6", v2_oof_w6, v2_ext_w6),
        ("B1_Meta6BB_5s", oof_b1_t, ext_b1_t),
        ("B2_VLM3Fam_3s", oof_b2_t, ext_b2_t),
        ("B3_DINO3Fam_3s", oof_b3_t, ext_b3_t),
        ("B4_Ridge6BB", oof_b4_t, ext_b4_t),
        ("MLP_SigLIP2_2s", oof_sig_full, ext_sig_full),
        ("MLP_DINOv3L_2s", oof_din_full, ext_din_full),
        ("MLP_DINOv3CNXB_2s", oof_cnxb_full, ext_cnxb_full),
    ]:
        print(
            f"  [Head] {hname:20s} | ClnF1={f1_score(y, po.argmax(1), average='macro'):.5f} "
            f"| ExtF1={f1_score(y, pe.argmax(1), average='macro'):.5f} "
            f"| ClnLL={log_loss(y, np.clip(po, 1e-7, 1.0), labels=[0,1,2,3,4]):.5f}",
            flush=True,
        )

    all_o2_heads = [v2_oof_o2, oof_b1_t, oof_b2_t, oof_b3_t, oof_b4_t, oof_cnxb_full, oof_sig_full, oof_din_full]
    all_c_heads = [v2_oof_w6, oof_b1_t, oof_b2_t, oof_b3_t, oof_b4_t, oof_cnxb_full, oof_sig_full, oof_din_full]
    all_e_heads = [v2_ext_w6, ext_b1_t, ext_b2_t, ext_b3_t, ext_b4_t, ext_cnxb_full, ext_sig_full, ext_din_full]
    all_t_o2 = [v2_te_o2, te_b1_t, te_b2_t, te_b3_t, te_b4_t, te_cnxb_full, te_sig_full, te_din_full]
    all_t_w6 = [v2_te_w6, te_b1_t, te_b2_t, te_b3_t, te_b4_t, te_cnxb_full, te_sig_full, te_din_full]

    oof_orig2_v3 = np.zeros((n_tr, 5), dtype=np.float64)
    oof_w6_v3 = np.zeros((n_tr, 5), dtype=np.float64)
    ext_w6_v3 = np.zeros((n_tr, 5), dtype=np.float64)
    ws_o2, ws_w6 = [], []

    sig_oof_o2, sig_oof_w6, sig_ext_w6 = np.zeros((n_tr, 5)), np.zeros((n_tr, 5)), np.zeros((n_tr, 5))
    din_oof_o2, din_oof_w6, din_ext_w6 = np.zeros((n_tr, 5)), np.zeros((n_tr, 5)), np.zeros((n_tr, 5))
    ws_sig_o2, ws_sig_w6, ws_din_o2, ws_din_w6 = [], [], [], []

    sig_c_heads = [oof_sig_full, oof_b2_t, pb_oof_c[0]]
    sig_e_heads = [ext_sig_full, ext_b2_t, pb_ext_c[0]]
    sig_t_heads = [te_sig_full, te_b2_t, pb_te_c[0]]

    din_c_heads = [oof_cnxb_full, oof_din_full, oof_b3_t, pb_oof_c[5]]
    din_e_heads = [ext_cnxb_full, ext_din_full, ext_b3_t, pb_ext_c[5]]
    din_t_heads = [te_cnxb_full, te_din_full, te_b3_t, pb_te_c[5]]

    for k in range(5):
        tr_m, va_m = fold_masks[k]
        wk_o2 = fit_slsqp_simplex([p[tr_m] for p in all_o2_heads], y[tr_m], max_w=0.85)
        wk_w6 = fit_slsqp_hybrid([p[tr_m] for p in all_c_heads], [p[tr_m] for p in all_e_heads], y[tr_m], w_clean_loss=0.75, max_w=0.85)
        ws_o2.append(wk_o2); ws_w6.append(wk_w6)
        oof_orig2_v3[va_m] = geo_pool([p[va_m] for p in all_o2_heads], wk_o2)
        oof_w6_v3[va_m] = geo_pool([p[va_m] for p in all_c_heads], wk_w6)
        ext_w6_v3[va_m] = geo_pool([p[va_m] for p in all_e_heads], wk_w6)

        w_so2 = fit_slsqp_simplex([p[tr_m] for p in sig_c_heads], y[tr_m], max_w=0.85)
        w_sw6 = fit_slsqp_hybrid([p[tr_m] for p in sig_c_heads], [p[tr_m] for p in sig_e_heads], y[tr_m], 0.75, 0.85)
        ws_sig_o2.append(w_so2); ws_sig_w6.append(w_sw6)
        sig_oof_o2[va_m] = geo_pool([p[va_m] for p in sig_c_heads], w_so2)
        sig_oof_w6[va_m] = geo_pool([p[va_m] for p in sig_c_heads], w_sw6)
        sig_ext_w6[va_m] = geo_pool([p[va_m] for p in sig_e_heads], w_sw6)

        w_do2 = fit_slsqp_simplex([p[tr_m] for p in din_c_heads], y[tr_m], max_w=0.85)
        w_dw6 = fit_slsqp_hybrid([p[tr_m] for p in din_c_heads], [p[tr_m] for p in din_e_heads], y[tr_m], 0.75, 0.85)
        ws_din_o2.append(w_do2); ws_din_w6.append(w_dw6)
        din_oof_o2[va_m] = geo_pool([p[va_m] for p in din_c_heads], w_do2)
        din_oof_w6[va_m] = geo_pool([p[va_m] for p in din_c_heads], w_dw6)
        din_ext_w6[va_m] = geo_pool([p[va_m] for p in din_e_heads], w_dw6)

    wm_o2, wm_w6 = np.mean(ws_o2, axis=0), np.mean(ws_w6, axis=0)
    te_orig2_v3 = geo_pool(all_t_o2, wm_o2)
    te_w6_v3 = geo_pool(all_t_w6, wm_w6)
    sig_te_o2 = geo_pool(sig_t_heads, np.mean(ws_sig_o2, axis=0))
    sig_te_w6 = geo_pool(sig_t_heads, np.mean(ws_sig_w6, axis=0))
    din_te_o2 = geo_pool(din_t_heads, np.mean(ws_din_o2, axis=0))
    din_te_w6 = geo_pool(din_t_heads, np.mean(ws_din_w6, axis=0))

    oof_u_v3, ext_u_v3 = np.zeros_like(oof_w6_v3), np.zeros_like(ext_w6_v3)
    oof_u_v2, ext_u_v2 = np.zeros_like(v2_oof_w6), np.zeros_like(v2_ext_w6)
    for k in range(5):
        tr_m, va_m = fold_masks[k]
        pi_k = np.bincount(y[tr_m], minlength=5).astype(np.float64) / tr_m.sum()
        oof_u_v3[va_m] = train_to_unif_prior(oof_w6_v3[va_m], pi_k)
        ext_u_v3[va_m] = train_to_unif_prior(ext_w6_v3[va_m], pi_k)
        oof_u_v2[va_m] = train_to_unif_prior(v2_oof_w6[va_m], pi_k)
        ext_u_v2[va_m] = train_to_unif_prior(v2_ext_w6[va_m], pi_k)

    f_tr_v3 = f1_score(y, oof_w6_v3.argmax(1), average="macro")
    f_un_v3 = f1_score(y, oof_u_v3.argmax(1), average="macro", sample_weight=sw_all)
    f_ex_v3 = f1_score(y, ext_u_v3.argmax(1), average="macro", sample_weight=sw_all)
    ll_v3 = log_loss(y, np.clip(oof_w6_v3, 1e-7, 1.0), labels=[0, 1, 2, 3, 4])

    f_un_v2 = f1_score(y, oof_u_v2.argmax(1), average="macro", sample_weight=sw_all)
    f_ex_v2 = f1_score(y, ext_u_v2.argmax(1), average="macro", sample_weight=sw_all)
    ll_v2 = log_loss(y, np.clip(v2_oof_w6, 1e-7, 1.0), labels=[0, 1, 2, 3, 4])

    rng = np.random.default_rng(SEED)
    p3c, p2c = oof_u_v3.argmax(1), oof_u_v2.argmax(1)
    p3e, p2e = ext_u_v3.argmax(1), ext_u_v2.argmax(1)
    win_c, win_e = 0, 0
    for _ in range(500):
        idx = rng.integers(0, n_tr, size=n_tr)
        if f1_score(y[idx], p3c[idx], average="macro", sample_weight=sw_all[idx]) > f1_score(
            y[idx], p2c[idx], average="macro", sample_weight=sw_all[idx]
        ):
            win_c += 1
        if f1_score(y[idx], p3e[idx], average="macro", sample_weight=sw_all[idx]) > f1_score(
            y[idx], p2e[idx], average="macro", sample_weight=sw_all[idx]
        ):
            win_e += 1

    print(f"\n=== [FoundVLM-v3 Fast] COMPLETE in {time.time()-t0:.1f}s ===", flush=True)
    print(f"  Outer orig2 w [v2, B1_Meta, B2_VLM, B3_DINO, B4_R6, CNXB, Sig2, DINO3L]: {np.round(wm_o2, 4).tolist()}", flush=True)
    print(f"  Outer hard  w [v2, B1_Meta, B2_VLM, B3_DINO, B4_R6, CNXB, Sig2, DINO3L]: {np.round(wm_w6, 4).tolist()}", flush=True)
    print(f"  v2 Baseline: UnifPriorF1={f_un_v2:.5f} | OOF_LL={ll_v2:.5f} | ExtHard UnifF1={f_ex_v2:.5f}", flush=True)
    print(
        f"  v3 Upgrade : TrainPriorF1={f_tr_v3:.5f} | UnifPriorF1={f_un_v3:.5f} ({f_un_v3-f_un_v2:+.5f}) "
        f"| OOF_LL={ll_v3:.5f} ({ll_v3-ll_v2:+.5f}) | ExtHard UnifF1={f_ex_v3:.5f} ({f_ex_v3-f_ex_v2:+.5f})",
        flush=True,
    )
    print(f"  500-Boot vs v2: CleanUnifWin={win_c/500.0:.3f} | ExtHardUnifWin={win_e/500.0:.3f}", flush=True)

    bak_npz = M5_OUT_DIR / "task4-foundvlm-v3-backup.npz"
    atomic_savez_compressed(
        bak_npz, folds=folds, oof_orig2=oof_orig2_v3.astype(np.float32), oof_w6=oof_w6_v3.astype(np.float32),
        ext_w6=ext_w6_v3.astype(np.float32), te_orig2=te_orig2_v3.astype(np.float32), te_w6=te_w6_v3.astype(np.float32)
    )
    sig_npz = M5_OUT_DIR / "task4-clean_siglip_m5-backup.npz"
    atomic_savez_compressed(
        sig_npz, folds=folds, oof_orig2=sig_oof_o2.astype(np.float32), oof_w6=sig_oof_w6.astype(np.float32),
        ext_w6=sig_ext_w6.astype(np.float32), te_orig2=sig_te_o2.astype(np.float32), te_w6=sig_te_w6.astype(np.float32)
    )
    din_npz = M5_OUT_DIR / "task4-clean_dinov3l_m5-backup.npz"
    atomic_savez_compressed(
        din_npz, folds=folds, oof_orig2=din_oof_o2.astype(np.float32), oof_w6=din_oof_w6.astype(np.float32),
        ext_w6=din_ext_w6.astype(np.float32), te_orig2=din_te_o2.astype(np.float32), te_w6=din_te_w6.astype(np.float32)
    )
    print(f"Saved backups -> {bak_npz.name}, {sig_npz.name}, {din_npz.name}", flush=True)

    best_winner = None
    if args.ablate_fusion or args.auto_install_best:
        best_winner = run_shadow_ablation(bak_npz, sig_npz, din_npz)

    if args.auto_install_best:
        assert best_winner is not None
        label, ex_keys, extra_map, wmetrics = best_winner
        assert wmetrics["comp"] > wmetrics["base_comp"], f"Winner did not beat Run 27 ({wmetrics['comp']:.6f} <= {wmetrics['base_comp']:.6f})"
        print(f"\n[AUTO-INSTALL] Installing winning configuration: {label}", flush=True)
        install_bundle_set("foundvlm", bak_npz, tr_df, te_df, folds, zip_name="t4_foundvlm.zip")
        for prefix, bpath in extra_map.items():
            install_bundle_set(prefix, bpath, tr_df, te_df, folds, zip_name=f"t4_{prefix}.zip")
        sys.path.insert(0, str(ML_T4_DIR / "scripts"))
        import fuse_t4_sota_trio as fts  # noqa: E402
        res_ml = fts.run_fusion(ML_T4_DIR, save_outputs=True, exclude_keys=ex_keys)
        res_cp = fts.run_fusion(CPE_T4_DIR, save_outputs=True, exclude_keys=ex_keys)
        assert res_ml["metrics"] == res_cp["metrics"], "ml00x vs cpe342 fusion metrics mismatch!"
        print(f"[FUSION UPDATED BOTH REPOS] ex_keys={ex_keys} | metrics={json.dumps(res_ml['metrics'])}", flush=True)
    elif args.install:
        install_bundle_set("foundvlm", bak_npz, tr_df, te_df, folds, zip_name="t4_foundvlm.zip")
        print("[INSTALLED] FoundVLM-v3 into both repos.", flush=True)
    else:
        print("[DRY-INSTALL] Pass --install or --auto-install-best to update repos.", flush=True)


if __name__ == "__main__":
    main()
