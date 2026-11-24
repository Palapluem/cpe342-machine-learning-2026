#!/usr/bin/env python3
"""1-Command Kaggle CleanSigLIP-SO400M Verifier, Installer, and Pure-Train Ablation Runner.

Validates downloaded `t4_clean_siglip.zip` (or `t4_found_bundles*.zip`), reports standalone
CleanSigLIP-SO400M SGKF metrics, runs dry-run shadow ablations against active Run 27 (0.86680)
with 500-resample paired bootstrap and test delta-rows, and optionally installs into both repos.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, log_loss

CPE_T4_DIR = Path("/Users/wisitsuwannao/Documents/Workspace/cpe342-machine-learning-2026/project/task4")
ML_T4_DIR = Path("/Users/wisitsuwannao/Documents/Workspace/ml00x/task4")
sys.path.insert(0, str(ML_T4_DIR / "scripts"))

import fuse_t4_sota_trio as fts  # noqa: E402

REQ_FILES = (
    "task4-clean_siglip-hard6-oof.csv",
    "task4-clean_siglip-hard6-test-predictions.csv",
    "task4-clean_siglip-orig2-oof.csv",
    "task4-clean_siglip-orig2-test-predictions.csv",
    "task4-clean_siglip-sgkf-hard-bundle.npz",
)


def auto_find_zip() -> Optional[Path]:
    dl = Path.home() / "Downloads"
    if not dl.is_dir():
        return None
    candidates = sorted(
        list(dl.glob("*clean_siglip*.zip")) + list(dl.glob("t4_found_bundles*.zip")),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for p in candidates:
        try:
            with zipfile.ZipFile(p, "r") as zf:
                names = {Path(n).name for n in zf.namelist()}
                if "task4-clean_siglip-sgkf-hard-bundle.npz" in names:
                    return p
        except zipfile.BadZipFile:
            continue
    return None


def train_to_unif(p: np.ndarray, pi_vec: np.ndarray) -> np.ndarray:
    q = np.clip(p, 1e-7, 1.0) / np.clip(pi_vec[None, :], 1e-7, 1.0)
    return q / q.sum(axis=1, keepdims=True)


def validate_zip_and_report(zip_path: Path) -> Tuple[Path, Dict[str, Path], np.ndarray, np.ndarray]:
    ref_folds = np.load(ML_T4_DIR / "predictions/model-level/task4-sgkf-e2e-bundle.npz")["folds"].astype(int)
    tr_df = pd.read_csv(ML_T4_DIR / "data/train.csv")
    y = (
        tr_df["label"].astype(int).to_numpy()
        if "label" in tr_df.columns
        else tr_df["class"].map(fts.CLASS_MAP).astype(int).to_numpy()
    )
    n_tr, n_te = len(y), 25889
    assert n_tr == 31546

    tmp_dir = Path(tempfile.mkdtemp(prefix="clean_siglip_verify_"))
    extracted: Dict[str, Path] = {}
    with zipfile.ZipFile(zip_path, "r") as zf:
        by_base = {Path(n).name: n for n in zf.namelist() if not n.endswith("/")}
        missing = [fn for fn in REQ_FILES if fn not in by_base]
        assert not missing, f"Missing required CleanSigLIP files in {zip_path}: {missing}"
        for fn in REQ_FILES:
            dst = tmp_dir / fn
            with zf.open(by_base[fn]) as src, open(dst, "wb") as out:
                shutil.copyfileobj(src, out)
            extracted[fn] = dst

    with np.load(extracted["task4-clean_siglip-sgkf-hard-bundle.npz"]) as zb:
        bfolds = zb["folds"].astype(int)
        assert np.array_equal(bfolds, ref_folds), "SGKF folds mismatch vs task4-sgkf-e2e-bundle.npz!"
        for k, exp_shape in [
            ("oof_orig2", (n_tr, 5)),
            ("oof_w6", (n_tr, 5)),
            ("ext_w6", (n_tr, 5)),
            ("te_orig2", (n_te, 5)),
            ("te_w6", (n_te, 5)),
        ]:
            arr = zb[k]
            assert arr.shape == exp_shape, f"{k} shape {arr.shape} != {exp_shape}"
            assert np.isfinite(arr).all(), f"{k} contains non-finite values!"
            rs = arr.sum(axis=1)
            assert np.allclose(rs, 1.0, atol=2e-3), f"{k} row sums deviate from 1.0: [{rs.min()}, {rs.max()}]"
        oof_o2 = zb["oof_orig2"].astype(np.float64)
        oof_w6 = zb["oof_w6"].astype(np.float64)
        ext_w6 = zb["ext_w6"].astype(np.float64)

    te_csv = pd.read_csv(extracted["task4-clean_siglip-hard6-test-predictions.csv"])
    assert len(te_csv) == n_te
    assert str(te_csv["id"].iloc[0]) == "ANS00001" and str(te_csv["file_name"].iloc[0]) == "sjqyfwc.jpg", (
        f"Test row 0 mismatch: id={te_csv['id'].iloc[0]}, file_name={te_csv['file_name'].iloc[0]}"
    )

    oof_u = np.zeros_like(oof_w6)
    ext_u = np.zeros_like(ext_w6)
    for k in range(5):
        tr_m, va_m = (ref_folds != k), (ref_folds == k)
        pi_k = np.bincount(y[tr_m], minlength=5).astype(np.float64) / tr_m.sum()
        oof_u[va_m] = train_to_unif(oof_w6[va_m], pi_k)
        ext_u[va_m] = train_to_unif(ext_w6[va_m], pi_k)

    pi_all = np.bincount(y, minlength=5).astype(np.float64) / n_tr
    sw_all = 1.0 / pi_all[y]
    print(f"=== Standalone CleanSigLIP-SO400M ({zip_path.name}) ===")
    print(
        f"  orig2: TrainPriorF1={f1_score(y, oof_o2.argmax(1), average='macro'):.5f} "
        f"| OOF_LL={log_loss(y, np.clip(oof_o2, 1e-7, 1.0), labels=[0,1,2,3,4]):.5f}"
    )
    print(
        f"  hard6: TrainPriorF1={f1_score(y, oof_w6.argmax(1), average='macro'):.5f} "
        f"| UnifPriorF1={f1_score(y, oof_u.argmax(1), average='macro', sample_weight=sw_all):.5f} "
        f"| OOF_LL={log_loss(y, np.clip(oof_w6, 1e-7, 1.0), labels=[0,1,2,3,4]):.5f}"
    )
    print(
        f"  ExtHard6: UnifF1={f1_score(y, ext_u.argmax(1), average='macro', sample_weight=sw_all):.5f} "
        f"| ExtLL={log_loss(y, np.clip(ext_w6, 1e-7, 1.0), labels=[0,1,2,3,4]):.5f}"
    )
    return tmp_dir, extracted, y, ref_folds


def compute_fusion_arrays(
    task4_dir: Path,
    exclude_keys: Tuple[str, ...],
) -> Tuple[Dict[str, object], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Replay fuse_t4_sota_trio exactly while returning OOF/Ext/Test arrays for paired bootstrap."""
    data_dir = task4_dir / "data"
    ml_dir = task4_dir / "predictions" / "model-level"
    train_df = pd.read_csv(data_dir / "train.csv")
    y = (
        train_df["label"].astype(int).to_numpy()
        if "label" in train_df.columns
        else train_df["class"].map(fts.CLASS_MAP).astype(int).to_numpy()
    )
    e2e = np.load(ml_dir / "task4-sgkf-e2e-bundle.npz")
    folds = e2e["folds"].astype(int)
    w_hard_iso = e2e["train_hard_iso_w"].astype(np.float64)
    iso_mask = e2e["train_iso_mask"].astype(bool)
    corr_mask = e2e["train_corr_mask"].astype(bool)
    p_cols = [f"prob_{i}" for i in range(5)]

    base_vit_names = ["dinov3-vitl", "eva02-large", "eva02-base", "dinov3-vitb"]
    vit_oof = {m: fts.norm_p(pd.read_csv(ml_dir / f"task4-{m}-oof.csv")[p_cols].to_numpy()) for m in base_vit_names}
    vit_te = {m: fts.norm_p(pd.read_csv(ml_dir / f"task4-{m}-test-predictions.csv")[p_cols].to_numpy()) for m in base_vit_names}

    excluded = set(exclude_keys)
    discovered_fullres = {}
    for key in fts.DEFAULT_FULLRES_KEYS:
        if key in excluded:
            continue
        bpath = ml_dir / f"task4-{key}-sgkf-hard-bundle.npz"
        if bpath.is_file():
            discovered_fullres[key] = np.load(bpath)

    pool_names = [
        "ConvNeXtV2-SGKF",
        "DINOv2-StageA",
        "MLP-V3-SGKF",
        "DINOv3-ViT-L",
        "EVA02-Large",
        "EVA02-Base",
        "DINOv3-ViT-B",
        "DINOv2-StageC-2TTA",
        "DINOv2-StageC-6TTA",
    ]
    pool_oof = [
        fts.norm_p(e2e["cnx_a_oof"]),
        fts.norm_p(e2e["din_a_oof"]),
        fts.norm_p(e2e["mlp_oof"]),
        vit_oof["dinov3-vitl"],
        vit_oof["eva02-large"],
        vit_oof["eva02-base"],
        vit_oof["dinov3-vitb"],
        fts.norm_p(e2e["din_c2_oof"]),
        fts.norm_p(e2e["din_c6_oof"]),
    ]
    pool_te = [
        fts.norm_p(e2e["cnx_a_te"]),
        fts.norm_p(e2e["din_a_te"]),
        fts.norm_p(e2e["mlp_te"]),
        vit_te["dinov3-vitl"],
        vit_te["eva02-large"],
        vit_te["eva02-base"],
        vit_te["dinov3-vitb"],
        fts.norm_p(e2e["din_c2_te"]),
        fts.norm_p(e2e["din_c6_te"]),
    ]
    if "cnxv3" not in discovered_fullres:
        pool_names += ["CNXv2-StageC-2TTA", "CNXv2-StageC-6TTA"]
        pool_oof += [fts.norm_p(e2e["cnx_c2_oof"]), fts.norm_p(e2e["cnx_c6_oof"])]
        pool_te += [fts.norm_p(e2e["cnx_c2_te"]), fts.norm_p(e2e["cnx_c6_te"])]

    for key, bun in discovered_fullres.items():
        lbl = fts.HARDENED_LABELS.get(key, key)
        pool_names += [f"{lbl}-Orig2", f"{lbl}-Hard6"]
        pool_oof += [fts.norm_p(bun["oof_orig2"]), fts.norm_p(bun["oof_w6"])]
        pool_te += [fts.norm_p(bun["te_orig2"]), fts.norm_p(bun["te_w6"])]

    T_pool = fts.fit_nested_temps_weighted(pool_oof, y, folds, w_hard_iso)
    w_clean, clean_oof_raw, _ = fts.fit_nested_pool_weights(
        pool_oof, T_pool, y, folds, w_hard_iso, [(0.0, 0.60)] * len(pool_oof)
    )
    clean_te_raw = fts.geo_pool(pool_te, w_clean, T_pool)

    hard_names = ["CNXv2-StageC-6TTA", "DINOv2-StageC-6TTA"]
    hard_ext_oof = [fts.norm_p(e2e["cnx_c6_ext"]), fts.norm_p(e2e["din_c6_ext"])]
    hard_std_oof = [fts.norm_p(e2e["cnx_c6_oof"]), fts.norm_p(e2e["din_c6_oof"])]
    hard_te = [fts.norm_p(e2e["cnx_c6_te"]), fts.norm_p(e2e["din_c6_te"])]
    for key, bun in discovered_fullres.items():
        lbl = fts.HARDENED_LABELS.get(key, key)
        hard_names.append(f"{lbl}-Hard6")
        hard_ext_oof.append(fts.norm_p(bun["ext_w6"]))
        hard_std_oof.append(fts.norm_p(bun["oof_w6"]))
        hard_te.append(fts.norm_p(bun["te_w6"]))

    T_hard = [T_pool[pool_names.index(n)] if n in pool_names else 1.0 for n in hard_names]
    if "CNXv2-StageC-6TTA" not in pool_names:
        T_hard[0] = fts.fit_nested_temps_weighted([hard_std_oof[0]], y, folds, w_hard_iso)[0]

    w_hard, ext_hard_oof, hard_fold_ws = fts.fit_nested_pool_weights(
        hard_ext_oof, T_hard, y, folds, np.ones(len(y), dtype=np.float64), [(0.05, 0.80)] * len(hard_names)
    )
    hard_std_oof_pool = np.zeros_like(clean_oof_raw)
    for k in range(5):
        va = folds == k
        hard_std_oof_pool[va] = fts.geo_pool([p[va] for p in hard_std_oof], hard_fold_ws[k], T_hard)
    hard_te_pool = fts.geo_pool(hard_te, w_hard, T_hard)

    js_oof = np.mean([fts.js_divergence(fts.norm_p(b["oof_orig2"]), fts.norm_p(b["oof_w6"])) for b in discovered_fullres.values()], axis=0)
    js_te = np.mean([fts.js_divergence(fts.norm_p(b["te_orig2"]), fts.norm_p(b["te_w6"])) for b in discovered_fullres.values()], axis=0)

    gated_oof_raw, tau_js, opt_gamma = fts.fit_nested_shift_gate(
        clean_oof_raw, hard_std_oof_pool, js_oof, y, folds, iso_mask, corr_mask
    )
    gated_te_raw, _ = fts.apply_shift_gate_blend(clean_te_raw, hard_te_pool, js_te, tau_js, opt_gamma)

    a_lo, a_hi, _, _, final_oof_deb, final_ext_deb = fts.fit_nested_entropy_prior_debias(
        gated_oof_raw, ext_hard_oof, y, folds, h0=0.20, h1=0.50
    )
    pi_tr = np.bincount(y, minlength=5).astype(np.float64) / len(y)
    final_te_probs = fts.apply_entropy_gated_prior_debias(gated_te_raw, pi_tr, a_lo, a_hi, h0=0.20, h1=0.50)

    summary = fts.run_fusion(task4_dir, save_outputs=False, exclude_keys=exclude_keys)
    return summary, final_oof_deb, final_ext_deb, final_te_probs.argmax(1).astype(int), corr_mask, pi_tr


def make_shadow_task4_dir(
    extracted: Dict[str, Path],
    foundvlm_npz: Optional[Path] = None,
) -> Path:
    sh_dir = Path(tempfile.mkdtemp(prefix="t4_shadow_"))
    os.symlink(ML_T4_DIR / "data", sh_dir / "data")
    os.symlink(ML_T4_DIR / "models", sh_dir / "models")
    ml_sh = sh_dir / "predictions" / "model-level"
    ml_sh.mkdir(parents=True, exist_ok=True)
    for src in (ML_T4_DIR / "predictions" / "model-level").iterdir():
        os.symlink(src, ml_sh / src.name)
    for fn, p in extracted.items():
        dst = ml_sh / fn
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        os.symlink(p, dst)
    if foundvlm_npz is not None:
        assert foundvlm_npz.is_file(), f"Missing --foundvlm-npz: {foundvlm_npz}"
        dst_fv = ml_sh / "task4-foundvlm-sgkf-hard-bundle.npz"
        if dst_fv.exists() or dst_fv.is_symlink():
            dst_fv.unlink()
        os.symlink(foundvlm_npz.resolve(), dst_fv)
    return sh_dir


def main() -> None:
    ap = argparse.ArgumentParser(description="Verify & ablate CleanSigLIP-SO400M against active Run 27")
    ap.add_argument("--zip", type=Path, default=None, help="Path to t4_clean_siglip.zip (auto-detects ~/Downloads if omitted)")
    ap.add_argument("--foundvlm-npz", type=Path, default=None, help="Optional FoundVLM-v3 backup npz to include in dry-run shadow")
    ap.add_argument("--install", action="store_true", help="Install CleanSigLIP files into ml00x & cpe342 predictions/model-level")
    args = ap.parse_args()

    zip_path = args.zip or auto_find_zip()
    if zip_path is None or not zip_path.is_file():
        raise FileNotFoundError("No t4_clean_siglip*.zip found in ~/Downloads; pass --zip explicitly.")

    tmp_ext, extracted, y, _ = validate_zip_and_report(zip_path)
    sh_dir = make_shadow_task4_dir(extracted, foundvlm_npz=args.foundvlm_npz)

    try:
        active_sub = pd.read_csv(ML_T4_DIR / "submit/index.csv")["task4"].to_numpy()
        b_sum, b_oof, b_ext, b_pred, corr_mask, pi_tr = compute_fusion_arrays(ML_T4_DIR, exclude_keys=("siglip_l16",))
        assert np.array_equal(b_pred, active_sub), "Active Run 27 replay mismatch vs ml00x/task4/submit/index.csv!"

        sw_all = 1.0 / pi_tr[y]
        rng = np.random.default_rng(42)
        n = len(y)

        print("\n=== Ablation vs Active Run 27 (0.86680) ===")
        print(f"Baseline Run 27 metrics: {json.dumps(b_sum['metrics'])}")

        for label, ex_keys in [
            ("+clean_siglip (excl siglip_l16)", ("siglip_l16",)),
            ("+clean_siglip (keep siglip_l16)", ()),
        ]:
            s_new, oof_new, ext_new, pred_new, _, _ = compute_fusion_arrays(sh_dir, exclude_keys=ex_keys)
            w_cln, w_cor, w_ext = 0, 0, 0
            p0c, p1c = b_oof.argmax(1), oof_new.argmax(1)
            p0e, p1e = b_ext.argmax(1), ext_new.argmax(1)
            idx_corr = np.where(corr_mask)[0]
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

            d_rows = int((pred_new != active_sub).sum())
            print(f"\n--- {label} ---")
            print(f"  Discovered models : {s_new['discovered_fullres_models']}")
            print(f"  Clean pool weights: {s_new['clean_pool_weights']}")
            print(f"  ExtHard weights   : {s_new['exthard_weights']}")
            print(f"  Shift gate        : {s_new['shift_gate']}")
            print(f"  Metrics           : {s_new['metrics']}")
            print(
                f"  Delta vs Run 27   : CleanUnifF1={s_new['metrics']['clean_unif_oof_f1'] - b_sum['metrics']['clean_unif_oof_f1']:+.6f} "
                f"| CorrF1={s_new['metrics']['corrupt_oof_f1'] - b_sum['metrics']['corrupt_oof_f1']:+.6f} "
                f"| ExtUnifF1={s_new['metrics']['exthard6_unif_oof_f1'] - b_sum['metrics']['exthard6_unif_oof_f1']:+.6f} "
                f"| OOF_LL={s_new['metrics']['oof_logloss'] - b_sum['metrics']['oof_logloss']:+.6f}"
            )
            print(
                f"  500-Boot win vs R27: CleanUnif={w_cln/500:.3f} | Corrupt={w_cor/500:.3f} | ExtHard6={w_ext/500:.3f} "
                f"| test Δrows vs Run27={d_rows:,}"
            )

        if args.install:
            for tgt_dir in (ML_T4_DIR / "predictions/model-level", CPE_T4_DIR / "predictions/model-level"):
                tgt_dir.mkdir(parents=True, exist_ok=True)
                for fn, src in extracted.items():
                    shutil.copy2(src, tgt_dir / fn)
            bundle_dir = CPE_T4_DIR / "t4_hard_bundles"
            bundle_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(zip_path, bundle_dir / "t4_clean_siglip.zip")
            print(f"\n[INSTALLED] 5 CleanSigLIP files into both repos & {bundle_dir / 't4_clean_siglip.zip'}")
        else:
            print("\n[DRY-RUN] Not installed. Pass --install to copy CleanSigLIP into ml00x & cpe342.")
    finally:
        shutil.rmtree(tmp_ext, ignore_errors=True)
        shutil.rmtree(sh_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
