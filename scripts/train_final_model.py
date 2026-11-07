"""Train the single final 3-class LightGBM model on C5 with T2 Dense features.

Rules:
- NO new modeling experiments: this fits the pre-registered production model on all 8,000 C5 pairs.
- Uses exact T2 features (1,537-d: u, v, |u-v|, u*v, cos).
- Hyperparameters: n_estimators=300, learning_rate=0.05, max_depth=6, num_leaves=31, random_state=42.
- Fits temperature scaling on inner OOF predictions.
- Saves model to models/c5_lightgbm_final.joblib and metadata to models/c5_lightgbm_final_meta.json.
"""

import os
import sys
import time
import json
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb
from scipy.optimize import minimize
from scipy.special import softmax

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.ats.adapters import get_adapter
from src.ats.embeddings import build_and_cache_document_embeddings
from scripts.run_c5_feature_ablation import extract_t2_dense


def fit_temperature_scaling(oof_probs: np.ndarray, y_true: np.ndarray) -> dict:
    """Optimize temperature T to minimize negative log likelihood on OOF predictions."""
    # Convert probabilities to unnormalized pseudo-logits (with epsilon clipping)
    eps = 1e-7
    clipped_probs = np.clip(oof_probs, eps, 1.0 - eps)
    logits = np.log(clipped_probs)

    def nll_loss(t_val):
        t = t_val[0]
        scaled_logits = logits / max(t, 0.05)
        scaled_probs = softmax(scaled_logits, axis=1)
        # NLL
        correct_probs = scaled_probs[np.arange(len(y_true)), y_true]
        return -np.mean(np.log(np.clip(correct_probs, eps, 1.0)))

    res = minimize(nll_loss, x0=[1.2], bounds=[(0.1, 5.0)], method="L-BFGS-B")
    opt_t = float(res.x[0])

    # Compute uncalibrated vs calibrated ECE
    def compute_ece(probs, y, n_bins=10):
        confidences = np.max(probs, axis=1)
        preds = np.argmax(probs, axis=1)
        accuracies = (preds == y)
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        for i in range(n_bins):
            in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
            prop_in_bin = np.mean(in_bin)
            if prop_in_bin > 0:
                acc_in_bin = np.mean(accuracies[in_bin])
                conf_in_bin = np.mean(confidences[in_bin])
                ece += np.abs(conf_in_bin - acc_in_bin) * prop_in_bin
        return float(ece)

    raw_ece = compute_ece(oof_probs, y_true)
    cal_probs = softmax(logits / opt_t, axis=1)
    cal_ece = compute_ece(cal_probs, y_true)

    return {
        "optimal_temperature": opt_t,
        "raw_ece": raw_ece,
        "calibrated_ece": cal_ece,
        "ece_reduction": raw_ece - cal_ece,
    }


def main():
    print("=" * 70)
    print("Fitting Single Final Production Model: LightGBM on C5 (T2 Dense)")
    print("=" * 70)

    models_dir = PROJECT_ROOT / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load C5 Dataset
    t0 = time.time()
    adapter = get_adapter("c5")
    df = adapter.load()
    print(f"[Data] Loaded C5: {len(df)} pairs, {df['resume_id'].nunique()} resumes, {df['jd_id'].nunique()} JDs.")

    label_map = {"No Fit": 0, "Potential Fit": 1, "Good Fit": 2}
    inv_label_map = {0: "No Fit", 1: "Potential Fit", 2: "Good Fit"}
    if df["label_class"].dtype == object or isinstance(df["label_class"].iloc[0], str):
        y = df["label_class"].map(label_map).fillna(0).astype(int).values
    else:
        y = df["label_class"].astype(int).values

    # 2. Document Embeddings & T2 Features
    doc_emb_map, _ = build_and_cache_document_embeddings(
        df,
        dataset_name="c5",
        cache_dir="cache/embeddings",
        include_sentences=False,
    )
    print(f"[Features] Building T2 Dense features (1,537-d)...")
    X_T2 = extract_t2_dense(df, doc_emb_map)
    print(f"[Features] Shape: {X_T2.shape}, dtype: {X_T2.dtype}")

    # 3. Train LightGBM Model
    lgb_params = {
        "objective": "multiclass",
        "num_class": 3,
        "n_estimators": 300,
        "max_depth": 6,
        "num_leaves": 31,
        "learning_rate": 0.05,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }
    print("[Training] Fitting LightGBMClassifier on all 8,000 pairs...")
    clf = lgb.LGBMClassifier(**lgb_params)
    t_train = time.time()
    clf.fit(X_T2, y)
    train_time = time.time() - t_train
    print(f"[Training] Fitted in {train_time:.2f}s.")

    # 4. Temperature Calibration from inner OOF predictions
    oof_path = PROJECT_ROOT / "results" / "c5_lightgbm_oof_predictions.parquet"
    if oof_path.exists():
        oof_df = pd.read_parquet(oof_path)
        print(f"[Calibration] Loaded OOF predictions: {len(oof_df)} rows.")
        s1_probs = oof_df[["s1_prob_0", "s1_prob_1", "s1_prob_2"]].values
        s2_probs = oof_df[["s2_prob_0", "s2_prob_1", "s2_prob_2"]].values
        y_oof = oof_df["y_true"].values

        calib_s1 = fit_temperature_scaling(s1_probs, y_oof)
        calib_s2 = fit_temperature_scaling(s2_probs, y_oof)
        print(f"[Calibration] S1 optimal T: {calib_s1['optimal_temperature']:.2f} (ECE: {calib_s1['raw_ece']:.4f} -> {calib_s1['calibrated_ece']:.4f})")
        print(f"[Calibration] S2 optimal T: {calib_s2['optimal_temperature']:.2f} (ECE: {calib_s2['raw_ece']:.4f} -> {calib_s2['calibrated_ece']:.4f})")
        opt_t = float(calib_s1["optimal_temperature"])
    else:
        print("[Calibration] OOF file not found, defaulting T=1.19.")
        opt_t = 1.19
        calib_s1 = {"optimal_temperature": 1.19}
        calib_s2 = {"optimal_temperature": 1.91}

    # 5. Save model and metadata
    model_file = models_dir / "c5_lightgbm_final.joblib"
    joblib.dump(clf, model_file)
    model_sha256 = hashlib.sha256(open(model_file, "rb").read()).hexdigest()
    model_size_mb = model_file.stat().st_size / (1024 * 1024)

    meta = {
        "model_type": "LightGBMClassifier",
        "library_version": lgb.__version__,
        "dataset": "cnamuangtoun/resume-job-description-fit",
        "training_samples": len(df),
        "feature_tier": "T2_Dense",
        "feature_dim": int(X_T2.shape[1]),
        "feature_definition": "[u (384), v (384), |u-v| (384), u*v (384), cos(u,v) (1)]",
        "embedding_backbone": "sentence-transformers/all-MiniLM-L6-v2",
        "embedding_dim": 384,
        "classes": ["No Fit", "Potential Fit", "Good Fit"],
        "class_mapping": label_map,
        "hyperparameters": lgb_params,
        "calibration": {
            "method": "Temperature Scaling",
            "optimal_temperature_s1": calib_s1["optimal_temperature"],
            "optimal_temperature_s2": calib_s2["optimal_temperature"],
            "serving_default_temperature": 1.19,
            "s1_ece_uncalibrated": calib_s1.get("raw_ece"),
            "s1_ece_calibrated": calib_s1.get("calibrated_ece"),
            "s2_ece_uncalibrated": calib_s2.get("raw_ece"),
            "s2_ece_calibrated": calib_s2.get("calibrated_ece"),
        },
        "model_file": "c5_lightgbm_final.joblib",
        "file_size_mb": round(model_size_mb, 2),
        "sha256": model_sha256,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
    }

    meta_file = models_dir / "c5_lightgbm_final_meta.json"
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"[OK] Model saved to {model_file} ({model_size_mb:.2f} MB, sha256={model_sha256[:16]}...)")
    print(f"[OK] Metadata saved to {meta_file}")


if __name__ == "__main__":
    main()
