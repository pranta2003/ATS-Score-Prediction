"""Like-for-like Leakage Comparison Study.

Evaluates the EXACT SAME T2 Dense + LightGBM model on C5 (8,000 pairs) under:
1. Random Stratified 5-Fold CV (Pair-level split; represents warm/leaked evaluation)
2. S1 Cold-Resume 5-Fold GroupKFold on resume_id (No resume leakage)
3. S2 Cold-JD 5-Fold GroupKFold on jd_id (No JD leakage)

Saves:
- results/leakage_comparison_results.json
- results/c5_leakage_oof_predictions.parquet
- reports/LEAKAGE_COMPARISON.md
"""

import os
import sys
import time
import json
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.ats.adapters import get_adapter
from src.ats.embeddings import build_and_cache_document_embeddings
from src.ats.evaluation import (
    compute_classification_metrics,
    vectorized_cluster_bootstrap_acc_f1,
    vectorized_paired_cluster_bootstrap_diff,
)


def extract_t2_dense(df: pd.DataFrame, doc_emb_map: dict) -> np.ndarray:
    """Build canonical T2 dense feature matrix."""
    X_res = np.array([doc_emb_map[rid] for rid in df["resume_id"]], dtype=np.float32)
    X_jd = np.array([doc_emb_map[jid] for jid in df["jd_id"]], dtype=np.float32)

    diff = np.abs(X_res - X_jd)
    mult = X_res * X_jd
    cos = np.sum(X_res * X_jd, axis=1, keepdims=True)

    return np.hstack([X_res, X_jd, diff, mult, cos])


def run_cv(X: np.ndarray, y: np.ndarray, split_gen, params: dict) -> tuple[np.ndarray, np.ndarray]:
    """Execute 5-fold CV returning out-of-fold probabilities and predictions."""
    oof_probs = np.zeros((len(y), 3), dtype=np.float32)
    oof_preds = np.zeros(len(y), dtype=np.int32)

    for tr_idx, val_idx in split_gen:
        clf = lgb.LGBMClassifier(**params)
        clf.fit(X[tr_idx], y[tr_idx])
        p = clf.predict_proba(X[val_idx])
        oof_probs[val_idx] = p
        oof_preds[val_idx] = np.argmax(p, axis=1)

    return oof_probs, oof_preds


def main():
    print("=" * 60)
    print("  LIKE-FOR-LIKE LEAKAGE COMPARISON: T2 DENSE + LIGHTGBM")
    print("=" * 60)

    adapter = get_adapter("c5")
    df = adapter.load()
    print(f"Loaded C5: {len(df)} pairs, {df['resume_id'].nunique()} resumes, {df['jd_id'].nunique()} JDs.")

    label_map = {"No Fit": 0, "Potential Fit": 1, "Good Fit": 2}
    y = np.array([label_map[l] for l in df["label_class"]], dtype=int)

    emb_dir = str(PROJECT_ROOT / "cache" / "embeddings")
    doc_emb_map, _ = build_and_cache_document_embeddings(df, dataset_name="c5", cache_dir=emb_dir)

    X = extract_t2_dense(df, doc_emb_map)
    print(f"Constructed T2 dense features: shape {X.shape}")

    lgb_params = {
        "objective": "multiclass",
        "num_class": 3,
        "n_estimators": 300,
        "max_depth": 6,
        "num_leaves": 31,
        "learning_rate": 0.05,
        "random_state": 42,
        "verbosity": -1,
        "n_jobs": -1,
    }

    results = {}

    # 1. Random Stratified 5-Fold
    print("\n--- 1. Running Random Stratified 5-Fold CV ---")
    t0 = time.time()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    rand_probs, rand_preds = run_cv(X, y, skf.split(X, y), lgb_params)
    rand_time = time.time() - t0
    rand_metrics = compute_classification_metrics(y, rand_preds, rand_probs)
    # For random split, cluster bootstrap on pair index (i.e. iid pairs)
    pair_clusters = np.arange(len(y))
    rand_ci_acc, rand_ci_f1 = vectorized_cluster_bootstrap_acc_f1(y, rand_preds, pair_clusters, n_boot=500, seed=42)
    print(f"Random 5-fold finished in {rand_time:.2f}s:")
    print(f"  Accuracy: {rand_metrics['accuracy']:.4f} (95% CI: [{rand_ci_acc[0]:.4f}, {rand_ci_acc[1]:.4f}])")
    print(f"  Macro-F1: {rand_metrics['macro_f1']:.4f} (95% CI: [{rand_ci_f1[0]:.4f}, {rand_ci_f1[1]:.4f}])")
    print(f"  Macro-AUC: {rand_metrics['macro_roc_auc']:.4f}")

    results["random_stratified"] = {
        "accuracy": float(rand_metrics["accuracy"]),
        "accuracy_ci95": [float(rand_ci_acc[0]), float(rand_ci_acc[1])],
        "macro_f1": float(rand_metrics["macro_f1"]),
        "macro_f1_ci95": [float(rand_ci_f1[0]), float(rand_ci_f1[1])],
        "macro_roc_auc": float(rand_metrics["macro_roc_auc"]),
        "time_seconds": round(rand_time, 2),
    }

    # 2. S1: Cold Resume (GroupKFold on resume_id)
    print("\n--- 2. Running S1: Cold Resume 5-Fold GroupKFold ---")
    t0 = time.time()
    gkf_s1 = GroupKFold(n_splits=5)
    s1_probs, s1_preds = run_cv(X, y, gkf_s1.split(X, y, groups=df["resume_id"]), lgb_params)
    s1_time = time.time() - t0
    s1_metrics = compute_classification_metrics(y, s1_preds, s1_probs)
    # Cluster bootstrap on resume_id
    resume_codes = pd.Categorical(df["resume_id"]).codes
    s1_ci_acc, s1_ci_f1 = vectorized_cluster_bootstrap_acc_f1(y, s1_preds, resume_codes, n_boot=500, seed=42)
    print(f"S1 finished in {s1_time:.2f}s:")
    print(f"  Accuracy: {s1_metrics['accuracy']:.4f} (95% CI: [{s1_ci_acc[0]:.4f}, {s1_ci_acc[1]:.4f}])")
    print(f"  Macro-F1: {s1_metrics['macro_f1']:.4f} (95% CI: [{s1_ci_f1[0]:.4f}, {s1_ci_f1[1]:.4f}])")
    print(f"  Macro-AUC: {s1_metrics['macro_roc_auc']:.4f}")

    results["s1_cold_resume"] = {
        "accuracy": float(s1_metrics["accuracy"]),
        "accuracy_ci95": [float(s1_ci_acc[0]), float(s1_ci_acc[1])],
        "macro_f1": float(s1_metrics["macro_f1"]),
        "macro_f1_ci95": [float(s1_ci_f1[0]), float(s1_ci_f1[1])],
        "macro_roc_auc": float(s1_metrics["macro_roc_auc"]),
        "time_seconds": round(s1_time, 2),
    }

    # 3. S2: Cold JD (GroupKFold on jd_id)
    print("\n--- 3. Running S2: Cold JD 5-Fold GroupKFold ---")
    t0 = time.time()
    gkf_s2 = GroupKFold(n_splits=5)
    s2_probs, s2_preds = run_cv(X, y, gkf_s2.split(X, y, groups=df["jd_id"]), lgb_params)
    s2_time = time.time() - t0
    s2_metrics = compute_classification_metrics(y, s2_preds, s2_probs)
    # Cluster bootstrap on jd_id
    jd_codes = pd.Categorical(df["jd_id"]).codes
    s2_ci_acc, s2_ci_f1 = vectorized_cluster_bootstrap_acc_f1(y, s2_preds, jd_codes, n_boot=500, seed=42)
    print(f"S2 finished in {s2_time:.2f}s:")
    print(f"  Accuracy: {s2_metrics['accuracy']:.4f} (95% CI: [{s2_ci_acc[0]:.4f}, {s2_ci_acc[1]:.4f}])")
    print(f"  Macro-F1: {s2_metrics['macro_f1']:.4f} (95% CI: [{s2_ci_f1[0]:.4f}, {s2_ci_f1[1]:.4f}])")
    print(f"  Macro-AUC: {s2_metrics['macro_roc_auc']:.4f}")

    results["s2_cold_jd"] = {
        "accuracy": float(s2_metrics["accuracy"]),
        "accuracy_ci95": [float(s2_ci_acc[0]), float(s2_ci_acc[1])],
        "macro_f1": float(s2_metrics["macro_f1"]),
        "macro_f1_ci95": [float(s2_ci_f1[0]), float(s2_ci_f1[1])],
        "macro_roc_auc": float(s2_metrics["macro_roc_auc"]),
        "time_seconds": round(s2_time, 2),
    }

    # 4. Compute Measured Leakage Gaps
    gap_rand_vs_s1_acc = float(rand_metrics["accuracy"] - s1_metrics["accuracy"])
    gap_rand_vs_s2_acc = float(rand_metrics["accuracy"] - s2_metrics["accuracy"])
    gap_rand_vs_s1_f1 = float(rand_metrics["macro_f1"] - s1_metrics["macro_f1"])
    gap_rand_vs_s2_f1 = float(rand_metrics["macro_f1"] - s2_metrics["macro_f1"])

    # Paired cluster-bootstrap on differences
    diff_s1_ci_acc, diff_s1_ci_f1 = vectorized_paired_cluster_bootstrap_diff(
        y, rand_preds, s1_preds, resume_codes, n_boot=500, seed=42
    )
    diff_s2_ci_acc, diff_s2_ci_f1 = vectorized_paired_cluster_bootstrap_diff(
        y, rand_preds, s2_preds, jd_codes, n_boot=500, seed=42
    )

    results["leakage_gaps"] = {
        "random_vs_s1_cold_resume": {
            "delta_accuracy": round(gap_rand_vs_s1_acc, 4),
            "delta_accuracy_ci95": [float(diff_s1_ci_acc[0]), float(diff_s1_ci_acc[1])],
            "delta_macro_f1": round(gap_rand_vs_s1_f1, 4),
            "delta_macro_f1_ci95": [float(diff_s1_ci_f1[0]), float(diff_s1_ci_f1[1])],
            "delta_macro_auc": round(float(rand_metrics["macro_roc_auc"] - s1_metrics["macro_roc_auc"]), 4),
        },
        "random_vs_s2_cold_jd": {
            "delta_accuracy": round(gap_rand_vs_s2_acc, 4),
            "delta_accuracy_ci95": [float(diff_s2_ci_acc[0]), float(diff_s2_ci_acc[1])],
            "delta_macro_f1": round(gap_rand_vs_s2_f1, 4),
            "delta_macro_f1_ci95": [float(diff_s2_ci_f1[0]), float(diff_s2_ci_f1[1])],
            "delta_macro_auc": round(float(rand_metrics["macro_roc_auc"] - s2_metrics["macro_roc_auc"]), 4),
        },
    }

    print("\n--- Leakage Gap Summary ---")
    print(f"Random vs S1 (Cold Resume): +{gap_rand_vs_s1_acc * 100:.2f} percentage points accuracy [95% CI: {diff_s1_ci_acc[0]*100:.2f}%, {diff_s1_ci_acc[1]*100:.2f}%]")
    print(f"Random vs S2 (Cold JD):     +{gap_rand_vs_s2_acc * 100:.2f} percentage points accuracy [95% CI: {diff_s2_ci_acc[0]*100:.2f}%, {diff_s2_ci_acc[1]*100:.2f}%]")

    # Save artifacts
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    json_path = results_dir / "leakage_comparison_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[OK] Results written to {json_path}")

    oof_df = pd.DataFrame({
        "pair_id": np.arange(len(df)),
        "resume_id": df["resume_id"],
        "jd_id": df["jd_id"],
        "y_true": y,
        "rand_pred": rand_preds,
        "rand_prob_0": rand_probs[:, 0],
        "rand_prob_1": rand_probs[:, 1],
        "rand_prob_2": rand_probs[:, 2],
        "s1_pred": s1_preds,
        "s1_prob_0": s1_probs[:, 0],
        "s1_prob_1": s1_probs[:, 1],
        "s1_prob_2": s1_probs[:, 2],
        "s2_pred": s2_preds,
        "s2_prob_0": s2_probs[:, 0],
        "s2_prob_1": s2_probs[:, 1],
        "s2_prob_2": s2_probs[:, 2],
    })
    parquet_path = results_dir / "c5_leakage_oof_predictions.parquet"
    oof_df.to_parquet(parquet_path, index=False)
    print(f"[OK] OOF predictions written to {parquet_path}")

    # Generate markdown report
    report_md = f"""# Like-for-Like Leakage Comparison Report

## Overview
Evaluates the exact same **Tier 2 Dense Semantic** feature set (`[u, v, |u-v|, u*v, cos(u,v)]`) fitted with **LightGBM** (300 trees, lr=0.05, max_depth=6) across three evaluation protocols on C5 (8,000 resume-JD pairs):
1. **Random Stratified 5-Fold (Warm/Leaked):** Pairs randomly split across folds; allows both resumes and JDs to overlap between train and validation.
2. **S1 Cold-Resume 5-Fold (Grouped by Resume):** Resumes strictly partitioned; no resume seen in training appears in validation.
3. **S2 Cold-JD 5-Fold (Grouped by JD):** Job descriptions strictly partitioned; simulates deploying to new job postings.

---

## Measured Performance Comparison

| Protocol | Split Grouping | Accuracy (95% CI) | Macro-F1 (95% CI) | Macro-AUC |
|---|---|---|---|---|
| **Random Stratified** | None (Pair-level) | **{results['random_stratified']['accuracy']*100:.2f}%** [{results['random_stratified']['accuracy_ci95'][0]*100:.2f}%, {results['random_stratified']['accuracy_ci95'][1]*100:.2f}%] | **{results['random_stratified']['macro_f1']*100:.2f}%** [{results['random_stratified']['macro_f1_ci95'][0]*100:.2f}%, {results['random_stratified']['macro_f1_ci95'][1]*100:.2f}%] | **{results['random_stratified']['macro_roc_auc']:.4f}** |
| **S1 Cold-Resume** | `resume_id` (643 groups) | **{results['s1_cold_resume']['accuracy']*100:.2f}%** [{results['s1_cold_resume']['accuracy_ci95'][0]*100:.2f}%, {results['s1_cold_resume']['accuracy_ci95'][1]*100:.2f}%] | **{results['s1_cold_resume']['macro_f1']*100:.2f}%** [{results['s1_cold_resume']['macro_f1_ci95'][0]*100:.2f}%, {results['s1_cold_resume']['macro_f1_ci95'][1]*100:.2f}%] | **{results['s1_cold_resume']['macro_roc_auc']:.4f}** |
| **S2 Cold-JD** | `jd_id` (351 groups) | **{results['s2_cold_jd']['accuracy']*100:.2f}%** [{results['s2_cold_jd']['accuracy_ci95'][0]*100:.2f}%, {results['s2_cold_jd']['accuracy_ci95'][1]*100:.2f}%] | **{results['s2_cold_jd']['macro_f1']*100:.2f}%** [{results['s2_cold_jd']['macro_f1_ci95'][0]*100:.2f}%, {results['s2_cold_jd']['macro_f1_ci95'][1]*100:.2f}%] | **{results['s2_cold_jd']['macro_roc_auc']:.4f}** |

---

## Measured Leakage Gaps

| Comparison | $\\Delta$ Accuracy (95% Paired CI) | $\\Delta$ Macro-F1 (95% Paired CI) | $\\Delta$ Macro-AUC |
|---|---|---|---|
| **Random vs S1 (Cold Resume)** | **+{results['leakage_gaps']['random_vs_s1_cold_resume']['delta_accuracy']*100:.2f} pp** [{results['leakage_gaps']['random_vs_s1_cold_resume']['delta_accuracy_ci95'][0]*100:.2f} pp, {results['leakage_gaps']['random_vs_s1_cold_resume']['delta_accuracy_ci95'][1]*100:.2f} pp] | **+{results['leakage_gaps']['random_vs_s1_cold_resume']['delta_macro_f1']*100:.2f} pp** [{results['leakage_gaps']['random_vs_s1_cold_resume']['delta_macro_f1_ci95'][0]*100:.2f} pp, {results['leakage_gaps']['random_vs_s1_cold_resume']['delta_macro_f1_ci95'][1]*100:.2f} pp] | **+{results['leakage_gaps']['random_vs_s1_cold_resume']['delta_macro_auc']:.4f}** |
| **Random vs S2 (Cold JD)** | **+{results['leakage_gaps']['random_vs_s2_cold_jd']['delta_accuracy']*100:.2f} pp** [{results['leakage_gaps']['random_vs_s2_cold_jd']['delta_accuracy_ci95'][0]*100:.2f} pp, {results['leakage_gaps']['random_vs_s2_cold_jd']['delta_accuracy_ci95'][1]*100:.2f} pp] | **+{results['leakage_gaps']['random_vs_s2_cold_jd']['delta_macro_f1']*100:.2f} pp** [{results['leakage_gaps']['random_vs_s2_cold_jd']['delta_macro_f1_ci95'][0]*100:.2f} pp, {results['leakage_gaps']['random_vs_s2_cold_jd']['delta_macro_f1_ci95'][1]*100:.2f} pp] | **+{results['leakage_gaps']['random_vs_s2_cold_jd']['delta_macro_auc']:.4f}** |

### Key Findings
1. Random pair splitting inflates accuracy by **{results['leakage_gaps']['random_vs_s1_cold_resume']['delta_accuracy']*100:.2f} percentage points** compared to Cold-Resume evaluation (S1), and by **{results['leakage_gaps']['random_vs_s2_cold_jd']['delta_accuracy']*100:.2f} percentage points** compared to Cold-JD evaluation (S2).
2. The measured leakage gap replaces any hypothetical estimates with directly audited, like-for-like experimental measurements.
"""
    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / "LEAKAGE_COMPARISON.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_md)
    print(f"[OK] Report written to {report_path}")


if __name__ == "__main__":
    main()
