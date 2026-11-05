import os
import sys
import json
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    balanced_accuracy_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize
import lightgbm as lgb
from datasets import load_dataset

from src.ats.adapters.registry import get_adapter
from src.ats.adapters.base import normalize_text, compute_sha1
from src.ats.embeddings import get_cached_embeddings
from src.ats.modeling import build_tfidf_features


def compute_comprehensive_metrics(y_true, y_pred, y_prob):
    """
    Computes primary and secondary metrics:
    Primary: macro_f1, macro_roc_auc
    Secondary: accuracy, balanced_accuracy, per_class_recall, binary_good_vs_nofit_auc
    """
    classes = [0, 1, 2] # 0: Good Fit, 1: No Fit, 2: Potential Fit (alphabetic)
    # y_true and y_pred are integer encoded
    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    
    # Per-class recall
    recalls = recall_score(y_true, y_pred, average=None, zero_division=0)
    per_class_rec = {f"class_{c}_recall": float(recalls[c]) if c < len(recalls) else 0.0 for c in classes}
    
    # Macro ROC-AUC (one-vs-rest)
    try:
        y_true_bin = label_binarize(y_true, classes=classes)
        macro_auc = float(roc_auc_score(y_true_bin, y_prob, average="macro", multi_class="ovr"))
    except Exception:
        macro_auc = None

    # Binary Good Fit vs No Fit AUC (Potential Fit dropped)
    # Find indices for Good Fit and No Fit
    # In alphabetical order: classes might be 0: Good Fit, 1: No Fit, 2: Potential Fit
    good_fit_idx = 0
    no_fit_idx = 1
    # Check binary mask
    binary_mask = (y_true == good_fit_idx) | (y_true == no_fit_idx)
    if binary_mask.sum() > 10 and len(np.unique(y_true[binary_mask])) > 1:
        y_bin = (y_true[binary_mask] == good_fit_idx).astype(int)
        # Prob of good fit normalized over (p_good + p_nofit)
        p_good = y_prob[binary_mask, good_fit_idx]
        p_nofit = y_prob[binary_mask, no_fit_idx]
        denom = p_good + p_nofit + 1e-9
        p_bin = p_good / denom
        try:
            bin_auc = float(roc_auc_score(y_bin, p_bin))
        except Exception:
            bin_auc = None
    else:
        bin_auc = None

    majority_bl = float(pd.Series(y_true).value_counts(normalize=True).max())

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "balanced_accuracy": bal_acc,
        "macro_roc_auc": macro_auc,
        "binary_good_vs_nofit_auc": bin_auc,
        "majority_baseline": majority_bl,
        **per_class_rec,
    }


def cluster_bootstrap_eval(df_eval, cluster_col, metric_fn, n_boot=500, seed=42):
    """Cluster bootstrap confidence interval."""
    rng = np.random.RandomState(seed)
    clusters = df_eval[cluster_col].unique()
    n_clusters = len(clusters)
    
    cluster_indices = df_eval.groupby(cluster_col).indices
    
    scores = []
    for _ in range(n_boot):
        sampled_clusters = rng.choice(clusters, size=n_clusters, replace=True)
        boot_idx = []
        for c in sampled_clusters:
            boot_idx.extend(cluster_indices[c])
        sub_df = df_eval.iloc[boot_idx]
        try:
            score = metric_fn(sub_df)
            if score is not None and not np.isnan(score):
                scores.append(score)
        except Exception:
            continue
            
    if len(scores) < 20:
        return None, None
    low = float(np.percentile(scores, 2.5))
    high = float(np.percentile(scores, 97.5))
    return low, high


def train_eval_lgbm_scenario(X_train, y_train, X_test, y_test, num_classes=3, seed=42):
    """Train multiclass LightGBM and evaluate."""
    clf = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=num_classes,
        n_estimators=100,
        random_state=seed,
        verbose=-1,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    probs = clf.predict_proba(X_test)
    preds = clf.predict(X_test)
    return clf, preds, probs


def run_s1_cold_resume(df, X_features, class_to_idx, seed=42):
    """S1: GroupKFold on resume_id (5 splits)."""
    print("\n--- Running S1: Cold-Resume (GroupKFold on resume_id) ---", flush=True)
    gkf = GroupKFold(n_splits=5)
    y = np.array([class_to_idx[c] for c in df["label_class"]])
    n_samples = len(df)
    oof_preds = np.zeros(n_samples, dtype=int)
    oof_probs = np.zeros((n_samples, len(class_to_idx)))
    
    for fold, (tr_idx, val_idx) in enumerate(gkf.split(df, y, groups=df["resume_id"])):
        clf, preds, probs = train_eval_lgbm_scenario(
            X_features[tr_idx], y[tr_idx], X_features[val_idx], y[val_idx], num_classes=len(class_to_idx), seed=seed
        )
        oof_preds[val_idx] = preds
        oof_probs[val_idx] = probs

    metrics = compute_comprehensive_metrics(y, oof_preds, oof_probs)
    
    df_eval = df.copy()
    df_eval["y_true"] = y
    df_eval["y_pred"] = oof_preds
    for c in range(len(class_to_idx)):
        df_eval[f"prob_{c}"] = oof_probs[:, c]

    # Bootstrap CIs
    ci_acc = cluster_bootstrap_eval(df_eval, "resume_id", lambda d: accuracy_score(d["y_true"], d["y_pred"]))
    ci_f1 = cluster_bootstrap_eval(df_eval, "resume_id", lambda d: f1_score(d["y_true"], d["y_pred"], average="macro", zero_division=0))
    ci_auc = cluster_bootstrap_eval(
        df_eval, "resume_id",
        lambda d: roc_auc_score(
            label_binarize(d["y_true"], classes=[0, 1, 2]),
            d[[f"prob_{c}" for c in range(3)]].values,
            average="macro", multi_class="ovr"
        )
    )

    metrics["ci_95_accuracy"] = ci_acc
    metrics["ci_95_macro_f1"] = ci_f1
    metrics["ci_95_macro_roc_auc"] = ci_auc
    return metrics, df_eval


def run_s2_cold_jd(df, X_features, class_to_idx, seed=42):
    """S2: GroupKFold on jd_id (5 splits)."""
    print("\n--- Running S2: Cold-JD (GroupKFold on jd_id) ---", flush=True)
    gkf = GroupKFold(n_splits=5)
    y = np.array([class_to_idx[c] for c in df["label_class"]])
    n_samples = len(df)
    oof_preds = np.zeros(n_samples, dtype=int)
    oof_probs = np.zeros((n_samples, len(class_to_idx)))
    
    for fold, (tr_idx, val_idx) in enumerate(gkf.split(df, y, groups=df["jd_id"])):
        clf, preds, probs = train_eval_lgbm_scenario(
            X_features[tr_idx], y[tr_idx], X_features[val_idx], y[val_idx], num_classes=len(class_to_idx), seed=seed
        )
        oof_preds[val_idx] = preds
        oof_probs[val_idx] = probs

    metrics = compute_comprehensive_metrics(y, oof_preds, oof_probs)
    
    df_eval = df.copy()
    df_eval["y_true"] = y
    df_eval["y_pred"] = oof_preds
    for c in range(len(class_to_idx)):
        df_eval[f"prob_{c}"] = oof_probs[:, c]

    ci_acc = cluster_bootstrap_eval(df_eval, "jd_id", lambda d: accuracy_score(d["y_true"], d["y_pred"]))
    ci_f1 = cluster_bootstrap_eval(df_eval, "jd_id", lambda d: f1_score(d["y_true"], d["y_pred"], average="macro", zero_division=0))
    ci_auc = cluster_bootstrap_eval(
        df_eval, "jd_id",
        lambda d: roc_auc_score(
            label_binarize(d["y_true"], classes=[0, 1, 2]),
            d[[f"prob_{c}" for c in range(3)]].values,
            average="macro", multi_class="ovr"
        )
    )

    metrics["ci_95_accuracy"] = ci_acc
    metrics["ci_95_macro_f1"] = ci_f1
    metrics["ci_95_macro_roc_auc"] = ci_auc
    return metrics, df_eval


def run_s3_double_cold(df, X_features, class_to_idx, seeds=[42, 43, 44, 45, 46]):
    """
    S3: Double-Cold Protocol.
    Hold out 15% unique resumes AND 15% unique JDs.
    Train only on pairs with neither.
    Test on pairs with both.
    Evaluated across multiple seeds.
    """
    print("\n--- Running S3: Double-Cold (Disjoint Resumes AND Disjoint JDs) ---", flush=True)
    all_resumes = np.array(df["resume_id"].unique())
    all_jds = np.array(df["jd_id"].unique())
    y = np.array([class_to_idx[c] for c in df["label_class"]])
    
    seed_results = []
    
    for s in seeds:
        rng = np.random.RandomState(s)
        test_res_count = int(np.ceil(0.15 * len(all_resumes)))
        test_jd_count = int(np.ceil(0.15 * len(all_jds)))
        
        test_resumes = set(rng.choice(all_resumes, size=test_res_count, replace=False))
        test_jds = set(rng.choice(all_jds, size=test_jd_count, replace=False))
        
        is_test = df["resume_id"].isin(test_resumes) & df["jd_id"].isin(test_jds)
        is_train = (~df["resume_id"].isin(test_resumes)) & (~df["jd_id"].isin(test_jds))
        
        train_idx = np.where(is_train)[0]
        test_idx = np.where(is_test)[0]
        
        n_tr = len(train_idx)
        n_te = len(test_idx)
        
        if n_te < 10:
            print(f"Warning: Seed {s} has only {n_te} test pairs. Skipping seed.", flush=True)
            continue
            
        clf, preds, probs = train_eval_lgbm_scenario(
            X_features[train_idx], y[train_idx], X_features[test_idx], y[test_idx], num_classes=len(class_to_idx), seed=s
        )
        
        m = compute_comprehensive_metrics(y[test_idx], preds, probs)
        m["n_train_pairs"] = n_tr
        m["n_test_pairs"] = n_te
        m["seed"] = s
        seed_results.append(m)
        print(f"  Seed {s}: n_tr={n_tr}, n_te={n_te} | Acc={m['accuracy']:.4f}, Macro-F1={m['macro_f1']:.4f}, AUC={m['macro_roc_auc']:.4f}", flush=True)

    # Average metrics across seeds
    avg_metrics = {}
    for k in seed_results[0].keys():
        if k in ["seed"]:
            continue
        vals = [r[k] for r in seed_results if r[k] is not None]
        avg_metrics[k] = float(np.mean(vals)) if len(vals) > 0 else None
        avg_metrics[f"{k}_std"] = float(np.std(vals)) if len(vals) > 0 else None

    # Compute CI from seed distribution
    avg_metrics["ci_95_accuracy"] = [float(np.percentile([r["accuracy"] for r in seed_results], 2.5)), float(np.percentile([r["accuracy"] for r in seed_results], 97.5))]
    avg_metrics["ci_95_macro_f1"] = [float(np.percentile([r["macro_f1"] for r in seed_results], 2.5)), float(np.percentile([r["macro_f1"] for r in seed_results], 97.5))]
    auc_vals = [r["macro_roc_auc"] for r in seed_results if r["macro_roc_auc"] is not None]
    avg_metrics["ci_95_macro_roc_auc"] = [float(np.percentile(auc_vals, 2.5)), float(np.percentile(auc_vals, 97.5))] if len(auc_vals) > 0 else None
    avg_metrics["individual_seed_runs"] = seed_results
    return avg_metrics


def run_ablations(df, res_emb, jd_emb, X_full, class_to_idx, seed=42):
    """
    Ablations on S3 and S1:
    A1: resume-only features
    A2: JD-only features
    A3: both
    A4: JD-swap test
    """
    print("\n--- Running Feature & Sensitivity Ablations (A1 - A4) ---", flush=True)
    y = np.array([class_to_idx[c] for c in df["label_class"]])
    
    # Define S3 train/test split for seed 42
    all_resumes = np.array(df["resume_id"].unique())
    all_jds = np.array(df["jd_id"].unique())
    rng = np.random.RandomState(seed)
    test_res_count = int(np.ceil(0.15 * len(all_resumes)))
    test_jd_count = int(np.ceil(0.15 * len(all_jds)))
    test_resumes = set(rng.choice(all_resumes, size=test_res_count, replace=False))
    test_jds = set(rng.choice(all_jds, size=test_jd_count, replace=False))
    
    is_test = df["resume_id"].isin(test_resumes) & df["jd_id"].isin(test_jds)
    is_train = (~df["resume_id"].isin(test_resumes)) & (~df["jd_id"].isin(test_jds))
    train_idx = np.where(is_train)[0]
    test_idx = np.where(is_test)[0]
    
    # Feature sets
    feature_sets = {
        "A1_resume_only": res_emb,
        "A2_jd_only": jd_emb,
        "A3_both_full": X_full,
    }
    
    ablation_results = {}
    models = {}
    
    for feat_name, X_mat in feature_sets.items():
        clf, preds, probs = train_eval_lgbm_scenario(
            X_mat[train_idx], y[train_idx], X_mat[test_idx], y[test_idx], num_classes=len(class_to_idx), seed=seed
        )
        m = compute_comprehensive_metrics(y[test_idx], preds, probs)
        ablation_results[feat_name] = m
        models[feat_name] = clf
        print(f"  {feat_name}: Acc={m['accuracy']:.4f}, Macro-F1={m['macro_f1']:.4f}, AUC={m['macro_roc_auc']:.4f}", flush=True)

    # A4: JD-Swap Sensitivity Test
    print("\n--- Running A4: JD-Swap Sensitivity Test ---", flush=True)
    # On the test set of S3, take each test row and swap its JD with a different test JD having a different Good Fit rate
    df_test = df.iloc[test_idx].copy().reset_index(drop=True)
    y_test = y[test_idx]
    
    # Compute JD label rates in df
    jd_rates = df.groupby("jd_id")["label_class"].apply(lambda s: (s == "Good Fit").mean()).to_dict()
    df_test["orig_jd_rate"] = df_test["jd_id"].map(jd_rates)
    
    # For each row, find another JD in test set with different rate (|rate1 - rate2| > 0.20 or max difference)
    unique_test_jds = df_test["jd_id"].unique()
    swapped_jd_ids = []
    
    for i, row in df_test.iterrows():
        orig_rate = row["orig_jd_rate"]
        # candidate JDs with different rate
        diffs = [abs(orig_rate - jd_rates.get(jid, 0.5)) for jid in unique_test_jds if jid != row["jd_id"]]
        if len(diffs) > 0:
            best_other_jd = [jid for jid in unique_test_jds if jid != row["jd_id"]][np.argmax(diffs)]
        else:
            best_other_jd = row["jd_id"]
        swapped_jd_ids.append(best_other_jd)
        
    df_test["swapped_jd_id"] = swapped_jd_ids
    
    # Extract swapped embeddings
    # Build swapped full feature matrix for test set
    orig_res = res_emb[test_idx]
    # Map swapped jd ids to embedding
    jd_id_to_idx = {jid: i for i, jid in enumerate(df["jd_id"])}
    swapped_jd_emb = np.array([jd_emb[jd_id_to_idx[jid]] for jid in swapped_jd_ids])
    
    diff_swapped = np.abs(orig_res - swapped_jd_emb)
    mult_swapped = orig_res * swapped_jd_emb
    cos_swapped = np.sum(orig_res * swapped_jd_emb, axis=1, keepdims=True)
    X_test_swapped = np.hstack([orig_res, swapped_jd_emb, diff_swapped, mult_swapped, cos_swapped])
    
    # Evaluate model A3 on swapped test set
    full_model = models["A3_both_full"]
    orig_probs = full_model.predict_proba(X_full[test_idx])
    orig_preds = full_model.predict(X_full[test_idx])
    
    swapped_probs = full_model.predict_proba(X_test_swapped)
    swapped_preds = full_model.predict(X_test_swapped)
    
    pred_change_rate = float(np.mean(orig_preds != swapped_preds))
    mean_prob_shift = float(np.mean(np.sum(np.abs(orig_probs - swapped_probs), axis=1)))
    orig_acc = float(accuracy_score(y_test, orig_preds))
    swapped_acc = float(accuracy_score(y_test, swapped_preds))
    acc_drop = float(orig_acc - swapped_acc)
    
    swap_results = {
        "prediction_flip_rate": pred_change_rate,
        "mean_l1_prob_shift": mean_prob_shift,
        "original_accuracy": orig_acc,
        "swapped_accuracy": swapped_acc,
        "accuracy_drop": acc_drop,
        "model_ignores_jd": bool(pred_change_rate < 0.05),
    }
    print(f"  JD-Swap Test Results:")
    print(f"    Predictions flipped: {pred_change_rate * 100:.2f}%")
    print(f"    Mean L1 probability shift: {mean_prob_shift:.4f}")
    print(f"    Accuracy drop: {orig_acc:.4f} -> {swapped_acc:.4f} (drop: {acc_drop:.4f})")
    print(f"    Verdict: {'IGNORES JD' if swap_results['model_ignores_jd'] else 'STRONGLY SENSITIVE TO JD'}")

    return ablation_results, swap_results


def run_a5_label_base_rates(df):
    """
    A5: Label base rates:
    - Per-JD and per-resume class rates (histogram summary, top/bottom 10 JDs)
    - Out-of-fold JD prior-only baseline
    """
    print("\n--- Running A5: Label Base Rates & JD Prior Baseline ---", flush=True)
    # Per-JD Good Fit rate
    jd_stats = df.groupby("jd_id")["label_class"].agg(
        count="count",
        good_fit_rate=lambda s: (s == "Good Fit").mean(),
        no_fit_rate=lambda s: (s == "No Fit").mean(),
        potential_fit_rate=lambda s: (s == "Potential Fit").mean()
    )
    
    # Per-resume stats
    res_stats = df.groupby("resume_id")["label_class"].agg(
        count="count",
        good_fit_rate=lambda s: (s == "Good Fit").mean(),
        no_fit_rate=lambda s: (s == "No Fit").mean(),
        potential_fit_rate=lambda s: (s == "Potential Fit").mean()
    )
    
    top_10_jds = jd_stats.sort_values(by="good_fit_rate", ascending=False).head(10).to_dict(orient="index")
    bottom_10_jds = jd_stats.sort_values(by="good_fit_rate", ascending=True).head(10).to_dict(orient="index")
    
    # JD prior-only baseline: for each row, predict majority class of that JD using out-of-row counts
    # If a JD only has 1 row or unseen, predict dataset majority class ("No Fit")
    dataset_majority = df["label_class"].mode()[0]
    
    oof_prior_preds = []
    for i, row in df.iterrows():
        jid = row["jd_id"]
        # other rows with same JD
        other_labels = df[(df["jd_id"] == jid) & (df.index != i)]["label_class"]
        if len(other_labels) > 0:
            pred = other_labels.mode()[0]
        else:
            pred = dataset_majority
        oof_prior_preds.append(pred)
        
    prior_acc = float(accuracy_score(df["label_class"], oof_prior_preds))
    prior_f1 = float(f1_score(df["label_class"], oof_prior_preds, average="macro", zero_division=0))
    print(f"  JD Prior-Only Baseline: Accuracy={prior_acc:.4f}, Macro-F1={prior_f1:.4f}")
    
    return {
        "jd_good_fit_rate_mean": float(jd_stats["good_fit_rate"].mean()),
        "jd_good_fit_rate_std": float(jd_stats["good_fit_rate"].std()),
        "jd_good_fit_rate_min": float(jd_stats["good_fit_rate"].min()),
        "jd_good_fit_rate_max": float(jd_stats["good_fit_rate"].max()),
        "res_good_fit_rate_mean": float(res_stats["good_fit_rate"].mean()),
        "res_good_fit_rate_std": float(res_stats["good_fit_rate"].std()),
        "prior_baseline_accuracy": prior_acc,
        "prior_baseline_macro_f1": prior_f1,
        "top_10_jds": top_10_jds,
        "bottom_10_jds": bottom_10_jds,
    }


def run_a6_official_split_leakage_check():
    """
    A6: Exact and near-duplicate check between official train and test splits of C5.
    """
    print("\n--- Running A6: Official Train/Test Split Leakage Check ---", flush=True)
    ds = load_dataset("cnamuangtoun/resume-job-description-fit")
    df_tr = ds["train"].to_pandas()
    df_te = ds["test"].to_pandas()
    
    # Compute hashes
    tr_res_hashes = df_tr["resume_text"].astype(str).apply(compute_sha1)
    tr_jd_hashes = df_tr["job_description_text"].astype(str).apply(compute_sha1)
    tr_pairs = set(zip(tr_res_hashes, tr_jd_hashes))
    
    te_res_hashes = df_te["resume_text"].astype(str).apply(compute_sha1)
    te_jd_hashes = df_te["job_description_text"].astype(str).apply(compute_sha1)
    te_pairs = set(zip(te_res_hashes, te_jd_hashes))
    
    res_overlap = len(set(te_res_hashes).intersection(set(tr_res_hashes)))
    jd_overlap = len(set(te_jd_hashes).intersection(set(tr_jd_hashes)))
    pair_overlap = len(te_pairs.intersection(tr_pairs))
    
    pct_test_res_in_tr = (res_overlap / len(set(te_res_hashes))) * 100
    pct_test_jd_in_tr = (jd_overlap / len(set(te_jd_hashes))) * 100
    
    print(f"  Official Train ({len(df_tr)} rows) vs Test ({len(df_te)} rows):")
    print(f"    Resumes overlapping: {res_overlap} / {len(set(te_res_hashes))} test resumes ({pct_test_res_in_tr:.2f}%)")
    print(f"    JDs overlapping: {jd_overlap} / {len(set(te_jd_hashes))} test JDs ({pct_test_jd_in_tr:.2f}%)")
    print(f"    Exact pairs overlapping: {pair_overlap}")
    
    return {
        "train_rows": len(df_tr),
        "test_rows": len(df_te),
        "test_unique_resumes": len(set(te_res_hashes)),
        "test_unique_jds": len(set(te_jd_hashes)),
        "test_resumes_in_train": res_overlap,
        "test_resumes_in_train_pct": pct_test_res_in_tr,
        "test_jds_in_train": jd_overlap,
        "test_jds_in_train_pct": pct_test_jd_in_tr,
        "exact_pairs_overlap": pair_overlap,
    }


def main():
    print("=======================================================", flush=True)
    print("   PRE-REGISTERED C5 VERIFICATION GATE SUITE", flush=True)
    print("=======================================================", flush=True)
    print("Pre-registered Primary Metrics: Macro-F1, Macro ROC-AUC (one-vs-rest)")
    print("Secondary Metrics: Accuracy, Balanced Accuracy, Per-class Recall, Binary Good-vs-No-Fit AUC")
    
    # 1. Load C5
    adapter = get_adapter("c5")
    df = adapter.load()
    print(f"Loaded C5: {len(df)} rows, {df['resume_id'].nunique()} unique resumes, {df['jd_id'].nunique()} unique JDs", flush=True)
    
    # Class mapping
    classes = sorted(df["label_class"].unique()) # ['Good Fit', 'No Fit', 'Potential Fit']
    class_to_idx = {c: i for i, c in enumerate(classes)}
    print(f"Class mapping: {class_to_idx}")
    
    # 2. Features
    print("Loading cached MiniLM embeddings...", flush=True)
    res_emb, jd_emb = get_cached_embeddings(df, dataset_name="c5")
    diff = np.abs(res_emb - jd_emb)
    mult = res_emb * jd_emb
    cos = np.sum(res_emb * jd_emb, axis=1, keepdims=True)
    X_minilm_full = np.hstack([res_emb, jd_emb, diff, mult, cos])
    print(f"Full feature matrix shape: {X_minilm_full.shape}", flush=True)
    
    # 3. Split Scenarios
    s1_metrics, df_s1_eval = run_s1_cold_resume(df, X_minilm_full, class_to_idx, seed=42)
    s2_metrics, df_s2_eval = run_s2_cold_jd(df, X_minilm_full, class_to_idx, seed=42)
    s3_metrics = run_s3_double_cold(df, X_minilm_full, class_to_idx, seeds=[42, 43, 44, 45, 46])
    
    # 4. Ablations
    ablation_metrics, swap_metrics = run_ablations(df, res_emb, jd_emb, X_minilm_full, class_to_idx, seed=42)
    a5_metrics = run_a5_label_base_rates(df)
    a6_metrics = run_a6_official_split_leakage_check()
    
    # 5. Compile and Save Results
    verification_results = {
        "dataset": "cnamuangtoun/resume-job-description-fit",
        "pre_registered_primary_metrics": ["macro_f1", "macro_roc_auc"],
        "class_mapping": class_to_idx,
        "scenarios": {
            "S1_cold_resume": s1_metrics,
            "S2_cold_jd": s2_metrics,
            "S3_double_cold": s3_metrics,
        },
        "ablations": ablation_metrics,
        "a4_jd_swap": swap_metrics,
        "a5_base_rates": a5_metrics,
        "a6_official_split_leakage": a6_metrics,
    }
    
    os.makedirs("reports", exist_ok=True)
    with open("reports/c5_verification_results.json", "w") as f:
        json.dump(verification_results, f, indent=2)
    print("\nSaved full verification results to reports/c5_verification_results.json", flush=True)


if __name__ == "__main__":
    main()

