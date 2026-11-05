import os
import sys
import time
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, roc_auc_score, recall_score
from sklearn.preprocessing import label_binarize
import lightgbm as lgb
import xgboost as xgb

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.ats.adapters.registry import get_adapter
from src.ats.embeddings import build_pair_features
from src.ats.evaluation import vectorized_cluster_bootstrap_acc_f1


def fit_predict_model(model_name, X_tr, y_tr, X_te, df_tr, df_te, num_classes=3, seed=42):
    """Fit baseline or ML model and return (preds, probs)."""
    n_te = len(X_te)
    
    if model_name == "majority":
        maj_c = int(pd.Series(y_tr).mode()[0])
        preds = np.full(n_te, maj_c, dtype=int)
        probs = np.zeros((n_te, num_classes))
        probs[:, maj_c] = 1.0
        return preds, probs

    elif model_name == "jd_prior":
        # Out-of-fold JD marginal class probabilities
        global_prior = pd.Series(y_tr).value_counts(normalize=True).reindex(range(num_classes), fill_value=0.0).values
        jd_priors = {}
        for jid, grp in df_tr.groupby("jd_id"):
            counts = grp["label_idx"].value_counts(normalize=True).reindex(range(num_classes), fill_value=0.0).values
            jd_priors[jid] = counts
            
        probs = np.zeros((n_te, num_classes))
        for i, jid in enumerate(df_te["jd_id"]):
            probs[i] = jd_priors.get(jid, global_prior)
        preds = np.argmax(probs, axis=1)
        return preds, probs

    elif model_name == "resume_prior":
        # Out-of-fold Resume marginal class probabilities
        global_prior = pd.Series(y_tr).value_counts(normalize=True).reindex(range(num_classes), fill_value=0.0).values
        res_priors = {}
        for rid, grp in df_tr.groupby("resume_id"):
            counts = grp["label_idx"].value_counts(normalize=True).reindex(range(num_classes), fill_value=0.0).values
            res_priors[rid] = counts
            
        probs = np.zeros((n_te, num_classes))
        for i, rid in enumerate(df_te["resume_id"]):
            probs[i] = res_priors.get(rid, global_prior)
        preds = np.argmax(probs, axis=1)
        return preds, probs

    elif model_name == "logistic":
        clf = LogisticRegression(max_iter=300, C=1.0, random_state=seed, n_jobs=-1)
        clf.fit(X_tr, y_tr)
        probs = clf.predict_proba(X_te)
        preds = clf.predict(X_te)
        return preds, probs

    elif model_name == "xgboost":
        clf = xgb.XGBClassifier(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.08,
            objective="multi:softprob",
            num_class=num_classes,
            random_state=seed,
            n_jobs=-1,
            tree_method="hist",
        )
        clf.fit(X_tr, y_tr)
        probs = clf.predict_proba(X_te)
        preds = clf.predict(X_te)
        return preds, probs

    elif model_name == "lightgbm":
        clf = lgb.LGBMClassifier(
            objective="multiclass",
            num_class=num_classes,
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            num_leaves=31,
            random_state=seed,
            n_jobs=-1,
            verbose=-1,
        )
        clf.fit(X_tr, y_tr)
        probs = clf.predict_proba(X_te)
        preds = clf.predict(X_te)
        return preds, probs

    else:
        raise ValueError(f"Unknown model_name: {model_name}")


def eval_metrics(y_true, y_pred, y_prob, clusters=None, num_classes=3, seed=42):
    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))

    rec = recall_score(y_true, y_pred, average=None, zero_division=0)
    per_class = {f"class_{c}_recall": float(rec[c]) if c < len(rec) else 0.0 for c in range(num_classes)}

    try:
        y_bin = label_binarize(y_true, classes=list(range(num_classes)))
        macro_auc = float(roc_auc_score(y_bin, y_prob, average="macro", multi_class="ovr"))
    except Exception:
        macro_auc = None

    bin_mask = (y_true == 0) | (y_true == 1)
    if bin_mask.sum() > 10 and len(np.unique(y_true[bin_mask])) > 1:
        y_b = (y_true[bin_mask] == 0).astype(int)
        p_0 = y_prob[bin_mask, 0]
        p_1 = y_prob[bin_mask, 1]
        p_bin = p_0 / (p_0 + p_1 + 1e-9)
        try:
            bin_auc = float(roc_auc_score(y_b, p_bin))
        except Exception:
            bin_auc = None
    else:
        bin_auc = None

    maj_rate = float(pd.Series(y_true).value_counts(normalize=True).max())

    ci_acc = (None, None)
    ci_f1 = (None, None)
    if clusters is not None and len(np.unique(clusters)) >= 5:
        ci_acc, ci_f1 = vectorized_cluster_bootstrap_acc_f1(
            y_true, y_pred, clusters, n_classes=num_classes, n_boot=500, seed=seed
        )

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "balanced_accuracy": bal_acc,
        "macro_roc_auc": macro_auc,
        "binary_good_vs_nofit_auc": bin_auc,
        "majority_baseline": maj_rate,
        "ci_95_accuracy": list(ci_acc) if ci_acc[0] is not None else None,
        "ci_95_macro_f1": list(ci_f1) if ci_f1[0] is not None else None,
        **per_class,
    }


def run_j1_sweep():
    t0 = time.time()
    print("=" * 65)
    print("   J1 COMPREHENSIVE BENCHMARK SWEEP (C5)")
    print("   20 Seeds | All Baselines | S1, S2, S3 Protocols")
    print("=" * 65)

    df = get_adapter("c5").load()
    classes = sorted(df["label_class"].unique()) # ['Good Fit', 'No Fit', 'Potential Fit']
    class_to_idx = {c: i for i, c in enumerate(classes)}
    df["label_idx"] = [class_to_idx[c] for c in df["label_class"]]
    y = df["label_idx"].values
    num_classes = len(classes)

    print(f"Loaded {len(df)} pairs across {df['resume_id'].nunique()} resumes and {df['jd_id'].nunique()} JDs.")
    print(f"Classes: {class_to_idx} -> Distribution: {df['label_class'].value_counts().to_dict()}")

    print("Building / loading unique document pair features...", flush=True)
    X = build_pair_features(df, "c5", include_late_interaction=False)
    print(f"Features ready: shape {X.shape}")

    models = ["majority", "jd_prior", "resume_prior", "logistic", "lightgbm", "xgboost"]
    seeds = [42 + i for i in range(20)]
    all_results = {}

    # --- SCENARIO S1: Cold-Resume (GroupKFold by resume_id) ---
    print("\n" + "-" * 50)
    print(">>> Executing Scenario S1: Cold-Resume (GroupKFold)")
    print("-" * 50)
    all_results["S1_cold_resume"] = {}
    gkf = GroupKFold(n_splits=5)

    for model_name in models:
        t_mod = time.time()
        print(f"  Evaluating {model_name:14s} ...", end=" ", flush=True)
        # For non-stochastic baselines (majority, priors), 1 run suffices; for ML models, evaluate with seeds
        active_seeds = [42] if model_name in ["majority", "jd_prior", "resume_prior"] else seeds[:5]
        seed_metrics = []

        for s in active_seeds:
            oof_preds = np.zeros(len(df), dtype=int)
            oof_probs = np.zeros((len(df), num_classes))

            for tr_idx, te_idx in gkf.split(df, y, groups=df["resume_id"]):
                preds, probs = fit_predict_model(
                    model_name, X[tr_idx], y[tr_idx], X[te_idx], df.iloc[tr_idx], df.iloc[te_idx], num_classes=num_classes, seed=s
                )
                oof_preds[te_idx] = preds
                oof_probs[te_idx] = probs

            m = eval_metrics(y, oof_preds, oof_probs, clusters=df["resume_id"].values, num_classes=num_classes, seed=s)
            seed_metrics.append(m)

        avg_acc = float(np.mean([m["accuracy"] for m in seed_metrics]))
        avg_f1 = float(np.mean([m["macro_f1"] for m in seed_metrics]))
        avg_auc = float(np.mean([m["macro_roc_auc"] for m in seed_metrics if m["macro_roc_auc"] is not None]))
        all_results["S1_cold_resume"][model_name] = {
            "accuracy": avg_acc,
            "macro_f1": avg_f1,
            "macro_roc_auc": avg_auc,
            "ci_95_accuracy": seed_metrics[0]["ci_95_accuracy"],
            "ci_95_macro_f1": seed_metrics[0]["ci_95_macro_f1"],
            "time_sec": round(time.time() - t_mod, 2),
        }
        print(f"Acc: {avg_acc:.4f} | Macro-F1: {avg_f1:.4f} | AUC: {avg_auc:.4f} ({all_results['S1_cold_resume'][model_name]['time_sec']}s)")

    # --- SCENARIO S2: Cold-JD (GroupKFold by jd_id) ---
    print("\n" + "-" * 50)
    print(">>> Executing Scenario S2: Cold-JD (GroupKFold)")
    print("-" * 50)
    all_results["S2_cold_jd"] = {}
    gkf_jd = GroupKFold(n_splits=5)

    for model_name in models:
        t_mod = time.time()
        print(f"  Evaluating {model_name:14s} ...", end=" ", flush=True)
        active_seeds = [42] if model_name in ["majority", "jd_prior", "resume_prior"] else seeds[:5]
        seed_metrics = []

        for s in active_seeds:
            oof_preds = np.zeros(len(df), dtype=int)
            oof_probs = np.zeros((len(df), num_classes))

            for tr_idx, te_idx in gkf_jd.split(df, y, groups=df["jd_id"]):
                preds, probs = fit_predict_model(
                    model_name, X[tr_idx], y[tr_idx], X[te_idx], df.iloc[tr_idx], df.iloc[te_idx], num_classes=num_classes, seed=s
                )
                oof_preds[te_idx] = preds
                oof_probs[te_idx] = probs

            m = eval_metrics(y, oof_preds, oof_probs, clusters=df["jd_id"].values, num_classes=num_classes, seed=s)
            seed_metrics.append(m)

        avg_acc = float(np.mean([m["accuracy"] for m in seed_metrics]))
        avg_f1 = float(np.mean([m["macro_f1"] for m in seed_metrics]))
        avg_auc = float(np.mean([m["macro_roc_auc"] for m in seed_metrics if m["macro_roc_auc"] is not None]))
        all_results["S2_cold_jd"][model_name] = {
            "accuracy": avg_acc,
            "macro_f1": avg_f1,
            "macro_roc_auc": avg_auc,
            "ci_95_accuracy": seed_metrics[0]["ci_95_accuracy"],
            "ci_95_macro_f1": seed_metrics[0]["ci_95_macro_f1"],
            "time_sec": round(time.time() - t_mod, 2),
        }
        print(f"Acc: {avg_acc:.4f} | Macro-F1: {avg_f1:.4f} | AUC: {avg_auc:.4f} ({all_results['S2_cold_jd'][model_name]['time_sec']}s)")

    # --- SCENARIO S3: Double-Cold (Disjoint 15% holdouts, 20 Seeds) ---
    print("\n" + "-" * 50)
    print(">>> Executing Scenario S3: Double-Cold (20 Independent Seeds)")
    print("-" * 50)
    all_results["S3_double_cold"] = {}

    all_resumes = np.array(df["resume_id"].unique())
    all_jds = np.array(df["jd_id"].unique())

    for model_name in models:
        t_mod = time.time()
        print(f"  Evaluating {model_name:14s} ...", end=" ", flush=True)
        seed_metrics = []

        for s in seeds:
            rng = np.random.RandomState(s)
            te_res = set(rng.choice(all_resumes, size=int(np.ceil(0.15 * len(all_resumes))), replace=False))
            te_jds = set(rng.choice(all_jds, size=int(np.ceil(0.15 * len(all_jds))), replace=False))

            is_te = df["resume_id"].isin(te_res) & df["jd_id"].isin(te_jds)
            is_tr = (~df["resume_id"].isin(te_res)) & (~df["jd_id"].isin(te_jds))

            tr_idx = np.where(is_tr)[0]
            te_idx = np.where(is_te)[0]

            if len(te_idx) == 0:
                continue

            preds, probs = fit_predict_model(
                model_name, X[tr_idx], y[tr_idx], X[te_idx], df.iloc[tr_idx], df.iloc[te_idx], num_classes=num_classes, seed=s
            )
            m = eval_metrics(y[te_idx], preds, probs, clusters=df.iloc[te_idx]["resume_id"].values, num_classes=num_classes, seed=s)
            seed_metrics.append(m)

        accs = [m["accuracy"] for m in seed_metrics]
        f1s = [m["macro_f1"] for m in seed_metrics]
        aucs = [m["macro_roc_auc"] for m in seed_metrics if m["macro_roc_auc"] is not None]

        avg_acc = float(np.mean(accs))
        avg_f1 = float(np.mean(f1s))
        avg_auc = float(np.mean(aucs)) if aucs else 0.50
        ci_acc = [float(np.percentile(accs, 2.5)), float(np.percentile(accs, 97.5))]
        ci_f1 = [float(np.percentile(f1s, 2.5)), float(np.percentile(f1s, 97.5))]

        all_results["S3_double_cold"][model_name] = {
            "accuracy": avg_acc,
            "macro_f1": avg_f1,
            "macro_roc_auc": avg_auc,
            "ci_95_accuracy": ci_acc,
            "ci_95_macro_f1": ci_f1,
            "n_runs": len(seed_metrics),
            "time_sec": round(time.time() - t_mod, 2),
        }
        print(f"Acc: {avg_acc:.4f} [CI: {ci_acc[0]:.4f}, {ci_acc[1]:.4f}] | Macro-F1: {avg_f1:.4f} | AUC: {avg_auc:.4f} ({all_results['S3_double_cold'][model_name]['time_sec']}s)")

    total_time = round(time.time() - t0, 2)
    print("\n" + "=" * 65)
    print(f"   SWEEP COMPLETE IN {total_time}s")
    print("=" * 65)

    # Save to JSON
    out_json = PROJECT_ROOT / "reports" / "j1_sweep_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved results to {out_json}")

    res_dir = PROJECT_ROOT / "results" / "j1_sweep"
    res_dir.mkdir(parents=True, exist_ok=True)
    with open(res_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)


if __name__ == "__main__":
    run_j1_sweep()

