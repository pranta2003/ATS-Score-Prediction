import os
import sys
import time
import json
import hashlib
import subprocess
import argparse
from pathlib import Path
import yaml
import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from sklearn.model_selection import GroupKFold, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, roc_auc_score, recall_score
from sklearn.preprocessing import label_binarize
import lightgbm as lgb
import xgboost as xgb

from src.ats.adapters.registry import get_adapter
from src.ats.embeddings import build_pair_features, get_cached_embeddings
from src.ats.modeling import build_tfidf_features
from src.ats.evaluation import (
    compute_classification_metrics,
    vectorized_cluster_bootstrap_acc_f1,
)


def get_git_commit():
    """Retrieve short commit hash."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(PROJECT_ROOT),
            stderr=subprocess.DEVNULL,
        ).decode("utf-8").strip()
        return commit
    except Exception:
        return "0000000"


def compute_run_id(config_dict, commit):
    """Generate run_id = hash of config + git commit."""
    serialized = json.dumps(config_dict, sort_keys=True) + commit
    h = hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:8]
    ds = config_dict.get("dataset", "dataset")
    sc = config_dict.get("scenario", "scen")
    md = config_dict.get("model", "mod")
    return f"{ds}_{sc}_{md}_{h}"


def train_eval_model(model_name, model_params, X_train, y_train, X_test, num_classes=3, seed=42):
    """Factory for model training and prediction."""
    if model_name == "majority":
        maj_class = int(pd.Series(y_train).mode()[0])
        preds = np.full(len(X_test), maj_class, dtype=int)
        probs = np.zeros((len(X_test), num_classes))
        probs[:, maj_class] = 1.0
        return None, preds, probs

    elif model_name == "logistic":
        clf = LogisticRegression(
            max_iter=model_params.get("max_iter", 300),
            C=model_params.get("C", 1.0),
            random_state=seed,
            n_jobs=model_params.get("n_jobs", -1),
        )
        clf.fit(X_train, y_train)
        probs = clf.predict_proba(X_test)
        preds = clf.predict(X_test)
        return clf, preds, probs

    elif model_name == "xgboost":
        clf = xgb.XGBClassifier(
            n_estimators=model_params.get("n_estimators", 150),
            max_depth=model_params.get("max_depth", 5),
            learning_rate=model_params.get("learning_rate", 0.08),
            objective="multi:softprob",
            num_class=num_classes,
            random_state=seed,
            n_jobs=model_params.get("n_jobs", -1),
            tree_method="hist",
        )
        clf.fit(X_train, y_train)
        probs = clf.predict_proba(X_test)
        preds = clf.predict(X_test)
        return clf, preds, probs

    else: # Default: lightgbm
        clf = lgb.LGBMClassifier(
            objective="multiclass",
            num_class=num_classes,
            n_estimators=model_params.get("n_estimators", 250),
            max_depth=model_params.get("max_depth", 6),
            learning_rate=model_params.get("learning_rate", 0.05),
            num_leaves=model_params.get("num_leaves", 31),
            random_state=seed,
            n_jobs=model_params.get("n_jobs", -1),
            verbose=-1,
        )
        clf.fit(X_train, y_train)
        probs = clf.predict_proba(X_test)
        preds = clf.predict(X_test)
        return clf, preds, probs


def evaluate_predictions(y_true, y_pred, y_prob, clusters=None, n_boot=500, seed=42):
    """Compute metrics with fast vectorized cluster bootstrap CIs."""
    num_classes = y_prob.shape[1] if y_prob is not None else 3
    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    
    # Per-class recall
    rec = recall_score(y_true, y_pred, average=None, zero_division=0)
    per_class = {f"class_{c}_recall": float(rec[c]) if c < len(rec) else 0.0 for c in range(num_classes)}

    # Macro ROC-AUC
    try:
        y_bin = label_binarize(y_true, classes=list(range(num_classes)))
        macro_auc = float(roc_auc_score(y_bin, y_prob, average="macro", multi_class="ovr"))
    except Exception:
        macro_auc = None

    # Binary Good (0) vs No Fit (1) AUC
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

    # Bootstrap CIs
    ci_acc = (None, None)
    ci_f1 = (None, None)
    if clusters is not None and len(np.unique(clusters)) >= 5:
        ci_acc, ci_f1 = vectorized_cluster_bootstrap_acc_f1(
            y_true, y_pred, clusters, n_classes=num_classes, n_boot=n_boot, seed=seed
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


def main():
    parser = argparse.ArgumentParser(description="Deterministic ML Experiment Runner")
    parser.add_argument("--config", type=str, required=True, help="Path to config yaml")
    parser.add_argument("--smoke", action="store_true", help="Run rapid smoke test (<60s)")
    args = parser.parse_args()

    t_start = time.time()
    timings = {}

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Smoke overrides
    is_smoke = args.smoke or config.get("smoke", False)
    if is_smoke:
        config["smoke"] = True
        config["seeds"] = [42, 43]
        config["n_boot"] = 50
        model_p = config.get("model_params", {})
        model_p["n_estimators"] = min(model_p.get("n_estimators", 20), 20)
        model_p["max_depth"] = min(model_p.get("max_depth", 3), 3)
        config["model_params"] = model_p

    commit = get_git_commit()
    run_id = compute_run_id(config, commit)
    out_dir = PROJECT_ROOT / "results" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    log_path = out_dir / "log.txt"
    log_file = open(log_path, "w", encoding="utf-8")

    def log(msg):
        print(msg, flush=True)
        log_file.write(msg + "\n")
        log_file.flush()

    log("=" * 65)
    log(f"   ATS SCORE PREDICTION - EXPERIMENT RUNNER")
    log(f"   Run ID: {run_id}")
    log(f"   Git Commit: {commit} | Smoke Mode: {is_smoke}")
    log("=" * 65)

    # Stage 1: Load Data
    t0 = time.time()
    dataset_name = config.get("dataset", "c5")
    adapter = get_adapter(dataset_name)
    df = adapter.load()
    
    if is_smoke:
        smoke_n = config.get("smoke_samples", 200)
        df = df.sample(n=min(smoke_n, len(df)), random_state=42).reset_index(drop=True)
        log(f"Smoke mode active: sliced data to {len(df)} pairs across classes: {df['label_class'].value_counts().to_dict()}.")

    classes = sorted(df["label_class"].unique())
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y = np.array([class_to_idx[c] for c in df["label_class"]], dtype=int)
    num_classes = len(classes)
    timings["data_loading_sec"] = float(time.time() - t0)
    log(f"[Stage 1] Loaded {len(df)} rows across {df['resume_id'].nunique()} resumes and {df['jd_id'].nunique()} JDs ({timings['data_loading_sec']:.2f}s)")

    # Stage 2: Feature Lookup / Construction
    t0 = time.time()
    feature_type = config.get("features", "minilm_interaction")
    if feature_type == "tfidf":
        X = build_tfidf_features(df["resume_text"].tolist(), df["jd_text"].tolist(), max_features=300)
    elif feature_type == "minilm_late_interaction":
        X = build_pair_features(df, dataset_name=dataset_name, include_late_interaction=True)
    else: # minilm_interaction
        X = build_pair_features(df, dataset_name=dataset_name, include_late_interaction=False)
        
    timings["feature_lookup_sec"] = float(time.time() - t0)
    log(f"[Stage 2] Features ready: {feature_type} shape {X.shape} ({timings['feature_lookup_sec']:.2f}s)")

    # Stage 3: Model Cross-Validation / Training
    t0 = time.time()
    scenario = config.get("scenario", "s1_cold_resume")
    model_name = config.get("model", "lightgbm")
    model_params = config.get("model_params", {})
    seeds = config.get("seeds", [42])
    n_splits = config.get("n_splits", 5)
    n_boot = config.get("n_boot", 500)

    log(f"[Stage 3] Executing Scenario '{scenario}' with Model '{model_name}'...")

    preds_records = []
    seed_metrics = []

    if scenario == "s1_cold_resume":
        gkf = GroupKFold(n_splits=n_splits)
        oof_preds = np.zeros(len(df), dtype=int)
        oof_probs = np.zeros((len(df), num_classes))
        
        for fold, (tr_idx, val_idx) in enumerate(gkf.split(df, y, groups=df["resume_id"])):
            _, preds, probs = train_eval_model(
                model_name, model_params, X[tr_idx], y[tr_idx], X[val_idx], num_classes=num_classes, seed=seeds[0]
            )
            oof_preds[val_idx] = preds
            oof_probs[val_idx] = probs

        m = evaluate_predictions(y, oof_preds, oof_probs, clusters=df["resume_id"].values, n_boot=n_boot, seed=seeds[0])
        seed_metrics.append(m)
        for i in range(len(df)):
            preds_records.append({
                "index": i, "resume_id": df.loc[i, "resume_id"], "jd_id": df.loc[i, "jd_id"],
                "y_true": int(y[i]), "y_pred": int(oof_preds[i]),
                **{f"prob_{c}": float(oof_probs[i, c]) for c in range(num_classes)}
            })

    elif scenario == "s2_cold_jd":
        gkf = GroupKFold(n_splits=n_splits)
        oof_preds = np.zeros(len(df), dtype=int)
        oof_probs = np.zeros((len(df), num_classes))
        
        for fold, (tr_idx, val_idx) in enumerate(gkf.split(df, y, groups=df["jd_id"])):
            _, preds, probs = train_eval_model(
                model_name, model_params, X[tr_idx], y[tr_idx], X[val_idx], num_classes=num_classes, seed=seeds[0]
            )
            oof_preds[val_idx] = preds
            oof_probs[val_idx] = probs

        m = evaluate_predictions(y, oof_preds, oof_probs, clusters=df["jd_id"].values, n_boot=n_boot, seed=seeds[0])
        seed_metrics.append(m)
        for i in range(len(df)):
            preds_records.append({
                "index": i, "resume_id": df.loc[i, "resume_id"], "jd_id": df.loc[i, "jd_id"],
                "y_true": int(y[i]), "y_pred": int(oof_preds[i]),
                **{f"prob_{c}": float(oof_probs[i, c]) for c in range(num_classes)}
            })

    elif scenario == "s3_double_cold":
        all_resumes = np.array(df["resume_id"].unique())
        all_jds = np.array(df["jd_id"].unique())
        holdout_pct = config.get("holdout_pct", 0.15)

        for s in seeds:
            rng = np.random.RandomState(s)
            te_res_n = max(1, int(np.ceil(holdout_pct * len(all_resumes))))
            te_jd_n = max(1, int(np.ceil(holdout_pct * len(all_jds))))
            
            te_res = set(rng.choice(all_resumes, size=te_res_n, replace=False))
            te_jds = set(rng.choice(all_jds, size=te_jd_n, replace=False))
            
            is_te = df["resume_id"].isin(te_res) & df["jd_id"].isin(te_jds)
            is_tr = (~df["resume_id"].isin(te_res)) & (~df["jd_id"].isin(te_jds))
            
            tr_idx = np.where(is_tr)[0]
            te_idx = np.where(is_te)[0]
            
            if len(te_idx) == 0:
                continue
                
            _, preds, probs = train_eval_model(
                model_name, model_params, X[tr_idx], y[tr_idx], X[te_idx], num_classes=num_classes, seed=s
            )
            m = evaluate_predictions(y[te_idx], preds, probs, clusters=df.iloc[te_idx]["resume_id"].values, n_boot=n_boot, seed=s)
            m["seed"] = s
            m["n_train"] = len(tr_idx)
            m["n_test"] = len(te_idx)
            seed_metrics.append(m)
            
            for k, orig_idx in enumerate(te_idx):
                preds_records.append({
                    "seed": s, "index": orig_idx, "resume_id": df.loc[orig_idx, "resume_id"], "jd_id": df.loc[orig_idx, "jd_id"],
                    "y_true": int(y[orig_idx]), "y_pred": int(preds[k]),
                    **{f"prob_{c}": float(probs[k, c]) for c in range(num_classes)}
                })
    else: # warm_random
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seeds[0])
        oof_preds = np.zeros(len(df), dtype=int)
        oof_probs = np.zeros((len(df), num_classes))
        for fold, (tr_idx, val_idx) in enumerate(kf.split(df, y)):
            _, preds, probs = train_eval_model(
                model_name, model_params, X[tr_idx], y[tr_idx], X[val_idx], num_classes=num_classes, seed=seeds[0]
            )
            oof_preds[val_idx] = preds
            oof_probs[val_idx] = probs
        m = evaluate_predictions(y, oof_preds, oof_probs, clusters=None, n_boot=n_boot, seed=seeds[0])
        seed_metrics.append(m)
        for i in range(len(df)):
            preds_records.append({
                "index": i, "resume_id": df.loc[i, "resume_id"], "jd_id": df.loc[i, "jd_id"],
                "y_true": int(y[i]), "y_pred": int(oof_preds[i]),
                **{f"prob_{c}": float(oof_probs[i, c]) for c in range(num_classes)}
            })

    timings["cv_training_sec"] = float(time.time() - t0)

    # Average metrics if multiple seeds
    if len(seed_metrics) > 1:
        final_metrics = {"n_runs": len(seed_metrics)}
        for k in seed_metrics[0].keys():
            if k in ["seed", "ci_95_accuracy", "ci_95_macro_f1"]:
                continue
            vals = [sm[k] for sm in seed_metrics if sm.get(k) is not None]
            if vals:
                final_metrics[k] = float(np.mean(vals))
                final_metrics[f"{k}_std"] = float(np.std(vals))
        final_metrics["ci_95_accuracy"] = [float(np.percentile([sm["accuracy"] for sm in seed_metrics], 2.5)), float(np.percentile([sm["accuracy"] for sm in seed_metrics], 97.5))]
        final_metrics["ci_95_macro_f1"] = [float(np.percentile([sm["macro_f1"] for sm in seed_metrics], 2.5)), float(np.percentile([sm["macro_f1"] for sm in seed_metrics], 97.5))]
        final_metrics["runs"] = seed_metrics
    else:
        final_metrics = seed_metrics[0]

    timings["total_sec"] = float(time.time() - t_start)

    log("\n" + "=" * 65)
    log("   EXPERIMENT RESULTS")
    log("=" * 65)
    log(f"Accuracy:        {final_metrics['accuracy']:.4f} (95% CI: {final_metrics.get('ci_95_accuracy')})")
    log(f"Macro-F1:        {final_metrics['macro_f1']:.4f} (95% CI: {final_metrics.get('ci_95_macro_f1')})")
    log(f"Macro ROC-AUC:   {final_metrics.get('macro_roc_auc')}")
    log(f"Majority Base:   {final_metrics.get('majority_baseline'):.4f}")
    log("-" * 65)
    log(f"Timings: Loading={timings['data_loading_sec']:.2f}s | Features={timings['feature_lookup_sec']:.2f}s | CV={timings['cv_training_sec']:.2f}s | Total={timings['total_sec']:.2f}s")
    log("=" * 65)

    # Save artifact files in results/{run_id}/
    with open(out_dir / "config.yaml", "w") as f:
        yaml.dump(config, f)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(final_metrics, f, indent=2)
    with open(out_dir / "timings.json", "w") as f:
        json.dump(timings, f, indent=2)
        
    df_preds = pd.DataFrame(preds_records)
    df_preds.to_parquet(out_dir / "preds.parquet")

    log_file.close()
    print(f"\n[OK] Run {run_id} completed successfully in {timings['total_sec']:.2f}s.")
    print(f"Artifacts saved to: {out_dir}")


if __name__ == "__main__":
    main()

