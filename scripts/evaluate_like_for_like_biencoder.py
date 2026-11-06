import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize
import lightgbm as lgb

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.ats.adapters import get_adapter
from src.ats.embeddings import build_pair_features
from src.ats.evaluation import compute_classification_metrics, vectorized_cluster_bootstrap_acc_f1

LABEL_MAP = {"No Fit": 0, "Potential Fit": 1, "Good Fit": 2}


def get_splits(df, scenario, seed=42):
    if scenario == "S2":
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
        train_idx, test_idx = next(gss.split(df, groups=df["jd_id"]))
        return train_idx, test_idx, "jd_id"
    elif scenario == "S3":
        unique_resumes = df["resume_id"].drop_duplicates().sample(frac=0.2, random_state=seed)
        unique_jds = df["jd_id"].drop_duplicates().sample(frac=0.2, random_state=seed)
        test_mask = df["resume_id"].isin(unique_resumes) & df["jd_id"].isin(unique_jds)
        train_mask = (~df["resume_id"].isin(unique_resumes)) & (~df["jd_id"].isin(unique_jds))
        train_idx = np.where(train_mask)[0]
        test_idx = np.where(test_mask)[0]
        return train_idx, test_idx, "jd_id"
    else:
        raise ValueError(f"Unknown scenario {scenario}")


def evaluate_like_for_like():
    print("=== LIKE-FOR-LIKE BASELINE EVALUATION FOR CROSS-ENCODER COMPARISON ===")
    adapter = get_adapter("c5")
    df = adapter.load()
    if df["label_class"].dtype == object or isinstance(df["label_class"].iloc[0], str):
        y = df["label_class"].map(LABEL_MAP).values
    else:
        y = df["label_class"].astype(int).values

    # Pre-extract or load bi-encoder features
    print("Building bi-encoder pair features...")
    X_bi = build_pair_features(df, dataset_name="c5", include_late_interaction=True, cache_dir="cache/embeddings")

    results = {}

    for scen in ["S2", "S3"]:
        print(f"\n--- Scenario {scen} ---")
        train_idx, test_idx, cluster_col = get_splits(df, scen, seed=42)
        df_tr, df_te = df.iloc[train_idx].reset_index(drop=True), df.iloc[test_idx].reset_index(drop=True)
        y_tr, y_te = y[train_idx], y[test_idx]
        X_tr, X_te = X_bi[train_idx], X_bi[test_idx]
        n_te = len(y_te)

        print(f"Train samples: {len(y_tr)}, Test samples: {len(y_te)}")
        clusters = df_te[cluster_col].values

        # 1. Majority baseline
        maj_c = int(pd.Series(y_tr).mode()[0])
        maj_preds = np.full(n_te, maj_c, dtype=int)
        maj_probs = np.zeros((n_te, 3))
        maj_probs[:, maj_c] = 1.0
        maj_acc = float(accuracy_score(y_te, maj_preds))
        maj_f1 = float(f1_score(y_te, maj_preds, average="macro", zero_division=0))
        maj_auc = 0.50

        # 2. JD Prior baseline
        global_prior = pd.Series(y_tr).value_counts(normalize=True).reindex(range(3), fill_value=0.0).values
        jd_priors = {}
        for jid, grp in df_tr.groupby("jd_id"):
            counts = pd.Series(y[train_idx][df_tr["jd_id"] == jid]).value_counts(normalize=True).reindex(range(3), fill_value=0.0).values
            jd_priors[jid] = counts
        jdp_probs = np.zeros((n_te, 3))
        for i, jid in enumerate(df_te["jd_id"]):
            jdp_probs[i] = jd_priors.get(jid, global_prior)
        jdp_preds = np.argmax(jdp_probs, axis=1)
        jdp_acc = float(accuracy_score(y_te, jdp_preds))
        jdp_f1 = float(f1_score(y_te, jdp_preds, average="macro", zero_division=0))
        try:
            y_te_bin = label_binarize(y_te, classes=[0, 1, 2])
            jdp_auc = float(roc_auc_score(y_te_bin, jdp_probs, multi_class="ovr", average="macro"))
        except Exception:
            jdp_auc = 0.50

        # 3. Bi-Encoder + LightGBM
        clf = lgb.LGBMClassifier(
            objective="multiclass",
            num_class=3,
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            num_leaves=31,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
        clf.fit(X_tr, y_tr)
        lgb_preds = clf.predict(X_te)
        lgb_probs = clf.predict_proba(X_te)
        lgb_acc = float(accuracy_score(y_te, lgb_preds))
        lgb_f1 = float(f1_score(y_te, lgb_preds, average="macro", zero_division=0))
        y_te_bin = label_binarize(y_te, classes=[0, 1, 2])
        lgb_auc = float(roc_auc_score(y_te_bin, lgb_probs, multi_class="ovr", average="macro"))

        # Bootstrap CIs for LightGBM
        ci_acc, ci_f1 = vectorized_cluster_bootstrap_acc_f1(y_te, lgb_preds, clusters, n_boot=1000, seed=42)

        results[scen] = {
            "test_size": n_te,
            "majority": {"accuracy": maj_acc, "macro_f1": maj_f1, "macro_roc_auc": maj_auc},
            "jd_prior": {"accuracy": jdp_acc, "macro_f1": jdp_f1, "macro_roc_auc": jdp_auc},
            "biencoder_lightgbm": {
                "accuracy": lgb_acc,
                "macro_f1": lgb_f1,
                "macro_roc_auc": lgb_auc,
                "ci_95_accuracy": ci_acc,
                "ci_95_macro_f1": ci_f1,
            },
            "y_true": y_te.tolist(),
            "lgb_preds": lgb_preds.tolist(),
            "lgb_probs": lgb_probs.tolist(),
        }
        print(f"  Majority Baseline:       Acc: {maj_acc:.4f} | F1: {maj_f1:.4f} | AUC: {maj_auc:.4f}")
        print(f"  JD-Prior Baseline:       Acc: {jdp_acc:.4f} | F1: {jdp_f1:.4f} | AUC: {jdp_auc:.4f}")
        print(f"  Bi-Encoder + LightGBM:   Acc: {lgb_acc:.4f} [CI: {ci_acc[0]:.4f}, {ci_acc[1]:.4f}] | F1: {lgb_f1:.4f} | AUC: {lgb_auc:.4f}")

    out_file = PROJECT_ROOT / "reports" / "like_for_like_biencoder.json"
    with open(out_file, "w", encoding="utf-8") as f:
        # Save without raw lists for clean output
        clean_res = {k: {m: v for m, v in val.items() if not m.startswith("y_") and not m.startswith("lgb_")} for k, val in results.items()}
        json.dump(clean_res, f, indent=2)
    print(f"\nSaved like-for-like results to {out_file}")

    # Now compute paired differences with cross-encoder results
    ce_res_path = PROJECT_ROOT / "reports" / "cross_encoder_results.json"
    if ce_res_path.exists():
        with open(ce_res_path, "r", encoding="utf-8") as f:
            ce_res = json.load(f)["scenarios"]
        print("\n=== PAIRED COMPARISON ON EXACT SAME TEST SPLITS ===")
        for scen, name in [("S2", "S2_cold_jd"), ("S3", "S3_double_cold")]:
            ce = ce_res[name]
            bi = results[scen]["biencoder_lightgbm"]
            maj = results[scen]["majority"]
            print(f"\n{scen} (Test size: {results[scen]['test_size']}, Majority: {maj['accuracy']:.4f}):")
            print(f"  Cross-Encoder (3 ep):   Acc: {ce['accuracy']:.4f} [CI: {ce['ci_95_accuracy'][0]:.4f}, {ce['ci_95_accuracy'][1]:.4f}] | Macro-F1: {ce['macro_f1']:.4f} | AUC: {ce['macro_roc_auc']:.4f}")
            print(f"  Bi-Encoder + LightGBM:  Acc: {bi['accuracy']:.4f} [CI: {bi['ci_95_accuracy'][0]:.4f}, {bi['ci_95_accuracy'][1]:.4f}] | Macro-F1: {bi['macro_f1']:.4f} | AUC: {bi['macro_roc_auc']:.4f}")
            diff_auc = bi['macro_roc_auc'] - ce['macro_roc_auc']
            diff_acc = bi['accuracy'] - ce['accuracy']
            diff_f1 = bi['macro_f1'] - ce['macro_f1']
            print(f"  Difference (Bi - CE):   Delta-AUC: {diff_auc:+.4f} | Delta-Acc: {diff_acc:+.4f} | Delta-F1: {diff_f1:+.4f}")


if __name__ == "__main__":
    evaluate_like_for_like()

