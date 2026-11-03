import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GroupKFold, KFold
import lightgbm as lgb
from .evaluation import (
    compute_classification_metrics,
    compute_regression_metrics,
    cluster_bootstrap_ci,
)


def build_tfidf_features(resumes, jds, max_features=300):
    """Compute separate TF-IDF representations and interaction cosine."""
    vec_res = TfidfVectorizer(max_features=max_features, stop_words="english")
    vec_jd = TfidfVectorizer(max_features=max_features, stop_words="english")
    
    X_res = vec_res.fit_transform(resumes).toarray()
    X_jd = vec_jd.fit_transform(jds).toarray()
    
    norm_res = np.linalg.norm(X_res, axis=1, keepdims=True) + 1e-9
    norm_jd = np.linalg.norm(X_jd, axis=1, keepdims=True) + 1e-9
    cos = np.sum((X_res / norm_res) * (X_jd / norm_jd), axis=1, keepdims=True)
    
    X = np.hstack([X_res, X_jd, np.abs(X_res - X_jd), cos])
    return X


def run_experiment_protocol(
    X: np.ndarray,
    df: pd.DataFrame,
    split_type: str,
    group_col: str = None,
    is_regression: bool = False,
    n_splits: int = 5,
    seed: int = 42,
):
    """Run CV loop with GroupKFold or KFold, collect OOF predictions, and compute metrics with 95% CIs."""
    if is_regression:
        valid_mask = df["label_continuous"].notnull().values
    else:
        valid_mask = df["label_class"].notnull().values

    if not valid_mask.all():
        X = X[valid_mask]
        df = df[valid_mask].reset_index(drop=True)

    n_samples = len(df)
    oof_preds = np.zeros(n_samples)
    oof_probs = None
    
    if is_regression:
        y = df["label_continuous"].values
    else:
        y_raw = df["label_class"].astype(str).values
        classes, y = np.unique(y_raw, return_inverse=True)
        num_classes = len(classes)
        oof_probs = np.zeros((n_samples, num_classes))

    if split_type == "grouped":
        groups = df[group_col].values
        # Ensure at least n_splits groups
        n_unique_groups = len(np.unique(groups))
        splits_to_use = min(n_splits, n_unique_groups)
        splitter = GroupKFold(n_splits=splits_to_use)
        splits = list(splitter.split(X, y, groups=groups))
    else:
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        splits = list(splitter.split(X, y))

    for train_idx, val_idx in splits:
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_val = X[val_idx]
        
        if is_regression:
            reg = lgb.LGBMRegressor(
                n_estimators=100,
                learning_rate=0.08,
                max_depth=5,
                num_leaves=24,
                random_state=seed,
                verbose=-1,
                n_jobs=-1,
            )
            reg.fit(X_tr, y_tr)
            oof_preds[val_idx] = reg.predict(X_val)
        else:
            if num_classes == 2:
                clf = lgb.LGBMClassifier(
                    n_estimators=100,
                    learning_rate=0.08,
                    max_depth=5,
                    num_leaves=24,
                    random_state=seed,
                    verbose=-1,
                    n_jobs=-1,
                )
            else:
                clf = lgb.LGBMClassifier(
                    n_estimators=100,
                    learning_rate=0.08,
                    max_depth=5,
                    num_leaves=24,
                    random_state=seed,
                    verbose=-1,
                    n_jobs=-1,
                    objective="multiclass",
                )
            clf.fit(X_tr, y_tr)
            oof_probs[val_idx] = clf.predict_proba(X_val)
            oof_preds[val_idx] = clf.predict(X_val)

    # Collect metrics
    df_eval = df.copy()
    df_eval["y_true"] = y
    df_eval["y_pred"] = oof_preds

    if is_regression:
        metrics = compute_regression_metrics(y, oof_preds)
        
        # Pre-registered 3-class bins: <40, 40-70, >70
        def to_bin(val):
            if val < 40.0:
                return 0
            elif val <= 70.0:
                return 1
            return 2

        y_true_bins = np.array([to_bin(v) for v in y])
        y_pred_bins = np.array([to_bin(v) for v in oof_preds])
        binned_clf_metrics = compute_classification_metrics(y_true_bins, y_pred_bins)
        metrics["accuracy"] = binned_clf_metrics["accuracy"]
        metrics["macro_f1"] = binned_clf_metrics["macro_f1"]
        metrics["balanced_accuracy"] = binned_clf_metrics["balanced_accuracy"]
        metrics["majority_baseline"] = binned_clf_metrics["majority_baseline"]
        metrics["macro_roc_auc"] = None

        # Cluster bootstrap 95% CIs
        cluster_key = group_col if group_col and group_col in df_eval.columns else "resume_id"
        low_acc, high_acc = cluster_bootstrap_ci(
            df_eval, cluster_key,
            lambda d: compute_classification_metrics(np.array([to_bin(v) for v in d["y_true"]]), np.array([to_bin(v) for v in d["y_pred"]]))["accuracy"],
            n_boot=200
        )
        low_f1, high_f1 = cluster_bootstrap_ci(
            df_eval, cluster_key,
            lambda d: compute_classification_metrics(np.array([to_bin(v) for v in d["y_true"]]), np.array([to_bin(v) for v in d["y_pred"]]))["macro_f1"],
            n_boot=200
        )
        metrics["ci_95_accuracy"] = [low_acc, high_acc]
        metrics["ci_95_macro_f1"] = [low_f1, high_f1]
    else:
        metrics = compute_classification_metrics(y, oof_preds, y_prob=oof_probs)
        cluster_key = group_col if group_col and group_col in df_eval.columns else "resume_id"
        low_acc, high_acc = cluster_bootstrap_ci(
            df_eval, cluster_key,
            lambda d: compute_classification_metrics(d["y_true"], d["y_pred"])["accuracy"],
            n_boot=200
        )
        low_f1, high_f1 = cluster_bootstrap_ci(
            df_eval, cluster_key,
            lambda d: compute_classification_metrics(d["y_true"], d["y_pred"])["macro_f1"],
            n_boot=200
        )
        metrics["ci_95_accuracy"] = [low_acc, high_acc]
        metrics["ci_95_macro_f1"] = [low_f1, high_f1]

    return metrics
