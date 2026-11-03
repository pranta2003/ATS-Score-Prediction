import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    balanced_accuracy_score,
    roc_auc_score,
    r2_score,
    mean_absolute_error,
)
from scipy.stats import spearmanr


def compute_classification_metrics(y_true, y_pred, y_prob=None, majority_label=None):
    """Compute standard classification metrics."""
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    
    # Majority baseline
    if majority_label is not None:
        maj_pred = np.full_like(y_pred, majority_label)
        majority_acc = accuracy_score(y_true, maj_pred)
    else:
        majority_label = pd.Series(y_true).mode()[0]
        majority_acc = (pd.Series(y_true) == majority_label).mean()

    # Macro ROC-AUC if probabilities provided
    roc_auc = np.nan
    if y_prob is not None:
        try:
            classes = np.unique(y_true)
            if len(classes) == 2:
                # Binary
                if y_prob.ndim == 2 and y_prob.shape[1] == 2:
                    roc_auc = roc_auc_score(y_true, y_prob[:, 1])
                else:
                    roc_auc = roc_auc_score(y_true, y_prob)
            elif len(classes) > 2:
                roc_auc = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
        except Exception:
            roc_auc = np.nan

    return {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "balanced_accuracy": float(bal_acc),
        "macro_roc_auc": float(roc_auc) if not np.isnan(roc_auc) else None,
        "majority_baseline": float(majority_acc),
    }


def compute_regression_metrics(y_true, y_pred):
    """Compute standard continuous regression metrics."""
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rho, _ = spearmanr(y_true, y_pred)
    return {
        "r2": float(r2),
        "mae": float(mae),
        "spearman_rho": float(rho) if not np.isnan(rho) else 0.0,
    }


def cluster_bootstrap_ci(df_oof: pd.DataFrame, cluster_col: str, metric_fn, n_boot: int = 1000, seed: int = 42):
    """
    Compute 95% bootstrap confidence interval by resampling clusters (e.g., resumes).
    df_oof must contain y_true, y_pred, and cluster_col.
    """
    clusters = df_oof[cluster_col].unique()
    n_clusters = len(clusters)
    rng = np.random.RandomState(seed)

    # Group rows by cluster
    cluster_groups = df_oof.groupby(cluster_col).indices

    scores = []
    for _ in range(n_boot):
        boot_clusters = rng.choice(clusters, size=n_clusters, replace=True)
        boot_indices = np.concatenate([cluster_groups[c] for c in boot_clusters])
        boot_df = df_oof.iloc[boot_indices]
        try:
            val = metric_fn(boot_df)
            if val is not None and not np.isnan(val):
                scores.append(val)
        except Exception:
            continue

    if len(scores) < 100:
        return None, None
    low = np.percentile(scores, 2.5)
    high = np.percentile(scores, 97.5)
    return float(low), float(high)

