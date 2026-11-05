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


def vectorized_cluster_bootstrap_acc_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    clusters: np.ndarray,
    n_classes: int = 3,
    n_boot: int = 500,
    seed: int = 42,
):
    """
    Vectorized cluster bootstrap for accuracy and macro-F1.
    Resamples cluster frequencies via multinomial distribution and aggregates sufficient statistics.
    Runs in milliseconds!
    """
    unique_clusters, cluster_inv = np.unique(clusters, return_inverse=True)
    K = len(unique_clusters)
    if K < 5:
        return (None, None), (None, None)

    # 1. Cluster size: shape (K,)
    cluster_n = np.bincount(cluster_inv)

    # 2. Accuracy: cluster sum of correct predictions: shape (K,)
    is_correct = (y_true == y_pred).astype(int)
    cluster_correct = np.bincount(cluster_inv, weights=is_correct)

    # 3. Macro-F1 sufficient statistics per cluster:
    # For each class c, compute TP_k, FP_k, FN_k
    cluster_tp = np.zeros((K, n_classes))
    cluster_fp = np.zeros((K, n_classes))
    cluster_fn = np.zeros((K, n_classes))

    for c in range(n_classes):
        tp_mask = ((y_pred == c) & (y_true == c)).astype(int)
        fp_mask = ((y_pred == c) & (y_true != c)).astype(int)
        fn_mask = ((y_pred != c) & (y_true == c)).astype(int)
        
        cluster_tp[:, c] = np.bincount(cluster_inv, weights=tp_mask, minlength=K)
        cluster_fp[:, c] = np.bincount(cluster_inv, weights=fp_mask, minlength=K)
        cluster_fn[:, c] = np.bincount(cluster_inv, weights=fn_mask, minlength=K)

    # Sample cluster frequency weights: shape (n_boot, K)
    rng = np.random.RandomState(seed)
    boot_weights = rng.multinomial(n=K, pvals=np.full(K, 1.0 / K), size=n_boot)

    # Vectorized Accuracy: shape (n_boot,)
    boot_corr = boot_weights @ cluster_correct
    boot_total = boot_weights @ cluster_n
    boot_acc = boot_corr / (boot_total + 1e-9)

    # Vectorized Macro-F1:
    # boot_tp: (n_boot, n_classes)
    boot_tp = boot_weights @ cluster_tp
    boot_fp = boot_weights @ cluster_fp
    boot_fn = boot_weights @ cluster_fn

    denom = 2 * boot_tp + boot_fp + boot_fn + 1e-9
    boot_f1_per_class = (2 * boot_tp) / denom # (n_boot, n_classes)
    boot_macro_f1 = np.mean(boot_f1_per_class, axis=1) # (n_boot,)

    ci_acc = (float(np.percentile(boot_acc, 2.5)), float(np.percentile(boot_acc, 97.5)))
    ci_f1 = (float(np.percentile(boot_macro_f1, 2.5)), float(np.percentile(boot_macro_f1, 97.5)))

    return ci_acc, ci_f1


def cluster_bootstrap_ci(df_oof: pd.DataFrame, cluster_col: str, metric_fn, n_boot: int = 500, seed: int = 42):
    """
    General cluster bootstrap confidence interval with fast vectorization fallback.
    """
    clusters = df_oof[cluster_col].values
    if "y_true" in df_oof.columns and "y_pred" in df_oof.columns:
        y_true = np.asarray(df_oof["y_true"])
        y_pred = np.asarray(df_oof["y_pred"])
        
        # If classes are discrete integers
        if np.issubdtype(y_true.dtype, np.integer) and np.issubdtype(y_pred.dtype, np.integer):
            n_classes = max(int(np.max(y_true)), int(np.max(y_pred))) + 1
            ci_acc, ci_f1 = vectorized_cluster_bootstrap_acc_f1(
                y_true, y_pred, clusters, n_classes=n_classes, n_boot=n_boot, seed=seed
            )
            return ci_acc

    # Fallback for arbitrary custom metric functions
    unique_clusters = np.unique(clusters)
    K = len(unique_clusters)
    if K < 5:
        return None, None
        
    rng = np.random.RandomState(seed)
    cluster_indices = df_oof.groupby(cluster_col).indices
    
    scores = []
    for _ in range(n_boot):
        boot_clusters = rng.choice(unique_clusters, size=K, replace=True)
        boot_idx = np.concatenate([cluster_indices[c] for c in boot_clusters])
        boot_df = df_oof.iloc[boot_idx]
        try:
            val = metric_fn(boot_df)
            if val is not None and not np.isnan(val):
                scores.append(val)
        except Exception:
            continue

    if len(scores) < 20:
        return None, None
    low = float(np.percentile(scores, 2.5))
    high = float(np.percentile(scores, 97.5))
    return low, high
