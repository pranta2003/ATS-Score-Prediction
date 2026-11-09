"""Investigate whether re-weighted rubric features on training folds show signal."""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.ats.adapters import get_adapter
from src.ats.rubric import (
    extract_keywords,
    extract_years_experience,
    extract_degree_level,
    extract_seniority_level,
)


def extract_rubric_components(df: pd.DataFrame):
    """Extract individual rubric components for all pairs."""
    features = []
    for _, row in df.iterrows():
        r_text = row["resume_text"]
        j_text = row["jd_text"]

        r_words = extract_keywords(r_text)
        j_words = extract_keywords(j_text)

        kw_match = len(r_words & j_words) / max(len(j_words), 1)

        r_yrs = extract_years_experience(r_text)
        j_yrs = extract_years_experience(j_text)
        exp_match = min(r_yrs / max(j_yrs, 1.0), 1.5)

        r_deg = extract_degree_level(r_text)
        j_deg = extract_degree_level(j_text)
        deg_match = 1.0 if r_deg >= j_deg else (0.5 if r_deg > 0 else 0.0)

        r_sen = extract_seniority_level(r_text)
        j_sen = extract_seniority_level(j_text)
        sen_match = 1.0 if r_sen >= j_sen else 0.5

        features.append([kw_match, exp_match, deg_match, sen_match])

    return np.array(features, dtype=np.float32)


def main():
    print("=== Testing Re-Weighted Rubric Signal on C5 ===")
    adapter = get_adapter("c5")
    df = adapter.load()

    label_map = {"No Fit": 0, "Potential Fit": 1, "Good Fit": 2}
    y = np.array([label_map[l] for l in df["label_class"]], dtype=int)

    X_rubric = extract_rubric_components(df)
    print(f"Rubric feature matrix: {X_rubric.shape}")

    # Inspect component correlations with true labels
    names = ["Keyword Overlap", "Years Exp Match", "Degree Match", "Seniority Match"]
    print("\nIndividual Component Correlations (Spearman r with 3-class label):")
    for col_idx, name in enumerate(names):
        r, p = spearmanr(X_rubric[:, col_idx], y)
        print(f"  {name:20s}: r = {r:+.4f} (p = {p:.4e})")

    # Evaluate Logistic Regression on S1 and S2
    for scen_name, group_col in [("S1 (Cold Resume)", "resume_id"), ("S2 (Cold JD)", "jd_id")]:
        gkf = GroupKFold(n_splits=5)
        oof_probs = np.zeros((len(y), 3))

        for tr_idx, val_idx in gkf.split(X_rubric, y, groups=df[group_col]):
            clf = LogisticRegression(max_iter=500, random_state=42)
            clf.fit(X_rubric[tr_idx], y[tr_idx])
            oof_probs[val_idx] = clf.predict_proba(X_rubric[val_idx])

        # Binary Good vs No Fit
        mask = (y != 1)
        y_bin = (y[mask] == 2).astype(int)
        p_good_bin = oof_probs[mask, 2] / (oof_probs[mask, 0] + oof_probs[mask, 2] + 1e-9)

        bin_auc = roc_auc_score(y_bin, p_good_bin)
        r_oof, _ = spearmanr(oof_probs[:, 2], y)

        print(f"\n{scen_name} Re-Weighted Logistic Regression (Training Folds Only):")
        print(f"  Spearman r (P(Good Fit) vs Class): {r_oof:+.4f}")
        print(f"  Binary Good-vs-No-Fit AUC:          {bin_auc:.4f}")


if __name__ == "__main__":
    main()
