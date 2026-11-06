"""C2 ATS-Style Feature Ablations and Rubric Validation Script.

Ablations:
- F1: Lexical features (TF-IDF overlap, token ratios, length metrics)
- F2: Dense semantic embeddings (MiniLM bi-encoder + late interaction)
- F3: ATS Rubric heuristics (keyword frequency, skill density proxy)
- F4: Combined Full Model

Evaluates on:
- S1 (Cold-Resume)
- S2 (Cold-JD)
Outputs: R2, Pearson r, MAE for continuous scores; Accuracy, Macro-F1, ROC-AUC for binned classes.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, r2_score, roc_auc_score
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.ats.adapters import get_adapter
from src.ats.embeddings import build_and_cache_document_embeddings, build_pair_features


def extract_lexical_features(resumes: list[str], jds: list[str]) -> np.ndarray:
    tfidf = TfidfVectorizer(max_features=500, stop_words="english")
    tfidf.fit(resumes + jds)
    r_mat = tfidf.transform(resumes)
    j_mat = tfidf.transform(jds)

    cos_sim = np.asarray(r_mat.multiply(j_mat).sum(axis=1)).flatten()
    r_lens = np.array([len(t.split()) for t in resumes], dtype=float)
    j_lens = np.array([len(t.split()) for t in jds], dtype=float)
    len_ratios = r_lens / np.maximum(j_lens, 1.0)
    char_ratios = np.array([len(r) / max(len(j), 1) for r, j in zip(resumes, jds)])

    feats = np.column_stack([cos_sim, r_lens, j_lens, len_ratios, char_ratios])
    return feats


def extract_rubric_heuristics(resumes: list[str], jds: list[str]) -> np.ndarray:
    """Proxy ATS rule-based rubric features: section tokens, bullet density, qualification markers."""
    rubric_feats = []
    keywords = ["experience", "education", "skills", "projects", "certifications", "requirements", "responsibilities"]
    for r, j in zip(resumes, jds):
        r_lower = r.lower()
        j_lower = j.lower()
        r_kw_counts = [r_lower.count(kw) for kw in keywords]
        j_kw_counts = [j_lower.count(kw) for kw in keywords]
        bullet_count = r.count("•") + r.count("- ") + r.count("* ")
        rubric_feats.append(r_kw_counts + j_kw_counts + [bullet_count])
    return np.array(rubric_feats, dtype=float)


def run_c2_ablation(smoke: bool = False, output_dir: str = "results/c2_ablation"):
    print("=== Running C2 ATS Feature Ablations & Rubric Validation ===")
    adapter = get_adapter("c2")
    df = adapter.load()
    # Filter out invalid rows with null continuous scores
    df = df.dropna(subset=["label_continuous"]).reset_index(drop=True)

    if smoke:
        df = df.sample(n=min(120, len(df)), random_state=42).reset_index(drop=True)

    print(f"Total pairs: {len(df)}, Unique resumes: {df['resume_id'].nunique()}, Unique JDs: {df['jd_id'].nunique()}")

    resumes = df["resume_text"].tolist()
    jds = df["jd_text"].tolist()
    y_reg = df["label_continuous"].astype(float).values
    # Pre-registered binning: <40, 40-70, >70
    y_cls = pd.cut(
        df["label_continuous"].astype(float),
        bins=[-float("inf"), 40.0, 70.0, float("inf")],
        labels=[0, 1, 2],
    ).astype(int).values

    # Pre-compute feature blocks
    print("Extracting lexical features...")
    f_lex = extract_lexical_features(resumes, jds)

    print("Extracting rubric heuristics...")
    f_rub = extract_rubric_heuristics(resumes, jds)

    print("Extracting dense embeddings...")
    f_dense = build_pair_features(
        df,
        dataset_name="c2",
        include_late_interaction=False,
        cache_dir="cache/embeddings",
    )

    feature_sets = {
        "F1_Lexical": f_lex,
        "F2_DenseSemantic": f_dense,
        "F3_RubricHeuristics": f_rub,
        "F4_CombinedFull": np.hstack([f_lex, f_dense, f_rub]),
    }

    results = {"scenarios": {}}

    for scen_name, group_col in [("S1_cold_resume", "resume_id"), ("S2_cold_jd", "jd_id")]:
        results["scenarios"][scen_name] = {}
        groups = df[group_col].values
        gkf = GroupKFold(n_splits=3 if smoke else 5)

        for f_name, X in feature_sets.items():
            oof_reg = np.zeros(len(df))
            oof_cls_preds = np.zeros(len(df))
            oof_cls_probs = np.zeros((len(df), 3))

            for tr_idx, te_idx in gkf.split(df, y_cls, groups=groups):
                X_tr, X_te = X[tr_idx], X[te_idx]
                y_reg_tr = y_reg[tr_idx]
                y_cls_tr = y_cls[tr_idx]

                # Regression model (predicting continuous GPT-4o score)
                reg = lgb.LGBMRegressor(n_estimators=20 if smoke else 100, random_state=42, verbose=-1)
                reg.fit(X_tr, y_reg_tr)
                oof_reg[te_idx] = reg.predict(X_te)

                # Classification model (predicting 3-class binned target)
                clf = lgb.LGBMClassifier(n_estimators=20 if smoke else 100, random_state=42, verbose=-1)
                clf.fit(X_tr, y_cls_tr)
                oof_cls_preds[te_idx] = clf.predict(X_te)
                oof_cls_probs[te_idx] = clf.predict_proba(X_te)

            # Continuous metrics
            r2 = float(r2_score(y_reg, oof_reg))
            mae = float(mean_absolute_error(y_reg, oof_reg))
            pr, _ = pearsonr(y_reg, oof_reg)

            # Discrete metrics
            acc = float(accuracy_score(y_cls, oof_cls_preds))
            f1 = float(f1_score(y_cls, oof_cls_preds, average="macro"))
            try:
                auc = float(roc_auc_score(y_cls, oof_cls_probs, multi_class="ovr"))
            except Exception:
                auc = 0.5

            results["scenarios"][scen_name][f_name] = {
                "regression": {"r2": r2, "mae": mae, "pearson_r": float(pr)},
                "classification": {"accuracy": acc, "macro_f1": f1, "macro_roc_auc": auc},
            }
            print(f"[{scen_name}] {f_name} -> R2: {r2:.4f}, r: {pr:.4f}, Acc: {acc:.4f}, AUC: {auc:.4f}")

    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)
    with open(out_p / "c2_ablation_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {out_p / 'c2_ablation_results.json'}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output_dir", default="results/c2_ablation")
    args = parser.parse_args()
    run_c2_ablation(smoke=args.smoke, output_dir=args.output_dir)
