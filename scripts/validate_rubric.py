"""Validation and empirical behavior analysis for the Interpretable Rubric Scorer."""

import os
import sys
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.ats.adapters import get_adapter
from src.ats.rubric import score_rubric, evaluate_formatting


def compute_within_jd_auc(df_scored: pd.DataFrame) -> dict:
    """Compute within-JD ranking metrics across eligible JDs (>=5 candidates with Good and No Fit)."""
    jd_aucs = []
    jd_counts = []
    
    for jd_id, group in df_scored.groupby("jd_id"):
        classes = set(group["label_num"])
        if len(group) >= 5 and 0 in classes and 2 in classes:
            # Binary subset (No Fit vs Good Fit)
            binary_sub = group[group["label_num"].isin([0, 2])]
            if len(binary_sub["label_num"].unique()) == 2:
                y_bin = (binary_sub["label_num"] == 2).astype(int)
                try:
                    auc = roc_auc_score(y_bin, binary_sub["rubric_score"])
                    jd_aucs.append(auc)
                    jd_counts.append(len(group))
                except Exception:
                    pass

    return {
        "eligible_jds": len(jd_aucs),
        "mean_candidates_per_jd": float(np.mean(jd_counts)) if jd_counts else 0.0,
        "mean_within_jd_auc": float(np.mean(jd_aucs)) if jd_aucs else 0.5,
        "median_within_jd_auc": float(np.median(jd_aucs)) if jd_aucs else 0.5,
    }


def run_behavior_tests() -> dict:
    """Evaluate behavioral properties: tailoring, keyword stuffing cap, JD swap, determinism."""
    base_jd = "Senior Python Developer with 5 years experience in Django, PostgreSQL, Docker, AWS, and Kubernetes. Bachelor degree required."
    base_resume = "Software Engineer with 4 years experience in Python and MySQL. Bachelor of Science in Computer Science. Email: dev@example.com."

    # 1. Determinism
    res1 = score_rubric(base_resume, base_jd)
    res2 = score_rubric(base_resume, base_jd)
    det_pass = (res1["rubric_score"] == res2["rubric_score"] and res1["breakdown"] == res2["breakdown"])

    # 2. Tailoring Effect Size
    # Add target keywords from JD
    tailored_resume = base_resume + " Extensive hands-on experience deploying microservices with Django, PostgreSQL, Docker, AWS, and Kubernetes."
    res_tailored = score_rubric(tailored_resume, base_jd)
    tailoring_delta = res_tailored["rubric_score"] - res1["rubric_score"]
    tailoring_pass = tailoring_delta >= 10.0

    # 3. Keyword Stuffing Cap Effect Size
    # Repeat a single matched keyword 50 times
    stuffed_resume = base_resume + " " + ("python " * 50)
    res_stuffed = score_rubric(stuffed_resume, base_jd)
    stuffing_delta = res_stuffed["rubric_score"] - res1["rubric_score"]
    stuffing_pass = (stuffing_delta == 0.0)  # Set-based keyword extraction gives 0 bonus for repetition

    # 4. JD Swap Sensitivity
    unrelated_jd = "Senior Clinical Nurse Specialist with 8 years experience in intensive care, triage, and patient trauma care. Registered Nursing Degree required."
    res_swapped = score_rubric(base_resume, unrelated_jd)
    swap_delta = abs(res1["rubric_score"] - res_swapped["rubric_score"])
    swap_pass = swap_delta >= 15.0

    return {
        "determinism": {
            "passed": bool(det_pass),
            "score_run_1": res1["rubric_score"],
            "score_run_2": res2["rubric_score"],
        },
        "tailoring_effect": {
            "passed": bool(tailoring_pass),
            "base_score": res1["rubric_score"],
            "tailored_score": res_tailored["rubric_score"],
            "delta_points": round(tailoring_delta, 1),
        },
        "keyword_stuffing_cap": {
            "passed": bool(stuffing_pass),
            "base_score": res1["rubric_score"],
            "stuffed_score": res_stuffed["rubric_score"],
            "delta_points": round(stuffing_delta, 1),
            "explanation": "Set-based tokenization prevents repeated keywords from gaming match score.",
        },
        "jd_swap_sensitivity": {
            "passed": bool(swap_pass),
            "original_jd_score": res1["rubric_score"],
            "swapped_jd_score": res_swapped["rubric_score"],
            "delta_points": round(swap_delta, 1),
        },
    }


def main():
    print("=" * 70)
    print("Validating Interpretable Rubric Scorer on C5 Benchmark")
    print("=" * 70)

    t0 = time.time()
    adapter = get_adapter("c5")
    df = adapter.load()
    print(f"[Data] Loaded C5: {len(df)} pairs, {df['resume_id'].nunique()} resumes, {df['jd_id'].nunique()} JDs.")

    label_map = {"No Fit": 0, "Potential Fit": 1, "Good Fit": 2}
    df["label_num"] = df["label_class"].map(label_map).fillna(0).astype(int)

    # Score sample of 1,000 pairs (or full if fast)
    # 1,000 pairs takes ~3 seconds with regex rules
    sample_df = df.sample(n=min(2000, len(df)), random_state=42).copy()
    print(f"[Evaluation] Scoring {len(sample_df)} candidate pairs...")

    scores = []
    formatting_scores = []
    for r_text, j_text in zip(sample_df["resume_text"], sample_df["jd_text"]):
        res = score_rubric(r_text, j_text)
        scores.append(res["rubric_score"])
        formatting_scores.append(res["formatting_score"])

    sample_df["rubric_score"] = scores
    sample_df["formatting_score"] = formatting_scores

    # 1. Spearman Rank Correlation
    spearman_corr, spearman_p = spearmanr(sample_df["rubric_score"], sample_df["label_num"])
    print(f"[Correlation] Spearman rank r: {spearman_corr:.4f} (p={spearman_p:.2e})")

    # 2. Binary Good-vs-No-Fit AUC
    bin_subset = sample_df[sample_df["label_num"].isin([0, 2])]
    y_bin = (bin_subset["label_num"] == 2).astype(int)
    bin_auc = roc_auc_score(y_bin, bin_subset["rubric_score"])
    print(f"[Discrimination] Binary Good-vs-No-Fit AUC: {bin_auc:.4f}")

    # 3. Macro AUC (One-vs-Rest)
    y_one_hot = pd.get_dummies(sample_df["label_num"])[[0, 1, 2]].values
    # Rubric score is a single scalar fit score, so we compute correlation/AUC for ranking
    # Good fit vs rest AUC:
    good_vs_rest_auc = roc_auc_score((sample_df["label_num"] == 2).astype(int), sample_df["rubric_score"])
    no_fit_vs_rest_auc = roc_auc_score((sample_df["label_num"] == 0).astype(int), -sample_df["rubric_score"])
    macro_auc = float((good_vs_rest_auc + no_fit_vs_rest_auc) / 2.0)
    print(f"[Discrimination] Macro AUC: {macro_auc:.4f} (Good-vs-Rest: {good_vs_rest_auc:.4f}, NoFit-vs-Rest: {no_fit_vs_rest_auc:.4f})")

    # 4. Within-JD Ranking Performance
    within_jd = compute_within_jd_auc(sample_df)
    print(f"[Within-JD] Eligible JDs: {within_jd['eligible_jds']}, Mean AUC: {within_jd['mean_within_jd_auc']:.4f}")

    # 5. Behavior Tests
    print("[Behavior] Running behavior tests (tailoring, stuffing, swap, determinism)...")
    behavior_results = run_behavior_tests()

    results = {
        "dataset": "cnamuangtoun/resume-job-description-fit",
        "sample_size": len(sample_df),
        "correlation": {
            "spearman_r": float(spearman_corr),
            "spearman_pvalue": float(spearman_p),
        },
        "discrimination": {
            "binary_good_vs_nofit_auc": float(bin_auc),
            "good_vs_rest_auc": float(good_vs_rest_auc),
            "nofit_vs_rest_auc": float(no_fit_vs_rest_auc),
            "macro_directional_auc": float(macro_auc),
        },
        "within_jd_ranking": within_jd,
        "behavior_tests": behavior_results,
        "formatting_decoupling": {
            "formatting_moved_out_of_fit_score": True,
            "mean_formatting_score": float(np.mean(formatting_scores)),
            "std_formatting_score": float(np.std(formatting_scores)),
        },
        "elapsed_seconds": round(time.time() - t0, 2),
    }

    out_file = PROJECT_ROOT / "reports" / "rubric_validation_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"[OK] Validation results saved to {out_file}")


if __name__ == "__main__":
    main()
