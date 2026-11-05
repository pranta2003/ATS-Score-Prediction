import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import json
import numpy as np
import pandas as pd
from rapidfuzz import fuzz

from src.ats.adapters.registry import get_adapter
from src.ats.adapters.base import normalize_text, compute_sha1
from src.ats.modeling import run_experiment_protocol, build_tfidf_features
from src.ats.embeddings import get_cached_embeddings

def main():
    print("=======================================================")
    print(" 1. RECONCILING UNIQUE RESUME COUNTS")
    print("=======================================================")
    
    # Load Control raw
    ctrl_adapter = get_adapter("control")
    df_ctrl_raw = ctrl_adapter.load_raw()
    
    # Split on first " SEP "
    s_raw = df_ctrl_raw["text"].astype(str).str.split(" SEP ", n=1, expand=True)
    res_raw_strip = s_raw[0].str.strip()
    res_norm = res_raw_strip.apply(normalize_text)
    
    print(f"Control Raw split strings (strip only, Phase 1 style): {res_raw_strip.nunique()} unique resumes")
    print(f"Control Normalized (lowercased + collapsed whitespace): {res_norm.nunique()} unique resumes")
    
    df_ctrl = ctrl_adapter.load()
    print(f"Control adapter standardized unique resumes: {df_ctrl['resume_id'].nunique()}")
    print(f"Control adapter standardized unique JDs: {df_ctrl['jd_id'].nunique()}")
    
    # Load C5
    c5_adapter = get_adapter("c5")
    df_c5 = c5_adapter.load()
    print(f"C5 standardized unique resumes: {df_c5['resume_id'].nunique()}")
    print(f"C5 standardized unique JDs: {df_c5['jd_id'].nunique()}")
    
    # Overlap between Control and C5 by normalized text hash
    ctrl_res_set = set(df_ctrl["resume_id"].unique())
    c5_res_set = set(df_c5["resume_id"].unique())
    exact_res_matches = ctrl_res_set.intersection(c5_res_set)
    
    ctrl_jd_set = set(df_ctrl["jd_id"].unique())
    c5_jd_set = set(df_c5["jd_id"].unique())
    exact_jd_matches = ctrl_jd_set.intersection(c5_jd_set)
    
    ctrl_pairs = set(zip(df_ctrl["resume_id"], df_ctrl["jd_id"]))
    c5_pairs = set(zip(df_c5["resume_id"], df_c5["jd_id"]))
    exact_pair_matches = ctrl_pairs.intersection(c5_pairs)
    
    print(f"\nExact Hash Overlap:")
    print(f"  Resumes: {len(exact_res_matches)} / {len(ctrl_res_set)} Control, {len(c5_res_set)} C5")
    print(f"  JDs: {len(exact_jd_matches)} / {len(ctrl_jd_set)} Control, {len(c5_jd_set)} C5")
    print(f"  Pairs: {len(exact_pair_matches)}")

    # Fuzzy Matching between Control and C5 (rapidfuzz ratio >= 95)
    print("\nRunning Fuzzy Matching (rapidfuzz ratio >= 95)...")
    
    # Resumes
    ctrl_res_texts = df_ctrl.drop_duplicates("resume_id")[["resume_id", "resume_text"]].reset_index(drop=True)
    c5_res_texts = df_c5.drop_duplicates("resume_id")[["resume_id", "resume_text"]].reset_index(drop=True)
    
    # Sample or full pairwise check
    fuzzy_res_matches = 0
    fuzzy_res_match_pairs = []
    
    # To be fast, check normalized text prefixes or exact length proximity
    for i, r_row in ctrl_res_texts.iterrows():
        r_text = normalize_text(r_row["resume_text"])
        r_id = r_row["resume_id"]
        if r_id in exact_res_matches:
            fuzzy_res_matches += 1
            continue
        # search in c5
        best_score = 0
        for _, c_row in c5_res_texts.iterrows():
            c_text = normalize_text(c_row["resume_text"])
            # quick length filter
            if abs(len(r_text) - len(c_text)) / max(len(r_text), len(c_text), 1) > 0.10:
                continue
            score = fuzz.ratio(r_text, c_text)
            if score > best_score:
                best_score = score
            if score >= 95:
                break
        if best_score >= 95:
            fuzzy_res_matches += 1
            fuzzy_res_match_pairs.append((r_id, best_score))
            
    print(f"Fuzzy Resume Matches (ratio >= 95): {fuzzy_res_matches} / {len(ctrl_res_set)} ({fuzzy_res_matches/len(ctrl_res_set)*100:.2f}%)")

    # JDs
    ctrl_jd_texts = df_ctrl.drop_duplicates("jd_id")[["jd_id", "jd_text"]].reset_index(drop=True)
    c5_jd_texts = df_c5.drop_duplicates("jd_id")[["jd_id", "jd_text"]].reset_index(drop=True)
    
    fuzzy_jd_matches = 0
    for i, r_row in ctrl_jd_texts.iterrows():
        r_text = normalize_text(r_row["jd_text"])
        r_id = r_row["jd_id"]
        if r_id in exact_jd_matches:
            fuzzy_jd_matches += 1
            continue
        best_score = 0
        for _, c_row in c5_jd_texts.iterrows():
            c_text = normalize_text(c_row["jd_text"])
            if abs(len(r_text) - len(c_text)) / max(len(r_text), len(c_text), 1) > 0.10:
                continue
            score = fuzz.ratio(r_text, c_text)
            if score > best_score:
                best_score = score
            if score >= 95:
                break
        if best_score >= 95:
            fuzzy_jd_matches += 1
            
    print(f"Fuzzy JD Matches (ratio >= 95): {fuzzy_jd_matches} / {len(ctrl_jd_set)} ({fuzzy_jd_matches/len(ctrl_jd_set)*100:.2f}%)")

    # Save reconciliation report numbers
    reconciliation = {
        "control_raw_unique_resumes_strip": int(res_raw_strip.nunique()),
        "control_normalized_unique_resumes": int(res_norm.nunique()),
        "c5_unique_resumes": int(df_c5["resume_id"].nunique()),
        "exact_res_matches": int(len(exact_res_matches)),
        "exact_jd_matches": int(len(exact_jd_matches)),
        "exact_pair_matches": int(len(exact_pair_matches)),
        "fuzzy_res_matches_ge_95": int(fuzzy_res_matches),
        "fuzzy_jd_matches_ge_95": int(fuzzy_jd_matches),
    }
    with open("reports/triage_results/control_c5_reconciliation.json", "w") as f:
        json.dump(reconciliation, f, indent=2)
    print("Saved reports/triage_results/control_c5_reconciliation.json")

    print("\n=======================================================")
    print(" 2. RE-RUNNING CONTROL AS DIRECT 3-CLASS CLASSIFIER")
    print("=======================================================")
    
    # Re-run Control with SAME 3-class classifier on original_label and GroupKFold on resume_id
    # Prepare features
    res_emb, jd_emb = get_cached_embeddings(df_ctrl, dataset_name="control")
    diff = np.abs(res_emb - jd_emb)
    mult = res_emb * jd_emb
    cos = np.sum(res_emb * jd_emb, axis=1, keepdims=True)
    X_minilm = np.hstack([res_emb, jd_emb, diff, mult, cos])
    
    # TF-IDF
    X_tfidf = build_tfidf_features(df_ctrl["resume_text"].tolist(), df_ctrl["jd_text"].tolist(), max_features=300)
    
    # Target: original_label
    # Map classes consistently with C5: "No Fit", "Potential Fit", "Good Fit"
    print("Control class counts in df_ctrl:")
    print(df_ctrl["label_class"].value_counts())
    
    print("\n--- Running Control 3-Class Classifier (Cold-Start Resume GroupKFold) ---")
    tfidf_res_metrics = run_experiment_protocol(
        X_tfidf, df_ctrl, split_type="grouped", group_col="resume_id", is_regression=False, seed=42
    )
    print("TF-IDF Cold Resume:")
    print(f"  Acc: {tfidf_res_metrics['accuracy']:.4f}, Macro-F1: {tfidf_res_metrics['macro_f1']:.4f}, AUC: {tfidf_res_metrics.get('macro_roc_auc')}")
    print(f"  95% CI Acc: {tfidf_res_metrics['ci_95_accuracy']}, CI F1: {tfidf_res_metrics['ci_95_macro_f1']}")
    
    minilm_res_metrics = run_experiment_protocol(
        X_minilm, df_ctrl, split_type="grouped", group_col="resume_id", is_regression=False, seed=42
    )
    print("MiniLM Cold Resume:")
    print(f"  Acc: {minilm_res_metrics['accuracy']:.4f}, Macro-F1: {minilm_res_metrics['macro_f1']:.4f}, AUC: {minilm_res_metrics.get('macro_roc_auc')}")
    print(f"  95% CI Acc: {minilm_res_metrics['ci_95_accuracy']}, CI F1: {minilm_res_metrics['ci_95_macro_f1']}")

    print("\n--- Running Control 3-Class Classifier (Warm-Start Random Split) ---")
    tfidf_warm_metrics = run_experiment_protocol(
        X_tfidf, df_ctrl, split_type="random", is_regression=False, seed=42
    )
    minilm_warm_metrics = run_experiment_protocol(
        X_minilm, df_ctrl, split_type="random", is_regression=False, seed=42
    )
    print("TF-IDF Warm:")
    print(f"  Acc: {tfidf_warm_metrics['accuracy']:.4f}, Macro-F1: {tfidf_warm_metrics['macro_f1']:.4f}, AUC: {tfidf_warm_metrics.get('macro_roc_auc')}")
    print("MiniLM Warm:")
    print(f"  Acc: {minilm_warm_metrics['accuracy']:.4f}, Macro-F1: {minilm_warm_metrics['macro_f1']:.4f}, AUC: {minilm_warm_metrics.get('macro_roc_auc')}")

    results_out = {
        "cold_start_resume": {
            "tfidf_lightgbm": tfidf_res_metrics,
            "minilm_lightgbm": minilm_res_metrics
        },
        "warm_start_random": {
            "tfidf_lightgbm": tfidf_warm_metrics,
            "minilm_lightgbm": minilm_warm_metrics
        }
    }
    with open("reports/triage_results/control_direct_classification.json", "w") as f:
        json.dump(results_out, f, indent=2)
    print("Saved reports/triage_results/control_direct_classification.json")

if __name__ == "__main__":
    main()
