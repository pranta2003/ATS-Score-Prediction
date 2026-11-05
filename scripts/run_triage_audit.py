import os
import sys
import json
import argparse
import pandas as pd
import numpy as np

# Ensure root in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.ats.adapters import get_adapter
from src.ats.audit import audit_dataset_profile
from src.ats.provenance import test_label_provenance
from src.ats.embeddings import get_cached_embeddings
from src.ats.modeling import build_tfidf_features, run_experiment_protocol
from sentence_transformers import SentenceTransformer


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types."""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


DATASET_METADATA = {
    "control": {
        "repo_id": "0xnbk/resume-ats-score-v1-en",
        "license": "Unknown / Research",
        "summary": "Derived from cnamuangtoun dataset. Resumes and JDs concatenated with ' SEP '. ats_score is an artificial projection of 3 original labels with added gaussian noise.",
        "leakage_cols": ["original_label (if predicting ats_score, eta^2=0.987; mutually derived)"],
        "is_regression": True,
        "sample_size": None,
    },
    "c1": {
        "repo_id": "harsh72877287/job_resume_fit",
        "license": "other",
        "summary": "Contains 2,385 resume-job pairs across 23 job categories. Evaluates skill matching via direct AI-based skill match score, skill string match score, and fuzzy match score.",
        "leakage_cols": ["skill_string_match_score", "fuzzy_match_score", "ai_matched_skills", "category"],
        "is_regression": True,
        "sample_size": None,
    },
    "c2": {
        "repo_id": "netsol/resume-score-details",
        "license": "cc / unknown",
        "summary": "1,031 samples of resumes and job descriptions generated and assessed using GPT-4o. Includes macro/micro evaluation scores and requirement check criteria.",
        "leakage_cols": ["micro_scores", "requirements", "valid_resume_and_jd", "justification"],
        "is_regression": True,
        "sample_size": None,
    },
    "c3": {
        "repo_id": "rahulrajpu676/job-resume-matching",
        "license": "mit",
        "summary": "Multi-table dataset (CV, Job, Matches) for matching/recommendation. Matches table computes final_score from semantic_score, skill_score, position_score, experience_score, etc.",
        "leakage_cols": ["semantic_score", "skill_score", "position_score", "category_score", "experience_score", "education_score", "raw_score"],
        "is_regression": True,
        "sample_size": 2500,
    },
    "c4": {
        "repo_id": "surendra365/recruitment-dataset",
        "license": "DbCL-1.0",
        "summary": "10,000 synthetic recruitment records with applicant demographics (Age, Gender, Race, Ethnicity), resume, job role, job description, and binary Best Match target.",
        "leakage_cols": ["Job Applicant Name", "Age", "Gender", "Race", "Ethnicity", "Job Roles"],
        "is_regression": False,
        "sample_size": 2500,
    },
    "c5": {
        "repo_id": "cnamuangtoun/resume-job-description-fit",
        "license": "unknown",
        "summary": "Original source dataset of 0xnbk containing 8,000 resume and job description pairs with 3 human candidate fit labels (No Fit, Potential Fit, Good Fit).",
        "leakage_cols": [],
        "is_regression": False,
        "sample_size": None,
    },
}


def run_single_dataset_triage(dataset_key: str, model=None):
    print(f"\n=======================================================", flush=True)
    print(f"   RUNNING TRIAGE AUDIT FOR: {dataset_key.upper()}", flush=True)
    print(f"=======================================================", flush=True)
    meta = DATASET_METADATA[dataset_key]
    
    # 1. Load data
    kwargs = {}
    if meta["sample_size"] is not None and dataset_key == "c3":
        kwargs["max_rows"] = meta["sample_size"]
    
    adapter = get_adapter(dataset_key, **kwargs)
    df = adapter.load()
    if meta["sample_size"] is not None and len(df) > meta["sample_size"]:
        df = df.sample(meta["sample_size"], random_state=42).reset_index(drop=True)
    
    print(f"Loaded {len(df)} standardized records. Unique resumes: {df['resume_id'].nunique()}, Unique JDs: {df['jd_id'].nunique()}", flush=True)

    # 2. Audit profile
    print("\n--- Running Audit Profile ---", flush=True)
    audit_res = audit_dataset_profile(
        df,
        dataset_name=meta["repo_id"],
        license_info=meta["license"],
        summary_info=meta["summary"],
    )
    print(f"Token stats: Resume > 256 toks: {audit_res['token_truncation_minilm']['resume_pct_over_256']:.1f}%, JD > 256 toks: {audit_res['token_truncation_minilm']['jd_pct_over_256']:.1f}%", flush=True)

    # 3. Cache & extract embeddings first (fast, unique texts only!)
    print("\n--- Encoding unique texts and caching MiniLM embeddings ---", flush=True)
    res_emb, jd_emb = get_cached_embeddings(df, dataset_name=dataset_key, model=model)
    diff = np.abs(res_emb - jd_emb)
    mult = res_emb * jd_emb
    cos = np.sum(res_emb * jd_emb, axis=1, keepdims=True)
    X_minilm = np.hstack([res_emb, jd_emb, diff, mult, cos])
    print(f"MiniLM feature matrix shape: {X_minilm.shape}", flush=True)

    # 4. Provenance test (re-uses cached embeddings for 0-second cosine calculation)
    print("\n--- Running Label Provenance Test ---", flush=True)
    prov_res = test_label_provenance(
        df,
        dataset_name=meta["repo_id"],
        card_info=meta["summary"],
        res_emb=res_emb,
        jd_emb=jd_emb,
        model=model,
    )
    print(f"Provenance Verdict: {prov_res['provenance']}", flush=True)
    print(f"Evidence: {prov_res['evidence']}", flush=True)
    if prov_res['lexical_only_r2'] is not None:
        print(f"Lexical-only R2: {prov_res['lexical_only_r2']:.4f}", flush=True)
    if prov_res['lexical_only_accuracy'] is not None:
        print(f"Lexical-only Accuracy: {prov_res['lexical_only_accuracy']:.4f}", flush=True)

    # 5. TF-IDF features
    print("\n--- Building TF-IDF Features ---", flush=True)
    X_tfidf = build_tfidf_features(df["resume_text"].tolist(), df["jd_text"].tolist(), max_features=300)
    print(f"TF-IDF feature matrix shape: {X_tfidf.shape}", flush=True)

    is_reg = meta["is_regression"] and df["label_continuous"].notnull().sum() > (0.5 * len(df))

    # Protocol A: Cold-start Resume (GroupKFold on resume_id)
    print("\n[Baseline] Cold-Start by Resume (GroupKFold on resume_id)...", flush=True)
    res_tfidf_cold_res = run_experiment_protocol(
        X_tfidf, df, split_type="grouped", group_col="resume_id", is_regression=is_reg
    )
    print(f"  TF-IDF Cold Resume: Acc={res_tfidf_cold_res.get('accuracy', 0):.4f}, F1={res_tfidf_cold_res.get('macro_f1', 0):.4f}, Maj={res_tfidf_cold_res.get('majority_baseline', 0):.4f}", flush=True)

    res_minilm_cold_res = run_experiment_protocol(
        X_minilm, df, split_type="grouped", group_col="resume_id", is_regression=is_reg
    )
    print(f"  MiniLM Cold Resume: Acc={res_minilm_cold_res.get('accuracy', 0):.4f}, F1={res_minilm_cold_res.get('macro_f1', 0):.4f}, Maj={res_minilm_cold_res.get('majority_baseline', 0):.4f}", flush=True)
    if is_reg:
        print(f"  MiniLM Cold Resume Reg: R2={res_minilm_cold_res.get('r2', 0):.4f}, MAE={res_minilm_cold_res.get('mae', 0):.4f}, Rho={res_minilm_cold_res.get('spearman_rho', 0):.4f}", flush=True)

    # Protocol B: Cold-start JD (GroupKFold on jd_id) if JDs repeat
    n_unique_jds = df["jd_id"].nunique()
    res_tfidf_cold_jd = None
    res_minilm_cold_jd = None
    if 5 <= n_unique_jds <= 150 and (len(df) / n_unique_jds) >= 3.0:
        print(f"\n[Baseline] Cold-Start by JD (GroupKFold on jd_id, {n_unique_jds} unique JDs)...", flush=True)
        res_tfidf_cold_jd = run_experiment_protocol(
            X_tfidf, df, split_type="grouped", group_col="jd_id", is_regression=is_reg
        )
        print(f"  TF-IDF Cold JD: Acc={res_tfidf_cold_jd.get('accuracy', 0):.4f}, F1={res_tfidf_cold_jd.get('macro_f1', 0):.4f}", flush=True)
        res_minilm_cold_jd = run_experiment_protocol(
            X_minilm, df, split_type="grouped", group_col="jd_id", is_regression=is_reg
        )
        print(f"  MiniLM Cold JD: Acc={res_minilm_cold_jd.get('accuracy', 0):.4f}, F1={res_minilm_cold_jd.get('macro_f1', 0):.4f}", flush=True)

    # Protocol C: Warm-start (Random KFold)
    print("\n[Baseline] Warm-Start Random Split (KFold, shuffle=True)...", flush=True)
    res_tfidf_warm = run_experiment_protocol(
        X_tfidf, df, split_type="random", is_regression=is_reg
    )
    print(f"  TF-IDF Warm: Acc={res_tfidf_warm.get('accuracy', 0):.4f}, F1={res_tfidf_warm.get('macro_f1', 0):.4f}", flush=True)
    res_minilm_warm = run_experiment_protocol(
        X_minilm, df, split_type="random", is_regression=is_reg
    )
    print(f"  MiniLM Warm: Acc={res_minilm_warm.get('accuracy', 0):.4f}, F1={res_minilm_warm.get('macro_f1', 0):.4f}", flush=True)

    # Compile result package
    full_result = {
        "dataset_key": dataset_key,
        "metadata": meta,
        "audit": audit_res,
        "provenance": prov_res,
        "leakage_columns_excluded": meta["leakage_cols"],
        "baselines": {
            "cold_start_resume": {
                "tfidf_lightgbm": res_tfidf_cold_res,
                "minilm_lightgbm": res_minilm_cold_res,
            },
            "cold_start_jd": {
                "tfidf_lightgbm": res_tfidf_cold_jd,
                "minilm_lightgbm": res_minilm_cold_jd,
            } if res_tfidf_cold_jd is not None else None,
            "warm_start_random": {
                "tfidf_lightgbm": res_tfidf_warm,
                "minilm_lightgbm": res_minilm_warm,
            },
        }
    }

    os.makedirs("reports/triage_results", exist_ok=True)
    out_file = f"reports/triage_results/{dataset_key}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(full_result, f, indent=2, cls=NumpyEncoder)
    print(f"\n[OK] Results successfully saved to {out_file}", flush=True)
    return full_result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="all", help="Dataset key or 'all'")
    args = parser.parse_args()

    print("Pre-loading MiniLM model...", flush=True)
    model = SentenceTransformer("all-MiniLM-L6-v2")

    if args.dataset == "all":
        target_keys = ["control", "c1", "c2", "c3", "c4", "c5"]
    else:
        target_keys = [args.dataset]

    for key in target_keys:
        run_single_dataset_triage(key, model=model)


if __name__ == "__main__":
    main()
