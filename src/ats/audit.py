import numpy as np
import pandas as pd
from transformers import AutoTokenizer


def audit_dataset_profile(df: pd.DataFrame, dataset_name: str, license_info: str = "Unknown", summary_info: str = ""):
    """Run comprehensive entity, length, truncation, and label audit on standardized dataframe."""
    n_rows = len(df)
    n_unique_res = df["resume_id"].nunique()
    n_unique_jds = df["jd_id"].nunique()
    n_unique_pairs = df.groupby(["resume_id", "jd_id"]).ngroups
    exact_duplicates = n_rows - n_unique_pairs

    res_counts = df["resume_id"].value_counts()
    jd_counts = df["jd_id"].value_counts()

    # Text length stats (characters and words)
    res_char_lens = df["resume_text"].str.len()
    jd_char_lens = df["jd_text"].str.len()
    res_word_lens = df["resume_text"].str.split().str.len()
    jd_word_lens = df["jd_text"].str.split().str.len()

    def summarize_series(s):
        return {
            "mean": float(s.mean()),
            "median": float(s.median()),
            "p90": float(np.percentile(s, 90)),
            "p99": float(np.percentile(s, 99)),
            "max": float(s.max()),
        }

    # MiniLM tokenization counts with 500-sample representative batch
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    tok_sample_size = min(len(df), 500)
    df_tok = df.sample(tok_sample_size, random_state=42)
    
    # Tokenize individually with max length check
    res_tokens = [len(tokenizer.encode(t, truncation=False)) for t in df_tok["resume_text"]]
    jd_tokens = [len(tokenizer.encode(t, truncation=False)) for t in df_tok["jd_text"]]

    res_tok_s = pd.Series(res_tokens)
    jd_tok_s = pd.Series(jd_tokens)

    res_pct_gt_256 = float((res_tok_s > 256).mean() * 100.0)
    res_pct_gt_512 = float((res_tok_s > 512).mean() * 100.0)
    jd_pct_gt_256 = float((jd_tok_s > 256).mean() * 100.0)
    jd_pct_gt_512 = float((jd_tok_s > 512).mean() * 100.0)

    # Label analysis
    class_counts = {str(k): int(v) for k, v in df["label_class"].value_counts().items()}
    has_continuous = df["label_continuous"].notnull().sum() > (0.5 * len(df))
    cont_stats = {}
    binned_stats = {}

    if has_continuous:
        cont = df["label_continuous"].dropna()
        cont_stats = {
            "count": int(len(cont)),
            "mean": float(cont.mean()),
            "std": float(cont.std()),
            "min": float(cont.min()),
            "p25": float(np.percentile(cont, 25)),
            "median": float(np.percentile(cont, 50)),
            "p75": float(np.percentile(cont, 75)),
            "max": float(cont.max()),
        }
        
        # Pre-registered bins: <40, 40-70, >70
        b_low = int((cont < 40.0).sum())
        b_mid = int(((cont >= 40.0) & (cont <= 70.0)).sum())
        b_high = int((cont > 70.0).sum())
        med_val = float(cont.median())
        b_med_low = int((cont <= med_val).sum())
        b_med_high = int((cont > med_val).sum())

        binned_stats = {
            "preregistered_bins": {
                "<40": b_low,
                "40-70": b_mid,
                ">70": b_high,
            },
            "median_binary_split": {
                "median_threshold": med_val,
                "<=median": b_med_low,
                ">median": b_med_high,
            }
        }

    return {
        "dataset_name": dataset_name,
        "license": license_info,
        "summary": summary_info,
        "rows": n_rows,
        "entities": {
            "unique_resumes": n_unique_res,
            "unique_jds": n_unique_jds,
            "unique_pairs": n_unique_pairs,
            "exact_duplicate_rows": exact_duplicates,
            "rows_per_resume": summarize_series(res_counts),
            "rows_per_jd": summarize_series(jd_counts),
        },
        "text_lengths": {
            "resume_chars": summarize_series(res_char_lens),
            "jd_chars": summarize_series(jd_char_lens),
            "resume_words": summarize_series(res_word_lens),
            "jd_words": summarize_series(jd_word_lens),
        },
        "token_truncation_minilm": {
            "sample_size": tok_sample_size,
            "resume_tokens_mean": float(res_tok_s.mean()),
            "resume_pct_over_256": res_pct_gt_256,
            "resume_pct_over_512": res_pct_gt_512,
            "jd_tokens_mean": float(jd_tok_s.mean()),
            "jd_pct_over_256": jd_pct_gt_256,
            "jd_pct_over_512": jd_pct_gt_512,
        },
        "label_analysis": {
            "classes": class_counts,
            "has_continuous": has_continuous,
            "continuous_stats": cont_stats,
            "continuous_binned": binned_stats,
        }
    }
