import pandas as pd
import numpy as np
from huggingface_hub import hf_hub_download
from .base import BaseDatasetAdapter, compute_sha1


class C3MatchingAdapter(BaseDatasetAdapter):
    """Adapter for rahulrajpu676/job-resume-matching (3 CSVs: cv, job, matches)."""

    def __init__(self, repo_id: str = "rahulrajpu676/job-resume-matching", max_rows: int = None):
        super().__init__(name=repo_id)
        self.max_rows = max_rows

    def load_raw(self) -> pd.DataFrame:
        p_cv = hf_hub_download(self.name, "cv.csv", repo_type="dataset")
        p_job = hf_hub_download(self.name, "job.csv", repo_type="dataset")
        p_matches = hf_hub_download(self.name, "matches.csv", repo_type="dataset")

        df_cv = pd.read_csv(p_cv)[["candidate_id", "resume_text"]]
        df_job = pd.read_csv(p_job)[["job_id", "job_description", "job_text"]]
        df_matches = pd.read_csv(p_matches)

        if self.max_rows is not None and len(df_matches) > self.max_rows:
            df_matches = df_matches.sample(self.max_rows, random_state=42)

        # Merge
        merged = df_matches.merge(df_cv, on="candidate_id", how="inner")
        merged = merged.merge(df_job, on="job_id", how="inner")
        return merged

    def standardize(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        df = df_raw.copy()
        
        # Choose jd_text
        jd_text = df["job_description"].fillna("").astype(str).str.strip()
        fallback_jd = df["job_text"].fillna("").astype(str).str.strip()
        final_jd = np.where(jd_text != "", jd_text, fallback_jd)

        resume_texts = df["resume_text"].fillna("").astype(str).str.strip()
        final_jd = pd.Series(final_jd, index=df.index).str.strip()

        mask = (resume_texts != "") & (final_jd != "")
        df = df[mask].copy()

        df["resume_text"] = resume_texts[mask]
        df["jd_text"] = final_jd[mask]
        df["resume_id"] = df["resume_text"].apply(compute_sha1)
        df["jd_id"] = df["jd_text"].apply(compute_sha1)

        # Continuous target: scale 0-1 final_score to 0-100 scale
        scores = pd.to_numeric(df["final_score"], errors="coerce") * 100.0
        df["label_continuous"] = scores
        df["label_class"] = df["label_strength"].fillna("unknown").astype(str)
        df["source_dataset"] = self.name

        return df

