import pandas as pd
import numpy as np
from datasets import load_dataset
from .base import BaseDatasetAdapter, compute_sha1


class C1FitAdapter(BaseDatasetAdapter):
    """Adapter for harsh72877287/job_resume_fit (and identical batuhanmtl/job_resume_fit)."""

    def __init__(self, repo_id: str = "harsh72877287/job_resume_fit"):
        super().__init__(name=repo_id)

    def load_raw(self) -> pd.DataFrame:
        ds = load_dataset(self.name, split="train")
        return ds.to_pandas()

    def standardize(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        df = df_raw.copy()
        resume_texts = df["resume_text"].fillna("").astype(str).str.strip()
        jd_texts = df["job_text"].fillna("").astype(str).str.strip()

        mask = (resume_texts != "") & (jd_texts != "")
        df = df[mask].copy()

        df["resume_text"] = resume_texts[mask]
        df["jd_text"] = jd_texts[mask]
        df["resume_id"] = df["resume_text"].apply(compute_sha1)
        df["jd_id"] = df["jd_text"].apply(compute_sha1)

        # Continuous target: ai_match_score
        scores = pd.to_numeric(df["ai_match_score"], errors="coerce")
        df["label_continuous"] = scores

        # Pre-registered 3-class bins: <40, 40-70, >70
        def to_class(s):
            if pd.isna(s):
                return "Unknown"
            if s < 40.0:
                return "<40"
            elif s <= 70.0:
                return "40-70"
            else:
                return ">70"

        df["label_class"] = scores.apply(to_class)
        df["source_dataset"] = self.name

        return df
