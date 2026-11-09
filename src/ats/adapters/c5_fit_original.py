import pandas as pd
import numpy as np
from .base import BaseDatasetAdapter, compute_sha1


class C5FitOriginalAdapter(BaseDatasetAdapter):
    """Adapter for cnamuangtoun/resume-job-description-fit (the original source of 0xnbk)."""

    def __init__(self, repo_id: str = "cnamuangtoun/resume-job-description-fit", split: str = "all"):
        super().__init__(name=repo_id)
        self.split = split

    def load_raw(self) -> pd.DataFrame:
        from datasets import load_dataset
        ds = load_dataset(self.name)
        if self.split in ds:
            return ds[self.split].to_pandas()
        return pd.concat([ds[s].to_pandas() for s in ds.keys()], ignore_index=True)

    def standardize(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        df = df_raw.copy()
        resume_texts = df["resume_text"].fillna("").astype(str).str.strip()
        jd_texts = df["job_description_text"].fillna("").astype(str).str.strip()

        mask = (resume_texts != "") & (jd_texts != "")
        df = df[mask].copy()

        df["resume_text"] = resume_texts[mask]
        df["jd_text"] = jd_texts[mask]
        df["resume_id"] = df["resume_text"].apply(compute_sha1)
        df["jd_id"] = df["jd_text"].apply(compute_sha1)

        # C5 natively has no continuous score, only 3-class label
        df["label_continuous"] = np.nan
        df["label_class"] = df["label"].astype(str)
        df["source_dataset"] = self.name

        return df

