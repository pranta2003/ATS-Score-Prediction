import os
import subprocess
import pandas as pd
import numpy as np
from .base import BaseDatasetAdapter, compute_sha1


class C4RecruitmentAdapter(BaseDatasetAdapter):
    """Adapter for Kaggle surendra365/recruitement-dataset."""

    def __init__(self, data_path: str = "data/c4_surendra/job_applicant_dataset.csv"):
        super().__init__(name="surendra365/recruitment-dataset")
        self.data_path = data_path

    def load_raw(self) -> pd.DataFrame:
        if not os.path.exists(self.data_path):
            os.makedirs(os.path.dirname(self.data_path), exist_ok=True)
            subprocess.run(
                ["kaggle", "datasets", "download", "-d", "surendra365/recruitement-dataset", "-p", "data/c4_surendra", "--unzip"],
                check=True
            )
        return pd.read_csv(self.data_path)

    def standardize(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        df = df_raw.copy()
        resume_texts = df["Resume"].fillna("").astype(str).str.strip()
        jd_texts = df["Job Description"].fillna("").astype(str).str.strip()

        mask = (resume_texts != "") & (jd_texts != "")
        df = df[mask].copy()

        df["resume_text"] = resume_texts[mask]
        df["jd_text"] = jd_texts[mask]
        df["resume_id"] = df["resume_text"].apply(compute_sha1)
        df["jd_id"] = df["jd_text"].apply(compute_sha1)

        # Discrete binary target: 0 or 1
        df["label_continuous"] = np.nan
        df["label_class"] = df["Best Match"].astype(str)
        df["source_dataset"] = self.name

        return df

