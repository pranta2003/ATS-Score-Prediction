import os
import pandas as pd
from datasets import load_dataset
from .base import BaseDatasetAdapter, compute_sha1


class ControlAdapter(BaseDatasetAdapter):
    """Adapter for 0xnbk/resume-ats-score-v1-en (Phase 1 dataset)."""

    def __init__(self, local_path: str = None, split: str = "all"):
        super().__init__(name="0xnbk/resume-ats-score-v1-en")
        self.local_path = local_path
        self.split = split

    def load_raw(self) -> pd.DataFrame:
        # Check local files first
        train_path = r"C:\Users\Hp\Downloads\ats_v2\train.csv"
        val_path = r"C:\Users\Hp\Downloads\ats_v2\validation.csv"
        if os.path.exists(train_path) and os.path.exists(val_path):
            df_tr = pd.read_csv(train_path)
            df_val = pd.read_csv(val_path)
            if self.split == "train":
                return df_tr
            elif self.split == "validation":
                return df_val
            else:
                return pd.concat([df_tr, df_val], ignore_index=True)
        else:
            ds = load_dataset("0xnbk/resume-ats-score-v1-en")
            if self.split in ds:
                return ds[self.split].to_pandas()
            return pd.concat([ds[s].to_pandas() for s in ds.keys()], ignore_index=True)

    def standardize(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        df = df_raw.copy()
        
        # Split text on first ' SEP '
        split_res = df["text"].astype(str).str.split(" SEP ", n=1, expand=True)
        resume_texts = split_res[0].fillna("").str.strip()
        jd_texts = split_res[1].fillna("").str.strip() if split_res.shape[1] > 1 else pd.Series([""] * len(df))

        # Filter out rows where either text is empty
        mask = (resume_texts != "") & (jd_texts != "")
        df = df[mask].copy()
        df["resume_text"] = resume_texts[mask]
        df["jd_text"] = jd_texts[mask]

        df["resume_id"] = df["resume_text"].apply(compute_sha1)
        df["jd_id"] = df["jd_text"].apply(compute_sha1)
        df["label_continuous"] = pd.to_numeric(df["ats_score"], errors="coerce")
        df["label_class"] = df["original_label"].astype(str)
        df["source_dataset"] = self.name

        return df
