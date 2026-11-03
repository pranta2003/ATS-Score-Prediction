import abc
import hashlib
import re
import pandas as pd
import numpy as np


STANDARD_COLUMNS = [
    "resume_id",
    "jd_id",
    "resume_text",
    "jd_text",
    "label_continuous",
    "label_class",
    "source_dataset",
]


def normalize_text(text: str) -> str:
    """Normalize text by lowercasing and collapsing whitespace for deterministic hashing."""
    if not isinstance(text, str):
        return ""
    return " ".join(text.strip().lower().split())


def compute_sha1(text: str) -> str:
    """Compute SHA1 hash of normalized text."""
    norm = normalize_text(text)
    return hashlib.sha1(norm.encode("utf-8")).hexdigest()


class BaseDatasetAdapter(abc.ABC):
    """Abstract base class for dataset adapters."""

    def __init__(self, name: str):
        self.name = name

    @abc.abstractmethod
    def load_raw(self) -> pd.DataFrame:
        """Load raw dataset records and return as a DataFrame."""
        pass

    @abc.abstractmethod
    def standardize(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        """Map raw columns to standard schema and format labels."""
        pass

    def load(self) -> pd.DataFrame:
        """Load, standardize, validate, and return the standard table."""
        df_raw = self.load_raw()
        df = self.standardize(df_raw)
        return self.validate(df)

    def validate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate standard table schema, non-empty texts, and deterministic IDs."""
        missing = [c for c in STANDARD_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"Adapter {self.name} missing required columns: {missing}")

        # Reorder to standard columns
        df = df[STANDARD_COLUMNS].copy()

        # Check types
        if not pd.api.types.is_string_dtype(df["resume_text"]) and not df["resume_text"].dtype == object:
            raise TypeError("resume_text must be string/object dtype")
        if not pd.api.types.is_string_dtype(df["jd_text"]) and not df["jd_text"].dtype == object:
            raise TypeError("jd_text must be string/object dtype")

        # Disallow empty texts
        empty_res = (df["resume_text"].fillna("").astype(str).str.strip() == "").sum()
        empty_jd = (df["jd_text"].fillna("").astype(str).str.strip() == "").sum()
        if empty_res > 0 or empty_jd > 0:
            raise ValueError(
                f"Adapter {self.name} contains empty texts: {empty_res} empty resumes, {empty_jd} empty JDs"
            )

        # Validate deterministic IDs
        sample_size = min(len(df), 200)
        sample = df.sample(sample_size, random_state=42)
        for _, row in sample.iterrows():
            expected_res_id = compute_sha1(row["resume_text"])
            expected_jd_id = compute_sha1(row["jd_text"])
            if row["resume_id"] != expected_res_id:
                raise ValueError(
                    f"resume_id mismatch: got {row['resume_id']}, expected {expected_res_id}"
                )
            if row["jd_id"] != expected_jd_id:
                raise ValueError(
                    f"jd_id mismatch: got {row['jd_id']}, expected {expected_jd_id}"
                )

        return df.reset_index(drop=True)

