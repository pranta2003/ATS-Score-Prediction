import os
import glob
import json
import pandas as pd
import numpy as np
from huggingface_hub import snapshot_download
from .base import BaseDatasetAdapter, compute_sha1


class C2DetailsAdapter(BaseDatasetAdapter):
    """Adapter for netsol/resume-score-details (and copies saryal01, callmeech, saptarshideveloper)."""

    def __init__(self, repo_id: str = "netsol/resume-score-details"):
        super().__init__(name=repo_id)

    def load_raw(self) -> pd.DataFrame:
        path = snapshot_download(self.name, repo_type="dataset", allow_patterns="*.json")
        json_files = glob.glob(os.path.join(path, "*.json"))

        records = []
        for jf in json_files:
            try:
                with open(jf, "r", encoding="utf-8") as f:
                    d = json.load(f)
                    inp = d.get("input", {})
                    out = d.get("output", {})
                    res_text = inp.get("resume", "")
                    jd_text = inp.get("job_description", "")
                    
                    scores = out.get("scores", {}) if isinstance(out, dict) else {}
                    agg = scores.get("aggregated_scores", {}) if isinstance(scores, dict) else {}
                    macro = agg.get("macro_scores") if isinstance(agg, dict) else None
                    micro = agg.get("micro_scores") if isinstance(agg, dict) else None
                    
                    fname = os.path.basename(jf)
                    prefix = fname.split("_")[0]
                    
                    records.append({
                        "file": fname,
                        "prefix": prefix,
                        "resume_text": str(res_text),
                        "jd_text": str(jd_text),
                        "macro_score": macro,
                        "micro_score": micro,
                        "valid": out.get("valid_resume_and_jd") if isinstance(out, dict) else None
                    })
            except Exception:
                continue

        return pd.DataFrame(records)

    def standardize(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        df = df_raw.copy()
        resume_texts = df["resume_text"].fillna("").astype(str).str.strip()
        jd_texts = df["jd_text"].fillna("").astype(str).str.strip()

        mask = (resume_texts != "") & (jd_texts != "")
        df = df[mask].copy()

        df["resume_text"] = resume_texts[mask]
        df["jd_text"] = jd_texts[mask]
        df["resume_id"] = df["resume_text"].apply(compute_sha1)
        df["jd_id"] = df["jd_text"].apply(compute_sha1)

        # Continuous target: scale 0-10 macro_score to 0-100 scale for ATS compatibility
        macro = pd.to_numeric(df["macro_score"], errors="coerce")
        df["label_continuous"] = macro * 10.0
        df["label_class"] = df["prefix"].astype(str)
        df["source_dataset"] = self.name

        return df
