import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


def get_cached_embeddings(df: pd.DataFrame, dataset_name: str, cache_dir: str = "cache/embeddings", model=None):
    """
    Encode unique resume texts and JD texts, caching to disk.
    Returns resume embeddings array (N, 384) and JD embeddings array (N, 384).
    """
    os.makedirs(cache_dir, exist_ok=True)
    clean_name = dataset_name.replace("/", "_").replace("-", "_")
    res_cache_path = os.path.join(cache_dir, f"{clean_name}_res_emb.npy")
    jd_cache_path = os.path.join(cache_dir, f"{clean_name}_jd_emb.npy")
    id_cache_path = os.path.join(cache_dir, f"{clean_name}_ids.parquet")

    # If already cached and matching shape
    if os.path.exists(res_cache_path) and os.path.exists(jd_cache_path) and os.path.exists(id_cache_path):
        cached_ids = pd.read_parquet(id_cache_path)
        if len(cached_ids) == len(df) and (cached_ids["resume_id"].values == df["resume_id"].values).all():
            res_emb = np.load(res_cache_path)
            jd_emb = np.load(jd_cache_path)
            return res_emb, jd_emb

    if model is None:
        model = SentenceTransformer("all-MiniLM-L6-v2")

    # Encode unique resumes
    unique_res = df[["resume_id", "resume_text"]].drop_duplicates("resume_id")
    res_enc = model.encode(unique_res["resume_text"].tolist(), batch_size=64, show_progress_bar=False, normalize_embeddings=True)
    res_map = dict(zip(unique_res["resume_id"], res_enc))

    # Encode unique JDs
    unique_jds = df[["jd_id", "jd_text"]].drop_duplicates("jd_id")
    jd_enc = model.encode(unique_jds["jd_text"].tolist(), batch_size=64, show_progress_bar=False, normalize_embeddings=True)
    jd_map = dict(zip(unique_jds["jd_id"], jd_enc))

    # Map back to full dataframe rows
    res_emb = np.array([res_map[rid] for rid in df["resume_id"]])
    jd_emb = np.array([jd_map[jid] for jid in df["jd_id"]])

    # Save cache
    np.save(res_cache_path, res_emb)
    np.save(jd_cache_path, jd_emb)
    df[["resume_id", "jd_id"]].to_parquet(id_cache_path)

    return res_emb, jd_emb

