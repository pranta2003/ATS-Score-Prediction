import os
import re
import numpy as np
import pandas as pd
from .adapters.base import compute_sha1, normalize_text


def split_into_sentences(text: str, min_chars: int = 15):
    """Split text into sentences using regex boundary."""
    if not isinstance(text, str) or not text.strip():
        return [""]
    # Split on punctuation followed by whitespace or newlines
    raw_sents = re.split(r"(?<=[.!?\n])\s+", text)
    sents = [s.strip() for s in raw_sents if len(s.strip()) >= min_chars]
    if not sents:
        # Fallback to entire text or first 200 chars
        sents = [text[:200].strip()]
    return sents


def split_into_chunks(text: str, max_words: int = 120, overlap: int = 20):
    """Chunk text by word count with overlap."""
    words = text.split()
    if len(words) <= max_words:
        return [text]
    chunks = []
    step = max_words - overlap
    for i in range(0, len(words), step):
        chunk = " ".join(words[i:i + max_words])
        if chunk:
            chunks.append(chunk)
    return chunks


def build_and_cache_document_embeddings(
    df: pd.DataFrame,
    dataset_name: str,
    cache_dir: str = "cache/embeddings",
    model_name: str = "all-MiniLM-L6-v2",
    include_sentences: bool = False,
    model=None,
):
    """
    Encode unique documents (resumes and JDs) once.
    Computes and caches:
    1. Document-level embeddings (384-d, normalized)
    2. Sentence-level embeddings for late interaction (if include_sentences=True)
    Returns:
    - doc_emb_map: dict[text_id, np.ndarray (384,)]
    - sent_emb_map: dict[text_id, np.ndarray (num_sents, 384)] or empty dict
    """
    os.makedirs(cache_dir, exist_ok=True)
    clean_name = dataset_name.replace("/", "_").replace("-", "_")
    
    doc_cache_path = os.path.join(cache_dir, f"{clean_name}_doc_emb.npy")
    doc_meta_path = os.path.join(cache_dir, f"{clean_name}_doc_meta.parquet")
    sent_cache_path = os.path.join(cache_dir, f"{clean_name}_sent_emb.npy")
    sent_meta_path = os.path.join(cache_dir, f"{clean_name}_sent_meta.parquet")

    # Check legacy pair cache migration first (instant reuse of precomputed embeddings)
    legacy_res = os.path.join(cache_dir, f"{clean_name}_res_emb.npy")
    legacy_jd = os.path.join(cache_dir, f"{clean_name}_jd_emb.npy")
    legacy_ids = os.path.join(cache_dir, f"{clean_name}_ids.parquet")
    if not os.path.exists(doc_cache_path) and os.path.exists(legacy_res) and os.path.exists(legacy_jd) and os.path.exists(legacy_ids):
        df_ids = pd.read_parquet(legacy_ids)
        r_arr = np.load(legacy_res)
        j_arr = np.load(legacy_jd)
        doc_dict = {}
        meta_recs = []
        for i, rid in enumerate(df_ids["resume_id"]):
            if rid not in doc_dict:
                doc_dict[rid] = r_arr[i]
                meta_recs.append({"text_id": rid, "doc_type": "resume", "array_idx": len(meta_recs)})
        for i, jid in enumerate(df_ids["jd_id"]):
            if jid not in doc_dict:
                doc_dict[jid] = j_arr[i]
                meta_recs.append({"text_id": jid, "doc_type": "jd", "array_idx": len(meta_recs)})
        np.save(doc_cache_path, np.array(list(doc_dict.values()), dtype=np.float32))
        pd.DataFrame(meta_recs).to_parquet(doc_meta_path)

    # Collect unique documents
    unique_resumes = df[["resume_id", "resume_text"]].drop_duplicates("resume_id").rename(
        columns={"resume_id": "text_id", "resume_text": "text"}
    )
    unique_resumes["doc_type"] = "resume"
    
    unique_jds = df[["jd_id", "jd_text"]].drop_duplicates("jd_id").rename(
        columns={"jd_id": "text_id", "jd_text": "text"}
    )
    unique_jds["doc_type"] = "jd"
    
    unique_docs = pd.concat([unique_resumes, unique_jds], ignore_index=True).drop_duplicates("text_id").reset_index(drop=True)
    all_text_ids = set(unique_docs["text_id"])

    # Load doc embeddings if already cached
    doc_emb_map = {}
    if os.path.exists(doc_cache_path) and os.path.exists(doc_meta_path):
        doc_meta = pd.read_parquet(doc_meta_path)
        cached_ids = set(doc_meta["text_id"])
        if all_text_ids.issubset(cached_ids):
            doc_arr = np.load(doc_cache_path)
            doc_id_to_idx = dict(zip(doc_meta["text_id"], doc_meta["array_idx"]))
            doc_emb_map = {tid: doc_arr[idx] for tid, idx in doc_id_to_idx.items() if tid in all_text_ids}

    # If some documents are missing from doc cache, encode unique docs
    if len(doc_emb_map) < len(all_text_ids):
        if model is None:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(model_name)
        missing_ids = all_text_ids - set(doc_emb_map.keys())
        missing_docs = unique_docs[unique_docs["text_id"].isin(missing_ids)].reset_index(drop=True)
        print(f"Encoding {len(missing_docs)} unique documents...", flush=True)
        texts_to_encode = missing_docs["text"].tolist()
        encs = model.encode(texts_to_encode, batch_size=64, show_progress_bar=False, normalize_embeddings=True)
        for tid, vec in zip(missing_docs["text_id"], encs):
            doc_emb_map[tid] = np.array(vec, dtype=np.float32)

    # Sentence embeddings (only if requested)
    sent_emb_map = {}
    if include_sentences:
        if os.path.exists(sent_cache_path) and os.path.exists(sent_meta_path):
            sent_arr = np.load(sent_cache_path)
            sent_meta = pd.read_parquet(sent_meta_path)
            cached_sent_ids = set(sent_meta["text_id"])
            if all_text_ids.issubset(cached_sent_ids):
                for _, row in sent_meta.iterrows():
                    tid = row["text_id"]
                    if tid in all_text_ids:
                        s_start, s_end = int(row["sent_start"]), int(row["sent_end"])
                        sent_emb_map[tid] = sent_arr[s_start:s_end]
                return doc_emb_map, sent_emb_map

        if model is None:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(model_name)
        all_sentences = []
        sent_meta_records = []
        current_sent_offset = 0
        for idx, row in unique_docs.iterrows():
            text_id = row["text_id"]
            text = str(row["text"])
            sents = split_into_sentences(text, min_chars=15)
            n_s = len(sents)
            all_sentences.extend(sents)
            sent_meta_records.append({
                "text_id": text_id,
                "sent_start": current_sent_offset,
                "sent_end": current_sent_offset + n_s,
                "num_sentences": n_s,
            })
            current_sent_offset += n_s

        print(f"Encoding {len(all_sentences)} total sentences for late-interaction...", flush=True)
        sent_arr = model.encode(all_sentences, batch_size=128, show_progress_bar=False, normalize_embeddings=True)
        sent_arr = np.array(sent_arr, dtype=np.float32)
        sent_meta_df = pd.DataFrame(sent_meta_records)

        np.save(sent_cache_path, sent_arr)
        sent_meta_df.to_parquet(sent_meta_path)
        for _, row in sent_meta_df.iterrows():
            tid = row["text_id"]
            sent_emb_map[tid] = sent_arr[row["sent_start"]:row["sent_end"]]

    return doc_emb_map, sent_emb_map


def build_pair_features(
    df: pd.DataFrame,
    dataset_name: str,
    include_late_interaction: bool = True,
    cache_dir: str = "cache/embeddings",
    model=None,
):
    """
    Build pair features by fast index lookup from cached document & sentence embeddings.
    Zero re-embedding computation!
    """
    doc_emb_map, sent_emb_map = build_and_cache_document_embeddings(
        df, dataset_name=dataset_name, cache_dir=cache_dir, include_sentences=include_late_interaction, model=model
    )

    # 1. Document-level features via lookup
    X_res = np.array([doc_emb_map[rid] for rid in df["resume_id"]], dtype=np.float32)
    X_jd = np.array([doc_emb_map[jid] for jid in df["jd_id"]], dtype=np.float32)
    
    diff = np.abs(X_res - X_jd)
    mult = X_res * X_jd
    cos = np.sum(X_res * X_jd, axis=1, keepdims=True)
    
    feature_blocks = [X_res, X_jd, diff, mult, cos]

    # 2. Late Interaction Features via sentence matrices
    if include_late_interaction:
        late_features = []
        for rid, jid in zip(df["resume_id"], df["jd_id"]):
            s_res = sent_emb_map.get(rid)
            s_jd = sent_emb_map.get(jid)
            
            if s_res is not None and s_jd is not None and len(s_res) > 0 and len(s_jd) > 0:
                # Cosine similarity matrix between all resume sentences and JD sentences
                # Since sentence embeddings are normalized, dot product = cosine similarity
                M = np.dot(s_res, s_jd.T) # (num_res_sents, num_jd_sents)
                
                # JD coverage: max match in resume for each JD sentence
                jd_max = np.max(M, axis=0) # (num_jd_sents,)
                res_max = np.max(M, axis=1) # (num_res_sents,)
                
                jd_mean = float(np.mean(jd_max))
                res_mean = float(np.mean(res_max))
                max_overall = float(np.max(M))
                
                cov_70 = float(np.mean(jd_max >= 0.70))
                cov_50 = float(np.mean(jd_max >= 0.50))
                
                top3_jd = float(np.mean(np.sort(jd_max)[-3:])) if len(jd_max) >= 3 else jd_mean
                
                late_features.append([jd_mean, res_mean, max_overall, cov_70, cov_50, top3_jd])
            else:
                late_features.append([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                
        X_late = np.array(late_features, dtype=np.float32)
        feature_blocks.append(X_late)

    X_full = np.hstack(feature_blocks)
    return X_full


def get_cached_embeddings(df: pd.DataFrame, dataset_name: str, cache_dir: str = "cache/embeddings", model=None):
    """Legacy backward-compatible wrapper returning row-aligned (res_emb, jd_emb)."""
    doc_emb_map, _ = build_and_cache_document_embeddings(
        df, dataset_name=dataset_name, cache_dir=cache_dir, model=model
    )
    res_emb = np.array([doc_emb_map[rid] for rid in df["resume_id"]], dtype=np.float32)
    jd_emb = np.array([doc_emb_map[jid] for jid in df["jd_id"]], dtype=np.float32)
    return res_emb, jd_emb
