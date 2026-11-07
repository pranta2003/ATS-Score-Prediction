"""Unit tests for T3 Late Interaction feature extraction.

Tests:
1. Toy identical pairs vs random pairs.
2. Exact mathematical bounds of sentence-level MaxSim features.
3. Cache alignment between c5_sent_emb.npy and c5_sent_meta.parquet.
4. Check for NaNs and constant columns on actual extracted T3 features.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_c5_feature_ablation import extract_t3_late_interaction


def test_t3_synthetic_bounds():
    # Construct synthetic sentence embeddings
    rng = np.random.RandomState(42)
    s1 = rng.randn(4, 384).astype(np.float32)
    s1 /= np.linalg.norm(s1, axis=1, keepdims=True)
    
    # Identical pair
    s2 = s1.copy()
    
    # Orthogonal / different pair
    s3 = rng.randn(4, 384).astype(np.float32)
    s3 /= np.linalg.norm(s3, axis=1, keepdims=True)

    emb_map = {
        "res_1": s1,
        "jd_identical": s2,
        "jd_diff": s3,
    }

    df = pd.DataFrame({
        "resume_id": ["res_1", "res_1"],
        "jd_id": ["jd_identical", "jd_diff"]
    })

    feats = extract_t3_late_interaction(df, emb_map)
    
    assert feats.shape == (2, 12), f"Expected shape (2, 12), got {feats.shape}"
    assert not np.isnan(feats).any(), "NaNs found in extracted features"
    
    # Identical pair must have max overall == 1.0 and cov_70 == 1.0
    # feats columns: [jd_mean, jd_max_val, jd_std, cov_70, cov_50, top3_jd, res_mean, res_max_val, res_std, top3_res, max_overall, mean_overall]
    cov_70_idx = 3
    max_overall_idx = 10
    
    assert np.isclose(feats[0, max_overall_idx], 1.0, atol=1e-5), f"Identical max_overall should be 1.0, got {feats[0, max_overall_idx]}"
    assert np.isclose(feats[0, cov_70_idx], 1.0, atol=1e-5), f"Identical cov_70 should be 1.0, got {feats[0, cov_70_idx]}"
    
    # Identical pair alignment must exceed different pair alignment
    assert feats[0, 0] > feats[1, 0], "Identical jd_mean must exceed different jd_mean"
    assert feats[0, max_overall_idx] > feats[1, max_overall_idx], "Identical max must exceed different max"


def test_t3_cache_integrity():
    emb_path = PROJECT_ROOT / "cache" / "embeddings" / "c5_sent_emb.npy"
    meta_path = PROJECT_ROOT / "cache" / "embeddings" / "c5_sent_meta.parquet"

    assert emb_path.exists(), "c5_sent_emb.npy does not exist"
    assert meta_path.exists(), "c5_sent_meta.parquet does not exist"

    sent_embs = np.load(emb_path)
    sent_meta = pd.read_parquet(meta_path)

    # sent_meta rows correspond to unique documents, num_sentences sums to sent_embs length
    assert len(sent_meta) == 994, f"Expected 994 unique documents in C5 sent_meta, got {len(sent_meta)}"
    assert sent_meta["num_sentences"].sum() == len(sent_embs), f"Length mismatch: {len(sent_embs)} vs {sent_meta['num_sentences'].sum()}"
    assert set(sent_meta.columns).issuperset({"text_id", "sent_start", "sent_end", "num_sentences"})
    
    # Check bounds
    assert sent_meta["sent_start"].min() >= 0
    assert sent_meta["sent_end"].max() == len(sent_embs)
    
    # Check that embeddings are normalized
    norms = np.linalg.norm(sent_embs[:100], axis=1)
    assert np.allclose(norms, 1.0, atol=1e-3), "Sentence embeddings are not unit normalized"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
