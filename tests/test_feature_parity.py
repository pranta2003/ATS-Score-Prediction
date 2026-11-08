"""Feature parity test: verifies serving-time features match training-time features."""

import sys
from pathlib import Path
import numpy as np
import pytest
import joblib

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.ats.adapters import get_adapter
from src.ats.embeddings import build_and_cache_document_embeddings
from scripts.run_c5_feature_ablation import extract_t2_dense
from app import extract_serving_features


@pytest.mark.heavy
def test_t2_feature_parity_50_pairs():
    adapter = get_adapter("c5")
    df = adapter.load()
    
    # Select 50 deterministic pairs
    sample_df = df.sample(n=50, random_state=42).reset_index(drop=True)
    
    # 1. Training pipeline features
    doc_emb_map, _ = build_and_cache_document_embeddings(
        sample_df,
        dataset_name="c5",
        cache_dir="cache/embeddings",
        include_sentences=False,
    )
    X_offline = extract_t2_dense(sample_df, doc_emb_map)
    
    # 2. Serving pipeline features (one by one or batched)
    from sentence_transformers import SentenceTransformer
    sbert = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    
    serving_feats = []
    for r_text, j_text in zip(sample_df["resume_text"], sample_df["jd_text"]):
        feats = extract_serving_features(r_text, j_text, model=sbert)
        serving_feats.append(feats[0])
    X_serving = np.array(serving_feats, dtype=np.float32)
    
    # Assert shapes
    assert X_offline.shape == (50, 1537), f"Expected shape (50, 1537), got {X_offline.shape}"
    assert X_serving.shape == (50, 1537), f"Expected shape (50, 1537), got {X_serving.shape}"
    
    # Assert numerical parity
    max_diff = np.max(np.abs(X_offline - X_serving))
    assert np.allclose(X_offline, X_serving, atol=1e-4), f"Feature mismatch! Max diff: {max_diff}"
    
    # 3. Model predictions parity
    model_path = PROJECT_ROOT / "models" / "c5_lightgbm_final.joblib"
    assert model_path.exists(), "c5_lightgbm_final.joblib does not exist"
    model = joblib.load(model_path)
    
    probs_offline = model.predict_proba(X_offline)
    probs_serving = model.predict_proba(X_serving)
    
    preds_offline = np.argmax(probs_offline, axis=1)
    preds_serving = np.argmax(probs_serving, axis=1)
    
    assert np.array_equal(preds_offline, preds_serving), "Model predictions differ between offline and serving!"
    assert np.allclose(probs_offline, probs_serving, atol=1e-4), "Predicted probabilities differ between offline and serving!"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
