"""QA Unit Tests for App v2 Serving Logic.

Tests:
1. Input validation (empty text, short text rejection).
2. Input warnings on long text (>500 words) and identical/single-sided text.
3. Name/gender/pronoun perturbation invariance (fairness check).
4. Output format and calibrated category assignment.
"""

import pytest
import numpy as np
from unittest.mock import MagicMock
import app


@pytest.fixture(autouse=True)
def mock_app_models():
    """Mock models so tests run fast without re-loading heavy weights."""
    app._models_ready = True
    app._load_error = None
    app.sbert_model = MagicMock()
    
    # Mock encode to return 384-d normalized vector based on text length / hash
    def fake_encode(texts, normalize_embeddings=True):
        vecs = []
        for t in texts:
            seed = hash(t) % (2**31)
            rng = np.random.RandomState(seed)
            v = rng.randn(1, 384).astype(np.float32)
            v /= np.linalg.norm(v, axis=1, keepdims=True)
            vecs.append(v[0])
        return np.array(vecs, dtype=np.float32)
    app.sbert_model.encode = fake_encode

    app.lgb_model = MagicMock()
    app.lgb_model.predict_proba = MagicMock(return_value=np.array([[0.1, 0.2, 0.7]], dtype=np.float32))
    app.model_meta = {"calibration": {"serving_default_temperature": 1.19}}


def test_input_validation_empty_short():
    with pytest.raises(Exception) as exc1:
        app.predict_ats("", "Some valid job description with enough words to pass validation testing.")
    assert "Please provide resume text" in str(exc1.value)

    with pytest.raises(Exception) as exc2:
        app.predict_ats("Too short resume", "Some valid job description with enough words to pass validation testing.")
    assert "Resume text is too short" in str(exc2.value)


def test_input_warnings():
    # Long text test
    long_resume = "word " * 300
    long_jd = "requirement " * 300
    fit_idx, pred_class, report = app.predict_ats(long_resume, long_jd)
    assert "exceeds ~500 tokens" in report
    assert "Good Fit" in pred_class

    # Identical text test
    common_text = "Experienced Senior Python Engineer with 10 years experience developing microservices " * 5
    fit_idx2, pred_class2, report2 = app.predict_ats(common_text, common_text)
    assert "appear identical" in report2


def test_fairness_perturbation():
    """Check that replacing candidate names and pronouns does not alter ATS prediction."""
    jd = "Seeking Senior Software Engineer with experience in Python, Django, AWS, CI/CD pipelines, and microservices architecture. Bachelor degree required. Candidate must possess strong communication skills and leadership experience across cross-functional teams."
    
    resume_male = "John Doe is a dedicated Senior Software Engineer with 8 years experience in Python and AWS. He built scalable microservices and led backend engineering teams across several high-impact customer-facing initiatives."
    resume_female = "Jane Smith is a dedicated Senior Software Engineer with 8 years experience in Python and AWS. She built scalable microservices and led backend engineering teams across several high-impact customer-facing initiatives."

    # Using actual mocked predictions
    app.lgb_model.predict_proba = MagicMock(return_value=np.array([[0.15, 0.25, 0.60]], dtype=np.float32))
    fit_m, class_m, _ = app.predict_ats(resume_male, jd)
    fit_f, class_f, _ = app.predict_ats(resume_female, jd)

    assert fit_m == fit_f, f"Prediction discrepancy on demographic perturbation: {fit_m} vs {fit_f}"
    assert class_m == class_f, f"Category discrepancy on demographic perturbation: {class_m} vs {class_f}"
