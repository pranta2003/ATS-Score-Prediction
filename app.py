import os
import threading
import time
import json
import numpy as np
from scipy.special import softmax
import gradio as gr
from src.ats.rubric import (
    evaluate_formatting,
    extract_keywords,
    extract_years_experience,
    extract_degree_level,
)

# ==========================================
# Background model loader
# ==========================================
# Opening the port FIRST prevents Render's "Port scan timeout reached, no open
# ports detected" error, while loading heavy neural models asynchronously.

_models_ready = False
_load_error = None
sbert_model = None
lgb_model = None
model_meta = {}


def _load_models():
    global _models_ready, _load_error, sbert_model, lgb_model, model_meta
    try:
        print("[Startup] Background thread: initializing dependencies and models...", flush=True)
        import joblib
        from sentence_transformers import SentenceTransformer

        # 1. Load SBERT embedding backbone
        sbert_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

        # 2. Load LightGBM 3-class production classifier
        base_dir = os.path.dirname(__file__)
        model_path = os.path.join(base_dir, "models", "c5_lightgbm_final.joblib")
        meta_path = os.path.join(base_dir, "models", "c5_lightgbm_final_meta.json")

        if os.path.exists(model_path):
            lgb_model = joblib.load(model_path)
        else:
            raise FileNotFoundError(f"Model file not found at {model_path}")

        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                model_meta = json.load(f)

        _models_ready = True
        print("[Startup] Background thread: models successfully loaded and ready.", flush=True)
    except Exception as e:
        _load_error = str(e)
        print(f"[Startup] Background thread ERROR loading models: {e}", flush=True)


# Kick off model loading in background daemon thread immediately
threading.Thread(target=_load_models, daemon=True).start()


# ==========================================
# Serving feature extraction (Feature Parity with T2 Dense)
# ==========================================

def extract_serving_features(resume_text: str, jd_text: str, model=None) -> np.ndarray:
    """
    Extract Tier 2 Dense semantic features (1,537-d) exactly matching training.
    Features: [u (384), v (384), |u-v| (384), u*v (384), cos(u,v) (1)]
    """
    m = model if model is not None else sbert_model
    if m is None:
        raise RuntimeError("Embedding model is not loaded yet.")

    # Unit-normalized embeddings (384-d each)
    r_emb = m.encode([resume_text], normalize_embeddings=True).astype(np.float32)
    j_emb = m.encode([jd_text], normalize_embeddings=True).astype(np.float32)

    diff = np.abs(r_emb - j_emb)
    mult = r_emb * j_emb
    cos = np.sum(r_emb * j_emb, axis=1, keepdims=True)

    return np.hstack([r_emb, j_emb, diff, mult, cos])


# ==========================================
# Prediction function with temperature calibration & validation
# ==========================================

def predict_ats(resume: str, job_description: str):
    if not _models_ready:
        if _load_error is not None:
            raise gr.Error(f"Model initialization failed: {_load_error}")
        raise gr.Error("Model is still loading in the background. Please wait ~15-30 seconds and retry.")

    r_text = (resume or "").strip()
    j_text = (job_description or "").strip()

    # Input validation
    if not r_text:
        raise gr.Error("Validation Error: Please provide resume text.")
    if not j_text:
        raise gr.Error("Validation Error: Please provide job description text.")

    r_words = r_text.split()
    j_words = j_text.split()

    if len(r_words) < 20:
        raise gr.Error("Validation Error: Resume text is too short (< 20 words). Please provide adequate career/skills text.")
    if len(j_words) < 20:
        raise gr.Error("Validation Error: Job description is too short (< 20 words). Please provide adequate job requirements.")

    # Warnings for length truncation & duplicate inputs
    warnings = []
    if len(r_words) + len(j_words) > 500:
        warnings.append("Combined length exceeds ~500 tokens; text is embedded with sentence chunk-pooling per MiniLM-L6 architecture.")

    if r_text.lower() == j_text.lower():
        warnings.append("Resume and Job Description text appear identical. Please verify input fields.")

    # Extract 1,537-d T2 Dense features
    features = extract_serving_features(r_text, j_text, sbert_model)

    # Raw model probabilities [P(No Fit), P(Potential Fit), P(Good Fit)]
    raw_probs = lgb_model.predict_proba(features)[0]

    # Temperature scaling calibration (T=1.91 pre-calibrated on C5 S2 Cold-JD OOF)
    temp = float(model_meta.get("calibration", {}).get("serving_default_temperature", 1.91))
    eps = 1e-7
    clipped_probs = np.clip(raw_probs, eps, 1.0 - eps)
    logits = np.log(clipped_probs)
    calibrated_probs = softmax(logits / temp)

    p_no_fit = float(calibrated_probs[0])
    p_potential = float(calibrated_probs[1])
    p_good = float(calibrated_probs[2])

    # Class selection
    classes = ["No Fit", "Potential Fit", "Good Fit"]
    pred_idx = int(np.argmax(calibrated_probs))
    pred_class = classes[pred_idx]
    confidence = float(calibrated_probs[pred_idx])

    # Composite Fit Index (0–100 scale)
    # Weights: Good Fit = 1.0, Potential Fit = 0.5, No Fit = 0.0
    composite_fit_index = round(100.0 * (p_good * 1.0 + p_potential * 0.5), 1)

    # Candidate Coverage Checklist (Strictly non-scoring audit)
    fmt_audit = evaluate_formatting(r_text)
    res_kw = extract_keywords(r_text)
    jd_kw = extract_keywords(j_text)
    matched_kw = res_kw.intersection(jd_kw)
    missing_kw = sorted(list(jd_kw - res_kw))
    res_exp = extract_years_experience(r_text)
    jd_req_exp = extract_years_experience(j_text)
    res_edu = extract_degree_level(r_text)
    jd_edu = extract_degree_level(j_text)

    deg_names = {0: "Not specified", 1: "Associate/Diploma", 2: "Bachelor's", 3: "Master's/MBA", 4: "PhD/Doctorate"}
    res_edu_str = deg_names.get(res_edu, "Not specified")
    jd_edu_str = deg_names.get(jd_edu, "Not specified")

    kw_pct = (len(matched_kw) / max(1, len(jd_kw))) * 100.0
    exp_status = "Meets requirement" if res_exp >= jd_req_exp else f"Below requirement ({res_exp:.0f} vs {jd_req_exp:.0f} yrs)"
    missing_str = ", ".join(f"`{k}`" for k in missing_kw[:6]) if missing_kw else "None (Comprehensive coverage)"

    warn_block = f"\n\n> ⚠️ **Alerts**: {' | '.join(warnings)}" if warnings else ""

    markdown_report = (
        f"### Assessment: **{pred_class}**\n\n"
        f"- **Composite Fit Index:** `{composite_fit_index} / 100`\n"
        f"- **Model Confidence:** `{confidence * 100:.1f}%`\n\n"
        f"#### Calibrated Class Probabilities\n"
        f"| Category | Calibrated Probability | Raw Model Probability |\n"
        f"| :--- | :---: | :---: |\n"
        f"| **Good Fit** | **{p_good * 100:.1f}%** | {raw_probs[2] * 100:.1f}% |\n"
        f"| **Potential Fit** | **{p_potential * 100:.1f}%** | {raw_probs[1] * 100:.1f}% |\n"
        f"| **No Fit** | **{p_no_fit * 100:.1f}%** | {raw_probs[0] * 100:.1f}% |\n"
        f"{warn_block}\n\n"
        f"#### Candidate Coverage Checklist (Non-Scoring Hygiene & Match Audit)\n"
        f"- **Formatting Hygiene:** Email: {'✅ Detected' if fmt_audit['has_email'] else '❌ Missing'} | Phone: {'✅ Detected' if fmt_audit['has_phone'] else '❌ Missing'} | Word Count: `{fmt_audit['word_count']} words`\n"
        f"- **Years of Experience:** Resume: `{res_exp:.0f} yrs` | JD Required: `{jd_req_exp:.0f} yrs` ({exp_status})\n"
        f"- **Education Level:** Resume: `{res_edu_str}` | JD Requested: `{jd_edu_str}`\n"
        f"- **Keyword Coverage:** `{len(matched_kw)} / {len(jd_kw)} terms matched` ({kw_pct:.1f}%)\n"
        f"- **Top Missing Key Terms:** {missing_str}\n\n"
        f"---\n"
        f"*Methodological Note: Evaluated using LightGBM with 1,537-d MiniLM-L6 interaction features and temperature scaling ($T = {temp:.2f}$). "
        f"On unseen job descriptions (cold evaluation), the expected empirical accuracy ceiling is ~52–56%; scores should inform, not replace, recruiter judgment.*"
    )

    return composite_fit_index, pred_class, markdown_report


# ==========================================
# Gradio interface
# ==========================================

interface = gr.Interface(
    fn=predict_ats,
    inputs=[
        gr.Textbox(
            lines=12,
            label="Resume Text",
            placeholder="Paste candidate resume text here (minimum 20 words)..."
        ),
        gr.Textbox(
            lines=12,
            label="Job Description Text",
            placeholder="Paste job description requirements here (minimum 20 words)..."
        )
    ],
    outputs=[
        gr.Number(label="Composite Fit Index (0-100)"),
        gr.Textbox(label="Predicted Candidate Fit Category"),
        gr.Markdown(label="Evaluation & Calibrated Probabilities")
    ],
    title="ATS Candidate Match & Fit Evaluator",
    description=(
        "Production ATS evaluator predicting candidate fit category and calibrated match probability. "
        "Powered by LightGBM on all-MiniLM-L6-v2 semantic interaction vectors (T2 Dense, 1,537-d) with temperature scaling calibration."
    )
)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    interface.launch(
        server_name="0.0.0.0",
        server_port=port
    )
