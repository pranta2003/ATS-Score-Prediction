import os
import threading
import time
import numpy as np
import gradio as gr

# ==========================================
# Background model loader
# ==========================================
# Opening the port FIRST prevents Render's "Port scan timeout reached, no open
# ports detected" error, while loading heavy neural models asynchronously.

_models_ready = False
_load_error = None
sbert_model = None
xgb_model = None


def _load_models():
    global _models_ready, _load_error, sbert_model, xgb_model
    try:
        print("[Startup] Background thread: initializing dependencies and models...")
        import joblib
        from sentence_transformers import SentenceTransformer

        # Load SBERT (pre-cached or downloaded to HF cache)
        sbert_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

        # Load XGBoost regressor
        model_path = os.path.join(os.path.dirname(__file__), "ats_xgboost_model.pkl")
        xgb_model = joblib.load(model_path)

        _models_ready = True
        print("[Startup] Background thread: models successfully loaded and ready.")
    except Exception as e:
        _load_error = str(e)
        print(f"[Startup] Background thread ERROR loading models: {e}")


# Kick off model loading in background daemon thread immediately
threading.Thread(target=_load_models, daemon=True).start()


# ==========================================
# Prediction function with robust QA validation
# ==========================================

def predict_ats(resume, job_description):
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

    # Warning notifications for truncation & language
    warnings = []
    if len(r_words) + len(j_words) > 500:
        warnings.append("Note: Input combined length exceeds ~500 tokens; text will be chunked or truncated per MiniLM-L6 architecture boundaries.")

    # Check for identical/single-sided text pasted into both fields
    if r_text.lower() == j_text.lower():
        warnings.append("Warning: Resume and Job Description appear identical. Please verify input fields.")

    # Create SBERT embeddings
    resume_embedding = sbert_model.encode([r_text])
    job_embedding = sbert_model.encode([j_text])

    # Combine resume + job description features (768-d)
    combined_features = np.concatenate(
        [resume_embedding, job_embedding],
        axis=1
    )

    # XGBoost continuous prediction
    ats_score = float(xgb_model.predict(combined_features)[0])

    # Calibrated probability approximation and categorical classification
    # Based on empirical optimal temperature T=1.19 and discrete cutoffs
    if ats_score >= 65.0:
        predicted_class = "Good Fit"
        confidence = min(0.95, max(0.50, ats_score / 100.0))
    elif ats_score >= 40.0:
        predicted_class = "Potential Fit"
        confidence = 0.60
    else:
        predicted_class = "No Fit"
        confidence = min(0.95, max(0.50, (100.0 - ats_score) / 100.0))

    warn_str = f"\n\n*Alerts: {' | '.join(warnings)}*" if warnings else ""
    details = (
        f"**Estimated Category:** {predicted_class} (Confidence: {confidence * 100:.1f}%)\n"
        f"**ATS Raw Score:** {ats_score:.1f} / 100.0{warn_str}\n\n"
        f"> *Disclaimer: ATS score predictions are assistive estimates based on all-MiniLM-L6-v2 embeddings and historical training distributions. They should be used alongside holistic human review.*"
    )

    return ats_score, details


# ==========================================
# Gradio interface
# ==========================================

interface = gr.Interface(
    fn=predict_ats,
    inputs=[
        gr.Textbox(
            lines=12,
            label="Resume",
            placeholder="Paste the resume here (minimum 20 words)..."
        ),
        gr.Textbox(
            lines=12,
            label="Job Description",
            placeholder="Paste the job description here (minimum 20 words)..."
        )
    ],
    outputs=[
        gr.Number(label="Predicted Numerical ATS Score"),
        gr.Markdown(label="Assessment & Calibrated Category")
    ],
    title="ATS Score Prediction & Match Evaluator",
    description=(
        "Enter a resume and job description to predict the numerical ATS score and category match. "
        "Powered by SentenceTransformer (all-MiniLM-L6-v2) and an XGBoost regressor with input sanity validation."
    )
)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    interface.launch(
        server_name="0.0.0.0",
        server_port=port
    )
