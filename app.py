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
# Prediction function
# ==========================================

def predict_ats(resume, job_description):
    if not _models_ready:
        if _load_error is not None:
            raise gr.Error(f"Model initialization failed: {_load_error}")
        raise gr.Error("Model is still loading in the background. Please wait ~15-30 seconds and retry.")

    if not resume or not resume.strip():
        raise gr.Error("Please provide resume text.")
    if not job_description or not job_description.strip():
        raise gr.Error("Please provide job description text.")

    # Create SBERT embeddings
    resume_embedding = sbert_model.encode([resume])
    job_embedding = sbert_model.encode([job_description])

    # Combine resume + job description features (768-d)
    combined_features = np.concatenate(
        [resume_embedding, job_embedding],
        axis=1
    )

    # XGBoost prediction
    ats_score = xgb_model.predict(combined_features)[0]

    return float(ats_score)


# ==========================================
# Gradio interface
# ==========================================

interface = gr.Interface(
    fn=predict_ats,
    inputs=[
        gr.Textbox(
            lines=12,
            label="Resume",
            placeholder="Paste the resume here..."
        ),
        gr.Textbox(
            lines=12,
            label="Job Description",
            placeholder="Paste the job description here..."
        )
    ],
    outputs=gr.Number(
        label="Predicted ATS Score"
    ),
    title="ATS Score Prediction",
    description=(
        "Enter a resume and job description to predict the numerical ATS score. "
        "The model uses SentenceTransformer embeddings with an XGBoost regressor."
    )
)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    interface.launch(
        server_name="0.0.0.0",
        server_port=port
    )
