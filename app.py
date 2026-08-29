import os
import numpy as np
import joblib
import gradio as gr

from sentence_transformers import SentenceTransformer


# ==========================================
# Load models
# ==========================================

sbert_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

xgb_model = joblib.load("ats_xgboost_model.pkl")


# ==========================================
# Prediction function
# ==========================================

def predict_ats(resume, job_description):

    # Create SBERT embeddings
    resume_embedding = sbert_model.encode([resume])
    job_embedding = sbert_model.encode([job_description])

    # Combine resume + job description features
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
        "Enter a resume and job description "
        "to predict the numerical ATS score."
    )
)


interface.launch(
    server_name="0.0.0.0",
    server_port=int(os.environ.get("PORT", 7860))
)