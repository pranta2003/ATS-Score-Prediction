import re
import joblib
import numpy as np
import gradio as gr

from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer


# -----------------------------------
# Load trained Bagging models
# -----------------------------------

models = joblib.load("ats_bagging_models.pkl")

bag_lr = models["bag_lr"]
bag_knn = models["bag_knn"]


# -----------------------------------
# Load Sentence-BERT
# -----------------------------------

sbert_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)


# -----------------------------------
# Text Cleaning
# -----------------------------------

def clean_text(text):

    text = str(text).lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", " ", text)

    # Remove HTML
    text = BeautifulSoup(text, "html.parser").get_text(" ")

    # Remove emails
    text = re.sub(r"\S+@\S+", " ", text)

    # Remove phone numbers
    text = re.sub(
        r"\+?\d[\d\s().-]{7,}\d",
        " ",
        text
    )

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


# -----------------------------------
# ATS Prediction
# -----------------------------------

def predict_ats(resume, job_description):

    if not resume or not job_description:
        return 0.0

    # Clean inputs
    resume_clean = clean_text(resume)
    job_clean = clean_text(job_description)

    # Generate SBERT embeddings
    resume_embedding = sbert_model.encode(
        resume_clean
    )

    job_embedding = sbert_model.encode(
        job_clean
    )

    # 384 + 384 = 768 features
    combined_features = np.concatenate([
        resume_embedding,
        job_embedding
    ])

    combined_features = combined_features.reshape(
        1, -1
    )

    # Bagged Linear Regression
    lr_prediction = bag_lr.predict(
        combined_features
    )[0]

    # Bagged KNN
    knn_prediction = bag_knn.predict(
        combined_features
    )[0]

    # Average predictions
    ats_score = (
        lr_prediction + knn_prediction
    ) / 2

    return round(float(ats_score), 2)


# -----------------------------------
# Gradio Interface
# -----------------------------------

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


if __name__ == "__main__":
    interface.launch()
