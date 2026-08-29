# ATS Score Prediction using Machine Learning

## 📌 Project Overview

Applicant Tracking Systems (ATS) are widely used to evaluate resumes against job descriptions during recruitment. This project develops a **machine learning-based regression system** that predicts a numerical ATS score by analyzing the semantic relationship between a resume and its corresponding job description.

Instead of treating ATS prediction as a classification problem, this project formulates it as a **regression problem**, where the output is a continuous ATS score.

The system uses **Sentence-BERT (SBERT)** to convert both resumes and job descriptions into semantic numerical representations and then applies several regression models to predict the ATS score.

---

## 🎯 Objectives

The main objectives of this project are:

- Predict a numerical ATS score from a resume and job description.
- Capture semantic relationships between resume content and job requirements.
- Compare multiple regression algorithms.
- Evaluate model performance using appropriate regression metrics.
- Investigate ensemble learning to improve prediction performance.
- Develop a simple interface for generating ATS score predictions.

---

## 📊 Dataset

The project uses the:

**resume-ats-score-v1-en**

Dataset from Hugging Face:

`Oxnbk/resume-ats-score-v1-en`

The dataset contains two independent text inputs:

- `resume`
- `job_description`

Target variable:

- `ats_score`

Additional column:

- `original_label`

The dataset is already separated into `resume` and `job_description`; therefore, no text-column splitting is required.

---

## 🔄 Project Pipeline

```text
Resume
   │
   ▼
Text Cleaning
   │
   ▼
Sentence-BERT
   │
   ▼
384-Dimensional Embedding
   │
   ├─────────────────────────┐
   │                         │
   │                         │
Job Description             │
   │                         │
   ▼                         │
Text Cleaning                │
   │                         │
   ▼                         │
Sentence-BERT                │
   │                         │
   ▼                         │
384-Dimensional Embedding    │
   │                         │
   └────────────┬────────────┘
                ▼
       Concatenation
                ▼
        768 Features
                ▼
       80% Train / 20% Test
                ▼
    ┌─────────────────────────┐
    │  Regression Models      │
    │                         │
    │  Linear Regression      │
    │  KNN Regressor          │
    │  Random Forest          │
    │  XGBoost                │
    └────────────┬────────────┘
                 ▼
        Model Evaluation
                 ▼
       Bagging Ensemble
       ┌─────────────────┐
       │ Bagged Linear   │
       │ Regression      │
       │                 │
       │ Bagged KNN      │
       └────────┬────────┘
                ▼
       Prediction Averaging
                ▼
        Final ATS Score
