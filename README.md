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
```
🧹 Text Preprocessing

The resume and job description are cleaned independently before generating embeddings.

The preprocessing includes:

Converting text to lowercase
Removing URLs
Removing HTML content
Removing email addresses
Removing phone numbers
Removing unnecessary symbols
Preserving important programming and technical keywords such as:
C++
C#
.NET
Node.js

The purpose of preprocessing is to reduce irrelevant textual noise while preserving information that may be important for matching a candidate with a job description.


🧠 Sentence-BERT Feature Extraction

The project uses:

sentence-transformers/all-MiniLM-L6-v2

Each text input is independently converted into a 384-dimensional semantic embedding.

Therefore:

Resume                → 384 features
Job Description       → 384 features
                         ───────────
Total                 → 768 features

The two embeddings are concatenated to form the final feature vector used by the regression models.

This allows the models to work with semantic representations rather than relying only on individual words or manually engineered features.

🤖 Regression Models

Four regression models were evaluated as individual benchmark models:

1. Linear Regression

Provides a simple linear relationship between the SBERT feature representation and the ATS score.

2. KNN Regressor

Predicts the ATS score based on the nearest observations in the embedding feature space.

3. Random Forest Regressor

Uses multiple decision trees and aggregates their predictions. Random Forest already incorporates a bagging mechanism internally.

4. XGBoost Regressor

A boosting-based regression algorithm that builds an ensemble of decision trees sequentially.


📏 Evaluation Metrics

Since ATS score prediction is a regression problem, the following metrics are used:

MAE (Mean Absolute Error)
RMSE (Root Mean Squared Error)
R² (Coefficient of Determination)

Lower MAE and RMSE indicate better prediction performance, while a higher R² indicates better explanatory performance.

