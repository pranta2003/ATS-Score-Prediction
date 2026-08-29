# ATS Score Prediction using Machine Learning

A machine learning regression system that predicts a **numerical ATS score** by analyzing the semantic relationship between a resume and a job description.

## Overview

The project uses the `resume-ats-score-v1-en` dataset, where `resume` and `job_description` are processed as separate text inputs and `ats_score` is the continuous target.
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
### Pipeline

Resume + Job Description  
↓  
Text Cleaning  
↓  
Sentence-BERT (`all-MiniLM-L6-v2`)  
↓  
384 + 384 = **768-dimensional features**  
↓  
Regression Models  
↓  
Bagging Ensemble  
↓  
Predicted ATS Score

## Models Evaluated

- Linear Regression
- KNN Regressor
- Random Forest Regressor
- XGBoost Regressor
- Bagging Ensemble (Bagged Linear Regression + Bagged KNN)

## Test Results

| Model | R² |
|---|---:|
| Linear Regression | 0.1468 |
| KNN | 0.2445 |
| Random Forest | 0.3614 |
| **XGBoost** | **0.4287** |
| Bagging Ensemble | 0.6865 |

XGBoost achieved the strongest performance among the individual benchmark models, with **R² = 0.4287**.

The Bagging Ensemble combines Bagged Linear Regression and Bagged KNN through prediction averaging. It achieved **R² = 0.2865**, improving over the original KNN result of **0.2445**.

## Evaluation

The project uses regression metrics:

- MAE
- RMSE
- R²

### Best Individual Model

**XGBoost**

- MAE: **15.4347**
- RMSE: **18.8966**
- R²: **0.4287**

## Technologies

Python • Pandas • NumPy • Scikit-learn • Sentence-Transformers • XGBoost • Joblib • Gradio

## Interface

A Gradio interface allows users to enter a **resume** and **job description** and receive a predicted numerical ATS score.

## Repository

The main implementation is available in:

`ats-score-prediction-project.ipynb`

## Interface

A Gradio interface allows users to enter a **resume** and **job description** and receive a predicted numerical ATS score.

## Repository

The main implementation is available in:

`ats-score-prediction-project.ipynb`

## 📄 License

This project is licensed under the **MIT License**.

See the [LICENSE](LICENSE) file for details.
