```
# ATS Score Prediction using Machine Learning

A machine learning regression system that predicts a **numerical ATS score** by analyzing the semantic relationship between a resume and a job description.

## Overview

The project uses the `resume-ats-score-v1-en` dataset, where `resume` and `job_description` are processed as separate text inputs and `ats_score` is the continuous target.

The system uses **Sentence-BERT (SBERT)** to convert the resume and job description into numerical embeddings. These embeddings are concatenated and provided to regression models for ATS score prediction.

## 🔄 Project Pipeline


Resume
   │
   ▼
Text Processing
   │
   ▼
Sentence-BERT
   │
   ▼
384-Dimensional Embedding
   │
   │
Job Description
   │
   ▼
Text Processing
   │
   ▼
Sentence-BERT
   │
   ▼
384-Dimensional Embedding
   │
   └──────────────┐
                  ▼
           Concatenation
                  ▼
            768 Features
                  ▼
           Train / Test Split
                  ▼
        ┌─────────────────────┐
        │ Regression Models   │
        │                     │
        │ Linear Regression   │
        │ KNN Regressor       │
        │ Random Forest       │
        │ XGBoost             │
        └──────────┬──────────┘
                   ▼
            Model Evaluation
                   ▼
          Final XGBoost Model
                   ▼
           Predicted ATS Score

Pipeline Summary

Resume + Job Description
↓
Text Processing
↓
Sentence-BERT (all-MiniLM-L6-v2)
↓
384 + 384 = 768-dimensional features
↓
Train / Test Split
↓
Regression Models
↓
Model Evaluation
↓
Final XGBoost Model
↓
Predicted ATS Score

Models Evaluated
Linear Regression
KNN Regressor
Random Forest Regressor
XGBoost Regressor
Cross-Validation Results

The models were evaluated using 5-fold cross-validation.

Model	Average Validation R²
Linear Regression	0.1233
KNN	0.2032
Random Forest	0.3390
XGBoost	0.3777
Bagging LR + KNN	0.2525

Among the evaluated models, XGBoost achieved the highest average 5-fold validation R² of 0.3777.

Final XGBoost Model

After model evaluation, XGBoost was selected as the final individual prediction model.

Final Test Results
Metric	Score
Training R²	0.8670
Test R²	0.3900
MAE	16.0956
RMSE	19.5251

The final XGBoost model achieved a test R² of 0.3900, indicating that the model explains approximately 39% of the variation in the ATS score on the held-out test data.

Hyperparameter Tuning

XGBoost hyperparameters were also tuned using 5-fold cross-validation.

Best Parameters
subsample = 0.8
n_estimators = 100
max_depth = 4
learning_rate = 0.1
colsample_bytree = 1.0

The best cross-validation R² obtained during the tuning process was:

0.2808

The final model selection was based on the overall model evaluation and validation results, with XGBoost providing the strongest individual-model performance.

Evaluation Metrics

The project uses standard regression metrics:

R² (Coefficient of Determination) — measures how well the model explains variation in ATS scores.
MAE (Mean Absolute Error) — measures the average absolute prediction error.
RMSE (Root Mean Squared Error) — measures prediction error while giving larger errors more weight.
Technologies
Python
Pandas
NumPy
Scikit-learn
Sentence-Transformers
XGBoost
Joblib
Gradio
💻 Gradio Interface

The project includes a Gradio web interface where users can enter:

A resume
A job description

The system processes both text inputs using Sentence-BERT and uses the trained XGBoost model to generate a predicted numerical ATS score.

🚀 Deployment

The Gradio application is deployed as a web service using Render.

The application uses:

app.py

and loads the trained model from:

ats_xgboost_model.pkl

The application automatically uses the port provided by the deployment environment.

📁 Project Structure
ATS-Score-Prediction/
│
├── app.py
├── ats_xgboost_model.pkl
├── ats-score-prediction-project.ipynb
├── requirements.txt
├── .python-version
└── README.md
📓 Main Notebook

The complete machine learning workflow, including data processing, feature generation, model training, evaluation, cross-validation, and model selection, is available in:

ats-score-prediction-project.ipynb
📄 License

This project is licensed under the MIT License.
