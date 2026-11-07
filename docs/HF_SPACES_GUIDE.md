# Hugging Face Spaces Deployment Guide

This guide provides step-by-step instructions for deploying the ATS Score Prediction Gradio web application to **Hugging Face Spaces**.

Hugging Face Spaces offers a **free CPU Basic tier** with:
- **16 GB RAM** (eliminates all memory pressure and OOM risks; the app consumes ~522 MB)
- **2 vCPUs** (faster inference and model loading)
- **50 GB persistent disk space**
- Native support for Gradio applications

---

## Step 1: Create a Space on Hugging Face

1. Log into your account at [huggingface.co](https://huggingface.co).
2. Navigate to [huggingface.co/new-space](https://huggingface.co/new-space).
3. Fill in the Space creation form:
   - **Space name:** e.g., `ats-score-predictor`
   - **License:** `mit` (or your preferred license)
   - **Space SDK:** Select **Gradio**
   - **Gradio template:** Select **Blank**
   - **Space hardware:** Select **CPU basic • 2 vCPU • 16GB • Free**
   - **Visibility:** Public (or Private)
4. Click **Create Space**.

---

## Step 2: Clone or Add Remote to Your Local Repository

After creating the space, Hugging Face provides a Git repository URL (e.g., `https://huggingface.co/spaces/<your-username>/ats-score-predictor`).

In your local project terminal, add Hugging Face as a git remote:

```bash
git remote add space https://huggingface.co/spaces/<your-username>/ats-score-predictor
```

*(Note: Replace `<your-username>` with your actual Hugging Face username).*

---

## Step 3: Required Files in Repository Root

Ensure the following files are present in your deployment branch / root:

1. **`app.py`**: The entry point for Gradio. Hugging Face Spaces automatically executes `python app.py`.
2. **`requirements.txt`**: Python dependencies:
   ```text
   numpy
   joblib
   gradio
   sentence-transformers
   xgboost
   scikit-learn
   torch
   ```
3. **`ats_xgboost_model.pkl`**: The trained XGBoost model artifact (382 KB, tracked directly in Git).
4. **`README.md`**: Hugging Face Spaces uses YAML frontmatter in `README.md` to configure the Space metadata. You can include:
   ```yaml
   ---
   title: ATS Score Prediction
   emoji: 📄
   colorFrom: blue
   colorTo: green
   sdk: gradio
   sdk_version: 5.9.1
   app_file: app.py
   pinned: false
   ---
   ```

---

## Step 4: Push to Hugging Face Spaces

Push your branch or `main` directly to the `space` remote:

```bash
git push space main:main
```

*(Hugging Face will prompt for your Hugging Face credentials/token via git credential manager).*

---

## Step 5: Verify Deployment

1. Return to your Space page on Hugging Face (`https://huggingface.co/spaces/<your-username>/ats-score-predictor`).
2. The Space status will display **Building**, then **Running**.
3. Once running, the interactive ATS Score Prediction UI is live and accessible globally!
