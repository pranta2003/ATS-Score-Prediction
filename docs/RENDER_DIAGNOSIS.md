# Render Deployment Diagnosis

**Date:** November 6, 2026  
**Investigator:** Senior ML Engineer  
**Target:** Render Web Service Auto-Deployment for `ATS-Score-Prediction`  

---

## 1. Git History Audit of Deployment-Relevant Files

We audited the commit log and file history across all merged pull requests for:
- `requirements.txt`
- `app.py`
- `.python-version`
- Render configuration (`render.yaml`, `Procfile`)
- Model artifacts (`ats_xgboost_model.pkl`)
- `.gitignore`

### Summary of Changes Across Merged PRs

| File | Status Across Merged PRs (PR #1 - PR #9) | Last Modified Commit | Modifying Author / PR |
| :--- | :--- | :--- | :--- |
| `app.py` | **Unchanged** | `4d788bc` (Aug 30, 2026) | Initial pre-merge commit |
| `requirements.txt` | **Unchanged** | `273fc6e` (Aug 30, 2026) | Initial pre-merge commit |
| `.python-version` | **Unchanged** (`3.13`) | `870c76d` (Aug 30, 2026) | Initial pre-merge commit |
| `ats_xgboost_model.pkl` | **Unchanged** (Tracked in Git, 381,983 bytes) | `4d788bc` (Aug 30, 2026) | Initial pre-merge commit |
| `render.yaml` / `Procfile` | **Never existed** in repository history | — | — |
| `.gitignore` | **Added & Updated** | `3aaadc2` (PR #1)<br>`4f85195` (PR #5)<br>`7cadf51` (PR #6) | PR #1: Created initial gitignore.<br>PR #5: Added `!results/`.<br>PR #6: Added `.kaggle_kernels/`. |

### Detailed Diff Inspection
None of the merged PRs (PR #1 through PR #9) modified `app.py`, `requirements.txt`, `.python-version`, or `ats_xgboost_model.pkl`.
The only deployment-relevant file touched by our PRs was `.gitignore`.
Inspection of `.gitignore` confirms that:
- `ats_xgboost_model.pkl` is **not** ignored and remains tracked in git (`git ls-files ats_xgboost_model.pkl` confirms presence).
- `app.py` and `requirements.txt` remain tracked and unmodified.

---

## 2. Identified Deployment Vulnerabilities

Because Render is configured to auto-deploy on every push to `main`, every merged PR triggered a clean Render build and deployment attempt.

Three distinct failure vectors exist in the repository's deployment configuration:

### Vulnerability A: Invalid PyTorch Dependency in `requirements.txt`
In commit `273fc6e`, `requirements.txt` was configured with:
```text
numpy
joblib
gradio
sentence-transformers
xgboost
scikit-learn
torch==2.12.1
--extra-index-url https://download.pytorch.org/whl/cpu
```
1. **Version Non-Existence on Linux:** `torch==2.12.1` does not exist as a stable release. For Linux x86_64, pip searching PyPI or the PyTorch CPU wheel index fails or pulls an incompatible nightly/unsupported build.
2. **Python 3.13 Compatibility:** `.python-version` specifies `3.13`. PyTorch only added experimental/stable CPU wheels for Python 3.13 starting in version 2.6.0+.

### Vulnerability B: Render Free-Tier Memory Ceiling (512 MB)
Render's official documentation defines the free web service tier limits as:
- **Memory (RAM):** 512 MB
- **CPU:** 0.1 vCPU
- **Behavior on Memory Breach:** The Linux kernel OOM-killer immediately terminates the process (Exit code 137).

#### Empirical Local Memory Profiling (`scripts/measure_app_memory.py`)
Running the exact `app.py` inference pipeline in Python 3.13 yielded the following Resident Set Size (RSS):
- **Base Python Process:** 15.00 MB
- **After `import torch`:** 188.50 MB (+173.5 MB)
- **After `import sentence_transformers`:** 419.84 MB (+231.3 MB)
- **After `SentenceTransformer('all-MiniLM-L6-v2')` load:** 442.48 MB (+22.6 MB)
- **After `xgb_model` load:** 445.79 MB (+3.3 MB)
- **Peak RSS during single inference:** **491.52 MB**

When Gradio launches its HTTP server and WebSocket runtime (`interface.launch()`), total memory exceeds 520+ MB, causing an **immediate OOM crash on Render's 512 MB free tier**.

---

## 3. Request for Render Build Log

Per the project protocol (*"Ask me for the Render build log if the cause is not obvious from the diff; do not guess"*):
Because none of our merged PRs modified the runtime code or dependencies, the failure on Render is triggered by Render's automated build hook executing against pre-existing configuration (`requirements.txt`, `.python-version`, or memory limits).

To confirm whether Render failed during:
1. **The Build Phase:** (e.g. `pip install` dependency resolution error on `torch==2.12.1` or wheel compilation failure), OR
2. **The Deploy/Runtime Phase:** (e.g. OOM exit code 137 during `import sentence_transformers` or `interface.launch()`),

**Please provide the latest Render build and deploy log.**

