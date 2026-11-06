# Render Deployment Diagnosis & Realistic Serving Options

**Date:** November 6, 2026  
**Investigator:** Senior ML Engineer  
**Scope:** Root cause diagnosis of Render web service auto-deployment failure and serving architecture options.  

---

## 1. OBSERVED: Code, Git History, and Empirical Measurements

### Git Audit of Deployment-Relevant Files Across Merged PRs
An audit of `git log --stat` across all merged pull requests (PR #1 through PR #9, PR #11, PR #13) confirms:
- `app.py`: **Unmodified** across all PRs. Last commit: `4d788bc` (Aug 30, 2026).
- `requirements.txt`: **Unmodified** across all PRs. Last commit: `273fc6e` (Aug 30, 2026).
- `.python-version`: **Unmodified** (`3.13`). Last commit: `870c76d` (Aug 30, 2026).
- `ats_xgboost_model.pkl`: **Unmodified** (381,983 bytes, tracked in Git). Last commit: `4d788bc` (Aug 30, 2026).
- `render.yaml` / `Procfile`: Never created or configured in the repository.
- `.gitignore`: Updated in PR #1 (`3aaadc2`), PR #5 (`4f85195`), and PR #6 (`7cadf51`) to ignore datasets, caches, and Kaggle bundles. `ats_xgboost_model.pkl` is explicitly tracked and is **not** ignored.

### Empirical Local Profiling Results (`scripts/profile_inference_memory.py`)
All measurements below were empirically gathered by executing isolated subprocess benchmarks across 50 realistic ATS resume-JD pairs (~300 words each). The results are saved in `reports/memory_benchmark_results.json`:

| Serving Architecture | Base Process RSS | Peak RSS During Inference | Cold-Start Startup Time | Inference Latency / Pair | Status on 512 MB Free Tier |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Current App (PyTorch + SentenceTransformers + XGBoost)** | 36.77 MB | **508.59 MB** | 32.811 s | 324.42 ms | **FATAL OOM Risk** (Exceeds 512 MB with Gradio) |
| **Minimal ONNX Runtime (Quantized MiniLM, NO torch)** | 36.95 MB | **521.79 MB** | 24.709 s | 447.09 ms | **FATAL OOM Risk** (C++ ONNX runtime buffers breach 512 MB) |
| **Precomputed Embedding Cache (Zero Serving Inference)** | 36.93 MB | **153.93 MB** | 3.373 s | 1.60 ms | **SAFE** (Saves 354 MB RAM; fits in 512 MB) |

---

## 2. INFERRED: Failure Vectors

Because Render auto-deploys on every push to `main`, every merged PR triggered Render's automated build and launch hooks against the pre-existing configuration. Three primary failure mechanisms are inferred:

### Failure Vector A: Render Free-Tier Memory Boundary Breach (OOM Killer)
- **Official Specification:** Render Free Web Service instances are strictly capped at **512 MB RAM** and 0.1 vCPU (Source: [Render Free Web Services Docs](https://render.com/docs/free#free-web-services), date checked: November 2026).
- **Inference:** Our profiling shows that `import sentence_transformers`, PyTorch weight initialization, and batch inference consume 508.59 MB of Resident Set Size (RSS). When combined with the Gradio web server runtime (`interface.launch()`), total RSS surges past 530+ MB. Under Linux container cgroups, any memory breach immediately triggers the kernel OOM killer (`kill -9`, Exit Code 137).

### Failure Vector B: Unresolvable PyTorch Pin in `requirements.txt`
- **Observed Pin:** `requirements.txt` specifies:
  ```text
  torch==2.12.1
  --extra-index-url https://download.pytorch.org/whl/cpu
  ```
- **Inference:** Version `2.12.1` does not exist in PyPI or PyTorch's official wheel repository. Furthermore, `.python-version` requests Python `3.13`, for which CPU wheels were only introduced in PyTorch 2.6.0+. The `pip install -r requirements.txt` build step will fail with dependency resolution errors unless pip falls back or fails the build entirely.

### Failure Vector C: Port Binding and Cold-Start Timeouts
- **Inference:** On Render's 0.1 vCPU container, downloading 90+ MB of transformer weights from Hugging Face Hub and initializing PyTorch takes over 30–60 seconds (measured at 32.81s even on a local Core i5 CPU). Render imposes a strict port binding timeout (typically 30–60 seconds). If the application does not bind to `$PORT` within this window, Render terminates the deploy.

---

## 3. NEEDS RENDER LOG: Unknowable Without Deployment Telemetry

While the git history and memory profiling identify three distinct critical failure modes, the **exact** stage at which the current deployment failed cannot be verified without the actual Render dashboard build and deploy logs:

1. **Build Step vs Deploy Step:** Did the pipeline fail during `pip install` (Failure Vector B: non-existent `torch==2.12.1` on Python 3.13), or did `pip` compile an alternate wheel and fail during container launch?
2. **OOM vs Port Timeout:** If the build passed, did the service crash with `Exit code 137 (OOM killed)` when Gradio and SentenceTransformer loaded into memory, or did it fail with `Timed out waiting for port` due to cold-start model downloading?

**Action Required:** Please retrieve and provide the latest build and deployment log from the Render dashboard to confirm the exact termination signal.

---

## 4. REALISTIC SERVING OPTIONS FOR PRODUCTION

Serving modern neural NLP models on cloud platforms requires adequate memory allocation. Below are the three viable architectures evaluated:

### Option 1: Hugging Face Spaces (CPU Basic) — RECOMMENDED FOR ZERO COST
- **Hardware Resources:** **16 GB RAM**, **2 vCPU**, 50 GB disk (Source: [Hugging Face Spaces Hardware Specs](https://huggingface.co/docs/hub/spaces-overview#hardware-resources), date checked: November 2026).
- **Cost:** **Free**.
- **Fit for Project:** Directly supports Gradio (`app.py`), accommodates PyTorch and SentenceTransformer with negligible memory pressure (~508 MB used out of 16,384 MB available, <3.5% capacity), eliminates all OOM risks, and allows instant deployment via Git push.

### Option 2: Render Paid Starter Tier
- **Hardware Resources:** 1 GB or 2 GB RAM, 0.5–1.0 vCPU (Source: [Render Pricing](https://render.com/pricing), date checked: November 2026).
- **Cost:** \$7.00/month.
- **Fit for Project:** Provides sufficient memory headroom for the existing `app.py` pipeline (508 MB peak RSS fits safely within 1,024 MB RAM).

### Option 3: Precomputed Embedding Architecture (Render Free Tier Compatible)
- **Architecture:** Decouple embedding extraction from runtime serving. Pre-extract embeddings for resumes and job descriptions using batch offline scripts. The web application only runs embedding lookups and the XGBoost model.
- **Memory Consumption:** **153.93 MB Peak RSS** (fits comfortably within Render's 512 MB ceiling with ~350 MB margin).
- **Cold Start:** 3.37 seconds (fast startup, zero HF Hub model download overhead).

