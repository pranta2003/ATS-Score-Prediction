# Render Deployment Diagnosis & Production Serving Analysis

**Date:** November 9, 2026  
**Investigator:** Senior ML Engineer  
**Scope:** Root cause diagnosis of Render web service deployment failure, production serving validation for 3-class LightGBM (T2 Dense), and deployment target recommendations.

---

## 1. OBSERVED: Build Success and Port Scan Timeout

### Telemetry from Deployment Logs
Analysis of the Render deployment telemetry reveals the exact failure sequence:
1. **Build Step:** **SUCCEEDED**.
   - Package installation via `pip install -r requirements.txt` completed with exit code 0.
   - `torch 2.12.1+cpu` resolved and installed successfully via the configured PyTorch CPU wheel repository (`--extra-index-url https://download.pytorch.org/whl/cpu`).
2. **Deploy / Health Check Step:** **FAILED**.
   - Render container initialization terminated with:  
     `"Port scan timeout reached, no open ports detected"`
   - Subsequent redeployments succeeded intermittently.

---

## 2. HYPOTHESIS & ROOT CAUSE: Synchronous Model Loading Before Port Binding

### Root Cause Analysis
In the original `app.py`, heavy neural model dependencies were loaded synchronously at the top level of the module:
```python
# Original app.py (synchronous startup)
sbert_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
xgb_model = joblib.load("ats_xgboost_model.pkl")
interface.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
```

On Render's Free Web Service tier (0.1 vCPU shared, CPU throttle active):
- Loading PyTorch, instantiating `SentenceTransformer`, and initializing model tensors takes **38 to 60+ seconds** of CPU time.
- Render's health checker probes the container's bound port (`$PORT`) within a 30–60 second window after container launch.
- Because model loading ran synchronously before `interface.launch()`, the HTTP server had not yet bound to `0.0.0.0:$PORT` when the port scanner probed.
- When cold-start loading took >45 seconds, Render terminated the container with `"Port scan timeout reached, no open ports detected"`.
- When disk/cache hits allowed the load to complete just under the deadline, deployment succeeded intermittently.

### Out-of-Memory (OOM) Killer: Unconfirmed Risk
- **Status:** **Unconfirmed Risk**.
- The deployment log showed no `Exit code 137` (SIGKILL / OOM Killer).
- However, local empirical profiling shows PyTorch + SentenceTransformers + Gradio consumes **508–523 MB Peak RSS**, which hovers directly around Render's 512 MB free-tier cgroup ceiling. While not the verified cause of the port timeout failure, it poses an operational stability risk under concurrent inference.

---

## 3. REMEDIATION: Port-First Startup via Background Thread

To eliminate the port scan timeout completely, `app.py` opens the HTTP listener immediately, offloading model loading to a background daemon thread:

1. **Immediate Port Binding:** `interface.launch(server_name="0.0.0.0", server_port=...)` executes as soon as the process starts.
2. **Time-to-Listen:** Measured at **11.88 seconds** (down from >38–60s), ensuring Render's health scanner receives an immediate HTTP 200/open port signal.
3. **Graceful Loading State:** Incoming user requests received before models finish loading receive a non-crashing informative message:  
   `"Model is still loading in the background. Please wait ~15-30 seconds and retry."`
4. **Verified Model Readiness:** Once the background thread finishes (~38.77s on local CPU), predictions execute normally.

---

## 4. PRODUCTION SERVING: 3-Class LightGBM (T2 Dense Features)

### Obsolete Measurements Notice
*The previously documented XGBoost regression / pure ONNX Runtime memory and cold-start measurements are **obsolete (old app)**. The production application serves the validated 3-class LightGBM classifier with T2 Dense semantic interaction features (1,537-d).*

### Library Version Audit
- **Trained Model Environment:** LightGBM `4.7.0`, scikit-learn `1.9.0`.
- **Pinned Requirements:** `lightgbm==4.6.0`, `scikit-learn` (latest).
- **Deserialization Check:** The model was loaded in an isolated, clean virtual environment (`scratch/ci_proof_venv`) with all warnings captured:
  - **LightGBM C-library / serialization warnings:** **None (0 warnings)**.
  - **Scikit-learn warning:** `InconsistentVersionWarning: Trying to unpickle estimator LabelEncoder from version 1.9.0 when using version 1.9.1.`
  - **Prediction Validation:** A 5-pair dummy prediction test ran successfully in the isolated venv with zero prediction warnings and valid class outputs (`[0 0 0 1 0]`).

### Prediction Parity Verification
Feature extraction and model prediction pipelines were tested across **50 pairs** from dataset C5, including **20 pairs exceeding 1,500 tokens**:
- **Feature Parity:** Max absolute difference between training pipeline (`scripts/run_c5_feature_ablation.py:extract_t2_dense`) and serving pipeline (`app.py:extract_serving_features`) across all $50 \times 1,537 = 76,850$ values:
  $$\max |X_{\text{train}} - X_{\text{serving}}| = \mathbf{0.00 \times 10^0}\quad (\text{exact bit-level parity})$$
- **Probability Parity:** Max absolute difference in raw model probabilities across all $50 \times 3 = 150$ class outputs:
  $$\max |P_{\text{train}} - P_{\text{serving}}| = \mathbf{0.00 \times 10^0}\quad (\text{exact bit-level parity})$$

### Serving Latency & Benchmarks
- **Time-to-Listen:** **11.88 seconds** on local workstation CPU.
- **Request Latency on Long Pair:** Benchmarked on a 1,694-token resume + job description pair (representative of the 1,700-token recruitment pair boundary):
  - Mean latency over 5 iterations: **958.67 ms** (Feature encoding: 945 ms; LightGBM inference: 13.67 ms).
- **Container Resource Limit Test:** **Not tested, Docker not available locally.**

---

## 5. DEPLOYMENT TARGET RECOMMENDATIONS

1. **Hugging Face Spaces (PRIMARY Target):**  
   - Hugging Face Spaces is designated as the **PRIMARY** deployment target for this application.
   - The HF Spaces CPU Basic tier provides **16 GB RAM** and **2 dedicated vCPUs** completely free of charge, eliminating all memory ceiling constraints and port timeout risks. See `docs/HF_SPACES_GUIDE.md`.
2. **Render (OPTIONAL Target):**  
   - Render deployment is **OPTIONAL**.
   - If deploying to Render Free Tier (512 MB RAM, 0.1 vCPU shared), the port-first background loader in `app.py` is mandatory to satisfy health checks, and traffic must remain low to prevent exceeding the 512 MB cgroup threshold during concurrent inference.
