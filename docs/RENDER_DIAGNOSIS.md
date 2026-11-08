# Render Deployment Diagnosis & Production Serving Analysis

**Date:** November 7, 2026  
**Investigator:** Senior ML Engineer  
**Scope:** Root cause diagnosis of Render web service deployment failure, empirical memory profiling, and serving architecture recommendations.

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

In the upgraded production deployment, the legacy XGBoost regression model has been replaced with the validated 3-class LightGBM classifier (`models/c5_lightgbm_final.joblib`) with temperature scaling calibration.


On Render's Free Web Service tier (0.1 vCPU shared, CPU throttle active):
- Loading PyTorch, instantiating `SentenceTransformer`, and initializing model tensors takes **38 to 60+ seconds** of CPU time.
- Render's health checker probes the container's bound port (`$PORT`) within a 30–60 second window after container launch.
- Because model loading ran synchronously before `interface.launch()`, the HTTP server had not yet bound to `0.0.0.0:$PORT` when the port scanner probed.
- When cold-start loading took >45 seconds, Render terminated the container with `"Port scan timeout reached, no open ports detected"`.
- When disk/cache hits allowed the load to complete just under the deadline, deployment succeeded intermittently.

### Out-of-Memory (OOM) Killer: Unconfirmed Risk
- **Status:** **Unconfirmed Risk**.
- The deployment log showed no `Exit code 137` (SIGKILL / OOM Killer).
- However, local empirical profiling shows PyTorch + SentenceTransformers + Gradio consumes **508–523 MB Peak RSS**, which hovers directly around Render's 512 MB free-tier cgroup ceiling. While not the verified cause of the port timeout failure, it poses a severe stability hazard under concurrent inference.

---

## 3. REMEDIATION: Port-First Startup via Background Thread

To eliminate the port scan timeout completely, `app.py` has been refactored to open the HTTP listener immediately, offloading model loading to a background daemon thread:

1. **Immediate Port Binding:** `interface.launch(server_name="0.0.0.0", server_port=...)` executes as soon as the process starts.
2. **Time-to-Listen:** Measured at **11.88 seconds** (down from >38–60s), ensuring Render's health scanner receives an immediate HTTP 200/open port signal.
3. **Graceful Loading State:** Incoming user requests received before models finish loading receive a non-crashing informative message:  
   `"Model is still loading in the background. Please wait ~15-30 seconds and retry."`
4. **Verified Model Readiness:** Once the background thread finishes (~38.77s on local CPU), predictions execute normally (verified ATS prediction: 61.68, latency: 466 ms).

---

## 4. EMPIRICAL BENCHMARKS: PyTorch vs Pure ONNX Runtime

### Retraction & Correction of Earlier ONNX Figures
In earlier diagnostics, ONNX Runtime was reported as consuming 521.79 MB (higher than PyTorch).  
**Diagnostic Retraction:** That measurement was corrupted because the script imported `transformers.AutoTokenizer`, which silently loaded `torch` into the Python process, allocating dual runtime arenas.

### Fresh Subprocess Benchmark (Strictly Isolated, `torch` Not Imported)
Empirical profiling in isolated subprocesses (`scratch/test_pure_onnx.py`) with `tokenizers` (Rust backend, zero PyTorch) and `onnxruntime` gives the true resource consumption:

| Metric | PyTorch + SentenceTransformers | Pure ONNX Runtime (Quantized MiniLM) | Difference / Honest Saving |
| :--- | :---: | :---: | :---: |
| **Model Artifact** | PyTorch weights (HF cache) | `onnx/model_quint8_avx2.onnx` (8-bit quantized) | -70 MB on disk |
| **Session Options** | PyTorch CPU threads = default | `intra_op_num_threads=1`, arena enabled | Low CPU footprint |
| **Torch Imported** | `True` | `False` (`torch in sys.modules == False`) | Zero PyTorch runtime |
| **Base RSS** | 145.89 MB | 15.80 MB | -130.09 MB |
| **Post Model Load RSS** | 489.48 MB | 240.10 MB | -249.38 MB |
| **Peak Inference RSS** | **522.77 MB** | **243.55 MB** | **-279.22 MB (~53% RAM reduction)** |
| **Cold-Start Startup Time** | 38.77 s | 3.82 s | **-34.95 s (10x faster startup)** |
| **Inference Latency** | 466.27 ms | 312.40 ms | **-153.87 ms** |
| **Render 512 MB Free Tier** | High OOM Risk (exceeds 512 MB) | **SAFE** (fits with 268 MB headroom) | **Fully viable on Free Tier** |

*Note on Precomputed Embedding Cache:* The option to serve using a precomputed embedding lookup has been **removed** from consideration, as the production application must accept arbitrary user-pasted resume and job description texts.

---

## 5. PRODUCTION RECOMMENDATION

1. **Primary Recommendation (Hugging Face Spaces - Recommended & Robust):**  
   - Deploy directly to **Hugging Face Spaces** (CPU Basic tier: **16 GB RAM**, **2 vCPUs**, 50 GB disk, free forever).
   - Providing 32x the memory headroom of Render Free (16,384 MB vs 512 MB), Hugging Face Spaces completely eliminates memory pressure, cold-start latency constraints, and OOM killer risks without complex container optimizations. See `docs/HF_SPACES_GUIDE.md`.

2. **Secondary / Optional Deployment (Render Web Service):**  
   - Render remains supported as an optional deployment target.
   - Requires keeping the background thread port-first startup in `app.py` to prevent health check port scan timeouts.
   - For free-tier long-term stability within Render's tight 512 MB cgroup limit, inference should be converted to pure ONNX runtime (`243.55 MB peak RSS`), or the service upgraded to a paid plan ($7/mo with 2 GB RAM).
