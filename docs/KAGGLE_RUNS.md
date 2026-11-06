# Kaggle Cloud Execution Audit

**Date:** November 6, 2026  
**Auditor:** Senior ML Engineer  
**Repository:** `ATS-Score-Prediction`  

---

## 1. Comprehensive Kernel Execution Log

Every Kaggle kernel executed across the project's cloud compute phases is cataloged below with its execution details, completion status, error analysis, and result ingestion provenance.

| Kernel Identifier | Job Type | Target Accelerator | Runtime | Status | Result Source | Error Message / Root Cause / Verification |
| :--- | :--- | :--- | :---: | :---: | :---: | :--- |
| `prantakumerpandit390/ats-probe-cpu` | Environment Probe | CPU (4-Core) | ~40s | **COMPLETE** | Written File (`probe_results.json`) | None (Probe succeeded). |
| `prantakumerpandit390/ats-probe-gpu` | Environment Probe | GPU (2x T4) | ~45s | **COMPLETE** | Written File (`probe_results.json`) | None (Probe succeeded). |
| `prantakumerpandit390/ats-run-ats-c5-scen-mod-81b51c15` | J1 Attempt 1 | CPU (4-Core) | ~12s | **ERROR** | None | Initial job launch aborted due to missing dataset mount. |
| `prantakumerpandit390/ats-c5-scen-mod-81b51c15` | J1 Benchmark Sweep | CPU (4-Core) | 7,482s (~2.08h) | **ERROR** | **Parsed from stdout** | `FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/working/reports/j1_sweep_results.json'` at line 323 in `scripts/run_j1_sweep.py`. The computation completed all 20 seeds across S1, S2, and S3, but failed on final write because directory was not pre-created. |
| `prantakumerpandit390/ats-c2-scen-mod-59809512` | J2 Attempt 1 | CPU (4-Core) | ~20s | **ERROR** | None | Missing dataset dependency during initialization. |
| `prantakumerpandit390/ats-c2-scen-mod-174278cf` | J2 C2 Ablation | CPU (4-Core) | ~1,780s (~29.6m) | **COMPLETE** | Written File (`c2_ablation_results.json`) | None (Completed successfully, 878 valid pairs, 26 JDs). |
| `prantakumerpandit390/ats-c5-scen-mod-ec22929c` | J3 Cross-Encoder | GPU (2x T4) | 1,065s (~17.75m) | **COMPLETE** | Written File (`cross_encoder_results.json`) | None (Pilot, S2, and S3 completed successfully). |
| `prantakumerpandit390/ats-c5-scen-mod-9dd67db7` | J1 Sweep Re-run | CPU (4-Core) | ~3,500s (~58m) | **COMPLETE** | Written File (`manifest.json` & `metrics.json`) | Validated: Max absolute diff vs reconstructed stdout was 0.0050 percentage points (strictly $\le 1.0$ point threshold). |
| `prantakumerpandit390/ats-c5-scen-lightgbm-068737a8` | C5 Feature Ablation | CPU (4-Core) | ~7,100s (~118m) | **COMPLETE** | Written File (`c5_feature_ablation_results.json`) | Completed successfully across Tiers T1 to T5, S1, S2, S3 (20 seeds) with paired cluster-bootstrap CIs. |
| `prantakumerpandit390/ats-c5-scen-mod-bd47431f` | Cross-Encoder vs Bi-Encoder Study | GPU (2x T4) | 3,186s (~53.1m) | **COMPLETE** | Written File (`study_results.json` & `study_predictions.parquet`) | Completed multi-seed (seeds 42, 43, 44) S2 & S3 evaluation with identical split ID files. Verified CE underperformed Bi-Encoder. |

---

## 2. Root Cause Analysis & Re-run Verification for J1

1. **Failure Mechanism of Initial Attempt (`ats-c5-scen-mod-81b51c15`):**
   The kernel executed all experimental sweeps over 7,426.07 seconds:
   - S1 (Cold-Resume): Majority, JD-prior, Resume-prior, Logistic Regression, LightGBM, XGBoost (5 folds).
   - S2 (Cold-JD): Majority, JD-prior, Resume-prior, Logistic Regression, LightGBM, XGBoost (5 folds).
   - S3 (Double-Cold): 20 independent seeds across all 6 models with 1,000 bootstrap iterations.
   At line 323 of `scripts/run_j1_sweep.py`:
   ```python
   with open("/kaggle/working/reports/j1_sweep_results.json", "w", encoding="utf-8") as f:
       json.dump(results, f, indent=2)
   ```
   Because `os.path.dirname(out_json)` (`/kaggle/working/reports`) did not exist, Python raised `FileNotFoundError: [Errno 2] No such file or directory`. The kernel terminated with exit code 1.
2. **Re-run Verification (`ats-c5-scen-mod-9dd67db7`):**
   The re-run completed with incremental checkpointing (`results/j1_sweep/incremental/*.json`) and an atomic SHA-256 `manifest.json`.
   A side-by-side comparison of every metric confirmed:
   $$\max |\text{Metric}_{\text{reconstructed}} - \text{Metric}_{\text{rerun}}| = 0.000050 \text{ (0.0050 percentage points)}$$
   Meeting the stopping threshold of $\le 1.0$ percentage point, verifying the exact integrity of the reported numbers.

---

## 3. Engineering Fix: Incremental Persistence & Checksum Manifest

To eliminate the vulnerability of late-stage batch persistence failure, `scripts/run_j1_sweep.py` was upgraded with:
1. **Directory Guarantee:** `os.makedirs(os.path.dirname(out_json), exist_ok=True)` executed before any evaluation.
2. **Incremental Checkpointing:** Each completed scenario, model, and seed writes an individual atomic JSON file into `results/j1_sweep/incremental/<scenario>_<model>_<seed>.json` with `fsync()`.
3. **Checksum Manifest:** A `manifest.json` recording SHA-256 hashes of each incremental result file is updated after each checkpoint, guaranteeing data integrity.
