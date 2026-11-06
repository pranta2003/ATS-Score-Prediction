# Kaggle Cloud Execution Audit

**Date:** November 6, 2026  
**Auditor:** Senior ML Engineer  
**Repository:** `ATS-Score-Prediction`  

---

## 1. Comprehensive Kernel Execution Log

Every Kaggle kernel executed across the project's cloud compute phases is cataloged below with its execution details, completion status, error analysis, and result ingestion provenance.

| Kernel Identifier | Job Type | Target Accelerator | Runtime | Status | Result Source | Error Message / Root Cause |
| :--- | :--- | :--- | :---: | :---: | :---: | :--- |
| `prantakumerpandit390/ats-probe-cpu` | Environment Probe | CPU (4-Core) | ~40s | **COMPLETE** | Written File (`probe_results.json`) | None (Probe succeeded). |
| `prantakumerpandit390/ats-probe-gpu` | Environment Probe | GPU (2x T4) | ~45s | **COMPLETE** | Written File (`probe_results.json`) | None (Probe succeeded). |
| `prantakumerpandit390/ats-run-ats-c5-scen-mod-81b51c15` | J1 Attempt 1 | CPU (4-Core) | ~12s | **ERROR** | None | Initial job launch aborted due to missing dataset mount. |
| `prantakumerpandit390/ats-c5-scen-mod-81b51c15` | J1 Benchmark Sweep | CPU (4-Core) | 7,482s (~2.08h) | **ERROR** | **Parsed from stdout** | `FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/working/reports/j1_sweep_results.json'` at line 323 in `scripts/run_j1_sweep.py`. The computation completed all 20 seeds across S1, S2, and S3, but failed on the final write because `/kaggle/working/reports` was not pre-created via `os.makedirs`. |
| `prantakumerpandit390/ats-c2-scen-mod-59809512` | J2 Attempt 1 | CPU (4-Core) | ~20s | **ERROR** | None | Missing dataset dependency during initialization. |
| `prantakumerpandit390/ats-c2-scen-mod-174278cf` | J2 C2 Ablation | CPU (4-Core) | ~1,780s (~29.6m) | **COMPLETE** | Written File (`c2_ablation_results.json`) | None (Completed successfully, 878 pairs). |
| `prantakumerpandit390/ats-c5-scen-mod-ec22929c` | J3 Cross-Encoder | GPU (2x T4) | 1,065s (~17.75m) | **COMPLETE** | Written File (`cross_encoder_results.json`) | None (Pilot, S2, and S3 completed successfully). |

---

## 2. Root Cause Analysis for J1 (`ats-c5-scen-mod-81b51c15`)

1. **Failure Mechanism:**
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
2. **Result Recovery:**
   All summary metrics had been logged to stdout before the crash. The initial results in `reports/j1_sweep_results.json` were reconstructed by parsing the kernel's stdout log.

---

## 3. Engineering Fix: Incremental Persistence & Checksum Manifest

To eliminate the vulnerability of late-stage batch persistence failure, `scripts/run_j1_sweep.py` is upgraded with:
1. **Directory Guarantee:** `os.makedirs(os.path.dirname(out_json), exist_ok=True)` executed before any evaluation.
2. **Incremental Checkpointing:** Each completed scenario, model, and seed writes an individual atomic JSON file into `results/j1_sweep/incremental/<scenario>_<model>_<seed>.json` with `fsync()`.
3. **Checksum Manifest:** A `manifest.json` recording SHA-256 hashes of each incremental result file is updated after each checkpoint, guaranteeing data integrity.
