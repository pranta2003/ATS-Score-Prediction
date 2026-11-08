# Kaggle Cloud Execution Audit & Artifact Provenance

**Date:** November 7, 2026  
**Auditor:** Senior ML Engineer  
**Repository:** `ATS-Score-Prediction`  
**Execution Policy:** Control on laptop (editing code, unit tests, git, docs); compute delegated to Kaggle for GPU / heavy tasks. Zero kernel deletions performed.

---

## 1. Remote Kernel Registry (ATS Project)

| Slug | Title | Accelerator | Purpose | Result Files | ARCHIVED-VERIFIED | Run Date | Status |
| :--- | :--- | :---: | :--- | :--- | :---: | :---: | :---: |
| `prantakumerpandit390/ats-probe-cpu` | Environment Probe (CPU) | CPU (4 vCPUs, 31.35 GB RAM) | Hardware, OS, and library audit on Kaggle CPU container | `.kaggle_kernels/probe_cpu_out/probe_results.json` | **yes** | 2026-11-05 | COMPLETE |
| `prantakumerpandit390/ats-probe-gpu` | Environment Probe (GPU) | GPU (2x NVIDIA Tesla T4, CUDA 13.0, 31.35 GB RAM) | Hardware, GPU, CUDA driver, and library audit on Kaggle GPU container | `reports/probe_results.json` | **yes** | 2026-11-05 | COMPLETE |
| `prantakumerpandit390/ats-c5-scen-mod-81b51c15` | J1 Architecture Sweep (Attempt 1) | CPU (4 vCPUs, 31.35 GB RAM) | Initial 20-seed sweep of 6 model families across S1, S2, S3 | Kernel execution stdout log | **no** | 2026-11-06 | ERROR (Exit 1)* |
| `prantakumerpandit390/ats-c2-scen-mod-174278cf` | C2 Dataset Ablation | CPU (4 vCPUs, 31.35 GB RAM) | C2 section ablation across 878 valid pairs and 26 JDs | `reports/c2_ablation_results.json` | **yes** | 2026-11-06 | COMPLETE |
| `prantakumerpandit390/ats-c5-scen-mod-ec22929c` | First Cross-Encoder GPU Run | GPU (2x NVIDIA Tesla T4, CUDA 13.0) | First Cross-Encoder GPU run evaluating Pilot, S2, and S3 scenarios | `reports/cross_encoder_results.json` | **yes** | 2026-11-06 | COMPLETE |
| `prantakumerpandit390/ats-c5-scen-mod-9dd67db7` | J1 Architecture Sweep Re-run | CPU (4 vCPUs, 31.35 GB RAM) | Verified J1 benchmark sweep re-run with atomic checkpointing and manifest verification | `reports/j1_sweep_results.json` | **yes** | 2026-11-06 | COMPLETE |
| `prantakumerpandit390/ats-c5-scen-lightgbm-068737a8` | C5 Feature Ablation Sweep | CPU (4 vCPUs, 31.35 GB RAM) | 5-tier dense feature ablation across S1, S2, S3 (20 seeds) with paired cluster bootstrap | `reports/c5_feature_ablation_results.json` | **yes** | 2026-11-06 | COMPLETE |
| `prantakumerpandit390/ats-c5-scen-mod-bd47431f` | Cross-Encoder vs Bi-Encoder Study | GPU (2x NVIDIA Tesla T4, CUDA 13.0) | Multi-seed (42, 43, 44) matched split evaluation of Cross-Encoder vs Bi-Encoder on S2 & S3 | `reports/ce_vs_bi_results.json` | **yes** | 2026-11-06 | COMPLETE |
| `prantakumerpandit390/ats-c5-leakage-comparison-cpu` | Ats C5 Leakage Comparison CPU | CPU (4 vCPUs, 31.35 GB RAM) | Like-for-like leakage comparison evaluating T2 Dense + LightGBM across Random Stratified, S1 Cold-Resume, and S2 Cold-JD (5 folds) | `results/leakage_comparison_results.json`, `reports/LEAKAGE_COMPARISON.md`, `results/c5_leakage_oof_predictions.parquet` | **yes** | 2026-11-08 | COMPLETE |

*\* Note on `ats-c5-scen-mod-81b51c15`: Computation completed 100% of folds and seeds over 7,426 seconds, but raised `FileNotFoundError` on final write due to an uncreated output directory. It was formally re-run and verified by `ats-c5-scen-mod-9dd67db7`.*

---

## 2. Checksum Verification of Ingested Artifacts

Every output artifact checked into the repository has been verified against its remote generation run via SHA-256 digest:

| Target Result File | Source Kaggle Kernel | Size (Bytes) | SHA-256 Checksum |
| :--- | :--- | :---: | :--- |
| `reports/probe_results.json` | `ats-probe-gpu` | 1,208 | `e9b8fea4bcff314270754d7af6a3540724febe389da45d626513b14e98791856` |
| `.kaggle_kernels/probe_cpu_out/probe_results.json` | `ats-probe-cpu` | 1,124 | `03fb8f6e85407ef3012b925939d49382d419b9257714f83a98d87ca4d0b44dcf` |
| `reports/j1_sweep_results.json` | `ats-c5-scen-mod-9dd67db7` | 6,533 | `dd0bf5107cf239c2ef09ab54d9c0b321f68197388483cd6e38eda4868649d0a9` |
| `reports/c2_ablation_results.json` | `ats-c2-scen-mod-174278cf` | 3,021 | `e3e18fb036ed325ff3589aa185358ecf23d8773bff317ed8a6d273cfe2ee130f` |
| `reports/cross_encoder_results.json` | `ats-c5-scen-mod-ec22929c` | 1,107 | `f5090f43a5537726d71ea526801db936e945294aad3e9da9e986bb2bd5b78f13` |
| `reports/c5_feature_ablation_results.json` | `ats-c5-scen-lightgbm-068737a8` | 12,461 | `488ad07504f98686ac87d4a79f0e873b96f1c49bfb8c630d870456ce90b27cc3` |
| `reports/ce_vs_bi_results.json` | `ats-c5-scen-mod-bd47431f` | 11,616 | `7c04f26c41cb497efb49d8b98032dba5fcbc21aa3e58ca12f12c48692f17c473` |
| `results/leakage_comparison_results.json` | `ats-c5-leakage-comparison-cpu` | 1,732 | `db3104db9612577bf7f0a7a46e4d9915ee78d040e662f2eb4f1449bbcaeb38da` |
| `results/c5_leakage_oof_predictions.parquet` | `ats-c5-leakage-comparison-cpu` | 524,343 | `110417bf3db8b0e9872abc925c7a0478deab3c0590f70d3182eaf9aac83380da` |

---

## 3. Root Cause Analysis & Re-run Verification for J1

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
   The re-run executed on a 4-vCPU Xeon instance (`cpu`) with incremental checkpointing (`results/j1_sweep/incremental/*.json`) and an atomic SHA-256 `manifest.json`.
   A side-by-side comparison of every metric confirmed:
   $$\max |\text{Metric}_{\text{reconstructed}} - \text{Metric}_{\text{rerun}}| = 0.000050 \text{ (0.0050 percentage points)}$$
   Strictly below the acceptance threshold of $\le 1.0$ percentage point, verifying the exact reproducibility of the reported figures.

---

## 4. Standardized Kaggle Automation Protocol

The submission helper (`scripts/kaggle_run.py`) enforces:
1. **Predictable Slugs**: Formatted as `ats-c5-<purpose>-<cpu|gpu>` (under 50 chars).
2. **Readable Titles**: Formatted as `"ATS-Score | C5 | <Purpose> | <CPU|GPU>"`.
3. **Execution Modes**: Explicitly sets `"kernel_type": "script"`, `"language": "python"`, `"enable_internet": "true"`.
4. **Draft Auto-Execution**: Pushes metadata and code payload, initiating remote execution via `kaggle kernels push` without manual web interaction.
