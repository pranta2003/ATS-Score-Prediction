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
| `reports/j1_sweep_results.json` | `ats-c5-scen-mod-9dd67db7` | 6,268 | `bf5bc4e51cbdae8b041bb2c2cbb062815d1d28c1ceace59b0f30563a8f8acd23` |
| `reports/c2_ablation_results.json` | `ats-c2-scen-mod-174278cf` | 2,918 | `9bf1d024702dc96e737948593acdfda5411e88b4d6a22056ca808e4c6731db79` |
| `reports/cross_encoder_results.json` | `ats-c5-scen-mod-ec22929c` | 1,068 | `bb2e29e60a7b386e5579a33786beddcae1bed7dbb5cda5ba220da67236d77fd1` |
| `reports/c5_feature_ablation_results.json` | `ats-c5-scen-lightgbm-068737a8` | 12,029 | `6f6e5e04ae0ae54fa4bfd5af7d12062afd14ef3ae9192936224ff0394a33bd87` |
| `reports/ce_vs_bi_results.json` | `ats-c5-scen-mod-bd47431f` | 11,228 | `0984af961436237cbc98c8f69a72729936a1c05162059ce6d497f6bd86a4e15e` |
| `results/leakage_comparison_results.json` | `ats-c5-leakage-comparison-cpu` | 1,637 | `db3104db9612577bf7f0a7a46e4d9915ee78d040e662f2eb4f1449bbcaeb38da` |
| `results/c5_leakage_oof_predictions.parquet` | `ats-c5-leakage-comparison-cpu` | 524,343 | `110417bf3db8b0e9872abc925c7a0478deab3c0590f70d3182eaf9aac83380da` |

> **Note on Final Serving Model Artifact (`models/c5_lightgbm_final.joblib`)**:
> The final serving model was fitted locally on the control laptop across all 8,000 C5 benchmark pairs using the exact hyperparameters established by the J1 benchmark sweep and cached T2 Dense embeddings (`cache/embeddings/c5_doc_emb.npy`). It is not an artifact produced or downloaded from Kaggle kernels. Its metadata and configuration are committed at `models/c5_lightgbm_final_meta.json`.

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
