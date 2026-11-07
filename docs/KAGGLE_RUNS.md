# Kaggle Remote Execution Registry & Lifecycle Inventory

**Auditor:** Senior ML Engineer  
**Date:** November 7, 2026  
**Account:** `prantakumerpandit390`  
**Execution Policy:** Controlling on laptop; compute delegated to Kaggle for GPU / heavy tasks. **Zero kernel deletions performed** (categorized strictly for user review).

---

## 1. Active Experiment Kernels (ATS Project)

| Kernel Slug / Reference | Human Title | Accelerator | Purpose / Artifact Output | Lifecycle Status | Action Recommendation |
|---|---|:---:|---|:---:|:---:|
| `prantakumerpandit390/ats-c5-scen-mod-bd47431f` | Cross-Encoder vs Bi-Encoder Study | GPU (T4x2) | `reports/ce_vs_bi_results.json` | COMPLETE | **KEEP** (Authoritative matched CE vs Bi-Encoder study across S2/S3 seeds 42, 43, 44) |
| `prantakumerpandit390/ats-c5-scen-lightgbm-068737a8` | C5 Feature Ablation Sweep | CPU | `reports/c5_feature_ablation_results.json` | COMPLETE | **KEEP** (Authoritative 5-tier ablation across S1, S2, S3 with cluster bootstrap) |
| `prantakumerpandit390/ats-c5-scen-mod-9dd67db7` | J1 Architecture Sweep Re-run | GPU (T4x2) | `reports/j1_sweep_results.json` | COMPLETE | **KEEP** (Re-run verifying multi-seed model family sweep) |
| `prantakumerpandit390/ats-c2-scen-mod-174278cf` | C2 Dataset Ablation | CPU | `reports/c2_ablation_results.json` | COMPLETE | **KEEP** (Authoritative multi-target section ablation on C2) |
| `prantakumerpandit390/ats-c5-scen-mod-ec22929c` | J1 Architecture Sweep (Draft 1) | GPU (T4x2) | Early J1 partial logs | SUPERSEDED | **SAFE-TO-DELETE** (Superseded by complete re-run `9dd67db7`) |
| `prantakumerpandit390/ats-probe-gpu` | Environment Probe (GPU) | GPU | Hardware / CUDA / library versions | COMPLETE | **KEEP** (Authoritative GPU environment audit) |
| `prantakumerpandit390/ats-probe-cpu` | Environment Probe (CPU) | CPU | Hardware / OS / library versions | COMPLETE | **KEEP** (Authoritative CPU environment audit) |

---

## 2. Legacy / Personal Account Notebooks (Untouched)

*Per instructions, none of these notebooks have been deleted or modified.*

| Kernel Slug / Reference | Title | Date | Lifecycle Category | Recommendation |
|---|---|:---:|:---:|:---:|
| `prantakumerpandit390/mid-evo-2026` | MID_EVO_2026 | 2026-11-01 | Academic / Personal | **KEEP** |
| `prantakumerpandit390/numpy-prectice-data-mining` | numpy_prectice Data mining | 2026-10-22 | Academic / Personal | **KEEP** |
| `prantakumerpandit390/lab-dm` | Lab DM | 2026-10-20 | Academic / Personal | **KEEP** |
| `prantakumerpandit390/pandas-52-4` | pandas_52_4 | 2026-10-06 | Academic / Personal | **KEEP** |
| `prantakumerpandit390/notebookf9d947937f` | notebookf9d947937f | 2026-10-06 | Auto-generated Draft | **DELETE-FAILED-DRAFT** |
| `prantakumerpandit390/notebookf07f26c5ef` | notebookf07f26c5ef | 2026-10-06 | Auto-generated Draft | **DELETE-FAILED-DRAFT** |
| `prantakumerpandit390/classification-with-knn` | classification_with_KNN | 2026-10-06 | Academic / Personal | **KEEP** |
| `prantakumerpandit390/52-4-307-ml-lab3` | 52_4_307_ML_Lab3 | 2026-06-11 | Academic / Personal | **KEEP** |
| `prantakumerpandit390/lab-02-evo` | Lab_02_evo | 2026-05-21 | Academic / Personal | **KEEP** |
| `prantakumerpandit390/lab-1-lab` | Lab_1_lab | 2026-05-14 | Academic / Personal | **KEEP** |
| `prantakumerpandit390/exercise-summary-functions-and-maps` | Exercise: Summary Functions and Maps | 2026-01-20 | Kaggle Tutorial | **KEEP** |
| `prantakumerpandit390/notebook7b93abafe8` | notebook7b93abafe8 | 2026-01-20 | Auto-generated Draft | **DELETE-FAILED-DRAFT** |
| `prantakumerpandit390/notebook59d26d08cd` | notebook59d26d08cd | 2026-01-20 | Auto-generated Draft | **DELETE-FAILED-DRAFT** |

---

## 3. Standardized Kaggle Automation Protocol

The submission helper (`scripts/kaggle_run.py`) enforces:
1. **Predictable Slugs**: Formatted as `ats-c5-<purpose>-<cpu|gpu>` (under 50 chars).
2. **Readable Titles**: Formatted as `"ATS-Score | C5 | <Purpose> | <CPU|GPU>"`.
3. **Execution Modes**: Explicitly sets `"kernel_type": "script"`, `"language": "python"`, `"enable_internet": "true"`.
4. **Draft Auto-Execution**: Pushes metadata and code payload, initiating remote execution via `kaggle kernels push` without manual web interaction.
