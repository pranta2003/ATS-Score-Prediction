# Kaggle Environment Probe Report

## Overview
This report documents the hardware, environment capabilities, network connectivity, and preinstalled machine learning dependencies probed on Kaggle Cloud containers (CPU and GPU instances) on November 5, 2026.

---

## Hardware & Compute Capabilities

| Metric | Kaggle CPU Worker | Kaggle GPU Worker | Local Control Laptop (Comparison) |
|---|---|---|---|
| **Processor / vCPUs** | 4 vCPUs | 4 vCPUs | 2 cores / 4 threads @ 3.50 GHz |
| **System RAM** | **31.35 GB** (30.40 GB available) | **31.35 GB** (30.29 GB available) | 8.00 GB (1.10 GB available) |
| **GPU Hardware** | None (`torch.cuda.is_available() = False`) | **2 × NVIDIA Tesla T4** | Intel HD Graphics 620 (No CUDA) |
| **CUDA Driver Version** | N/A | **CUDA 13.0** (`torch 2.11.0+cu130`) | N/A |
| **Operating System** | Linux 6.18.55+ x86_64 | Linux 6.18.55+ x86_64 | Windows 10 Pro 22H2 |
| **Python Version** | 3.13.18 | 3.13.16 | 3.13.7 |

---

## Network & Internet Reachability

Kaggle kernel execution with `"enable_internet": "true"` was probed for external endpoints:

- **PyPI (`https://pypi.org`)**: **Reachable (HTTP 200)**
- **HuggingFace Hub (`https://huggingface.co`)**: **Reachable (HTTP 200)**
- **Google (`https://www.google.com`)**: **Reachable (HTTP 200)**

### Operational Protocol
Because **Internet is ON**:
1. Kernels can download Hugging Face dataset splits (`cnamuangtoun/resume-job-description-fit`) and model checkpoints (`sentence-transformers/all-MiniLM-L6-v2`, cross-encoders) directly during execution.
2. Code synchronization from the repository is pushed directly via the driver kernel script.

---

## Preinstalled Machine Learning Packages

All target libraries required for the project are natively preinstalled:

| Library | CPU Version | GPU Version | Local Version | Status |
|---|---|---|---|---|
| `torch` | 2.11.0+cpu | 2.11.0+cu130 | 2.6.0+cpu | Ready |
| `transformers` | 5.20.2 | 5.20.2 | 4.49.0 | Ready |
| `sentence_transformers` | 5.7.0 | 5.7.0 | 3.4.1 | Ready |
| `lightgbm` | 4.6.0 | 4.6.0 | 4.6.0 | Ready |
| `xgboost` | 3.4.1 | 3.4.1 | 2.1.4 | Ready |
| `scikit-learn` | 1.6.1 | 1.6.1 | 1.6.1 | Ready |
| `datasets` | 4.8.5 | 4.8.5 | 3.3.2 | Ready |
| `optuna` | 5.0.0 | 5.0.0 | 4.2.1 | Ready |
| `accelerate` | 1.15.0 | 1.15.0 | N/A | Ready |
| `scipy` | 1.16.3 | 1.16.3 | 1.15.2 | Ready |
| `pandas` | 2.3.3 | 2.3.3 | 2.2.3 | Ready |
| `numpy` | 2.1.3 | 2.1.3 | 2.2.3 | Ready |

---

## Verification & Status
- **CPU Probe Kernel**: `prantakumerpandit390/ats-probe-cpu` -> `KernelWorkerStatus.COMPLETE`
- **GPU Probe Kernel**: `prantakumerpandit390/ats-probe-gpu` -> `KernelWorkerStatus.COMPLETE`

