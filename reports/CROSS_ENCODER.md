# Cross-Encoder Transformer Evaluation Report (J3)

**Author:** Senior Machine Learning Engineer  
**Date:** November 6, 2026  
**Status:** Completed via Remote Kaggle Compute (`ats-c5-scen-mod-ec22929c`)  
**Model:** `cross-encoder/ms-marco-MiniLM-L-6-v2` (sequence pair cross-attention fine-tuning)  
**Dataset:** `cnamuangtoun/resume-job-description-fit` (C5)  
**Compute Execution:** Kaggle Cloud GPU (Kernel: `prantakumerpandit390/ats-c5-scen-mod-ec22929c`, Total Runtime: 1,065.24s (~17.75 minutes))  
**Branch:** `exp/cross-encoder`  

---

## 1. Motivation & Pre-Registered Protocol

To establish whether full token-level cross-attention bridges the out-of-distribution generalization gap identified in Step 2, we fine-tuned a cross-encoder transformer (`cross-encoder/ms-marco-MiniLM-L-6-v2`) directly on concatenated Resume + Job Description pairs with full bidirectional self-attention across the pair.

The protocol evaluated:
1. **Pilot Run (1 Epoch):** Measured on GPU to establish training throughput and memory stability.
2. **S2 (Cold-JD):** Grouped holdout of 20% job descriptions (train on 80% JDs, test on unseen 20% JDs; resumes repeat).
3. **S3 (Double-Cold):** Disjoint holdout of 20% unique resumes AND 20% unique job descriptions (train only on pairs containing *neither*; test on pairs containing *both*).
4. **Uncertainty Quantification:** 95% cluster-bootstrap confidence intervals (1,000 resamples clustered by JD).

---

## 2. Empirical Results (Exact Measurements)

### GPU Pilot Runtime Projection
- **1-Epoch Training Time (S3):** **129.61 seconds** (~2.16 minutes) on CUDA GPU.
- **Projected 3-Epoch Training Time:** **388.84 seconds** (~6.48 minutes).
- **Total Suite Execution Time:** **1,065.24 seconds** (~17.75 minutes) across Pilot, S2, and S3 fine-tuning and evaluation.

### Cross-Encoder Master Results Table

| Scenario | Epochs | Test Pairs | Accuracy [95% CI] | Macro-F1 [95% CI] | Balanced Accuracy | Macro ROC-AUC | Majority Baseline |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **S2 (Cold-JD)** | 3 | 1,596 | **51.63%** [47.13%, 56.40%] | **0.3339** [0.2762, 0.3817] | 40.19% | **0.6196** | 52.10% |
| **S3 (Double-Cold)** | 3 | 272 | **51.84%** [43.55%, 61.25%] | **0.2910** [0.2312, 0.3383] | 34.65% | **0.5688** | 54.41% |

---

## 3. Comparative Benchmark: Bi-Encoder vs. Cross-Encoder

We compare the fine-tuned Cross-Encoder directly against our gradient-boosted trees trained on frozen bi-encoder chunked embeddings + late-interaction features (from J1 sweep on C5):

| Scenario | Model Architecture | Accuracy | Macro-F1 | Macro ROC-AUC | vs. Majority Baseline |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **S2 (Cold-JD)** | **Bi-Encoder + LightGBM** | **55.19%** | **0.4348** | **0.7267** | **+3.09%** |
| **S2 (Cold-JD)** | **Cross-Encoder Transformer** | 51.63% | 0.3339 | 0.6196 | -0.47% (Collapses to Majority) |
| | | | | | |
| **S3 (Double-Cold)** | **Bi-Encoder + LightGBM** | **51.15%** | **0.3751** | **0.6691** | **+1.92%** |
| **S3 (Double-Cold)** | **Bi-Encoder + XGBoost** | **51.33%** | **0.3789** | **0.6678** | **+2.10%** |
| **S3 (Double-Cold)** | **Cross-Encoder Transformer** | 51.84% | 0.2910 | **0.5688** | -2.57% (Below Majority) |

---

## 4. Key Scientific Discoveries

1. **Cross-Attention Does NOT Bridge the Out-of-Distribution Gap:**
   Contrary to intuitive hypotheses that full token-level cross-attention between resume and JD would resolve semantic mismatch, fine-tuning the 22M-parameter cross-encoder on ~5,000 pairs resulted in **inferior out-of-distribution generalization** compared to frozen bi-encoders:
   - On S2 (Cold-JD), ROC-AUC dropped from **0.7267 $\rightarrow$ 0.6196** ($\Delta = -0.1071$).
   - On S3 (Double-Cold), ROC-AUC dropped from **0.6691 $\rightarrow$ 0.5688** ($\Delta = -0.1003$).
2. **Severe Overfitting to Training Formulations:**
   Under small-sample fine-tuning without hundreds of thousands of domain-specific recruiter judgments, cross-encoder self-attention overfits rapidly to the phrasing of training resumes and job descriptions. When presented with unseen vocabulary in both resumes and JDs (S3), its discrimination collapses toward chance (AUC = 0.5688).
3. **The Regularization Advantage of Frozen Embeddings:**
   Frozen representations from `all-MiniLM-L6-v2` pre-trained on 1B+ sentence pairs preserve broad general semantic geometries. Tree ensembles constrained by shallow depth and feature subsampling regularize effectively, outperforming end-to-end fine-tuning by over +10% AUC on cold entities.
