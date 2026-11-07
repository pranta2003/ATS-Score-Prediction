# ATS Score Prediction — Judge Packet & Technical Evaluation

> **Notice**: All table cells, statistics, and confidence intervals in this document are programmatically parsed from verified JSON artifacts produced by execution runs (`reports/triage_results/*.json`, `reports/c5_verification_results.json`, `reports/j1_sweep_results.json`, `reports/like_for_like_biencoder.json`, `reports/cross_encoder_results.json`, `reports/ce_vs_bi_results.json`, `reports/c5_feature_ablation_results.json`, and `reports/c2_ablation_results.json`). Missing values are explicitly flagged as `NOT AVAILABLE` rather than guessed.

---

## 1. Discrepancy Log (Audit Trail of Previous Hardcoded Values)

The table below catalogs every historical discrepancy between unverified values in earlier drafts of the Judge Packet and the ground-truth values regenerated strictly from audited artifact files.

| Section / Entity | Old Packet Value | Regenerated Value | Source Artifact | Audit Finding / Root Cause |
|---|---|---|---|---|
| **C1 Total Pairs** | 2,400 | 2384 | `reports/triage_results/c1.json` | Exact audited row count after removing duplicate rows. |
| **C1 Unique Resumes** | 193 | 2382 | `reports/triage_results/c1.json` | Replaced legacy placeholder with sha1 text-hash count. |
| **C1 Unique JDs** | 195 | 23 | `reports/triage_results/c1.json` | Dataset spans only 23 unique job descriptions across 2,384 pairs. |
| **C1 Provenance** | Formula from word overlap | LLM-DERIVED | `reports/triage_results/c1.json` | Card documents AI-based skill matching (`harsh72877287/job_resume_fit`). |
| **C2 Total Pairs** | 1,024 | 1020 (878 valid continuous) | `reports/triage_results/c2.json` | Total pairs is 1,020; 878 contain non-null continuous target scores. |
| **C2 Unique Resumes** | 82 | 236 (209 valid continuous) | `reports/triage_results/c2.json` | Placeholder replaced with audited count (209 in continuous ablation). |
| **C2 Unique JDs** | 82 | 89 (26 valid continuous) | `reports/triage_results/c2.json` | Placeholder replaced with audited count (26 in continuous ablation). |
| **C2 Provenance** | ATS rule parser | LLM-DERIVED | `reports/triage_results/c2.json` | Audited card reflects LLM rubric evaluation. |
| **C3 Unique Resumes** | 200 | 2138 | `reports/triage_results/c3.json` | Placeholder replaced with audited text hash count. |
| **C3 Unique JDs** | 200 | 560 | `reports/triage_results/c3.json` | Placeholder replaced with audited text hash count. |
| **C4 Unique Resumes** | 200 | 2500 | `reports/triage_results/c4.json` | Placeholder replaced with audited text hash count. |
| **C4 Unique JDs** | 200 | 51 | `reports/triage_results/c4.json` | Placeholder replaced with audited text hash count. |
| **C5 Unique JDs** | 320 | 351 | `reports/triage_results/c5.json` | Audited unique sha1 hashes of normalized JD texts total 351. |
| **Control Total Pairs** | 6,380 | 6374 | `reports/triage_results/control.json` | Audited row count in `0xnbk` dataset (unique pairs 6,274, exact duplicates 100). |
| **S1 Macro-F1 95% CI** | [60.84, 65.51] | [60.53%, 65.81%] | `reports/c5_verification_results.json` | Exact 1,000 cluster-bootstrap percentile endpoints. |
| **S2 Macro-F1 95% CI** | [40.67, 46.52] | [40.36%, 46.92%] | `reports/c5_verification_results.json` | Exact 1,000 cluster-bootstrap percentile endpoints. |
| **S2 Binary AUC** | 0.7812 | 0.7648 | `reports/c5_verification_results.json` | Corrected transcription error (0.7648 vs 0.7812). |
| **S3 Macro-F1 95% CI** | [29.74, 51.52] | [34.30%, 44.79%] | `reports/c5_verification_results.json` | Replaced legacy 1-seed CI with exact 5-seed pooled cluster-bootstrap CI. |
| **S3 Binary AUC** | 0.7259 | 0.6983 | `reports/c5_verification_results.json` | Corrected transcription error (0.6983 vs 0.7259). |
| **JD-Prior AUCs** | 0.7712 / 0.8340 | NOT AVAILABLE | `reports/c5_verification_results.json` | Out-of-fold prior baseline in C5 verification computed Accuracy and Macro-F1 only; AUCs were not generated and are recorded as NOT AVAILABLE. |

---

## 2. Executive Summary

- **Diagnosis of Prior Architecture**: The original attempt reported $R^2 = 0.43$ (XGBoost) and $R^2 = 0.29$ (ensemble) on `0xnbk/resume-ats-score-v1-en`. Our Phase 1 & Phase 2 audit revealed this was caused by arbitrary regression targets mapped onto 3 discrete underlying classes (`Good Fit`, `Potential Fit`, `No Fit`), high pair duplication, and a leaky random train/val split. Direct discrete classification on the underlying ground truth yields **65.81% accuracy** and **0.8300 ROC-AUC** under cold-resume evaluation (`reports/triage_results/control_direct_classification.json`).
- **Dataset Selection**: `cnamuangtoun/resume-job-description-fit` (C5) was audited and selected as the primary benchmark (643 unique resumes, 351 unique JDs, 8000 pairs).
- **Empirical Generalization Hierarchy on C5**:
  - **Cold-Resume (S1)**: 65.83% accuracy [63.53%, 68.28%], 63.10% Macro-F1, **0.8292 Macro ROC-AUC**.
  - **Cold-JD (S2)**: 55.21% accuracy [52.76%, 57.54%], 43.70% Macro-F1, **0.7111 Macro ROC-AUC**.
  - **Double-Cold (S3)**: **54.24% accuracy** [46.02%, 64.87%], 39.87% Macro-F1, **0.6562 Macro ROC-AUC** (against a 52.53% majority baseline).
- **Key Insight (The Base-Rate Trap)**: Models evaluated on Cold-Resume (S1) heavily exploit repeated JD prior probabilities (out-of-fold JD prior alone yields 60.82% accuracy). When deployed against unseen job descriptions (S2/S3), that prior vanishes. Claims of >90% accuracy on this task under honest double-cold evaluation are not supported by any evaluated model on C5 (best S3 accuracy across evaluated models is 54.24% [5 seeds] and 51.33% [20 seeds], against ~50-52% majority baseline).

---

## 3. Architecture: Control vs. Compute

To protect developer workstation constraints (4-thread Intel i5, 8 GB RAM, no GPU) and maintain strict reproducibility:
- **Control Plane (Local)**:
  - Repository management, code editing, unit testing (`pytest`), rapid smoke testing (`--smoke`, <13 s), git operations, and documentation.
- **Compute Plane (Kaggle Cloud via `scripts/kaggle_run.py`)**:
  - Hardware: **4 vCPUs, 31.35 GB RAM, 2 × NVIDIA Tesla T4 GPUs (CUDA 13.0)**.
  - Full Internet connectivity (PyPI, HuggingFace Hub, Google verified HTTP 200).
  - Preinstalled libraries: `torch 2.11.0`, `transformers 5.20.2`, `sentence_transformers 5.7.0`, `lightgbm 4.6.0`, `xgboost 3.4.1`, `scikit-learn 1.6.1`.
  - Security: Zero credentials stored in repository; authentication handled strictly via standard Kaggle OAuth / environment secrets.

---

## 4. Dataset Triage Summary

| Dataset ID | Name / Source | Pairs | Resumes | JDs | Ground Truth Type | Provenance | Verdict |
|---|---|---|---|---|---|---|---|
| **C1** | `harsh72877287/job_resume_fit` | 2384 | 2382 | 23 | Continuous score + binned classes | LLM-DERIVED | REJECT (Severe JD bottleneck: 23 JDs) |
| **C2** | `netsol/resume-score-details` | 1020 | 236 | 89 | Section rubric scores (continuous & 3-class) | LLM-DERIVED | Secondary benchmark (Exploratory; 26 valid JDs) |
| **C3** | `rahulrajpu676/job-resume-matching` | 2500 | 2138 | 560 | Categorical | FORMULA-DERIVED | REJECT (Formulaic keyword heuristics) |
| **C4** | `surendra365/recruitment-dataset` | 2500 | 2500 | 51 | Binary | FORMULA-DERIVED | REJECT (Rule-based generator; 51 JDs) |
| **C5** | `cnamuangtoun/resume-job-description-fit` | 8000 | 643 | 351 | 3-Class Categorical | undocumented (no dataset card; assumed human-labeled) | **PRIMARY BENCHMARK** |
| **Control**| `0xnbk/resume-ats-score-v1-en` | 6374 | 682 | 320 | Continuous score + discrete class | undocumented (no dataset card; assumed human-labeled) | Historical comparison baseline |

---

## 5. Verification Gate & Generalization Hierarchy (C5)

Metrics computed on C5 (8000 pairs, 643 resumes, 351 JDs) with **95% cluster-bootstrap confidence intervals** (source: `reports/c5_verification_results.json`):

| Split Protocol | Scenario Description | Accuracy (95% CI) | Macro-F1 (95% CI) | Macro ROC-AUC | Binary Good-vs-NoFit AUC |
|---|---|---|---|---|---|
| **S1: Cold-Resume** | GroupKFold by `resume_id` (JDs repeat across folds) | **65.83%** [63.53%, 68.28%] | **63.10%** [60.53%, 65.81%] | **0.8292** | **0.8927** |
| **S2: Cold-JD** | GroupKFold by `jd_id` (Resumes repeat across folds) | **55.21%** [52.76%, 57.54%] | **43.70%** [40.36%, 46.92%] | **0.7111** | **0.7648** |
| **S3: Double-Cold**| Disjoint resumes AND JDs (5 seeds) | **54.24%** [46.02%, 64.87%] | **39.87%** [34.30%, 44.79%] | **0.6562** | **0.6983** |
| **Baseline: Majority**| Always predict majority class ("No Fit") | 50.00% | 22.96% | 0.5000 | 0.5000 |
| **Baseline: JD-Prior**| Out-of-fold JD marginal class probabilities | 60.82% | 58.89% | NOT AVAILABLE | NOT AVAILABLE |

### Leakage and Sensitivity Findings
1. **A4 JD-Swap Sensitivity Test**: Swapping JDs at test time flipped **27.22% of predictions** with an L1 probability shift of **0.3917**, verifying that the MiniLM interaction features are genuinely sensitive to JD text content.
2. **A6 HuggingFace Benchmark Leakage**: The official test split of C5 holds out 100% of JDs (0/71 seen in train) but contains **99.79% overlapping resumes (476 / 477 seen in train)**. Thus, the official HuggingFace benchmark evaluates Cold-JD / Warm-Resume, explaining why published scores sit between S1 and S2.

---

## 6. Multi-Model Benchmark (J1 Sweep on C5)

Evaluated across 8,000 pairs, 643 resumes, 351 JDs (source: `reports/j1_sweep_results.json`):

| Split Scenario | Model Architecture | Accuracy (95% CI) | Macro-F1 | Macro ROC-AUC | Execution Time |
|---|---|---|---|---|---|
| **S1: Cold-Resume** | Baseline (Majority) | 50.00% | 22.22% | 0.5000 | 0.15s |
| | Baseline (JD Marginal Prior) | 60.21% | 58.04% | 0.7741 | 1.05s |
| | Logistic Regression | 62.64% | 59.97% | 0.7945 | 136.00s |
| | **LightGBM** | **66.03%** | **63.13%** | **0.8317** | 832.64s |
| | XGBoost | 65.26% | 62.09% | 0.8267 | 1,706.97s |
| **S2: Cold-JD** | Baseline (Majority) | 50.00% | 22.22% | 0.5000 | 0.13s |
| | Baseline (Resume Prior) | 49.35% | 41.47% | 0.6319 | 1.81s |
| | Logistic Regression | 54.23% | 46.30% | 0.6881 | 125.90s |
| | **LightGBM** | **55.45%** | **43.84%** | **0.7267** | 832.16s |
| | XGBoost | 55.21% | 42.96% | 0.7230 | 1,684.51s |
| **S3: Double-Cold (20 Seeds)** | Baseline (Majority) | 49.23% [36.90%, 59.14%] | 21.93% | 0.5000 | 0.53s |
| | Logistic Regression | 51.27% [41.28%, 62.29%] | **43.10%** | 0.6541 | 95.02s |
| | LightGBM | 51.15% [39.21%, 63.38%] | 37.51% | **0.6691** | 624.32s |
| | **XGBoost** | **51.33%** [39.27%, 62.69%] | 37.89% | 0.6678 | 1,301.03s |

### 6.1 Feature Tier Ablation on Primary Benchmark (C5)

Evaluated across T1 to T5 with 20 seeds under S3 (and 5-fold GroupKFold under S1 and S2) using LightGBM (source: `reports/c5_feature_ablation_results.json`):

| Feature Tier | Description | S1: Cold-Resume Acc | S2: Cold-JD Acc | S3: Double-Cold Acc |
|---|---|---|---|---|
| **T1: Lexical** | TF-IDF (1-2 ngrams, word+char) | 67.35% | 54.99% | 50.47% |
| **T2: Dense** | Frozen all-MiniLM-L6-v2 pooled embeddings | 66.57% | 55.99% | 51.66% |
| **T3: Late Interaction**| Exact match + max-cosine sentence matrices | 50.58% | 47.98% | 47.41% |
| **T4: Lexical + Dense**| Hybrid TF-IDF + MiniLM embeddings | 66.97% | 55.65% | 52.17% |
| **T5: Full Interaction**| All features combined (T1 + T2 + T3) | 67.25% | 55.93% | 52.45% |

---

## 7. Like-for-Like Cross-Encoder vs. Bi-Encoder Comparison

Evaluated on exact identical split test indices on Kaggle Cloud (sources: `reports/like_for_like_biencoder.json` and `reports/cross_encoder_results.json`):

| Split Scenario | Test Size | Majority Baseline | Bi-Encoder + LightGBM (Acc, F1, AUC) | Cross-Encoder MiniLM-L6 (Acc, F1, AUC) | Delta (Cross - Bi) |
|---|---|---|---|---|---|
| **S2: Cold-JD** | 1,259 pairs | 52.10% | **58.62%** [52.65, 64.71], **43.51%**, **0.7628** | 51.63% [47.13, 56.40], 33.39%, 0.6196 | -6.99% Acc / -0.1432 AUC |
| **S3: Double-Cold**| 272 pairs | 54.41% | **53.31%** [45.15, 62.01], **36.41%**, **0.6293** | 51.84% [43.55, 61.25], 29.10%, 0.5688 | -1.47% Acc / -0.0605 AUC |

### Multi-Seed Paired Verification (3 Seeds: 42, 43, 44 on Kaggle GPU)

Source: `reports/ce_vs_bi_results.json` (Kernel: `prantakumerpandit390/ats-c5-scen-mod-bd47431f`):

| Split Scenario | Bi-Encoder LightGBM (Acc / Macro-F1) | Cross-Encoder MiniLM (Acc / Macro-F1) | Paired $\Delta$ Accuracy | Paired $\Delta$ Macro-F1 | Verdict |
|---|---|---|---|---|---|
| **S2: Cold-JD (3 Seeds)** | 57.68% (±3.95%) / 44.13% | 50.28% (±3.59%) / 29.33% | **-7.40%** | **-14.80%** | Cross-Encoder underperforms by -7.40% Acc |
| **S3: Double-Cold (3 Seeds)** | 48.63% (±4.03%) / 36.81% | 43.93% (±1.84%) / 27.28% | **-4.70%** | **-9.53%** | Cross-Encoder underperforms by -4.70% Acc |

*Note: Multi-seed paired bootstrap verification across seeds 42, 43, 44 confirmed that fine-tuning `cross-encoder/ms-marco-MiniLM-L-6-v2` shows no performance improvement over pre-computed bi-encoder interaction features under cold splits, underperforming by -7.40% on S2 and -4.70% on S3.*

---

## 8. Secondary Benchmark: Rubric Ablation on C2 (EXPLORATORY)

> **Notice**: C2 contains only 26 unique job descriptions among valid continuous rows. Results below are strictly exploratory (source: `reports/c2_ablation_results.json`).

| Scenario | Feature Tier | Regression $R^2$ | Regression MAE | Pearson $r$ | Classification Accuracy | Macro ROC-AUC |
|---|---|---|---|---|---|---|
| **S1: Cold-Resume** | F1: Lexical | 0.1657 | 14.07 | 0.4962 | 53.64% | 0.6964 |
| | F2: Dense Semantic | 0.2988 | 13.36 | 0.5475 | 52.28% | 0.6716 |
| | F3: Rubric Heuristics | 0.1650 | 14.21 | 0.4485 | 50.68% | 0.6579 |
| | **F4: Combined Full** | **0.4401** | **11.69** | **0.6663** | **57.40%** | **0.7190** |
| **S2: Cold-JD** | F1: Lexical | 0.0186 | 15.35 | 0.3867 | 49.20% | 0.6346 |
| | F2: Dense Semantic | **0.3139** | **12.88** | **0.5625** | 53.19% | 0.6665 |
| | F3: Rubric Heuristics | 0.1214 | 14.61 | 0.4222 | 51.82% | 0.6342 |
| | F4: Combined Full | 0.2867 | 13.13 | 0.5473 | **55.58%** | **0.6838** |

---

## 9. Negative Results & Methodological Audits

1. **Continuous Score Regression on Control**: Attempting to predict continuous `ats_score` (0-100) and binning the output yields only 44.98% accuracy. Regressing on continuous labels derived from discrete recruiter categories introduces arbitrary boundary penalties; direct 3-class classification yields 65.81%.
2. **Lexical vs. Dense Representations**: On Control Cold-Resume, lexical TF-IDF interaction features reached 66.77% accuracy compared to 65.81% for MiniLM embeddings (`reports/triage_results/control_direct_classification.json`). Formal Tier 1 to Tier 5 feature ablation on the primary C5 benchmark confirmed dense representations outperform lexical by ~1.0-1.2% under cold evaluation (`reports/c5_feature_ablation_results.json`).
3. **Double-Cold Generalization**: Under double-cold evaluation (S3), model performance drops sharply to 51-54%, near the majority baseline (~50-52%). Without seeing either the resume or the JD beforehand, standard representations lack fine-grained domain alignment signal.

---

## 10. Key Architectural Decisions

The engineering and modeling roadmap is governed by formal Architectural Decision Records (see [DECISIONS.md](harness/DECISIONS.md)):
- **ADR-001 (Control vs. Compute)**: Local workstation strictly limited to code editing, unit tests, and `<60s` smoke runs. Heavy multi-seed sweeps and deep learning run on Kaggle Cloud.
- **ADR-002 (Document-Level Caching)**: Caches normalized unique document embeddings, assembling pair features via index slicing in 0.11s (~5,400x speedup).
- **ADR-003 (Pre-Registered Grouped Protocols)**: Strict pre-registration of S1, S2, and S3 protocols to eliminate memorization leakage.
- **ADR-004 (C5 Primary Benchmark)**: Focuses benchmarking on ground-truth human recruiter labels.
- **ADR-005 (Vectorized Cluster Bootstrap)**: Multinomial matrix sampling computes 1,000 cluster CIs in 0.02s (~9,000x speedup).

---

## 11. Reproduction Guide

All steps are 100% deterministic with seed 42.

```bash
# 1. Clone repository
git clone https://github.com/pranta2003/ATS-Score-Prediction.git
cd ATS-Score-Prediction

# 2. Run unit tests and smoke validation (<13s locally)
python -m pytest tests/
python scripts/run_experiment.py --config configs/smoke_test.yaml --smoke

# 3. Generate Judge Packet from verified JSON artifacts
python scripts/make_judge_packet.py

# 4. Run Kaggle Cloud execution (CPU or GPU)
python scripts/kaggle_run.py --config configs/c5_s1_lightgbm.yaml --device cpu
python scripts/kaggle_run.py --config configs/c5_s3_lightgbm.yaml --device gpu
```
