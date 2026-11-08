# Empirical & Engineering Claims

This document catalogs testable scientific and architectural claims established in the ATS score prediction project. Every claim is validated against artifacts in the repository by automated tests (`tests/test_claims.py`).

---

## 1. Data Integrity & Cardinality Claims
- **Claim 1.1 (C5 Entity Cardinality):** Dataset C5 (`cnamuangtoun/resume-job-description-fit`) consists of exactly 8,000 pairs, 643 unique resume entities, and 351 unique job description entities (`reports/triage_results/c5.json`).
- **Claim 1.2 (C5 Class Distribution):** The class distribution in C5 is balanced across 3 classes: Class 0 ("No Fit") comprises exactly 50.0% of rows (4,000 pairs), Class 1 ("Potential Fit") comprises 25.0% (2,000 pairs), and Class 2 ("Good Fit") comprises 25.0% (2,000 pairs).
- **Claim 1.3 (Control vs C5 Overlap):** Control (`0xnbk/resume-ats-score-v1-en`) is an English translation/re-packaging derived from C5, sharing identical underlying entities.

## 2. Generalization Hierarchy Claims
- **Claim 2.1 (S1 Exploitation of JD Prior):** Under scenario S1 (Cold Resume), models can exploit repetitive JD entities and their associated class priors. A simple JD-prior baseline achieves >60% accuracy without reading the resume.
- **Claim 2.2 (S3 Double-Cold Generalization Ceiling):** Under strict double-cold evaluation (S3, where both resumes and JDs in the test split are unseen), machine learning models (LightGBM, XGBoost, Logistic Regression) achieve ~54% accuracy (Macro ROC-AUC $\approx 0.65\text{--}0.67$), only marginally above the majority baseline (~50--52%).
- **Claim 2.3 (Performance Degradation from S1 to S3):** Across identical feature sets, model performance strictly degrades as evaluation constraints tighten:
  $$\text{Macro ROC-AUC}(S1) > \text{Macro ROC-AUC}(S2) > \text{Macro ROC-AUC}(S3)$$
  Specifically: S1 = 0.8292, S2 = 0.7111, S3 = 0.6562.

## 3. Engineering & Performance Claims
- **Claim 3.1 (Document-Level Caching Acceleration):** Pre-computing and caching unique document embeddings allows pair features to be constructed via index slicing in $<0.5$ seconds, compared to $>300$ seconds when embedding full text pairs iteratively.
- **Claim 3.2 (Vectorized Cluster Bootstrap Speed):** Vectorized matrix multinomial sampling computes 1,000 cluster bootstrap iterations in $<0.1$ seconds.
- **Claim 3.3 (Smoke Test Runtime Guarantee):** The end-to-end smoke test (`scripts/run_experiment.py --config configs/smoke_test.yaml --smoke`) completes in $<60$ seconds on local CPU hardware.

## 4. Compute Environment Claims
- **Claim 4.1 (Remote Kaggle Specifications):** Remote compute probes verified Kaggle Cloud container specifications: 4 vCPUs, $\ge 30$ GB RAM, and dual NVIDIA Tesla T4 GPUs with unrestricted outbound internet connectivity.

## 5. Architectural & Truncation Claims
- **Claim 5.1 (Bi-Encoder vs Cross-Encoder Truncation Constraint):** On C5, 99.58% of pairs exceed 512 tokens. Under this setup (MiniLM-L6, 512 tokens), bi-encoder interaction features statistically significantly outperform the fine-tuned cross-encoder ($\Delta$ AUC = -0.1716 on S2, -0.1034 on S3, paired CIs exclude zero; `reports/ce_vs_bi_results.json`).

## 6. Rubric & Calibration Claims
- **Claim 6.1 (Decoupled Rubric & Keyword Cap):** The explainable rubric scorer decouples formatting hygiene from content fit. Its set-based keyword extraction guarantees that repeating an identical keyword produces 0 additional score increase (`tests/test_rubric_validation.py`).
- **Claim 6.2 (Temperature Scaling Calibration):** Temperature scaling ($T=1.19$ on S1, $T=1.91$ on S2) reduces expected calibration error by $>60\%$ (S1: 0.0496 to 0.0188; S2: 0.1333 to 0.0357; `results/missing_statistics.json`).

## 7. Like-for-Like Leakage Comparison Claims
- **Claim 7.1 (Measured Random vs Grouped Leakage Gap):** Evaluating the exact same Tier 2 Dense + LightGBM model on C5 (8,000 pairs) under Random Stratified 5-Fold CV yields 73.16% accuracy [95% CI: 72.17%, 74.15%], compared to 66.16% [63.85%, 68.32%] under Cold-Resume (S1) and 56.20% [53.79%, 58.65%] under Cold-JD (S2). Random pair splitting inflates accuracy by a measured +7.00 percentage points [5.48, 8.58 pp] over S1, and by +16.96 percentage points [14.58, 19.44 pp] over S2 (`results/leakage_comparison_results.json`).


