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
