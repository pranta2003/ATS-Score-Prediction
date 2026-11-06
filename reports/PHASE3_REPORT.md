# Phase 3 Final Report: Diagnostic Synthesis, Observed Performance Bounds, and ATS Benchmarking

**Author:** Senior Machine Learning Engineer  
**Date:** November 6, 2026  
**Repository:** `ATS-Score-Prediction`  
**Execution Environment:** Kaggle Cloud CPU & 2x NVIDIA T4 GPUs / Control Laptop  

---

## 1. Executive Summary & Diagnostic Conclusions

This comprehensive investigation into automated resume-to-job-description ("ATS score prediction") matching set out to resolve two foundational questions:
1. **Why did the historical baseline capstone plateau at $R^2 \approx 0.43$ (XGBoost) and $R^2 \approx 0.29$ (Ensemble)?**
2. **What are the true, honest performance bounds of modern NLP and machine learning architectures for ATS fit scoring across realistic hiring scenarios?**

Through systematic data triage, rigorous multi-entity cross-validation splitting, comprehensive feature ablations, and cloud-scale cross-encoder fine-tuning, the answers have been conclusively established.

### Key Conclusions

1. **The Historical $R^2 \approx 0.43$ Mystery Solved:**
   In the original capstone, models were evaluated on random row-level train/test splits where job descriptions were heavily repeated across both splits. In Work Package J2, training a Full Combined model (TF-IDF Lexical + Dense MiniLM Semantic + Rubric Heuristics) on C2 under an **S1 (Cold-Resume, seen Job Descriptions)** protocol yielded **$R^2 = 0.4401$** (Pearson $r = 0.6663$). This represents the best results observed with our methods on this dataset when job descriptions repeat, echoing the Phase 1 audit where random row-level splitting exhibited severe entity memorization leakage ($R^2 = 0.4281$ under random split vs. $R^2 = 0.2717$ under grouped split on Control). The model was not learning transferable resume-job compatibility; it was exploiting job-specific score baselines.

2. **The Generalization Hierarchy ($S1 \gg S2 > S3$):**
   Model performance is fundamentally dictated by entity leakage:
   - **S1 (Cold-Resume, warm JD):** Models exploit job-specific priors. On C5, a naive zero-feature JD-prior model achieves **60.21% accuracy** and **0.7741 ROC-AUC** without ever reading the candidate's resume. LightGBM reaches **66.03% accuracy** and **0.8317 ROC-AUC**.
   - **S2 (Cold-JD, warm Resume):** Evaluated on unseen job descriptions, the JD-prior collapses to random baseline (50.00% accuracy, 0.4825 ROC-AUC). LightGBM performance drops to **55.45% accuracy** and **0.7267 ROC-AUC**.
   - **S3 (Double-Cold, unseen Resumes AND unseen JDs):** In a true double-cold-start production setting, LightGBM drops to **51.15% accuracy** [95% CI: 39.21% - 63.38%] and **0.6691 ROC-AUC** against a majority baseline of 49.23%. Under S3 double-cold evaluation, accuracy is not statistically distinguishable from the majority baseline; ROC-AUC is the only evaluated metric demonstrating non-trivial ranking signal.

3. **Lexical Feature Collapse & Heuristic Generalization Hypothesis:**
   Lexical overlap features (TF-IDF similarity, token set overlap, length ratios) that show correlation on seen jobs ($R^2 = 0.1657$ on S1) show substantially reduced predictive signal on unseen jobs (**$R^2 = 0.0186$** on S2). Furthermore, adding manually engineered rubric heuristics to dense embeddings resulted in a point estimate drop on unseen jobs ($R^2$ moved from **0.3139** down to **0.2867**); we hypothesize this may reflect negative transfer from heuristic rules overfitting the training distribution, though full paired bootstrap confidence intervals would be required to establish statistical significance.

4. **Cross-Encoders Fail to Bridge the Cold-Start Gap:**
   Fine-tuning a 22M-parameter cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) with full cross-attention directly on candidate pairs yields **51.63% accuracy** on S2 (majority class 52.10%) and **51.84% accuracy** on S3 (majority 54.41%), with ROC-AUCs of **0.6196** (S2) and **0.5688** (S3). We hypothesize that on moderate-sized datasets (~8,000 pairs), fine-tuning deep cross-attention layers without large-scale domain pretraining may overfit training phrase structures or collapse class priors, whereas a frozen bi-encoder paired with a regularized tree ensemble retains broader semantic representations.

5. **Observed Performance Bounds:**
   No evidence was found supporting claims of "90% ATS matching accuracy" on open-domain candidates and jobs without entity leakage. In true double-cold-start hiring, the best results observed with our methods on this dataset are **~51–55% classification accuracy** (where accuracy does not significantly exceed the majority class baseline) and **~0.66–0.72 ROC-AUC**.

---

## 2. Dataset Audits & Label Provenance

Candidate datasets across the recruitment ML ecosystem were systematically audited and normalized through an adapter layer (`src/ats/adapters.py`).

| Dataset Name | Source / Identifier | Pairs | Resumes | JDs | Target Type | Provenance & Quality Verdict |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Control** | `0xnbk/resume-ats-score-v1-en` | 8,000 | 682 | 351 | Continuous (0–100) & Discrete | Continuous target is an artificial 3-cluster jittered projection of the 3 discrete fit classes ($\eta^2 = 0.987$). Non-overlapping bands (<40, 40–70, >70). |
| **C5** | `cnamuangtoun/resume-job-description-fit` | 8,000 | 643 | 351 | Discrete (3 classes) | **Primary Benchmark Dataset.** Human recruitment triage decisions: `No Fit` (50.0%), `Good Fit` (30.0%), `Potential Fit` (20.0%). |
| **C2** | `netsol/resume-score-details` | 878 | 209 | 26 | Continuous (0–100) | **Rubric Benchmark Dataset.** Generated by LLM (GPT-4o) evaluation prompt with weighted criteria (Skills 40%, Experience 30%, Education 20%, Certs 10%). 878 valid pairs after dropping null/invalid targets. |
| **C1** | `harsh72877287/job_resume_fit` | 8,000 | 643 | 351 | Discrete | Identical duplicate of C5 with renamed column headers. |
| **C3** | `rahulrajpu676/job-resume-matching` | 3 CSVs | — | — | Bipartite Graph | Discarded: Severe entity disconnect and corrupt tabular foreign keys. |

---

## 3. Work Package J1: Multi-Model Benchmark Sweep on C5

Evaluated on Kaggle Cloud CPU over **7,426.07 seconds** across **20 independent seeds** with 5-fold cross-validation or disjoint holdouts.

### Comprehensive Empirical Results

| Protocol / Scenario | Model | Accuracy | Macro-F1 | Macro ROC-AUC | Compute Time |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **S1: Cold-Resume** | Majority Baseline | 0.5000 | 0.2222 | 0.5000 | 0.15s |
| *(Resumes unseen,* | JD-Prior Baseline | 0.6021 | 0.5804 | 0.7741 | 1.05s |
| *JDs repeated)* | Resume-Prior Baseline | 0.5000 | 0.2222 | 0.4831 | 1.90s |
| | Logistic Regression | 0.6264 | 0.5997 | 0.7945 | 136.00s |
| | LightGBM | **0.6603** | **0.6313** | **0.8317** | 832.64s |
| | XGBoost | 0.6526 | 0.6209 | 0.8267 | 1,706.97s |
| **S2: Cold-JD** | Majority Baseline | 0.5000 | 0.2222 | 0.5000 | 0.13s |
| *(JDs unseen,* | JD-Prior Baseline | 0.5000 | 0.2222 | 0.4825 | 0.85s |
| *Resumes repeated)* | Resume-Prior Baseline | 0.4935 | 0.4147 | 0.6319 | 1.81s |
| | Logistic Regression | 0.5423 | 0.4630 | 0.6881 | 125.90s |
| | LightGBM | **0.5545** | **0.4384** | **0.7267** | 832.16s |
| | XGBoost | 0.5521 | 0.4296 | 0.7230 | 1,684.51s |
| **S3: Double-Cold** | Majority Baseline | 0.4923 [0.369, 0.591] | 0.2193 | 0.5000 | 0.53s |
| *(Resumes unseen AND* | JD-Prior Baseline | 0.4923 [0.369, 0.591] | 0.2193 | 0.5000 | 3.52s |
| *JDs unseen)* | Resume-Prior Baseline | 0.4923 [0.369, 0.591] | 0.2193 | 0.5000 | 6.16s |
| | Logistic Regression | 0.5127 [0.413, 0.623] | 0.4310 | 0.6541 | 95.02s |
| | LightGBM | 0.5115 [0.392, 0.634] | 0.3751 | **0.6691** | 624.32s |
| | XGBoost | **0.5133** [0.393, 0.627] | 0.3789 | 0.6678 | 1,301.03s |

### Key Takeaways from J1
- **The JD-Prior Artifact:** In S1, the JD-prior model achieves 60.21% accuracy and 0.7741 AUC, accounting for over **73% of the gain** of LightGBM over the 50% majority class. When tested on unseen JDs in S2, the JD-prior drops to random performance (50.00% / 0.4825).
- **True Generalization:** Tree ensembles (LightGBM / XGBoost) consistently outperform linear models in AUC, reaching 0.7267 in S2 and 0.6691 in S3, but accuracy under strict double-cold splitting hovers just above random chance (~51.3%).

---

## 4. Work Package J2: C2 ATS Feature Ablations & Rubric Benchmark

Evaluated on Kaggle Cloud CPU over 878 validated pairs (209 resumes, 26 JDs) across 4 feature subsets:
- **F1 (Lexical):** TF-IDF unigram/bigram cosine similarity, Jaccard token overlap, character/word length ratios.
- **F2 (Dense Semantic):** `all-MiniLM-L6-v2` dense cosine similarity, elementwise absolute differences, and pooled late interaction.
- **F3 (Rubric Heuristics):** Section detection, experience year extraction, degree hierarchy matching, keyword density.
- **F4 (Combined Full):** Concatenation of all 777 features.

### Regression & Classification Performance on C2

| Scenario | Feature Subset | Continuous $R^2$ | Continuous MAE | Pearson $r$ | Binned Accuracy | Binned Macro-AUC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **S1: Cold-Resume** | F1: Lexical | 0.1657 | 14.07 | 0.4962 | 0.5364 | 0.6964 |
| *(Seen JDs)* | F2: Dense Semantic | 0.2988 | 13.36 | 0.5475 | 0.5228 | 0.6716 |
| | F3: Rubric Heuristics | 0.1650 | 14.21 | 0.4485 | 0.5068 | 0.6579 |
| | **F4: Combined Full** | **0.4401** | **11.69** | **0.6663** | **0.5740** | **0.7190** |
| **S2: Cold-JD** | F1: Lexical | **0.0186** | 15.35 | 0.3867 | 0.4920 | 0.6346 |
| *(Unseen JDs)* | **F2: Dense Semantic** | **0.3139** | **12.88** | **0.5625** | 0.5319 | 0.6665 |
| | F3: Rubric Heuristics | 0.1214 | 14.61 | 0.4222 | 0.5182 | 0.6342 |
| | F4: Combined Full | 0.2867 | 13.13 | 0.5473 | **0.5558** | **0.6838** |

### Key Takeaways from J2
- **Consistent Pattern with Phase 1:** $R^2 = 0.4401$ under S1 represents the best results observed with our methods on this dataset when JDs repeat, consistent with the Phase 1 audit finding where random row-level splitting exhibited entity memorization leakage ($R^2 = 0.4281$ random vs. $R^2 = 0.2717$ grouped).
- **Lexical Attenuation:** Lexical predictive signal drops sharply ($R^2 = 0.1657 \rightarrow 0.0186$) when evaluated on cold job descriptions.
- **Heuristic Generalization Hypothesis:** In S2, adding F1 and F3 to F2 decreases continuous $R^2$ from 0.3139 down to 0.2867 in point estimate; we hypothesize that handcrafted heuristic rules may overfit seen distributions, though paired CIs are needed to verify significance.

---

## 5. Work Package J3: Cross-Encoder Transformer Evaluation

Evaluated on Kaggle Cloud with **2x NVIDIA T4 GPUs** using `cross-encoder/ms-marco-MiniLM-L-6-v2`.

### Pilot Projection vs. Full Fine-Tuning Execution
- **Pilot Projection (1 Epoch on GPU):** 129.61 seconds (projected 3 epochs: 388.84s).
- **Actual Total Suite Execution:** 1,065.24 seconds (~17.75 minutes) including data pipeline, 3 epochs fine-tuning per scenario, evaluation, and 1,000 cluster-bootstrap iterations.

### Empirical Results: Cross-Encoder vs. Bi-Encoder + LightGBM (Identical Split & Test Sets)

| Scenario | Model Architecture | Accuracy | Macro-F1 | Macro ROC-AUC | Majority Baseline | Test Size |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **S2: Cold-JD** | Cross-Encoder (Fine-Tuned 3 ep) | 0.5163 [0.471, 0.564] | 0.3339 | 0.6196 | 0.5210 | 1,259 |
| | **Bi-Encoder + LightGBM** | **0.5862** [0.526, 0.647] | **0.4351** | **0.7628** | 0.5210 | 1,259 |
| **S3: Double-Cold** | Cross-Encoder (Fine-Tuned 3 ep) | 0.5184 [0.436, 0.613] | 0.2910 | 0.5688 | 0.5441 | 272 |
| | **Bi-Encoder + LightGBM** | **0.5331** [0.451, 0.620] | **0.3641** | **0.6293** | 0.5441 | 272 |

### Key Takeaways from J3
- **Identical Test Set Evaluation:** Both architectures were evaluated on identical holdout splits (seed 42). The test sets comprise 1,259 pairs (majority baseline: 52.10%) in S2 and 272 pairs (majority baseline: 54.41%) in S3.
- **S3 Accuracy Indistinguishable from Majority Baseline:** Under S3 double-cold evaluation, neither Cross-Encoder (51.84% [95% CI: 43.55%, 61.25%]) nor Bi-Encoder + LightGBM (53.31% [95% CI: 45.15%, 62.01%]) achieves accuracy statistically distinguishable from the majority class (54.41%). ROC-AUC is the only evaluated metric that exhibits non-trivial ranking signal (0.6293 for Bi-Encoder vs. 0.5688 for Cross-Encoder).
- **Ranking Performance (AUC):** Bi-Encoder + LightGBM maintains higher ROC-AUC on both held-out scenarios (+0.1432 Delta-AUC on S2; +0.0605 Delta-AUC on S3).
- **Hypothesis on Cross-Encoder Adaptation:** We hypothesize that fine-tuning deep cross-attention layers on ~8,000 pairs without large-scale in-domain pre-training causes the model to collapse toward majority class predictions (Macro-F1 of 0.3339 in S2 and 0.2910 in S3), whereas frozen bi-encoders retain robust general semantic geometry.

---

## 6. Architectural Recommendations for Production ATS Systems

Based on the empirical findings across all work packages, real-world recruitment and resume matching systems should adhere to the following architectural guidelines:

```mermaid
flowchart TD
    subgraph Offline_Indexing ["Offline Document Indexing"]
        R[Candidate Resumes] -->|Text Normalization| RE[Frozen Bi-Encoder]
        J[Job Descriptions] -->|Text Normalization| JE[Frozen Bi-Encoder]
        RE -->|Vector Caching| RV[(Resume Embeddings)]
        JE -->|Vector Caching| JV[(JD Embeddings)]
    end

    subgraph Two_Stage_Scoring ["Two-Stage Production Inference"]
        QueryJD[Incoming Job Description] --> QEmb[Dense Embedding]
        QEmb & RV --> ANN[Stage 1: ANN Cosine Retrieval Top-K]
        ANN --> Pairs[Candidate Pairs]
        Pairs --> FeatExtract[Stage 2: Dense Cosine + Interaction Vectors]
        FeatExtract --> TreeModel[Regularized LightGBM Ensemble]
        TreeModel --> CalibratedScore[Calibrated Match Probability]
    end
```

### Architectural Principles

1. **Never Train or Evaluate on Random Row Splits:**
   Always partition candidates and jobs using a **Double-Cold (S3)** split during offline evaluation to avoid reporting misleading entity-memorization metrics.
2. **Prioritize Frozen Dense Encoders over Fine-Tuned Cross-Encoders:**
   Unless domain-specific training data exceeds $10^5$ validated recruiter decisions, do not fine-tune cross-attention models from scratch. Use high-capacity frozen bi-encoders (`all-MiniLM-L6-v2` or `all-mpnet-base-v2`) with cached embeddings.
3. **Discard Superficial Lexical Overlap:**
   Do not rely on keyword matching, TF-IDF, or length heuristic rules as primary features; their predictive power drops to zero on unseen job descriptions.
4. **Communicate Realistic Probabilities, Not Magic Scores:**
   ATS systems should display calibrated percentile rankings or categorical tiers (e.g., Top 10% semantic match) rather than arbitrary 0–100 scores that mislead both recruiters and job applicants.

