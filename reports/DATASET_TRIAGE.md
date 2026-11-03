# Dataset Triage & Cross-Candidate Audit Report (WP-D2)

**Author:** Senior Machine Learning Engineer  
**Date:** November 3, 2026  
**Status:** Complete — Awaiting User Decision  
**Code Repository:** [ATS-Score-Prediction](https://github.com/pranta2003/ATS-Score-Prediction)  
**Branches:** `exp/dataset-triage` (audit & baselines), `docs/dataset-triage` (triage report)  

---

## Executive Summary

We conducted an exhaustive empirical triage across six candidate ATS / resume-matching datasets to establish:
1. **Label Provenance:** What does each target actually measure (human evaluation, LLM judge imitation, or arithmetic formula)?
2. **Cold-Start Generalization:** What accuracy is honestly achievable when holding out unseen candidates (`GroupKFold` on `resume_id`) and unseen jobs (`GroupKFold` on `jd_id`)?
3. **The "90% Accuracy" Feasibility Test:** Can any available open dataset support 90% accuracy under an honest evaluation protocol without data leakage or trivial class collapse?

### Master Triage Summary Table

| Dataset | Total Rows | # Unique Resumes | # Unique JDs | Label Provenance | License | Cold-Start Accuracy (CI 95%) | Macro-F1 | Macro-AUC | Majority Baseline | Warm Accuracy (Random Split) | Excluded Leakage Columns | Empirical Verdict |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Control** (`0xnbk/resume-ats-score-v1-en`) | 6,374 | 682 | 320 | **HUMAN-LABELED** (Corrupted) | Unknown | 44.98% [42.77%, 47.05%] | 40.06% | N/A (reg) | 50.30% | 48.02% | `original_label` ($\eta^2=0.987$) | **REJECT:** Human data corrupted by Gaussian noise. Models perform below majority class. |
| **C1** (`harsh72877287/job_resume_fit`) | 2,384 | 2,382 | 23 | **LLM-DERIVED** | Other | 69.71% [67.85%, 71.52%] *(Cold-JD: 46.43%)* | 66.98% | N/A (reg) | 49.24% | 70.68% | `skill_string_match_score`, `fuzzy_match_score`, `ai_matched_skills`, `category` | **REJECT:** Only 23 JDs. Memorizes templates. Generalization collapses to 46.4% on unseen JDs. |
| **C2** (`netsol/resume-score-details`) | 1,020 | 236 | 89 | **LLM-DERIVED** (GPT-4o) | CC / Unknown | 54.56% [50.39%, 58.19%] | 43.75% | N/A (reg) | 52.73% | 64.58% | `micro_scores`, `requirements`, `valid_resume_and_jd`, `justification` | **HONEST LLM BENCHMARK:** Rich GPT-4o rubric, but honest cold accuracy is ~55-59% ($R^2 \approx 0.29\text{--}0.34$). |
| **C3** (`rahulrajpu676/job-resume-matching`) | 2,500 (sampled) | 2,138 | 560 | **FORMULA-DERIVED** | MIT | 99.88% [99.72%, 100.0%] *(Majority match)* | 49.97% | N/A (reg) | 99.88% | 99.88% | `semantic_score`, `skill_score`, `position_score`, `category_score`, `experience_score`, etc. | **REJECT (ILLUSORY):** 99.88% of scores fall in one pre-registered bin (`40-70`). Regression $R^2=0.3986$. |
| **C4** (`surendra365/recruitment-dataset`) | 2,500 (sampled) | 2,500 | 51 | **FORMULA-DERIVED** | DbCL-1.0 | 52.28% [50.36%, 54.24%] | 52.28% | 0.5337 | 50.64% | 52.44% | `Job Applicant Name`, `Age`, `Gender`, `Race`, `Ethnicity`, `Job Roles` | **REJECT:** Pure demographic synthetic table. Text has zero signal (AUC = 0.53; random coin flip). |
| **C5** (`cnamuangtoun/resume-job-description-fit`) | 8,000 | 643 | 351 | **HUMAN-LABELED** (Original) | Unknown | 65.66% [63.64%, 67.95%] | 62.76% | 0.8297 | 50.00% | 71.61% | None | **RECOMMENDED PRIMARY BENCHMARK:** Uncorrupted human recruiter labels. Solid AUC (0.83) and 66% Cold Acc. |

*Notes:*
- *Cold-Start Accuracy reports MiniLM chunked embeddings + interaction features + LightGBM under `GroupKFold(n_splits=5, groups=resume_id)`. 95% confidence intervals are estimated via 500-iteration cluster bootstrap.*
- *Continuous targets (Control, C1, C2, C3) were converted to discrete classes using the pre-registered bins: `<40`, `40-70`, `>70` before running any baseline models.*
- *"WARM: not a generalization claim" indicates standard randomized `KFold`, which inflates scores by leaking identical resumes/JDs across train and validation.*

---

## 1. Candidate Identity & Dataset Verification

Before running audits, candidate hashes and remote repository contents were verified to eliminate duplicate investigations:
1. **C1 Identity Check:** `harsh72877287/job_resume_fit` is **100% byte-identical** to `batuhanmtl/job_resume_fit` (exact SHA-256 match across all 2,385 JSON lines).
2. **C2 Identity Check:** `netsol/resume-score-details` is **identical** to forks hosted by `saryal01`, `callmeech`, and `saptarshideveloper` (1,031 GPT-4o evaluation records).
3. **Cross-Dataset Entity Overlap:**
   Cross-dataset text hash comparison (`resume_id` and `jd_id` via SHA1 of normalized text) revealed:
   - `control vs c5`: **7 identical JDs** (`0xnbk` derived directly from `cnamuangtoun`). Zero exact resume hash matches due to separator splitting and whitespace artifacts in `control`.
   - **Zero cross-dataset entity overlap** across all other candidate pairs (`control`, `c1`, `c2`, `c3`, `c4`, `c5`). All candidate datasets represent completely distinct underlying corpuses.

---

## 2. In-Depth Per-Dataset Audits & Baselines

### Dataset Control: `0xnbk/resume-ats-score-v1-en`
* **Source & License:** Hugging Face `0xnbk/resume-ats-score-v1-en` (Unknown / Research License). Derived from `cnamuangtoun` with resumes and job descriptions joined by `" SEP "`.
* **Entities & Schema:**
  - Rows: 6,374 | Unique Resumes: 682 | Unique JDs: 320 | Unique Pairs: 6,274 | Exact Duplicate Rows: 100
  - Text Lengths: Resume mean 5,717.5 characters (721.0 words; max 14,972 chars); JD mean 2,729.7 characters (374.3 words; max 11,469 chars).
  - MiniLM Token Truncation: **100.0% of resumes exceed 256 tokens**; **92.6% exceed 512 tokens**. For JDs, 71.8% exceed 256 tokens and 45.6% exceed 512 tokens.
* **Target Distribution & Pre-registered Bins:**
  - Continuous `ats_score`: Mean 47.23, std 24.97, min 19.16, max 90.05.
  - Pre-registered bins: `<40`: 3,206 (50.3%), `40-70`: 1,592 (25.0%), `>70`: 1,576 (24.7%).
* **Label Provenance Test:**
  - Correlations with lexical match features: TF-IDF cosine ($r=0.242$, $\rho=0.251$), keyword overlap ($r=0.093$, $\rho=0.100$), fuzzy token match ($r=0.091$, $\rho=0.094$), embedding cosine ($r=0.236$, $\rho=0.225$).
  - LightGBM on lexical features alone yielded $R^2 = 0.0796$.
  - **Provenance Verdict:** **HUMAN-LABELED (Corrupted)**. The ground-truth categorical fit from recruitment scraping was projected into arbitrary clusters and smoothed with synthetic Gaussian noise.
* **Leakage Excluded:** `original_label` ($\eta^2 = 0.987$).
* **Empirical Baselines:**
  - **Cold-Start Resume (GroupKFold on `resume_id`):**
    - TF-IDF + LGBM: Accuracy = 45.04% [95% CI: 43.13%, 46.49%], Macro-F1 = 39.26% [37.39%, 41.04%], Balanced Acc = 44.10%, $R^2 = 0.2911$, MAE = 17.85, Spearman $\rho = 0.4821$, Majority Baseline = 50.30%.
    - MiniLM + LGBM: Accuracy = 44.98% [95% CI: 42.77%, 47.05%], Macro-F1 = 40.06% [37.46%, 42.29%], Balanced Acc = 44.33%, $R^2 = 0.2850$, MAE = 17.88, Spearman $\rho = 0.4841$, Majority Baseline = 50.30%.
  - **Warm-Start Random Split (KFold):**
    - TF-IDF + LGBM: Accuracy = 49.11%, Macro-F1 = 44.39%, $R^2 = 0.3720$, MAE = 16.69.
    - MiniLM + LGBM: Accuracy = 48.02%, Macro-F1 = 43.31%, $R^2 = 0.3542$, MAE = 16.84.
* **Finding:** Both cold and warm models fail to exceed the 50.3% majority baseline. The synthetic noise corrupts the boundaries, ensuring high error rates.

---

### Candidate C1: `harsh72877287/job_resume_fit`
* **Source & License:** Hugging Face `harsh72877287/job_resume_fit` (License: Other). Contains 2,385 resume-job pairs across 23 job categories.
* **Entities & Schema:**
  - Rows: 2,384 | Unique Resumes: 2,382 | **Unique JDs: 23** | Unique Pairs: 2,382 | Exact Duplicate Rows: 2
  - Text Lengths: Resume mean 6,285.7 chars (811.6 words); JD mean 4,805.3 chars (599.1 words).
  - MiniLM Token Truncation: 99.2% of resumes exceed 256 tokens; 90.4% exceed 512. **100.0% of JDs exceed 512 tokens**.
* **Target Distribution & Pre-registered Bins:**
  - Continuous `ai_job_resume_fit`: Mean 44.05, std 24.62, min 0.0, max 100.0.
  - Pre-registered bins: `<40`: 1,174 (49.2%), `40-70`: 745 (31.3%), `>70`: 465 (19.5%). Median binary split threshold: 40.65.
* **Label Provenance Test:**
  - Correlations: TF-IDF cosine ($r=0.673$, $\rho=0.690$), keyword overlap ($r=0.596$, $\rho=0.601$), fuzzy match ($r=0.273$, $\rho=0.262$), embedding cosine ($r=0.582$, $\rho=0.579$).
  - LightGBM on lexical features alone yielded $R^2 = 0.5896$.
  - **Provenance Verdict:** **LLM-DERIVED**. Generated by an LLM prompt scoring skill overlap.
* **Leakage Excluded:** `skill_string_match_score`, `fuzzy_match_score`, `ai_matched_skills`, `category`.
* **Empirical Baselines:**
  - **Cold-Start Resume (GroupKFold on `resume_id`):**
    - TF-IDF + LGBM: Accuracy = 73.03% [95% CI: 71.22%, 75.00%], Macro-F1 = 70.82% [68.99%, 72.72%], Balanced Acc = 69.29%, $R^2 = 0.7078$, MAE = 10.36, Spearman $\rho = 0.8354$, Majority Baseline = 49.24%.
    - MiniLM + LGBM: Accuracy = 69.71% [95% CI: 67.85%, 71.52%], Macro-F1 = 66.98% [65.02%, 68.80%], Balanced Acc = 65.69%, $R^2 = 0.6652$, MAE = 11.10, Spearman $\rho = 0.8117$, Majority Baseline = 49.24%.
  - **Cold-Start Job Description (GroupKFold on `jd_id` — Critical Failure Test):**
    - TF-IDF + LGBM: Accuracy = 54.70% [95% CI: 46.17%, 62.99%], Macro-F1 = 38.92% [32.90%, 42.63%], $R^2 = 0.1970$, MAE = 17.75.
    - MiniLM + LGBM: **Accuracy collapses to 46.43%** [95% CI: 36.35%, 56.55%], Macro-F1 = 34.97% [28.47%, 40.04%], **$R^2 = -0.0142$**, MAE = 20.28, Majority Baseline = 49.24%.
  - **Warm-Start Random Split:**
    - TF-IDF: Accuracy = 72.36%, Macro-F1 = 70.41%, $R^2 = 0.7020$.
    - MiniLM: Accuracy = 70.68%, Macro-F1 = 68.12%, $R^2 = 0.6563$.
* **Fatal Defect:** Because there are only 23 distinct job descriptions, any model evaluated by holding out resumes has already trained on the exact same 23 JDs. When evaluated on *unseen job descriptions* (Cold JD), performance completely collapses ($R^2 < 0$, accuracy drops below majority). It cannot generalize to unseen jobs.

---

### Candidate C2: `netsol/resume-score-details`
* **Source & License:** Hugging Face `netsol/resume-score-details` (License: CC / Unknown). 1,031 samples of resumes and job descriptions generated and assessed using GPT-4o.
* **Entities & Schema:**
  - Valid Rows: 1,020 (878 with complete numeric `macro_score`) | Unique Resumes: 236 | Unique JDs: 89 | Unique Pairs: 770 | Exact Duplicate Rows: 250
  - Text Lengths: Resume mean 4,169.2 chars (550.3 words; max 36,942 chars); JD mean 1,826.7 chars (250.7 words).
  - MiniLM Token Truncation: 95.4% of resumes exceed 256 tokens; 60.2% exceed 512 tokens. For JDs, 81.8% exceed 256 tokens and 3.6% exceed 512 tokens.
* **Target Distribution & Pre-registered Bins:**
  - Continuous `macro_score`: Mean 57.61, std 19.68, min 0.0, max 95.5.
  - Pre-registered bins: `<40`: 162 (18.5%), `40-70`: 463 (52.7%), `>70`: 253 (28.8%). Median binary split: 60.0.
* **Label Provenance Test:**
  - Correlations: TF-IDF cosine ($r=0.133$, $\rho=0.399$), keyword overlap ($r=0.184$, $\rho=0.402$), fuzzy match ($r=0.066$, $\rho=0.095$), embedding cosine ($r=0.187$, $\rho=0.263$).
  - LightGBM on lexical features alone yielded $R^2 = 0.3938$.
  - **Provenance Verdict:** **LLM-DERIVED**. Generated and rubric-assessed directly by GPT-4o.
* **Leakage Excluded:** `micro_scores`, `requirements`, `valid_resume_and_jd`, `justification`.
* **Empirical Baselines:**
  - **Cold-Start Resume (GroupKFold on `resume_id`):**
    - TF-IDF + LGBM: Accuracy = 59.11% [95% CI: 54.83%, 63.30%], Macro-F1 = 51.62% [47.34%, 55.81%], Balanced Acc = 50.17%, $R^2 = 0.3432$, MAE = 12.85, Spearman $\rho = 0.5629$, Majority Baseline = 52.73%.
    - MiniLM + LGBM: Accuracy = 54.56% [95% CI: 50.39%, 58.19%], Macro-F1 = 43.75% [39.04%, 47.77%], Balanced Acc = 43.86%, $R^2 = 0.2896$, MAE = 13.44, Spearman $\rho = 0.4902$, Majority Baseline = 52.73%.
  - **Cold-Start Job Description (GroupKFold on `jd_id`):**
    - TF-IDF + LGBM: Accuracy = 60.71% [95% CI: 53.79%, 66.15%], Macro-F1 = 49.74% [42.13%, 56.28%], $R^2 = 0.2653$, MAE = 13.03.
    - MiniLM + LGBM: Accuracy = 56.95% [95% CI: 50.59%, 62.17%], Macro-F1 = 47.19% [40.14%, 52.87%], $R^2 = 0.2760$, MAE = 13.06.
  - **Warm-Start Random Split:**
    - TF-IDF: Accuracy = 67.77%, Macro-F1 = 64.78%, $R^2 = 0.5902$, MAE = 9.70.
    - MiniLM: Accuracy = 64.58%, Macro-F1 = 60.18%, $R^2 = 0.5695$, MAE = 10.21.
* **Assessment:** C2 represents high-quality structured GPT-4o evaluations with reasoning. However, standard feature baselines achieve only 55-59% cold accuracy and $R^2 \approx 0.29\text{--}0.34$. It cannot support a 90% accuracy claim.

---

### Candidate C3: `rahulrajpu676/job-resume-matching`
* **Source & License:** Hugging Face `rahulrajpu676/job-resume-matching` (License: MIT). Multi-table dataset (`CVs.csv`, `Job_Description.csv`, `matches.csv`).
* **Entities & Schema:**
  - Sample Evaluated: 2,500 paired matches | Unique Resumes: 2,138 | Unique JDs: 560 | Pairs: 2,500 | Exact Duplicates: 0
  - Text Lengths: Resume mean 1,492.2 chars (175.1 words); JD mean 3,030.2 chars (408.5 words).
  - MiniLM Token Truncation: 65.4% of resumes exceed 256 tokens; 2.6% exceed 512 tokens. 82.8% of JDs exceed 256 tokens; 46.0% exceed 512 tokens.
* **Target Distribution & Pre-registered Bins:**
  - Continuous `final_score`: Mean 58.47, std 3.54, min 43.47, max 72.78.
  - Pre-registered bins: `<40`: 0 (0.0%), **`40-70`: 2,497 (99.88%)**, `>70`: 3 (0.12%).
* **Label Provenance Test:**
  - Correlations: TF-IDF cosine ($r=0.180$, $\rho=0.243$), keyword overlap ($r=0.123$, $\rho=0.132$), fuzzy match ($r=-0.071$), embedding cosine ($r=0.195$, $\rho=0.212$).
  - LightGBM on lexical features alone yielded $R^2 = 0.0573$.
  - **Provenance Verdict:** **FORMULA-DERIVED**. Computed as an explicit linear weighting of synthetic sub-scores: $0.35 \times \text{semantic} + 0.25 \times \text{skill} + 0.15 \times \text{position} + 0.10 \times \text{category} + 0.10 \times \text{experience} + 0.05 \times \text{education}$.
* **Leakage Excluded:** All sub-score components (`semantic_score`, `skill_score`, `position_score`, `category_score`, `experience_score`, `education_score`, `raw_score`).
* **Empirical Baselines:**
  - **Cold-Start Resume (GroupKFold on `resume_id`):**
    - TF-IDF + LGBM: **Accuracy = 99.88%** [95% CI: 99.72%, 100.0%], Macro-F1 = 49.97%, Balanced Acc = 50.00%, Majority Baseline = 99.88%, $R^2 = 0.4388$, MAE = 1.90, Spearman $\rho = 0.6686$.
    - MiniLM + LGBM: **Accuracy = 99.88%** [95% CI: 99.72%, 100.0%], Macro-F1 = 49.97%, Balanced Acc = 50.00%, Majority Baseline = 99.88%, $R^2 = 0.3986$, MAE = 1.97, Spearman $\rho = 0.6475$.
  - **Warm-Start Random Split:**
    - TF-IDF: Accuracy = 99.88%, Macro-F1 = 49.97%, $R^2 = 0.4423$, MAE = 1.90.
    - MiniLM: Accuracy = 99.88%, Macro-F1 = 49.97%, $R^2 = 0.3995$, MAE = 1.96.
* **Why 99.88% Accuracy is an Illusion:** The 99.88% accuracy is purely due to extreme class collapse under pre-registered bins: 2,497 out of 2,500 samples fall into the `40-70` bin. A degenerate model predicting constant `40-70` achieves 99.88% accuracy with Macro-F1 of 49.97% and Balanced Accuracy of 50.0%. In reality, continuous regression achieves only $R^2 = 0.3986$.

---

### Candidate C4: `surendra365/recruitment-dataset`
* **Source & License:** Kaggle `surendra365/recruitement-dataset` (License: DbCL-1.0). 10,000 synthetic records containing applicant demographics, resume snippets, job roles, and a binary target (`Best Match`).
* **Entities & Schema:**
  - Sample Evaluated: 2,500 records | Unique Resumes: 2,500 | Unique JDs: 51 | Pairs: 2,500 | Exact Duplicates: 0
  - Text Lengths: Resume mean 294.1 chars (37.8 words); JD mean 645.0 chars (90.8 words) — short templated boilerplate.
  - MiniLM Token Truncation: 0.0% of resumes or JDs exceed 256 tokens.
* **Target Distribution & Pre-registered Bins:**
  - Binary target `Best Match`: Class 0: 1,266 (50.64%), Class 1: 1,234 (49.36%). Majority baseline = 50.64%.
* **Label Provenance Test:**
  - Lexical-only Accuracy: 49.04%.
  - **Provenance Verdict:** **FORMULA-DERIVED**. Synthetic table created for algorithmic bias exercises with demographic features.
* **Leakage Excluded:** `Job Applicant Name`, `Age`, `Gender`, `Race`, `Ethnicity`, `Job Roles`.
* **Empirical Baselines:**
  - **Cold-Start Resume (GroupKFold on `resume_id`):**
    - TF-IDF + LGBM: Accuracy = 51.60% [95% CI: 49.88%, 53.72%], Macro-F1 = 51.59%, Balanced Acc = 51.62%, Macro ROC-AUC = 0.5325, Majority Baseline = 50.64%.
    - MiniLM + LGBM: Accuracy = 52.28% [95% CI: 50.36%, 54.24%], Macro-F1 = 52.28%, Balanced Acc = 52.28%, Macro ROC-AUC = 0.5337, Majority Baseline = 50.64%.
  - **Cold-Start Job Description (GroupKFold on `jd_id`):**
    - TF-IDF + LGBM: Accuracy = 47.00% [95% CI: 44.70%, 49.32%], Macro-F1 = 47.00%, Macro ROC-AUC = 0.4664.
    - MiniLM + LGBM: Accuracy = 51.52% [95% CI: 49.19%, 54.29%], Macro-F1 = 51.05%, Macro ROC-AUC = 0.5198.
  - **Warm-Start Random Split:**
    - TF-IDF: Accuracy = 51.76%, Macro-F1 = 51.76%, ROC-AUC = 0.5360.
    - MiniLM: Accuracy = 52.44%, Macro-F1 = 52.43%, ROC-AUC = 0.5366.
* **Finding:** When demographic leakage columns are stripped, resume and JD text have essentially zero predictive power (ROC-AUC = 0.53; accuracy indistinguishable from random chance).

---

### Candidate C5: `cnamuangtoun/resume-job-description-fit`
* **Source & License:** Hugging Face `cnamuangtoun/resume-job-description-fit` (License: Unknown / Web Scraped). The original source dataset from which `0xnbk` was derived.
* **Entities & Schema:**
  - Rows: 8,000 | Unique Resumes: 643 | Unique JDs: 351 | Unique Pairs: 7,993 | Exact Duplicate Rows: 7
  - Text Lengths: Resume mean 5,773.4 chars (704.4 words; max 25,364 chars); JD mean 2,776.8 chars (378.4 words; max 8,171 chars).
  - MiniLM Token Truncation: 100.0% of resumes exceed 256 tokens; 96.4% exceed 512 tokens. For JDs, 69.0% exceed 256 tokens and 41.8% exceed 512 tokens.
* **Target Distribution:**
  - Categorical 3-class target: `No Fit`: 4,000 (50.00%), `Potential Fit`: 2,000 (25.00%), `Good Fit`: 2,000 (25.00%). Majority baseline = 50.00%.
* **Label Provenance Test:**
  - Lexical-only Accuracy: 51.71%.
  - **Provenance Verdict:** **HUMAN-LABELED**. Authentic human recruiter fit judgments collected from job board applicant matching.
* **Leakage Excluded:** None (clean 2-column text input).
* **Empirical Baselines:**
  - **Cold-Start Resume (GroupKFold on `resume_id`):**
    - TF-IDF + LGBM: Accuracy = 66.18% [95% CI: 64.06%, 68.46%], Macro-F1 = 63.21% [60.39%, 65.91%], Balanced Acc = 62.16%, **Macro ROC-AUC = 0.8291**, Majority Baseline = 50.00%.
    - MiniLM + LGBM: Accuracy = 65.66% [95% CI: 63.64%, 67.95%], Macro-F1 = 62.76% [60.27%, 65.48%], Balanced Acc = 62.23%, **Macro ROC-AUC = 0.8297**, Majority Baseline = 50.00%.
  - **Warm-Start Random Split (KFold):**
    - TF-IDF + LGBM: Accuracy = 72.11% [95% CI: 70.59%, 73.69%], Macro-F1 = 69.60%, Macro ROC-AUC = 0.8687.
    - MiniLM + LGBM: Accuracy = 71.61% [95% CI: 69.65%, 73.45%], Macro-F1 = 68.99%, Macro ROC-AUC = 0.8679.
* **Assessment:** C5 is the cleanest ground truth human dataset available. Removing the artificial Gaussian noise from `0xnbk` immediately recovers a competitive baseline: Cold-Start Accuracy of 66.2% (vs 50% majority baseline) and a Macro ROC-AUC of 0.83.

---

## 3. Plain-Language Analysis: The "90% Accuracy" Question

### Does Any Dataset Honestly Exceed 90% Cold-Start Accuracy?
**No.** No dataset evaluated achieves an honest 90% accuracy on unseen candidates and unseen job descriptions for real-world candidate-job suitability:
1. **The C3 "99.88%" is a Mathematical Artifact:** In C3, 99.88% of samples sit within the single pre-registered range of 40-70. An empty majority-class classifier obtains 99.88% accuracy while failing completely on minority classes (Macro-F1 = 49.97%). Its continuous regression $R^2$ is only 0.3986.
2. **The C1 "73%" is Template Memorization:** C1 appears to reach 73% accuracy under resume splitting only because it evaluates across a tiny set of **23 fixed job descriptions**. When evaluated under a proper cold-job protocol (`GroupKFold` on `jd_id`), its $R^2$ plunges from +0.665 to **-0.014**, and accuracy falls to 46.4% (worse than random guessing).
3. **The Human Ceilings (C5 & Control) Peak at 66% Cold Accuracy (AUC 0.83):** In real-world human recruitment annotations, recruiter disagreement and subjective criteria establish an empirical Bayes error rate of approximately 25-35%. A cold-start macro ROC-AUC of 0.83 is a solid, honest baseline.

### What Does Each Label Actually Measure?
- **Control (`0xnbk`):** Measures the ability to fit a 3-component Gaussian mixture with artificial noise.
- **C1 (`harsh72877287`):** Measures keyword and embedding similarity across 23 static job description prompts.
- **C2 (`netsol`):** Measures the ability to imitate GPT-4o's internal rubric across structured evaluation criteria.
- **C3 (`rahulrajpu`):** Measures the ability to reconstruct an explicit 6-part weighted arithmetic equation.
- **C4 (`surendra365`):** Measures random noise once demographic variables are removed.
- **C5 (`cnamuangtoun`):** Measures actual human recruiter fit decisions (No Fit / Potential Fit / Good Fit).

### Honest Framing Recommendations for Documentation / Reports

| Dataset | Unacceptable / Misleading Claim | Honest, Rigorous Claim |
| :--- | :--- | :--- |
| **C5 (`cnamuangtoun`)** | *"Achieves 90% accuracy in hiring selection."* | *"Achieves 66.2% cold-start accuracy (AUC 0.83, macro-F1 63.2%) at predicting authentic human recruiter candidate screening decisions on unseen candidates."* |
| **C2 (`netsol`)** | *"Accurately determines whether a candidate is qualified for a job."* | *"Achieves 55-59% cold-start accuracy ($R^2 \approx 0.34$) at predicting GPT-4o multi-criteria candidate evaluation scores without querying the LLM."* |
| **C1 (`harsh72877287`)** | *"Predicts ATS scores with 73% accuracy."* | *"Exhibits severe template overfitting: reaches 70% accuracy on 23 fixed job descriptions, but collapses to 46.4% ($R^2 < 0$) on unseen job postings."* |
| **C3 (`rahulrajpu676`)** | *"Achieves 99.9% matching precision."* | *"Exhibits extreme target clustering (99.88% in a single bin); regression models achieve an $R^2$ of 0.40 against the synthetic weighted formula."* |

---

## 4. Ranked Technical Recommendation

### Rank 1: Candidate C5 (`cnamuangtoun/resume-job-description-fit`) — PRIMARY RECOMMENDATION
* **Reasoning:** C5 represents uncorrupted, real-world human recruiting decisions. It is the exact parent corpus of our Control dataset, but without the synthetic Gaussian noise injection that destroyed the signal in `0xnbk`. It provides 8,000 balanced pairs, realistic text distributions, authentic recruiter noise, and solid baseline discriminative power (Cold-Start Macro ROC-AUC = 0.83, Cold Accuracy = 66.2% vs 50.0% majority baseline).
* **Next Steps if Chosen:** Train chunked sentence-transformer cross-encoders or multi-task interaction models to establish an honest benchmark in the 70-75% accuracy range (AUC 0.86-0.88).

### Rank 2: Candidate C2 (`netsol/resume-score-details`) — SECONDARY BENCHMARK (LLM Distillation)
* **Reasoning:** If the project goal is specifically to build a fast, local ML model that mimics advanced LLM recruiter reasoning (GPT-4o), C2 offers 1,020 rich records with continuous scores, qualitative justifications, and multi-criteria rubrics.
* **Caveat:** Sample size is modest (1,020 rows), and cold generalization with standard tree baselines is ~55-59% ($R^2 \approx 0.29\text{--}0.34$).

### Rank 3: REJECT Candidates C1, C3, C4, and Control
* **C1:** Structurally flawed (only 23 JDs; memorizes job prompts and fails completely on new jobs).
* **C3:** Pathological target clustering (99.88% in one bin); synthetic formula with no genuine recruiting signal.
* **C4:** Demographic toy dataset with zero text predictive signal (AUC = 0.53).
* **Control (`0xnbk`):** Artificially corrupted variant of C5; inferior to C5 in every dimension.

---

> [!IMPORTANT]
> **Decision Gate:** In accordance with prompt rules, execution is stopped here. Please review the empirical triage results and indicate which dataset should be selected as the foundation for the final model.
