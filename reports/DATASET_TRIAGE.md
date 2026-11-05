# Dataset Triage & Cross-Candidate Audit Report (WP-D2)

**Author:** Senior Machine Learning Engineer  
**Date:** November 5, 2026 (Updated with Reconciled Control Direct Classification & Verification)  
**Status:** Complete — Decision Implemented (C5 Primary Benchmark, Control Labeled Comparison)  
**Code Repository:** [ATS-Score-Prediction](https://github.com/pranta2003/ATS-Score-Prediction)  
**Branches:** `exp/dataset-triage` (audit & baselines), `docs/dataset-triage` (triage report)  

---

## Executive Summary

We conducted an empirical triage across six candidate ATS / resume-matching datasets to establish:
1. **Label Provenance:** What does each target actually measure (human evaluation, LLM judge imitation, or arithmetic formula)?
2. **Cold-Start Generalization:** What accuracy is honestly achievable when holding out unseen candidates (`GroupKFold` on `resume_id`) and unseen jobs (`GroupKFold` on `jd_id`)?
3. **The "90% Accuracy" Feasibility Test:** Can any available open dataset support 90% accuracy under an honest evaluation protocol without data leakage or trivial class collapse?

### Master Triage Summary Table

| Dataset | Total Rows | # Unique Resumes | # Unique JDs | Label Provenance | License | Cold-Start Accuracy (CI 95%) | Macro-F1 | Macro-AUC | Majority Baseline | Warm Accuracy (Random Split) | Excluded Leakage Columns | Empirical Verdict |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Control** (`0xnbk/resume-ats-score-v1-en`) — *Direct 3-Class* | 6,374 | 682 | 320 | **HUMAN-LABELED** (Provenance not documented) | Unknown | **65.81%** [63.36%, 68.27%] *(TF-IDF: 66.77%)* | **62.78%** *(TF-IDF: 64.05%)* | **0.8300** *(TF-IDF: 0.8339)* | 50.30% | 71.45% *(TF-IDF: 72.45%)* | `original_label` (when evaluating continuous score) | **LABELED COMPARISON:** Direct 3-class classification matches C5 (65.8% acc, AUC 0.83). Class labels intact. |
| **Control** (`0xnbk/resume-ats-score-v1-en`) — *Regression Binned* | 6,374 | 682 | 320 | **HUMAN-LABELED** (Provenance not documented) | Unknown | **44.98%** [42.77%, 47.05%] *(TF-IDF: 45.04%)* | **40.06%** *(TF-IDF: 39.26%)* | N/A (reg) | 50.30% | 48.02% *(TF-IDF: 49.11%)* | `original_label` ($\eta^2=0.987$) | Continuous regression then binning underperforms (45.0%) due to boundary scattering. |
| **C1** (`harsh72877287/job_resume_fit`) | 2,384 | 2,382 | 23 | **LLM-DERIVED** | Other | 69.71% [67.85%, 71.52%] *(Cold-JD: 46.43%)* | 66.98% | N/A (reg) | 49.24% | 70.68% | `skill_string_match_score`, `fuzzy_match_score`, `ai_matched_skills`, `category` | **REJECT:** Only 23 JDs. Memorizes templates. Generalization collapses to 46.4% on unseen JDs. |
| **C2** (`netsol/resume-score-details`) | 1,020 | 236 | 89 | **LLM-DERIVED** (GPT-4o) | CC / Unknown | 54.56% [50.39%, 58.19%] | 43.75% | N/A (reg) | 52.73% | 64.58% | `micro_scores`, `requirements`, `valid_resume_and_jd`, `justification` | **HONEST LLM BENCHMARK:** Rich GPT-4o rubric, but honest cold accuracy is ~55-59% ($R^2 \approx 0.29\text{--}0.34$). |
| **C3** (`rahulrajpu676/job-resume-matching`) | 2,500 (sampled) | 2,138 | 560 | **FORMULA-DERIVED** | MIT | 99.88% [99.72%, 100.0%] *(Majority match)* | 49.97% | N/A (reg) | 99.88% | 99.88% | `semantic_score`, `skill_score`, `position_score`, `category_score`, `experience_score`, etc. | **REJECT (ILLUSORY):** 99.88% of scores fall in one pre-registered bin (`40-70`). Regression $R^2=0.3986$. |
| **C4** (`surendra365/recruitment-dataset`) | 2,500 (sampled) | 2,500 | 51 | **FORMULA-DERIVED** | DbCL-1.0 | 52.28% [50.36%, 54.24%] | 52.28% | 0.5337 | 50.64% | 52.44% | `Job Applicant Name`, `Age`, `Gender`, `Race`, `Ethnicity`, `Job Roles` | **REJECT:** Pure demographic synthetic table. Text has zero signal (AUC = 0.53; random coin flip). |
| **C5** (`cnamuangtoun/resume-job-description-fit`) | 8,000 | 643 | 351 | **HUMAN-LABELED** (Provenance not documented) | Unknown | **65.66%** [63.64%, 67.95%] *(TF-IDF: 66.18%)* | **62.76%** *(TF-IDF: 63.21%)* | **0.8297** *(TF-IDF: 0.8291)* | 50.00% | 71.61% *(TF-IDF: 72.11%)* | None | **PRIMARY BENCHMARK:** Clean multi-class setup without separator artifacts. AUC 0.83 and 65.7% Cold Acc. |

*Notes:*
- *Cold-Start Accuracy reports MiniLM chunked embeddings + interaction features + LightGBM under `GroupKFold(n_splits=5, groups=resume_id)`. 95% confidence intervals are estimated via 500-iteration cluster bootstrap.*
- *Continuous targets (Control regression, C1, C2, C3) were converted to discrete classes using pre-registered bins: `<40`, `40-70`, `>70` before running models.*
- *"WARM: not a generalization claim" indicates standard randomized `KFold`, which inflates scores by leaking identical resumes/JDs across train and validation.*

---

## 1. Candidate Identity, Entity Reconciliation & Verification

### Entity Reconciliation: Phase 1 (778) vs. Audit (682) vs. C5 (643)
An explicit reconciliation of unique resume counts across the project phases was conducted:
1. **Phase 1 Count (778 Unique Resumes):**
   In Phase 1, `train.csv` and `validation.csv` were split on `" SEP "` using raw string slicing with `strip()` only. Resumes with different casing (e.g. uppercase headings vs lowercase) or variations in whitespace/newlines were counted as distinct strings, resulting in 778 unique resumes.
2. **Adapter Normalization (682 Unique Resumes):**
   The universal adapter layer applies deterministic normalization (`normalize_text`: lowercase conversion and collapsing consecutive whitespace/newlines to single spaces). Under this normalization, the 778 strings collapse to **682 unique resumes** (and 320 unique JDs).
3. **C5 Count (643 Unique Resumes):**
   The raw C5 dataset contains 8,000 rows, which resolve to **643 unique normalized resumes** and **351 unique JDs**.
4. **Cross-Dataset Matching between Control and C5:**
   - **Exact Text Hash Overlap:** 0 exact resume hashes match, 7 JD hashes match. Exact hash equality fails because `0xnbk` introduced minor whitespace, formatting, or separator artifacts when concatenating resume and JD text with `" SEP "`.
   - **Fuzzy Text Matching (`rapidfuzz` ratio $\ge 95$):**
     - **Resumes:** **636 / 682 Control resumes (93.26%)** match C5 resumes with $\ge 95\%$ string similarity.
     - **Job Descriptions:** **280 / 320 Control JDs (87.50%)** match C5 JDs with $\ge 95\%$ string similarity.
   This confirms that Control (`0xnbk`) is an export of C5 (`cnamuangtoun`) that underwent text concatenation and minor formatting mutations.

### Other Candidate Identity Checks:
- **C1 Identity Check:** `harsh72877287/job_resume_fit` is **100% byte-identical** to `batuhanmtl/job_resume_fit` (exact SHA-256 match across all 2,385 JSON lines).
- **C2 Identity Check:** `netsol/resume-score-details` is **identical** to forks hosted by `saryal01`, `callmeech`, and `saptarshideveloper` (1,031 GPT-4o evaluation records).
- **Entity Independence:** Aside from Control and C5, all other candidates (`c1`, `c2`, `c3`, `c4`) share zero entities (0 resumes, 0 JDs).

---

## 2. In-Depth Per-Dataset Audits & Baselines

### Dataset Control: `0xnbk/resume-ats-score-v1-en`
* **Source & License:** Hugging Face `0xnbk/resume-ats-score-v1-en` (Unknown / Research License). Derived from `cnamuangtoun` with resumes and job descriptions joined by `" SEP "`.
* **Entities & Schema:**
  - Rows: 6,374 | Unique Resumes: 682 | Unique JDs: 320 | Unique Pairs: 6,274 | Exact Duplicate Rows: 100
  - Text Lengths: Resume mean 5,717.5 characters (721.0 words); JD mean 2,729.7 characters (374.3 words).
  - MiniLM Token Truncation: 100.0% of resumes exceed 256 tokens; 92.6% exceed 512 tokens. For JDs, 71.8% exceed 256 tokens and 45.6% exceed 512 tokens.
* **Target Distribution & Intact Discrete Labels:**
  - Continuous `ats_score`: Mean 47.23, std 24.97, min 19.16, max 90.05.
  - Discrete `original_label`: `No Fit`: 3,206 (50.3%), `Potential Fit`: 1,592 (25.0%), `Good Fit`: 1,576 (24.7%).
  - The discrete class labels are completely intact ($\eta^2 = 0.987$ between `original_label` and `ats_score` with non-overlapping score clusters).
* **Direct Classification vs. Binned Regression Comparison:**
  The discrepancy in prior reports stemmed from methodological differences (Control was previously evaluated via continuous regression then binned into classes, whereas C5 was evaluated as a direct 3-class classifier). When evaluated under the **identical 3-class classification protocol** with `GroupKFold` on `resume_id`:
  - **Direct 3-Class Classification (on `original_label`):**
    - TF-IDF + LGBM: **Accuracy = 66.77%** [95% CI: 64.61%, 69.57%], Macro-F1 = **64.05%** [61.42%, 66.68%], **Macro ROC-AUC = 0.8339**.
    - MiniLM + LGBM: **Accuracy = 65.81%** [95% CI: 63.36%, 68.27%], Macro-F1 = **62.78%** [60.11%, 65.69%], **Macro ROC-AUC = 0.8300**.
    - Warm-Start Random Split: TF-IDF Acc = 72.45% (AUC 0.8730), MiniLM Acc = 71.45% (AUC 0.8670).
  - **Continuous Regression then Binned (<40, 40-70, >70):**
    - TF-IDF + LGBM: Accuracy = 45.04% [43.13%, 46.49%], Macro-F1 = 39.26%, $R^2 = 0.2911$, MAE = 17.85.
    - MiniLM + LGBM: Accuracy = 44.98% [42.77%, 47.05%], Macro-F1 = 40.06%, $R^2 = 0.2850$, MAE = 17.88.
* **Conclusion for Control:** The underlying class signal in Control is completely preserved. The low 45.0% accuracy in the regression pipeline occurred because regression predictions disperse across arbitrary bin cutoffs, whereas direct multiclass training achieves **65.8% to 66.8% accuracy (AUC 0.83)**, matching C5. Control remains a valid labeled comparison.

---

### Candidate C1: `harsh72877287/job_resume_fit`
* **Source & License:** Hugging Face `harsh72877287/job_resume_fit` (License: Other). Contains 2,385 resume-job pairs across 23 job categories.
* **Entities & Schema:**
  - Rows: 2,384 | Unique Resumes: 2,382 | **Unique JDs: 23** | Unique Pairs: 2,382 | Exact Duplicate Rows: 2
  - Text Lengths: Resume mean 6,285.7 chars (811.6 words); JD mean 4,805.3 chars (599.1 words).
* **Label Provenance Test:**
  - Correlations: TF-IDF cosine ($r=0.673$), keyword overlap ($r=0.596$), embedding cosine ($r=0.582$).
  - LightGBM on lexical features alone yielded $R^2 = 0.5896$.
  - **Provenance Verdict:** **LLM-DERIVED**. Generated by an LLM prompt scoring skill overlap.
* **Leakage Excluded:** `skill_string_match_score`, `fuzzy_match_score`, `ai_matched_skills`, `category`.
* **Empirical Baselines:**
  - **Cold-Start Resume (GroupKFold on `resume_id`):** TF-IDF Acc = 73.03%, MiniLM Acc = 69.71% ($R^2 = 0.6652$).
  - **Cold-Start Job Description (GroupKFold on `jd_id` — Critical Failure Test):**
    - TF-IDF + LGBM: Accuracy = 54.70%, Macro-F1 = 38.92%, $R^2 = 0.1970$.
    - MiniLM + LGBM: **Accuracy collapses to 46.43%**, Macro-F1 = 34.97%, **$R^2 = -0.0142$** (worse than mean guessing; majority baseline is 49.24%).
* **Fatal Defect:** Only 23 distinct JDs. The model simply memorizes the 23 job templates; on unseen JDs, performance entirely collapses.

---

### Candidate C2: `netsol/resume-score-details`
* **Source & License:** Hugging Face `netsol/resume-score-details` (License: CC / Unknown). 1,031 samples generated and assessed using GPT-4o.
* **Entities & Schema:**
  - Valid Rows: 1,020 (878 with numeric `macro_score`) | Unique Resumes: 236 | Unique JDs: 89
* **Label Provenance Test:**
  - Lexical $R^2 = 0.3938$. **Provenance Verdict:** **LLM-DERIVED** (GPT-4o rubric assessment).
* **Leakage Excluded:** `micro_scores`, `requirements`, `valid_resume_and_jd`, `justification`.
* **Empirical Baselines:**
  - Cold-Start Resume: TF-IDF Acc = 59.11%, MiniLM Acc = 54.56% ($R^2 = 0.2896$).
  - Cold-Start JD: TF-IDF Acc = 60.71%, MiniLM Acc = 56.95% ($R^2 = 0.2760$).
* **Assessment:** Legitimate LLM distillation benchmark, but cold accuracy is modest (~55-59%, $R^2 \approx 0.29\text{--}0.34$).

---

### Candidate C3: `rahulrajpu676/job-resume-matching`
* **Source & License:** Hugging Face `rahulrajpu676/job-resume-matching` (License: MIT).
* **Target Distribution & Pre-registered Bins:**
  - Continuous `final_score`: Mean 58.47, std 3.54, range [43.47, 72.78].
  - Bins: `<40`: 0, **`40-70`: 2,497 (99.88%)**, `>70`: 3 (0.12%).
* **Label Provenance Test:** **FORMULA-DERIVED** ($0.35 \times \text{semantic} + 0.25 \times \text{skill} + \dots$).
* **Leakage Excluded:** All sub-score components (`semantic_score`, `skill_score`, etc.).
* **Empirical Baselines:**
  - Cold-Start Resume: Accuracy = **99.88%** (matches majority baseline 99.88%; Macro-F1 = 49.97%; $R^2 = 0.3986$).
* **Why 99.88% Accuracy is an Illusion:** Pathological class collapse. 99.88% of samples sit within a single bin. A constant-predicting dummy model achieves 99.88% accuracy. Continuous regression achieves only $R^2 = 0.3986$.

---

### Candidate C4: `surendra365/recruitment-dataset`
* **Source & License:** Kaggle `surendra365/recruitement-dataset` (License: DbCL-1.0). 10,000 synthetic records with demographic features.
* **Target Distribution:** Binary `Best Match` (Class 0: 50.64%, Class 1: 49.36%).
* **Label Provenance:** **FORMULA-DERIVED**.
* **Leakage Excluded:** Demographic columns (`Age`, `Gender`, `Race`, `Ethnicity`, `Job Roles`).
* **Empirical Baselines:**
  - Cold-Start Resume: Acc = 52.28%, Macro ROC-AUC = **0.5337** (indistinguishable from a coin flip). Text has zero predictive signal.

---

### Candidate C5: `cnamuangtoun/resume-job-description-fit`
* **Source & License:** Hugging Face `cnamuangtoun/resume-job-description-fit` ([URL](https://huggingface.co/datasets/cnamuangtoun/resume-job-description-fit)). License: Unknown / Web Scraped.
* **Documentation & Label Provenance:**
  - The repository contains only `.gitattributes`, `test.csv`, and `train.csv`. No README or dataset card exists (HTTP 404).
  - Therefore, the exact annotation procedure is **label provenance not documented**.
* **Entities & Schema:**
  - Rows: 8,000 | Unique Resumes: 643 | Unique JDs: 351 | Unique Pairs: 7,993 | Exact Duplicate Rows: 7
  - Text Lengths: Resume mean 5,773.4 chars (704.4 words); JD mean 2,776.8 chars (378.4 words).
  - MiniLM Token Truncation: 100.0% of resumes exceed 256 tokens; 96.4% exceed 512 tokens. For JDs, 69.0% exceed 256 tokens and 41.8% exceed 512 tokens.
* **Target Distribution:**
  - Categorical 3-class target: `No Fit`: 4,000 (50.00%), `Potential Fit`: 2,000 (25.00%), `Good Fit`: 2,000 (25.00%). Majority baseline = 50.00%.
* **Empirical Baselines:**
  - **Cold-Start Resume (GroupKFold on `resume_id`):**
    - TF-IDF + LGBM: **Accuracy = 66.18%** [95% CI: 64.06%, 68.46%], Macro-F1 = **63.21%** [60.39%, 65.91%], Balanced Acc = 62.16%, **Macro ROC-AUC = 0.8291**, Majority Baseline = 50.00%.
    - MiniLM + LGBM: **Accuracy = 65.66%** [95% CI: 63.64%, 67.95%], Macro-F1 = **62.76%** [60.27%, 65.48%], Balanced Acc = 62.23%, **Macro ROC-AUC = 0.8297**, Majority Baseline = 50.00%.
  - **Warm-Start Random Split (KFold):**
    - TF-IDF + LGBM: Accuracy = 72.11% [95% CI: 70.59%, 73.69%], Macro-F1 = 69.60%, Macro ROC-AUC = 0.8687.
    - MiniLM + LGBM: Accuracy = 71.61% [95% CI: 69.65%, 73.45%], Macro-F1 = 68.99%, Macro ROC-AUC = 0.8679.
* **Assessment:** C5 is our selected primary benchmark: clean multi-class target without text separator artifacts, authentic text distributions, and a solid baseline (Cold Macro ROC-AUC = 0.83, Cold Acc = 65.7% vs 50.0% majority baseline).

---

## 3. Plain-Language Analysis: The "90% Accuracy" Question

### Does Any Dataset Honestly Exceed 90% Cold-Start Accuracy?
**No.** No dataset evaluated achieves an honest 90% accuracy on unseen candidates and unseen job descriptions:
1. **The C3 "99.88%" is a Mathematical Artifact:** In C3, 99.88% of samples sit within the single pre-registered range of 40-70. An empty majority-class classifier obtains 99.88% accuracy while failing completely on minority classes (Macro-F1 = 49.97%). Its continuous regression $R^2$ is only 0.3986.
2. **The C1 "73%" is Template Memorization:** C1 appears to reach 73% accuracy under resume splitting only because it evaluates across a tiny set of **23 fixed job descriptions**. When evaluated under a proper cold-job protocol (`GroupKFold` on `jd_id`), its $R^2$ plunges from +0.665 to **-0.014**, and accuracy falls to 46.4% (worse than random guessing).
3. **The Authentic Ceilings (C5 & Control) Peak at ~66% Cold Accuracy (AUC 0.83):** In realistic recruitment datasets, cold-start accuracy settles at ~66% with a macro ROC-AUC of 0.83.

### What Does Each Label Actually Measure?
- **Control (`0xnbk`):** Measures 3-class fit categories concatenated with `" SEP "`.
- **C1 (`harsh72877287`):** Measures keyword and embedding similarity across 23 static job description prompts.
- **C2 (`netsol`):** Measures the ability to imitate GPT-4o's internal rubric across structured evaluation criteria.
- **C3 (`rahulrajpu`):** Measures the ability to reconstruct an explicit 6-part weighted arithmetic equation.
- **C4 (`surendra365`):** Measures random noise once demographic variables are removed.
- **C5 (`cnamuangtoun`):** Measures candidate fit classes (`No Fit`, `Potential Fit`, `Good Fit`), though specific annotation methodology is not documented.

### Honest Framing Recommendations for Documentation / Reports

| Dataset | Unacceptable / Misleading Claim | Honest, Rigorous Claim |
| :--- | :--- | :--- |
| **C5 (`cnamuangtoun`)** | *"Achieves 90% accuracy in hiring selection."* | *"Achieves 65.7% cold-start accuracy (AUC 0.83, macro-F1 62.8%) at predicting candidate fit classes on cnamuangtoun/resume-job-description-fit (label provenance not documented) on unseen candidates."* |
| **Control (`0xnbk`)** | *"Achieves 45% accuracy due to label corruption."* | *"Achieves 65.8% cold-start accuracy (AUC 0.83, macro-F1 62.8%) under direct 3-class classification, but drops to 45.0% when framed as continuous regression with post-hoc binning."* |
| **C2 (`netsol`)** | *"Accurately determines whether a candidate is qualified for a job."* | *"Achieves 55-59% cold-start accuracy ($R^2 \approx 0.34$) at predicting GPT-4o multi-criteria candidate evaluation scores without querying the LLM."* |
| **C1 (`harsh72877287`)** | *"Predicts ATS scores with 73% accuracy."* | *"Exhibits severe template overfitting: reaches 70% accuracy on 23 fixed job descriptions, but collapses to 46.4% ($R^2 < 0$) on unseen job postings."* |
| **C3 (`rahulrajpu676`)** | *"Achieves 99.9% matching precision."* | *"Exhibits extreme target clustering (99.88% in a single bin); regression models achieve an $R^2$ of 0.40 against the synthetic weighted formula."* |

---

## 4. Final Recommendation & Decision Gate

1. **Candidate C5 (`cnamuangtoun/resume-job-description-fit`) — PRIMARY BENCHMARK (ACCEPTED):**
   Selected as the primary dataset. Provides 8,000 balanced pairs, realistic text distributions, and clean native columns without artificial string concatenation.
2. **Control (`0xnbk/resume-ats-score-v1-en`) — LABELED COMPARISON (ACCEPTED):**
   Retained as a labeled comparison. Re-run confirms intact class labels matching C5 performance (65.8% accuracy, AUC 0.83).
3. **REJECT Candidates C1, C3, and C4:**
   - C1 fails generalization to new jobs (only 23 JDs, $R^2 < 0$ on cold JDs).
   - C3 suffers from extreme class collapse (99.88% in one bin).
   - C4 is a demographic toy dataset with near-zero textual predictive signal (AUC 0.53).
