# C5 Verification Gate & Generalization Audit (Step 2)

**Author:** Senior Machine Learning Engineer  
**Date:** November 5, 2026  
**Status:** Verification Gate Completed — Stop Condition Triggered for User Consultation  
**Dataset:** `cnamuangtoun/resume-job-description-fit` (8,000 pairs, 643 unique resumes, 351 unique JDs)  
**Branch:** `exp/c5-verification`  

---

## 1. Pre-Registered Metrics & Evaluation Protocol

Before running the verification suite, the evaluation metrics were pre-registered as follows:
- **Primary Metrics:** Macro-F1 and Macro ROC-AUC (one-vs-rest across the 3 classes: `Good Fit`, `No Fit`, `Potential Fit`).
- **Secondary Metrics:** Overall Accuracy, Balanced Accuracy, Per-class Recall, and Binary Good-vs-No-Fit ROC-AUC (dropping `Potential Fit` rows, disclosed as such).
- **Uncertainty Quantification:** 95% cluster-bootstrap confidence intervals (500 iterations).

---

## 2. Generalization Scenarios (Master Results Table)

We evaluated the baseline chunked MiniLM + interaction representation with LightGBM under three distinct cross-validation and split regimes:
1. **S1 (Cold-Resume):** 5-fold `GroupKFold` on `resume_id`. Resumes in validation never appear in training; JDs repeat.
2. **S2 (Cold-JD):** 5-fold `GroupKFold` on `jd_id`. JDs in validation never appear in training; resumes repeat.
3. **S3 (Double-Cold):** Disjoint holdout of 15% unique resumes AND 15% unique JDs. Train only on pairs containing *neither* test entity; test on pairs containing *both*. Evaluated across 5 random seeds (42, 43, 44, 45, 46).

| Scenario | Split Type | Test N (avg) | Accuracy [95% CI] | Macro-F1 [95% CI] | Balanced Accuracy | Macro ROC-AUC [95% CI] | Binary Good vs No-Fit AUC | Majority Baseline |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **S1 (Cold-Resume)** | GroupKFold (`resume_id`) | 8,000 (OOF) | **65.83%** [63.53%, 68.28%] | **63.10%** [60.53%, 65.81%] | 62.44% | **0.8292** [0.8110, 0.8469] | 0.8927 | 50.00% |
| **S2 (Cold-JD)** | GroupKFold (`jd_id`) | 8,000 (OOF) | **55.21%** [52.76%, 57.54%] | **43.70%** [40.36%, 46.92%] | 44.58% | **0.7111** [0.6877, 0.7361] | 0.7648 | 50.00% |
| **S3 (Double-Cold)** | Disjoint Resumes & JDs (5 seeds) | 186.8 / seed | **54.24%** [46.02%, 64.87%] | **39.87%** [34.30%, 44.79%] | 41.18% | **0.6562** [0.5974, 0.7372] | 0.6983 | 52.53% |

### Per-Class Recall Breakdown:
- **S1 (Cold-Resume):** Good Fit = **67.40%**, No Fit = **75.98%**, Potential Fit = **43.95%**.
- **S2 (Cold-JD):** Good Fit = **18.55%**, No Fit = **87.13%**, Potential Fit = **28.05%**.
- **S3 (Double-Cold):** Good Fit = **15.97%**, No Fit = **85.93%**, Potential Fit = **21.65%**.

---

## 3. Shortcut & Leakage Ablations

### A1–A3: Feature Modality Ablations (Evaluated on S3 Protocol, Seed 42)
| Feature Representation | Input Dimensions | Accuracy | Macro-F1 | Macro ROC-AUC | Binary Good vs No-Fit AUC | Majority Baseline |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **A1: Resume-Only ($X_{res}$)** | 384 | 43.89% | 22.55% | 0.5949 | 0.6090 | 43.89% |
| **A2: JD-Only ($X_{jd}$)** | 384 | 43.89% | 20.33% | 0.5520 | 0.4181 | 43.89% |
| **A3: Full Interaction ($X_{full}$)** | 1,537 | **48.89%** | **37.98%** | **0.6482** | **0.6404** | 43.89% |

*Interpretation:* Neither text alone can predict candidate fit (Macro-F1 ~20-22%, collapsing to majority class). Full cross-modal interaction features significantly improve Macro-F1 (+15.4% to +17.6%) and AUC (+0.05 to +0.10), proving the model relies on the relationship between resume and job description.

---

### A4: JD-Swap Sensitivity Test
To test whether the model genuinely evaluates the specific job description or merely predicts candidate competence / resume priors, each test pair's JD was replaced at test time with a randomly selected test JD having a different historical Good Fit rate.
- **Predictions Flipped:** **27.22%** of test samples changed class prediction when the JD was swapped.
- **Mean L1 Probability Shift:** **0.3917** across the 3-class probability simplex.
- **Accuracy Drop:** Accuracy dropped from **48.89% down to 43.89%** ($\Delta = -5.00\%$).
- **Verdict:** **The model does NOT ignore the JD.** It exhibits significant sensitivity to the job description text.

---

### A5: Label Base Rates & The "JD Prior" Discovery
Why does S1 (Cold-Resume) achieve 65.8% accuracy while S2 (Cold-JD) and S3 (Double-Cold) drop to 54-55%?
We analyzed the per-JD class distributions across all 351 unique JDs:
- Mean Good Fit rate across JDs: **16.84%** (std: 24.35%, range: **0.0% to 84.38%**).
- Top 10 JDs have Good Fit rates between **74.3% and 84.4%**!
- Bottom 10 JDs have **0.0%** Good Fit rates (almost 100% No Fit).
- **Out-of-Fold JD Prior-Only Baseline:**
  A dummy model that ignores the resume text and simply predicts the historical majority class of the specific JD achieves:
  - **Accuracy = 60.83%**
  - **Macro-F1 = 58.89%**

*Critical Takeaway:* In S1 (Cold-Resume), the job descriptions are seen during training. Because individual job postings have strong idiosyncratic base rates (some jobs accept 80% of applicants, others 0%), a model evaluated with unseen resumes but *seen JDs* exploits the JD base rate prior. When JDs are strictly held out (S2 and S3), this prior is unavailable, and accuracy drops to 54-55%.

---

### A6: Official HuggingFace Train/Test Split Leakage Audit
We audited the official `train` (6,241 rows) and `test` (1,759 rows) partitions on HuggingFace:
- **Test Unique Resumes:** 477 | **Test Unique JDs:** 71
- **Test JDs appearing in Train:** **0 / 71 (0.00%)** — Strictly cold JDs.
- **Test Resumes appearing in Train:** **476 / 477 (99.79%)**!
- **Exact (Resume, JD) Pairs Overlap:** 0.

*Critical Finding:* The official HuggingFace split published by `cnamuangtoun` is **not a cold-resume split**; it is an **S2 Cold-JD split with near-100% warm resumes**! Any benchmark claiming ~66% on the official test set is evaluating under Cold-JD / Warm-Resume conditions.

---

## 4. Headline Generalization Number & Stop Rule Analysis

Per prompt instructions:
> "State the number we can honestly claim as the 'headline' = the LOWER of S2 and S3 results."
> "Stop and report to me if A4 shows the model ignores the JD, or if the S3 macro-AUC drops below 0.65."

### Comparing S2 vs. S3:
- **S2 (Cold-JD):** Accuracy = 55.21%, Macro-F1 = 43.70%, Macro ROC-AUC = 0.7111
- **S3 (Double-Cold):** Accuracy = **54.24%**, Macro-F1 = **39.87%**, Macro ROC-AUC = **0.6562**

The lower result is **S3 (Double-Cold)**.

### Honest Headline Claim:
> *"Under a rigorous double-cold evaluation protocol where both candidates and job descriptions are held out, our baseline model achieves **54.24% accuracy** (vs. 52.53% majority baseline), **39.87% macro-F1**, and a **macro ROC-AUC of 0.6562** [95% CI: 0.5974, 0.7372]. When job descriptions are known (cold-resume), performance rises to 65.83% accuracy and 0.8292 AUC by leveraging job-specific acceptance priors."*

---

## 5. Decision Gate & Stop Condition

### Stop Condition Assessment:
1. **Did A4 show the model ignores the JD?**  
   **No.** Predictions flip 27.22% with an L1 probability shift of 0.3917 and an accuracy drop of 5.0%.
2. **Did S3 Macro-AUC drop below 0.65?**  
   - Mean S3 Macro ROC-AUC across 5 random seeds is **0.6562** (just above 0.6500).
   - However, **3 out of the 5 seeds dropped below 0.65** (Seed 42: 0.6482, Seed 45: 0.5944, Seed 46: 0.6249), and the 95% bootstrap confidence interval lower bound is **0.5974**.

Because individual seeds and the lower confidence bound dip below 0.65, and because the honest double-cold accuracy (54.24%) is only ~1.7% above the majority baseline (52.53%), this threshold boundary is explicitly flagged for your decision before proceeding to Step 3 model building.
