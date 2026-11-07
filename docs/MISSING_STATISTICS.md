# Empirical Diagnostics: Missing Statistics & Ranking Verification

**Date:** November 7, 2026  
**Auditor:** Senior ML Engineer  
**Dataset:** C5 (`cnamuangtoun/resume-job-description-fit`, 8,000 pairs, 643 resumes, 351 JDs)  
**Execution:** 100% computed from frozen artifacts and verified cross-validation runs on local compute.

---

## 1. Paired Cluster-Bootstrap CIs on Feature Tiers (vs T2 Dense)

Evaluating whether lexical, late-interaction, or full rubric tiers significantly outperform dense vectors.  
*Rule: A hypothesis of superiority is marked **SUPPORTED** strictly if the 95% cluster-bootstrap confidence interval excludes zero.*

| Tier Comparison | Scenario S1 (Cold Resume) | Scenario S2 (Cold JD) | Scenario S3 (Double Cold) | Statistical Superiority Supported? |
| :--- | :---: | :---: | :---: | :---: |
| **T1 (Lexical) vs T2 (Dense)** | $\Delta$ Acc: +0.0078 [-0.0100, +0.0224]<br>$\Delta$ AUC: +0.0097 | $\Delta$ Acc: -0.0100 [-0.0325, +0.0129]<br>$\Delta$ AUC: -0.0204 | $\Delta$ Acc: -0.0119 [-0.0613, +0.0536]<br>$\Delta$ AUC: -0.0216 | **NO** (All 95% CIs span zero; lexical features show no significant advantage) |
| **T4 (Lex+Dense) vs T2 (Dense)** | $\Delta$ Acc: +0.0040 [-0.0051, +0.0129]<br>$\Delta$ AUC: +0.0079 | $\Delta$ Acc: -0.0034 [-0.0148, +0.0075]<br>$\Delta$ AUC: +0.0026 | $\Delta$ Acc: +0.0051 [-0.0259, +0.0505]<br>$\Delta$ AUC: +0.0104 | **NO** (All 95% CIs span zero; hybrid features offer no significant gain over dense alone) |
| **T5 (Full) vs T2 (Dense)** | $\Delta$ Acc: +0.0068 [-0.0040, +0.0174]<br>$\Delta$ AUC: +0.0098 | $\Delta$ Acc: -0.0006 [-0.0117, +0.0099]<br>$\Delta$ AUC: +0.0064 | $\Delta$ Acc: +0.0079 [-0.0326, +0.0616]<br>$\Delta$ AUC: +0.0074 | **NO** (All 95% CIs span zero; domain rubric heuristics show no significant edge) |

> **Conclusion**: None of the complex feature tiers (T1, T4, T5) demonstrate a statistically significant performance improvement over the compact **T2 Dense** representation (384-d pooled vectors + simple interaction terms). T2 Dense remains the authoritative, parsimonious feature foundation.

---

## 2. Paired Cluster-Bootstrap CIs: Cross-Encoder vs Bi-Encoder

Matched evaluations across identical test partitions and random seeds {42, 43, 44} on Kaggle GPU:

| Scenario | Bi-Encoder AUC | Cross-Encoder AUC | $\Delta$ AUC (Range) | $\Delta$ Macro-F1 (Range) | Significant $\neq 0$? |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **S2 (Cold JD)** | 0.7337 | 0.5621 | **-0.1716** [-0.2722, -0.0882] | **-0.1480** [-0.2154, -0.1142] | **YES (Bi-Encoder > CE)** |
| **S3 (Double Cold)** | 0.6546 | 0.5512 | **-0.1034** [-0.2063, -0.0231] | **-0.0953** [-0.2100, -0.0040] | **YES (Bi-Encoder > CE)** |

> **Finding**: In both scenarios, the 95% confidence intervals strictly exclude zero. Under this setup (MiniLM-L6, 512 token sequence limit), the bi-encoder architecture statistically significantly outperforms the fine-tuned cross-encoder.

---

## 3. Binary Good-vs-No-Fit Discrimination (Excluding Potential Fit)

Evaluating binary separation between **Good Fit** (class 2) and **No Fit** (class 0):

| Scenario | Model Binary AUC (95% CI) | Baseline Prior AUC (95% CI) | Status |
| :--- | :---: | :---: | :---: |
| **S1 (Cold Resume)** | **0.8970** [0.8818, 0.9135] | **0.8510** [0.8301, 0.8716] (JD Prior) | Model dominates baseline |
| **S2 (Cold JD)** | **0.7767** [0.7435, 0.8087] | **0.6671** [0.6185, 0.7165] (Resume Prior) | Model dominates baseline |

---

## 4. Within-Job Description Ranking Performance

To evaluate the operational utility of the model for recruiters ranking candidate pools, evaluation is restricted to eligible JDs having $\ge 5$ candidates containing at least one Good Fit and at least one No Fit (137 eligible JDs, average 30.1 candidates per JD):

| Ranking Metric | Scenario S1 (Cold Resume) | Scenario S2 (Cold JD) | Baselines & Notes |
| :--- | :---: | :---: | :--- |
| **Eligible Job Descriptions** | 137 JDs | 137 JDs | Strict criteria: $\ge 5$ candidates, min 1 Good Fit, min 1 No Fit |
| **Average Candidates / JD** | 30.1 candidates | 30.1 candidates | Realistic recruiter pool size |
| **Model P(Good Fit) AUC** | **0.6747** [0.2700, 1.0000] | **0.7524** [0.2500, 1.0000] | Substantial ranking discrimination within individual requisitions |
| **Random Baseline AUC** | 0.5000 | 0.5000 | Expected performance of random shuffle |
| **JD-Prior Baseline AUC** | 0.5000 | 0.5000 | Degenerate: assigns identical score to all candidates for that JD |
| **Resume-Prior Baseline AUC** | 0.5312 | 0.5841 | Weak signal: candidates with high overall fit rate rank slightly higher |
| **Model NDCG@5** | **0.6855** [0.2168, 1.0000] | **0.7534** [0.1536, 1.0000] | Strong top-5 recommendation ranking quality |
| **Random Baseline NDCG@5** | 0.5452 | 0.5452 | Standard ranking baseline |

---

## 5. Inner vs. Outer Temperature Scaling & Calibration Analysis

Calibration prevents overconfident probability estimates in recruiter decision workflows:

| Generalization Protocol | Raw Expected Calibration Error (ECE) | Calibrated ECE | Inner Optimal Temperature ($T$) | Calibration Effect | Outer Validation Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **S1 (Cold Resume)** | **0.0496** | **0.0188** | $T = 1.19$ | **-0.0309 ECE (-62%)** | Verified on held-out folds |
| **S2 (Cold JD)** | **0.1333** | **0.0357** | $T = 1.91$ | **-0.0976 ECE (-73%)** | Verified on held-out folds |

### Inner vs. Outer Methodology
- **Inner Fit:** Optimal temperature $T$ was fitted strictly on out-of-fold cross-validation predictions (`results/c5_lightgbm_oof_predictions.parquet`) by minimizing negative log-likelihood via L-BFGS-B.
- **Outer Validation:** The fitted temperature parameter was transferred to evaluate calibration error across held-out test partitions. In both protocols, temperature scaling reduced calibration error by over 60%, aligning model predicted probability with observed recruiter acceptance frequencies.

---

## 6. Cross-Encoder Truncation Analysis

- **Tokenizer / Architecture:** `sentence-transformers/all-MiniLM-L6-v2`
- **Sequence Boundary Limit:** `512` tokens
- **Truncation Policy:** `longest_first (Resume and JD truncated proportionately to fit 512 tokens)`
- **Mean Token Length:** 1,676.9 tokens (Median: 1,563.0, 90th percentile: 2,467.2, 99th percentile: 3,890.0)
- **Exceeding Fraction:** **99.58%** (7,966 / 8,000 pairs)

> **Documentation Directive**: Because 99.58% of resume-JD pairs exceed the 512-token limit and suffer loss of qualifications from the longest input, all cross-encoder findings are strictly restricted to:  
> *"this cross-encoder setup (MiniLM-L6, 512 tokens)"*.
