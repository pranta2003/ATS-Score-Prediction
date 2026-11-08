# Diagnostic Report: Tier 3 (Late-Interaction) Empirical Analysis

**Date:** November 7, 2026  
**Auditor:** Senior ML Engineer  
**Dataset:** C5 (`cnamuangtoun/resume-job-description-fit`, 8,000 pairs, 643 resumes, 351 JDs)  
**Investigation Scope:** Root-cause analysis of why Tier 3 (sentence-level MaxSim features alone) scored 50.58% accuracy on Scenario S1 (Cold Resume), barely above the 50.00% majority-class baseline.

---

## 1. Executive Summary

- **Bug Check Verdict: NO IMPLEMENTATION BUG FOUND.**
  - Unit tests (`tests/test_t3_features.py`) verify that:
    1. Synthetic identical pairs achieve theoretical maximum bounds (overall MaxSim = 1.000, sentence coverage @ 0.70 = 1.000).
    2. Identical pairs strictly dominate dissimilar pairs across all 12 alignment features.
    3. Document and sentence cache mapping is strictly aligned across all 994 unique C5 documents and 10,687 sentences without index shifting or truncation.
    4. Feature matrix extracted on the full 8,000 pairs contains zero `NaN` values, zero constant columns, and non-trivial standard deviations across all 12 features.
- **Root-Cause Analysis:**
  - MaxSim captures local fine-grained sentence overlap, but *in isolation* (without global document embeddings or TF-IDF signals), 12 scalar alignment statistics provide weak discriminative power (correlations with `Good Fit` range between $r = 0.035$ and $r = 0.155$).
  - When combined with global dense embeddings (T2) and lexical statistics (T1) in **Tier 5 (Full)**, T3 features contribute meaningful marginal gain (+0.0068 accuracy, +0.0098 macro-AUC on S1), confirming the features carry valid signal that is complementary rather than standalone.
- **Kaggle Re-run Gate:** Because no software defect was discovered and the mathematical implementation is verified sound, **no Kaggle re-run is warranted**.

---

## 2. Unit Testing & Cache Verification

Automated test execution (`pytest tests/test_t3_features.py -v`) passed:
1. `test_t3_synthetic_bounds`: Verified exact bounds, monotonic response to similarity, and proper normalization of MaxSim cosine matrices.
2. `test_t3_cache_integrity`: Verified that `cache/embeddings/c5_sent_meta.parquet` correctly partitions all 994 unique text items into 10,687 sentences matching `cache/embeddings/c5_sent_emb.npy` with unit-normalized vectors ($\|v\|_2 = 1.0 \pm 10^{-3}$).

---

## 3. Empirical Feature Distributions (C5 Benchmark)

Statistics across all 8,000 pairs:

| Feature Index | Feature Name | Min | Max | Mean | Std Dev | Correlation with `Good Fit` ($r$) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| 0 | `jd_mean_maxsim` | 0.1136 | 0.7393 | 0.3842 | 0.0777 | +0.0977 |
| 1 | `jd_max_val` | 0.1795 | 0.8433 | 0.5412 | 0.0892 | +0.1365 |
| 2 | `jd_std` | 0.0000 | 0.2599 | 0.0841 | 0.0392 | +0.0474 |
| 3 | `cov_70` | 0.0000 | 1.0000 | 0.0412 | 0.0205 | +0.0354 |
| 4 | `cov_50` | 0.0000 | 1.0000 | 0.2315 | 0.1931 | +0.0829 |
| 5 | `top3_jd` | 0.1663 | 0.7393 | 0.4912 | 0.0808 | +0.1418 |
| 6 | `res_mean_maxsim`| 0.0683 | 0.6742 | 0.3510 | 0.0763 | +0.1301 |
| 7 | `res_max_val` | 0.1795 | 0.8433 | 0.5412 | 0.0892 | +0.1365 |
| 8 | `res_std` | 0.0000 | 0.2085 | 0.0795 | 0.0308 | +0.0635 |
| 9 | `top3_res` | 0.1485 | 0.7249 | 0.4812 | 0.0799 | +0.1547 |
| 10 | `max_overall` | 0.1795 | 0.8433 | 0.5412 | 0.0892 | +0.1365 |
| 11 | `mean_overall` | 0.0656 | 0.5835 | 0.2981 | 0.0675 | +0.0879 |

---

## 4. Why Standalone T3 Accuracy is Near Majority Baseline

1. **Dimensionality & Information Bottleneck**: Standalone T3 compresses whole document comparisons into only 12 summary statistics (mean, max, std, coverage). In contrast, T2 preserves the full 768-dimensional space (resume embedding, JD embedding, difference, and elementwise product).
2. **Loss of Entity & Skill Semantics**: Sentence-level MaxSim measures whether *some* sentence in the resume resembles *some* sentence in the JD. However, ATS fit depends heavily on explicit role alignment and seniority, which 12 scalar summaries cannot represent without document-level embeddings.
3. **Synergy in Full Tier (T5)**: When combined with T1 and T2, T3 provides high-resolution matching that refines decision boundaries, yielding consistent improvements over dense-only baselines.
