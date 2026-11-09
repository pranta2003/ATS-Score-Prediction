# Like-for-Like Leakage Comparison Report

## Overview
Evaluates the exact same **Tier 2 Dense Semantic** feature set (`[u, v, |u-v|, u*v, cos(u,v)]`) fitted with **LightGBM** (300 trees, lr=0.05, max_depth=6) across three evaluation protocols on C5 (8,000 resume-JD pairs):
1. **Random Stratified 5-Fold (Warm/Leaked):** Pairs randomly split across folds; allows both resumes and JDs to overlap between train and validation.
2. **S1 Cold-Resume 5-Fold (Grouped by Resume):** Resumes strictly partitioned; no resume seen in training appears in validation.
3. **S2 Cold-JD 5-Fold (Grouped by JD):** Job descriptions strictly partitioned; simulates deploying to new job postings.

---

## Measured Performance Comparison

| Protocol | Split Grouping | Accuracy (95% CI) | Macro-F1 (95% CI) | Macro-AUC |
|---|---|---|---|---|
| **Random Stratified** | None (Pair-level) | **73.16%** [72.17%, 74.15%] | **70.75%** [69.69%, 71.82%] | **0.8781** |
| **S1 Cold-Resume** | `resume_id` (643 groups) | **66.16%** [63.85%, 68.32%] | **63.40%** [60.67%, 65.84%] | **0.8307** |
| **S2 Cold-JD** | `jd_id` (351 groups) | **56.20%** [53.79%, 58.65%] | **45.70%** [42.44%, 48.73%] | **0.7294** |

---

## Measured Leakage Gaps

| Comparison | $\Delta$ Accuracy (95% Paired CI) | $\Delta$ Macro-F1 (95% Paired CI) | $\Delta$ Macro-AUC |
|---|---|---|---|
| **Random vs S1 (Cold Resume)** | **+7.00 pp** [5.48 pp, 8.58 pp] | **+7.35 pp** [5.72 pp, 9.01 pp] | **+0.0474** |
| **Random vs S2 (Cold JD)** | **+16.96 pp** [14.58 pp, 19.44 pp] | **+25.05 pp** [21.72 pp, 28.41 pp] | **+0.1487** |

### Key Findings
1. Random pair splitting inflates accuracy by **7.00 percentage points** compared to Cold-Resume evaluation (S1), and by **16.96 percentage points** compared to Cold-JD evaluation (S2).
2. The measured leakage gap replaces any hypothetical estimates with directly audited, like-for-like experimental measurements.
