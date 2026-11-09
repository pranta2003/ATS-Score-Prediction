# Empirical Diagnostics: Missing Statistics & Ranking Verification

**Date:** November 7, 2026  
**Auditor:** Senior ML Engineer  
**Dataset:** C5 (`cnamuangtoun/resume-job-description-fit`, 8,000 pairs, 643 resumes, 351 JDs)  
**Execution:** 100% computed from frozen artifacts and verified cross-validation runs on local compute.

---

## 1. Paired Cluster-Bootstrap CIs on Feature Tiers (vs T2 Dense)

Evaluating whether lexical, late-interaction, or full rubric tiers significantly outperform dense vectors:

| Tier Comparison | Scenario S1 (Cold Resume) | Scenario S2 (Cold JD) | Scenario S3 (Double Cold) |
| :--- | :---: | :---: | :---: |
| **T1 (Lexical) vs T2 (Dense)** | $\Delta$ Acc: +0.0078 [-0.0100, +0.0224]<br>$\Delta$ AUC: +0.0097 | $\Delta$ Acc: -0.0100 [-0.0325, +0.0129]<br>$\Delta$ AUC: -0.0204 | $\Delta$ Acc: -0.0119 [-0.0613, +0.0536]<br>$\Delta$ AUC: -0.0216 |
| **T4 (Lex+Dense) vs T2 (Dense)** | $\Delta$ Acc: +0.0040 [-0.0051, +0.0129]<br>$\Delta$ AUC: +0.0079 | $\Delta$ Acc: -0.0034 [-0.0148, +0.0075]<br>$\Delta$ AUC: +0.0026 | $\Delta$ Acc: +0.0051 [-0.0259, +0.0505]<br>$\Delta$ AUC: +0.0104 |
| **T5 (Full) vs T2 (Dense)** | $\Delta$ Acc: +0.0068 [-0.0040, +0.0174]<br>$\Delta$ AUC: +0.0098 | $\Delta$ Acc: -0.0006 [-0.0117, +0.0099]<br>$\Delta$ AUC: +0.0064 | $\Delta$ Acc: +0.0079 [-0.0326, +0.0616]<br>$\Delta$ AUC: +0.0074 |

---

## 2. Paired Cluster-Bootstrap CIs: Cross-Encoder vs Bi-Encoder

Matched evaluations across identical test partitions and random seeds {42, 43, 44}:

| Scenario | Bi-Encoder AUC | Cross-Encoder AUC | $\Delta$ AUC (Range) | $\Delta$ Macro-F1 (Range) | Significant $\neq 0$? |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **S2 (Cold JD)** | 0.7337 | 0.5621 | **-0.1716** [-0.2722, -0.0882] | **-0.1480** [-0.2154, -0.1142] | **YES (Bi-Encoder > CE)** |
| **S3 (Double Cold)** | 0.6546 | 0.5512 | **-0.1034** [-0.2063, -0.0231] | **-0.0953** [-0.2100, -0.0040] | **YES (Bi-Encoder > CE)** |

> **Finding**: In both scenarios, the 95% range strictly excludes zero. The bi-encoder statistically significantly outperforms the cross-encoder under this setup (MiniLM-L6, 512 tokens).

---

## 3. Binary Good-vs-No-Fit Discrimination (Excluding Potential Fit)

Evaluating binary separation between **Good Fit** (relevance 2) and **No Fit** (relevance 0):

| Scenario | Model Binary AUC (95% CI) | Baselines Prior AUC (95% CI) | Status |
| :--- | :---: | :---: | :---: |
| **S1 (Cold Resume)** | **0.8970** [0.8818, 0.9135] | **0.8510** [0.8301, 0.8716] (JD Prior) | Model dominates baseline |
| **S2 (Cold JD)** | **0.7767** [0.7435, 0.8087] | **0.6671** [0.6185, 0.7165] (Resume Prior) | Model dominates baseline |

---

## 4. Within-Job Description Ranking Performance

Evaluated on eligible JDs ($\ge 5$ candidates containing at least one Good Fit and at least one No Fit, total 137 eligible JDs). Primary evaluation is **S2 (Cold JD)** where a single fold model scores all candidates for a held-out JD. In **S1 (Cold Resume)**, candidates originate from different fold models, so S1 metrics are strictly indicative.

### Primary Benchmark: S2 (Cold JD) — Single Model Per JD
| Metric | Full Model $P(\text{Good Fit})$ | Resume-Only Model | Resume-Prior Baseline | Random Baseline |
| :--- | :---: | :---: | :---: | :---: |
| **Mean AUC (95% CI of Mean)** | **0.7524** [0.7172, 0.7850] | 0.6662 [0.6272, 0.7063] | 0.6637 [0.6216, 0.7025] | 0.4906 [0.4565, 0.5246] |
| **Median AUC (IQR [Q25, Q75])** | **0.7880** [0.6667, 0.9093] | 0.7304 [0.5556, 0.8365] | 0.7222 [0.5502, 0.8365] | 0.5000 [0.3862, 0.5964] |
| **Mean NDCG@5 (95% CI of Mean)** | **0.7534** [0.7090, 0.7951] | 0.6343 [0.5873, 0.6797] | 0.6345 [0.5878, 0.6840] | 0.5702 [0.5251, 0.6132] |
| **Median NDCG@5 (IQR [Q25, Q75])** | **0.8304** [0.6309, 1.0000] | 0.6367 [0.4469, 0.9152] | 0.6516 [0.4307, 0.9270] | 0.5774 [0.3869, 0.8304] |

#### Paired Differences on S2 (Within-JD Bootstrap CIs)
- **Model vs Resume Prior:** $\Delta$ AUC = **+0.0887** [+0.0490, +0.1281], $\Delta$ NDCG@5 = **+0.1189** [+0.0741, +0.1644].
- **Full Model vs Resume-Only Model:** $\Delta$ AUC = **+0.0862** [+0.0450, +0.1215], $\Delta$ NDCG@5 = **+0.1190** [+0.0752, +0.1652].

*Analysis:* Because both paired 95% confidence intervals strictly exclude zero, the full interaction model significantly outperforms both the empirical resume prior and the resume-only embedding model, demonstrating genuine resume-JD matching signal beyond candidate-level quality.

### Indicative Benchmark: S1 (Cold Resume) — Multi-Model Aggregation
| Metric | Model $P(\text{Good Fit})$ | Random Baseline |
| :--- | :---: | :---: |
| **Mean AUC (95% CI of Mean)** | **0.6747** [0.6413, 0.7071] | 0.4719 [0.4368, 0.5036] |
| **Median AUC (IQR [Q25, Q75])** | **0.6691** [0.5590, 0.8000] | 0.4638 [0.3667, 0.5818] |
| **Mean NDCG@5 (95% CI of Mean)** | **0.6855** [0.6474, 0.7264] | 0.5414 [0.4937, 0.5857] |
| **Median NDCG@5 (IQR [Q25, Q75])** | **0.6844** [0.5317, 0.8539] | 0.5538 [0.3960, 0.7227] |

*Note on S1 Baseline:* The JD-prior baseline is omitted because a JD prior is constant inside a JD, yielding AUC 0.5000 by definition; previous 0.2520 values were a leave-one-out fold artifact.

---

## 5. Calibration & Temperature Scaling Analysis (Nested GroupKFold Evaluation)

| Generalization Scenario | Raw Expected Calibration Error (ECE) | Calibrated ECE (Temperature Scaled) | Median Temperature ($T$) | Calibration Effect |
| :--- | :---: | :---: | :---: | :---: |
| **S1 (Cold Resume)** | **0.0496** | **0.0188** | 1.19 | **-0.0309 ECE** |
| **S2 (Cold JD)** | **0.1333** | **0.0357** | 1.89 | **-0.0976 ECE** |

> **Operational Recommendation**: Calibrated probabilities are strongly recommended. Nested temperature scaling reduces ECE on S1 from 0.0496 to 0.0188 (median T=1.19) and on S2 from 0.1333 to 0.0357 (median T=1.89), aligning confidence with empirical accuracy for recruiter decisions.

---

## 6. Cross-Encoder Truncation Analysis

- **Tokenizer / Architecture:** `sentence-transformers/all-MiniLM-L6-v2`
- **Sequence Boundary Limit:** `512` tokens
- **Truncation Policy:** `longest_first (Resume and JD truncated proportionately to fit 512 tokens)`
- **Mean Token Length:** 1676.9 tokens (Median: 1563.0, 90th percentile: 2467.2, 99th percentile: 3890.0)
- **Exceeding Fraction:** **99.58%** (7966/8000 pairs)

> **Documentation Directive**: Because 99.58% of resume-JD pairs exceed the 512-token limit and suffer loss of qualifications from the longest input, all cross-encoder findings are strictly restricted to:  
> *"this cross-encoder setup (MiniLM-L6, 512 tokens)"*.
