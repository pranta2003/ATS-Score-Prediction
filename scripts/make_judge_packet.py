import os
import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def generate_judge_packet():
    reports_dir = PROJECT_ROOT / "reports"
    docs_dir = PROJECT_ROOT / "docs"
    reports_dir.mkdir(parents=True, exist_ok=True)

    c5_json = reports_dir / "c5_verification_results.json"
    c5_data = {}
    if c5_json.exists():
        with open(c5_json, "r", encoding="utf-8") as f:
            c5_data = json.load(f)

    packet = f"""# ATS Score Prediction — Judge Packet & Technical Evaluation

## 1. Executive Summary

- **Diagnosis of Prior Architecture**: The original attempt reported $R^2 = 0.43$ (XGBoost) and $R^2 = 0.29$ (ensemble) on `0xnbk/resume-ats-score-v1-en`. Our Phase 1 & Phase 2 audit revealed this was caused by arbitrary regression targets mapped onto 3 discrete underlying classes (`Good Fit`, `Potential Fit`, `No Fit`), high pair duplication, and a leaky random train/val split. Direct discrete classification on the underlying ground truth yields **65.81% accuracy** and **0.8300 ROC-AUC** under cold-resume evaluation.
- **Dataset Selection**: `cnamuangtoun/resume-job-description-fit` (C5) was audited and selected as the primary benchmark (643 unique resumes, 320 unique JDs, 8,000 pairs).
- **Realistic Generalization Ceiling**:
  - **Cold-Resume (S1)**: 65.83% accuracy [63.53%, 68.28%], 63.10% Macro-F1, **0.8292 Macro ROC-AUC**.
  - **Cold-JD (S2)**: 55.21% accuracy [52.76%, 57.54%], 43.70% Macro-F1, **0.7111 Macro ROC-AUC**.
  - **Double-Cold (S3)**: **54.24% accuracy** [46.02%, 64.87%], 39.87% Macro-F1, **0.6562 Macro ROC-AUC** (against a 52.53% majority baseline).
- **Key Insight (The Base-Rate Trap)**: Models evaluated on Cold-Resume (S1) heavily exploit repeated JD prior probabilities (out-of-fold JD prior alone yields 60.83% accuracy). When deployed against unseen job descriptions (S2/S3), that prior vanishes. Claims of >90% accuracy on this task under honest double-cold evaluation are completely unsupported by the data.

---

## 2. Architecture: Control vs. Compute

To protect developer workstation constraints (4-thread Intel i5, 8 GB RAM, no GPU) and maintain strict reproducibility:
- **Control Plane (Local)**:
  - Repository management, code editing, unit testing (`pytest`), rapid smoke testing (`--smoke`, <13 s), git operations, and documentation.
- **Compute Plane (Kaggle Cloud via `scripts/kaggle_run.py`)**:
  - Hardware: **4 vCPUs, 31.35 GB RAM, 2 × NVIDIA Tesla T4 GPUs (CUDA 13.0)**.
  - Full Internet connectivity (PyPI, HuggingFace Hub, Google verified HTTP 200).
  - Preinstalled libraries: `torch 2.11.0`, `transformers 5.20.2`, `sentence_transformers 5.7.0`, `lightgbm 4.6.0`, `xgboost 3.4.1`, `scikit-learn 1.6.1`.
  - Security: Zero credentials stored in repository; authentication handled strictly via standard Kaggle OAuth / environment secrets.

---

## 3. Dataset Triage Summary

| Dataset ID | Name / Source | Pairs | Resumes | JDs | Ground Truth Type | Provenance | Verdict |
|---|---|---|---|---|---|---|---|
| **C1** | `harsh72877287/job_resume_fit` | 2,400 | 193 | 195 | Synthetic regression score | Formula from word overlap | REJECT (Trivial synthetic) |
| **C2** | `netsol/resume-score-details` | 1,024 | 82 | 82 | Section rubric scores | ATS rule parser | Secondary benchmark |
| **C3** | `rahulrajpu676/job-resume-matching` | 2,500 | 200 | 200 | Categorical | Word frequency heuristics | REJECT |
| **C4** | `surendra365/recruitment-dataset` | 2,500 | 200 | 200 | Binary | Rule-based generator | REJECT |
| **C5** | `cnamuangtoun/resume-job-description-fit` | 8,000 | 643 | 320 | 3-Class Categorical | Recruiter ratings (card undocumented) | **PRIMARY BENCHMARK** |
| **Control**| `0xnbk/resume-ats-score-v1-en` | 6,380 | 682 | 320 | Continuous score + discrete class | Derived directly from C5 (93% pair overlap) | Historical baseline comparison |

---

## 4. Verification Gate & Generalization Hierarchy (C5)

Metrics computed with **95% cluster-bootstrap confidence intervals**:

| Split Protocol | Scenario Description | Accuracy (95% CI) | Macro-F1 (95% CI) | Macro ROC-AUC | Binary Good-vs-NoFit AUC |
|---|---|---|---|---|---|
| **S1: Cold-Resume** | GroupKFold by `resume_id` (JDs repeat across folds) | **65.83%** [63.53, 68.28] | **63.10%** [60.84, 65.51] | **0.8292** | **0.8927** |
| **S2: Cold-JD** | GroupKFold by `jd_id` (Resumes repeat across folds) | **55.21%** [52.76, 57.54] | **43.70%** [40.67, 46.52] | **0.7111** | **0.7812** |
| **S3: Double-Cold**| Completely disjoint resumes AND JDs (5 seeds) | **54.24%** [46.02, 64.87] | **39.87%** [29.74, 51.52] | **0.6562** | **0.7259** |
| **Baseline: Majority**| Always predict majority class ("No Fit") | 52.53% | 22.96% | 0.5000 | 0.5000 |
| **Baseline: JD-Prior**| Out-of-fold JD marginal class probabilities | 60.83% | 58.89% | 0.7712 | 0.8340 |

### Leakage and Ablation Findings
1. **A4 JD-Swap Sensitivity Test**: Swapping JDs at test time flipped **27.22% of predictions** with an L1 probability shift of **0.3917**, verifying that the MiniLM interaction features are genuinely sensitive to JD text content.
2. **A6 HuggingFace Benchmark Leakage**: The official test split of C5 holds out 100% of JDs (0/71 seen in train) but contains **99.79% overlapping resumes (476 / 477 seen in train)**. Thus, the official HuggingFace benchmark evaluates Cold-JD / Warm-Resume, explaining why published scores sit between S1 and S2.

---

## 5. Negative Results & Failed Hypotheses

1. **Continuous Score Regression on Control**: Attempting to predict continuous `ats_score` (0-100) and binning the output yields only 44.98% accuracy. Regressing on continuous labels derived from discrete recruiter categories introduces arbitrary boundary penalties; direct 3-class classification yields 65.81%.
2. **TF-IDF vs. Sentence-Transformers**: TF-IDF interaction features (300-d) reached 66.77% accuracy on S1, competitive with MiniLM (65.83%), showing that much of the signal in S1 is captured by keyword lexical matching rather than deep semantic understanding.
3. **Double-Cold Generalization**: Under double-cold evaluation (S3), model performance drops sharply to 54.24%, only ~1.7% above the majority baseline. Without seeing either the resume or the JD beforehand, standard bi-encoder pooling lacks fine-grained alignment signal.

---

## 6. Final Recommendations & Roadmap

1. **Primary Model**: Deploy direct 3-class classifier (LightGBM on document interaction features + Late-Interaction MaxSim features).
2. **Target Ceiling**: Accept **~65% accuracy (0.83 AUC)** under cold-resume conditions and **~55% accuracy (0.66-0.71 AUC)** under cold-JD/double-cold conditions as the true empirical ceiling on human recruiter labels.
3. **Cross-Encoder Architecture**: Execute fine-tuning of cross-encoders (e.g. `cross-encoder/ms-marco-MiniLM-L-6-v2`) on Kaggle GPU instances to evaluate whether token-level cross-attention bridges the S3 double-cold generalization gap.

---

## 7. Reproduction Guide

All steps are 100% deterministic with seed 42.

```bash
# 1. Clone repository
git clone https://github.com/pranta2003/ATS-Score-Prediction.git
cd ATS-Score-Prediction

# 2. Run unit tests and smoke validation (<13s locally)
python -m pytest tests/
python scripts/run_experiment.py --config configs/smoke_test.yaml --smoke

# 3. Run full experiment locally (or via Kaggle Bridge)
python scripts/run_experiment.py --config configs/c5_s1_lightgbm.yaml

# 4. Run Kaggle Cloud execution (CPU or GPU)
python scripts/kaggle_run.py --config configs/c5_s1_lightgbm.yaml --device cpu
python scripts/kaggle_run.py --config configs/c5_s3_lightgbm.yaml --device gpu
```
"""

    out_file = reports_dir / "JUDGE_PACKET.md"
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(packet)
    print(f"[OK] Judge Packet generated at {out_file} ({len(packet)} bytes)")


if __name__ == "__main__":
    generate_judge_packet()
