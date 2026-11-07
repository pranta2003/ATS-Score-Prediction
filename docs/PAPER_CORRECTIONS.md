# Scientific Corrections & Forensic Retraction Log

**Auditor:** Senior ML Engineer  
**Date:** November 7, 2026  
**Document Purpose:** Formally record all retractions, reconciliations, and methodological corrections made during the diagnostic review of the ATS Score Prediction project.

---

## 1. Retraction of Claims Regarding Control Dataset (0xnbk)

1. **Retracted Claim**: *"The control dataset was corrupted by artificial Gaussian noise."*
   - **Correction**: The underlying class labels in `0xnbk/resume-ats-score-v1-en` (`original_label`) are completely intact ($\eta^2 = 0.987$ correlation with continuous score ranges). The previous poor result ($R^2 = 0.29$ to $0.43$) was caused by fitting a continuous regression model to synthetic score targets, then binning continuous outputs into discrete classes.
   - **Evidence**: When evaluated with direct 3-class classification on identical group splits, Control achieves **65.81% accuracy** and **0.8300 ROC-AUC**, closely matching C5 (65.66% accuracy, 0.8297 ROC-AUC).

2. **Reconciliation of Duplicate Counts (Phase 1 vs Triage Audit)**:
   - **Discrepancy**: Phase 1 reported 0 duplicates in `train.csv`; subsequent triage reported 100 duplicates across the full dataset.
   - **Root Cause**:
     - Phase 1 evaluated exact literal raw string equality on `train.csv` alone (which has 0 literal byte duplicates).
     - Full triage evaluated whitespace-normalized and case-folded text across combined `train.csv` and `validation.csv` (which reveals 56 normalized duplicates within train, and 100 normalized duplicates across train and validation combined).

---

## 2. Retraction of Claims Regarding C5 Provenance

1. **Retracted Claim**: *"C5 consists of authentic human recruiter annotations with 25-35% subjective disagreement."*
   - **Correction**: The dataset card for `cnamuangtoun/resume-job-description-fit` does not provide formal documentation of annotation protocols, recruit panels, or inter-rater reliability.
   - **Corrected Designation**: All technical reports and model cards now explicitly record provenance as:  
     `"undocumented (no dataset card; assumed human-labeled)"`.

---

## 3. Retraction of Memory Claims Regarding ONNX vs PyTorch

1. **Retracted Claim**: *"ONNX Runtime consumes more memory than PyTorch (580 MB vs 522 MB)."*
   - **Correction**: The initial test script imported `transformers.AutoTokenizer`, which pulled PyTorch and CUDA bindings into memory prior to launching ONNX.
   - **Verified Ground Truth**: When profiled in clean, isolated processes using Rust `tokenizers` and pure `onnxruntime`, the quantized ONNX model consumes only **243.55 MB Peak RSS**, saving **279.22 MB (~53% RAM reduction)** compared to PyTorch (522.77 MB Peak RSS).

---

## 4. Retraction of Unbounded Cross-Encoder Superiority Claims

1. **Retracted Claim**: *"Cross-encoders are universally superior to bi-encoders for recruitment matching."*
   - **Correction**: In realistic candidate pools where documents average 1,677 tokens, cross-encoders restricted to standard 512-token limits truncate 99.58% of pairs via `longest_first`, lopping off extensive qualifications or JD requirements.
   - **Verified Ground Truth**: Across matched runs on identical S2 and S3 test partitions over seeds {42, 43, 44}, the bi-encoder statistically significantly outperforms the cross-encoder ($\Delta \text{AUC} = -0.1716$ on S2, $\Delta \text{AUC} = -0.1034$ on S3; 95% range strictly excludes zero).
   - **Bound Directive**: All statements are formally bounded to *"this cross-encoder setup (MiniLM-L6, 512 tokens)"*.
