# Academic Research Paper Outline & Publication Package

**Title:** Unmasking Leakage, Generalization Barriers, and Evaluation Baselines in Automated Resume Screening (ATS)  
**Authors:** Senior ML Engineer & Technical Research Team  
**Artifact Venue:** Empirical Machine Learning / NLP in Recruitment Systems  
**Date:** November 7, 2026  

---

## 1. Abstract

Automated Applicant Tracking Systems (ATS) increasingly employ pre-trained language models to rank candidate resumes against job descriptions (JDs). Prior literature and open-source benchmarks routinely report regression fits of $R^2 \approx 0.40$ or classification accuracies approaching 90% without examining whether models learn genuine alignment semantics or exploit task-irrelevant base rates. In this work, we conduct an exhaustive forensic audit of standard ATS benchmarks (`0xnbk/resume-ats-score-v1-en`, `cnamuangtoun/resume-job-description-fit`, and candidate corpus variants) spanning 8,000 resume-job pairs. 

We reveal four structural pathologies:
1. **Data Leakage and Split Corruption**: Random cross-validation leaks identical resumes and job descriptions across splits, inflating accuracy by up to 18 percentage points over cold evaluation.
2. **Artificial Target Artifacts**: Arbitrary continuous scores mapped across discrete recruiter labels introduce noise that collapses regression $R^2$ to 0.29–0.43, whereas direct multi-class classification achieves 65.7% accuracy.
3. **The Generalization Cliff**: Under rigorous cold-entity protocols, performance drops monotonically from **65.66%** (S1: Cold Resume) to **55.99%** (S2: Cold JD) and **51.66%** (S3: Double-Cold) against a 50.00% majority baseline.
4. **Bi-Encoder Dominance under Length Constraints**: In real-world recruitment pairs where 99.58% exceed standard sequence boundaries (mean length 1,677 tokens), bi-encoders with chunk-level representations statistically significantly outperform 512-token cross-encoders ($\Delta \text{AUC} = +0.1716$, CI excludes zero).

We provide an open-source reproduction suite, interpretable rubric baselines, and temperature-calibrated serving guidelines.

---

## 2. Paper Structure & Section Blueprint

### Section 1: Introduction
- Motivation: Proliferation of LLM-based and neural ATS matchers in recruitment pipelines.
- The 90% Accuracy Fallacy: Distinguishing authentic multi-class discrimination from non-representative synthetic benchmarks and majority-class guessing.
- Contributions:
  - Complete forensic audit of public ATS benchmark corpora.
  - Formal definition and empirical evaluation of 3 cold-entity generalization regimes: S1 (Cold Resume), S2 (Cold JD), and S3 (Double Cold).
  - Feature ablation quantifying lexical, dense, late-interaction, and rubric signals.
  - Rigorous within-JD ranking evaluation using NDCG@5 and binary good-vs-no-fit separation.

### Section 2: Forensic Diagnosis of Prior Baselines
- Audit of `0xnbk/resume-ats-score-v1-en`:
  - 100 whitespace-normalized duplicate pairs across train and validation sets.
  - Regression binned accuracy (44.98%) vs Direct 3-class classification (65.81%).
  - Separator token (`" SEP "`) artifacts and leakage dynamics.
- Triage across candidate benchmarks C1–C5:
  - Severe JD bottleneck in C1 (only 23 unique JDs across 2,384 pairs).
  - Selection of C5 (`cnamuangtoun/resume-job-description-fit`) as primary benchmark (643 resumes, 351 JDs, 8,000 pairs).

### Section 3: Generalization Regimes & Experimental Setup
- Mathematical formulations of GroupKFold partitions:
  - S1 (Grouping on `resume_id`).
  - S2 (Grouping on `jd_id`).
  - S3 (Strictly disjoint bipartition of both resumes and JDs over 20 random seeds).
- Statistical Significance: Cluster-bootstrap confidence intervals (1,000 resamples) with paired hypothesis testing.
- Remote Kaggle Execution Framework: Zero-drift execution scripts and caching architecture.

### Section 4: Results & Empirical Findings
- **Generalization Hierarchy**: Fig 1 presentation demonstrating the drop from S1 (65.66%) to S2 (55.99%) and S3 (51.66%).
- **Feature Tier Ablation**: Fig 2 presentation comparing T1 (Lexical), T2 (Dense), T3 (Late Interaction), T4 (Lex+Dense), and T5 (Full).
- **Cross-Encoder vs Bi-Encoder Study**: Token truncation analysis (Fig 4) explaining why MiniLM-L6 at 512 tokens fails against bi-encoder late interaction.
- **Within-JD Ranking & Calibration**: NDCG@5 performance (0.7534 on S2) and temperature scaling reducing ECE from 0.1333 to 0.0357 (Fig 3).

### Section 5: Transparent Rule-Based Alternative (Rubric Scorer)
- Why black-box neural scores fail recruiter explainability standards.
- Architecture of the deterministic rubric engine: keyword match with diminishing returns, regex experience extraction, degree hierarchy, and seniority scoring.

### Section 6: Discussion, Ethical Considerations & Deployment
- Mitigating demographic bias: name/pronoun perturbation invariance testing.
- Low-memory deployment: ONNX Runtime vs PyTorch on resource-constrained containers.
- Limitations: Label provenance undocumented in public datasets; ceiling imposed by human recruiter inter-annotator disagreement (~25–35%).

---

## 3. Publication Figures Reference

| Figure | Source File | Description |
|:---:|---|---|
| **Figure 1** | `reports/figures/fig1_generalization_hierarchy.png` | Accuracy across S1, S2, S3 with 95% cluster-bootstrap error bars vs majority baseline. |
| **Figure 2** | `reports/figures/fig2_feature_ablation.png` | Grouped bar chart comparing T1 through T5 on S1 and S2 scenarios. |
| **Figure 3** | `reports/figures/fig3_calibration_reliability.png` | 10-bin reliability diagram before and after temperature scaling on S2. |
| **Figure 4** | `reports/figures/fig4_token_truncation_distribution.png` | Token length histogram showing 99.58% truncation beyond 512 tokens. |
