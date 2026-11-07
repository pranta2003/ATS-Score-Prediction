# Academic Research Paper Outline & Publication Package

**Title:** Unmasking Leakage, Generalization Barriers, and Evaluation Baselines in Automated Resume Screening (ATS)  
**Authors:** Pranta Kumer Pandit, Md Abu Nasr Farabi, Mesbaul Hossain Limon  
**Artifact Venue:** Empirical Machine Learning / NLP in Recruitment Systems  
**Date:** November 7, 2026  

---

## 1. Abstract

Automated Applicant Tracking Systems (ATS) increasingly employ pre-trained language models to rank candidate resumes against job descriptions (JDs). Prior literature and open-source benchmarks routinely report regression fits of $R^2 \approx 0.40$ or classification accuracies approaching 90% without examining whether models learn genuine alignment semantics or exploit task-irrelevant base rates. In this work, we conduct an exhaustive forensic audit of standard ATS benchmarks (`0xnbk/resume-ats-score-v1-en`, `cnamuangtoun/resume-job-description-fit`, and candidate corpus variants) spanning 8,000 resume-job pairs. 

We reveal four structural pathologies:
1. **Data Leakage and Split Corruption**: Random cross-validation leaks identical resumes and job descriptions across splits, inflating accuracy by up to 18 percentage points over cold evaluation.
2. **Artificial Target Artifacts**: Arbitrary continuous scores mapped across discrete recruiter labels introduce boundary noise that collapses regression $R^2$ to 0.29–0.43, whereas direct multi-class classification achieves 65.7% accuracy.
3. **The Generalization Cliff**: Under rigorous cold-entity protocols, performance drops monotonically from **66.03%** (S1: Cold Resume) to **55.45%** (S2: Cold JD) and **51.15%–51.66%** (S3: Double-Cold) against a ~49–52% majority baseline.
4. **Bi-Encoder Dominance under Length Constraints**: In real-world recruitment pairs where 99.58% exceed standard sequence boundaries (mean length 1,677 tokens), bi-encoders with chunk-level representations statistically significantly outperform 512-token cross-encoders ($\Delta \text{AUC} = +0.1716$, CI strictly excludes zero).

We provide an open-source reproduction suite, an interpretable decoupled rubric baseline, and temperature-calibrated serving guidelines.

---

## 2. Paper Structure & Section Blueprint

### Section 1: Introduction
- Motivation: Proliferation of neural ATS matchers in talent acquisition pipelines.
- The 90% Accuracy Fallacy: Distinguishing authentic multi-class discrimination from non-representative synthetic benchmarks and majority-class guessing.
- Summary of Contributions:
  - Forensic audit of public ATS benchmark corpora (C1 through C5 and Control).
  - Formal definition and empirical evaluation of 3 cold-entity generalization regimes: S1 (Cold Resume), S2 (Cold JD), and S3 (Double Cold).
  - Feature ablation quantifying lexical, dense, late-interaction, and rubric signals.
  - Rigorous within-JD ranking evaluation using NDCG@5 and binary good-vs-no-fit separation.
  - Decoupled, explainable rule-based rubric baseline.

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
  - S1: Grouping on `resume_id` (JD priors repeat).
  - S2: Grouping on `jd_id` (resume priors repeat).
  - S3: Strictly disjoint bipartition of both resumes and JDs over 20 random seeds.
- Statistical Significance: Cluster-bootstrap confidence intervals (1,000 resamples) with paired hypothesis testing.
- Remote Kaggle Execution Framework: Zero-drift execution scripts, caching architecture, and artifact checksum verification.

### Section 4: Results & Empirical Findings
- **Generalization Hierarchy**: Fig 1 presentation demonstrating the drop from S1 (66.03%) to S2 (55.45%) and S3 (51.15%–51.66%).
- **Feature Tier Ablation**: Fig 2 presentation comparing T1 (Lexical), T2 (Dense), T3 (Late Interaction), T4 (Lex+Dense), and T5 (Full), confirming T2 Dense is the most parsimonious foundation.
- **Cross-Encoder vs Bi-Encoder Study**: Token truncation analysis (Fig 4) explaining why MiniLM-L6 at 512 tokens fails against bi-encoder late interaction.
- **Within-JD Ranking & Calibration**: NDCG@5 performance (0.7534 on S2) and temperature scaling reducing ECE from 0.1333 to 0.0357 (Fig 3).

### Section 5: Transparent Rule-Based Alternative (Rubric Scorer)
- Why black-box neural scores fail recruiter explainability standards.
- Architecture of the deterministic rubric engine: keyword match with diminishing returns, regex experience extraction, degree hierarchy, and seniority scoring.
- Independent Formatting Audit: Decoupling formatting hygiene from candidate qualification match score.
- Empirical Rubric Findings: Surface keyword overlap alone yields Spearman $r \approx -0.017$ and binary AUC $0.481$, confirming dense semantic interaction is necessary for recruiter alignment.

### Section 6: Discussion & Production Deployment
- Mitigating demographic bias: name/pronoun perturbation invariance testing.
- Low-memory deployment: Port-first startup and background loading to prevent cloud container port scan timeouts.
- Memory optimization: Comparing PyTorch serving (522 MB Peak RSS) with ONNX Runtime (243 MB Peak RSS).

---

## 3. Limitations & Threats to Validity

1. **Sequence Truncation in Cross-Encoders**:
   - 99.58% of resume-job pairs in C5 exceed the 512-token limit of standard cross-encoder architectures (`all-MiniLM-L6-v2`), with a mean pair length of 1,677 tokens.
   - Longest-first truncation disproportionately drops text from either the resume or JD. Consequently, findings demonstrating bi-encoder superiority are strictly restricted to *this cross-encoder setup (MiniLM-L6, 512 tokens)*. Long-context models (e.g. 8k+ token cross-encoders) may exhibit different trade-offs.

2. **Dataset Provenance & Recruiter Subjectivity**:
   - The primary dataset C5 lacks detailed annotation guidelines and demographic metadata on annotators.
   - Ground-truth labels reflect inherent recruiter disagreement and subjectivity, which industry literature estimates at 25% to 35% inter-annotator disagreement. Models approaching human consistency on this task cannot achieve 90% multi-class accuracy without overfitting to noise.

3. **Single Pretrained Embedding Backbone**:
   - All dense feature evaluations were grounded on `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions) to maintain strict workstation and free-tier serving feasibility.
   - Larger modern architectures (e.g., 7B parameter instruction-tuned embedding models) were not benchmarked.

4. **Exploratory Scope of Secondary Datasets**:
   - C2 (`netsol/resume-score-details`) contains continuous rubric scores but is constrained by only 26 unique job descriptions among valid continuous rows, limiting generalizability.

---

## 4. Publication Figures Reference

| Figure | Source File | Description |
|:---:|---|---|
| **Figure 1** | `reports/figures/fig1_generalization_hierarchy.png` | Accuracy across S1, S2, S3 with 95% cluster-bootstrap error bars vs majority baseline. |
| **Figure 2** | `reports/figures/fig2_feature_ablation.png` | Grouped bar chart comparing T1 through T5 on S1 and S2 scenarios. |
| **Figure 3** | `reports/figures/fig3_calibration_reliability.png` | 10-bin reliability diagram before and after temperature scaling on S2. |
| **Figure 4** | `reports/figures/fig4_token_truncation_distribution.png` | Token length histogram showing 99.58% truncation beyond 512 tokens. |
