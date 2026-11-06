# ATS Project Task Board

## Status Overview
- **Phase 1 (Diagnosis & Audit):** COMPLETED
- **Phase 2 (Dataset Triage & Adapter Layer):** COMPLETED
- **Phase 3 (Kaggle Cloud Scaling & Rigorous Verification):** COMPLETED (100%)

---

## Task Breakdown

### Completed (Done)
- [x] **Audit & Baseline Diagnosis (Phase 1):** Identified label distribution collapse, regression target discreteness, and test leakage in initial capstone.
- [x] **WP-D0 Dataset Adapter Layer:** Unified schema (`resume_id`, `jd_id`, `resume_text`, `jd_text`, `label_continuous`, `label_class`, `source_dataset`) across candidate datasets (`src/ats/adapters.py`).
- [x] **WP-D1 Candidate Dataset Triage:** Audited candidates C1 through C5 alongside Control; documented in `reports/DATASET_TRIAGE.md` (PR #3).
- [x] **C5 Verification Gate (Initial):** Established S1, S2, and S3 protocols; proved double-cold baseline ceiling on C5 (PR #4).
- [x] **K0 Fast Experiment Runner:** Unique-document caching (`src/ats/embeddings.py`), vectorized cluster bootstrap (`src/ats/evaluation.py`), and `--smoke` runner (`scripts/run_experiment.py`) executing in <15s (PR #5).
- [x] **K1 Kaggle Bridge:** Automated packaging, deployment, monitoring, and validation harness for Kaggle Cloud CPU/GPU execution (`scripts/kaggle_run.py`, `docs/KAGGLE_PROBE.md`, PR #6).
- [x] **J1 Benchmark Sweep (Kaggle CPU):** 20 seeds across S1, S2, S3 with 6 baseline models (Majority, JD-prior, Resume-prior, Logistic Regression, LightGBM, XGBoost) and 95% cluster-bootstrap CIs (`reports/C5_VERIFICATION.md`, `reports/j1_sweep_results.json`, PR #4).
- [x] **J2 Feature Ablations & Rubric Benchmark (Kaggle CPU):** C2 (`netsol/resume-score-details`) GPT-4o rubric imitation and feature importance analysis (`reports/C2_RUBRIC_ABLATION.md`, `reports/c2_ablation_results.json`, PR #8).
- [x] **J3 Cross-Encoder Transformer Pilot & S2/S3 Suite (Kaggle GPU):** Fine-tuning pilot and 3-epoch evaluation with `cross-encoder/ms-marco-MiniLM-L-6-v2` on Kaggle 2x T4 GPUs (`reports/CROSS_ENCODER.md`, `reports/cross_encoder_results.json`, PR #9).
- [x] **K3 Automated Verification & Phase 3 Report:**
  - Automated tests: `tests/test_claims.py` and `tests/test_judge_packet.py` passing (PR #8).
  - Judge packet compilation: `reports/JUDGE_PACKET.md` generated via `scripts/make_judge_packet.py`.
  - Comprehensive final report: `reports/PHASE3_REPORT.md` synthesizing all diagnostic findings, empirical ceilings, and architectural blueprints.
