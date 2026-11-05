# ATS Project Task Board

## Status Overview
- **Phase 1 (Diagnosis & Audit):** COMPLETED
- **Phase 2 (Dataset Triage & Adapter Layer):** COMPLETED
- **Phase 3 (Kaggle Cloud Scaling & Rigorous Verification):** IN PROGRESS

---

## Task Breakdown

### Completed (Done)
- [x] **Audit & Baseline Diagnosis (Phase 1):** Identified label distribution collapse, regression target discreteness, and test leakage in initial capstone.
- [x] **WP-D0 Dataset Adapter Layer:** Unified schema (`resume_id`, `jd_id`, `resume_text`, `jd_text`, `label_continuous`, `label_class`, `source_dataset`) across candidate datasets (`src/ats/adapters.py`).
- [x] **WP-D1 Candidate Dataset Triage:** Audited candidates C1 through C5 alongside Control; documented in `reports/DATASET_TRIAGE.md`.
- [x] **C5 Verification Gate (Initial):** Established S1, S2, and S3 protocols; proved double-cold baseline ceiling on C5.
- [x] **K0 Fast Experiment Runner:** Unique-document caching (`src/ats/embeddings.py`), vectorized cluster bootstrap (`src/ats/evaluation.py`), and `--smoke` runner (`scripts/run_experiment.py`) executing in <15s. (PR #5).
- [x] **K1 Kaggle Bridge:** Automated packaging, deployment, monitoring, and validation harness for Kaggle Cloud CPU/GPU execution (`scripts/kaggle_run.py`, `docs/KAGGLE_PROBE.md`, PR #6).
- [x] **Initial Judge Packet:** Automated markdown compilation script (`scripts/make_judge_packet.py`) generating `reports/JUDGE_PACKET.md` (PR #7).

### In Progress
- [ ] **J1 Benchmark Sweep (Kaggle CPU):** 20 seeds across S1, S2, S3 with 6 baseline models (Majority, JD-prior, Resume-prior, Logistic Regression, LightGBM, XGBoost) and 95% cluster-bootstrap CIs. (Kernel `ats-c5-scen-mod-81b51c15` currently running on Kaggle).
- [ ] **Update C5 Verification & Merge PR #4:** Update `reports/C5_VERIFICATION.md` with 20-seed empirical numbers and merge into `main`.

### Up Next (K2 & K3)
- [ ] **J2 Feature Ablations & Rubric Benchmark (Kaggle CPU):** C2 (`netsol/resume-score-details`) GPT-4o rubric imitation and feature importance analysis.
- [ ] **J3 Cross-Encoder Transformer Pilot (Kaggle GPU):** Fine-tuning pilot with `cross-encoder/ms-marco-MiniLM-L-6-v2` on Kaggle 2x T4 GPUs for S2/S3.
- [ ] **K3 Automated Verification & Phase 3 Report:**
  - Automated tests: `tests/test_claims.py` and `tests/test_judge_packet.py`.
  - Comprehensive final report: `reports/PHASE3_REPORT.md`.

