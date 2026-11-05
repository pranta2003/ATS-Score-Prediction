# ATS Project: Agent & Engineering Operating Rules

## 1. Environment & Architecture: Control vs. Compute
- **Control Machine (Laptop):**
  - Specs: Intel Core i5-7300U (2 cores, 4 threads), 8 GB RAM, no discrete GPU, Windows 10.
  - Permitted tasks: code editing, static analysis, unit tests, `--smoke` runs (<60s on tiny data, e.g. $\le 200$ samples), git operations, report writing.
  - Prohibited tasks: embedding full datasets, multi-seed sweeps, hyperparameter optimization, deep learning model training.
- **Compute Machine (Kaggle Cloud):**
  - Specs: 4 vCPUs, 31.35 GB RAM, 2 $\times$ NVIDIA Tesla T4 GPUs (16 GB VRAM each), full internet access.
  - Permitted tasks: all sweeps, multi-seed benchmarks (J1), feature ablations (J2), transformer fine-tuning / cross-encoders (J3), Optuna searches (J4).
  - Protocol: code packaged via `scripts/kaggle_run.py`, executed asynchronously, results downloaded to `results/` and verified.

## 2. Waiting Rule & Resource Polling
- When awaiting asynchronous Kaggle jobs or long operations, never poll in a tight loop or spam status checks.
- Use sparse polling: sleep/wait for minutes between status checks.
- Read only relevant log lines (e.g. tail last 20 lines) when checking progress.

## 3. Empirical Honesty & Scientific Rigor
- **Zero Hallucinated Metrics:** Every metric, score, count, or p-value in any report or table must derive directly from executed code and reproducible result files (`results/*.json`, `reports/*.json`).
- **Pre-Registration:** Evaluation protocols, split strategies, metric definitions (Macro-F1, Macro ROC-AUC OVR, Balanced Accuracy), and continuous bin cutoffs (<40, 40–70, >70) must be pre-registered before running models.
- **Leakage Prevention:**
  - Standard random splits leak JD and resume entity identifiers, producing artificially inflated metrics.
  - Primary benchmarks must evaluate on grouped splits: S1 (Cold Resume), S2 (Cold Job Description), and S3 (Double Cold: disjoint resumes and disjoint JDs).
- **Uncertainty Quantification:** Always report 95% cluster-bootstrap confidence intervals (clustered by resume or JD group) rather than point estimates alone.

## 4. Code & Data Integrity
- Never modify or overwrite raw source datasets.
- Cache computationally expensive representations (document embeddings) by text hash to avoid redundant computation.
- Never commit private API credentials or large binary caches to Git. Small result metadata and summary JSONs must be committed.

