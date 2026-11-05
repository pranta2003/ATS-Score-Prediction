# Local Pipeline Timing Breakdown & Bottleneck Analysis

## Benchmark Environment
- **Device**: HP EliteBook 840 G4
- **CPU**: Intel(R) Core(TM) i5-7300U (2 cores, 4 threads @ 3.50 GHz)
- **RAM**: 8.00 GB (86% utilized at baseline)
- **GPU**: Intel HD Graphics 620 (No CUDA)
- **OS**: Windows 10 Pro 22H2 (x86_64)
- **Python**: 3.13.7

---

## Stage Timings (Smoke Test Protocol: 200 Pairs, 3-Class C5, LightGBM)

| Pipeline Stage | Implementation | Time (seconds) | % of Total Runtime | Bottleneck Assessment |
|---|---|---|---|---|
| **Stage 1: Data Loading & Adapter Normalization** | HuggingFace dataset cache + adapter schema mapping | **10.02 s** | 77.4% | Network handshake + Arrow IPC deserialization on cold start. |
| **Stage 2: Pair Feature Construction** | Unique document embedding lookup (1,537-d) | **0.11 s** | 0.8% | **Completely solved.** Index lookup replaced re-embedding 8,000 pairs (previously ~10 min). |
| **Stage 3: Cross-Validation Training** | 5-fold LightGBM (hist, max_depth=3, n_estimators=20) | **0.15 s** | 1.2% | Ultra-fast histogram gradient boosting. |
| **Stage 4: Evaluation & Cluster Bootstrap CIs** | `vectorized_cluster_bootstrap_acc_f1` (matrix multinomial) | **0.02 s** | 0.2% | **Completely solved.** Matrix math replaced O(B*N) dataframe resampling (previously ~180 s). |
| **Stage 5: Serialization** | JSON metrics, timings, and Parquet predictions | **0.05 s** | 0.4% | Minimal disk I/O. |
| **Total Runtime** | End-to-end execution | **12.95 s** | 100.0% | **Comfortably under the 60-second smoke limit.** |

---

## Full Dataset (8,000 Pairs) Local Extrapolation vs. Compute Shift (Kaggle)

| Workload | Local Laptop (Estimated) | Kaggle Cloud CPU / GPU | Decision / Rationale |
|---|---|---|---|
| **Smoke Run (200 pairs, 1 model)** | 12.95 s | N/A | **Local** (unit tests & rapid verification) |
| **Single Full CV (8,000 pairs, LightGBM)** | ~35 s | ~8 s | Local for smoke check only |
| **J1 Full Sweep (S1/S2/S3 × 20 seeds × 6 baselines = 360 models)** | **~3.5 - 4.5 hours** | **~15 - 20 minutes** | **Move to Kaggle (K2 - J1)**: Laptop CPU/RAM throttles under sustained load. |
| **J2 ATS Feature Ablation & Rubric Validation** | ~45 minutes | ~5 minutes | **Move to Kaggle (K2 - J2)** |
| **J3 Cross-Encoder Transformer Fine-Tuning** | **>24 hours (No GPU)** | **~25 minutes (T4 GPU)** | **Mandatory Kaggle GPU (K2 - J3)**: Impossible on laptop CPU. |
