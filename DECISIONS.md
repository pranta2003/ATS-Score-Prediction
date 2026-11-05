# Architectural Decision Records (ADR)

## ADR-001: Separation of Control and Compute via Kaggle Cloud Bridge
- **Context:** The local development workstation is an HP EliteBook 840 G4 (2-core / 4-thread Intel Core i5-7300U, 8 GB RAM, no discrete GPU). Large multi-seed sweeps, transformer fine-tuning, and extensive bootstrap iterations take hours locally and risk out-of-memory crashes.
- **Decision:** Split the execution pipeline strictly into Control (local) and Compute (remote via Kaggle Cloud API). The local environment is reserved for editing, git management, static tests, and `<60s` smoke runs. All jobs expected to exceed 10 minutes run on Kaggle Cloud (4 vCPUs, 31.35 GB RAM, 2x NVIDIA T4 GPUs).
- **Consequences:** Eliminates local CPU bottlenecks and crashes; requires bundle packager and remote execution monitor (`scripts/kaggle_run.py`).

## ADR-002: Unique Document Embedding Caching with Index Lookup
- **Context:** In resume-JD pair datasets, each resume or JD appears in multiple pairs (e.g. in C5, 643 unique resumes and ~384 unique JDs form 2,484 pairs). Re-embedding text on every pair repeatedly consumed ~10 minutes per experiment.
- **Decision:** Hash normalized document texts, compute embeddings once per unique document, and cache them to disk (`cache/embeddings/`). Pair features are assembled dynamically in memory via fast array slicing/index lookup.
- **Consequences:** Reduced pair feature construction time from ~600 seconds to 0.11 seconds (~5,400x speedup), enabling sub-second feature loading.

## ADR-003: Pre-Registration of Grouped Evaluation Protocols (S1, S2, S3)
- **Context:** In recruitment datasets, standard random train/test splits inadvertently place the same resumes or job descriptions in both train and test partitions. Models exploit candidate or employer specific priors rather than learning true semantic compatibility.
- **Decision:** Pre-register three strict evaluation scenarios:
  1. **S1 (Cold Resume):** Resumes are strictly held out; JDs may overlap. Measures ability to rank unseen candidates for existing roles.
  2. **S2 (Cold JD):** JDs are strictly held out; resumes may overlap. Measures ability to match candidates to newly opened job listings.
  3. **S3 (Double Cold):** Both resumes and JDs in the test split are completely disjoint from the training split. Measures true, uncheatable generalization.
- **Consequences:** Prevents memorization leaks and establishes an honest generalization ceiling (~54% accuracy on S3 vs random baseline ~52.5%).

## ADR-004: Selection of C5 as Primary Empirical Dataset
- **Context:** Multiple public ATS datasets exist with varying provenance and label noise. Control (`0xnbk`) suffers from post-hoc synthetic continuous mapping; C1 (`harsh72877287`) and C3 (`rahulrajpu676`) are synthetic/deterministic mappings; C2 (`netsol`) is LLM-derived rubric scores.
- **Decision:** Adopt C5 (`cnamuangtoun/resume-job-description-fit`) as the primary ground-truth dataset for classification benchmarking, while retaining C2 for LLM rubric alignment and Control for historical comparison.
- **Consequences:** Focuses model validation on authentic recruiter-labeled pairs while documenting exact dataset limitations honestly.

## ADR-005: Vectorized Cluster Bootstrap for Uncertainty Quantification
- **Context:** Evaluating performance confidence intervals across clusters (e.g., grouping by resume or JD) using standard iterative bootstrap required 1,000 passes and took ~180 seconds.
- **Decision:** Implement matrix multinomial sampling to generate cluster weights in a single vectorized numpy operation (`src/ats/evaluation.py`).
- **Consequences:** Accelerated cluster bootstrap CI calculation from 180 seconds to 0.02 seconds (~9,000x speedup) with zero loss of statistical precision.

