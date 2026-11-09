# CI Diagnosis and Failure Analysis

This document catalogs and diagnoses GitHub Actions workflow runs across all open pull requests (`#17` through `#30`), recording the latest run per open PR, identifying historical failing steps with verbatim logs, and establishing lazy import protections.

---

## 1. Latest Workflow Run Per Open PR

The table below reflects the latest workflow run executed for each open pull request as of November 8, 2026:

| PR | Branch Name | Latest Run ID | UTC Start Time | Conclusion | Workflow File Version Used |
|:---:|---|:---:|:---:|:---:|---|
| **#17** | `fix/render-startup` | `42333148024` | 2026-11-07T03:15:18Z | **success** | Legacy (`python-app.yml`, Python 3.10) |
| **#18** | `fix/missing-statistics` | `42338113489` | 2026-11-07T04:21:44Z | **success** | Legacy (`python-app.yml`, Python 3.10) |
| **#19** | `fix/t3-investigation` | `42338660656` | 2026-11-07T04:29:15Z | **success** | Legacy (`python-app.yml`, Python 3.10) |
| **#20** | `chore/kaggle-organization` | `42339047141` | 2026-11-07T04:34:28Z | **success** | Legacy (`python-app.yml`, Python 3.10) |
| **#21** | `feat/rubric-scorer` | `42339139836` | 2026-11-07T04:35:43Z | **success** | Legacy (`python-app.yml`, Python 3.10) |
| **#22** | `feat/app-v2-qa` | `42339295921` | 2026-11-07T04:37:58Z | **success** | Legacy (`python-app.yml`, Python 3.10) |
| **#23** | `docs/paper-package` | `42339477972` | 2026-11-07T04:40:37Z | **success** | Legacy (`python-app.yml`, Python 3.10) |
| **#24** | `docs/final-reconciliation` | `42340099413` | 2026-11-07T04:49:26Z | **success** | Legacy (`python-app.yml`, Python 3.10) |
| **#25** | `fix/kaggle-runs` | `42368352279` | 2026-11-07T11:28:59Z | **success** | Legacy (`python-app.yml`, Python 3.10) |
| **#26** | `fix/packet-provenance` | `42368711788` | 2026-11-07T11:33:57Z | **success** | Legacy (`python-app.yml`, Python 3.10) |
| **#27** | `feat/app-final-model` | `42487115669` | 2026-11-08T13:53:42Z | **success** | Modern (`python-app.yml`, Python 3.13, `-m "not heavy"`) |
| **#28** | `fix/rubric-validation` | `42369831407` | 2026-11-07T11:50:00Z | **success** | Legacy (`python-app.yml`, Python 3.10) |
| **#29** | `docs/paper-outline-fix` | `42475415783` | 2026-11-08T11:21:50Z | **success** | Legacy (`python-app.yml`, Python 3.10) |
| **#30** | `release/final-package` | `42487088817` | 2026-11-08T13:53:22Z | **success** | Modern (`python-app.yml`, Python 3.13, `-m "not heavy"`) |

---

## 2. Forensic Diagnosis of Historical Failures

### Incident: 4 Collection Errors ("No module named 'datasets'") on Python 3.13.18
- **Matching Runs:** 
  - `42478336124` on `release/final-package` (PR #30, started 2026-11-08T12:01:57Z).
  - `42478324834` on `feat/app-final-model` (PR #27, started 2026-11-08T12:01:48Z).
- **Environment:** Linux Ubuntu runner, Python `3.13.18`, pytest `9.2.0`.
- **Command:** `pytest -m "not heavy" -v`.
- **Verbatim Error:**
  ```
  scripts/run_experiment.py:24: in <module>
      from src.ats.adapters.registry import get_adapter
  src/ats/adapters/__init__.py:2: in <module>
      from .control import ControlAdapter
  src/ats/adapters/control.py:3: in <module>
      from datasets import load_dataset
  E   ModuleNotFoundError: No module named 'datasets'
  !!!!!!!!!!!!!!!!!!! Interrupted: 4 errors during collection !!!!!!!!!!!!!!!!!!!
  ============================== 4 errors in 3.55s ===============================
  ```
- **Did Later Runs on the Same Branch Pass?**
  - **YES.** 
    - On `release/final-package` (PR #30): Run `42478561190` (started 12:04:54Z) passed in 1m 46s, Run `42479575012` (started 12:17:55Z) passed in 1m 30s, and Run `42479606507` (started 12:18:18Z) passed in 1m 33s.
    - On `feat/app-final-model` (PR #27): Run `42479301096` (started 12:14:29Z) passed in 1m 37s.
- **Root Cause & Permanent Fix:**
  - *Immediate fix applied in commits `32c5206` / `bdd1a1d`:* Added `datasets` and `lightgbm` to `requirements-dev.txt`.
  - *Permanent architectural hardening applied in Step 0b:* Refactored all adapter classes (`src/ats/adapters/control.py`, `c1_fit.py`, `c5_fit_original.py`, `c2_details.py`, `c3_matching.py`) and embedding utilities (`src/ats/embeddings.py`) to import heavy libraries lazily inside methods (`load_raw()`, `build_and_cache_document_embeddings()`), guarded test modules with `pytest.importorskip("datasets")` and `pytest.importorskip("yaml")`, and retained `datasets` and `lightgbm` in `requirements-dev.txt`.
  - *Proof:* Executed in a fresh Python 3.13 virtual environment without `datasets`: `pytest -m "not heavy" --collect-only` collected 29 selected tests across 41 items with **0 errors**. When `datasets` was installed, full execution passed 34 selected tests in **4.95 seconds** with **0 failures**.

---

## 3. Incident Log: Additional Prior Failures

### Incident A: SciPy Wheel Resolution on Python 3.10 (Run 42476260925 on PR #27)
- **Failing Step:** `Install dependencies` (failed at 12s, 2026-11-08T11:33:16Z).
- **Verbatim Error:** `ERROR: Could not find a version that satisfies the requirement scipy==1.16.3 ... ERROR: No matching distribution found for scipy==1.16.3`.
- **Root Cause:** Runner executed on hardcoded Python 3.10 while `requirements.txt` required `scipy==1.16.3` (which requires Python >= 3.11).
- **Remediation:** Configured `python-version-file: ".python-version"` (Python 3.13) in `.github/workflows/python-app.yml`.

### Incident B: Unsynchronized Fixtures on App Branch (Run 42479011425 on PR #27)
- **Failing Step:** `Run fast unit tests` (failed at 1m 25s, 2026-11-08T12:12:08Z).
- **Verbatim Error:** `FAILED tests/test_claims.py::test_no_banned_phrases_in_outline_or_packet ... FAILED tests/test_rubric_validation.py::test_rubric_validation_results_file`.
- **Root Cause:** PR #27 branched before `release/final-package` consolidated the updated paper outline and generated `reports/rubric_validation_results.json`.
- **Remediation:** Merged `release/final-package` into `feat/app-final-model`. Subsequent run `42479301096` passed in 1m 37s.
