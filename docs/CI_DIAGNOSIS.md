# CI Diagnosis and Failure Analysis

This document catalogs and diagnoses workflow run failures in GitHub Actions across all open pull requests (`#17` through `#30`), identifying failing steps, verbatim error logs, and root cause classifications.

---

## 1. Summary of PR Checks (Status Audit)

| PR | Branch | Status | Check Run Duration | Workflow Run ID |
|---|---|---|---|---|
| #17 | `fix/render-startup` | PASS | 1m 25s | `42333148024` |
| #18 | `fix/missing-statistics` | PASS | 1m 23s | `42338113489` |
| #19 | `fix/t3-investigation` | PASS | 1m 12s | `42338660656` |
| #20 | `chore/kaggle-organization` | PASS | 1m 20s | `42339047141` |
| #21 | `feat/rubric-scorer` | PASS | 1m 11s | `42339139836` |
| #22 | `feat/app-v2-qa` | PASS | 1m 22s | `42339295921` |
| #23 | `docs/paper-package` | PASS | 1m 29s | `42339477972` |
| #24 | `docs/final-reconciliation` | PASS | 57s | `42340099413` |
| #25 | `fix/kaggle-runs` | PASS | 1m 04s | `42368352279` |
| #26 | `fix/packet-provenance` | PASS | 57s | `42368711788` |
| #27 | `feat/app-final-model` | PENDING / RE-RUN | In Progress | `42478324834` |
| #28 | `fix/rubric-validation` | PASS | 1m 05s | `42369831407` |
| #29 | `docs/paper-outline-fix` | PASS | 1m 21s | `42475415783` |
| #30 | `release/final-package` | PENDING / RE-RUN | In Progress | `42478336124` |

---

## 2. Detailed Incident Log

### Incident 1: Workflow Run 42476260925 (PR #27)
- **Pull Request:** #27
- **Branch:** `feat/app-final-model`
- **Workflow Name:** `Python application`
- **Failing Job:** `build`
- **Failing Step:** `Install dependencies`
- **Run Duration:** 14s (failed at 12s)
- **Timestamp:** 2026-11-08T11:33:16Z
- **Root Cause Classification:** `(i) dependency install / Python version`

#### Last 15 Error Lines Verbatim
```
build	Install dependencies	2026-11-08T11:33:28.1237647Z Collecting scikit-learn (from -r requirements.txt (line 6))
build	Install dependencies	2026-11-08T11:33:28.1311737Z   Downloading scikit_learn-1.7.2-cp310-cp310-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (11 kB)
build	Install dependencies	2026-11-08T11:33:28.1831274Z Collecting lightgbm==4.6.0 (from -r requirements.txt (line 7))
build	Install dependencies	2026-11-08T11:33:28.1903093Z   Downloading lightgbm-4.6.0-py3-none-manylinux_2_28_x86_64.whl.metadata (17 kB)
build	Install dependencies	2026-11-08T11:33:28.3201352Z ERROR: Ignored the following yanked versions: 1.11.0, 1.14.0rc1
build	Install dependencies	2026-11-08T11:33:28.3207409Z ERROR: Ignored the following versions that require a different python version: 1.16.0 Requires-Python >=3.11; 1.16.0rc1 Requires-Python >=3.11; 1.16.0rc2 Requires-Python >=3.11; 1.16.1 Requires-Python >=3.11; 1.16.2 Requires-Python >=3.11; 1.16.3 Requires-Python >=3.11; 1.17.0 Requires-Python >=3.11; 1.17.0rc1 Requires-Python >=3.11; 1.17.0rc2 Requires-Python >=3.11; 1.17.1 Requires-Python >=3.11; 1.18.0 Requires-Python >=3.12; 1.18.0rc1 Requires-Python >=3.12; 1.18.0rc2 Requires-Python >=3.12; 1.18.1 Requires-Python >=3.12; 1.6.2 Requires-Python >=3.7,<3.10; 1.6.3 Requires-Python >=3.7,<3.10; 1.7.0 Requires-Python >=3.7,<3.10; 1.7.1 Requires-Python >=3.7,<3.10; 1.8.0 Requires-Python >=3.11; 1.8.0rc1 Requires-Python >=3.11; 1.9.0 Requires-Python >=3.11; 1.9.0rc1 Requires-Python >=3.11; 1.9.1 Requires-Python >=3.11; 2.3.0 Requires-Python >=3.11; 2.3.1 Requires-Python >=3.11; 2.3.2 Requires-Python >=3.11; 2.3.3 Requires-Python >=3.11; 2.3.4 Requires-Python >=3.11; 2.3.5 Requires-Python >=3.11; 2.4.0 Requires-Python >=3.11; 2.4.0rc1 Requires-Python >=3.11; 2.4.1 Requires-Python >=3.11; 2.4.2 Requires-Python >=3.11; 2.4.3 Requires-Python >=3.11; 2.4.4 Requires-Python >=3.11; 2.4.5 Requires-Python >=3.11; 2.4.6 Requires-Python >=3.11; 2.5.0 Requires-Python >=3.12; 2.5.0rc1 Requires-Python >=3.12; 2.5.1 Requires-Python >=3.12; 2.5.2 Requires-Python >=3.12; 2.5.3 Requires-Python >=3.12; 2.5.4 Requires-Python >=3.12; 3.3.0 Requires-Python >=3.12; 3.4.0 Requires-Python >=3.12; 3.4.1 Requires-Python >=3.12
build	Install dependencies	2026-11-08T11:33:28.3215789Z ERROR: Could not find a version that satisfies the requirement scipy==1.16.3 (from versions: 0.8.0, 0.9.0, 0.10.0, 0.10.1, 0.11.0, 0.12.0, 0.12.1, 0.13.0, 0.13.1, 0.13.2, 0.13.3, 0.14.0, 0.14.1, 0.15.0, 0.15.1, 0.16.0, 0.16.1, 0.17.0, 0.17.1, 0.18.0, 0.18.1, 0.19.0, 0.19.1, 1.0.0, 1.0.1, 1.1.0, 1.2.0, 1.2.1, 1.2.2, 1.2.3, 1.3.0, 1.3.1, 1.3.2, 1.3.3, 1.4.0, 1.4.1, 1.5.0, 1.5.1, 1.5.2, 1.5.3, 1.5.4, 1.6.0, 1.6.1, 1.7.2, 1.7.3, 1.8.0rc1, 1.8.0rc2, 1.8.0rc3, 1.8.0rc4, 1.8.0, 1.8.1, 1.9.0rc1, 1.9.0rc2, 1.9.0rc3, 1.9.0, 1.9.1, 1.9.2, 1.9.3, 1.10.0rc1, 1.10.0rc2, 1.10.0, 1.10.1, 1.11.0rc1, 1.11.0rc2, 1.11.1, 1.11.2, 1.11.3, 1.11.4, 1.12.0rc1, 1.12.0rc2, 1.12.0, 1.13.0rc1, 1.13.0, 1.13.1, 1.14.0rc2, 1.14.0, 1.14.1, 1.15.0rc1, 1.15.0rc2, 1.15.0, 1.15.1, 1.15.2, 1.15.3)
build	Install dependencies	2026-11-08T11:33:28.3218364Z ERROR: No matching distribution found for scipy==1.16.3
build	Install dependencies	2026-11-08T11:33:28.3824266Z ##[error]Process completed with exit code 1.
```

#### Diagnostic Explanation
The repository specifies Python `3.13` in `.python-version`. However, `.github/workflows/python-app.yml` configured `actions/setup-python@v5` with hardcoded `python-version: "3.10"`.
When `pip install -r requirements.txt` executed on PR #27, it attempted to resolve `scipy==1.16.3`. Per PyPI release metadata, SciPy 1.16 requires `Python >= 3.11`. Under Python 3.10, `pip` rejected all 1.16.x wheels and failed.

---

### Incident 2: Workflow Runs 42478324834 (PR #27) and 42478336124 (PR #30)
- **Pull Request:** #27 (`feat/app-final-model`) and #30 (`release/final-package`)
- **Workflow Name:** `Python application`
- **Failing Job:** `build`
- **Failing Step:** `Run fast unit tests`
- **Run Duration:** 1m 11s (failed at test collection)
- **Timestamp:** 2026-11-08T12:02:59Z
- **Root Cause Classification:** `(ii) test needs data, network, Kaggle login, cache, or model file that CI lacks` (dev dependencies missing during test module collection)

#### Last 15 Error Lines Verbatim
```
build	Run fast unit tests	2026-11-08T12:02:59.6224029Z scripts/run_experiment.py:24: in <module>
build	Run fast unit tests	2026-11-08T12:02:59.6224244Z     from src.ats.adapters.registry import get_adapter
build	Run fast unit tests	2026-11-08T12:02:59.6224460Z src/ats/adapters/__init__.py:2: in <module>
build	Run fast unit tests	2026-11-08T12:02:59.6224651Z     from .control import ControlAdapter
build	Run fast unit tests	2026-11-08T12:02:59.6224836Z src/ats/adapters/control.py:3: in <module>
build	Run fast unit tests	2026-11-08T12:02:59.6225017Z     from datasets import load_dataset
build	Run fast unit tests	2026-11-08T12:02:59.6225211Z E   ModuleNotFoundError: No module named 'datasets'
build	Run fast unit tests	2026-11-08T12:02:59.6229426Z =========================== short test summary info ============================
build	Run fast unit tests	2026-11-08T12:02:59.6229688Z ERROR tests/test_adapters.py
build	Run fast unit tests	2026-11-08T12:02:59.6230006Z ERROR tests/test_feature_parity.py
build	Run fast unit tests	2026-11-08T12:02:59.6230188Z ERROR tests/test_kaggle_bridge.py
build	Run fast unit tests	2026-11-08T12:02:59.6230362Z ERROR tests/test_t3_features.py
build	Run fast unit tests	2026-11-08T12:02:59.6230574Z !!!!!!!!!!!!!!!!!!! Interrupted: 4 errors during collection !!!!!!!!!!!!!!!!!!!!
build	Run fast unit tests	2026-11-08T12:02:59.6230835Z ============================== 4 errors in 3.55s ===============================
build	Run fast unit tests	2026-11-08T12:03:00.2081674Z ##[error]Process completed with exit code 2.
```

#### Diagnostic Explanation
Pytest imports all `test_*.py` files during the collection phase before filtering out tests marked with `@pytest.mark.heavy`. Test modules such as `test_adapters.py`, `test_feature_parity.py`, `test_kaggle_bridge.py`, and `test_t3_features.py` import `src.ats.adapters`, which imports `datasets` (`from datasets import load_dataset`) and `lightgbm`.
Because `datasets` and `lightgbm` were omitted from `requirements-dev.txt`, the CI runner failed with `ModuleNotFoundError: No module named 'datasets'` at collection time.

---

## 3. Remediation Architecture

To align CI with production standards and ensure green builds across all PRs:
1. **Dynamic Python Versioning:** `.github/workflows/python-app.yml` references `.python-version` via `python-version-file: ".python-version"` (Python 3.13).
2. **Dependency Management:** Pip dependency caching is enabled (`cache: "pip"`), and `requirements-dev.txt` is installed alongside `requirements.txt`, specifying `pytest`, `flake8`, `datasets`, and `lightgbm`.
3. **Execution Guardrails:** `timeout-minutes: 10` is added to prevent stalled jobs.
4. **Flake8 Linting Filter:** Linting runs `flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics` so syntax and logic fatal errors fail the build without tripping on stylistic warnings.
5. **Fast Test Suite Separation:** Tests requiring Hugging Face downloads, caches, Kaggle credentials, model files, or long execution are marked with `@pytest.mark.heavy`. CI automatically filters these via `pytest -m "not heavy"`.
6. **Heavy Test Automation:** A separate manual dispatch workflow (`.github/workflows/heavy-tests.yml`) is provided for deep integration testing on demand.
7. **Graceful Skipped Fixtures:** Tests that read `results/` or `reports/` check for file existence and invoke `pytest.skip` rather than erroring out in environments without pre-computed artifacts.
