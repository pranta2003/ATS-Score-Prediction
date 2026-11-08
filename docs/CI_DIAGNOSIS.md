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
| #27 | `feat/app-final-model` | FAIL | 12s | `42476260925` |
| #28 | `fix/rubric-validation` | PASS | 1m 05s | `42369831407` |
| #29 | `docs/paper-outline-fix` | PASS | 1m 21s | `42475415783` |
| #30 | `release/final-package` | PASS | 1m 10s | `42476398422` |

---

## 2. Detailed Incident Log

### Workflow Run 42476260925
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

## 3. Remediation Architecture

To align CI with production standards and prevent future failures, the following changes are implemented:
1. **Dynamic Python Versioning:** `.github/workflows/python-app.yml` references `.python-version` via `python-version-file: ".python-version"` (Python 3.13).
2. **Dependency Management:** Pip dependency caching is enabled (`cache: "pip"`), and `requirements-dev.txt` is installed alongside `requirements.txt`.
3. **Execution Guardrails:** `timeout-minutes: 10` is added to prevent stalled jobs.
4. **Flake8 Linting Filter:** Linting runs `flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics` so syntax and logic fatal errors fail the build without tripping on stylistic warnings.
5. **Fast Test Suite Separation:** Tests requiring Hugging Face downloads, caches, Kaggle credentials, model files, or long execution are marked with `@pytest.mark.heavy`. CI automatically filters these via `pytest -m "not heavy"`.
6. **Heavy Test Automation:** A separate manual dispatch workflow (`.github/workflows/heavy-tests.yml`) is provided for deep integration testing on demand.
7. **Graceful Skipped Fixtures:** Tests that read `results/` or `reports/` check for file existence and invoke `pytest.skip` rather than erroring out in environments without pre-computed artifacts.
