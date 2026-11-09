import os
import sys
import json
import time
import subprocess
from pathlib import Path
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.heavy
def test_experiment_runner_smoke():
    """Verify that run_experiment.py in --smoke mode finishes in under 60s and produces valid artifacts."""
    config_path = PROJECT_ROOT / "configs" / "smoke_test.yaml"
    script_path = PROJECT_ROOT / "scripts" / "run_experiment.py"

    assert config_path.exists(), f"Config file not found at {config_path}"
    assert script_path.exists(), f"Runner script not found at {script_path}"

    t0 = time.time()
    res = subprocess.run(
        [sys.executable, str(script_path), "--config", str(config_path), "--smoke"],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
    )
    elapsed = time.time() - t0

    assert res.returncode == 0, f"run_experiment.py failed with code {res.returncode}:\nSTDERR:\n{res.stderr}\nSTDOUT:\n{res.stdout}"
    assert elapsed < 60.0, f"Smoke test took {elapsed:.2f}s, exceeding 60s limit!"

    # Locate the output results folder
    results_base = PROJECT_ROOT / "results"
    assert results_base.exists()

    # Find the run directory matching c5_s1_cold_resume_lightgbm
    matching_dirs = [d for d in results_base.glob("c5_s1_cold_resume_lightgbm_*") if d.is_dir()]
    assert len(matching_dirs) > 0, "No result directory generated for smoke test run!"
    
    # Check the newest one
    run_dir = max(matching_dirs, key=lambda p: p.stat().st_mtime)

    # Validate generated artifacts
    for required_file in ["config.yaml", "metrics.json", "preds.parquet", "timings.json", "log.txt"]:
        fpath = run_dir / required_file
        assert fpath.exists(), f"Missing expected artifact {required_file} in {run_dir}"

    with open(run_dir / "metrics.json", "r") as f:
        metrics = json.load(f)
    assert "accuracy" in metrics
    assert "macro_f1" in metrics
    assert 0.0 <= metrics["accuracy"] <= 1.0

    with open(run_dir / "timings.json", "r") as f:
        timings = json.load(f)
    assert "total_sec" in timings
    assert timings["total_sec"] < 60.0

    df_preds = pd.read_parquet(run_dir / "preds.parquet")
    assert len(df_preds) > 0
    for col in ["index", "resume_id", "jd_id", "y_true", "y_pred"]:
        assert col in df_preds.columns

