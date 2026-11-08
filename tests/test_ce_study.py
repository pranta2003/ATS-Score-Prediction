import subprocess
import sys
import time
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.heavy
def test_cross_encoder_study_smoke():
    script_path = PROJECT_ROOT / "scripts" / "run_cross_encoder_study.py"
    t0 = time.time()
    res = subprocess.run(
        [sys.executable, str(script_path), "--smoke", "--output_dir", "results/test_ce_smoke"],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
    )
    elapsed = time.time() - t0
    print(f"Smoke test output:\n{res.stdout}\nErrors:\n{res.stderr}")
    assert res.returncode == 0, f"Smoke test failed: {res.stderr}"
    assert elapsed < 60.0, f"Smoke run exceeded 60s limit: {elapsed:.2f}s"
    assert (PROJECT_ROOT / "results" / "test_ce_smoke" / "study_results.json").exists()

