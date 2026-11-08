import subprocess
import sys
import time
from pathlib import Path
import pytest


@pytest.mark.heavy
def test_c2_ablation_smoke():
    start_time = time.time()
    cmd = [
        sys.executable,
        "scripts/run_c2_ablation.py",
        "--smoke",
        "--output_dir",
        "results/test_smoke_c2",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    duration = time.time() - start_time
    
    # Clean up
    test_dir = Path("results/test_smoke_c2")
    if test_dir.exists():
        import shutil
        shutil.rmtree(test_dir, ignore_errors=True)

    assert res.returncode == 0, f"C2 ablation smoke failed:\n{res.stderr}\n{res.stdout}"
    assert duration < 60.0, f"C2 smoke took {duration:.2f}s, exceeding 60s limit"

