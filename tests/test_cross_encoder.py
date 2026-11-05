import subprocess
import sys
import time
from pathlib import Path


def test_cross_encoder_smoke():
    start_time = time.time()
    cmd = [
        sys.executable,
        "scripts/train_cross_encoder.py",
        "--smoke",
        "--output_dir",
        "results/test_smoke_ce",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    duration = time.time() - start_time
    
    # Clean up test output dir
    test_dir = Path("results/test_smoke_ce")
    if test_dir.exists():
        import shutil
        shutil.rmtree(test_dir, ignore_errors=True)

    assert res.returncode == 0, f"Cross-encoder smoke run failed:\n{res.stderr}\n{res.stdout}"
    assert duration < 60.0, f"Smoke run exceeded 60s target (took {duration:.2f}s)"

