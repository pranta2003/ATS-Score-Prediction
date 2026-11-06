import sys
import time
import json
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_c5_feature_ablation_smoke():
    """Verify that run_c5_feature_ablation.py in --smoke mode finishes in under 60s."""
    script_path = PROJECT_ROOT / "scripts" / "run_c5_feature_ablation.py"
    assert script_path.exists(), f"Script not found at {script_path}"

    t0 = time.time()
    res = subprocess.run(
        [sys.executable, str(script_path), "--smoke", "--output_dir", "results/smoke_c5_ablation"],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
    )
    elapsed = time.time() - t0

    assert res.returncode == 0, f"run_c5_feature_ablation.py failed with code {res.returncode}:\nSTDERR:\n{res.stderr}\nSTDOUT:\n{res.stdout}"
    assert elapsed < 60.0, f"Smoke test took {elapsed:.2f}s, exceeding 60s limit!"

    metrics_file = PROJECT_ROOT / "results" / "smoke_c5_ablation" / "metrics.json"
    assert metrics_file.exists(), "Smoke test must output metrics.json"
    with open(metrics_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "tiers" in data, "Output must contain 'tiers'"
    for tier in ["T1_Lexical", "T2_Dense", "T3_LateInteraction", "T4_LexicalDense", "T5_Full"]:
        assert tier in data["tiers"], f"Tier {tier} must be present in results"
        assert "S1_cold_resume" in data["tiers"][tier]
        assert "S2_cold_jd" in data["tiers"][tier]
        assert "S3_double_cold" in data["tiers"][tier]
