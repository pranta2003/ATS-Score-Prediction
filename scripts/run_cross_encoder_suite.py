"""Cross-Encoder Training and Evaluation Suite for ATS Score Prediction (J3).

Orchestrates:
1. Pilot run (1 epoch) to verify GPU timing projection (<1 min target).
2. S2 (Cold-JD) 3-epoch fine-tuning and evaluation with cluster-bootstrap CIs.
3. S3 (Double-Cold) 3-epoch fine-tuning and evaluation with cluster-bootstrap CIs.
4. Generates aggregated results in results/cross_encoder/cross_encoder_results.json and reports/cross_encoder_results.json.
"""

import argparse
import json
import sys
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_cross_encoder import run_cross_encoder


def run_cross_encoder_suite(smoke: bool = False, output_dir: str = "results/cross_encoder"):
    print("=" * 70)
    print("   CROSS-ENCODER TRANSFORMER EVALUATION SUITE (J3)")
    print("=" * 70)

    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)
    suite_start = time.time()
    results = {
        "pilot_projection": {},
        "scenarios": {},
    }

    # 1. Pilot Run (1 epoch, S3)
    print("\n[Step 1/3] Running Pilot Run (1 Epoch) to verify timing projection...")
    pilot_metrics = run_cross_encoder(
        scenario="S3",
        epochs=1,
        pilot=True,
        smoke=smoke,
        output_dir=str(out_p / "pilot"),
    )
    # Read timings from pilot
    pilot_timings_path = out_p / "pilot" / "timings.json"
    if pilot_timings_path.exists():
        with open(pilot_timings_path, "r", encoding="utf-8") as f:
            pilot_timings = json.load(f)
            results["pilot_projection"] = pilot_timings.get("pilot_projection", {})
            print(f"[PILOT RESULT] {results['pilot_projection']}")

    # 2. S2 (Cold-JD) Full Run (3 epochs)
    print("\n[Step 2/3] Running S2 (Cold-JD) Cross-Encoder Fine-Tuning (3 Epochs)...")
    s2_metrics = run_cross_encoder(
        scenario="S2",
        epochs=1 if smoke else 3,
        pilot=False,
        smoke=smoke,
        output_dir=str(out_p / "s2_cold_jd"),
    )
    results["scenarios"]["S2_cold_jd"] = s2_metrics

    # 3. S3 (Double-Cold) Full Run (3 epochs)
    print("\n[Step 3/3] Running S3 (Double-Cold) Cross-Encoder Fine-Tuning (3 Epochs)...")
    s3_metrics = run_cross_encoder(
        scenario="S3",
        epochs=1 if smoke else 3,
        pilot=False,
        smoke=smoke,
        output_dir=str(out_p / "s3_double_cold"),
    )
    results["scenarios"]["S3_double_cold"] = s3_metrics

    total_time = time.time() - suite_start
    results["suite_total_time_seconds"] = total_time

    # Save aggregated outputs
    with open(out_p / "cross_encoder_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved suite results to {out_p / 'cross_encoder_results.json'}")

    reports_p = PROJECT_ROOT / "reports"
    reports_p.mkdir(parents=True, exist_ok=True)
    with open(reports_p / "cross_encoder_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved suite results to {reports_p / 'cross_encoder_results.json'}")

    # Also save top-level metrics.json so Kaggle runner validation passes
    with open(out_p / "metrics.json", "w", encoding="utf-8") as f:
        json.dump({
            "accuracy": s3_metrics.get("accuracy", 0.0),
            "macro_f1": s3_metrics.get("macro_f1", 0.0),
            "macro_roc_auc": s3_metrics.get("macro_roc_auc", 0.0),
            "s2_accuracy": s2_metrics.get("accuracy", 0.0),
            "s2_macro_f1": s2_metrics.get("macro_f1", 0.0),
        }, f, indent=2)

    print("\n" + "=" * 70)
    print("   CROSS-ENCODER SUITE COMPLETED SUCCESSFULLY")
    print(f"   S2 (Cold-JD) Accuracy:      {s2_metrics.get('accuracy', 0.0)*100:.2f}% | AUC: {s2_metrics.get('macro_roc_auc', 0.0):.4f}")
    print(f"   S3 (Double-Cold) Accuracy:  {s3_metrics.get('accuracy', 0.0)*100:.2f}% | AUC: {s3_metrics.get('macro_roc_auc', 0.0):.4f}")
    print(f"   Total Suite Runtime:        {total_time:.2f}s")
    print("=" * 70)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Run rapid smoke test on tiny sample")
    parser.add_argument("--output_dir", default="results/cross_encoder", help="Output directory")
    args = parser.parse_args()

    run_cross_encoder_suite(smoke=args.smoke, output_dir=args.output_dir)
