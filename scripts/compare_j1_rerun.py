import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

with open(PROJECT_ROOT / "results/j1_sweep/metrics.json") as f:
    old = json.load(f)
with open(PROJECT_ROOT / "results/j1_sweep_rerun/metrics.json") as f:
    new = json.load(f)

max_diff = 0.0
print("=== J1 SWEEP RECONSTRUCTED VS RE-RUN COMPARISON ===")
header = f"{'Scenario':<16} {'Model':<12} {'Old Acc':<10} {'New Acc':<10} {'Diff Acc':<10} {'Old F1':<10} {'New F1':<10} {'Diff F1':<10}"
print(header)
print("-" * len(header))

diffs = []

for scen in ["S1_cold_resume", "S2_cold_jd", "S3_double_cold"]:
    for mod in ["majority", "jd_prior", "resume_prior", "logistic", "lightgbm", "xgboost"]:
        old_item = old[scen][mod]
        new_item = new[scen][mod]

        old_acc = old_item["accuracy"] if isinstance(old_item["accuracy"], (int, float)) else old_item["accuracy"]["mean"]
        new_acc = new_item["accuracy"] if isinstance(new_item["accuracy"], (int, float)) else new_item["accuracy"]["mean"]
        diff_acc = abs(old_acc - new_acc)

        old_f1 = old_item["macro_f1"] if isinstance(old_item["macro_f1"], (int, float)) else old_item["macro_f1"]["mean"]
        new_f1 = new_item["macro_f1"] if isinstance(new_item["macro_f1"], (int, float)) else new_item["macro_f1"]["mean"]
        diff_f1 = abs(old_f1 - new_f1)

        max_diff = max(max_diff, diff_acc, diff_f1)
        diffs.append((scen, mod, diff_acc, diff_f1))
        print(f"{scen:<16} {mod:<12} {old_acc:<10.4f} {new_acc:<10.4f} {diff_acc:<10.4f} {old_f1:<10.4f} {new_f1:<10.4f} {diff_f1:<10.4f}")

print("-" * len(header))
print(f"MAX ABSOLUTE DIFFERENCE: {max_diff:.6f} ({max_diff*100:.4f} percentage points)")
if max_diff <= 0.01:
    print("SUCCESS: All differences are <= 1.0 percentage point! Re-run validated.")
else:
    print(f"WARNING: Max difference {max_diff*100:.4f}% exceeds 1.0% limit.")
