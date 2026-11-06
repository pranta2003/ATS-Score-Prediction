import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.ats.adapters import get_adapter

print("=== ADAPTER AUDIT FOR C5 & C2 ===")
# C5
c5 = get_adapter("c5").load()
print(f"C5 Rows: {len(c5)}")
print(f"C5 Unique Resumes: {c5['resume_id'].nunique()}")
print(f"C5 Unique JDs: {c5['jd_id'].nunique()}")
print(f"C5 Null Class Labels: {c5['label_class'].isna().sum()}")
print(f"C5 Null Continuous Labels: {c5['label_continuous'].isna().sum()}")

# C2 raw vs filtered
c2_raw = get_adapter("c2").load()
print(f"\nC2 Raw Rows: {len(c2_raw)}")
print(f"C2 Raw Unique Resumes: {c2_raw['resume_id'].nunique()}")
print(f"C2 Raw Unique JDs: {c2_raw['jd_id'].nunique()}")
print(f"C2 Raw Null Continuous: {c2_raw['label_continuous'].isna().sum()}")
print(f"C2 Raw Null Class: {c2_raw['label_class'].isna().sum()}")

# Filter used by run_c2_ablation.py: dropna(subset=['label_continuous'])
c2_valid = c2_raw.dropna(subset=["label_continuous"]).reset_index(drop=True)
print(f"\nC2 Filtered Rows (dropna continuous): {len(c2_valid)}")
print(f"C2 Filtered Unique Resumes: {c2_valid['resume_id'].nunique()}")
print(f"C2 Filtered Unique JDs: {c2_valid['jd_id'].nunique()}")
print(f"Removed Rows: {len(c2_raw) - len(c2_valid)} rows where label_continuous is NaN")

