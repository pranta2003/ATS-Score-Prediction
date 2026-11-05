import json
from pathlib import Path
import pytest


def test_claim_1_c5_cardinality_and_distribution():
    """Validates Claim 1.1 and 1.2: C5 cardinality and class distribution."""
    c5_json = Path("reports/triage_results/c5.json")
    assert c5_json.exists(), "reports/triage_results/c5.json must exist"
    
    with open(c5_json, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    audit = data["audit"]
    assert audit["rows"] == 8000, f"Expected 8000 rows in C5, got {audit['rows']}"
    assert audit["entities"]["unique_resumes"] == 643, f"Expected 643 unique resumes, got {audit['entities']['unique_resumes']}"
    assert audit["entities"]["unique_jds"] == 351, f"Expected 351 unique JDs, got {audit['entities']['unique_jds']}"
    
    classes = audit["label_analysis"]["classes"]
    assert classes.get("No Fit") == 4000, f"Expected 4000 'No Fit', got {classes.get('No Fit')}"
    assert classes.get("Potential Fit") == 2000, f"Expected 2000 'Potential Fit', got {classes.get('Potential Fit')}"
    assert classes.get("Good Fit") == 2000, f"Expected 2000 'Good Fit', got {classes.get('Good Fit')}"


def test_claim_2_generalization_hierarchy():
    """Validates Claim 2.3: Macro ROC-AUC strictly degrades S1 > S2 > S3."""
    results_path = Path("reports/c5_verification_results.json")
    if not results_path.exists():
        pytest.skip("reports/c5_verification_results.json not found")
        
    with open(results_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    auc_s1 = data["scenarios"]["S1_cold_resume"]["macro_roc_auc"]
    auc_s2 = data["scenarios"]["S2_cold_jd"]["macro_roc_auc"]
    auc_s3 = data["scenarios"]["S3_double_cold"]["macro_roc_auc"]
    
    assert auc_s1 > auc_s2 > auc_s3, (
        f"Hierarchy violated: S1 ({auc_s1:.4f}) > S2 ({auc_s2:.4f}) > S3 ({auc_s3:.4f})"
    )
    # S3 ceiling is close to random/majority baseline
    acc_s3 = data["scenarios"]["S3_double_cold"]["accuracy"]
    assert 0.50 <= acc_s3 <= 0.60, f"Expected S3 accuracy around ~54%, got {acc_s3:.4f}"


def test_claim_3_remote_kaggle_specs():
    """Validates Claim 4.1: Remote probe recorded correct specs."""
    probe_path = Path("docs/KAGGLE_PROBE.md")
    assert probe_path.exists(), "docs/KAGGLE_PROBE.md must exist"
    content = probe_path.read_text(encoding="utf-8")
    assert "Tesla T4" in content, "Kaggle probe should record Tesla T4"
    assert "Internet Access" in content or "HTTP 200" in content, "Internet access must be verified"
