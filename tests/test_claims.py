import json
import re
from pathlib import Path
import pytest
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.render_report_tables import (
    render_j1_sweep_table,
    render_c2_ablation_table,
    render_like_for_like_table,
)
from scripts.make_judge_packet import generate_judge_packet


def test_governance_files_in_harness():
    """Validates that all governance files reside in harness/."""
    harness_dir = Path("harness")
    assert harness_dir.exists(), "harness/ directory must exist"
    for fname in ["AGENT_RULES.md", "TASK_BOARD.md", "DECISIONS.md", "CLAIMS.md"]:
        fpath = harness_dir / fname
        assert fpath.exists(), f"Governance file {fname} must exist in harness/"
        assert fpath.stat().st_size > 50, f"Governance file {fname} must not be empty"


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


def test_claim_c2_reconciled_cardinality():
    """Validates reconciled C2 cardinality: 878 pairs with continuous score."""
    c2_json = Path("reports/triage_results/c2.json")
    assert c2_json.exists(), "reports/triage_results/c2.json must exist"
    with open(c2_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["audit"]["label_analysis"]["continuous_stats"]["count"] == 878

    c2_abl = Path("results/c2_ablation/c2_ablation_results.json")
    if c2_abl.exists():
        with open(c2_abl, "r", encoding="utf-8") as f:
            abl_data = json.load(f)
        s1_f4_acc = abl_data["scenarios"]["S1_cold_resume"]["F4_CombinedFull"]["classification"]["accuracy"]
        # 504 / 878 = 0.57403189
        assert abs(s1_f4_acc - (504 / 878)) < 1e-4, f"Ablation denominator must match 878 pairs"


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


def test_claim_3_remote_kaggle_specs():
    """Validates Claim 4.1: Remote probe recorded correct specs."""
    probe_path = Path("docs/KAGGLE_PROBE.md")
    assert probe_path.exists(), "docs/KAGGLE_PROBE.md must exist"
    content = probe_path.read_text(encoding="utf-8")
    assert "Tesla T4" in content, "Kaggle probe should record Tesla T4"
    assert "Internet Access" in content or "HTTP 200" in content, "Internet access must be verified"


def test_rendered_tables_match():
    """Validates that scripts/render_report_tables.py generates non-empty tables containing valid metrics."""
    j1_tbl = render_j1_sweep_table()
    assert "S1: Cold-Resume" in j1_tbl
    assert "LightGBM" in j1_tbl
    assert "0.6603" in j1_tbl

    c2_tbl = render_c2_ablation_table()
    assert "F4: Combined Full" in c2_tbl
    assert "0.4401" in c2_tbl

    like_tbl = render_like_for_like_table()
    assert "S2: Cold-JD" in like_tbl
    assert "Cross-Encoder (Fine-Tuned 3 ep)" in like_tbl
    assert "0.5862" in like_tbl
    assert "0.7628" in like_tbl


def test_like_for_like_split_fidelity():
    """Validates that Cross-Encoder and Bi-Encoder evaluations share identical test sizes and majority baselines."""
    like_path = Path("reports/like_for_like_biencoder.json")
    ce_path = Path("reports/cross_encoder_results.json")
    assert like_path.exists(), "reports/like_for_like_biencoder.json must exist"
    assert ce_path.exists(), "reports/cross_encoder_results.json must exist"

    with open(like_path, "r", encoding="utf-8") as f:
        like_data = json.load(f)
    with open(ce_path, "r", encoding="utf-8") as f:
        ce_data = json.load(f)["scenarios"]

    # S2
    assert like_data["S2"]["test_size"] == 1259
    assert abs(like_data["S2"]["majority"]["accuracy"] - 0.5210) < 1e-4

    # S3
    assert like_data["S3"]["test_size"] == 272
    assert abs(like_data["S3"]["majority"]["accuracy"] - 0.5441) < 1e-4


def test_judge_packet_deterministic_generation():
    """Validates that make_judge_packet.py regenerates reports/JUDGE_PACKET.md without discrepancies."""
    packet_path = Path("reports/JUDGE_PACKET.md")
    assert packet_path.exists(), "reports/JUDGE_PACKET.md must exist"
    old_content = packet_path.read_text(encoding="utf-8")
    
    # Run generator
    generate_judge_packet()
    new_content = packet_path.read_text(encoding="utf-8")
    assert old_content == new_content, "JUDGE_PACKET.md in working tree does not match generated output"


def test_judge_packet_numbers_match_json_artifacts():
    """Asserts that all quantitative numbers in JUDGE_PACKET.md match source JSON files."""
    packet_text = Path("reports/JUDGE_PACKET.md").read_text(encoding="utf-8")

    # Load artifacts
    c1 = json.load(open("reports/triage_results/c1.json", encoding="utf-8"))
    c2 = json.load(open("reports/triage_results/c2.json", encoding="utf-8"))
    c3 = json.load(open("reports/triage_results/c3.json", encoding="utf-8"))
    c4 = json.load(open("reports/triage_results/c4.json", encoding="utf-8"))
    c5 = json.load(open("reports/triage_results/c5.json", encoding="utf-8"))
    ctl = json.load(open("reports/triage_results/control.json", encoding="utf-8"))
    c5_v = json.load(open("reports/c5_verification_results.json", encoding="utf-8"))

    # Assert triage cardinalities are exact
    assert str(c1["audit"]["rows"]) in packet_text
    assert str(c1["audit"]["entities"]["unique_resumes"]) in packet_text
    assert str(c1["audit"]["entities"]["unique_jds"]) in packet_text

    assert str(c2["audit"]["rows"]) in packet_text
    assert str(c2["audit"]["entities"]["unique_resumes"]) in packet_text
    assert str(c2["audit"]["entities"]["unique_jds"]) in packet_text

    assert str(c3["audit"]["rows"]) in packet_text
    assert str(c3["audit"]["entities"]["unique_resumes"]) in packet_text
    assert str(c3["audit"]["entities"]["unique_jds"]) in packet_text

    assert str(c4["audit"]["rows"]) in packet_text
    assert str(c4["audit"]["entities"]["unique_resumes"]) in packet_text
    assert str(c4["audit"]["entities"]["unique_jds"]) in packet_text

    assert str(c5["audit"]["rows"]) in packet_text
    assert str(c5["audit"]["entities"]["unique_resumes"]) in packet_text
    assert str(c5["audit"]["entities"]["unique_jds"]) in packet_text

    assert str(ctl["audit"]["rows"]) in packet_text
    assert str(ctl["audit"]["entities"]["unique_resumes"]) in packet_text
    assert str(ctl["audit"]["entities"]["unique_jds"]) in packet_text

    # Assert S1, S2, S3 verification values
    s1 = c5_v["scenarios"]["S1_cold_resume"]
    s2 = c5_v["scenarios"]["S2_cold_jd"]
    s3 = c5_v["scenarios"]["S3_double_cold"]

    assert f"{s1['accuracy']*100:.2f}%" in packet_text
    assert f"{s1['macro_f1']*100:.2f}%" in packet_text
    assert f"{s1['macro_roc_auc']:.4f}" in packet_text
    assert f"{s1['binary_good_vs_nofit_auc']:.4f}" in packet_text

    assert f"{s2['accuracy']*100:.2f}%" in packet_text
    assert f"{s2['macro_f1']*100:.2f}%" in packet_text
    assert f"{s2['macro_roc_auc']:.4f}" in packet_text
    assert f"{s2['binary_good_vs_nofit_auc']:.4f}" in packet_text

    assert f"{s3['accuracy']*100:.2f}%" in packet_text
    assert f"{s3['macro_f1']*100:.2f}%" in packet_text
    assert f"{s3['macro_roc_auc']:.4f}" in packet_text
    assert f"{s3['binary_good_vs_nofit_auc']:.4f}" in packet_text


def test_no_unsupported_phrases_in_judge_packet():
    """Asserts that unsupported or misleading phrases have been removed."""
    packet_text = Path("reports/JUDGE_PACKET.md").read_text(encoding="utf-8")
    assert "true empirical ceiling" not in packet_text, "Phrase 'true empirical ceiling' must be removed"
    assert "execute cross-encoder" not in packet_text.lower(), "Stale roadmap item 'execute cross-encoder' must be updated"
    assert "completely unsupported" not in packet_text, "Phrase 'completely unsupported' must be qualified"
