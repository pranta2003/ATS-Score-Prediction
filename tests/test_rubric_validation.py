"""Unit tests for rubric validation and behavioral properties."""

import json
from pathlib import Path
import pytest
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.ats.rubric import score_rubric, evaluate_formatting


def test_rubric_determinism():
    """Verify that scoring the same pair twice yields identical scores and breakdown."""
    resume = "Senior Software Engineer with 6 years experience in Python, AWS, Docker, and PostgreSQL. BS in Computer Science. Email: dev@example.com Phone: 555-123-4567."
    jd = "Seeking Senior Software Engineer with 5+ years experience in Python, AWS, and Docker. Bachelor degree required."
    
    res1 = score_rubric(resume, jd)
    res2 = score_rubric(resume, jd)
    
    assert res1["rubric_score"] == res2["rubric_score"]
    assert res1["content_fit_score"] == res2["content_fit_score"]
    assert res1["formatting_score"] == res2["formatting_score"]
    assert res1["breakdown"] == res2["breakdown"]


def test_rubric_tailoring_effect():
    """Verify that tailoring a resume to include target JD skills increases fit score."""
    base_jd = "Senior Backend Engineer with 5 years experience in Go, Kubernetes, Kafka, and Redis. BS degree required."
    generic_resume = "Software Engineer with 4 years experience in Python and MySQL. BS in Computer Science. Email: dev@example.com."
    
    tailored_resume = generic_resume + " Proven production experience developing microservices in Go, orchestrating Kubernetes clusters, and streaming data with Kafka and Redis."
    
    base_res = score_rubric(generic_resume, base_jd)
    tailored_res = score_rubric(tailored_resume, base_jd)
    
    delta = tailored_res["rubric_score"] - base_res["rubric_score"]
    assert delta >= 10.0, f"Expected tailoring to increase score by >= 10 pts, got delta={delta}"


def test_keyword_stuffing_cap_effect():
    """Verify that repeating a single matched keyword 50 times produces 0 additional score increase."""
    jd = "Seeking Python Developer with 3 years experience. Bachelor degree required."
    base_resume = "Python Developer with 3 years experience. Bachelor of Science. Email: dev@example.com."
    stuffed_resume = base_resume + " " + ("python " * 50)
    
    base_res = score_rubric(base_resume, jd)
    stuffed_res = score_rubric(stuffed_resume, jd)
    
    delta = stuffed_res["rubric_score"] - base_res["rubric_score"]
    assert delta == 0.0, f"Keyword stuffing must be capped by set-based extraction, got delta={delta}"


def test_jd_swap_sensitivity():
    """Verify that swapping to an entirely unrelated JD substantially changes the rubric score."""
    resume = "Experienced Senior Java Architect with 10 years experience building enterprise Spring Boot microservices, Oracle DB, and Maven builds. Master degree in Computer Engineering."
    jd_tech = "Senior Java Engineer with 8+ years experience in Spring Boot, microservices, and Oracle DB."
    jd_unrelated = "Certified Dental Hygienist with 3 years clinical dental care, patient periodontal therapy, and radiology. Dental Hygiene diploma required."
    
    score_tech = score_rubric(resume, jd_tech)["rubric_score"]
    score_unrelated = score_rubric(resume, jd_unrelated)["rubric_score"]
    
    diff = score_tech - score_unrelated
    assert diff >= 15.0, f"Expected significant drop on swapped JD, got score_tech={score_tech}, score_unrelated={score_unrelated}"


def test_formatting_decoupled_from_fit_score():
    """Verify that formatting hygiene is tracked independently and does not corrupt content fit score."""
    resume = "Python Developer with 3 years experience. Bachelor of Science."
    jd = "Python Developer with 3 years experience. Bachelor degree required."
    
    res = score_rubric(resume, jd)
    assert "formatting_score" in res
    assert "content_fit_score" in res
    assert res["rubric_score"] == res["content_fit_score"]
    
    # Assert breakdown scores sum equals content_fit_score
    score_keys = [k for k in res["breakdown"] if k.endswith("_score")]
    breakdown_sum = sum(res["breakdown"][k] for k in score_keys)
    assert abs(breakdown_sum - res["content_fit_score"]) < 1e-4, f"Breakdown sum ({breakdown_sum}) != content_fit_score ({res['content_fit_score']})"


def test_rubric_validation_results_file():
    """Verify that reports/rubric_validation_results.json exists and has verified metrics."""
    res_path = PROJECT_ROOT / "reports" / "rubric_validation_results.json"
    assert res_path.exists(), "rubric_validation_results.json must exist"
    
    with open(res_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    assert "correlation" in data
    assert "spearman_r" in data["correlation"]
    assert "discrimination" in data
    assert "binary_good_vs_nofit_auc" in data["discrimination"]
    assert "behavior_tests" in data
    assert data["behavior_tests"]["determinism"]["passed"] is True
    assert data["behavior_tests"]["tailoring_effect"]["passed"] is True
    assert data["behavior_tests"]["keyword_stuffing_cap"]["passed"] is True
    assert data["behavior_tests"]["jd_swap_sensitivity"]["passed"] is True
