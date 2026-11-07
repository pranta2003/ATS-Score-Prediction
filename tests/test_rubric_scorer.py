"""Unit tests for Interpretable Rubric Scorer."""

import pytest
from src.ats.rubric import (
    extract_years_experience,
    extract_degree_level,
    extract_seniority_level,
    score_rubric,
)


def test_experience_extraction():
    text1 = "Senior Software Engineer with 7 years of experience in Python and AWS."
    assert extract_years_experience(text1) == 7.0

    text2 = "Looking for candidate with 5+ yrs experience."
    assert extract_years_experience(text2) == 5.0

    text3 = "Recent graduate seeking entry level position."
    assert extract_years_experience(text3) == 0.0


def test_education_extraction():
    text1 = "Holds a Master of Science in Computer Science and Bachelor of Science."
    assert extract_degree_level(text1) == 3

    text2 = "PhD in Machine Learning with publications in NeurIPS."
    assert extract_degree_level(text2) == 4

    text3 = "Completed High School diploma."
    assert extract_degree_level(text3) == 1


def test_seniority_extraction():
    text1 = "Principal Architect leading distributed teams."
    assert extract_seniority_level(text1) == 5

    text2 = "Junior Web Developer eager to learn."
    assert extract_seniority_level(text2) == 1


def test_score_rubric_monotonicity():
    jd = "Seeking Senior Data Scientist with 5 years experience in Python, PyTorch, SQL, and Docker. Master degree required."
    
    # Highly matching resume
    res_good = "Senior Data Scientist with 6 years experience in Python, PyTorch, SQL, Docker, and Kubernetes. Holds a Master of Science. Contact: test@example.com"
    score_good = score_rubric(res_good, jd)

    # Weakly matching resume
    res_weak = "Graphic Designer with 1 year experience in Photoshop and Illustrator. High school diploma."
    score_weak = score_rubric(res_weak, jd)

    assert score_good["rubric_score"] > score_weak["rubric_score"]
    assert score_good["rubric_score"] >= 80.0
    assert score_weak["rubric_score"] <= 55.0
    assert len(score_good["suggestions"]) < len(score_weak["suggestions"])
    assert "rubric_score" in score_good
