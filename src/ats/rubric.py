"""Interpretable Rubric-Based ATS Match Scorer.

Provides transparent, rule-based matching across:
1. Hard keyword match with diminishing returns (capped terms).
2. Years-of-experience extraction and comparison against JD requirement.
3. Degree level / Education hierarchy matching.
4. Role title / Seniority level matching.
5. Actionable missing-keyword and qualification suggestions.

Governance Policy:
- Formatting score is tracked INDEPENDENTLY from the 0-100 content fit score.
- Output is an explainable rubric fit score (0-100), NEVER called 'accuracy'.
"""

import re
from typing import Dict, Any, List, Set, Tuple


# Common degree levels hierarchy
DEGREE_LEVELS = {
    "phd": 4,
    "doctorate": 4,
    "master": 3,
    "ms": 3,
    "msc": 3,
    "mba": 3,
    "bachelor": 2,
    "bs": 2,
    "bsc": 2,
    "ba": 2,
    "associate": 1,
    "diploma": 1,
}

SENIORITY_LEVELS = {
    "intern": 0,
    "junior": 1,
    "entry": 1,
    "associate": 1,
    "mid": 2,
    "senior": 3,
    "sr": 3,
    "lead": 4,
    "principal": 5,
    "staff": 5,
    "director": 6,
    "head": 6,
    "vp": 7,
}

STOPWORDS = {
    "a", "an", "the", "and", "or", "in", "on", "at", "to", "for", "with",
    "by", "as", "is", "are", "was", "were", "be", "been", "have", "has",
    "had", "do", "does", "did", "of", "from", "that", "this", "these",
    "those", "will", "would", "shall", "should", "can", "could", "may",
    "might", "must", "we", "you", "they", "he", "she", "it", "our", "your",
    "their", "his", "her", "its", "role", "job", "position", "candidate",
    "work", "experience", "looking", "seeking", "required", "preferred"
}


def extract_keywords(text: str, min_len: int = 3) -> Set[str]:
    """Tokenize and extract significant alphanumeric keywords (set-based to cap keyword stuffing)."""
    words = re.findall(r"\b[a-zA-Z0-9\+\#\.\-]{2,}\b", text.lower())
    clean_words = {w for w in words if len(w) >= min_len and w not in STOPWORDS}
    return clean_words


def extract_years_experience(text: str) -> float:
    """Extract maximum claimed years of experience using regex patterns."""
    text_lower = text.lower()
    patterns = [
        r"(\d{1,2})\+?\s*(?:to\s*\d{1,2}\s*)?(?:years?|yrs?)(?:\s+of)?(?:\s+experience)?",
        r"(?:experience\s*(?:of)?\s*)(\d{1,2})\+?\s*(?:years?|yrs?)",
    ]
    years = []
    for pat in patterns:
        matches = re.findall(pat, text_lower)
        for m in matches:
            try:
                y = float(m)
                if y <= 45:  # Reasonable sanity bound
                    years.append(y)
            except ValueError:
                continue
    return max(years) if years else 0.0


def extract_degree_level(text: str) -> int:
    """Extract highest education level mentioned."""
    text_lower = text.lower()
    highest = 0
    for deg, lvl in DEGREE_LEVELS.items():
        if re.search(rf"\b{deg}\b", text_lower):
            if lvl > highest:
                highest = lvl
    return highest


def extract_seniority_level(text: str) -> int:
    """Extract highest seniority mentioned."""
    text_lower = text.lower()
    highest = 1  # Default mid/entry
    for sen, lvl in SENIORITY_LEVELS.items():
        if re.search(rf"\b{sen}\b", text_lower):
            if lvl > highest:
                highest = lvl
    return highest


def evaluate_formatting(resume_text: str) -> Dict[str, Any]:
    """
    Independent formatting and structural hygiene audit (0-100 scale).
    Kept strictly outside the content match fit score.
    """
    score = 0.0
    issues = []
    
    words = resume_text.split()
    word_count = len(words)
    if 250 <= word_count <= 1200:
        score += 40.0
    elif 150 <= word_count < 250:
        score += 25.0
        issues.append("Resume is somewhat brief (<250 words); expand on project scope and responsibilities.")
    elif word_count > 1200:
        score += 20.0
        issues.append("Resume is very lengthy (>1,200 words); consider condensing to 1-2 pages.")
    else:
        score += 10.0
        issues.append("Resume is extremely short (<150 words); provide complete job history and skills.")

    has_email = bool(re.search(r"[\w\.-]+@[\w\.-]+", resume_text))
    has_phone = bool(re.search(r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b", resume_text))
    
    if has_email:
        score += 25.0
    else:
        issues.append("Missing parsed email address.")
        
    if has_phone:
        score += 25.0
    else:
        issues.append("Missing parsed phone number.")

    # Section headers detection
    headers = ["experience", "education", "skills", "projects", "certifications", "summary"]
    text_lower = resume_text.lower()
    found_headers = sum(1 for h in headers if h in text_lower)
    if found_headers >= 3:
        score += 10.0
    else:
        issues.append("Add clear standard section headers (Experience, Education, Skills, Projects).")

    return {
        "formatting_score": round(min(100.0, score), 1),
        "word_count": word_count,
        "has_email": has_email,
        "has_phone": has_phone,
        "formatting_issues": issues,
    }


def score_rubric(resume_text: str, jd_text: str) -> Dict[str, Any]:
    """Calculate transparent rubric ATS content fit score (0-100) and actionable feedback.
    
    Content Fit Weights (Sum = 100 pts):
    - Keyword Coverage (diminishing returns, set-capped): 50 pts
    - Experience Match: 25 pts
    - Education Match: 15 pts
    - Seniority Match: 10 pts
    
    Formatting score is computed and reported separately.
    """
    res_kw = extract_keywords(resume_text)
    jd_kw = extract_keywords(jd_text)

    # 1. Keyword coverage with diminishing returns
    matched_kw = res_kw.intersection(jd_kw)
    missing_kw = sorted(list(jd_kw - res_kw))
    
    if len(jd_kw) > 0:
        raw_cov = len(matched_kw) / len(jd_kw)
        # Diminishing returns: exponent 0.7 curves rewards toward early key matches
        dim_cov = (raw_cov ** 0.7)
        kw_points = min(50.0, dim_cov * 50.0)
    else:
        raw_cov = 0.5
        kw_points = 25.0

    # 2. Years of Experience (25 pts)
    res_exp = extract_years_experience(resume_text)
    jd_req_exp = extract_years_experience(jd_text)
    if jd_req_exp > 0:
        if res_exp >= jd_req_exp:
            exp_points = 25.0
        else:
            exp_points = max(0.0, (res_exp / jd_req_exp) * 25.0)
    else:
        # JD did not specify explicit years
        exp_points = 25.0 if res_exp >= 2.0 else 18.0

    # 3. Education Level (15 pts)
    res_edu = extract_degree_level(resume_text)
    jd_edu = extract_degree_level(jd_text)
    if jd_edu > 0:
        if res_edu >= jd_edu:
            edu_points = 15.0
        elif res_edu == jd_edu - 1:
            edu_points = 10.0
        else:
            edu_points = 5.0
    else:
        edu_points = 15.0 if res_edu >= 2 else 12.0

    # 4. Seniority Level (10 pts)
    res_sen = extract_seniority_level(resume_text)
    jd_sen = extract_seniority_level(jd_text)
    if res_sen >= jd_sen:
        sen_points = 10.0
    elif res_sen == jd_sen - 1:
        sen_points = 7.0
    else:
        sen_points = 4.0

    # Content Fit Score (0-100)
    content_fit_score = round(kw_points + exp_points + edu_points + sen_points, 1)
    content_fit_score = max(0.0, min(100.0, content_fit_score))

    # Independent formatting audit
    fmt_audit = evaluate_formatting(resume_text)

    # Actionable suggestions
    suggestions = []
    if missing_kw:
        top_missing = missing_kw[:8]
        suggestions.append(f"Incorporate relevant key skills requested in the job description: {', '.join(top_missing)}.")
    if jd_req_exp > 0 and res_exp < jd_req_exp:
        suggestions.append(f"Job specifies {jd_req_exp:.0f}+ years experience; your resume indicates {res_exp:.0f} years. Highlight relevant project duration or equivalent experience.")
    if jd_edu > 0 and res_edu < jd_edu:
        suggestions.append(f"Target role prefers degree level {jd_edu}; ensure all degree coursework or certifications are explicitly detailed.")
    
    suggestions.extend(fmt_audit["formatting_issues"])

    return {
        "rubric_score": content_fit_score,
        "content_fit_score": content_fit_score,
        "formatting_score": fmt_audit["formatting_score"],
        "breakdown": {
            "keyword_match_score": round(kw_points, 1),
            "keyword_max": 50.0,
            "experience_match_score": round(exp_points, 1),
            "experience_max": 25.0,
            "education_match_score": round(edu_points, 1),
            "education_max": 15.0,
            "seniority_match_score": round(sen_points, 1),
            "seniority_max": 10.0,
        },
        "formatting_audit": fmt_audit,
        "extracted_entities": {
            "resume_years_exp": res_exp,
            "jd_required_years_exp": jd_req_exp,
            "resume_degree_level": res_edu,
            "jd_degree_level": jd_edu,
            "matched_keywords_count": len(matched_kw),
            "total_jd_keywords_count": len(jd_kw),
        },
        "suggestions": suggestions,
    }
