"""Interpretable Rubric-Based ATS Match Scorer.

Provides transparent, rule-based matching across:
1. Hard keyword match with diminishing returns (capped terms).
2. Years-of-experience extraction and comparison against JD requirement.
3. Degree level / Education hierarchy matching.
4. Role title / Seniority level matching.
5. Actionable missing-keyword and qualification suggestions.

Note: The output is an explainable rubric fit score (0-100), NEVER called 'accuracy'.
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
    """Tokenize and extract significant alphanumeric keywords."""
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


def score_rubric(resume_text: str, jd_text: str) -> Dict[str, Any]:
    """Calculate transparent rubric ATS score and actionable recommendations.
    
    Weights:
    - Keyword Coverage (diminishing returns): 40 pts
    - Experience Match: 25 pts
    - Education Match: 15 pts
    - Seniority Match: 10 pts
    - Structure & Clarity: 10 pts
    Total: 100 pts.
    """
    res_kw = extract_keywords(resume_text)
    jd_kw = extract_keywords(jd_text)

    # 1. Keyword coverage with diminishing returns
    matched_kw = res_kw.intersection(jd_kw)
    missing_kw = sorted(list(jd_kw - res_kw))
    
    if len(jd_kw) > 0:
        raw_cov = len(matched_kw) / len(jd_kw)
        # Diminishing returns: square root curve so first matches matter most
        dim_cov = (raw_cov ** 0.7)
        kw_points = min(40.0, dim_cov * 40.0)
    else:
        raw_cov = 0.5
        kw_points = 20.0

    # 2. Years of Experience
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

    # 3. Education Level
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

    # 4. Seniority Level
    res_sen = extract_seniority_level(resume_text)
    jd_sen = extract_seniority_level(jd_text)
    if res_sen >= jd_sen:
        sen_points = 10.0
    elif res_sen == jd_sen - 1:
        sen_points = 7.0
    else:
        sen_points = 4.0

    # 5. Structure & Formatting Clarity
    struct_points = 0.0
    res_len = len(resume_text.split())
    if 150 <= res_len <= 1500:
        struct_points += 5.0
    else:
        struct_points += 2.0
        
    has_email = bool(re.search(r"[\w\.-]+@[\w\.-]+", resume_text))
    has_phone = bool(re.search(r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b", resume_text))
    if has_email or has_phone:
        struct_points += 5.0

    total_score = round(kw_points + exp_points + edu_points + sen_points + struct_points, 1)
    total_score = max(0.0, min(100.0, total_score))

    # Actionable suggestions
    suggestions = []
    if missing_kw:
        top_missing = missing_kw[:8]
        suggestions.append(f"Incorporate relevant key skills requested in the job description: {', '.join(top_missing)}.")
    if jd_req_exp > 0 and res_exp < jd_req_exp:
        suggestions.append(f"Job specifies {jd_req_exp:.0f}+ years experience; your resume indicates {res_exp:.0f} years. Highlight relevant project duration or equivalent experience.")
    if jd_edu > 0 and res_edu < jd_edu:
        suggestions.append(f"Target role prefers degree level {jd_edu}; ensure all degree coursework or certifications are explicitly detailed.")
    if not (has_email or has_phone):
        suggestions.append("Ensure clear contact information (email/phone) is easily parsed at the top of your resume.")

    return {
        "rubric_score": total_score,
        "breakdown": {
            "keyword_match_score": round(kw_points, 1),
            "keyword_max": 40.0,
            "experience_match_score": round(exp_points, 1),
            "experience_max": 25.0,
            "education_match_score": round(edu_points, 1),
            "education_max": 15.0,
            "seniority_match_score": round(sen_points, 1),
            "seniority_max": 10.0,
            "formatting_structure_score": round(struct_points, 1),
            "formatting_max": 10.0,
        },
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
