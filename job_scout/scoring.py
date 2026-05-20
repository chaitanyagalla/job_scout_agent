"""Deterministic resume-profile vs job-description scoring."""

from __future__ import annotations

import re
import unicodedata
from typing import Optional

from job_scout.domain_models import JobMatchScore
from job_scout.domain_models import ResumeProfile

STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "into", "your",
    "our", "their", "will", "have", "has", "had", "you", "they", "them",
    "are", "was", "were", "be", "been", "being", "who", "what", "when",
    "where", "why", "how", "but", "not", "all", "any", "can", "may",
    "must", "such", "than", "then", "job", "role", "team", "work",
    "working", "candidate", "experience", "years", "year", "skills",
    "ability", "preferred", "plus", "good", "strong", "using", "used",
    "build", "building", "develop", "developing", "developer",
}

SKILL_ONTOLOGY = {
    "python": {"weight": 1.4, "aliases": []},
    "java": {"weight": 1.4, "aliases": []},
    "javascript": {"weight": 1.3, "aliases": ["js"]},
    "typescript": {"weight": 1.3, "aliases": ["ts"]},
    "sql": {"weight": 1.2, "aliases": []},
    "spring boot": {"weight": 1.5, "aliases": ["springboot"]},
    "spring": {"weight": 1.2, "aliases": []},
    "node.js": {"weight": 1.3, "aliases": ["nodejs", "node js", "node"]},
    "express": {"weight": 1.1, "aliases": ["express.js", "expressjs"]},
    "fastapi": {"weight": 1.1, "aliases": []},
    "django": {"weight": 1.1, "aliases": []},
    "react": {"weight": 1.3, "aliases": ["react.js", "reactjs"]},
    "angular": {"weight": 1.3, "aliases": []},
    "next.js": {"weight": 1.2, "aliases": ["nextjs", "next js"]},
    "html": {"weight": 0.7, "aliases": []},
    "css": {"weight": 0.7, "aliases": []},
    "tailwind": {"weight": 0.8, "aliases": ["tailwindcss"]},
    "postgresql": {"weight": 1.2, "aliases": ["postgres", "postgre sql"]},
    "mysql": {"weight": 1.1, "aliases": []},
    "mongodb": {"weight": 1.1, "aliases": ["mongo db"]},
    "redis": {"weight": 1.0, "aliases": []},
    "aws": {"weight": 1.2, "aliases": ["amazon web services"]},
    "gcp": {"weight": 1.1, "aliases": ["google cloud", "google cloud platform"]},
    "azure": {"weight": 1.0, "aliases": []},
    "docker": {"weight": 1.2, "aliases": []},
    "kubernetes": {"weight": 1.2, "aliases": ["k8s"]},
    "ci/cd": {"weight": 1.0, "aliases": ["cicd", "ci cd"]},
    "git": {"weight": 0.8, "aliases": []},
    "rest api": {"weight": 1.1, "aliases": ["rest", "restful api", "restful apis"]},
    "microservices": {"weight": 1.1, "aliases": ["micro services"]},
    "llm": {"weight": 1.4, "aliases": ["large language model", "llms"]},
    "rag": {"weight": 1.4, "aliases": ["retrieval augmented generation"]},
    "langchain": {"weight": 1.2, "aliases": []},
    "genai": {"weight": 1.1, "aliases": ["generative ai"]},
}

REQUIRED_HINTS = (
    "required", "must", "need to", "needs to", "minimum", "at least",
    "hands-on", "expert in", "proficient in", "strong in", "mandatory",
)
PREFERRED_HINTS = (
    "preferred", "nice to have", "good to have", "plus", "bonus", "desired",
)
TITLE_SYNONYMS = {
    "sde": "software engineer",
    "developer": "engineer",
    "backend": "back end",
    "frontend": "front end",
    "fullstack": "full stack",
}

_WS = re.compile(r"\s+")
_SPLIT_LINES = re.compile(r"[\n\r]+|[.;•\u2022]+")
_YEAR_PATTERNS = [
    re.compile(r"(?<!\d)(\d+(?:\.\d+)?)\s*(?:to|-)\s*(\d+(?:\.\d+)?)\s+(?:years?|yrs?)\b"),
    re.compile(r"minimum\s+of\s+(\d+(?:\.\d+)?)\+?\s+(?:years?|yrs?)\b"),
    re.compile(r"minimum\s+(\d+(?:\.\d+)?)\+?\s+(?:years?|yrs?)\b"),
    re.compile(r"at\s+least\s+(\d+(?:\.\d+)?)\+?\s+(?:years?|yrs?)\b"),
    re.compile(r"(\d+(?:\.\d+)?)\+\s+(?:years?|yrs?)\b"),
    re.compile(r"(?<![-\d])(\d+(?:\.\d+)?)\s+(?:years?|yrs?)\b(?:\s+of\s+experience|\s+experience)?"),
]


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    text = text.lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.replace("&", " and ")
    for source, target in TITLE_SYNONYMS.items():
        text = re.sub(rf"(?<!\w){re.escape(source)}(?!\w)", target, text)
    text = _WS.sub(" ", text).strip()
    return text


def _tokenize(text: str) -> set[str]:
    tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9.+#/:-]{1,}", _normalize(text))
    return {token for token in tokens if token not in STOPWORDS and len(token) > 2}


def _count_phrase(haystack: str, phrase: str) -> int:
    pattern = r"(?<!\w)" + re.escape(phrase) + r"(?!\w)"
    return len(re.findall(pattern, haystack))


def _extract_required_years(jd_text: str) -> Optional[float]:
    def _extract_segment_years(segment: str) -> Optional[float]:
        for index, pattern in enumerate(_YEAR_PATTERNS):
            match = pattern.search(segment)
            if not match:
                continue
            if index == 0 and match.group(2):
                return float(match.group(1))
            return float(match.group(1))
        return None

    required_years: list[float] = []
    general_years: list[float] = []
    preferred_years: list[float] = []
    segments = _split_requirement_segments(jd_text) or [_normalize(jd_text)]

    for segment in segments:
        years = _extract_segment_years(segment)
        if years is None:
            continue
        bucket = _classify_requirement_segment(segment)
        if bucket == "required":
            required_years.append(years)
        elif bucket == "preferred":
            preferred_years.append(years)
        else:
            general_years.append(years)

    if required_years:
        return max(required_years)
    if general_years:
        return max(general_years)
    if preferred_years:
        return max(preferred_years)
    return None


def _sanitize_years_experience(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    return value if value >= 0 else 0.0


def _coerce_profile_for_scoring(profile: dict | ResumeProfile) -> ResumeProfile:
    if isinstance(profile, ResumeProfile):
        years_experience = _sanitize_years_experience(profile.years_experience)
        if years_experience == profile.years_experience:
            return profile
        return profile.model_copy(update={"years_experience": years_experience})

    payload = dict(profile)
    payload["years_experience"] = _sanitize_years_experience(payload.get("years_experience"))
    return ResumeProfile.model_validate(payload)


def _experience_alignment(required_years: Optional[float], actual_years: Optional[float]) -> float:
    if required_years is None:
        return 0.7
    if actual_years is None:
        return 0.35
    if actual_years >= required_years:
        return 1.0
    gap = required_years - actual_years
    if gap <= 0.5:
        return 0.9
    if gap <= 1:
        return 0.75
    if gap <= 2:
        return 0.55
    return 0.2


def _title_alignment(job_title: str, role_titles: list[str]) -> float:
    if not role_titles:
        return 0.35

    job_tokens = _tokenize(job_title)
    if not job_tokens:
        return 0.35

    best = 0.0
    for role_title in role_titles:
        role_tokens = _tokenize(role_title)
        if not role_tokens:
            continue
        precision = len(job_tokens & role_tokens) / max(1, len(job_tokens))
        recall = len(job_tokens & role_tokens) / max(1, len(role_tokens))
        best = max(best, 0.7 * precision + 0.3 * recall)

    return min(1.0, best)


def _general_overlap(profile_text: str, jd_text: str) -> float:
    profile_tokens = _tokenize(profile_text)
    jd_tokens = _tokenize(jd_text)

    if not profile_tokens or not jd_tokens:
        return 0.0

    meaningful_jd_tokens = {
        token for token in jd_tokens
        if len(token) > 3 and not token.isdigit()
    }
    if not meaningful_jd_tokens:
        return 0.3

    overlap = len(profile_tokens & meaningful_jd_tokens) / max(1, len(meaningful_jd_tokens))
    return min(1.0, overlap)


def _extract_skill_mentions(text: str) -> set[str]:
    normalized = _normalize(text)
    found: set[str] = set()
    for canonical, meta in SKILL_ONTOLOGY.items():
        forms = [canonical] + meta["aliases"]
        if any(_count_phrase(normalized, form) > 0 for form in forms):
            found.add(canonical)
    return found


def _split_requirement_segments(jd_text: str) -> list[str]:
    segments = []
    for part in _SPLIT_LINES.split(jd_text):
        segment = _normalize(part)
        if len(segment) < 8:
            continue
        segments.append(segment)
    return segments


def _classify_requirement_segment(segment: str) -> str:
    if any(hint in segment for hint in PREFERRED_HINTS):
        return "preferred"
    if any(hint in segment for hint in REQUIRED_HINTS):
        return "required"
    return "general"


def _extract_job_requirements(job_description: str) -> dict:
    jd_text = _normalize(job_description)
    required_skills: set[str] = set()
    preferred_skills: set[str] = set()
    general_skills: set[str] = set()

    for segment in _split_requirement_segments(jd_text):
        found = _extract_skill_mentions(segment)
        if not found:
            continue
        bucket = _classify_requirement_segment(segment)
        if bucket == "required":
            required_skills.update(found)
        elif bucket == "preferred":
            preferred_skills.update(found)
        else:
            general_skills.update(found)

    if not required_skills and not preferred_skills and not general_skills:
        general_skills = _extract_skill_mentions(jd_text)

    general_skills -= required_skills
    general_skills -= preferred_skills
    preferred_skills -= required_skills

    return {
        "required": sorted(required_skills),
        "preferred": sorted(preferred_skills),
        "general": sorted(general_skills),
    }


def _weighted_coverage(required_skills: list[str], profile_skills: set[str]) -> tuple[float, list[str], list[str]]:
    if not required_skills:
        return 0.0, [], []

    matched = [skill for skill in required_skills if skill in profile_skills]
    missing = [skill for skill in required_skills if skill not in profile_skills]
    total_weight = sum(SKILL_ONTOLOGY[skill]["weight"] for skill in required_skills)
    matched_weight = sum(SKILL_ONTOLOGY[skill]["weight"] for skill in matched)
    score = matched_weight / max(total_weight, 0.0001)

    matched_sorted = sorted(matched, key=lambda skill: -SKILL_ONTOLOGY[skill]["weight"])
    missing_sorted = sorted(missing, key=lambda skill: -SKILL_ONTOLOGY[skill]["weight"])
    return score, matched_sorted, missing_sorted


def _combine_skill_lists(*skill_lists: list[str]) -> list[str]:
    seen = set()
    ordered: list[str] = []
    for skill_list in skill_lists:
        for skill in skill_list:
            if skill in seen:
                continue
            seen.add(skill)
            ordered.append(skill)
    return ordered


def _summarize_fit_verdict(score: int, has_blockers: bool) -> str:
    if score >= 80 and not has_blockers:
        return "Strong apply"
    if score >= 65:
        return "Good match"
    if score >= 45:
        return "Possible stretch"
    return "Low match"


def _summarize_reason_to_apply(
    matched_skills: list[str],
    title_score: float,
    experience_score: float,
) -> str:
    if matched_skills:
        top_skills = ", ".join(matched_skills[:3])
        return f"Strongest alignment comes from {top_skills}."
    if title_score >= 0.7:
        return "Role alignment is stronger than direct skill evidence."
    if experience_score >= 0.75:
        return "Experience alignment is a positive signal for this role."
    return "There is limited direct overlap, so apply only if the role is strategic."


def _summarize_blocker_risk(
    missing_required: list[str],
    insufficient_experience: bool,
    title_mismatch: bool,
) -> str:
    if missing_required or insufficient_experience:
        return "High"
    if title_mismatch:
        return "Medium"
    return "Low"


def build_profile_text(profile: dict | ResumeProfile) -> str:
    profile_model = _coerce_profile_for_scoring(profile)
    parts = [
        profile_model.candidate_name,
        profile_model.professional_summary,
        " ".join(profile_model.role_titles),
        " ".join(profile_model.core_skills),
        " ".join(profile_model.additional_skills),
        " ".join(profile_model.preferred_locations),
        " ".join(profile_model.work_preferences),
        " ".join(profile_model.notable_projects),
    ]
    return _normalize(" ".join(part for part in parts if part))


def score_resume_vs_jd(job_title: str, job_description: str, profile: dict) -> dict:
    profile_model = _coerce_profile_for_scoring(profile)
    jd_title = _normalize(job_title)
    jd_text = _normalize(job_description)
    profile_text = build_profile_text(profile_model)
    profile_skills = _extract_skill_mentions(profile_text)

    job_requirements = _extract_job_requirements(jd_text)
    required_score, matched_required, missing_required = _weighted_coverage(
        job_requirements["required"],
        profile_skills,
    )
    preferred_score, matched_preferred, missing_preferred = _weighted_coverage(
        job_requirements["preferred"],
        profile_skills,
    )
    general_score, matched_general, missing_general = _weighted_coverage(
        job_requirements["general"],
        profile_skills,
    )

    if not job_requirements["required"]:
        required_score = general_score if job_requirements["general"] else 0.6
    if not job_requirements["preferred"]:
        preferred_score = 0.6
    if not job_requirements["general"]:
        general_score = max(general_score, required_score)

    title_score = _title_alignment(jd_title, profile_model.role_titles)
    required_years = _extract_required_years(jd_text)
    experience_score = _experience_alignment(required_years, profile_model.years_experience)
    overlap_score = _general_overlap(profile_text, jd_text)

    final_score = (
        0.45 * required_score
        + 0.15 * preferred_score
        + 0.10 * general_score
        + 0.12 * title_score
        + 0.10 * experience_score
        + 0.08 * overlap_score
    )

    required_count = len(job_requirements["required"])
    if required_count:
        missing_required_ratio = len(missing_required) / required_count
        if missing_required_ratio >= 0.5:
            final_score *= 0.72
        elif missing_required_ratio >= 0.3:
            final_score *= 0.85

    if required_years is not None and experience_score < 0.55:
        final_score *= 0.9

    score = max(0, min(100, round(final_score * 100)))

    matched_skills = _combine_skill_lists(matched_required, matched_preferred, matched_general)
    missing_skills = _combine_skill_lists(missing_required, missing_preferred, missing_general)
    insufficient_experience = bool(
        required_years is not None
        and profile_model.years_experience is not None
        and profile_model.years_experience + 0.5 < required_years
    )
    title_mismatch = title_score < 0.35
    blocker_reasons = []
    if missing_required:
        blocker_reasons.append(
            "Missing required skills: " + ", ".join(missing_required[:3]) + "."
        )
    if insufficient_experience:
        blocker_reasons.append(
            f"Experience gap: {profile_model.years_experience} yrs vs {required_years:g} yrs required."
        )
    if title_mismatch:
        blocker_reasons.append("Role title alignment is weak.")

    explanation_parts = []
    if required_score >= 0.8:
        explanation_parts.append("Strong match on core requirements.")
    elif required_score >= 0.5:
        explanation_parts.append("Partial match on core requirements.")
    else:
        explanation_parts.append("Weak match on core requirements.")

    if preferred_score >= 0.7:
        explanation_parts.append("Preferred skills align well.")

    if title_score >= 0.7:
        explanation_parts.append("Role titles align well.")
    elif title_score < 0.35:
        explanation_parts.append("Role title alignment is weak.")

    if required_years is not None and profile_model.years_experience is not None:
        explanation_parts.append(
            f"Experience: {profile_model.years_experience} yrs vs {required_years:g} yrs required."
        )

    if missing_required:
        explanation_parts.append("Missing required: " + ", ".join(missing_required[:3]) + ".")
    elif missing_preferred:
        explanation_parts.append("Top gaps: " + ", ".join(missing_preferred[:3]) + ".")

    has_blockers = bool(missing_required or insufficient_experience or title_mismatch)
    if score >= 80 and has_blockers:
        explanation_parts.append("Overall score is strong, but blockers still need review.")
    fit_verdict = _summarize_fit_verdict(score, has_blockers)
    reason_to_apply = _summarize_reason_to_apply(matched_skills, title_score, experience_score)
    blocker_risk = _summarize_blocker_risk(
        missing_required,
        insufficient_experience,
        title_mismatch,
    )

    return JobMatchScore.model_validate({
        "score": score,
        "signals": {
            "required_skill_coverage": round(required_score, 3),
            "preferred_skill_coverage": round(preferred_score, 3),
            "overall_skill_coverage": round(general_score, 3),
            "title_alignment": round(title_score, 3),
            "experience_alignment": round(experience_score, 3),
            "keyword_overlap": round(overlap_score, 3),
        },
        "matched_skills": matched_skills[:10],
        "missing_skills": missing_skills[:10],
        "explanation": " ".join(explanation_parts),
        "fit_verdict": fit_verdict,
        "reason_to_apply": reason_to_apply,
        "blocker_risk": blocker_risk,
        "evidence": {
            "required_skills": job_requirements["required"],
            "preferred_skills": job_requirements["preferred"],
            "general_skills": job_requirements["general"],
            "matched_required_skills": matched_required,
            "missing_required_skills": missing_required,
            "matched_preferred_skills": matched_preferred,
            "missing_preferred_skills": missing_preferred,
            "matched_general_skills": matched_general,
            "missing_general_skills": missing_general,
            "required_years": required_years,
            "actual_years": profile_model.years_experience,
        },
        "blockers": {
            "has_blockers": has_blockers,
            "missing_required_skills": missing_required,
            "insufficient_experience": insufficient_experience,
            "title_mismatch": title_mismatch,
            "blocker_reasons": blocker_reasons,
        },
    }).model_dump(exclude_none=True)
