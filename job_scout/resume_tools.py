from __future__ import annotations

import re
from typing import Optional

from google.adk.tools.tool_context import ToolContext

from job_scout.domain_models import JobMatchScore
from job_scout.domain_models import JobPosting
from job_scout.domain_models import ResumeProfile
from job_scout import resume_support
from job_scout.model_config import resume_gemini_attachment_fallback_enabled
from job_scout.scoring import score_resume_vs_jd
from job_scout.search_support import fetch_job_details
from job_scout.state_keys import LAST_SEARCH_RESULTS_STATE_KEY
from job_scout.state_keys import RESUME_PROFILE_EXTRACTION_METHOD_STATE_KEY
from job_scout.state_keys import RESUME_PROFILE_STATE_KEY
from job_scout.search_tools import search_jobs


async def _load_resume_part_from_current_user_message(
    tool_context: ToolContext,
    artifact_name: Optional[str] = None,
) -> tuple[Optional[str], Optional[object]]:
    user_content = getattr(tool_context, "user_content", None)
    candidate_parts = resume_support._extract_resume_parts_from_user_content(user_content)
    if not candidate_parts:
        return None, None

    requested_key = (artifact_name or "").strip().lower()
    selected_name: Optional[str] = None
    selected_part: Optional[object] = None

    if requested_key:
        for candidate_name, candidate_part in candidate_parts:
            if candidate_name.lower() == requested_key:
                selected_name = candidate_name
                selected_part = candidate_part
                break

    if selected_part is None:
        selected_name, selected_part = candidate_parts[0]

    if getattr(selected_part, "inline_data", None) is not None:
        return selected_name, selected_part
    if getattr(selected_part, "text", None):
        return selected_name, selected_part

    file_data = getattr(selected_part, "file_data", None)
    if file_data is None:
        return selected_name, selected_part

    file_uri = getattr(file_data, "file_uri", None)
    if not file_uri:
        return selected_name, selected_part

    parsed_uri = resume_support.artifact_util.parse_artifact_uri(str(file_uri))
    if parsed_uri is None:
        return selected_name, selected_part

    try:
        loaded_part = await tool_context.load_artifact(
            parsed_uri.filename,
            version=parsed_uri.version,
        )
    except TypeError:
        loaded_part = await tool_context.load_artifact(parsed_uri.filename)
    if loaded_part is not None:
        return parsed_uri.filename, loaded_part

    loaded_part = await tool_context.load_artifact(parsed_uri.filename)
    if loaded_part is not None:
        return parsed_uri.filename, loaded_part

    return selected_name, selected_part


async def extract_resume_profile_from_artifact(
    artifact_name: Optional[str] = None,
    tool_context: Optional[ToolContext] = None,
) -> dict:
    """Extract and save a structured resume profile from an uploaded artifact.

    Use this when the user uploads a resume or asks for matching based on the
    uploaded file rather than prompt-only search.

    Args:
        artifact_name: Optional uploaded artifact name to target explicitly.
        tool_context: ADK tool context used to read artifacts and persist state.

    Returns:
        A result payload describing whether extraction succeeded and the basic
        resume metadata saved into session state.
    """
    if tool_context is None:
        return {
            "status": "error",
            "message": "Tool context is required for resume extraction.",
        }

    extracted = await _extract_resume_profile_payload_from_artifact(
        artifact_name=artifact_name,
        tool_context=tool_context,
    )
    if extracted.get("status") != "ok":
        return extracted

    profile = extracted["profile"]
    extraction_method = extracted["extraction_method"]
    _persist_resume_profile(
        tool_context,
        profile,
        extraction_method=extraction_method,
    )
    return {
        "status": "ok",
        "message": "Resume profile extracted from uploaded artifact and saved.",
        "candidate_name": profile["candidate_name"],
        "role_titles": profile["role_titles"],
        "skills_count": len(profile["core_skills"]) + len(profile["additional_skills"]),
        "resume_source": profile["resume_source"],
        "extraction_method": extraction_method,
    }


def _persist_resume_profile(
    tool_context: ToolContext,
    profile: dict,
    *,
    extraction_method: str,
) -> None:
    tool_context.state[RESUME_PROFILE_STATE_KEY] = ResumeProfile.model_validate(profile).model_dump()
    tool_context.state[RESUME_PROFILE_EXTRACTION_METHOD_STATE_KEY] = extraction_method


async def _extract_resume_profile_payload_from_artifact(
    *,
    tool_context: ToolContext,
    artifact_name: Optional[str] = None,
) -> dict:
    try:
        artifact_names = await tool_context.list_artifacts()
    except Exception:
        artifact_names = []
    selected_artifact = resume_support._choose_resume_artifact_name(artifact_names, artifact_name)
    artifact_part = None

    if selected_artifact:
        artifact_part = await tool_context.load_artifact(selected_artifact)

    if artifact_part is None:
        selected_artifact, artifact_part = await _load_resume_part_from_current_user_message(
            tool_context,
            artifact_name=artifact_name or selected_artifact,
        )

    if not selected_artifact or artifact_part is None:
        return {
            "status": "error",
            "message": (
                "No uploaded resume artifact is available in this session or current user message."
            ),
        }

    local_result = resume_support._extract_resume_profile_locally(selected_artifact, artifact_part)
    if local_result.get("status") == "ok":
        profile = local_result["profile"]
        return {
            "status": "ok",
            "profile": profile,
            "resume_source": profile["resume_source"],
            "extraction_method": "local",
        }

    def _normalize_extracted_profile(extracted_payload: dict) -> Optional[dict]:
        try:
            normalized_profile = resume_support._normalize_resume_profile_payload(
                extracted_payload,
                resume_source=selected_artifact,
            )
        except ValueError:
            return None
        if not resume_support._profile_has_meaningful_resume_signal(normalized_profile):
            return None
        return normalized_profile

    text_fallback_error: Optional[str] = None
    profile = None
    extraction_method = None

    resume_text = (local_result.get("resume_text") or "").strip()
    if resume_text:
        try:
            extracted = await resume_support._parse_resume_with_text_model(
                selected_artifact,
                resume_text,
            )
            profile = _normalize_extracted_profile(extracted)
            if profile is None:
                text_fallback_error = (
                    "The text-model parser returned an incomplete or invalid resume profile."
                )
        except Exception as exc:
            text_fallback_error = str(exc)

    if profile is None:
        best_effort_profile = local_result.get("best_effort_profile")
        if best_effort_profile:
            profile = best_effort_profile
            extraction_method = "local_best_effort"

    if profile is None and resume_gemini_attachment_fallback_enabled():
        try:
            extracted = await resume_support._parse_resume_with_gemini(selected_artifact, artifact_part)
            profile = _normalize_extracted_profile(extracted)
            extraction_method = "gemini_attachment_fallback"
            if profile is None:
                return {
                    "status": "error",
                    "message": (
                        "The uploaded resume could not be parsed into a meaningful profile. "
                        "Try uploading a clearer PDF or DOCX resume."
                    ),
                    "resume_source": selected_artifact,
                }
        except Exception as exc:
            details = []
            if local_result.get("message"):
                details.append(f"local extraction: {local_result['message']}")
            if text_fallback_error:
                details.append(f"text-model fallback: {text_fallback_error}")
            details.append(f"attachment fallback: {exc}")
            return {
                "status": "error",
                "message": "Resume parsing failed after all extraction fallbacks.",
                "resume_source": selected_artifact,
                "details": details,
            }

    if profile is None:
        details = []
        if local_result.get("message"):
            details.append(f"local extraction: {local_result['message']}")
        if text_fallback_error:
            details.append(f"text-model fallback: {text_fallback_error}")
        details.append(
            "gemini attachment fallback: disabled by JOB_SCOUT_ENABLE_GEMINI_RESUME_FALLBACK"
        )
        return {
            "status": "error",
            "message": (
                "Resume parsing failed without using Gemini. Try uploading a text-based "
                "PDF/DOCX, install an optional PDF parser such as pypdf, or configure a "
                "non-Gemini resume parser model."
            ),
            "resume_source": selected_artifact,
            "details": details,
        }

    return {
        "status": "ok",
        "profile": profile,
        "resume_source": profile["resume_source"],
        "extraction_method": (
            extraction_method
            or ("llm_text_fallback" if text_fallback_error is None and resume_text else "local_best_effort")
        ),
    }


async def _ensure_resume_profile(tool_context: ToolContext) -> Optional[dict]:
    profile = tool_context.state.get(RESUME_PROFILE_STATE_KEY)
    if profile:
        return ResumeProfile.model_validate(profile).model_dump()

    extracted = await _extract_resume_profile_payload_from_artifact(
        tool_context=tool_context,
    )
    if extracted.get("status") != "ok":
        return None

    profile = extracted["profile"]
    _persist_resume_profile(
        tool_context,
        profile,
        extraction_method=extracted["extraction_method"],
    )
    return profile


async def get_resume_status(tool_context: ToolContext) -> dict:
    """Return whether a structured resume profile is available in session state.

    Use this near the start of a conversation to decide whether the agent can
    operate in resume-aware mode or should remain in prompt-only mode.

    Args:
        tool_context: ADK tool context used to read cached resume state.

    Returns:
        A small status payload indicating whether a resume profile is loaded,
        along with basic metadata when it exists.
    """
    profile = tool_context.state.get(RESUME_PROFILE_STATE_KEY)
    if not profile:
        return {"status": "missing", "resume_loaded": False}

    profile = ResumeProfile.model_validate(profile).model_dump()

    return {
        "status": "ok",
        "resume_loaded": True,
        "candidate_name": profile.get("candidate_name"),
        "role_titles": profile.get("role_titles", []),
        "core_skills": profile.get("core_skills", []),
        "years_experience": profile.get("years_experience"),
        "resume_source": profile.get("resume_source"),
        "extraction_method": tool_context.state.get(RESUME_PROFILE_EXTRACTION_METHOD_STATE_KEY),
    }


def clear_resume_profile(tool_context: ToolContext) -> dict:
    """Remove the saved resume profile from the current session.

    Args:
        tool_context: ADK tool context whose session state should be cleared.

    Returns:
        A confirmation payload indicating the stored resume profile was cleared.
    """
    tool_context.state[RESUME_PROFILE_STATE_KEY] = None
    tool_context.state[RESUME_PROFILE_EXTRACTION_METHOD_STATE_KEY] = None
    return {"status": "ok", "message": "Stored resume profile cleared."}


def save_resume_profile(
    candidate_name: str,
    professional_summary: str,
    role_titles: list[str],
    core_skills: list[str],
    additional_skills: list[str],
    years_experience: Optional[float],
    preferred_locations: list[str],
    work_preferences: list[str],
    notable_projects: list[str],
    resume_source: str,
    tool_context: ToolContext,
) -> dict:
    """Save a normalized resume profile in session state.

    Use this when a resume has already been read by some other process and the
    caller needs to persist the extracted fields for later scoring.

    Args:
        candidate_name: Candidate name found in the resume.
        professional_summary: Candidate summary or profile headline.
        role_titles: Inferred or explicit role titles from the resume.
        core_skills: Primary supported skills from the resume.
        additional_skills: Secondary tools or skills from the resume.
        years_experience: Estimated years of experience, if known.
        preferred_locations: Locations mentioned or preferred by the candidate.
        work_preferences: Work setup preferences like remote or hybrid.
        notable_projects: Key projects or achievements from the resume.
        resume_source: File name or label for the resume source.
        tool_context: ADK tool context used to persist the profile.

    Returns:
        A success or error payload. On success, the normalized profile is saved
        into the current session for later matching and scoring.
    """
    try:
        profile = resume_support._normalize_resume_profile_payload(
            {
                "candidate_name": candidate_name,
                "professional_summary": professional_summary,
                "role_titles": role_titles,
                "core_skills": core_skills,
                "additional_skills": additional_skills,
                "years_experience": years_experience,
                "preferred_locations": preferred_locations,
                "work_preferences": work_preferences,
                "notable_projects": notable_projects,
            },
            resume_source=resume_source,
        )
    except ValueError as exc:
        return {
            "status": "error",
            "message": str(exc),
        }

    if not resume_support._profile_has_meaningful_resume_signal(profile):
        return {
            "status": "error",
            "message": (
                "Resume profile is too empty or contains placeholder values only. "
                "Read the uploaded resume again and save only fields that are actually present."
            ),
        }

    _persist_resume_profile(tool_context, profile, extraction_method="manual")
    return {
        "status": "ok",
        "message": "Resume profile saved in session state.",
        "candidate_name": profile["candidate_name"],
        "role_titles": profile["role_titles"],
        "skills_count": len(profile["core_skills"]) + len(profile["additional_skills"]),
    }


async def score_job_match(job_title: str, job_description: str, tool_context: ToolContext) -> dict:
    """Score a job against the saved resume profile in the current session.

    Use this only when a resume profile exists or can be restored from the
    current session's uploaded artifact.

    Args:
        job_title: Title of the job posting to evaluate.
        job_description: Full job description text to score.
        tool_context: ADK tool context used to read resume state.

    Returns:
        A deterministic scoring payload with a score, signal breakdown, matched
        skills, missing skills, and a short explanation.
    """
    profile = await _ensure_resume_profile(tool_context)
    if not profile:
        return {
            "status": "error",
            "message": (
                "No resume profile is loaded yet, and no uploaded resume artifact "
                "could be restored for scoring. If the user wants resume-based "
                "matching, upload a resume first or save a structured profile with "
                "`save_resume_profile`."
            ),
        }

    result = score_resume_vs_jd(
        job_title=job_title,
        job_description=job_description,
        profile=profile,
    )
    result["resume_source"] = profile.get("resume_source")
    return JobMatchScore.model_validate(result).model_dump(exclude_none=True)


def _infer_search_role_from_profile(profile: dict) -> str:
    role_titles = [role.strip() for role in profile.get("role_titles", []) if role and role.strip()]
    if role_titles:
        return ", ".join(role_titles[:3])

    skills = [skill.strip() for skill in profile.get("core_skills", []) if skill and skill.strip()]
    if any("react" in skill.lower() for skill in skills):
        return "Frontend Developer"
    if any("node" in skill.lower() for skill in skills):
        return "Backend Developer"
    return "Software Developer"


def _infer_search_location_from_profile(profile: dict) -> str:
    locations = [location.strip() for location in profile.get("preferred_locations", []) if location and location.strip()]
    if locations:
        return locations[0]
    return "Remote"


def _job_description_needs_expansion(job: dict) -> bool:
    description = (job.get("description") or "").strip()
    if len(description) >= 500:
        return False

    snippet = (job.get("snippet") or "").strip()
    if len(snippet) < 180:
        return True

    vague_markers = (
        "see job description",
        "click to apply",
        "read more",
        "apply now",
    )
    snippet_lower = snippet.lower()
    return any(marker in snippet_lower for marker in vague_markers)


def _summarize_job_description_for_output(job_description: str, *, max_chars: int = 420) -> str:
    cleaned = " ".join((job_description or "").split())
    if not cleaned:
        return ""

    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", cleaned)
        if sentence.strip()
    ]
    summary = " ".join(sentences[:3]) if sentences else cleaned
    if len(summary) <= max_chars:
        return summary
    return summary[:max_chars].rsplit(" ", 1)[0].rstrip(".,;:") + "..."


def _score_job_payload_against_profile(job: dict, profile: dict) -> dict:
    enriched_job = dict(job)

    if enriched_job.get("url") and _job_description_needs_expansion(enriched_job):
        details = fetch_job_details(enriched_job["url"])
        if details.get("status") == "ok":
            description = (details.get("description") or "").strip()
            if description:
                enriched_job["description"] = description[:4000]

    job_description = (
        (enriched_job.get("description") or "").strip()
        or (enriched_job.get("snippet") or "").strip()
    )
    score_result = score_resume_vs_jd(
        job_title=enriched_job.get("title", ""),
        job_description=job_description,
        profile=profile,
    )
    score_result["resume_source"] = profile.get("resume_source")
    normalized_score = JobMatchScore.model_validate(score_result).model_dump(exclude_none=True)

    return JobPosting.model_validate({
        **enriched_job,
        "description": enriched_job.get("description"),
    }).model_dump(exclude_none=True) | {
        "apply_url": enriched_job.get("url"),
        "job_description_summary": _summarize_job_description_for_output(job_description),
        "score": normalized_score["score"],
        "matched_skills": normalized_score.get("matched_skills", []),
        "missing_skills": normalized_score.get("missing_skills", []),
        "fit_verdict": normalized_score.get("fit_verdict"),
        "reason_to_apply": normalized_score.get("reason_to_apply"),
        "before_applying": normalized_score.get("before_applying", []),
        "blocker_risk": normalized_score.get("blocker_risk"),
        "score_explanation": normalized_score.get("explanation", ""),
        "score_signals": normalized_score.get("signals", {}),
        "score_evidence": normalized_score.get("evidence", {}),
        "score_blockers": normalized_score.get("blockers", {}),
    }


async def score_saved_jobs(tool_context: ToolContext) -> dict:
    """Score every job from the most recent saved search against the resume.

    Use this for follow-ups like "score all these jobs" after a job search has
    already returned results. It preserves the job URL, company, location,
    description context, score evidence, and before-applying guidance.
    """
    profile = await _ensure_resume_profile(tool_context)
    if not profile:
        return {
            "status": "error",
            "message": (
                "No resume profile is loaded yet, so saved jobs cannot be scored. "
                "Upload or save a resume profile first."
            ),
            "jobs": [],
        }

    saved_search = tool_context.state.get(LAST_SEARCH_RESULTS_STATE_KEY)
    saved_jobs = (saved_search or {}).get("jobs", [])
    if not saved_jobs:
        return {
            "status": "error",
            "message": "No saved job search results are available to score. Search for jobs first.",
            "jobs": [],
        }

    scored_jobs = [
        _score_job_payload_against_profile(dict(job), profile)
        for job in saved_jobs
    ]
    scored_jobs.sort(key=lambda job: job.get("score", 0), reverse=True)

    return {
        "status": "ok",
        "message": f"Scored all {len(scored_jobs)} saved jobs against the resume.",
        "candidate_name": profile.get("candidate_name"),
        "resume_source": profile.get("resume_source"),
        "extraction_method": tool_context.state.get(RESUME_PROFILE_EXTRACTION_METHOD_STATE_KEY),
        "search_context": {
            "role": (saved_search or {}).get("role"),
            "expanded_roles": (saved_search or {}).get("expanded_roles", []),
            "location": (saved_search or {}).get("location"),
            "country": (saved_search or {}).get("country"),
        },
        "jobs": scored_jobs,
    }


async def find_resume_matched_jobs(
    location: Optional[str] = None,
    role: Optional[str] = None,
    max_results: int = 25,
    country: Optional[str] = None,
    min_years: Optional[float] = None,
    max_years: Optional[float] = None,
    tool_context: Optional[ToolContext] = None,
) -> dict:
    """Run resume extraction/restoration, job search, and scoring in one tool call.

    Use this when the user wants jobs matched according to an uploaded or saved
    resume. This is especially useful with providers that are less reliable at
    multi-step tool chaining.
    """
    if tool_context is None:
        return {
            "status": "error",
            "message": "Tool context is required for resume-based job matching.",
            "jobs": [],
        }

    profile = await _ensure_resume_profile(tool_context)
    if not profile:
        return {
            "status": "error",
            "message": (
                "No resume profile is loaded yet, and no uploaded resume artifact "
                "could be restored for matching. Upload a resume first or save a "
                "structured profile with `save_resume_profile`."
            ),
            "jobs": [],
        }

    resolved_role = (role or "").strip() or _infer_search_role_from_profile(profile)
    resolved_location = (location or "").strip() or _infer_search_location_from_profile(profile)

    search_result = search_jobs(
        role=resolved_role,
        location=resolved_location,
        max_results=max_results,
        country=country,
        min_years=min_years,
        max_years=max_years,
        tool_context=tool_context,
    )
    if search_result.get("status") not in {"ok", "demo_mode"}:
        return search_result

    ranked_jobs = []
    for raw_job in search_result.get("jobs", []):
        ranked_jobs.append(_score_job_payload_against_profile(dict(raw_job), profile))

    ranked_jobs.sort(key=lambda job: job.get("score", 0), reverse=True)

    return {
        "status": search_result.get("status", "ok"),
        "message": (
            f"Found {len(ranked_jobs)} resume-matched jobs for {resolved_role} in {resolved_location}."
        ),
        "candidate_name": profile.get("candidate_name"),
        "resume_source": profile.get("resume_source"),
        "extraction_method": tool_context.state.get(RESUME_PROFILE_EXTRACTION_METHOD_STATE_KEY),
        "role_used": resolved_role,
        "location_used": resolved_location,
        "provider": search_result.get("provider"),
        "expanded_roles": search_result.get("expanded_roles", []),
        "experience_filter": search_result.get("experience_filter"),
        "jobs": ranked_jobs,
    }
