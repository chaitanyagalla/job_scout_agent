from __future__ import annotations

from typing import Optional

from google.adk.tools.tool_context import ToolContext

from job_scout.domain_models import JobMatchScore
from job_scout.domain_models import ResumeProfile
from job_scout import resume_support
from job_scout.scoring import score_resume_vs_jd
from job_scout.state_keys import RESUME_PROFILE_EXTRACTION_METHOD_STATE_KEY
from job_scout.state_keys import RESUME_PROFILE_STATE_KEY


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
        try:
            extracted = await resume_support._parse_resume_with_gemini(selected_artifact, artifact_part)
            profile = _normalize_extracted_profile(extracted)
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

    return {
        "status": "ok",
        "profile": profile,
        "resume_source": profile["resume_source"],
        "extraction_method": "llm_text_fallback" if text_fallback_error is None and resume_text else "gemini_attachment_fallback",
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
    profile = await _ensure_resume_profile(tool_context)
    if not profile:
        return {"status": "missing", "resume_loaded": False}

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
