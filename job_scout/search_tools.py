from __future__ import annotations

import os
import re
from typing import Optional

from google.adk.tools.tool_context import ToolContext

from job_scout.domain_models import JobPosting
from job_scout.domain_models import SearchContext
from job_scout.search_support import _extract_experience_requirements
from job_scout.search_support import _is_entry_level_text
from job_scout.search_support import _is_senior_level_text
from job_scout.search_support import _matches_requested_experience
from job_scout.search_support import _merge_job_lists
from job_scout.search_support import _normalize_country
from job_scout.search_support import _resolve_job_search_provider
from job_scout.search_support import _search_jobs_once
from job_scout.search_support import fetch_job_details
from job_scout.state_keys import LAST_SEARCH_RESULTS_STATE_KEY

_ROLE_ALIAS_PATTERNS = (
    (re.compile(r"\bfull[\s-]*stack\b"), "Full Stack Developer"),
    (re.compile(r"\bback[\s-]*end\b"), "Backend Developer"),
    (re.compile(r"\bfront[\s-]*end\b"), "Frontend Developer"),
)


def _dedupe(items: list[str]) -> list[str]:
    seen = set()
    output = []
    for item in items:
        value = item.strip()
        if not value:
            continue
        key = value.lower()
        if key in seen:
            continue
        seen.add(key)
        output.append(value)
    return output


def _split_requested_roles(role: str) -> list[str]:
    normalized_role = " ".join((role or "").split())
    if not normalized_role:
        return []

    detected_roles = []
    normalized_lower = normalized_role.lower()
    for pattern, canonical_role in _ROLE_ALIAS_PATTERNS:
        if pattern.search(normalized_lower):
            detected_roles.append(canonical_role)

    if detected_roles:
        return _dedupe(detected_roles)

    split_candidates = re.split(r"\s*(?:,|/|&|\band\b|\bor\b)\s*", normalized_role, flags=re.IGNORECASE)
    cleaned_candidates = _dedupe([candidate.strip() for candidate in split_candidates if candidate.strip()])
    return cleaned_candidates or [normalized_role]


def _build_job_summary_fields(
    *,
    requested_role: str,
    job_title: str,
    min_years: Optional[float],
    max_years: Optional[float],
    explicit_entry_level: bool = False,
    explicit_senior_level: bool = False,
) -> dict[str, str]:
    title_lower = (job_title or "").lower()
    role_lower = (requested_role or "").lower()
    role_overlap = any(token and token in title_lower for token in role_lower.split())

    if explicit_senior_level and max_years is not None and max_years <= 1:
        blocker_risk = "High"
        fit_verdict = "Risky fit"
        reason_to_apply = "The role lines up on title, but the seniority signal is likely above the requested range."
    elif explicit_entry_level or (max_years is not None and max_years <= 1):
        blocker_risk = "Low"
        fit_verdict = "Good early-career fit"
        reason_to_apply = "The role appears aligned to an entry-level or junior search."
    elif role_overlap:
        blocker_risk = "Medium"
        fit_verdict = "Relevant match"
        reason_to_apply = "The job title aligns closely with the requested role."
    else:
        blocker_risk = "Medium"
        fit_verdict = "Potential match"
        reason_to_apply = "The role may still be relevant, but needs deeper review of the job description."

    return {
        "fit_verdict": fit_verdict,
        "reason_to_apply": reason_to_apply,
        "blocker_risk": blocker_risk,
    }


def _validate_search_inputs(
    *,
    max_results: int,
    min_years: Optional[float],
    max_years: Optional[float],
) -> Optional[str]:
    if max_results < 1:
        return "max_results must be greater than or equal to 1."
    if min_years is not None and min_years < 0:
        return "min_years must be greater than or equal to 0."
    if max_years is not None and max_years < 0:
        return "max_years must be greater than or equal to 0."
    if min_years is not None and max_years is not None and min_years > max_years:
        return "min_years must be less than or equal to max_years."
    return None


def search_jobs(
    role: str,
    location: str,
    max_results: int = 5,
    country: Optional[str] = None,
    min_years: Optional[float] = None,
    max_years: Optional[float] = None,
    tool_context: Optional[ToolContext] = None,
) -> dict:
    """Search job postings for the requested role and location.

    Use this in prompt-only mode or resume-aware mode whenever the user asks
    to find jobs.

    Args:
        role: Target role such as ``Frontend Developer`` or ``Data Engineer``.
        location: Requested location such as ``Hyderabad`` or ``Remote``.
        max_results: Maximum number of jobs to return.
        country: Optional country override such as ``IN`` or ``US``.
        min_years: Optional minimum experience requested by the user.
        max_years: Optional maximum experience requested by the user.
        tool_context: Optional ADK tool context used to persist the last search.

    Returns:
        A result payload containing status, selected jobs, and search metadata.
        When no supported provider credentials are configured, the function
        returns demo data.
    """
    validation_error = _validate_search_inputs(
        max_results=max_results,
        min_years=min_years,
        max_years=max_years,
    )
    if validation_error:
        return {
            "status": "error",
            "message": validation_error,
            "jobs": [],
        }

    apify_token = os.getenv("APIFY_TOKEN")
    adzuna_app_id = os.getenv("ADZUNA_APP_ID")
    adzuna_app_key = os.getenv("ADZUNA_APP_KEY")
    provider = _resolve_job_search_provider()
    resolved_country = _normalize_country(location, country)
    requested_roles = _split_requested_roles(role)
    if not requested_roles:
        requested_roles = [role.strip()] if role.strip() else ["Generalist"]

    if provider == "demo":
        demo_jobs = []
        for index in range(1, max_results + 1):
            requested_role = requested_roles[(index - 1) % len(requested_roles)]
            demo_title = requested_role if max_years is None or max_years > 1 else f"Junior {requested_role}"
            summary_fields = _build_job_summary_fields(
                requested_role=requested_role,
                job_title=demo_title,
                min_years=min_years,
                max_years=max_years,
                explicit_entry_level=max_years is not None and max_years <= 1,
            )
            demo_jobs.append(
                JobPosting.model_validate(
                    {
                        "id": f"demo-{index}",
                        "title": demo_title,
                        "company": "TechCorp",
                        "location": location,
                        "snippet": (
                            f"Looking for {demo_title} with "
                            f"{min_years or 0:g}-{max_years or 2:g} years of experience."
                        ),
                        "url": f"https://example.com/job/demo-{index}",
                        "matched_role": requested_role,
                        **summary_fields,
                    }
                ).model_dump(exclude_none=True)
            )
        result = {
            "status": "demo_mode",
            "jobs": demo_jobs,
            "note": "No job search provider credentials are configured; returning demo data.",
            "expanded_roles": requested_roles,
            "provider": "demo",
        }
        if tool_context:
            tool_context.state[LAST_SEARCH_RESULTS_STATE_KEY] = SearchContext.model_validate({
                "role": role,
                "expanded_roles": requested_roles,
                "location": location,
                "country": resolved_country,
                "jobs": result["jobs"],
            }).model_dump(exclude_none=True)
        return result

    try:
        per_role_limit = max(1, min(5, (max_results + len(requested_roles) - 1) // max(1, len(requested_roles))))
        search_batches = [
            _search_jobs_once(
                role=requested_role,
                location=location,
                max_results=per_role_limit,
                resolved_country=resolved_country,
                min_years=min_years,
                max_years=max_years,
                apify_token=apify_token,
                adzuna_app_id=adzuna_app_id,
                adzuna_app_key=adzuna_app_key,
            )
            for requested_role in requested_roles
        ]
        selected_jobs = _merge_job_lists(
            [batch["jobs"] for batch in search_batches],
            max_results=max_results,
        )

        result = {
            "status": "ok",
            "jobs": [JobPosting.model_validate(job).model_dump(exclude_none=True) for job in selected_jobs],
            "country_used": next(
                (batch.get("country_used") for batch in search_batches if batch.get("country_used")),
                resolved_country,
            ),
            "query_used": [batch["query_used"] for batch in search_batches],
            "expanded_roles": requested_roles,
            "provider": provider,
            "experience_filter": {
                "min_years": min_years,
                "max_years": max_years,
            },
            "filter_note": (
                "Applied strict entry-level experience filtering to the search results."
                if min_years is not None or max_years is not None
                else None
            ),
        }
        if tool_context:
            tool_context.state[LAST_SEARCH_RESULTS_STATE_KEY] = SearchContext.model_validate({
                "role": role,
                "expanded_roles": requested_roles,
                "location": location,
                "country": resolved_country,
                "jobs": result["jobs"],
            }).model_dump(exclude_none=True)
        return result
    except Exception as exc:
        return {"status": "error", "message": str(exc), "jobs": []}


def filter_saved_jobs_by_experience(
    min_years: float,
    max_years: float,
    tool_context: ToolContext,
) -> dict:
    """Filter the most recently searched jobs by experience requirement.

    Use this for follow-up requests like ``show only 0-1 year roles`` after a
    search has already been performed.

    Args:
        min_years: Minimum acceptable experience for the user.
        max_years: Maximum acceptable experience for the user.
        tool_context: ADK tool context used to read the last saved search.

    Returns:
        A payload containing matching jobs, jobs with unknown experience, and
        the requested range that was applied.
    """
    validation_error = _validate_search_inputs(
        max_results=1,
        min_years=min_years,
        max_years=max_years,
    )
    if validation_error:
        return {
            "status": "error",
            "message": validation_error,
            "jobs": [],
        }

    saved_search = tool_context.state.get(LAST_SEARCH_RESULTS_STATE_KEY)
    if not saved_search or not saved_search.get("jobs"):
        return {
            "status": "error",
            "message": "No previous job search is available yet. Search for jobs first.",
            "jobs": [],
        }

    matched_jobs = []
    unknown_jobs = []

    for job in saved_search["jobs"]:
        text_for_analysis = " ".join(
            part for part in (job.get("title", ""), job.get("snippet", "")) if part
        )
        requirement = _extract_experience_requirements(text_for_analysis)
        description = None
        explicit_entry_level = _is_entry_level_text(text_for_analysis)
        explicit_senior_level = _is_senior_level_text(text_for_analysis)

        if not requirement["detected"] and job.get("url"):
            details = fetch_job_details(job["url"])
            if details.get("status") == "ok":
                description = details.get("description", "")
                requirement = _extract_experience_requirements(description)
                explicit_entry_level = explicit_entry_level or _is_entry_level_text(description)
                explicit_senior_level = explicit_senior_level or _is_senior_level_text(description)

        summary_fields = _build_job_summary_fields(
            requested_role=saved_search.get("role", ""),
            job_title=job.get("title", ""),
            min_years=min_years,
            max_years=max_years,
            explicit_entry_level=explicit_entry_level,
            explicit_senior_level=explicit_senior_level,
        )

        enriched_job = JobPosting.model_validate({
            "id": job.get("id"),
            "title": job.get("title"),
            "company": job.get("company"),
            "location": job.get("location"),
            "url": job.get("url"),
            "snippet": job.get("snippet"),
            "experience_min_years": requirement.get("min_years"),
            "experience_max_years": requirement.get("max_years"),
            "experience_evidence": requirement.get("evidence"),
            **summary_fields,
        }).model_dump(exclude_none=True)
        if description:
            enriched_job["description"] = description[:2500]

        if max_years <= 1 and explicit_senior_level and not explicit_entry_level:
            continue

        if _matches_requested_experience(requirement, min_years, max_years):
            matched_jobs.append(enriched_job)
        elif max_years <= 1 and explicit_entry_level:
            matched_jobs.append(enriched_job)
        elif not requirement["detected"]:
            unknown_jobs.append(enriched_job)

    return {
        "status": "ok",
        "requested_experience_range": {
            "min_years": min_years,
            "max_years": max_years,
        },
        "search_context": {
            "role": saved_search.get("role"),
            "location": saved_search.get("location"),
            "country": saved_search.get("country"),
        },
        "jobs": matched_jobs,
        "unknown_experience_jobs": unknown_jobs,
        "message": (
            f"Found {len(matched_jobs)} jobs matching the requested experience range "
            f"{min_years:g}-{max_years:g} years."
        ),
    }
