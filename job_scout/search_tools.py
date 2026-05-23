from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
import os
import re
from typing import Optional

from google.adk.tools.tool_context import ToolContext

from job_scout.domain_models import JobPosting
from job_scout.domain_models import SearchContext
from job_scout.input_normalization import normalize_role_title
from job_scout.search_support import _extract_experience_requirements
from job_scout.search_support import _is_entry_level_text
from job_scout.search_support import _is_senior_level_text
from job_scout.search_support import _matches_requested_experience
from job_scout.search_support import _merge_job_lists
from job_scout.search_support import _normalize_country
from job_scout.search_support import _resolve_job_search_provider
from job_scout.search_support import _resolve_job_search_providers
from job_scout.search_support import _search_jobs_once
from job_scout.search_support import fetch_job_details
from job_scout.state_keys import LAST_SEARCH_RESULTS_STATE_KEY

_ROLE_ALIAS_PATTERNS = (
    (re.compile(r"\bfull[\s-]*stack\b"), "Full Stack Developer"),
    (re.compile(r"\bback[\s-]*end\b"), "Backend Developer"),
    (re.compile(r"\bfront[\s-]*end\b"), "Frontend Developer"),
    (re.compile(r"\bmern\b"), "MERN Stack Developer"),
    (re.compile(r"\b(?:gen[\s-]*ai|ai|llm)\b"), "GenAI Engineer"),
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

    split_candidates = re.split(r"\s*(?:,|/|&|\band\b|\bor\b)\s*", normalized_role, flags=re.IGNORECASE)
    if len([candidate for candidate in split_candidates if candidate.strip()]) <= 1:
        detected_roles = []
        normalized_lower = normalized_role.lower()
        for pattern, canonical_role in _ROLE_ALIAS_PATTERNS:
            if pattern.search(normalized_lower):
                detected_roles.append(canonical_role)
        if detected_roles:
            return _dedupe(detected_roles)

    cleaned_candidates = []
    for candidate in split_candidates:
        if not candidate.strip():
            continue

        candidate_roles = []
        candidate_lower = candidate.lower()
        for pattern, canonical_role in _ROLE_ALIAS_PATTERNS:
            if pattern.search(candidate_lower):
                candidate_roles.append(canonical_role)

        if candidate_roles:
            cleaned_candidates.extend(candidate_roles)
            continue

        normalized_candidate = normalize_role_title(candidate)
        if normalized_candidate:
            cleaned_candidates.append(normalized_candidate)

    cleaned_candidates = _dedupe(cleaned_candidates)
    return cleaned_candidates or [normalize_role_title(normalized_role) or normalized_role]


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


def _search_requested_roles(
    *,
    provider: str,
    requested_roles: list[str],
    location: str,
    max_results: int,
    resolved_country: Optional[str],
    min_years: Optional[float],
    max_years: Optional[float],
    apify_token: Optional[str],
    adzuna_app_id: Optional[str],
    adzuna_app_key: Optional[str],
) -> list[dict]:
    per_role_limit = max_results

    def run_search(requested_role: str) -> dict:
        batch = _search_jobs_once(
            role=requested_role,
            location=location,
            max_results=per_role_limit,
            resolved_country=resolved_country,
            min_years=min_years,
            max_years=max_years,
            apify_token=apify_token,
            adzuna_app_id=adzuna_app_id,
            adzuna_app_key=adzuna_app_key,
            provider=provider,
        )
        batch["requested_role"] = requested_role
        batch["requested_results"] = per_role_limit
        return batch

    if len(requested_roles) == 1:
        return [run_search(requested_roles[0])]

    max_workers = max(
        1,
        min(
            len(requested_roles),
            int(os.getenv("JOB_SCOUT_SEARCH_CONCURRENCY") or "3"),
        ),
    )
    ordered_batches: list[Optional[dict]] = [None] * len(requested_roles)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(run_search, requested_role): index
            for index, requested_role in enumerate(requested_roles)
        }
        for future in as_completed(futures):
            ordered_batches[futures[future]] = future.result()

    return [batch for batch in ordered_batches if batch is not None]


def _effective_max_results(max_results: int, requested_roles: list[str]) -> int:
    if len(requested_roles) <= 1:
        return max_results
    return max_results * len(requested_roles)


def _build_role_result_counts(search_batches: list[dict]) -> dict[str, int]:
    return {
        str(batch.get("requested_role") or ""): len(batch.get("jobs", []) or [])
        for batch in search_batches
        if batch.get("requested_role")
    }


def _build_role_result_shortfalls(search_batches: list[dict]) -> dict[str, dict[str, int]]:
    shortfalls = {}
    for batch in search_batches:
        requested_role = str(batch.get("requested_role") or "")
        requested_results = int(batch.get("requested_results") or 0)
        found_results = len(batch.get("jobs", []) or [])
        if requested_role and requested_results > found_results:
            shortfalls[requested_role] = {
                "requested": requested_results,
                "found": found_results,
            }
    return shortfalls


def _format_provider_attempt_errors(provider_attempt_errors: dict[str, str]) -> str:
    if not provider_attempt_errors:
        return ""

    return "; ".join(
        f"{provider}: {message}"
        for provider, message in provider_attempt_errors.items()
    )


def _sanitize_provider_error(message: str) -> str:
    sanitized = str(message)
    for secret_name in ("APIFY_TOKEN", "ADZUNA_APP_KEY", "BROWSERACT_API_KEY"):
        secret = (os.getenv(secret_name) or "").strip()
        if secret:
            sanitized = sanitized.replace(secret, "[redacted]")

    sanitized = re.sub(r"(?i)(token|app_key)=([^&\s]+)", r"\1=[redacted]", sanitized)
    return sanitized


def search_jobs(
    role: str,
    location: str,
    max_results: int = 25,
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
        max_results: Maximum jobs to return for a single role. Defaults to 25.
            When multiple
            role families are requested, the tool returns up to this many jobs
            for each expanded role.
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
    provider_candidates = _resolve_job_search_providers()
    resolved_country = _normalize_country(location, country)
    requested_roles = _split_requested_roles(role)
    if not requested_roles:
        requested_roles = [role.strip()] if role.strip() else ["Generalist"]
    effective_max_results = _effective_max_results(max_results, requested_roles)

    if provider == "demo":
        demo_jobs = []
        for index in range(1, effective_max_results + 1):
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
            "requested_results_per_role": max_results,
            "requested_results_total": effective_max_results,
            "role_result_counts": {
                requested_role: len([
                    job for job in demo_jobs if job.get("matched_role") == requested_role
                ])
                for requested_role in requested_roles
            },
            "role_result_shortfalls": {},
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

    provider_attempt_errors: dict[str, str] = {}
    for provider_candidate in provider_candidates:
        try:
            search_batches = _search_requested_roles(
                provider=provider_candidate,
                requested_roles=requested_roles,
                location=location,
                max_results=max_results,
                resolved_country=resolved_country,
                min_years=min_years,
                max_years=max_years,
                apify_token=apify_token,
                adzuna_app_id=adzuna_app_id,
                adzuna_app_key=adzuna_app_key,
            )
            provider = next(
                (batch.get("provider") for batch in search_batches if batch.get("provider")),
                provider_candidate,
            )
        except Exception as exc:
            provider_attempt_errors[provider_candidate] = _sanitize_provider_error(str(exc))
            continue

        selected_jobs = _merge_job_lists(
            [batch["jobs"] for batch in search_batches],
            max_results=effective_max_results,
        )
        role_result_shortfalls = _build_role_result_shortfalls(search_batches)

        fallback_note = None
        if provider_attempt_errors:
            fallback_note = (
                "The first configured job search provider failed, so another "
                "configured provider was used."
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
            "requested_results_per_role": max_results,
            "requested_results_total": effective_max_results,
            "role_result_counts": _build_role_result_counts(search_batches),
            "role_result_shortfalls": role_result_shortfalls,
            "experience_filter": {
                "min_years": min_years,
                "max_years": max_years,
            },
            "filter_note": (
                "Applied requested experience filtering to the search results."
                if min_years is not None or max_years is not None
                else None
            ),
            "shortfall_note": (
                "One or more role families returned fewer jobs than requested from the configured provider."
                if role_result_shortfalls
                else None
            ),
            "fallback_note": fallback_note,
            "provider_attempt_errors": provider_attempt_errors or None,
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

    provider_attempt_error_summary = _format_provider_attempt_errors(provider_attempt_errors)
    message = "All configured job search providers failed."
    if provider_attempt_error_summary:
        message = f"{message} {provider_attempt_error_summary}"
    return {"status": "error", "message": message, "jobs": []}


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
