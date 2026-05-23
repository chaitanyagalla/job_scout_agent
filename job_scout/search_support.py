from __future__ import annotations

import json
import os
import re
import time
from typing import Optional
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup

DEFAULT_APIFY_ACTOR_TIMEOUT_SECONDS = 300
DEFAULT_APIFY_HTTP_TIMEOUT_SECONDS = 360

_EXPERIENCE_RANGE_PATTERNS = [
    re.compile(r"(\d+(?:\.\d+)?)\s*(?:to|-)\s*(\d+(?:\.\d+)?)\s+years?"),
    re.compile(r"(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s+yrs?"),
]
_EXPERIENCE_MIN_PATTERNS = [
    re.compile(r"minimum\s+of\s+(\d+(?:\.\d+)?)\+?\s+years?"),
    re.compile(r"minimum\s+(\d+(?:\.\d+)?)\+?\s+years?"),
    re.compile(r"at\s+least\s+(\d+(?:\.\d+)?)\+?\s+years?"),
    re.compile(r"(\d+(?:\.\d+)?)\+\s+years?"),
]
_EXPERIENCE_EXACT_PATTERNS = [
    re.compile(r"(\d+(?:\.\d+)?)\s+years?\s+of\s+experience"),
    re.compile(r"experience\s+of\s+(\d+(?:\.\d+)?)\s+years?"),
]
_ENTRY_LEVEL_TERMS = (
    "fresher", "freshers", "entry level", "entry-level", "junior",
    "graduate", "graduate trainee", "intern", "internship", "trainee",
    "associate", "0-1 year", "0 to 1 year", "0 - 1 year",
)
_REMOTE_LOCATION_TERMS = (
    "remote",
    "work from home",
    "wfh",
    "anywhere",
)


def _resolve_positive_int_env(name: str, default_value: int) -> int:
    raw_value = (os.getenv(name) or "").strip()
    if not raw_value:
        return default_value
    try:
        value = int(raw_value)
    except ValueError:
        return default_value
    return value if value > 0 else default_value


def _resolve_apify_actor_timeout_seconds() -> int:
    return _resolve_positive_int_env(
        "JOB_SCOUT_APIFY_ACTOR_TIMEOUT_SECONDS",
        DEFAULT_APIFY_ACTOR_TIMEOUT_SECONDS,
    )


def _resolve_apify_http_timeout_seconds() -> int:
    actor_timeout = _resolve_apify_actor_timeout_seconds()
    configured_timeout = _resolve_positive_int_env(
        "JOB_SCOUT_APIFY_HTTP_TIMEOUT_SECONDS",
        DEFAULT_APIFY_HTTP_TIMEOUT_SECONDS,
    )
    return max(configured_timeout, actor_timeout + 30)
_REMOTE_JOB_TERMS = (
    "remote",
    "work from home",
    "wfh",
    "telecommute",
    "anywhere",
)
_SENIOR_LEVEL_TERMS = (
    "senior", "sr.", "sr ", "lead", "principal", "staff engineer",
    "architect", "manager", "head of",
)
_MERN_ROLE_TERMS = (
    "mern",
    "full stack",
    "fullstack",
    "react",
    "node",
    "node.js",
    "javascript",
    "typescript",
    "web developer",
)
_GENAI_ROLE_TERMS = (
    "genai",
    "gen ai",
    "generative ai",
    "llm",
    "large language model",
    "prompt engineer",
    "machine learning",
    "ml engineer",
    "nlp",
)
_MERN_UNRELATED_TITLE_TERMS = (
    "php",
    "laravel",
    "codeigniter",
    "rpa",
    "content developer",
    "curriculum",
)
_GENAI_UNRELATED_TITLE_TERMS = (
    "content developer",
    "curriculum",
    "rpa",
    "php",
    "laravel",
)
_COUNTRY_CODE_ALIASES = {
    "US": {"us", "usa"},
    "IN": {"in"},
    "GB": {"gb", "uk"},
    "CA": {"ca"},
    "AU": {"au", "aus"},
}
_AMBIGUOUS_COUNTRY_ALIASES = {"in"}
_COUNTRY_PHRASE_ALIASES = {
    "US": ("united states", "united states of america"),
    "IN": ("india",),
    "GB": ("united kingdom", "great britain"),
    "CA": ("canada",),
    "AU": ("australia",),
}
_DEFAULT_ADZUNA_COUNTRY = "gb"


def _normalize_country(location: str, country: Optional[str]) -> Optional[str]:
    def _match_country(raw_value: str, *, allow_ambiguous_aliases: bool) -> Optional[str]:
        normalized_text = " ".join(re.findall(r"[a-z]+", raw_value))
        if not normalized_text:
            return None

        tokens = set(normalized_text.split())
        for code, phrases in _COUNTRY_PHRASE_ALIASES.items():
            for phrase in phrases:
                if re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", normalized_text):
                    return code

        for code, aliases in _COUNTRY_CODE_ALIASES.items():
            for alias in aliases:
                if alias not in tokens:
                    continue
                if allow_ambiguous_aliases or alias not in _AMBIGUOUS_COUNTRY_ALIASES:
                    return code
                if normalized_text == alias:
                    return code
        return None

    explicit_country = " ".join((country or "").strip().lower().split())
    if explicit_country:
        return _match_country(explicit_country, allow_ambiguous_aliases=True)

    normalized_location = " ".join((location or "").strip().lower().split())
    if not normalized_location:
        return None
    return _match_country(normalized_location, allow_ambiguous_aliases=False)


def _extract_experience_requirements(text: str) -> dict:
    normalized = " ".join((text or "").lower().split())
    if not normalized:
        return {
            "detected": False,
            "min_years": None,
            "max_years": None,
            "evidence": None,
        }

    for pattern in _EXPERIENCE_RANGE_PATTERNS:
        match = pattern.search(normalized)
        if match:
            return {
                "detected": True,
                "min_years": float(match.group(1)),
                "max_years": float(match.group(2)),
                "evidence": match.group(0),
            }

    for pattern in _EXPERIENCE_MIN_PATTERNS:
        match = pattern.search(normalized)
        if match:
            value = float(match.group(1))
            return {
                "detected": True,
                "min_years": value,
                "max_years": None,
                "evidence": match.group(0),
            }

    for pattern in _EXPERIENCE_EXACT_PATTERNS:
        match = pattern.search(normalized)
        if match:
            value = float(match.group(1))
            return {
                "detected": True,
                "min_years": value,
                "max_years": value,
                "evidence": match.group(0),
            }

    if any(
        term in normalized
        for term in ("fresher", "freshers", "entry level", "entry-level", "graduate trainee", "new grad")
    ):
        return {
            "detected": True,
            "min_years": 0.0,
            "max_years": 1.0,
            "evidence": "entry-level/fresher wording",
        }

    return {
        "detected": False,
        "min_years": None,
        "max_years": None,
        "evidence": None,
    }


def _matches_requested_experience(
    job_requirement: dict,
    min_years: Optional[float],
    max_years: Optional[float],
) -> bool:
    job_min = job_requirement.get("min_years")
    job_max = job_requirement.get("max_years")

    if job_min is None and job_max is None:
        return False
    if max_years is not None and job_min is not None and job_min > max_years:
        return False
    if min_years is not None and job_max is not None and job_max < min_years:
        return False
    return True


def _is_entry_level_text(text: str) -> bool:
    normalized = " ".join((text or "").lower().split())
    return any(term in normalized for term in _ENTRY_LEVEL_TERMS)


def _is_senior_level_text(text: str) -> bool:
    normalized = " ".join((text or "").lower().split())
    return any(term in normalized for term in _SENIOR_LEVEL_TERMS)


def _is_remote_location_query(location: str) -> bool:
    normalized = " ".join((location or "").lower().split())
    return normalized in _REMOTE_LOCATION_TERMS or normalized.startswith("remote ")


def _looks_like_remote_job(job: dict) -> bool:
    combined_text = " ".join(
        part for part in (
            job.get("title", ""),
            job.get("location", ""),
            job.get("snippet", ""),
            job.get("description", ""),
        )
        if part
    ).lower()
    return any(term in combined_text for term in _REMOTE_JOB_TERMS)


def _build_position_query(role: str, min_years: Optional[float], max_years: Optional[float]) -> str:
    role = role.strip()
    if max_years is not None and max_years <= 1:
        return f"{role} fresher entry level junior"
    if max_years is not None and max_years <= 2:
        return f"{role} junior entry level"
    return role


def _with_experience_terms(query: str, min_years: Optional[float], max_years: Optional[float]) -> str:
    query = " ".join((query or "").split())
    if max_years is not None and max_years <= 1:
        return f"{query} fresher entry level junior"
    if max_years is not None and max_years <= 2:
        return f"{query} junior entry level"
    return query


def _build_position_query_variants(
    role: str,
    min_years: Optional[float],
    max_years: Optional[float],
) -> list[str]:
    role = " ".join((role or "").split())
    family = _requested_role_family(role)
    if family == "mern":
        base_queries = [
            role,
            "MERN Stack Developer",
            "Full Stack Developer React Node",
            "React Node.js Developer",
        ]
    elif family == "genai":
        base_queries = [
            role,
            "AI Engineer",
            "Generative AI Engineer",
            "LLM Engineer",
            "Machine Learning Engineer",
            "Prompt Engineer",
        ]
    else:
        base_queries = [role]

    seen = set()
    queries = []
    for query in base_queries:
        normalized_query = _with_experience_terms(query, min_years, max_years)
        key = normalized_query.lower()
        if not normalized_query or key in seen:
            continue
        seen.add(key)
        queries.append(normalized_query)
    return queries or [_build_position_query(role, min_years, max_years)]


def _provider_fetch_limit(max_results: int) -> int:
    configured_limit = int(os.getenv("JOB_SCOUT_PROVIDER_FETCH_LIMIT") or "50")
    return min(max(max_results * 5, max_results), configured_limit)


def _merge_job_lists(job_groups: list[list[dict]], max_results: int) -> list[dict]:
    merged = []
    seen = set()

    for jobs in job_groups:
        for job in jobs:
            key = (
                (job.get("url") or "").strip().lower(),
                (job.get("title") or "").strip().lower(),
                (job.get("company") or "").strip().lower(),
            )
            if key in seen:
                continue
            seen.add(key)
            merged.append(job)
            if len(merged) >= max_results:
                return merged

    return merged


def _requested_role_family(role: str) -> Optional[str]:
    normalized = " ".join((role or "").lower().split())
    if any(term in normalized for term in ("mern", "full stack", "fullstack", "react", "node")):
        return "mern"
    if any(
        term in normalized
        for term in (
            "genai",
            "gen ai",
            "generative ai",
            "llm",
            "ai engineer",
            "machine learning",
            "ml engineer",
            "prompt engineer",
        )
    ):
        return "genai"
    return None


def _role_relevance_score(job: dict, role: str) -> int:
    family = _requested_role_family(role)
    if family is None:
        return 1

    title = (job.get("title") or "").lower()
    combined_text = " ".join(
        part for part in (
            job.get("title", ""),
            job.get("snippet", ""),
            job.get("description", ""),
        )
        if part
    ).lower()

    if family == "mern":
        score = 0
        if "mern" in title:
            score += 5
        if "full stack" in title or "fullstack" in title:
            score += 4
        if any(term in title for term in ("react", "node", "node.js", "web developer")):
            score += 3
        if "mern" in combined_text:
            score += 4
        if "react" in combined_text and ("node" in combined_text or "node.js" in combined_text):
            score += 3
        if any(term in combined_text for term in _MERN_ROLE_TERMS):
            score += 1
        if any(term in title for term in _MERN_UNRELATED_TITLE_TERMS):
            score -= 4
        return score

    score = 0
    if any(term in title for term in ("genai", "gen ai", "generative ai", "llm")):
        score += 5
    if any(term in title for term in ("ai engineer", "ml engineer", "machine learning", "nlp", "prompt")):
        score += 4
    if any(term in combined_text for term in _GENAI_ROLE_TERMS):
        score += 2
    if any(term in title for term in _GENAI_UNRELATED_TITLE_TERMS):
        score -= 4
    return score


def _rank_jobs_for_requested_role(jobs: list[dict], role: str) -> list[dict]:
    family = _requested_role_family(role)
    if family is None:
        return jobs

    scored_jobs = [
        (_role_relevance_score(job, role), index, job)
        for index, job in enumerate(jobs)
    ]
    relevant_jobs = [
        (score, index, job)
        for score, index, job in scored_jobs
        if score > 0
    ]
    if not relevant_jobs:
        return []

    relevant_jobs.sort(key=lambda item: (-item[0], item[1]))
    return [job for _, _, job in relevant_jobs]


def _select_jobs_for_requested_experience(
    jobs: list[dict],
    min_years: Optional[float],
    max_years: Optional[float],
    max_results: int,
) -> list[dict]:
    filtered_jobs, fallback_jobs = _filter_jobs_for_experience(jobs, min_years, max_years)

    if max_years is not None and max_years <= 1:
        return filtered_jobs[:max_results]

    return _merge_job_lists(
        [filtered_jobs, fallback_jobs, jobs],
        max_results=max_results,
    )


def _filter_jobs_for_experience(
    jobs: list[dict],
    min_years: Optional[float],
    max_years: Optional[float],
) -> tuple[list[dict], list[dict]]:
    if min_years is None and max_years is None:
        return jobs, []

    matched = []
    fallback = []

    for job in jobs:
        combined_text = " ".join(
            part for part in (
                job.get("title", ""),
                job.get("snippet", ""),
            )
            if part
        )
        requirement = _extract_experience_requirements(combined_text)
        explicit_entry_level = _is_entry_level_text(combined_text)
        explicit_senior_level = _is_senior_level_text(combined_text)

        job["experience_min_years"] = requirement.get("min_years")
        job["experience_max_years"] = requirement.get("max_years")
        job["experience_evidence"] = requirement.get("evidence")

        if max_years is not None and max_years <= 1 and explicit_senior_level and not explicit_entry_level:
            continue

        if requirement["detected"]:
            if _matches_requested_experience(requirement, min_years, max_years):
                matched.append(job)
            continue

        if explicit_entry_level:
            matched.append(job)
            continue

        if max_years is not None and max_years <= 1:
            continue

        if not explicit_senior_level:
            fallback.append(job)

    return matched, fallback


def _resolve_job_search_provider() -> str:
    return _resolve_job_search_providers()[0]


def _resolve_job_search_providers() -> list[str]:
    preferred_provider = (os.getenv("JOB_SCOUT_JOB_SEARCH_PROVIDER") or "").strip().lower()
    apify_token = (os.getenv("APIFY_TOKEN") or "").strip()
    adzuna_app_id = (os.getenv("ADZUNA_APP_ID") or "").strip()
    adzuna_app_key = (os.getenv("ADZUNA_APP_KEY") or "").strip()
    browseract_api_key = (os.getenv("BROWSERACT_API_KEY") or "").strip()
    browseract_workflow_id = (os.getenv("BROWSERACT_WORKFLOW_ID") or "").strip()

    provider_credentials = {
        "browseract": bool(browseract_api_key and browseract_workflow_id),
        "adzuna": bool(adzuna_app_id and adzuna_app_key),
        "apify": bool(apify_token),
    }

    if preferred_provider in provider_credentials and not provider_credentials[preferred_provider]:
        return ["demo"]

    providers = []
    if provider_credentials.get(preferred_provider):
        providers.append(preferred_provider)

    for provider in ("browseract", "adzuna", "apify"):
        if provider_credentials[provider] and provider not in providers:
            providers.append(provider)

    return providers or ["demo"]


def _resolve_adzuna_country(resolved_country: Optional[str]) -> str:
    explicit_country = (os.getenv("ADZUNA_COUNTRY") or "").strip().lower()
    if explicit_country:
        return explicit_country
    if resolved_country:
        return resolved_country.lower()
    return _DEFAULT_ADZUNA_COUNTRY


def _adzuna_country_label(country_code: str) -> str:
    return {
        "in": "India",
        "us": "United States",
        "gb": "United Kingdom",
        "ca": "Canada",
        "au": "Australia",
    }.get((country_code or "").lower(), (country_code or "").upper())


def _build_adzuna_search_requests(
    *,
    position_query: str,
    location: str,
    country_code: str,
) -> list[dict]:
    if not _is_remote_location_query(location):
        return [{
            "what": position_query,
            "where": location,
            "remote_only": False,
        }]

    country_label = _adzuna_country_label(country_code)
    return [
        {
            "what": f"{position_query} remote work from home",
            "where": country_label,
            "remote_only": True,
        },
        {
            "what": f"{position_query} remote",
            "where": country_label,
            "remote_only": True,
        },
        {
            "what": f"{position_query} work from home",
            "where": country_label,
            "remote_only": True,
        },
        {
            "what": f"{position_query} remote",
            "where": "",
            "remote_only": True,
        },
    ]


def _search_jobs_once_adzuna(
    *,
    role: str,
    location: str,
    max_results: int,
    resolved_country: Optional[str],
    min_years: Optional[float],
    max_years: Optional[float],
    adzuna_app_id: str,
    adzuna_app_key: str,
) -> dict:
    position_queries = _build_position_query_variants(role, min_years, max_years)
    country_code = _resolve_adzuna_country(resolved_country)
    url = f"https://api.adzuna.com/v1/api/jobs/{country_code}/search/1"

    collected_jobs: list[dict] = []
    queries_used: list[str] = []

    with httpx.Client(timeout=60, follow_redirects=True) as client:
        for position_query in position_queries:
            search_requests = _build_adzuna_search_requests(
                position_query=position_query,
                location=location,
                country_code=country_code,
            )
            for request in search_requests:
                params = {
                    "app_id": adzuna_app_id,
                    "app_key": adzuna_app_key,
                    "results_per_page": _provider_fetch_limit(max_results),
                    "what": request["what"],
                    "sort_by": "date",
                    "content-type": "application/json",
                }
                if request.get("where"):
                    params["where"] = request["where"]

                response = client.get(
                    url,
                    params=params,
                    headers={"Accept": "application/json"},
                )
                response.raise_for_status()
                payload = response.json()

                items = payload.get("results", []) if isinstance(payload, dict) else []
                jobs = [
                    {
                        "id": item.get("id") or item.get("adref"),
                        "title": item.get("title"),
                        "company": (item.get("company") or {}).get("display_name", ""),
                        "location": (item.get("location") or {}).get("display_name", "") or location,
                        "snippet": (item.get("description") or "")[:500],
                        "url": item.get("redirect_url") or item.get("redirectUrl"),
                        "matched_role": role,
                    }
                    for item in items
                ]
                if request.get("remote_only"):
                    jobs = [job for job in jobs if _looks_like_remote_job(job)]

                if jobs:
                    collected_jobs = _merge_job_lists([collected_jobs, jobs], max_results=_provider_fetch_limit(max_results))
                    queries_used.append(request["what"])

                relevant_jobs = _rank_jobs_for_requested_role(collected_jobs, role)
                selected_jobs = _select_jobs_for_requested_experience(
                    relevant_jobs,
                    min_years,
                    max_years,
                    max_results,
                )
                if len(selected_jobs) >= max_results:
                    return {
                        "jobs": selected_jobs,
                        "query_used": " | ".join(queries_used) or request["what"],
                        "provider": "adzuna",
                        "country_used": country_code.upper(),
                    }
                if jobs:
                    break

    relevant_jobs = _rank_jobs_for_requested_role(collected_jobs, role)
    selected_jobs = _select_jobs_for_requested_experience(
        relevant_jobs,
        min_years,
        max_years,
        max_results,
    )

    return {
        "jobs": selected_jobs,
        "query_used": " | ".join(queries_used) or (position_queries[0] if position_queries else role),
        "provider": "adzuna",
        "country_used": country_code.upper(),
    }


def _search_jobs_once_apify(
    *,
    role: str,
    location: str,
    max_results: int,
    resolved_country: Optional[str],
    min_years: Optional[float],
    max_years: Optional[float],
    apify_token: str,
) -> dict:
    position_queries = _build_position_query_variants(role, min_years, max_years)

    url = "https://api.apify.com/v2/acts/misceres~indeed-scraper/run-sync-get-dataset-items"
    actor_timeout_seconds = _resolve_apify_actor_timeout_seconds()
    params = {
        "token": apify_token,
        "memory": 1024,
        "timeout": actor_timeout_seconds,
    }

    collected_jobs: list[dict] = []
    queries_used: list[str] = []
    with httpx.Client(timeout=_resolve_apify_http_timeout_seconds()) as client:
        for position_query in position_queries:
            payload = {
                "position": position_query,
                "location": location,
                "maxItems": _provider_fetch_limit(max_results),
            }
            if resolved_country:
                payload["country"] = resolved_country

            response = client.post(url, params=params, json=payload)
            response.raise_for_status()
            items = response.json()

            jobs = [
                {
                    "id": item.get("id"),
                    "title": item.get("positionName"),
                    "company": item.get("company"),
                    "location": item.get("location"),
                    "snippet": (item.get("description") or "")[:500],
                    "url": item.get("url"),
                    "matched_role": role,
                }
                for item in items
            ]
            if jobs:
                collected_jobs = _merge_job_lists([collected_jobs, jobs], max_results=_provider_fetch_limit(max_results))
                queries_used.append(position_query)

            relevant_jobs = _rank_jobs_for_requested_role(collected_jobs, role)
            selected_jobs = _select_jobs_for_requested_experience(
                relevant_jobs,
                min_years,
                max_years,
                max_results,
            )
            if len(selected_jobs) >= max_results:
                return {
                    "jobs": selected_jobs,
                    "query_used": " | ".join(queries_used) or position_query,
                    "provider": "apify",
                    "country_used": resolved_country,
                }

    relevant_jobs = _rank_jobs_for_requested_role(collected_jobs, role)
    selected_jobs = _select_jobs_for_requested_experience(
        relevant_jobs,
        min_years,
        max_years,
        max_results,
    )

    return {
        "jobs": selected_jobs,
        "query_used": " | ".join(queries_used) or (position_queries[0] if position_queries else role),
        "provider": "apify",
        "country_used": resolved_country,
    }


def _browseract_base_url() -> str:
    return (os.getenv("BROWSERACT_BASE_URL") or "https://api.browseract.com/v2/workflow").strip().rstrip("/")


def _browseract_headers(api_key: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


def _browseract_input_name(setting_name: str, default: str) -> str:
    return (os.getenv(setting_name) or default).strip()


def _build_browseract_input_parameters(
    *,
    position_query: str,
    location: str,
    max_results: int,
    resolved_country: Optional[str],
) -> list[dict[str, object]]:
    inputs = [
        {
            "name": _browseract_input_name("BROWSERACT_ROLE_PARAM", "keyword"),
            "value": position_query,
        },
        {
            "name": _browseract_input_name("BROWSERACT_LOCATION_PARAM", "location"),
            "value": location,
        },
        {
            "name": _browseract_input_name("BROWSERACT_LIMIT_PARAM", "datalimit"),
            "value": min(max(max_results * 4, max_results), 50),
        },
    ]

    country_param = _browseract_input_name("BROWSERACT_COUNTRY_PARAM", "country")
    if resolved_country and country_param:
        inputs.append({"name": country_param, "value": resolved_country})

    return inputs


def _browseract_extract_first(payload: dict, *keys: str) -> Optional[object]:
    for key in keys:
        if key in payload and payload[key] not in (None, ""):
            return payload[key]
    return None


def _browseract_parse_output_string(value: object) -> list[dict]:
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]

    if isinstance(value, dict):
        for nested_key in ("data", "items", "jobs", "results", "Request"):
            nested_value = value.get(nested_key)
            if isinstance(nested_value, list):
                return [item for item in nested_value if isinstance(item, dict)]
        return [value]

    if not isinstance(value, str):
        return []

    normalized = value.strip()
    if not normalized:
        return []

    try:
        parsed = json.loads(normalized)
    except json.JSONDecodeError:
        return []

    return _browseract_parse_output_string(parsed)


def _extract_browseract_jobs(task_payload: dict, *, role: str, location: str) -> list[dict]:
    candidates = []

    for container_key in ("output", "data", "result"):
        container = task_payload.get(container_key)
        if isinstance(container, dict):
            for nested_key in ("string", "data", "items", "jobs", "results"):
                candidates.extend(_browseract_parse_output_string(container.get(nested_key)))
        elif container is not None:
            candidates.extend(_browseract_parse_output_string(container))

    if not candidates:
        candidates.extend(_browseract_parse_output_string(task_payload.get("output_string")))
        candidates.extend(_browseract_parse_output_string(task_payload.get("jobs")))

    normalized_jobs = []
    for index, item in enumerate(candidates, start=1):
        title = _browseract_extract_first(
            item,
            "title",
            "job_title",
            "jobTitle",
            "Job Title",
            "position",
            "positionName",
            "Title",
        )
        if not title:
            continue

        description = _browseract_extract_first(
            item,
            "description",
            "job_description",
            "jobDescription",
            "summary",
            "about",
            "About",
            "Needs",
        )
        url = _browseract_extract_first(
            item,
            "url",
            "job_url",
            "jobUrl",
            "job_link",
            "jobLink",
            "link",
            "Link",
            "Job URL",
        )

        normalized_jobs.append(
            {
                "id": _browseract_extract_first(item, "id", "job_id", "jobId") or f"browseract-{index}",
                "title": str(title),
                "company": str(
                    _browseract_extract_first(
                        item,
                        "company",
                        "company_name",
                        "companyName",
                        "Company Name",
                        "company_name_display",
                    )
                    or ""
                ),
                "location": str(
                    _browseract_extract_first(item, "location", "job_location", "Location", "work_arrangement")
                    or location
                ),
                "snippet": str(description or "")[:500],
                "url": str(url) if url else None,
                "matched_role": role,
            }
        )

    return normalized_jobs


def _browseract_task_error(task_payload: dict) -> str:
    for key in ("message", "error", "failure_reason", "failureReason"):
        value = task_payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return "BrowserAct task failed."


def _wait_for_browseract_task(
    client: httpx.Client,
    *,
    base_url: str,
    api_key: str,
    task_id: str,
) -> dict:
    timeout_seconds = float(os.getenv("BROWSERACT_TASK_TIMEOUT_SECONDS") or "180")
    poll_seconds = max(0.5, float(os.getenv("BROWSERACT_TASK_POLL_INTERVAL_SECONDS") or "2"))
    deadline = time.monotonic() + timeout_seconds

    while True:
        response = client.get(
            f"{base_url}/get-task",
            params={"task_id": task_id},
            headers=_browseract_headers(api_key),
        )
        response.raise_for_status()
        payload = response.json()

        status = str(payload.get("status") or "").strip().lower()
        if status in {"finished", "completed", "success", "succeeded"}:
            return payload
        if status in {"failed", "error", "cancelled", "canceled"}:
            raise ValueError(_browseract_task_error(payload))
        if time.monotonic() >= deadline:
            raise TimeoutError(f"BrowserAct task {task_id} did not finish within {timeout_seconds:g} seconds.")

        time.sleep(poll_seconds)


def _search_jobs_once_browseract(
    *,
    role: str,
    location: str,
    max_results: int,
    resolved_country: Optional[str],
    min_years: Optional[float],
    max_years: Optional[float],
    browseract_api_key: str,
    browseract_workflow_id: str,
) -> dict:
    position_query = _build_position_query(role, min_years, max_years)
    base_url = _browseract_base_url()
    payload = {
        "workflow_id": browseract_workflow_id,
        "input_parameters": _build_browseract_input_parameters(
            position_query=position_query,
            location=location,
            max_results=max_results,
            resolved_country=resolved_country,
        ),
    }

    with httpx.Client(timeout=60, follow_redirects=True) as client:
        response = client.post(
            f"{base_url}/run-task",
            headers=_browseract_headers(browseract_api_key),
            json=payload,
        )
        response.raise_for_status()
        task_data = response.json()
        task_id = task_data.get("id")
        if not task_id:
            raise ValueError("BrowserAct run-task response did not include a task id.")

        task_payload = _wait_for_browseract_task(
            client,
            base_url=base_url,
            api_key=browseract_api_key,
            task_id=str(task_id),
        )

    jobs = _extract_browseract_jobs(task_payload, role=role, location=location)
    relevant_jobs = _rank_jobs_for_requested_role(jobs, role)
    selected_jobs = _select_jobs_for_requested_experience(
        relevant_jobs,
        min_years,
        max_years,
        max_results,
    )

    return {
        "jobs": selected_jobs,
        "query_used": position_query,
        "provider": "browseract",
        "country_used": resolved_country,
    }


def _search_jobs_once(
    *,
    role: str,
    location: str,
    max_results: int,
    resolved_country: Optional[str],
    min_years: Optional[float],
    max_years: Optional[float],
    apify_token: Optional[str] = None,
    adzuna_app_id: Optional[str] = None,
    adzuna_app_key: Optional[str] = None,
    provider: Optional[str] = None,
) -> dict:
    provider = provider or _resolve_job_search_provider()
    if provider == "adzuna":
        if not adzuna_app_id or not adzuna_app_key:
            raise ValueError("ADZUNA_APP_ID and ADZUNA_APP_KEY are required for Adzuna job search.")
        return _search_jobs_once_adzuna(
            role=role,
            location=location,
            max_results=max_results,
            resolved_country=resolved_country,
            min_years=min_years,
            max_years=max_years,
            adzuna_app_id=adzuna_app_id,
            adzuna_app_key=adzuna_app_key,
        )
    if provider == "apify":
        if not apify_token:
            raise ValueError("APIFY_TOKEN is required for Apify job search.")
        return _search_jobs_once_apify(
            role=role,
            location=location,
            max_results=max_results,
            resolved_country=resolved_country,
            min_years=min_years,
            max_years=max_years,
            apify_token=apify_token,
        )
    if provider == "browseract":
        browseract_api_key = (os.getenv("BROWSERACT_API_KEY") or "").strip()
        browseract_workflow_id = (os.getenv("BROWSERACT_WORKFLOW_ID") or "").strip()
        if not browseract_api_key or not browseract_workflow_id:
            raise ValueError(
                "BROWSERACT_API_KEY and BROWSERACT_WORKFLOW_ID are required for BrowserAct job search."
            )
        return _search_jobs_once_browseract(
            role=role,
            location=location,
            max_results=max_results,
            resolved_country=resolved_country,
            min_years=min_years,
            max_years=max_years,
            browseract_api_key=browseract_api_key,
            browseract_workflow_id=browseract_workflow_id,
        )
    raise ValueError("No job search provider is configured.")


def fetch_job_details(job_url: str) -> dict:
    """Fetch and clean the full text of a job description from a job URL.

    Use this when a job snippet is too short or vague for reliable resume-aware
    scoring and the caller needs the fuller posting text.

    Args:
        job_url: Direct job-posting URL. Only ``http`` and ``https`` URLs are supported.

    Returns:
        A payload containing either the cleaned job description text or an
        error message if the URL is invalid or the page could not be fetched.
    """
    normalized_url = (job_url or "").strip()
    if not normalized_url:
        return {"status": "error", "message": "Job URL is required."}

    parsed_url = urlparse(normalized_url)
    if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
        return {
            "status": "error",
            "message": "Only http(s) job URLs are supported.",
        }

    try:
        with httpx.Client(timeout=30, follow_redirects=True) as client:
            response = client.get(normalized_url, headers={"User-Agent": "Mozilla/5.0"})
            response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()

        text = soup.get_text(separator=" ", strip=True)
        return {"status": "ok", "description": text[:7000]}
    except Exception as exc:
        return {"status": "error", "message": str(exc)}
