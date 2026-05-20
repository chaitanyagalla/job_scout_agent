from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from typing import Optional

from pydantic import BaseModel
from pydantic import Field

from job_scout.domain_models import JobPosting
from job_scout.domain_models import ResumeProfile
from job_scout.scoring import score_resume_vs_jd
from job_scout.search_support import _filter_jobs_for_experience


class ScoringEvaluationCase(BaseModel):
    name: str
    profile: ResumeProfile
    job_title: str
    job_description: str
    min_score: Optional[int] = None
    max_score: Optional[int] = None
    expected_fit_verdict: Optional[str] = None
    expected_blocker_risk: Optional[str] = None
    required_matched_skills: list[str] = Field(default_factory=list)
    required_missing_skills: list[str] = Field(default_factory=list)
    expect_has_blockers: Optional[bool] = None


class RankingJobCase(BaseModel):
    title: str
    description: str


class RankingEvaluationCase(BaseModel):
    name: str
    profile: ResumeProfile
    jobs: list[RankingJobCase]
    expected_top_job_title: str


class ExperienceFilterEvaluationCase(BaseModel):
    name: str
    jobs: list[JobPosting]
    min_years: float
    max_years: float
    expected_matched_titles: list[str] = Field(default_factory=list)
    expected_fallback_titles: list[str] = Field(default_factory=list)


class EvaluationSuite(BaseModel):
    scoring_cases: list[ScoringEvaluationCase] = Field(default_factory=list)
    ranking_cases: list[RankingEvaluationCase] = Field(default_factory=list)
    experience_filter_cases: list[ExperienceFilterEvaluationCase] = Field(default_factory=list)


def default_fixture_path() -> Path:
    """Return the default path to the evaluation fixture file."""
    return Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "evaluation_cases.json"


def load_evaluation_suite(path: str | Path | None = None) -> EvaluationSuite:
    """Load evaluation cases from a JSON fixture file."""
    fixture_path = Path(path) if path is not None else default_fixture_path()
    payload = json.loads(fixture_path.read_text(encoding="utf-8"))
    return EvaluationSuite.model_validate(payload)


def _normalize_skill_list(items: list[str]) -> set[str]:
    return {item.strip().lower() for item in items if item.strip()}


def evaluate_scoring_case(case: ScoringEvaluationCase) -> dict[str, Any]:
    """Evaluate one deterministic resume-vs-JD scoring case."""
    result = score_resume_vs_jd(
        job_title=case.job_title,
        job_description=case.job_description,
        profile=case.profile.model_dump(),
    )
    failures: list[str] = []

    if case.min_score is not None and result["score"] < case.min_score:
        failures.append(f"score {result['score']} < expected min {case.min_score}")
    if case.max_score is not None and result["score"] > case.max_score:
        failures.append(f"score {result['score']} > expected max {case.max_score}")
    if case.expected_fit_verdict and result.get("fit_verdict") != case.expected_fit_verdict:
        failures.append(
            f"fit_verdict {result.get('fit_verdict')!r} != {case.expected_fit_verdict!r}"
        )
    if case.expected_blocker_risk and result.get("blocker_risk") != case.expected_blocker_risk:
        failures.append(
            f"blocker_risk {result.get('blocker_risk')!r} != {case.expected_blocker_risk!r}"
        )
    if case.expect_has_blockers is not None and result["blockers"]["has_blockers"] != case.expect_has_blockers:
        failures.append(
            f"has_blockers {result['blockers']['has_blockers']} != {case.expect_has_blockers}"
        )

    matched_skills = _normalize_skill_list(result.get("matched_skills", []))
    missing_skills = _normalize_skill_list(result.get("missing_skills", []))

    for skill in case.required_matched_skills:
        if skill.lower() not in matched_skills:
            failures.append(f"missing expected matched skill {skill!r}")
    for skill in case.required_missing_skills:
        if skill.lower() not in missing_skills:
            failures.append(f"missing expected gap skill {skill!r}")

    return {
        "name": case.name,
        "passed": not failures,
        "failures": failures,
        "score": result["score"],
        "fit_verdict": result.get("fit_verdict"),
        "blocker_risk": result.get("blocker_risk"),
    }


def evaluate_ranking_case(case: RankingEvaluationCase) -> dict[str, Any]:
    """Evaluate whether job ranking prefers the expected best match."""
    scored_jobs = []
    for job in case.jobs:
        result = score_resume_vs_jd(
            job_title=job.title,
            job_description=job.description,
            profile=case.profile.model_dump(),
        )
        scored_jobs.append({
            "title": job.title,
            "score": result["score"],
        })

    ordered = sorted(scored_jobs, key=lambda item: item["score"], reverse=True)
    top_title = ordered[0]["title"] if ordered else None
    failures: list[str] = []
    if top_title != case.expected_top_job_title:
        failures.append(f"top title {top_title!r} != {case.expected_top_job_title!r}")

    return {
        "name": case.name,
        "passed": not failures,
        "failures": failures,
        "ranking": ordered,
    }


def evaluate_experience_filter_case(case: ExperienceFilterEvaluationCase) -> dict[str, Any]:
    """Evaluate deterministic entry-level/experience filtering behavior."""
    matched, fallback = _filter_jobs_for_experience(
        [job.model_dump(exclude_none=True) for job in case.jobs],
        min_years=case.min_years,
        max_years=case.max_years,
    )
    matched_titles = [job.get("title", "") for job in matched]
    fallback_titles = [job.get("title", "") for job in fallback]
    failures: list[str] = []

    if matched_titles != case.expected_matched_titles:
        failures.append(f"matched titles {matched_titles!r} != {case.expected_matched_titles!r}")
    if fallback_titles != case.expected_fallback_titles:
        failures.append(f"fallback titles {fallback_titles!r} != {case.expected_fallback_titles!r}")

    return {
        "name": case.name,
        "passed": not failures,
        "failures": failures,
        "matched_titles": matched_titles,
        "fallback_titles": fallback_titles,
    }


def run_evaluation_suite(path: str | Path | None = None) -> dict[str, Any]:
    """Run all deterministic evaluation fixtures and summarize the results."""
    suite = load_evaluation_suite(path)
    case_results: list[dict[str, Any]] = []

    for case in suite.scoring_cases:
        case_results.append({"type": "scoring", **evaluate_scoring_case(case)})
    for case in suite.ranking_cases:
        case_results.append({"type": "ranking", **evaluate_ranking_case(case)})
    for case in suite.experience_filter_cases:
        case_results.append({"type": "experience_filter", **evaluate_experience_filter_case(case)})

    passed = sum(1 for case in case_results if case["passed"])
    failed = len(case_results) - passed
    return {
        "total": len(case_results),
        "passed": passed,
        "failed": failed,
        "cases": case_results,
    }
