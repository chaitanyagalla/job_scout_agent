"""Edge-case tests for Job Scout — exposes bugs and boundary conditions."""
from __future__ import annotations

import asyncio
import os

import pytest

from job_scout.scoring import (
    _extract_required_years,
    _experience_alignment,
    _summarize_fit_verdict,
    score_resume_vs_jd,
)
from job_scout.search_support import (
    _extract_experience_requirements,
    _filter_jobs_for_experience,
    _normalize_country,
)
from job_scout.resume_support import _extract_years_experience
from job_scout.search_tools import _split_requested_roles, search_jobs
from job_scout.tools import save_resume_profile, filter_saved_jobs_by_experience


# ---------------------------------------------------------------------------
# Dummy contexts
# ---------------------------------------------------------------------------

class DummyToolContext:
    def __init__(self):
        self.state = {}


class AsyncDummyToolContext(DummyToolContext):
    def __init__(self, saved_search=None):
        super().__init__()
        if saved_search:
            self.state["last_search_results"] = saved_search

    async def list_artifacts(self):
        return []

    async def load_artifact(self, name):
        return None


# ===========================================================================
# 1. _normalize_country: "in" as a preposition matches India (false positive)
# ===========================================================================

def test_normalize_country_preposition_in_should_not_match_india():
    """'in' as a preposition in a phrase like 'jobs in New York' should not map to India."""
    result = _normalize_country("jobs in New York", None)
    # BUG: "in" is in _COUNTRY_CODE_ALIASES["IN"], so this currently returns "IN"
    assert result is None, (
        f"'jobs in New York' incorrectly resolved to {result!r} instead of None. "
        "The word 'in' as a preposition matches India's country code alias."
    )


def test_normalize_country_remote_job_in_phrase():
    """'looking for a job in Hyderabad' should not map to India just from 'in'."""
    result = _normalize_country("looking for a job in Hyderabad", None)
    # This may return None (from Hyderabad not matching) but NOT because of "in"
    # The test checks that "in" alone doesn't cause a false India match
    # when the intended location is not India
    assert result is None, (
        f"Expected None but got {result!r}. The preposition 'in' is causing a false India match."
    )


# ===========================================================================
# 2. _extract_required_years: dead-code branch (both if/else do the same)
# ===========================================================================

def test_extract_required_years_range_appends_min_not_max():
    """A range like '1-3 years' should use the lower bound as the threshold."""
    result_range = _extract_required_years("candidates must have 1-3 years of experience")
    assert result_range == 1.0, (
        f"Expected 1.0 as the minimum acceptable experience in the stated range but got {result_range}"
    )


def test_extract_required_years_prefers_minimum_over_preferred_years():
    """Preferred experience should not override a lower explicit minimum requirement."""
    jd = "Minimum 1 year of experience required. 5+ years preferred for senior candidates."
    required = _extract_required_years(jd)
    # Currently returns max([1.0, 5.0]) = 5.0 — the "preferred" requirement becomes the
    # "required" threshold, making a 2-year candidate look under-qualified.
    # Expected: should return the minimum stated requirement (1.0), not 5.0
    assert required == 1.0, (
        f"Expected 1.0 (minimum stated) but got {required}. "
        "Taking max() across all year mentions inflates required experience when "
        "'preferred' mentions co-exist with a lower minimum."
    )


# ===========================================================================
# 3. _extract_years_experience: misses common resume wording without "of"
# ===========================================================================

def test_extract_years_experience_without_of():
    """'3 years experience' (no 'of') is common resume wording but not matched."""
    result = _extract_years_experience("Full Stack Developer with 3 years experience in React.")
    assert result == 3.0, (
        f"Expected 3.0 but got {result!r}. "
        "The pattern requires 'years of experience' but '3 years experience' is also common."
    )


def test_extract_years_experience_hyphenated():
    """'2-year experience' is sometimes written on resumes but never matched."""
    result = _extract_years_experience("Frontend developer with 2-year experience in JavaScript.")
    assert result == 2.0, (
        f"Expected 2.0 but got {result!r}. '2-year experience' is not matched by any pattern."
    )


# ===========================================================================
# 4. Demo mode returns fewer jobs than max_results (role-count bound)
# ===========================================================================

def test_demo_mode_returns_fewer_jobs_than_requested_when_roles_are_fewer():
    """
    When APIFY_TOKEN is absent, demo mode creates one job per detected role.
    If the user asks for 10 jobs but only 1 role is detected, only 1 is returned.
    This can silently under-deliver.
    """
    token = os.environ.pop("APIFY_TOKEN", None)
    try:
        result = search_jobs(
            role="React Developer",
            location="Remote",
            max_results=10,
        )
    finally:
        if token:
            os.environ["APIFY_TOKEN"] = token

    assert result["status"] == "demo_mode"
    # Only 1 role → only 1 demo job, even though 10 were requested
    job_count = len(result["jobs"])
    assert job_count == 10, (
        f"Expected 10 demo jobs but got {job_count}. "
        "Demo mode is bounded by role count, not max_results."
    )


# ===========================================================================
# 5. Scoring: years_experience=0 (fresh graduate) should not produce None gaps
# ===========================================================================

def test_scoring_with_zero_years_experience_fresh_graduate():
    """A profile with years_experience=0 is a valid fresh graduate, not missing data."""
    profile = {
        "candidate_name": "Fresh Grad",
        "professional_summary": "Recent graduate with React and JavaScript skills.",
        "role_titles": ["Frontend Developer"],
        "core_skills": ["React", "JavaScript", "HTML", "CSS"],
        "additional_skills": [],
        "years_experience": 0.0,
        "preferred_locations": [],
        "work_preferences": [],
        "notable_projects": [],
        "resume_source": "test",
    }
    jd = "Required: React, JavaScript, HTML. Entry level position. 0-1 years of experience."
    result = score_resume_vs_jd("Junior Frontend Developer", jd, profile)

    # Should not error; experience_alignment should be computed, not use the None fallback
    assert isinstance(result["score"], int)
    assert result["evidence"]["actual_years"] == 0.0
    assert result["evidence"]["required_years"] is not None
    # With 0 years vs 0-1 required (captures 0.0 as min), alignment should be reasonable
    assert result["signals"]["experience_alignment"] >= 0.7, (
        f"Fresh grad applying for 0-1 year job got experience_alignment "
        f"{result['signals']['experience_alignment']} — should be high."
    )


# ===========================================================================
# 6. Scoring: negative years_experience should be normalized safely
# ===========================================================================

def test_scoring_clamps_negative_years_experience_to_zero():
    """Negative experience should be normalized to 0 years for conservative scoring."""
    profile = {
        "candidate_name": "Test",
        "professional_summary": "Developer",
        "role_titles": ["Frontend Developer"],
        "core_skills": ["React"],
        "additional_skills": [],
        "years_experience": -1.0,
        "preferred_locations": [],
        "work_preferences": [],
        "notable_projects": [],
        "resume_source": "test",
    }
    jd = "Required: React. 1 year of experience."
    result = score_resume_vs_jd("Frontend Developer", jd, profile)

    # Negative experience should not be treated as "unknown" because that can
    # gap = 1 - (-1) = 2 → experience_alignment = 0.55
    # This should ideally be rejected or treated as 0 at normalization time
    assert result["evidence"]["actual_years"] == 0.0
    assert result["blockers"]["insufficient_experience"] is True
    assert result["blockers"]["has_blockers"] is True
    assert result["blocker_risk"] == "High"


# ===========================================================================
# 7. _split_requested_roles: empty string returns [""] which causes empty search
# ===========================================================================

def test_split_requested_roles_empty_string():
    """Empty role string should not produce a search with an empty role."""
    from job_scout.search_tools import _split_requested_roles
    result = _split_requested_roles("")
    assert result == [], f"Expected [] for empty input but got {result!r}"


def test_split_requested_roles_only_whitespace():
    """Whitespace-only role string should not produce a search."""
    from job_scout.search_tools import _split_requested_roles
    result = _split_requested_roles("   ")
    assert result == [], f"Expected [] for whitespace-only input but got {result!r}"


# ===========================================================================
# 8. filter_saved_jobs_by_experience: jobs with None URL should not error
# ===========================================================================

def test_filter_saved_jobs_with_none_url_does_not_crash():
    """Jobs with url=None should not cause an AttributeError in filter_saved_jobs_by_experience."""
    saved_search = {
        "role": "Software Engineer",
        "location": "Remote",
        "country": None,
        "jobs": [
            {
                "id": "job-1",
                "title": "Junior Software Engineer",
                "company": "TechCorp",
                "location": "Remote",
                "snippet": "0-1 years of experience. Entry level role.",
                "url": None,
            }
        ],
    }
    ctx = AsyncDummyToolContext(saved_search=saved_search)
    result = filter_saved_jobs_by_experience(min_years=0, max_years=1, tool_context=ctx)
    assert result["status"] == "ok"
    assert len(result["jobs"]) >= 0  # Should not raise an exception


# ===========================================================================
# 9. Scoring: fit_verdict boundary — score=80 with blockers should NOT be "Strong apply"
# ===========================================================================

def test_fit_verdict_at_80_with_blockers_is_not_strong_apply():
    """score=80 with has_blockers=True must return 'Good match', not 'Strong apply'."""
    verdict = _summarize_fit_verdict(score=80, has_blockers=True)
    assert verdict == "Good match", (
        f"Expected 'Good match' for score=80 with blockers but got {verdict!r}. "
        "The has_blockers guard only applies to the first branch."
    )


def test_fit_verdict_at_80_without_blockers_is_strong_apply():
    """score=80 with has_blockers=False must return 'Strong apply'."""
    verdict = _summarize_fit_verdict(score=80, has_blockers=False)
    assert verdict == "Strong apply", f"Expected 'Strong apply' but got {verdict!r}"


# ===========================================================================
# 10. Scoring: job with no skill mentions at all produces non-zero score
# ===========================================================================

def test_scoring_job_description_with_no_skill_mentions():
    """A job description that mentions no known skills should still produce a valid score."""
    profile = {
        "candidate_name": "Dev",
        "professional_summary": "Software developer.",
        "role_titles": ["Software Engineer"],
        "core_skills": ["React", "Node.js"],
        "additional_skills": [],
        "years_experience": 2.0,
        "preferred_locations": [],
        "work_preferences": [],
        "notable_projects": [],
        "resume_source": "test",
    }
    jd = "We are looking for a motivated individual to join our passionate team."
    result = score_resume_vs_jd("Software Engineer", jd, profile)

    assert 0 <= result["score"] <= 100
    # With no recognized skills in the JD, general_score falls back to 0.6
    # and required_score falls back to general_score
    # So the score should not be 0
    assert result["score"] > 0, (
        "A valid profile against a vague JD should score above 0 "
        "because fallback coverage scores are applied."
    )


# ===========================================================================
# 11. Scoring: completely empty profile gets a predictable score
# ===========================================================================

def test_scoring_empty_profile_vs_rich_jd():
    """An empty profile should score very low but not error."""
    profile = {
        "candidate_name": "",
        "professional_summary": "",
        "role_titles": [],
        "core_skills": [],
        "additional_skills": [],
        "years_experience": None,
        "preferred_locations": [],
        "work_preferences": [],
        "notable_projects": [],
        "resume_source": "test",
    }
    jd = (
        "Required: React, TypeScript, Node.js, PostgreSQL. Must have 3+ years of experience. "
        "Looking for a senior full stack developer."
    )
    result = score_resume_vs_jd("Senior Full Stack Developer", jd, profile)

    assert 0 <= result["score"] <= 100
    # Empty profile should score below 50 — it matches nothing
    assert result["score"] < 50, (
        f"Empty profile got score {result['score']} which seems too high."
    )


# ===========================================================================
# 12. _normalize_country: "in" country code alias vs common English word
# ===========================================================================

def test_normalize_country_standalone_in_matches_india():
    """Confirm that passing country='in' correctly resolves to India."""
    result = _normalize_country("", "in")
    assert result == "IN"


def test_normalize_country_location_containing_in_as_word():
    """
    'Remote in US' should resolve to US, not India.
    Currently, tokenization produces {'remote', 'in', 'us'}, and both 'in' (→IN)
    and 'us' (→US) are in the alias maps. The phrase lookup for 'united states'
    runs first and wins, so this specific case is correct — but 'Remote in Europe'
    would wrongly return 'IN' (from 'in') if no phrase alias matches 'europe'.
    """
    result = _normalize_country("Remote in Europe", None)
    # 'europe' has no entry, but 'in' matches India's code alias → returns IN
    # This is a false positive
    assert result is None, (
        f"'Remote in Europe' resolved to {result!r} instead of None. "
        "The preposition 'in' incorrectly matches India's country code alias."
    )


# ===========================================================================
# 13. Experience alignment: required_years from range vs actual
# ===========================================================================

def test_experience_alignment_candidate_within_stated_range():
    """
    A candidate with 2 years for a '1-3 years' job.
    _extract_required_years captures group(1)=1 for the range pattern,
    so required_years=1.0 and the candidate with 2 years gets alignment=1.0.
    This is correct behavior — test documents it.
    """
    required = _extract_required_years("We need 1-3 years of experience.")
    alignment = _experience_alignment(required, 2.0)
    assert required == 1.0  # range pattern appends group(1) = min of range
    assert alignment == 1.0  # 2 >= 1 → full alignment


# ===========================================================================
# 14. search_jobs demo mode: max_results is ignored for single-role searches
# ===========================================================================

def test_demo_mode_single_role_honors_max_results():
    """Demo mode should still honor max_results for a single detected role."""
    token = os.environ.pop("APIFY_TOKEN", None)
    try:
        result = search_jobs(role="Data Scientist", location="Bangalore", max_results=10)
    finally:
        if token:
            os.environ["APIFY_TOKEN"] = token

    assert result["status"] == "demo_mode"
    assert len(result["jobs"]) == 10, (
        f"Expected 10 demo jobs for a single-role search requesting 10, got {len(result['jobs'])}. "
        "Demo mode should honor max_results even when only one role is detected."
    )


# ===========================================================================
# 15. save_resume_profile: years_experience=0 (valid entry-level) must be accepted
# ===========================================================================

def test_save_resume_profile_accepts_zero_years_experience():
    """years_experience=0 is a valid value for a fresh graduate and must not be rejected."""
    ctx = DummyToolContext()
    result = save_resume_profile(
        candidate_name="Fresh Graduate",
        professional_summary="Recent CS graduate with React and Python projects.",
        role_titles=["Frontend Developer"],
        core_skills=["React", "JavaScript"],
        additional_skills=["Python"],
        years_experience=0,
        preferred_locations=["Hyderabad"],
        work_preferences=["Remote"],
        notable_projects=["Portfolio Website"],
        resume_source="resume.pdf",
        tool_context=ctx,
    )
    assert result["status"] == "ok", (
        f"years_experience=0 was unexpectedly rejected: {result.get('message')}"
    )


def test_save_resume_profile_rejects_negative_years_experience():
    """years_experience must be rejected when it is negative."""
    ctx = DummyToolContext()
    result = save_resume_profile(
        candidate_name="Test Candidate",
        professional_summary="Frontend developer.",
        role_titles=["Frontend Developer"],
        core_skills=["React"],
        additional_skills=[],
        years_experience=-1,
        preferred_locations=[],
        work_preferences=[],
        notable_projects=[],
        resume_source="resume.pdf",
        tool_context=ctx,
    )
    assert result["status"] == "error"
    assert "years_experience" in result["message"]


# ===========================================================================
# 16. filter_saved_jobs_by_experience: no previous search returns clear error
# ===========================================================================

def test_filter_saved_jobs_without_prior_search_returns_error():
    """Calling filter_saved_jobs_by_experience before any search should return a clear error."""
    ctx = AsyncDummyToolContext()  # no saved_search
    result = filter_saved_jobs_by_experience(min_years=0, max_years=1, tool_context=ctx)
    assert result["status"] == "error"
    assert "search" in result["message"].lower()


def test_search_jobs_rejects_inverted_experience_range():
    """min_years > max_years should fail fast instead of generating contradictory results."""
    result = search_jobs(
        role="Frontend Developer",
        location="Remote",
        min_years=3,
        max_years=1,
    )
    assert result["status"] == "error"
    assert "min_years" in result["message"]


def test_filter_saved_jobs_rejects_inverted_experience_range():
    """Experience filtering should reject impossible ranges before inspecting saved jobs."""
    saved_search = {
        "role": "Software Engineer",
        "location": "Remote",
        "country": None,
        "jobs": [
            {
                "id": "job-1",
                "title": "Junior Software Engineer",
                "company": "TechCorp",
                "location": "Remote",
                "snippet": "0-1 years of experience. Entry level role.",
                "url": None,
            }
        ],
    }
    ctx = AsyncDummyToolContext(saved_search=saved_search)
    result = filter_saved_jobs_by_experience(min_years=2, max_years=1, tool_context=ctx)
    assert result["status"] == "error"
    assert "min_years" in result["message"]


# ===========================================================================
# 17. Scoring: skills from ontology aliases are recognized in JD
# ===========================================================================

def test_scoring_recognizes_js_alias_for_javascript():
    """'js' in a JD should be recognized as 'javascript' via the alias map."""
    profile = {
        "candidate_name": "Dev",
        "professional_summary": "JavaScript developer.",
        "role_titles": ["Frontend Developer"],
        "core_skills": ["JavaScript"],
        "additional_skills": [],
        "years_experience": 2.0,
        "preferred_locations": [],
        "work_preferences": [],
        "notable_projects": [],
        "resume_source": "test",
    }
    jd = "Required: js, react, html. Strong js skills needed."
    result = score_resume_vs_jd("Frontend Developer", jd, profile)
    # javascript is in the profile; "js" is an alias for javascript in SKILL_ONTOLOGY
    assert "javascript" in result["evidence"]["matched_required_skills"], (
        f"'js' alias should resolve to 'javascript' but matched skills are: "
        f"{result['evidence']['matched_required_skills']}"
    )
