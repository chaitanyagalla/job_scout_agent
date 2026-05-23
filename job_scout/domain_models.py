from __future__ import annotations

from typing import Optional

from pydantic import BaseModel
from pydantic import Field


class ResumeProfile(BaseModel):
    candidate_name: str = ""
    professional_summary: str = ""
    role_titles: list[str] = Field(default_factory=list)
    core_skills: list[str] = Field(default_factory=list)
    additional_skills: list[str] = Field(default_factory=list)
    years_experience: Optional[float] = Field(default=None, ge=0)
    preferred_locations: list[str] = Field(default_factory=list)
    work_preferences: list[str] = Field(default_factory=list)
    notable_projects: list[str] = Field(default_factory=list)
    resume_source: str = "uploaded_resume"


class JobPosting(BaseModel):
    id: Optional[str] = None
    title: str = ""
    company: str = ""
    location: str = ""
    snippet: str = ""
    url: Optional[str] = None
    matched_role: Optional[str] = None
    experience_min_years: Optional[float] = None
    experience_max_years: Optional[float] = None
    experience_evidence: Optional[str] = None
    description: Optional[str] = None
    fit_verdict: Optional[str] = None
    reason_to_apply: Optional[str] = None
    blocker_risk: Optional[str] = None


class SearchContext(BaseModel):
    role: str = ""
    expanded_roles: list[str] = Field(default_factory=list)
    location: str = ""
    country: Optional[str] = None
    jobs: list[JobPosting] = Field(default_factory=list)


class JobMatchSignals(BaseModel):
    required_skill_coverage: float
    preferred_skill_coverage: float
    overall_skill_coverage: float
    title_alignment: float
    experience_alignment: float
    keyword_overlap: float


class JobMatchEvidence(BaseModel):
    required_skills: list[str] = Field(default_factory=list)
    preferred_skills: list[str] = Field(default_factory=list)
    general_skills: list[str] = Field(default_factory=list)
    matched_required_skills: list[str] = Field(default_factory=list)
    missing_required_skills: list[str] = Field(default_factory=list)
    matched_preferred_skills: list[str] = Field(default_factory=list)
    missing_preferred_skills: list[str] = Field(default_factory=list)
    matched_general_skills: list[str] = Field(default_factory=list)
    missing_general_skills: list[str] = Field(default_factory=list)
    required_years: Optional[float] = None
    actual_years: Optional[float] = None


class JobMatchBlockers(BaseModel):
    has_blockers: bool = False
    missing_required_skills: list[str] = Field(default_factory=list)
    insufficient_experience: bool = False
    title_mismatch: bool = False
    blocker_reasons: list[str] = Field(default_factory=list)


class JobMatchScore(BaseModel):
    score: int
    signals: JobMatchSignals
    matched_skills: list[str] = Field(default_factory=list)
    missing_skills: list[str] = Field(default_factory=list)
    before_applying: list[str] = Field(default_factory=list)
    explanation: str = ""
    resume_source: Optional[str] = None
    evidence: JobMatchEvidence = Field(default_factory=JobMatchEvidence)
    blockers: JobMatchBlockers = Field(default_factory=JobMatchBlockers)
    fit_verdict: str = ""
    reason_to_apply: str = ""
    blocker_risk: str = ""
