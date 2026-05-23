from __future__ import annotations

from dotenv import load_dotenv

from job_scout.resume_support import ResumeProfileExtraction
from job_scout.resume_support import _choose_resume_artifact_name
from job_scout.resume_support import _extract_text_from_pdf_bytes
from job_scout.resume_support import _normalize_resume_profile_payload
from job_scout.resume_support import _parse_resume_with_gemini
from job_scout.resume_tools import _ensure_resume_profile
from job_scout.resume_tools import _extract_resume_profile_payload_from_artifact
from job_scout.resume_tools import _persist_resume_profile
from job_scout.resume_tools import clear_resume_profile
from job_scout.resume_tools import extract_resume_profile_from_artifact
from job_scout.resume_tools import find_resume_matched_jobs
from job_scout.resume_tools import get_resume_status
from job_scout.resume_tools import save_resume_profile
from job_scout.resume_tools import score_job_match
from job_scout.resume_tools import score_saved_jobs
from job_scout.search_support import _filter_jobs_for_experience
from job_scout.search_support import _normalize_country
from job_scout.search_support import fetch_job_details
from job_scout.search_tools import filter_saved_jobs_by_experience
from job_scout.search_tools import search_jobs
from job_scout.state_keys import LAST_SEARCH_RESULTS_STATE_KEY
from job_scout.state_keys import RESUME_PROFILE_EXTRACTION_METHOD_STATE_KEY
from job_scout.state_keys import RESUME_PROFILE_STATE_KEY

load_dotenv()

__all__ = [
    "LAST_SEARCH_RESULTS_STATE_KEY",
    "RESUME_PROFILE_EXTRACTION_METHOD_STATE_KEY",
    "RESUME_PROFILE_STATE_KEY",
    "ResumeProfileExtraction",
    "_choose_resume_artifact_name",
    "_ensure_resume_profile",
    "_extract_resume_profile_payload_from_artifact",
    "_extract_text_from_pdf_bytes",
    "_filter_jobs_for_experience",
    "_normalize_country",
    "_normalize_resume_profile_payload",
    "_parse_resume_with_gemini",
    "_persist_resume_profile",
    "clear_resume_profile",
    "extract_resume_profile_from_artifact",
    "fetch_job_details",
    "filter_saved_jobs_by_experience",
    "find_resume_matched_jobs",
    "get_resume_status",
    "save_resume_profile",
    "score_job_match",
    "score_saved_jobs",
    "search_jobs",
]
