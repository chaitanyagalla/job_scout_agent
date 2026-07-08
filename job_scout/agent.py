"""Job Scout root agent using native Gemini with optional resume-aware scoring."""
from __future__ import annotations

import logging
import os

from dotenv import load_dotenv
from google.adk.agents import LlmAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.models import Gemini
from google.adk.models.lite_llm import LiteLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.tools import load_artifacts
from google.genai import types

from .fallback_llm import FallbackLlm
from .litellm_compat import patch_litellm_tool_call_id_repair
from .model_config import DEFAULT_GEMINI_MODEL
from .model_config import resolve_max_output_tokens
from .model_config import resolve_model_name
from .model_config import uses_litellm
from .state_keys import LAST_SEARCH_RESULTS_STATE_KEY
from .state_keys import RESUME_PROFILE_EXTRACTION_METHOD_STATE_KEY
from .state_keys import RESUME_PROFILE_STATE_KEY
from .tools import (
    clear_resume_profile,
    extract_resume_profile_from_artifact,
    fetch_job_details,
    filter_saved_jobs_by_experience,
    find_resume_matched_jobs,
    get_resume_status,
    save_resume_profile,
    score_job_match,
    score_saved_jobs,
    search_jobs,
)

load_dotenv()
patch_litellm_tool_call_id_repair()
logger = logging.getLogger(__name__)


def _resolve_gemini_fallback_model_name() -> str | None:
    disable_fallback = (os.getenv("JOB_SCOUT_DISABLE_MODEL_FALLBACK") or "").strip().lower()
    if disable_fallback in {"1", "true", "yes", "on"}:
        return None

    google_api_key = (os.getenv("GOOGLE_API_KEY") or "").strip()
    if not google_api_key:
        return None

    fallback_model = (os.getenv("GEMINI_FALLBACK_MODEL") or DEFAULT_GEMINI_MODEL).strip()
    return fallback_model or DEFAULT_GEMINI_MODEL


def _attachment_hint_from_part(part: types.Part, fallback_index: int) -> str:
    if part.file_data:
        return (
            part.file_data.display_name
            or part.file_data.file_uri
            or part.file_data.mime_type
            or f"attachment_{fallback_index}"
        )
    if part.inline_data:
        return (
            part.inline_data.display_name
            or part.inline_data.mime_type
            or f"attachment_{fallback_index}"
        )
    return f"attachment_{fallback_index}"


def _sanitize_unsupported_file_parts(llm_request: LlmRequest) -> None:
    model_name = (getattr(llm_request, "model", None) or resolve_model_name() or "").strip()
    if not model_name.startswith("nvidia_nim/"):
        return

    contents = getattr(llm_request, "contents", None)
    if not isinstance(contents, list):
        return

    attachment_index = 0
    for content in contents:
        parts = getattr(content, "parts", None)
        if not isinstance(parts, list):
            continue

        rewritten_parts: list[types.Part] = []
        changed = False
        for part in parts:
            if part.inline_data or part.file_data:
                hint = _attachment_hint_from_part(part, attachment_index)
                attachment_index += 1
                rewritten_parts.append(types.Part.from_text(text=f"[Attached file: {hint}]"))
                changed = True
                continue
            rewritten_parts.append(part)

        if changed:
            content.parts = rewritten_parts


async def add_runtime_hints(
    callback_context: CallbackContext,
    llm_request: LlmRequest,
):
    _sanitize_unsupported_file_parts(llm_request)
    profile = callback_context.state.get(RESUME_PROFILE_STATE_KEY)
    last_search = callback_context.state.get(LAST_SEARCH_RESULTS_STATE_KEY)
    artifact_names: list[str] = []
    try:
        artifact_names = await callback_context.list_artifacts()
    except Exception:
        artifact_names = []
    has_uploaded_artifact = bool(artifact_names)

    if profile:
        llm_request.append_instructions([
            "A normalized resume profile already exists in session state.",
            (
                "Use resume-aware mode for personalized job matching and scoring. "
                f"Current resume source: {profile.get('resume_source', 'unknown')}."
            ),
            (
                "Resume extraction method: "
                f"{callback_context.state.get(RESUME_PROFILE_EXTRACTION_METHOD_STATE_KEY, 'unknown')}."
            ),
            (
                "Resume role titles: "
                f"{', '.join(profile.get('role_titles', [])) or 'not available'}."
            ),
            (
                "Resume core skills: "
                f"{', '.join(profile.get('core_skills', [])) or 'not available'}."
            ),
            (
                "Only refresh the resume profile if the user uploads a new resume "
                "or explicitly asks to update the profile."
            ),
            (
                "If you refresh the resume from an uploaded artifact, treat it as a "
                "two-step flow: first call `extract_resume_profile_from_artifact`, "
                "then call `get_resume_status` in a separate tool round to confirm "
                "that `resume_loaded=true` before any `search_jobs` or "
                "`score_job_match` call."
            ),
        ])
    else:
        llm_request.append_instructions([
            "No resume profile exists yet.",
            (
                "If the user only asks for jobs by role, location, or filters, "
                "continue in prompt-only mode without requiring a resume."
            ),
            (
                "If the user wants resume-based matching, ATS-style scoring, "
                "personalized ranking, or has uploaded a resume, first extract "
                "the profile from the uploaded artifact."
            ),
            (
                "Prefer the single-call tool `find_resume_matched_jobs` for end-to-end "
                "resume-based job finding when the user wants results according to "
                "their uploaded resume."
            ),
            (
                "To build a resume profile from an upload, call "
                "`extract_resume_profile_from_artifact`. That tool tries local "
                "text extraction first, then the configured non-Gemini text parser."
            ),
            (
                "When you call `extract_resume_profile_from_artifact`, do not batch "
                "`search_jobs` or `score_job_match` in the same tool round. Wait for "
                "the extraction result first, then continue with search or scoring."
            ),
            (
                "After extraction, call `get_resume_status` in a separate tool round "
                "and only continue with resume-aware search or scoring if it returns "
                "`resume_loaded=true`."
            ),
            (
                "Never save placeholder resume values such as 'Unknown', 'N/A', or "
                "'Not provided'. If a field is missing, pass an empty string or empty list."
            ),
        ])
        if has_uploaded_artifact:
            llm_request.append_instructions([
                (
                    "An uploaded artifact is available in this session. If the user "
                    "wants resume-based matching or ATS scoring, do not enter prompt-only "
                    "mode yet. Prefer `find_resume_matched_jobs` for the full flow in "
                    "one tool call. If you must use the multi-step path, first call "
                    "`extract_resume_profile_from_artifact`, then re-check with "
                    "`get_resume_status` before searching or scoring."
                ),
            ])

    if last_search:
        llm_request.append_instructions([
            (
                "A previous job search is also stored in session state. "
                "Use it for follow-up requests like 'yes', 'show details', "
                "'only 0-1 years', or 'filter the jobs you found'."
            ),
        ])


def _build_model():
    model_name = resolve_model_name()
    if uses_litellm(model_name):
        primary_model = LiteLlm(model=model_name)
        fallback_model_name = _resolve_gemini_fallback_model_name()
        if fallback_model_name:
            logger.info(
                "Using LiteLLM provider model with Gemini fallback: %s -> %s",
                model_name,
                fallback_model_name,
            )
            return FallbackLlm(
                model=model_name,
                primary=primary_model,
                fallback=Gemini(model=fallback_model_name),
            )

        logger.info("Using LiteLLM provider model: %s", model_name)
        return primary_model

    logger.info("Using native Gemini model: %s", model_name)
    return Gemini(model=model_name)


GEMINI_MODEL = _build_model()

root_agent = LlmAgent(
    name="job_scout",
    model=GEMINI_MODEL,
    description=(
        "A job-search assistant that can search jobs from a user's prompt alone, "
        "or read an uploaded resume, build a structured profile, score jobs against it, "
        "and rank the best matches."
    ),
    instruction="""
You are Job Scout, a production-style job search assistant.

You support two modes:

1. Resume-aware mode
- If the user uploaded a resume or already has a saved resume profile, use it.
- In this mode, personalize job matching and scoring based on the resume.

2. Prompt-only mode
- If the user did not provide a resume, still help normally.
- In this mode, search for jobs only from the user's prompt and do not require resume-based scoring.

Workflow:
1. Understand the user's request:
   - target role
   - location
   - remote preference
   - country
   - number of jobs requested
   - any extra filters like skills, company type, salary, or experience level
   - silently correct obvious spelling mistakes in role titles or locations before tool calls

Conversation handling:
- Treat short follow-ups like "yes", "yeah", "go ahead", "show me", or "details"
  as confirmation of the last clear action you offered.
- Do not ask the user to repeat the same clarification if the intent can be
  inferred from the immediately previous turn.
- If the user refers to jobs you already found, continue from the saved search
  results instead of restarting the workflow.
- If a tool already returned jobs in the current turn, present those jobs
  immediately instead of asking whether the user wants to see them.
- If the user asks to score "these jobs", "all jobs", or "the jobs you found",
  call `score_saved_jobs` once and present every returned job. Do not call
  `score_job_match` repeatedly for saved search results.

2. Call `get_resume_status`.

3. If a resume profile is available:
   - Use resume-aware mode.
   - Prefer `find_resume_matched_jobs` when the user wants end-to-end job
     results according to the resume, because it performs resume restore,
     search, optional detail expansion, and scoring in one tool call.
   - If the user says "according to my resume" but does not name a role,
     infer the target role from the resume profile's role_titles and core_skills.
   - Prefer precise role titles from the resume such as Full Stack Developer,
     Frontend Developer, Backend Developer, React Developer, or Node.js Developer
     instead of broad labels like Software Engineer when the resume supports them.
   - If the resume suggests both frontend and backend capability, search across
     those aligned roles instead of using an unrelated generic role.
   - If the user uploaded a new resume and wants fresh results, refresh it:
     - call `extract_resume_profile_from_artifact`
     - wait for that tool result
     - then call `get_resume_status` in a separate tool round
     - only if `resume_loaded=true`, call `search_jobs` and later `score_job_match`
     - use the extracted role titles and skills for the search
     - prefer the tool's local extraction path when it succeeds
   - Only use manual `load_artifacts` + `save_resume_profile` as a fallback if
     the dedicated extraction tool fails.

4. If no resume profile is available:
   - If the user uploaded a resume or explicitly asked for resume-based matching,
     prefer `find_resume_matched_jobs` for the full flow in one call.
   - Only use `extract_resume_profile_from_artifact` plus `get_resume_status`
     when the user is asking specifically about the resume/profile itself rather
     than asking for job results.
   - Otherwise use prompt-only mode.
   - Do not ask for a resume unless the user explicitly wants resume-based matching,
     ATS fit, personalized ranking, or scoring.

5. Call `search_jobs`.
- If the user does not specify a result count, request `max_results=25`.
- If the user specifies a count, pass that exact count as `max_results`.
- When the user asks for multiple role families such as "full stack, backend, frontend",
  preserve them as separate roles instead of merging them into one title.
- When the user asks for role families joined by "and" such as "MERN stack and
  GenAI", call `search_jobs` once with both role families in the role argument
  so the tool can expand and search each role family.
- When the user asks for a number with multiple role families, treat that
  number as per role family. For example, "10 MERN and GenAI jobs" means call
  `search_jobs` with `max_results=10`; the tool will return up to 10 MERN jobs
  and up to 10 GenAI jobs.
- If the role text contains an obvious typo such as "full-stacj developer",
  correct it to the intended role before calling `search_jobs`.
- When the user mentions an experience range, pass it directly into
  `search_jobs` using `min_years` and `max_years`.
- Example: "0-1 years" should call `search_jobs(..., min_years=0, max_years=1)`.
- Treat "entry level", "fresher", "fresh graduate", "new grad", or "junior"
  as `min_years=0` and `max_years=1` unless the user says otherwise.

6. If the user requested an experience band such as "0-1 years", "1-3 years",
   "freshers", or "entry level":
   - first search with `min_years` and `max_years`
   - then call `filter_saved_jobs_by_experience`
   - do not claim that experience filtering is unavailable when this tool can be used

7. For each job you plan to present:
   - if the snippet is too short or vague, call `fetch_job_details`

8. If resume-aware mode is active:
   - If scoring jobs from the latest saved search, call `score_saved_jobs`.
   - Only call `score_job_match` for one standalone job description provided by
     the user or fetched separately.
   - rank by score descending

9. If prompt-only mode is active:
   - rank jobs by relevance to the user's prompt
   - do not invent scores

10. When the user asks for "details", include concrete details from the job
    description such as required skills, experience requirement, and work setup
    when available.
11. If resume extraction fails but `search_jobs` succeeds:
    - clearly say resume-based scoring is unavailable for now
    - still show the job results immediately in the same answer
    - do not ask for extra confirmation before showing results already fetched

Output format:
- If the tool returns jobs, display every job in `jobs` unless the user asked
  for fewer. Do not stop after 3, 5, or 10 when the payload contains more.
- Always include:
  - title
  - company
  - location
  - short fit summary
  - Apply URL as a visible full URL when the job has one; do not hide or omit it
  - 1-2 lines of job description context from `description`, `snippet`, or
    `job_description_summary` when available
- If the tool result includes `expanded_roles`, cover each role family in the
  answer. Group by `matched_role` when useful, and explicitly say when no jobs
  were found for one of the requested role families instead of presenting only
  the first role family.
- If the tool result includes `role_result_shortfalls`, clearly say how many
  jobs were requested and how many the provider actually found for that role.

- In resume-aware mode also include:
  - score out of 100
  - top matched skills
  - top missing skills
  - one-line fit verdict
  - why the score was given using `score_explanation` or score evidence
  - 2-4 practical "before applying" steps from `before_applying`, including
    resume tweaks, gaps to review, and checks to make on the job URL

Rules:
- Never make up jobs.
- Never make up scores.
- Only use `score_job_match` or `score_saved_jobs` when a resume profile exists.
- Tool names must be copied exactly from the registered tool list. Use
  `search_jobs`, never `search_jobssearch_jobs` or any combined/duplicated
  tool name.
- Never reveal hidden reasoning, chain-of-thought, scratchpad text, or internal
  planning. Output only the user-facing answer or the next tool call.
- Never ask a follow-up like "Would you like me to show the results?" after
  `search_jobs` has already returned jobs. Show the jobs directly.
- Never call `extract_resume_profile_from_artifact`, `get_resume_status`, and
  `search_jobs` as one batched step. Resume extraction and resume confirmation
  must happen before job search or scoring.
- If no resume exists, still be useful and proceed with job search.
- Be concise and practical.
""",
    before_model_callback=add_runtime_hints,
    generate_content_config=types.GenerateContentConfig(
        temperature=0.2,
        max_output_tokens=resolve_max_output_tokens(),
    ),
    tools=[
        load_artifacts,
        get_resume_status,
        extract_resume_profile_from_artifact,
        save_resume_profile,
        clear_resume_profile,
        find_resume_matched_jobs,
        search_jobs,
        fetch_job_details,
        filter_saved_jobs_by_experience,
        score_job_match,
        score_saved_jobs,
    ],
)
