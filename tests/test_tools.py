import os
import asyncio
import zlib
from types import SimpleNamespace

import job_scout.resume_support as resume_support_module
import job_scout.search_support as search_support_module
from job_scout.litellm_compat import _normalize_groq_content_blocks
from job_scout.litellm_compat import _normalize_content_payload
from job_scout.litellm_compat import _provider_rejects_file_parts
from job_scout.litellm_compat import _requires_content_block_normalization
from job_scout.litellm_compat import _repair_missing_tool_call_ids
from job_scout.tools import _filter_jobs_for_experience
from job_scout.tools import _normalize_country
from job_scout.tools import _extract_text_from_pdf_bytes
from job_scout.tools import _choose_resume_artifact_name
from job_scout.tools import extract_resume_profile_from_artifact
from job_scout.tools import fetch_job_details
from job_scout.tools import get_resume_status
from job_scout.tools import save_resume_profile
from job_scout.tools import score_job_match
from job_scout.tools import search_jobs


class DummyToolContext:
    def __init__(self):
        self.state = {}


class AsyncDummyToolContext(DummyToolContext):
    def __init__(self, artifact_names=None, artifacts=None, user_content=None):
        super().__init__()
        self._artifact_names = artifact_names or []
        self._artifacts = artifacts or {}
        self.user_content = user_content

    async def list_artifacts(self):
        return list(self._artifact_names)

    async def load_artifact(self, name):
        return self._artifacts.get(name)


def make_pdf_part(pdf_bytes):
    return SimpleNamespace(
        inline_data=SimpleNamespace(
            data=pdf_bytes,
            mime_type="application/pdf",
        )
    )


def make_user_content_with_inline_resume(pdf_bytes, display_name="resume.pdf"):
    return SimpleNamespace(
        parts=[
            SimpleNamespace(
                inline_data=SimpleNamespace(
                    data=pdf_bytes,
                    mime_type="application/pdf",
                    display_name=display_name,
                ),
                file_data=None,
                text=None,
            )
        ]
    )


def make_user_content_with_artifact_ref(display_name="resume.pdf", file_uri=None):
    return SimpleNamespace(
        parts=[
            SimpleNamespace(
                inline_data=None,
                file_data=SimpleNamespace(
                    file_uri=file_uri or f"artifact://apps/job_scout/users/test-user/sessions/test-session/artifacts/{display_name}/versions/0",
                    mime_type="application/pdf",
                    display_name=display_name,
                ),
                text=None,
            )
        ]
    )


def test_repair_missing_tool_call_id_for_single_tool_result():
    messages = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_123",
                    "type": "function",
                    "function": {"name": "search_jobs", "arguments": "{}"},
                }
            ],
        },
        {
            "role": "tool",
            "content": '{"status":"ok"}',
        },
    ]

    repaired = _repair_missing_tool_call_ids(messages)

    assert repaired[1]["tool_call_id"] == "call_123"


def test_repair_missing_tool_call_ids_in_tool_order():
    messages = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_a",
                    "type": "function",
                    "function": {"name": "search_jobs", "arguments": "{}"},
                },
                {
                    "id": "call_b",
                    "type": "function",
                    "function": {"name": "fetch_job_details", "arguments": "{}"},
                },
            ],
        },
        {
            "role": "tool",
            "content": '{"status":"ok"}',
        },
        {
            "role": "tool",
            "content": '{"status":"ok"}',
        },
    ]

    repaired = _repair_missing_tool_call_ids(messages)

    assert repaired[1]["tool_call_id"] == "call_a"
    assert repaired[2]["tool_call_id"] == "call_b"


def test_repair_synthesizes_assistant_and_tool_ids_when_both_missing():
    messages = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "",
                    "type": "function",
                    "function": {"name": "search_jobs", "arguments": "{}"},
                }
            ],
        },
        {
            "role": "tool",
            "content": '{"status":"ok"}',
        },
    ]

    repaired = _repair_missing_tool_call_ids(messages)

    assert repaired[0]["tool_calls"][0]["id"] == "jobscout_tool_call_0_0"
    assert repaired[1]["tool_call_id"] == "jobscout_tool_call_0_0"


def test_normalize_groq_content_blocks_rewrites_file_attachment_to_text():
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "score my resume for these jobs"},
                {
                    "type": "file",
                    "file": {
                        "file_id": "artifact://resume.pdf/0",
                        "format": "application/pdf",
                    },
                },
            ],
        }
    ]

    normalized = _normalize_groq_content_blocks(messages)

    assert isinstance(normalized[0]["content"], str)
    assert "score my resume for these jobs" in normalized[0]["content"]
    assert "artifact://resume.pdf/0" in normalized[0]["content"]


def test_nvidia_models_use_content_block_normalization():
    assert _requires_content_block_normalization(
        "nvidia_nim/nvidia/nemotron-3-nano-omni-30b-a3b-reasoning"
    ) is True
    assert _requires_content_block_normalization("groq/llama-3.3-70b-versatile") is True
    assert _requires_content_block_normalization("gemini-2.0-flash") is False


def test_nvidia_provider_rejects_file_parts():
    assert _provider_rejects_file_parts("nvidia_nim") is True
    assert _provider_rejects_file_parts("groq") is False


def test_normalize_content_payload_rewrites_file_blocks_to_text():
    content = [
        {"type": "text", "text": "score this resume"},
        {
            "type": "file",
            "file": {
                "file_id": "artifact://resume.pdf/0",
                "format": "application/pdf",
            },
        },
    ]

    normalized = _normalize_content_payload(content)

    assert isinstance(normalized, str)
    assert "score this resume" in normalized
    assert "artifact://resume.pdf/0" in normalized


def test_save_resume_profile_rejects_placeholder_values():
    tool_context = DummyToolContext()

    result = save_resume_profile(
        candidate_name="Unknown",
        professional_summary="N/A",
        role_titles=["Unknown"],
        core_skills=["Unknown"],
        additional_skills=["None"],
        years_experience=0,
        preferred_locations=["Hyderabad"],
        work_preferences=["Unknown"],
        notable_projects=["Unknown"],
        resume_source="attachment",
        tool_context=tool_context,
    )

    assert result["status"] == "error"
    assert tool_context.state == {}


def test_search_jobs_splits_multi_role_requests_in_demo_mode():
    original_token = os.environ.pop("APIFY_TOKEN", None)
    original_adzuna_id = os.environ.pop("ADZUNA_APP_ID", None)
    original_adzuna_key = os.environ.pop("ADZUNA_APP_KEY", None)
    original_provider = os.environ.pop("JOB_SCOUT_JOB_SEARCH_PROVIDER", None)

    try:
        result = search_jobs(
            role="full stack backend frontend",
            location="Hyderabad",
            max_results=10,
            country="India",
        )
    finally:
        if original_token is not None:
            os.environ["APIFY_TOKEN"] = original_token
        if original_adzuna_id is not None:
            os.environ["ADZUNA_APP_ID"] = original_adzuna_id
        if original_adzuna_key is not None:
            os.environ["ADZUNA_APP_KEY"] = original_adzuna_key
        if original_provider is not None:
            os.environ["JOB_SCOUT_JOB_SEARCH_PROVIDER"] = original_provider

    assert result["status"] == "demo_mode"
    assert result["expanded_roles"] == [
        "Full Stack Developer",
        "Backend Developer",
        "Frontend Developer",
    ]
    assert len(result["jobs"]) == 10
    assert [job["matched_role"] for job in result["jobs"][:3]] == result["expanded_roles"]
    assert all(job["fit_verdict"] for job in result["jobs"])
    assert all(job["reason_to_apply"] for job in result["jobs"])
    assert all(job["blocker_risk"] for job in result["jobs"])


def test_search_jobs_uses_adzuna_when_configured():
    original_token = os.environ.get("APIFY_TOKEN")
    original_adzuna_id = os.environ.get("ADZUNA_APP_ID")
    original_adzuna_key = os.environ.get("ADZUNA_APP_KEY")
    original_provider = os.environ.get("JOB_SCOUT_JOB_SEARCH_PROVIDER")
    original_client = search_support_module.httpx.Client

    class FakeResponse:
        def __init__(self, payload):
            self._payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def get(self, url, params=None, headers=None):
            assert "/jobs/in/search/1" in url.lower()
            assert params["app_id"] == "adzuna-id"
            assert params["app_key"] == "adzuna-key"
            assert params["where"] == "Hyderabad"
            if "backend developer" in params["what"].lower():
                return FakeResponse(
                    {
                        "results": [
                            {
                                "id": "adzuna-2",
                                "title": "Backend Developer Trainee",
                                "company": {"display_name": "DataNest"},
                                "location": {"display_name": "Hyderabad, Telangana"},
                                "description": "Trainee backend role with Python and APIs.",
                                "redirect_url": "https://example.com/jobs/adzuna-2",
                            }
                        ]
                    }
                )
            return FakeResponse(
                {
                    "results": [
                        {
                            "id": "adzuna-1",
                            "title": "Junior Full Stack Developer",
                            "company": {"display_name": "Acme Tech"},
                            "location": {"display_name": "Hyderabad, Telangana"},
                            "description": "Entry level React and Node.js role for freshers.",
                            "redirect_url": "https://example.com/jobs/adzuna-1",
                        }
                    ]
                }
            )

    os.environ["APIFY_TOKEN"] = "exhausted-but-present"
    os.environ["ADZUNA_APP_ID"] = "adzuna-id"
    os.environ["ADZUNA_APP_KEY"] = "adzuna-key"
    os.environ["JOB_SCOUT_JOB_SEARCH_PROVIDER"] = "adzuna"
    search_support_module.httpx.Client = FakeClient

    try:
        result = search_jobs(
            role="full-stack developer, backend developer",
            location="Hyderabad",
            max_results=2,
            country="India",
            min_years=0,
            max_years=1,
        )
    finally:
        search_support_module.httpx.Client = original_client
        if original_token is None:
            os.environ.pop("APIFY_TOKEN", None)
        else:
            os.environ["APIFY_TOKEN"] = original_token
        if original_adzuna_id is None:
            os.environ.pop("ADZUNA_APP_ID", None)
        else:
            os.environ["ADZUNA_APP_ID"] = original_adzuna_id
        if original_adzuna_key is None:
            os.environ.pop("ADZUNA_APP_KEY", None)
        else:
            os.environ["ADZUNA_APP_KEY"] = original_adzuna_key
        if original_provider is None:
            os.environ.pop("JOB_SCOUT_JOB_SEARCH_PROVIDER", None)
        else:
            os.environ["JOB_SCOUT_JOB_SEARCH_PROVIDER"] = original_provider

    assert result["status"] == "ok"
    assert result["provider"] == "adzuna"
    assert result["country_used"] == "IN"
    assert len(result["jobs"]) == 2
    assert result["jobs"][0]["company"] == "Acme Tech"
    assert result["jobs"][0]["url"] == "https://example.com/jobs/adzuna-1"


def test_search_jobs_uses_remote_aware_adzuna_strategy_for_india():
    original_token = os.environ.get("APIFY_TOKEN")
    original_adzuna_id = os.environ.get("ADZUNA_APP_ID")
    original_adzuna_key = os.environ.get("ADZUNA_APP_KEY")
    original_provider = os.environ.get("JOB_SCOUT_JOB_SEARCH_PROVIDER")
    original_country = os.environ.get("ADZUNA_COUNTRY")
    original_client = search_support_module.httpx.Client
    seen_calls = []

    class FakeResponse:
        def __init__(self, payload):
            self._payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def get(self, url, params=None, headers=None):
            seen_calls.append({"url": url, "params": dict(params or {})})
            assert "/jobs/in/search/1" in url.lower()
            assert params["app_id"] == "adzuna-id"
            assert params["app_key"] == "adzuna-key"
            assert params["where"] == "India"
            assert "remote" in params["what"].lower()
            return FakeResponse(
                {
                    "results": [
                        {
                            "id": "adzuna-remote-1",
                            "title": "Remote Full Stack Developer",
                            "company": {"display_name": "RemoteBase"},
                            "location": {"display_name": "Remote, India"},
                            "description": "Remote work from home role with React and Node.js.",
                            "redirect_url": "https://example.com/jobs/adzuna-remote-1",
                        },
                        {
                            "id": "adzuna-onsite-2",
                            "title": "Full Stack Developer",
                            "company": {"display_name": "OfficeOnly"},
                            "location": {"display_name": "Hyderabad, Telangana"},
                            "description": "Onsite role with React and Node.js.",
                            "redirect_url": "https://example.com/jobs/adzuna-onsite-2",
                        },
                    ]
                }
            )

    os.environ["APIFY_TOKEN"] = "exhausted-but-present"
    os.environ["ADZUNA_APP_ID"] = "adzuna-id"
    os.environ["ADZUNA_APP_KEY"] = "adzuna-key"
    os.environ["ADZUNA_COUNTRY"] = "in"
    os.environ["JOB_SCOUT_JOB_SEARCH_PROVIDER"] = "adzuna"
    search_support_module.httpx.Client = FakeClient

    try:
        result = search_jobs(
            role="full-stack developer",
            location="Remote",
            max_results=5,
            country="India",
        )
    finally:
        search_support_module.httpx.Client = original_client
        if original_token is None:
            os.environ.pop("APIFY_TOKEN", None)
        else:
            os.environ["APIFY_TOKEN"] = original_token
        if original_adzuna_id is None:
            os.environ.pop("ADZUNA_APP_ID", None)
        else:
            os.environ["ADZUNA_APP_ID"] = original_adzuna_id
        if original_adzuna_key is None:
            os.environ.pop("ADZUNA_APP_KEY", None)
        else:
            os.environ["ADZUNA_APP_KEY"] = original_adzuna_key
        if original_provider is None:
            os.environ.pop("JOB_SCOUT_JOB_SEARCH_PROVIDER", None)
        else:
            os.environ["JOB_SCOUT_JOB_SEARCH_PROVIDER"] = original_provider
        if original_country is None:
            os.environ.pop("ADZUNA_COUNTRY", None)
        else:
            os.environ["ADZUNA_COUNTRY"] = original_country

    assert result["status"] == "ok"
    assert result["provider"] == "adzuna"
    assert result["country_used"] == "IN"
    assert len(result["jobs"]) == 1
    assert result["jobs"][0]["title"] == "Remote Full Stack Developer"
    assert seen_calls[0]["params"]["where"] == "India"
    assert seen_calls[0]["params"]["what"].lower().count("remote") >= 1


def test_search_jobs_uses_browseract_when_configured():
    original_token = os.environ.get("APIFY_TOKEN")
    original_adzuna_id = os.environ.get("ADZUNA_APP_ID")
    original_adzuna_key = os.environ.get("ADZUNA_APP_KEY")
    original_provider = os.environ.get("JOB_SCOUT_JOB_SEARCH_PROVIDER")
    original_browseract_key = os.environ.get("BROWSERACT_API_KEY")
    original_browseract_workflow_id = os.environ.get("BROWSERACT_WORKFLOW_ID")
    original_client = search_support_module.httpx.Client

    class FakeResponse:
        def __init__(self, payload):
            self._payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    class FakeClient:
        def __init__(self, *args, **kwargs):
            self.post_calls = []
            self.get_calls = []

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def post(self, url, headers=None, json=None):
            self.post_calls.append({"url": url, "headers": headers, "json": json})
            assert url.endswith("/run-task")
            assert headers["Authorization"] == "Bearer browseract-key"
            assert json["workflow_id"] == "wf_jobs_123"
            input_parameters = {item["name"]: item["value"] for item in json["input_parameters"]}
            assert input_parameters["keyword"] == "Frontend Developer junior entry level"
            assert input_parameters["location"] == "Remote"
            assert input_parameters["datalimit"] == 8
            assert input_parameters["country"] == "IN"
            return FakeResponse({"id": "task_123"})

        def get(self, url, params=None, headers=None):
            self.get_calls.append({"url": url, "params": params, "headers": headers})
            assert url.endswith("/get-task")
            assert params == {"task_id": "task_123"}
            assert headers["Authorization"] == "Bearer browseract-key"
            return FakeResponse(
                {
                    "status": "finished",
                    "output": {
                        "string": (
                            '[{"title":"Frontend Developer","company":"RemoteBase",'
                            '"location":"Remote, India","description":"React role",'
                            '"url":"https://example.com/jobs/browseract-1"}]'
                        )
                    },
                }
            )

    os.environ["APIFY_TOKEN"] = "apify-fallback"
    os.environ["ADZUNA_APP_ID"] = "adzuna-id"
    os.environ["ADZUNA_APP_KEY"] = "adzuna-key"
    os.environ["BROWSERACT_API_KEY"] = "browseract-key"
    os.environ["BROWSERACT_WORKFLOW_ID"] = "wf_jobs_123"
    os.environ["JOB_SCOUT_JOB_SEARCH_PROVIDER"] = "browseract"
    search_support_module.httpx.Client = FakeClient

    try:
        result = search_jobs(
            role="Frontend Developer",
            location="Remote",
            max_results=2,
            country="India",
            min_years=0,
            max_years=2,
        )
    finally:
        search_support_module.httpx.Client = original_client
        if original_token is None:
            os.environ.pop("APIFY_TOKEN", None)
        else:
            os.environ["APIFY_TOKEN"] = original_token
        if original_adzuna_id is None:
            os.environ.pop("ADZUNA_APP_ID", None)
        else:
            os.environ["ADZUNA_APP_ID"] = original_adzuna_id
        if original_adzuna_key is None:
            os.environ.pop("ADZUNA_APP_KEY", None)
        else:
            os.environ["ADZUNA_APP_KEY"] = original_adzuna_key
        if original_provider is None:
            os.environ.pop("JOB_SCOUT_JOB_SEARCH_PROVIDER", None)
        else:
            os.environ["JOB_SCOUT_JOB_SEARCH_PROVIDER"] = original_provider
        if original_browseract_key is None:
            os.environ.pop("BROWSERACT_API_KEY", None)
        else:
            os.environ["BROWSERACT_API_KEY"] = original_browseract_key
        if original_browseract_workflow_id is None:
            os.environ.pop("BROWSERACT_WORKFLOW_ID", None)
        else:
            os.environ["BROWSERACT_WORKFLOW_ID"] = original_browseract_workflow_id

    assert result["status"] == "ok"
    assert result["provider"] == "browseract"
    assert result["country_used"] == "IN"
    assert len(result["jobs"]) == 1
    assert result["jobs"][0]["company"] == "RemoteBase"
    assert result["jobs"][0]["url"] == "https://example.com/jobs/browseract-1"


def test_entry_level_filter_rejects_senior_results_when_experience_is_unknown():
    jobs = [
        {
            "title": "Senior Full Stack Engineer",
            "snippet": "Build web applications with React and Node.js.",
        },
        {
            "title": "Lead Backend Developer",
            "snippet": "Work on APIs and distributed systems.",
        },
        {
            "title": "Frontend Developer",
            "snippet": "Freshers and entry level candidates can apply.",
        },
    ]

    matched, fallback = _filter_jobs_for_experience(jobs, min_years=0, max_years=1)

    assert [job["title"] for job in matched] == ["Frontend Developer"]
    assert fallback == []


def test_choose_resume_artifact_name_prefers_pdf_resume():
    artifact = _choose_resume_artifact_name(
        ["notes.txt", "resume.docx", "chaitanya_resume.pdf"],
    )

    assert artifact == "chaitanya_resume.pdf"


def test_extract_text_from_pdf_bytes_reads_text_stream():
    content_stream = "\n".join([
        "BT",
        "(GALLA CHAITANYA) Tj",
        "(Full Stack Developer | React Developer | Node.js Developer) Tj",
        "(PROFESSIONAL SUMMARY) Tj",
        "(Results-driven Full Stack Developer with 1+ year of experience building React and Node.js apps.) Tj",
        "(TECHNICAL SKILLS) Tj",
        "(Frontend: React, Next.js, TypeScript, Tailwind CSS) Tj",
        "(Backend: Node.js, Express.js, REST API, PostgreSQL, Redis) Tj",
        "(PROJECTS) Tj",
        "(AI-Powered YouTube Shorts Automation | Python, FFmpeg) Tj",
        "ET",
    ]).encode("utf-8")
    pdf_bytes = (
        b"%PDF-1.4\n"
        b"1 0 obj\n<< /Length 2 0 R /Filter /FlateDecode >>\nstream\n"
        + zlib.compress(content_stream)
        + b"\nendstream\nendobj\n"
    )

    extracted = _extract_text_from_pdf_bytes(pdf_bytes)

    assert "GALLA CHAITANYA" in extracted
    assert "TECHNICAL SKILLS" in extracted
    assert "React, Next.js, TypeScript" in extracted


def test_extract_text_from_pdf_bytes_tolerates_non_octal_digit_escapes():
    content_stream = "\n".join([
        "BT",
        r"(GALLA CHAITANYA\9Resume) Tj",
        r"(Frontend Developer with React\8 TypeScript experience) Tj",
        "(TECHNICAL SKILLS) Tj",
        "(React, TypeScript, Tailwind CSS, HTML5, CSS3) Tj",
        "(PROJECTS) Tj",
        "(Interactive Dashboard Platform) Tj",
        "ET",
    ]).encode("utf-8")
    pdf_bytes = (
        b"%PDF-1.4\n"
        b"1 0 obj\n<< /Length 2 0 R /Filter /FlateDecode >>\nstream\n"
        + zlib.compress(content_stream)
        + b"\nendstream\nendobj\n"
    )

    extracted = _extract_text_from_pdf_bytes(pdf_bytes)

    assert "GALLA CHAITANYA9Resume" in extracted
    assert "React8 TypeScript" in extracted


def test_normalize_country_avoids_substring_false_positives():
    assert _normalize_country("Austin, TX", None) is None
    assert _normalize_country("Remote - United States", None) == "US"
    assert _normalize_country("Hyderabad", "India") == "IN"


def test_fetch_job_details_rejects_non_http_urls():
    javascript_result = fetch_job_details("javascript:alert(1)")
    file_result = fetch_job_details("file:///tmp/job.html")
    blank_result = fetch_job_details("")

    assert javascript_result["status"] == "error"
    assert javascript_result["message"] == "Only http(s) job URLs are supported."
    assert file_result["status"] == "error"
    assert file_result["message"] == "Only http(s) job URLs are supported."
    assert blank_result["status"] == "error"
    assert blank_result["message"] == "Job URL is required."


def test_extract_resume_profile_from_artifact_prefers_local_parser():
    content_stream = "\n".join([
        "BT",
        "(GALLA CHAITANYA) Tj",
        "(Full Stack Developer | React Developer | Node.js Developer) Tj",
        "(Hyderabad, Telangana) Tj",
        "(PROFESSIONAL SUMMARY) Tj",
        "(Results-driven Full Stack Developer with 1+ year of experience architecting scalable web applications using React, Next.js, Node.js, Express.js, PostgreSQL, Redis, and AWS.) Tj",
        "(TECHNICAL SKILLS) Tj",
        "(Frontend: React, Next.js, JavaScript, TypeScript, Tailwind CSS, HTML5, CSS3) Tj",
        "(Backend: Node.js, Express.js, REST API, PostgreSQL, MongoDB, Redis, Prisma ORM) Tj",
        "(DevOps & Cloud: AWS, Docker, CI/CD, GitHub Actions, Linux, Nginx) Tj",
        "(PROJECTS) Tj",
        "(AI-Powered YouTube Shorts Automation | Python, FFmpeg) Tj",
        "(VideoSave | Next.js, TypeScript, Express.js, Prisma, Socket.io, Zod) Tj",
        "ET",
    ]).encode("utf-8")
    pdf_bytes = (
        b"%PDF-1.4\n"
        b"1 0 obj\n<< /Length 2 0 R /Filter /FlateDecode >>\nstream\n"
        + zlib.compress(content_stream)
        + b"\nendstream\nendobj\n"
    )
    original_parser = resume_support_module._parse_resume_with_gemini

    async def fake_parser(artifact_name, artifact_part):
        raise AssertionError("Gemini fallback should not be called when local parsing succeeds")

    resume_support_module._parse_resume_with_gemini = fake_parser
    tool_context = AsyncDummyToolContext(
        artifact_names=["chaitanya_resume.pdf"],
        artifacts={"chaitanya_resume.pdf": make_pdf_part(pdf_bytes)},
    )

    try:
        result = asyncio.run(extract_resume_profile_from_artifact(tool_context=tool_context))
    finally:
        resume_support_module._parse_resume_with_gemini = original_parser

    assert result["status"] == "ok"
    assert result["extraction_method"] == "local"
    assert tool_context.state["resume_profile"]["candidate_name"] == "Galla Chaitanya"
    assert tool_context.state["resume_profile"]["role_titles"][:3] == [
        "Full Stack Developer",
        "Frontend Developer",
        "Backend Developer",
    ]
    assert "React" in tool_context.state["resume_profile"]["core_skills"]


def test_extract_resume_profile_from_current_user_message_when_session_artifacts_are_empty():
    content_stream = "\n".join([
        "BT",
        "(GALLA CHAITANYA) Tj",
        "(Full Stack Developer | React Developer | Node.js Developer) Tj",
        "(Hyderabad, Telangana) Tj",
        "(PROFESSIONAL SUMMARY) Tj",
        "(Results-driven Full Stack Developer with 1+ year of experience architecting scalable web applications using React, Next.js, Node.js, Express.js, PostgreSQL, Redis, and AWS.) Tj",
        "(TECHNICAL SKILLS) Tj",
        "(Frontend: React, Next.js, JavaScript, TypeScript, Tailwind CSS, HTML5, CSS3) Tj",
        "(Backend: Node.js, Express.js, REST API, PostgreSQL, MongoDB, Redis, Prisma ORM) Tj",
        "(PROJECTS) Tj",
        "(AI-Powered YouTube Shorts Automation | Python, FFmpeg) Tj",
        "ET",
    ]).encode("utf-8")
    pdf_bytes = (
        b"%PDF-1.4\n"
        b"1 0 obj\n<< /Length 2 0 R /Filter /FlateDecode >>\nstream\n"
        + zlib.compress(content_stream)
        + b"\nendstream\nendobj\n"
    )
    tool_context = AsyncDummyToolContext(
        artifact_names=[],
        artifacts={},
        user_content=make_user_content_with_inline_resume(pdf_bytes, display_name="chaitanya_resume.pdf"),
    )

    result = asyncio.run(extract_resume_profile_from_artifact(tool_context=tool_context))

    assert result["status"] == "ok"
    assert result["extraction_method"] == "local"
    assert tool_context.state["resume_profile"]["candidate_name"] == "Galla Chaitanya"


def test_extract_resume_profile_can_load_user_message_artifact_ref_even_when_list_is_empty():
    content_stream = "\n".join([
        "BT",
        "(GALLA CHAITANYA) Tj",
        "(Frontend Developer | React Developer | Full Stack Developer) Tj",
        "(Hyderabad, Telangana) Tj",
        "(PROFESSIONAL SUMMARY) Tj",
        "(Frontend developer with React, TypeScript, Tailwind CSS, Next.js, Node.js, REST API, and PostgreSQL experience building responsive web applications.) Tj",
        "(TECHNICAL SKILLS) Tj",
        "(Frontend: React, Next.js, TypeScript, Tailwind CSS, HTML5, CSS3) Tj",
        "(Backend: Node.js, Express.js, REST API, PostgreSQL, Redis) Tj",
        "(PROJECTS) Tj",
        "(Interactive Dashboard Platform | React, TypeScript, Tailwind CSS, REST API) Tj",
        "ET",
    ]).encode("utf-8")
    pdf_bytes = (
        b"%PDF-1.4\n"
        b"1 0 obj\n<< /Length 2 0 R /Filter /FlateDecode >>\nstream\n"
        + zlib.compress(content_stream)
        + b"\nendstream\nendobj\n"
    )
    tool_context = AsyncDummyToolContext(
        artifact_names=[],
        artifacts={"resume.pdf": make_pdf_part(pdf_bytes)},
        user_content=make_user_content_with_artifact_ref(display_name="resume.pdf"),
    )

    result = asyncio.run(extract_resume_profile_from_artifact(tool_context=tool_context))

    assert result["status"] == "ok"
    assert result["resume_source"] == "resume.pdf"
    assert tool_context.state["resume_profile"]["candidate_name"] == "Galla Chaitanya"


def test_extract_resume_profile_from_artifact_falls_back_to_gemini_when_local_is_weak():
    original_text_parser = resume_support_module._parse_resume_with_text_model
    original_parser = resume_support_module._parse_resume_with_gemini

    async def fake_text_parser(artifact_name, resume_text):
        raise ValueError("nvidia parser could not recover enough structure")

    async def fake_parser(artifact_name, artifact_part):
        return {
            "candidate_name": "Galla Chaitanya",
            "professional_summary": "Full stack developer building React and Node.js applications.",
            "role_titles": ["Full Stack Developer"],
            "core_skills": ["React", "Node.js", "PostgreSQL"],
            "additional_skills": ["AWS"],
            "years_experience": 1.0,
            "preferred_locations": ["Hyderabad"],
            "work_preferences": [],
            "notable_projects": ["VideoSave"],
        }

    resume_support_module._parse_resume_with_text_model = fake_text_parser
    resume_support_module._parse_resume_with_gemini = fake_parser
    weak_pdf_bytes = b"%PDF-1.4\n1 0 obj\n<<>>\nstream\nabc\nendstream\nendobj\n"
    tool_context = AsyncDummyToolContext(
        artifact_names=["resume.pdf"],
        artifacts={"resume.pdf": make_pdf_part(weak_pdf_bytes)},
    )

    try:
        result = asyncio.run(extract_resume_profile_from_artifact(tool_context=tool_context))
    finally:
        resume_support_module._parse_resume_with_text_model = original_text_parser
        resume_support_module._parse_resume_with_gemini = original_parser

    assert result["status"] == "ok"
    assert result["extraction_method"] == "gemini_attachment_fallback"
    assert tool_context.state["resume_profile"]["candidate_name"] == "Galla Chaitanya"


def test_extract_resume_profile_prefers_text_model_fallback_before_gemini():
    original_text_parser = resume_support_module._parse_resume_with_text_model
    original_parser = resume_support_module._parse_resume_with_gemini

    async def fake_text_parser(artifact_name, resume_text):
        return {
            "candidate_name": "Galla Chaitanya",
            "professional_summary": "Backend-focused full stack developer building React and Node.js applications.",
            "role_titles": ["Full Stack Developer", "Backend Developer"],
            "core_skills": ["React", "Node.js", "PostgreSQL"],
            "additional_skills": ["AWS"],
            "years_experience": 1.0,
            "preferred_locations": ["Hyderabad"],
            "work_preferences": [],
            "notable_projects": [
                {
                    "title": "AI Powered YouTube Shorts Automation",
                    "summary": "Automated short-form video generation workflow.",
                },
                {
                    "title": "VideoSave",
                    "technologies": ["Next.js", "TypeScript", "Express.js"],
                },
            ],
        }

    async def fake_parser(artifact_name, artifact_part):
        raise AssertionError("Gemini attachment fallback should not run when text-model parsing succeeds")

    resume_support_module._parse_resume_with_text_model = fake_text_parser
    resume_support_module._parse_resume_with_gemini = fake_parser
    weak_pdf_bytes = (
        b"%PDF-1.4\n"
        b"1 0 obj\n<< /Length 2 0 R /Filter /FlateDecode >>\nstream\n"
        + zlib.compress(b"BT\n(GALLA CHAITANYA) Tj\n(React Node PostgreSQL Hyderabad) Tj\nET")
        + b"\nendstream\nendobj\n"
    )
    tool_context = AsyncDummyToolContext(
        artifact_names=["resume.pdf"],
        artifacts={"resume.pdf": make_pdf_part(weak_pdf_bytes)},
    )

    try:
        result = asyncio.run(extract_resume_profile_from_artifact(tool_context=tool_context))
    finally:
        resume_support_module._parse_resume_with_text_model = original_text_parser
        resume_support_module._parse_resume_with_gemini = original_parser

    assert result["status"] == "ok"
    assert result["extraction_method"] == "llm_text_fallback"
    assert tool_context.state["resume_profile"]["candidate_name"] == "Galla Chaitanya"
    assert tool_context.state["resume_profile"]["notable_projects"] == [
        "AI Powered YouTube Shorts Automation - Automated short-form video generation workflow.",
        "VideoSave | Next.js, TypeScript, Express.js",
    ]


def test_sanitize_part_for_gemini_removes_display_name():
    part = make_user_content_with_inline_resume(b"pdf-bytes", display_name="resume.pdf").parts[0]

    sanitized = resume_support_module._sanitize_part_for_gemini(part)

    assert sanitized.inline_data.display_name is None
    assert sanitized.inline_data.mime_type == "application/pdf"


def test_get_resume_status_restores_profile_from_uploaded_artifact():
    content_stream = "\n".join([
        "BT",
        "(GALLA CHAITANYA) Tj",
        "(Full Stack Developer | React Developer | Node.js Developer) Tj",
        "(Hyderabad, Telangana) Tj",
        "(PROFESSIONAL SUMMARY) Tj",
        "(Results-driven Full Stack Developer with 1+ year of experience building React, Next.js, Node.js, Express.js, PostgreSQL, MongoDB, Redis, AWS, and Docker based web applications.) Tj",
        "(TECHNICAL SKILLS) Tj",
        "(Frontend: React, Next.js, TypeScript, Tailwind CSS, HTML5, CSS3) Tj",
        "(Backend: Node.js, Express.js, REST API, PostgreSQL, MongoDB, Redis) Tj",
        "(DevOps & Cloud: AWS, Docker, CI/CD, GitHub Actions, Linux, Nginx) Tj",
        "(PROJECTS) Tj",
        "(AI-Powered YouTube Shorts Automation | Python, FFmpeg) Tj",
        "(VideoSave | Next.js, TypeScript, Express.js, Prisma, Socket.io, Zod) Tj",
        "ET",
    ]).encode("utf-8")
    pdf_bytes = (
        b"%PDF-1.4\n"
        b"1 0 obj\n<< /Length 2 0 R /Filter /FlateDecode >>\nstream\n"
        + zlib.compress(content_stream)
        + b"\nendstream\nendobj\n"
    )
    tool_context = AsyncDummyToolContext(
        artifact_names=["resume.pdf"],
        artifacts={"resume.pdf": make_pdf_part(pdf_bytes)},
    )

    result = asyncio.run(get_resume_status(tool_context))

    assert result["status"] == "ok"
    assert result["resume_loaded"] is True
    assert result["candidate_name"] == "Galla Chaitanya"
    assert tool_context.state["resume_profile"]["candidate_name"] == "Galla Chaitanya"


def test_score_job_match_restores_profile_from_uploaded_artifact():
    content_stream = "\n".join([
        "BT",
        "(GALLA CHAITANYA) Tj",
        "(Frontend Developer | React Developer | Full Stack Developer) Tj",
        "(Hyderabad, Telangana) Tj",
        "(PROFESSIONAL SUMMARY) Tj",
        "(Frontend Developer with React, Next.js, TypeScript, Tailwind CSS, Node.js, and REST API experience building responsive web applications.) Tj",
        "(TECHNICAL SKILLS) Tj",
        "(Frontend: React, Next.js, TypeScript, Tailwind CSS, HTML5, CSS3) Tj",
        "(Backend: Node.js, Express.js, REST API, PostgreSQL) Tj",
        "(PROJECTS) Tj",
        "(Interactive Dashboard Platform | React, TypeScript, Tailwind CSS, REST API) Tj",
        "ET",
    ]).encode("utf-8")
    pdf_bytes = (
        b"%PDF-1.4\n"
        b"1 0 obj\n<< /Length 2 0 R /Filter /FlateDecode >>\nstream\n"
        + zlib.compress(content_stream)
        + b"\nendstream\nendobj\n"
    )
    tool_context = AsyncDummyToolContext(
        artifact_names=["resume.pdf"],
        artifacts={"resume.pdf": make_pdf_part(pdf_bytes)},
    )

    result = asyncio.run(score_job_match(
        job_title="Frontend Developer",
        job_description=(
            "We need a frontend developer with strong React, TypeScript, "
            "Tailwind CSS, and HTML skills."
        ),
        tool_context=tool_context,
    ))

    assert result["score"] >= 1
    assert result["resume_source"] == "resume.pdf"
    assert tool_context.state["resume_profile"]["candidate_name"] == "Galla Chaitanya"


def test_score_job_match_includes_evidence_and_blockers_without_breaking_existing_fields():
    tool_context = DummyToolContext()
    saved = save_resume_profile(
        candidate_name="Galla Chaitanya",
        professional_summary="Frontend developer with React and TypeScript experience.",
        role_titles=["Frontend Developer"],
        core_skills=["React", "TypeScript", "HTML5", "CSS3"],
        additional_skills=["REST API"],
        years_experience=1.0,
        preferred_locations=["Hyderabad"],
        work_preferences=["Remote"],
        notable_projects=["Interactive Dashboard Platform"],
        resume_source="manual",
        tool_context=tool_context,
    )
    assert saved["status"] == "ok"

    result = asyncio.run(score_job_match(
        job_title="Frontend Developer",
        job_description=(
            "Required: React, TypeScript, AWS. "
            "Must have at least 3 years of experience. "
            "Preferred: Tailwind CSS."
        ),
        tool_context=tool_context,
    ))

    assert result["score"] >= 0
    assert "signals" in result
    assert "matched_skills" in result
    assert "missing_skills" in result
    assert "explanation" in result
    assert result["evidence"]["required_years"] == 3.0
    assert "aws" in result["evidence"]["missing_required_skills"]
    assert "react" in result["evidence"]["matched_required_skills"]
    assert result["blockers"]["has_blockers"] is True
    assert result["blockers"]["insufficient_experience"] is True
    assert "aws" in result["blockers"]["missing_required_skills"]
    assert result["fit_verdict"]
    assert result["reason_to_apply"]
    assert result["blocker_risk"] == "High"
