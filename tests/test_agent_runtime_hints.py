from __future__ import annotations

import asyncio

from job_scout.agent import add_runtime_hints
from job_scout.agent import root_agent
from google.adk.models.llm_request import LlmRequest
from google.genai import types
from job_scout.state_keys import LAST_SEARCH_RESULTS_STATE_KEY
from job_scout.state_keys import RESUME_PROFILE_EXTRACTION_METHOD_STATE_KEY
from job_scout.state_keys import RESUME_PROFILE_STATE_KEY


class DummyCallbackContext:
    def __init__(self, state=None, artifact_names=None):
        self.state = state or {}
        self._artifact_names = artifact_names or []

    async def list_artifacts(self):
        return list(self._artifact_names)


class DummyLlmRequest:
    def __init__(self):
        self.model = None
        self.contents = []
        self.instructions = []

    def append_instructions(self, instructions):
        self.instructions.extend(instructions)


def test_add_runtime_hints_describes_saved_resume_and_previous_search():
    context = DummyCallbackContext(
        state={
            RESUME_PROFILE_STATE_KEY: {
                "resume_source": "resume.pdf",
                "role_titles": ["Frontend Developer"],
                "core_skills": ["React", "TypeScript"],
            },
            RESUME_PROFILE_EXTRACTION_METHOD_STATE_KEY: "local",
            LAST_SEARCH_RESULTS_STATE_KEY: {
                "role": "Frontend Developer",
                "location": "Remote",
                "jobs": [{"id": "demo-1", "title": "Frontend Developer"}],
            },
        }
    )
    request = DummyLlmRequest()

    asyncio.run(add_runtime_hints(context, request))

    combined = "\n".join(request.instructions)
    assert "A normalized resume profile already exists in session state." in combined
    assert "Current resume source: resume.pdf." in combined
    assert "Resume extraction method: local." in combined
    assert "Resume role titles: Frontend Developer." in combined
    assert "Resume core skills: React, TypeScript." in combined
    assert "A previous job search is also stored in session state." in combined


def test_add_runtime_hints_promotes_resume_extraction_when_artifact_exists():
    context = DummyCallbackContext(artifact_names=["resume.pdf"])
    request = DummyLlmRequest()

    asyncio.run(add_runtime_hints(context, request))

    combined = "\n".join(request.instructions)
    assert "No resume profile exists yet." in combined
    assert "An uploaded artifact is available in this session." in combined
    assert "`extract_resume_profile_from_artifact`" in combined
    assert "`get_resume_status`" in combined


def test_add_runtime_hints_rewrites_file_parts_for_nvidia_requests():
    context = DummyCallbackContext(artifact_names=["resume.pdf"])
    request = LlmRequest(
        model="nvidia_nim/nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
        contents=[
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(text="according to this resume find entry level jobs"),
                    types.Part.from_uri(
                        file_uri="artifact://resume.pdf/0",
                        mime_type="application/pdf",
                    ),
                ],
            )
        ],
    )

    asyncio.run(add_runtime_hints(context, request))

    assert len(request.contents) == 1
    parts = request.contents[0].parts
    assert parts[0].text == "according to this resume find entry level jobs"
    assert parts[1].text == "[Attached file: artifact://resume.pdf/0]"


def test_root_agent_instruction_requires_showing_jobs_without_extra_confirmation():
    instruction = root_agent.instruction

    assert "present those jobs" in instruction
    assert "immediately instead of asking whether the user wants to see them" in instruction
    assert "do not ask for extra confirmation before showing results already fetched" in instruction
    assert 'Never ask a follow-up like "Would you like me to show the results?"' in instruction
