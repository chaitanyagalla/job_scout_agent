from __future__ import annotations

from google.adk.models import lite_llm as lite_llm_module

from job_scout.litellm_compat import patch_litellm_tool_call_id_repair
from job_scout.litellm_compat import _repair_json_arguments_payload
from job_scout.litellm_compat import _repair_tool_call_name


def test_litellm_reasoning_payloads_are_suppressed_by_default(monkeypatch):
    patch_litellm_tool_call_id_repair()
    monkeypatch.delenv("JOB_SCOUT_INCLUDE_REASONING_PARTS", raising=False)

    message = {"reasoning_content": "internal reasoning"}

    assert lite_llm_module._extract_reasoning_value(message) is None


def test_litellm_reasoning_payloads_can_be_opted_back_in(monkeypatch):
    patch_litellm_tool_call_id_repair()
    monkeypatch.setenv("JOB_SCOUT_INCLUDE_REASONING_PARTS", "true")

    message = {"reasoning_content": "internal reasoning"}

    assert lite_llm_module._extract_reasoning_value(message) == "internal reasoning"


def test_repair_json_arguments_payload_merges_concatenated_objects():
    repaired = _repair_json_arguments_payload(
        '{"role":"MERN Stack Developer"} {"location":"Hyderabad","max_results":10}'
    )

    assert repaired == (
        '{"role": "MERN Stack Developer", "location": "Hyderabad", "max_results": 10}'
    )


def test_repair_json_arguments_payload_trims_extra_tail_text():
    repaired = _repair_json_arguments_payload(
        '{"role":"GenAI Engineer","location":"Hyderabad"} trailing text'
    )

    assert repaired == '{"role": "GenAI Engineer", "location": "Hyderabad"}'


def test_repair_tool_call_name_deduplicates_registered_tool_name():
    assert _repair_tool_call_name("search_jobssearch_jobs") == "search_jobs"
    assert _repair_tool_call_name("fetch_job_detailsfetch_job_details") == "fetch_job_details"


def test_repair_tool_call_name_leaves_unknown_name_unchanged():
    assert _repair_tool_call_name("unknown_toolunknown_tool") == "unknown_toolunknown_tool"
