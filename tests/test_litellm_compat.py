from __future__ import annotations

from google.adk.models import lite_llm as lite_llm_module

from job_scout.litellm_compat import patch_litellm_tool_call_id_repair


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
