"""Compatibility helpers for LiteLLM-backed tool calling."""
from __future__ import annotations

import logging
import os
from typing import Iterable
from typing import Any

logger = logging.getLogger(__name__)
CONTENT_NORMALIZATION_MODEL_PREFIXES = ("groq/", "nvidia_nim/")
FILE_PART_UNSUPPORTED_PROVIDERS = {"nvidia_nim"}


def _env_flag_is_true(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


def reasoning_parts_enabled() -> bool:
    return _env_flag_is_true(os.getenv("JOB_SCOUT_INCLUDE_REASONING_PARTS"))


def _message_get(message: Any, key: str, default: Any = None) -> Any:
    if isinstance(message, dict):
        return message.get(key, default)

    getter = getattr(message, "get", None)
    if callable(getter):
        try:
            return getter(key, default)
        except TypeError:
            pass

    return getattr(message, key, default)


def _message_set(message: Any, key: str, value: Any) -> None:
    if isinstance(message, dict):
        message[key] = value
        return

    try:
        message[key] = value
        return
    except Exception:
        pass

    setattr(message, key, value)


def _synthesized_tool_call_id(block_index: int, call_index: int) -> str:
    return f"jobscout_tool_call_{block_index}_{call_index}"


def _extract_file_hint(content_block: dict[str, Any]) -> str:
    file_obj = content_block.get("file")
    if isinstance(file_obj, dict):
        for key in ("file_id", "format", "filename"):
            value = file_obj.get(key)
            if value:
                return str(value)

    document_obj = content_block.get("document")
    if isinstance(document_obj, dict):
        for key in ("url", "filename", "mime_type"):
            value = document_obj.get(key)
            if value:
                return str(value)

    image_obj = content_block.get("image_url")
    if isinstance(image_obj, dict):
        value = image_obj.get("url")
        if value:
            return str(value)

    for key in ("mime_type", "name", "uri"):
        value = content_block.get(key)
        if value:
            return str(value)

    return "attachment"


def _coerce_block_to_text(content_block: Any) -> dict[str, str]:
    if isinstance(content_block, str):
        return {"type": "text", "text": content_block}

    if not isinstance(content_block, dict):
        return {"type": "text", "text": str(content_block)}

    block_type = content_block.get("type")
    if block_type == "text":
        return {"type": "text", "text": str(content_block.get("text", ""))}

    hint = _extract_file_hint(content_block)
    return {
        "type": "text",
        "text": f"[Attached {block_type or 'content'}: {hint}]",
    }


def _flatten_text_blocks(content_blocks: Iterable[dict[str, str]]) -> str:
    return "\n".join(
        block.get("text", "")
        for block in content_blocks
        if block.get("text", "").strip()
    ).strip()


def _normalize_groq_content_blocks(messages: list[Any]) -> list[Any]:
    """Rewrite unsupported multipart content blocks into provider-safe blocks."""
    supported_types = {"text", "image_url", "document"}

    for message in messages:
        content = _message_get(message, "content")
        if not isinstance(content, list):
            continue

        normalized_blocks: list[dict[str, Any]] = []
        saw_non_text_supported_block = False

        for block in content:
            if isinstance(block, dict) and block.get("type") in supported_types:
                normalized_blocks.append(block)
                if block.get("type") != "text":
                    saw_non_text_supported_block = True
                continue

            normalized_blocks.append(_coerce_block_to_text(block))

        if not saw_non_text_supported_block:
            _message_set(message, "content", _flatten_text_blocks(normalized_blocks))
        else:
            _message_set(message, "content", normalized_blocks)

    return messages


def _normalize_content_payload(content: Any) -> Any:
    if isinstance(content, str) or content is None:
        return content

    if isinstance(content, dict):
        return _coerce_block_to_text(content)

    try:
        blocks = list(content)
    except TypeError:
        return content

    normalized_blocks: list[dict[str, Any]] = []
    saw_non_text_supported_block = False
    supported_types = {"text", "image_url", "document"}

    for block in blocks:
        if isinstance(block, dict) and block.get("type") in supported_types:
            normalized_blocks.append(block)
            if block.get("type") != "text":
                saw_non_text_supported_block = True
            continue
        normalized_blocks.append(_coerce_block_to_text(block))

    if not saw_non_text_supported_block:
        return _flatten_text_blocks(normalized_blocks)
    return normalized_blocks


def _requires_content_block_normalization(model: Any) -> bool:
    return isinstance(model, str) and model.startswith(CONTENT_NORMALIZATION_MODEL_PREFIXES)


def _provider_rejects_file_parts(provider: Any) -> bool:
    return isinstance(provider, str) and provider in FILE_PART_UNSUPPORTED_PROVIDERS


def _repair_missing_tool_call_ids(messages: list[Any]) -> list[Any]:
    """Fill missing tool_call_id values from the preceding assistant tool calls.

    Some ADK + LiteLLM provider combinations can emit `role="tool"` messages
    without a `tool_call_id`, even though the immediately preceding assistant
    message contains the matching tool call metadata. Providers like Groq reject
    those histories.
    """
    pending_calls: list[dict[str, Any]] = []
    assistant_block_index = 0

    for message in messages:
        role = _message_get(message, "role")

        if role == "assistant":
            pending_calls = []
            for call_index, tool_call in enumerate(_message_get(message, "tool_calls", []) or []):
                tool_call_id = _message_get(tool_call, "id")
                if not tool_call_id:
                    tool_call_id = _synthesized_tool_call_id(assistant_block_index, call_index)
                    _message_set(tool_call, "id", tool_call_id)

                if not tool_call_id:
                    continue
                function_meta = _message_get(tool_call, "function", {}) or {}
                pending_calls.append({
                    "id": str(tool_call_id),
                    "name": _message_get(function_meta, "name"),
                    "used": False,
                })
            assistant_block_index += 1
            continue

        if role != "tool":
            pending_calls = []
            continue

        existing_tool_call_id = _message_get(message, "tool_call_id")
        if existing_tool_call_id:
            for pending_call in pending_calls:
                if pending_call["id"] == str(existing_tool_call_id):
                    pending_call["used"] = True
                    break
            continue

        if not pending_calls:
            continue

        tool_name = _message_get(message, "name")
        selected_call = None

        if tool_name:
            for pending_call in pending_calls:
                if not pending_call["used"] and pending_call["name"] == tool_name:
                    selected_call = pending_call
                    break

        if selected_call is None:
            for pending_call in pending_calls:
                if not pending_call["used"]:
                    selected_call = pending_call
                    break

        if selected_call is None:
            continue

        _message_set(message, "tool_call_id", selected_call["id"])
        selected_call["used"] = True
        logger.debug("Repaired missing tool_call_id with %s", selected_call["id"])

    return messages


def patch_litellm_tool_call_id_repair() -> None:
    """Patch ADK's LiteLLM bridge to repair malformed tool result messages."""
    from google.adk.models import lite_llm as lite_llm_module

    if not getattr(lite_llm_module, "_job_scout_content_patch", False):
        original_get_content = lite_llm_module._get_content

        async def patched_get_content(parts, *, provider="", model=""):
            content = await original_get_content(parts, provider=provider, model=model)
            if _provider_rejects_file_parts(provider):
                return _normalize_content_payload(content)
            return content

        lite_llm_module._get_content = patched_get_content
        lite_llm_module._job_scout_content_patch = True

    if not getattr(lite_llm_module, "_job_scout_tool_call_id_patch", False):
        original_get_completion_inputs = lite_llm_module._get_completion_inputs

        async def patched_get_completion_inputs(llm_request, model):
            messages, tools, response_format, generation_params = (
                await original_get_completion_inputs(llm_request, model)
            )
            _repair_missing_tool_call_ids(messages)
            if _requires_content_block_normalization(model):
                _normalize_groq_content_blocks(messages)
            return messages, tools, response_format, generation_params

        lite_llm_module._get_completion_inputs = patched_get_completion_inputs
        lite_llm_module._job_scout_tool_call_id_patch = True

    if not getattr(lite_llm_module, "_job_scout_reasoning_patch", False):
        original_extract_reasoning_value = lite_llm_module._extract_reasoning_value

        def patched_extract_reasoning_value(message):
            if reasoning_parts_enabled():
                return original_extract_reasoning_value(message)
            return None

        lite_llm_module._extract_reasoning_value = patched_extract_reasoning_value
        lite_llm_module._job_scout_reasoning_patch = True
