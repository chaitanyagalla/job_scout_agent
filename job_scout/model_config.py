"""Helpers for choosing the configured LLM provider."""
from __future__ import annotations

import os
from collections.abc import Mapping

DEFAULT_GEMINI_MODEL = "gemini-2.0-flash"
DEFAULT_GROQ_MODEL = "groq/llama-3.3-70b-versatile"
DEFAULT_NVIDIA_MODEL = "nvidia_nim/nvidia/llama-3.3-nemotron-super-49b-v1.5"
DEFAULT_MAX_OUTPUT_TOKENS = 2048
DEFAULT_RESUME_PARSER_MAX_OUTPUT_TOKENS = 1024
LITELLM_PROVIDER_PREFIXES = ("groq/", "anthropic/", "openai/", "nvidia_nim/")
REASONING_MODEL_MARKERS = ("reasoning",)


def _env_flag_is_true(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


def reasoning_models_allowed(env: Mapping[str, str] | None = None) -> bool:
    env = env if env is not None else os.environ
    return _env_flag_is_true(env.get("JOB_SCOUT_ALLOW_REASONING_MODEL"))


def is_reasoning_model(model_name: str) -> bool:
    normalized = (model_name or "").strip().lower()
    return bool(normalized) and any(marker in normalized for marker in REASONING_MODEL_MARKERS)


def _safe_non_reasoning_model(model_name: str, env: Mapping[str, str]) -> str:
    normalized = (model_name or "").strip()
    fallback_gemini_model = (
        env.get("GEMINI_FALLBACK_MODEL")
        or DEFAULT_GEMINI_MODEL
    ).strip()

    if normalized.startswith("nvidia_nim/"):
        return DEFAULT_NVIDIA_MODEL
    if normalized.startswith("groq/"):
        return DEFAULT_GROQ_MODEL
    if normalized.startswith(("openai/", "anthropic/")):
        return fallback_gemini_model
    return fallback_gemini_model


def _sanitize_model_name(model_name: str, env: Mapping[str, str]) -> str:
    if not is_reasoning_model(model_name):
        return model_name
    if reasoning_models_allowed(env):
        return model_name
    return _safe_non_reasoning_model(model_name, env)


def resolve_model_name(env: Mapping[str, str] | None = None) -> str:
    """Resolve the model name from environment variables.

    Explicit model configuration always wins. Provider-specific fallback logic
    is only used when no model override is present.
    """
    env = env if env is not None else os.environ

    configured_model = (
        env.get("JOB_SCOUT_MODEL")
        or env.get("GEMINI_MODEL")
        or ""
    ).strip()
    if configured_model:
        return _sanitize_model_name(configured_model, env)

    fallback_gemini_model = (
        env.get("GEMINI_FALLBACK_MODEL")
        or DEFAULT_GEMINI_MODEL
    ).strip()
    has_google_api_key = bool((env.get("GOOGLE_API_KEY") or "").strip())
    has_groq_api_key = bool((env.get("GROQ_API_KEY") or "").strip())
    has_nvidia_api_key = bool((env.get("NVIDIA_NIM_API_KEY") or "").strip())

    if has_google_api_key:
        return _sanitize_model_name(fallback_gemini_model, env)
    if has_groq_api_key:
        return _sanitize_model_name(DEFAULT_GROQ_MODEL, env)
    if has_nvidia_api_key:
        return _sanitize_model_name(DEFAULT_NVIDIA_MODEL, env)
    return _sanitize_model_name(fallback_gemini_model, env)


def uses_litellm(model_name: str) -> bool:
    """Return whether the model should be instantiated via LiteLLM."""
    return model_name.startswith(LITELLM_PROVIDER_PREFIXES)


def _resolve_positive_int_setting(
    env: Mapping[str, str],
    setting_name: str,
    default_value: int,
) -> int:
    raw_value = (env.get(setting_name) or "").strip()
    if not raw_value:
        return default_value

    try:
        value = int(raw_value)
    except ValueError:
        return default_value

    return value if value > 0 else default_value


def resolve_max_output_tokens(env: Mapping[str, str] | None = None) -> int:
    env = env if env is not None else os.environ
    return _resolve_positive_int_setting(
        env,
        "JOB_SCOUT_MAX_OUTPUT_TOKENS",
        DEFAULT_MAX_OUTPUT_TOKENS,
    )


def resolve_resume_parser_max_output_tokens(env: Mapping[str, str] | None = None) -> int:
    env = env if env is not None else os.environ
    return _resolve_positive_int_setting(
        env,
        "JOB_SCOUT_RESUME_PARSER_MAX_OUTPUT_TOKENS",
        DEFAULT_RESUME_PARSER_MAX_OUTPUT_TOKENS,
    )


def resume_gemini_attachment_fallback_enabled(env: Mapping[str, str] | None = None) -> bool:
    """Return whether resume extraction may use native Gemini file parsing."""
    env = env if env is not None else os.environ
    return _env_flag_is_true(env.get("JOB_SCOUT_ENABLE_GEMINI_RESUME_FALLBACK"))


def resolve_resume_parser_model(env: Mapping[str, str] | None = None) -> str:
    """Choose the preferred model for structuring extracted resume text."""
    env = env if env is not None else os.environ

    configured_parser_model = (
        env.get("JOB_SCOUT_RESUME_PARSER_MODEL")
        or ""
    ).strip()
    if configured_parser_model:
        return _sanitize_model_name(configured_parser_model, env)

    fallback_gemini_model = (
        env.get("GEMINI_FALLBACK_MODEL")
        or DEFAULT_GEMINI_MODEL
    ).strip()
    configured_model = (
        env.get("JOB_SCOUT_MODEL")
        or env.get("GEMINI_MODEL")
        or ""
    ).strip()
    has_google_api_key = bool((env.get("GOOGLE_API_KEY") or "").strip())
    has_nvidia_api_key = bool((env.get("NVIDIA_NIM_API_KEY") or "").strip())

    if configured_model:
        sanitized_model = _sanitize_model_name(configured_model, env)
        if sanitized_model.startswith("nvidia_nim/"):
            return sanitized_model
        if not uses_litellm(sanitized_model):
            return sanitized_model

    if has_nvidia_api_key:
        return _sanitize_model_name(DEFAULT_NVIDIA_MODEL, env)
    if has_google_api_key:
        return _sanitize_model_name(fallback_gemini_model, env)

    return resolve_model_name(env)


def resolve_resume_attachment_model(env: Mapping[str, str] | None = None) -> str:
    """Choose a native Gemini model for attachment-based resume parsing."""
    env = env if env is not None else os.environ

    fallback_gemini_model = (
        env.get("GEMINI_FALLBACK_MODEL")
        or DEFAULT_GEMINI_MODEL
    ).strip()
    configured_model = (
        env.get("JOB_SCOUT_MODEL")
        or env.get("GEMINI_MODEL")
        or ""
    ).strip()
    has_google_api_key = bool((env.get("GOOGLE_API_KEY") or "").strip())

    if has_google_api_key:
        if configured_model and not uses_litellm(configured_model):
            return _sanitize_model_name(configured_model, env)
        return _sanitize_model_name(fallback_gemini_model, env)

    return resolve_resume_parser_model(env)
