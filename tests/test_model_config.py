from job_scout.model_config import DEFAULT_GEMINI_MODEL
from job_scout.model_config import DEFAULT_GROQ_MODEL
from job_scout.model_config import DEFAULT_MAX_OUTPUT_TOKENS
from job_scout.model_config import DEFAULT_NVIDIA_MODEL
from job_scout.model_config import DEFAULT_RESUME_PARSER_MAX_OUTPUT_TOKENS
from job_scout.model_config import is_reasoning_model
from job_scout.model_config import resolve_max_output_tokens
from job_scout.model_config import resolve_model_name
from job_scout.model_config import resolve_resume_attachment_model
from job_scout.model_config import resolve_resume_parser_max_output_tokens
from job_scout.model_config import resolve_resume_parser_model
from job_scout.model_config import reasoning_models_allowed
from job_scout.model_config import resume_gemini_attachment_fallback_enabled
from job_scout.model_config import uses_litellm


def test_explicit_groq_model_is_honored_even_with_google_api_key():
    env = {
        "GOOGLE_API_KEY": "google-key",
        "GROQ_API_KEY": "groq-key",
        "GEMINI_MODEL": "groq/llama-3.3-70b-versatile",
    }

    assert resolve_model_name(env) == "groq/llama-3.3-70b-versatile"
    assert uses_litellm(resolve_model_name(env)) is True


def test_google_fallback_is_used_only_when_no_explicit_model_is_set():
    env = {
        "GOOGLE_API_KEY": "google-key",
        "GEMINI_FALLBACK_MODEL": "gemini-2.5-flash",
    }

    assert resolve_model_name(env) == "gemini-2.5-flash"
    assert uses_litellm(resolve_model_name(env)) is False


def test_explicit_nvidia_model_is_honored_even_with_google_api_key():
    env = {
        "GOOGLE_API_KEY": "google-key",
        "NVIDIA_NIM_API_KEY": "nvidia-key",
        "GEMINI_MODEL": "nvidia_nim/nvidia/llama-3.3-nemotron-super-49b-v1.5",
    }

    assert resolve_model_name(env) == "nvidia_nim/nvidia/llama-3.3-nemotron-super-49b-v1.5"
    assert uses_litellm(resolve_model_name(env)) is True


def test_explicit_reasoning_model_is_downgraded_to_safe_chat_model_by_default():
    env = {
        "GOOGLE_API_KEY": "google-key",
        "NVIDIA_NIM_API_KEY": "nvidia-key",
        "GEMINI_MODEL": "nvidia_nim/nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
    }

    assert is_reasoning_model(env["GEMINI_MODEL"]) is True
    assert reasoning_models_allowed(env) is False
    assert resolve_model_name(env) == DEFAULT_NVIDIA_MODEL


def test_explicit_reasoning_model_can_be_allowed_with_opt_in_flag():
    env = {
        "GOOGLE_API_KEY": "google-key",
        "NVIDIA_NIM_API_KEY": "nvidia-key",
        "GEMINI_MODEL": "nvidia_nim/nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
        "JOB_SCOUT_ALLOW_REASONING_MODEL": "true",
    }

    assert reasoning_models_allowed(env) is True
    assert resolve_model_name(env) == env["GEMINI_MODEL"]


def test_groq_becomes_default_when_only_groq_key_is_available():
    env = {
        "GROQ_API_KEY": "groq-key",
    }

    assert resolve_model_name(env) == DEFAULT_GROQ_MODEL


def test_nvidia_becomes_default_when_only_nvidia_key_is_available():
    env = {
        "NVIDIA_NIM_API_KEY": "nvidia-key",
    }

    assert resolve_model_name(env) == DEFAULT_NVIDIA_MODEL


def test_gemini_default_is_used_when_no_provider_keys_are_present():
    assert resolve_model_name({}) == DEFAULT_GEMINI_MODEL


def test_resume_parser_model_prefers_nvidia_when_available_for_text_parsing():
    env = {
        "GOOGLE_API_KEY": "google-key",
        "NVIDIA_NIM_API_KEY": "nvidia-key",
        "GEMINI_MODEL": "groq/llama-3.3-70b-versatile",
        "GEMINI_FALLBACK_MODEL": "gemini-2.5-flash",
    }

    assert resolve_resume_parser_model(env) == DEFAULT_NVIDIA_MODEL


def test_resume_attachment_model_still_prefers_native_gemini_for_file_parts():
    env = {
        "GOOGLE_API_KEY": "google-key",
        "NVIDIA_NIM_API_KEY": "nvidia-key",
        "GEMINI_MODEL": "nvidia_nim/nvidia/llama-3.3-nemotron-super-49b-v1.5",
        "GEMINI_FALLBACK_MODEL": "gemini-2.5-flash",
    }

    assert resolve_resume_attachment_model(env) == "gemini-2.5-flash"


def test_resume_gemini_attachment_fallback_is_opt_in():
    assert resume_gemini_attachment_fallback_enabled({}) is False
    assert resume_gemini_attachment_fallback_enabled({
        "JOB_SCOUT_ENABLE_GEMINI_RESUME_FALLBACK": "true"
    }) is True


def test_resume_parser_model_honors_explicit_parser_override():
    env = {
        "GOOGLE_API_KEY": "google-key",
        "NVIDIA_NIM_API_KEY": "nvidia-key",
        "JOB_SCOUT_RESUME_PARSER_MODEL": "groq/llama-3.3-70b-versatile",
    }

    assert resolve_resume_parser_model(env) == "groq/llama-3.3-70b-versatile"


def test_max_output_tokens_can_be_configured():
    assert resolve_max_output_tokens({"JOB_SCOUT_MAX_OUTPUT_TOKENS": "1536"}) == 1536
    assert resolve_max_output_tokens({"JOB_SCOUT_MAX_OUTPUT_TOKENS": "0"}) == DEFAULT_MAX_OUTPUT_TOKENS
    assert resolve_max_output_tokens({"JOB_SCOUT_MAX_OUTPUT_TOKENS": "bad"}) == DEFAULT_MAX_OUTPUT_TOKENS


def test_resume_parser_max_output_tokens_can_be_configured():
    assert resolve_resume_parser_max_output_tokens({
        "JOB_SCOUT_RESUME_PARSER_MAX_OUTPUT_TOKENS": "768"
    }) == 768
    assert resolve_resume_parser_max_output_tokens({}) == DEFAULT_RESUME_PARSER_MAX_OUTPUT_TOKENS
