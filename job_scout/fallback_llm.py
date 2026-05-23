from __future__ import annotations

import copy
import logging
from typing import Any
from typing import AsyncGenerator

from pydantic import ConfigDict

from google.adk.models.base_llm import BaseLlm

logger = logging.getLogger(__name__)


def _clone_llm_request(llm_request: Any) -> Any:
    copier = getattr(llm_request, "model_copy", None)
    if callable(copier):
        return copier(deep=True)
    return copy.deepcopy(llm_request)


class FallbackLlm(BaseLlm):
    """Retry a failed primary model with a fallback provider.

    This keeps the chat usable when the preferred LiteLLM-backed provider fails
    before returning any content for the current turn.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    primary: BaseLlm
    fallback: BaseLlm

    async def generate_content_async(
        self,
        llm_request,
        stream: bool = False,
    ) -> AsyncGenerator[Any, None]:
        primary_request = _clone_llm_request(llm_request)
        yielded_primary_response = False

        try:
            async for response in self.primary.generate_content_async(primary_request, stream=stream):
                yielded_primary_response = True
                yield response
            return
        except Exception as exc:
            if yielded_primary_response:
                raise

            logger.warning(
                "Primary model %s failed before yielding a response; retrying with fallback model %s.",
                self.primary.model,
                self.fallback.model,
                exc_info=exc,
            )

        fallback_request = _clone_llm_request(llm_request)
        async for response in self.fallback.generate_content_async(fallback_request, stream=stream):
            yield response
