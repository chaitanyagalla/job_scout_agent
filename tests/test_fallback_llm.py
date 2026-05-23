from __future__ import annotations

import asyncio

from pydantic import ConfigDict

from google.adk.models.base_llm import BaseLlm

from job_scout.fallback_llm import FallbackLlm


class DummyRequest:
    def __init__(self, marker: str):
        self.marker = marker

    def model_copy(self, deep: bool = False):
        return DummyRequest(self.marker)


class StubLlm(BaseLlm):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    outputs: list[str] = []
    error: Exception | None = None

    async def generate_content_async(self, llm_request, stream: bool = False):
        for output in self.outputs:
            yield output
        if self.error is not None:
            raise self.error


def test_fallback_llm_retries_with_fallback_when_primary_fails_before_output():
    async def run():
        model = FallbackLlm(
            model="primary",
            primary=StubLlm(model="primary", error=RuntimeError("boom")),
            fallback=StubLlm(model="fallback", outputs=["fallback-response"]),
        )
        results = []
        async for response in model.generate_content_async(DummyRequest("request")):
            results.append(response)
        return results

    assert asyncio.run(run()) == ["fallback-response"]


def test_fallback_llm_does_not_restart_after_partial_primary_output():
    async def run():
        model = FallbackLlm(
            model="primary",
            primary=StubLlm(
                model="primary",
                outputs=["partial-response"],
                error=RuntimeError("mid-stream failure"),
            ),
            fallback=StubLlm(model="fallback", outputs=["fallback-response"]),
        )
        results = []
        try:
            async for response in model.generate_content_async(DummyRequest("request")):
                results.append(response)
        except RuntimeError as exc:
            return results, str(exc)
        return results, None

    results, error_message = asyncio.run(run())

    assert results == ["partial-response"]
    assert error_message == "mid-stream failure"
