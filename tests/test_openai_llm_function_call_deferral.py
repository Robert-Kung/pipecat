#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for function-call deferral in OpenAI LLM services.

Function calls that arrive alongside generated text are deferred until TTS
finishes and released only by a BotStoppedSpeakingFrame. Content with nothing
speakable (whitespace, bare punctuation, markup) never reaches the synthesizer,
so it must not arm the deferral, or the call would wait forever and leave the
conversation silent.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.services.openai.llm import OpenAILLMService


class _FakeStream:
    """Stands in for the provider's chat completion stream."""

    def __init__(self, chunks):
        self._chunks = list(chunks)

    def __aiter__(self):
        return self._iterate()

    async def _iterate(self):
        for chunk in self._chunks:
            yield chunk

    async def close(self):
        pass


def _content_chunk(text: str):
    return SimpleNamespace(
        usage=None,
        model=None,
        choices=[SimpleNamespace(delta=SimpleNamespace(tool_calls=None, content=text))],
    )


def _tool_call_chunk(name: str, tool_call_id: str = "call-1"):
    tool_call = SimpleNamespace(
        index=0,
        id=tool_call_id,
        function=SimpleNamespace(name=name, arguments="{}"),
    )
    return SimpleNamespace(
        usage=None,
        model=None,
        choices=[SimpleNamespace(delta=SimpleNamespace(tool_calls=[tool_call], content=None))],
    )


def _streaming_service(chunks) -> OpenAILLMService:
    with patch.object(OpenAILLMService, "create_client"):
        service = OpenAILLMService(api_key="test-key")
    service.get_chat_completions = AsyncMock(return_value=_FakeStream(chunks))
    service.start_ttfb_metrics = AsyncMock()
    service.stop_ttfb_metrics = AsyncMock()
    service._push_llm_text = AsyncMock()
    service.push_frame = AsyncMock()
    service.run_function_calls = AsyncMock()
    return service


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "content", ["\n", "...", "<break/>", '<function=transfer_call>{"reason":"x"}</function>']
)
async def test_unspeakable_content_does_not_defer_function_call(content):
    service = _streaming_service([_content_chunk(content), _tool_call_chunk("transfer_call")])

    await service._process_context(LLMContext())

    service.run_function_calls.assert_awaited_once()
    assert service._pending_function_calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("content", ["好的。", "請稍候<function=x>{}</function>"])
async def test_speakable_content_defers_function_call(content):
    service = _streaming_service([_content_chunk(content), _tool_call_chunk("transfer_call")])

    await service._process_context(LLMContext())

    service.run_function_calls.assert_not_awaited()
    [function_call] = service._pending_function_calls
    assert function_call.function_name == "transfer_call"


@pytest.mark.asyncio
async def test_speakable_content_split_across_chunks_defers_function_call():
    service = _streaming_service(
        [_content_chunk(" "), _content_chunk("ok"), _tool_call_chunk("transfer_call")]
    )

    await service._process_context(LLMContext())

    service.run_function_calls.assert_not_awaited()
    assert len(service._pending_function_calls) == 1
