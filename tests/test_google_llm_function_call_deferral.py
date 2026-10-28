#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for function-call deferral in GoogleLLMService.

Function calls that arrive alongside generated text are deferred until TTS
finishes and released only by a BotStoppedSpeakingFrame. Content with nothing
speakable (whitespace, bare punctuation, markup) never reaches the synthesizer,
so it must not arm the deferral, or the call would wait forever and leave the
conversation silent.
"""

from unittest.mock import AsyncMock, patch

import pytest
from google.genai.types import (
    Candidate,
    Content,
    FinishReason,
    FunctionCall,
    GenerateContentResponse,
    Part,
)

from pipecat.frames.frames import BotStoppedSpeakingFrame
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.google.llm import GoogleLLMService


def _text_chunk(text: str) -> GenerateContentResponse:
    return GenerateContentResponse(
        candidates=[Candidate(content=Content(role="model", parts=[Part(text=text)]))]
    )


def _tool_call_chunk(name: str) -> GenerateContentResponse:
    return GenerateContentResponse(
        candidates=[
            Candidate(
                content=Content(
                    role="model",
                    parts=[Part(function_call=FunctionCall(id="call-1", name=name, args={}))],
                ),
                finish_reason=FinishReason.STOP,
            )
        ]
    )


async def _process(chunks) -> GoogleLLMService:
    """Run one context through the service against a canned stream."""
    service = GoogleLLMService(api_key="test-key")
    service.run_function_calls = AsyncMock()

    async def fake_stream(context):
        async def generator():
            for chunk in chunks:
                yield chunk

        return generator()

    with (
        patch.object(service, "push_frame", AsyncMock()),
        patch.object(service, "_stream_content", fake_stream),
    ):
        await service._process_context(LLMContext())
    return service


@pytest.mark.asyncio
@pytest.mark.parametrize("content", ["\n", " ", "...", "。", "<break/>"])
async def test_unspeakable_content_does_not_defer_function_call(content):
    service = await _process([_text_chunk(content), _tool_call_chunk("transfer_call")])

    service.run_function_calls.assert_awaited_once()
    [function_call] = service.run_function_calls.await_args.args[0]
    assert function_call.function_name == "transfer_call"
    assert service._pending_function_calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("content", ["好的，幫您轉接。", "OK", "5"])
async def test_speakable_content_defers_function_call(content):
    service = await _process([_text_chunk(content), _tool_call_chunk("transfer_call")])

    service.run_function_calls.assert_not_awaited()
    [function_call] = service._pending_function_calls
    assert function_call.function_name == "transfer_call"


@pytest.mark.asyncio
async def test_speakable_content_split_across_chunks_defers_function_call():
    service = await _process(
        [_text_chunk(" "), _text_chunk("好"), _tool_call_chunk("transfer_call")]
    )

    service.run_function_calls.assert_not_awaited()
    assert len(service._pending_function_calls) == 1


@pytest.mark.asyncio
async def test_deferred_function_call_runs_after_tts():
    service = await _process([_text_chunk("好的。"), _tool_call_chunk("transfer_call")])
    pending = list(service._pending_function_calls)

    with patch.object(service, "push_frame", AsyncMock()):
        await service.process_frame(BotStoppedSpeakingFrame(), FrameDirection.DOWNSTREAM)

    service.run_function_calls.assert_awaited_once_with(pending)
    assert service._pending_function_calls == []
