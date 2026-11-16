#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Unit tests for ``fallback_user_message`` on OpenAI-compatible LLMs.

Some OpenAI-compatible endpoints (Meta Muse) reject a request whose messages
hold no ``user`` or ``tool`` entry, e.g. a voice agent's opening turn that
carries only the system prompt.
"""

from unittest.mock import AsyncMock, patch

import pytest

from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.services.openai.llm import OpenAILLMService

TRIGGER = "(call connected, greet the caller)"


def _service(**kwargs) -> OpenAILLMService:
    with patch.object(OpenAILLMService, "create_client"):
        service = OpenAILLMService(settings=OpenAILLMService.Settings(model="muse-x"), **kwargs)
    service._client = AsyncMock()
    return service


async def _sent_messages(service: OpenAILLMService, context: LLMContext) -> list:
    await service.get_chat_completions(context)
    return service._client.chat.completions.create.call_args.kwargs["messages"]


@pytest.mark.asyncio
async def test_system_only_turn_gets_trigger_appended_without_touching_context():
    service = _service(fallback_user_message=TRIGGER)
    context = LLMContext(messages=[{"role": "system", "content": "You are a receptionist."}])

    sent = await _sent_messages(service, context)

    assert sent == [
        {"role": "system", "content": "You are a receptionist."},
        {"role": "user", "content": TRIGGER},
    ]
    assert context.get_messages() == [{"role": "system", "content": "You are a receptionist."}]


@pytest.mark.asyncio
async def test_turn_with_user_message_is_unchanged():
    service = _service(fallback_user_message=TRIGGER)
    messages = [
        {"role": "system", "content": "You are a receptionist."},
        {"role": "assistant", "content": "Hello."},
        {"role": "user", "content": "I need a human."},
    ]

    sent = await _sent_messages(service, LLMContext(messages=list(messages)))

    assert sent == messages


@pytest.mark.asyncio
async def test_turn_with_tool_result_is_unchanged():
    service = _service(fallback_user_message=TRIGGER)
    messages = [
        {"role": "system", "content": "You are a receptionist."},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}}
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "ok"},
    ]

    sent = await _sent_messages(service, LLMContext(messages=list(messages)))

    assert [m["role"] for m in sent] == ["system", "assistant", "tool"]


@pytest.mark.asyncio
async def test_default_sends_system_only_turn_as_is():
    service = _service()
    context = LLMContext(messages=[{"role": "system", "content": "You are a receptionist."}])

    sent = await _sent_messages(service, context)

    assert sent == [{"role": "system", "content": "You are a receptionist."}]
