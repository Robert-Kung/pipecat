#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Fork tests (la-only-go-live 5.1 review S1): a stalled Gemini TTS stream is bounded.

The Gemini API client applies no timeout of its own, so a TTS request that is
accepted and then produces nothing (a proxy or upstream that stalls after the
connection is made) used to hold the utterance open indefinitely: the caller
heard silence and no error was ever reported. With ``stream_idle_timeout_secs``
set, every chunk, the first included, must arrive within that time; otherwise
the stream is closed and a non-fatal ``ErrorFrame`` is pushed, as for any other
TTS failure. The default (``None``) keeps upstream behavior.

All responses are test doubles; network connections are refused.
"""

import asyncio
import socket
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from pipecat.frames.frames import (
    ErrorFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    TextFrame,
    TTSAudioRawFrame,
    TTSSpeakFrame,
)
from pipecat.services.google.tts import GeminiTTSService, GeminiTTSStreamTimeoutError
from pipecat.tests.utils import SleepFrame, run_test

# Fixed fake key (design D8 scan rule: the 1.1/1.8 unit-test value).
FAKE_KEY = "AIza" + "0" * 35

# Short enough to keep tests quick, long enough to outlast scheduling jitter.
IDLE_TIMEOUT = 0.2

# How long a "stalled" stream goes quiet: far beyond the timeout, but finite, so
# that without the timeout the stream ends on its own and the test fails on its
# assertions instead of hanging the suite.
STALL = 3.0

# Generous upper bound for one run_test.
RUN_TIMEOUT = 10.0

SPOKEN_TEXT = "您好，這裡是客服中心。"


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def refuse(*args, **kwargs):
        raise RuntimeError("network access is not allowed in this test")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket.socket, "connect_ex", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)


# One second of 24 kHz 16-bit audio: larger than the service's frame size, so
# each chunk is pushed downstream as soon as it arrives.
ONE_SECOND = b"\x00\x01" * 24000


def audio_chunk(data=ONE_SECOND, finish_reason=None):
    part = SimpleNamespace(inline_data=SimpleNamespace(data=data), text=None)
    return SimpleNamespace(
        candidates=[
            SimpleNamespace(content=SimpleNamespace(parts=[part]), finish_reason=finish_reason)
        ],
        prompt_feedback=None,
    )


async def run_tts(mock_client_class, generator, frames=None, **service_kwargs):
    """Speak one sentence against a canned stream.

    ``frames`` replaces the default single ``TTSSpeakFrame``.

    Returns:
        The audio frames pushed downstream and the error frames pushed upstream.
    """
    mock_client = MagicMock()
    mock_client_class.return_value = mock_client

    async def stream(*args, **kwargs):
        return generator()

    mock_client.aio.models.generate_content_stream = stream
    tts = GeminiTTSService(
        api_key=FAKE_KEY,
        settings=GeminiTTSService.Settings(model="gemini-2.5-flash-preview-tts", voice="Kore"),
        **service_kwargs,
    )
    down, up = await asyncio.wait_for(
        run_test(tts, frames_to_send=frames or [TTSSpeakFrame(text=SPOKEN_TEXT)]),
        timeout=RUN_TIMEOUT,
    )
    audio = [f for f in down if isinstance(f, TTSAudioRawFrame)]
    errors = [f for f in up if isinstance(f, ErrorFrame)]
    return audio, errors


@pytest.mark.asyncio
@pytest.mark.parametrize("chunks_before_stall", [0, 2], ids=["before-audio", "mid-utterance"])
@patch("google.genai.Client")
async def test_stalled_stream_times_out_with_a_non_fatal_error(
    mock_client_class, chunks_before_stall
):
    """A stream that stops producing, before or after its first chunk, ends in a
    non-fatal error and is closed; the audio that did arrive is still played."""
    stalled_for = []

    async def generator():
        started = time.monotonic()
        try:
            for _ in range(chunks_before_stall):
                yield audio_chunk()
            await asyncio.sleep(STALL)
        finally:
            stalled_for.append(time.monotonic() - started)

    audio, errors = await run_tts(
        mock_client_class, generator, stream_idle_timeout_secs=IDLE_TIMEOUT
    )

    assert len(errors) == 1
    assert errors[0].fatal is False
    assert isinstance(errors[0].exception, GeminiTTSStreamTimeoutError)
    assert isinstance(errors[0].exception, TimeoutError)
    assert errors[0].exception.timeout_secs == IDLE_TIMEOUT
    assert bool(audio) == (chunks_before_stall > 0)
    # The stream was closed when the timeout fired, not when it ended on its own.
    [stalled] = stalled_for
    assert stalled < STALL / 2


@pytest.mark.asyncio
@pytest.mark.parametrize("chunks_before_stall", [0, 1], ids=["before-audio", "mid-utterance"])
@pytest.mark.parametrize("path", ["speak", "llm-turn"])
@patch("google.genai.Client")
async def test_stall_error_is_delivered_when_it_matches_the_stop_frame_timeout(
    mock_client_class, path, chunks_before_stall
):
    """A stall timeout equal to ``stop_frame_timeout_s`` still reaches upstream.

    The audio context's idle timer (``stop_frame_timeout_s``) and the stall
    timer restart on the same chunk, so with equal values both expire in the
    same instant. An ErrorFrame queued in the audio context then was dropped
    with the context as it closed (5.1 fix review R1): the utterance was
    skipped in silence and nothing upstream heard of it. The stall error now
    goes upstream directly.
    """

    async def generator():
        for _ in range(chunks_before_stall):
            yield audio_chunk()
        await asyncio.sleep(STALL)

    keep_running = SleepFrame(IDLE_TIMEOUT * 4)
    if path == "speak":
        frames = [TTSSpeakFrame(text=SPOKEN_TEXT), keep_running]
    else:
        frames = [
            LLMFullResponseStartFrame(),
            TextFrame(text=SPOKEN_TEXT),
            LLMFullResponseEndFrame(),
            keep_running,
        ]

    _, errors = await run_tts(
        mock_client_class,
        generator,
        frames=frames,
        stream_idle_timeout_secs=IDLE_TIMEOUT,
        stop_frame_timeout_s=IDLE_TIMEOUT,
    )

    assert [type(e.exception) for e in errors] == [GeminiTTSStreamTimeoutError]
    assert errors[0].fatal is False


@pytest.mark.asyncio
@patch("google.genai.Client")
async def test_stream_own_timeout_is_not_reported_as_a_stall(mock_client_class):
    """A TimeoutError raised by the stream itself (e.g. an HTTP read timeout)
    is reported as itself, not as a stall of ``stream_idle_timeout_secs``."""

    class ReadTimeout(TimeoutError):
        pass

    async def generator():
        yield audio_chunk()
        raise ReadTimeout("socket read timed out")

    _, errors = await run_tts(mock_client_class, generator, stream_idle_timeout_secs=STALL)

    assert len(errors) == 1
    assert type(errors[0].exception) is ReadTimeout
    assert errors[0].fatal is False


@pytest.mark.asyncio
@patch("google.genai.Client")
async def test_timeout_error_does_not_carry_the_spoken_text(mock_client_class):
    """Only the text's length is reported, never its content (call content may
    be personal data)."""

    async def generator():
        await asyncio.sleep(STALL)
        yield audio_chunk()

    _, [error] = await run_tts(mock_client_class, generator, stream_idle_timeout_secs=IDLE_TIMEOUT)

    assert SPOKEN_TEXT not in error.error
    assert SPOKEN_TEXT not in str(error.exception)
    assert error.exception.text_length == len(SPOKEN_TEXT)


@pytest.mark.asyncio
@patch("google.genai.Client")
async def test_slow_but_healthy_stream_is_not_cut_short(mock_client_class):
    """The timeout bounds each gap, not the utterance: a stream whose total
    time outlasts it several times over plays out without an error."""

    async def generator():
        for _ in range(5):
            await asyncio.sleep(IDLE_TIMEOUT / 2)
            yield audio_chunk()
        yield audio_chunk(data=b"", finish_reason="STOP")

    audio, errors = await run_tts(
        mock_client_class, generator, stream_idle_timeout_secs=IDLE_TIMEOUT
    )

    assert errors == []
    assert audio


@patch("google.genai.Client")
def test_stream_is_not_bounded_by_default(mock_client_class):
    """Upstream behavior is kept unless a caller opts in."""
    tts = GeminiTTSService(
        api_key=FAKE_KEY,
        settings=GeminiTTSService.Settings(model="gemini-2.5-flash-preview-tts", voice="Kore"),
    )

    assert tts._stream_idle_timeout_secs is None
