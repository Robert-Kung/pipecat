#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Fork tests (gemini-key-voice 1.1 / 1.8): Gemini API key path failure signals.

Covers what the application's never-silent fallback (C4) relies on when the
Gemini API key is invalid or the project quota is exhausted:

- Gemini STT reports every failed connection attempt -- including the first
  one, made when the pipeline starts -- and every stream error as one
  non-fatal ``ErrorFrame`` carrying the original exception, keeps retrying at a
  fixed interval, and never reconnects once stopped.
- Gemini TTS (GenAI backend) reports an invalid key / blocked API / 429 as a
  non-fatal ``ErrorFrame`` carrying the original exception.
- An empty ``api_key`` never falls back to environment keys or ADC (SEC-8).
- google-genai sends the key in the ``x-goog-api-key`` header, never in the URL
  (design D8), for the STT (Live websocket), TTS and LLM (REST) paths.

All failures are injected with test doubles; nothing here touches the network.
"""

import asyncio
import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from google.genai import errors as genai_errors

import pipecat.services.google.gemini_live.stt as gemini_stt
from pipecat.frames.frames import (
    ErrorFrame,
    TranscriptionFrame,
    TTSSpeakFrame,
)
from pipecat.services.google.gemini_live.stt import GeminiSTTService
from pipecat.services.google.tts import GeminiTTSService
from pipecat.tests.utils import SleepFrame, run_test

# Fixed fake key (design D8 scan rule: the 1.1/1.8 unit-test value).
FAKE_KEY = "AIza" + "0" * 35


def invalid_key_close():
    # What google-genai raises when the Live websocket is closed with 1008.
    return genai_errors.APIError(1008, "API key not valid. Please pass a valid API key.", None)


def quota_close():
    return genai_errors.APIError(1011, "Resource has been exhausted (e.g. check quota).", None)


def rest_error(code: int, status: str, message: str):
    return genai_errors.ClientError(
        code, {"error": {"code": code, "status": status, "message": message}}, None
    )


class FakeLiveClient:
    """Stands in for ``google.genai.Client``; ``aio.live.connect`` is scripted.

    ``outcomes`` is consumed one entry per connection attempt: an exception is
    raised from ``__aenter__`` (a connection-phase failure); a session is
    returned as the live session. When exhausted, the last entry repeats.
    """

    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.attempts = 0
        self.client_kwargs = None
        self.aio = SimpleNamespace(live=SimpleNamespace(connect=self._connect))

    def __call__(self, **kwargs):
        self.client_kwargs = kwargs
        return self

    def _connect(self, *, model, config):
        index = min(self.attempts, len(self.outcomes) - 1)
        outcome = self.outcomes[index]
        client = self

        class Ctx:
            async def __aenter__(self):
                client.attempts += 1
                if isinstance(outcome, BaseException):
                    raise outcome
                return outcome

            async def __aexit__(self, exc_type, exc, tb):
                return None

        return Ctx()


class FakeSession:
    """A live session that yields ``messages`` and then fails with ``error``."""

    def __init__(self, messages, error):
        self._messages = list(messages)
        self._error = error

    async def send_realtime_input(self, **kwargs):
        pass

    async def receive(self):
        while self._messages:
            yield self._messages.pop(0)
        raise self._error


def final_transcript(text):
    transcription = SimpleNamespace(text=text, language_code=None)
    return SimpleNamespace(
        go_away=None,
        server_content=SimpleNamespace(
            interim_input_transcription=None, input_transcription=transcription
        ),
    )


@pytest.fixture
def fast_retry(monkeypatch):
    monkeypatch.setattr(gemini_stt, "RECONNECT_INTERVAL_SECONDS", 0.02)


def stt_errors(up_frames):
    return [f for f in up_frames if isinstance(f, ErrorFrame)]


# --- STT: connection phase ---------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error",
    [invalid_key_close(), quota_close(), rest_error(429, "RESOURCE_EXHAUSTED", "Quota exceeded")],
    ids=["invalid-key-1008", "quota-1011", "http-429"],
)
async def test_stt_connection_phase_failure_pushes_non_fatal_error_frames(fast_retry, error):
    fake = FakeLiveClient([error])
    with patch.object(gemini_stt, "Client", fake):
        stt = GeminiSTTService(api_key=FAKE_KEY)
        _, up_frames = await run_test(stt, frames_to_send=[SleepFrame(sleep=0.15)])

    errors = stt_errors(up_frames)
    # The very first attempt (made when the pipeline starts) is reported, and
    # the service keeps retrying: one ErrorFrame per attempt.
    assert len(errors) >= 2
    assert len(errors) == fake.attempts
    assert all(not f.fatal for f in errors)
    assert all(f.exception is error for f in errors)


@pytest.mark.asyncio
async def test_stt_does_not_reconnect_after_stop(fast_retry):
    fake = FakeLiveClient([invalid_key_close()])
    with patch.object(gemini_stt, "Client", fake):
        stt = GeminiSTTService(api_key=FAKE_KEY)
        await run_test(stt, frames_to_send=[SleepFrame(sleep=0.1)])

        attempts_at_stop = fake.attempts
        await asyncio.sleep(0.1)
        # Nothing restarts it: not the retry loop, a reconnect request, or a
        # settings-driven reconnect.
        await stt._connect()
        await stt._request_reconnect()
        await stt._do_reconnect()

    assert attempts_at_stop >= 1
    assert fake.attempts == attempts_at_stop
    assert stt._receive_task is None


# --- STT: mid-call stream errors --------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error", [invalid_key_close(), quota_close()], ids=["invalid-key-1008", "quota-1011"]
)
async def test_stt_stream_error_mid_call_pushes_non_fatal_error_frame(fast_retry, error):
    # The first session works and then dies (key revoked / quota exhausted mid
    # call); every reconnect attempt after that fails at connection time.
    session = FakeSession([final_transcript("hello")], error)
    fake = FakeLiveClient([session, error])
    with patch.object(gemini_stt, "Client", fake):
        stt = GeminiSTTService(api_key=FAKE_KEY)
        down_frames, up_frames = await run_test(stt, frames_to_send=[SleepFrame(sleep=0.15)])

    assert any(isinstance(f, TranscriptionFrame) and f.text == "hello" for f in down_frames)
    errors = stt_errors(up_frames)
    assert len(errors) >= 2  # the stream error, then failed reconnects
    assert all(not f.fatal for f in errors)
    assert all(f.exception is error for f in errors)
    # One ErrorFrame per failure: the stream error plus each failed reconnect.
    assert len(errors) == fake.attempts


# --- TTS ----------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error",
    [
        rest_error(400, "INVALID_ARGUMENT", "API key not valid. Please pass a valid API key."),
        rest_error(403, "PERMISSION_DENIED", "Requests to this API method are blocked."),
        rest_error(429, "RESOURCE_EXHAUSTED", "Resource has been exhausted (e.g. check quota)."),
    ],
    ids=["invalid-key-400", "api-blocked-403", "quota-429"],
)
@patch("google.genai.Client")
async def test_tts_key_and_quota_errors_push_non_fatal_error_frame(mock_client_class, error):
    mock_client = MagicMock()
    mock_client_class.return_value = mock_client

    async def failing_stream(*args, **kwargs):
        raise error

    mock_client.aio.models.generate_content_stream = failing_stream

    tts = GeminiTTSService(
        api_key=FAKE_KEY,
        settings=GeminiTTSService.Settings(model="gemini-2.5-flash-preview-tts", voice="Kore"),
    )
    _, up_frames = await run_test(tts, frames_to_send=[TTSSpeakFrame(text="Hello.")])

    errors = [f for f in up_frames if isinstance(f, ErrorFrame)]
    assert len(errors) == 1
    assert not errors[0].fatal
    assert errors[0].exception is error
    assert errors[0].exception.code == error.code


# --- SEC-8: no fallback to environment keys, Vertex AI or ADC ----------------

HOSTILE_ENV = {
    "GOOGLE_API_KEY": "env-google-key",
    "GEMINI_API_KEY": "env-gemini-key",
    "GOOGLE_GENAI_USE_VERTEXAI": "true",
    "GOOGLE_CLOUD_PROJECT": "env-project",
}


@pytest.mark.parametrize("api_key", ["", "   "])
def test_stt_empty_api_key_is_rejected(api_key):
    fake = FakeLiveClient([invalid_key_close()])
    with patch.dict(os.environ, HOSTILE_ENV), patch.object(gemini_stt, "Client", fake):
        with pytest.raises(ValueError, match="non-empty api_key"):
            GeminiSTTService(api_key=api_key)
    assert fake.client_kwargs is None


def test_stt_client_is_pinned_to_the_gemini_api():
    fake = FakeLiveClient([invalid_key_close()])
    with patch.dict(os.environ, HOSTILE_ENV), patch.object(gemini_stt, "Client", fake):
        GeminiSTTService(api_key=FAKE_KEY)
    assert fake.client_kwargs["api_key"] == FAKE_KEY
    assert fake.client_kwargs["vertexai"] is False


@pytest.mark.parametrize("kwargs", [{"api_key": ""}, {"api_key": "  "}, {"use_genai": True}])
@patch("google.genai.Client")
def test_tts_genai_backend_requires_an_explicit_key(mock_client_class, kwargs):
    with patch.dict(os.environ, HOSTILE_ENV):
        with pytest.raises(ValueError, match="non-empty api_key"):
            GeminiTTSService(**kwargs)
    mock_client_class.assert_not_called()


@patch("google.genai.Client")
def test_tts_genai_client_is_pinned_to_the_gemini_api(mock_client_class):
    with patch.dict(os.environ, HOSTILE_ENV):
        GeminiTTSService(api_key=FAKE_KEY)
    _, kwargs = mock_client_class.call_args
    assert kwargs["api_key"] == FAKE_KEY
    assert kwargs["vertexai"] is False


# The LLM (``google`` provider, Gemini API key) shares the same key and the same
# rule (gemini-key-voice 1.2, SEC-8 LLM side).


@pytest.mark.parametrize("api_key", ["", "   ", None])
@patch("google.genai.Client")
def test_llm_empty_api_key_is_rejected(mock_client_class, api_key):
    from pipecat.services.google.llm import GoogleLLMService

    with patch.dict(os.environ, HOSTILE_ENV):
        with pytest.raises(ValueError, match="non-empty api_key"):
            GoogleLLMService(api_key=api_key)
    mock_client_class.assert_not_called()


@patch("google.genai.Client")
def test_llm_client_is_pinned_to_the_gemini_api(mock_client_class):
    from pipecat.services.google.llm import GoogleLLMService

    with patch.dict(os.environ, HOSTILE_ENV):
        GoogleLLMService(api_key=FAKE_KEY)
    _, kwargs = mock_client_class.call_args
    assert kwargs["api_key"] == FAKE_KEY
    assert kwargs["vertexai"] is False


def test_llm_real_client_ignores_vertex_environment():
    from pipecat.services.google.llm import GoogleLLMService

    with patch.dict(os.environ, HOSTILE_ENV):
        llm = GoogleLLMService(api_key=FAKE_KEY)
    api_client = llm._client._api_client
    assert api_client.vertexai is False
    assert api_client.api_key == FAKE_KEY


def test_vertex_llm_still_uses_vertex_client():
    from pipecat.services.google.vertex import llm as vertex_llm

    with (
        patch.object(vertex_llm.GoogleVertexLLMService, "_get_credentials", return_value=None),
        patch.object(vertex_llm, "Client") as mock_client_class,
    ):
        vertex_llm.GoogleVertexLLMService(project_id="demo-project", location="us-east4")
    _, kwargs = mock_client_class.call_args
    assert kwargs["vertexai"] is True
    assert kwargs["project"] == "demo-project"


# --- D8: the key travels in a header, never in the URL -----------------------


class Captured(Exception):
    """Raised by the transport doubles once the outgoing request is recorded."""


def assert_key_only_in_header(url: str, headers: dict):
    assert FAKE_KEY not in url
    assert "key=" not in url
    assert url.split("/")[2] == "generativelanguage.googleapis.com"
    assert headers["x-goog-api-key"] == FAKE_KEY


@pytest.mark.asyncio
async def test_stt_live_websocket_sends_key_in_header():
    captured = {}

    class FakeWsConnect:
        def __init__(self, uri, additional_headers=None, **kwargs):
            captured.update(uri=uri, headers=dict(additional_headers or {}), kwargs=kwargs)

        async def __aenter__(self):
            raise Captured()

        async def __aexit__(self, *args):
            return None

    with patch.dict(os.environ, HOSTILE_ENV), patch("google.genai.live.ws_connect", FakeWsConnect):
        stt = GeminiSTTService(api_key=FAKE_KEY)
        ctx = stt._client.aio.live.connect(
            model="gemini-3.5-transcribe-live", config=stt._build_live_config()
        )
        with pytest.raises(Captured):
            await ctx.__aenter__()

    assert captured["uri"].startswith("wss://")
    assert_key_only_in_header(captured["uri"], captured["headers"])
    # google-genai leaves websockets' proxy setting at its default (proxy=True:
    # use the proxy from the environment); whether that works through the egress
    # proxy is verified by tasks 1.3.
    assert "proxy" not in captured["kwargs"]


async def capture_rest_request(make_call):
    """Run ``make_call`` with aiohttp's request intercepted; return what was sent."""
    import aiohttp

    captured = {}

    async def fake_request(self, *, method, url, headers=None, **kwargs):
        captured.update(url=str(url), headers=dict(headers or {}), trust_env=self.trust_env)
        raise Captured()

    with (
        patch.dict(os.environ, HOSTILE_ENV),
        patch.object(aiohttp.ClientSession, "request", fake_request),
    ):
        with pytest.raises(Captured):
            await make_call()
    return captured


@pytest.mark.asyncio
async def test_tts_rest_request_sends_key_in_header():
    from google.genai import types

    with patch.dict(os.environ, HOSTILE_ENV):
        tts = GeminiTTSService(api_key=FAKE_KEY)

    async def call():
        stream = await tts._client.aio.models.generate_content_stream(
            model="gemini-2.5-flash-preview-tts",
            contents="hi",
            config=types.GenerateContentConfig(response_modalities=["AUDIO"]),
        )
        async for _ in stream:  # the request is sent on first iteration
            pass

    captured = await capture_rest_request(call)
    assert_key_only_in_header(captured["url"], captured["headers"])
    # The async REST path goes through aiohttp with trust_env=True (reads the
    # proxy environment variables); the egress itself is verified by 1.3.
    assert captured["trust_env"] is True


@pytest.mark.asyncio
async def test_llm_rest_request_sends_key_in_header():
    from pipecat.services.google.llm import GoogleLLMService

    llm = GoogleLLMService(api_key=FAKE_KEY)

    async def call():
        stream = await llm._client.aio.models.generate_content_stream(
            model="gemini-3.8-flash", contents="hi"
        )
        async for _ in stream:  # the request is sent on first iteration
            pass

    captured = await capture_rest_request(call)
    assert_key_only_in_header(captured["url"], captured["headers"])
    assert captured["trust_env"] is True
