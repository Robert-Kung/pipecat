#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Fork tests (gemini-key-voice 1.5 review-r1): Gemini key-path resilience.

- REV-72: a Gemini Live STT session the server announces it will end
  (``go_away``) is replaced without an ``ErrorFrame`` and without losing the
  caller's audio; audio arriving while no session is usable is replayed.
- REV-73: a connection attempt whose setup reply never arrives times out, so
  ``start()`` returns and the failure is reported as a non-fatal ``ErrorFrame``.
- REV-78: a Gemini TTS response without audio is reported as a non-fatal
  ``ErrorFrame``.
- REV-28: the Gemini File API client sends the key in the ``x-goog-api-key``
  header (never in the URL) and honors the proxy environment variables.

All failures are injected with test doubles; network connections are refused.
"""

import asyncio
import socket
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from google.genai import errors as genai_errors

import pipecat.services.google.gemini_live.stt as gemini_stt
from pipecat.frames.frames import ErrorFrame, InputAudioRawFrame, TranscriptionFrame, TTSSpeakFrame
from pipecat.services.google.gemini_live.file_api import GeminiFileAPI
from pipecat.services.google.gemini_live.stt import GeminiSTTService
from pipecat.services.google.tts import GeminiTTSNoAudioError, GeminiTTSService
from pipecat.tests.utils import SleepFrame, run_test

# Fixed fake key (design D8 scan rule: the 1.1/1.8 unit-test value).
FAKE_KEY = "AIza" + "0" * 35

# Generous upper bound for one run_test; a hang (e.g. start() never returning)
# fails the test instead of blocking the suite.
RUN_TIMEOUT = 5.0


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def refuse(*args, **kwargs):
        raise RuntimeError("network access is not allowed in this test")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket.socket, "connect_ex", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)


@pytest.fixture
def fast_timings(monkeypatch):
    monkeypatch.setattr(gemini_stt, "RECONNECT_INTERVAL_SECONDS", 0.02)
    monkeypatch.setattr(gemini_stt, "SETUP_TIMEOUT_SECONDS", 0.3)
    monkeypatch.setattr(gemini_stt, "GO_AWAY_DRAIN_SECONDS", 0.2)


def server_close(code=1011, reason="ABORTED"):
    # What google-genai raises when the server closes the Live websocket.
    return genai_errors.APIError(code, reason, None)


def go_away(time_left="1s"):
    return SimpleNamespace(server_content=None, go_away=SimpleNamespace(time_left=time_left))


def final_transcript(text):
    transcription = SimpleNamespace(text=text, language_code=None)
    return SimpleNamespace(
        go_away=None,
        server_content=SimpleNamespace(
            interim_input_transcription=None, input_transcription=transcription
        ),
    )


def audio(tag: int) -> InputAudioRawFrame:
    # 10 ms of 16 kHz 16-bit mono PCM, filled with a marker byte.
    return InputAudioRawFrame(audio=bytes([tag]) * 320, sample_rate=16000, num_channels=1)


class ScriptedSession:
    """A live session whose server side follows ``script``.

    Script steps: ``("sleep", seconds)``, ``("msg", message)``, ``("raise",
    error)``. After the script the session stays open without messages.
    ``receive()`` may be called again and continues where the script left off.
    """

    def __init__(self, *script):
        self.script = list(script)
        self.audio = b""
        self.stream_ends = 0
        self.closed = False

    async def send_realtime_input(self, audio=None, audio_stream_end=None):
        if audio is not None:
            self.audio += audio.data
        if audio_stream_end:
            self.stream_ends += 1

    async def receive(self):
        while self.script:
            kind, value = self.script.pop(0)
            if kind == "sleep":
                await asyncio.sleep(value)
            elif kind == "msg":
                yield value
            else:
                raise value
        await asyncio.Event().wait()


class Stall:
    """A connection attempt whose setup reply does not arrive.

    The attempt only ends after ``give_up_after`` seconds, like a websocket
    whose keepalive finally notices the stalled tunnel; that is far longer than
    the setup timeout, so a timed-out attempt never gets there.
    """

    def __init__(self, give_up_after=3.0):
        self.give_up_after = give_up_after
        self.cancelled = False


class Delayed:
    """A connection attempt that succeeds after ``delay`` seconds."""

    def __init__(self, delay, session):
        self.delay = delay
        self.session = session


class ScriptedLiveClient:
    """Stands in for ``google.genai.Client``; one outcome per connection attempt.

    An outcome is a session (live at once), ``Delayed``, ``Stall`` or an
    exception raised by the attempt. When exhausted, the last outcome repeats.
    """

    def __init__(self, *outcomes):
        self.outcomes = list(outcomes)
        self.attempts = 0
        self.aio = SimpleNamespace(live=SimpleNamespace(connect=self._connect))

    def __call__(self, **kwargs):
        return self

    def _connect(self, *, model, config):
        outcome = self.outcomes[min(self.attempts, len(self.outcomes) - 1)]
        client = self

        class Ctx:
            async def __aenter__(self):
                client.attempts += 1
                if isinstance(outcome, Stall):
                    try:
                        await asyncio.sleep(outcome.give_up_after)
                    except asyncio.CancelledError:
                        outcome.cancelled = True
                        raise
                    raise server_close(1011, "keepalive ping timeout")
                if isinstance(outcome, Delayed):
                    await asyncio.sleep(outcome.delay)
                    return outcome.session
                if isinstance(outcome, BaseException):
                    raise outcome
                return outcome

            async def __aexit__(self, exc_type, exc, tb):
                if isinstance(outcome, Delayed):
                    outcome.session.closed = True
                elif isinstance(outcome, ScriptedSession):
                    outcome.closed = True
                return None

        return Ctx()


async def run_stt(client, frames):
    with patch.object(gemini_stt, "Client", client):
        stt = GeminiSTTService(api_key=FAKE_KEY)
        down, up = await asyncio.wait_for(run_test(stt, frames_to_send=frames), timeout=RUN_TIMEOUT)
    return stt, down, [f for f in up if isinstance(f, ErrorFrame)]


# --- REV-73: setup timeout ----------------------------------------------------


@pytest.mark.asyncio
async def test_stt_stalled_setup_times_out_and_reports_non_fatal_error(fast_timings, monkeypatch):
    monkeypatch.setattr(gemini_stt, "SETUP_TIMEOUT_SECONDS", 0.05)
    stall = Stall()
    client = ScriptedLiveClient(stall)

    _, _, errors = await run_stt(client, [SleepFrame(sleep=0.3)])

    # start() returned (the pipeline ran), and every timed-out attempt was
    # reported and retried (the attempt in progress at stop is not reported).
    assert len(errors) >= 2
    assert client.attempts - 1 <= len(errors) <= client.attempts
    assert all(not f.fatal for f in errors)
    assert all(isinstance(f.exception, TimeoutError) for f in errors)
    # The abandoned attempt was cancelled, not left running.
    assert stall.cancelled


@pytest.mark.asyncio
async def test_stt_setup_within_timeout_is_not_an_error(fast_timings):
    session = ScriptedSession(("sleep", 0.02), ("msg", final_transcript("hello")))
    client = ScriptedLiveClient(Delayed(0.02, session))

    _, down, errors = await run_stt(client, [SleepFrame(sleep=0.15)])

    assert errors == []
    assert client.attempts == 1
    assert any(isinstance(f, TranscriptionFrame) and f.text == "hello" for f in down)


# --- REV-72: go_away / session rotation ---------------------------------------


@pytest.mark.asyncio
async def test_stt_go_away_rotates_session_without_error_or_audio_loss(fast_timings):
    old = ScriptedSession(
        ("sleep", 0.05),
        ("msg", go_away()),
        # What the old session still delivers after being flushed, then the
        # server ends it as announced.
        ("sleep", 0.02),
        ("msg", final_transcript("tail of the old session")),
        ("raise", server_close()),
    )
    new = ScriptedSession(("sleep", 0.1), ("msg", final_transcript("new session")))
    client = ScriptedLiveClient(old, new)

    _, down, errors = await run_stt(
        client,
        [audio(1), SleepFrame(sleep=0.15), audio(2), SleepFrame(sleep=0.3)],
    )

    assert errors == []
    assert client.attempts == 2
    # Audio before the swap went to the old session, after it to the new one.
    assert old.audio == audio(1).audio
    assert new.audio == audio(2).audio
    # The old session was flushed, its last result kept, and then closed.
    assert old.stream_ends == 1
    assert old.closed
    texts = [f.text for f in down if isinstance(f, TranscriptionFrame)]
    assert "tail of the old session" in texts
    assert "new session" in texts


@pytest.mark.asyncio
async def test_stt_go_away_without_replacement_reconnects_and_replays_audio(fast_timings):
    old = ScriptedSession(
        ("sleep", 0.03),
        ("msg", go_away()),
        ("sleep", 0.05),
        ("raise", server_close()),
    )
    reopened = ScriptedSession()
    # The replacement attempt fails; after the announced end the reconnect
    # takes a while, and audio sent meanwhile must reach the new session.
    client = ScriptedLiveClient(old, server_close(1011, "unavailable"), Delayed(0.15, reopened))

    _, _, errors = await run_stt(
        client,
        [SleepFrame(sleep=0.12), audio(3), audio(4), SleepFrame(sleep=0.3)],
    )

    # Neither the failed replacement attempt nor the announced end is an error.
    assert errors == []
    assert client.attempts == 3
    assert reopened.audio == audio(3).audio + audio(4).audio
    assert old.audio == b""


@pytest.mark.asyncio
async def test_stt_go_away_reconnect_failure_is_still_reported(fast_timings):
    old = ScriptedSession(("sleep", 0.03), ("msg", go_away()), ("raise", server_close()))
    failure = server_close(1008, "API key not valid.")
    client = ScriptedLiveClient(old, failure)

    _, _, errors = await run_stt(client, [SleepFrame(sleep=0.2)])

    # attempts: the first session, the failed replacement (not reported), then
    # reconnects after the announced end, each reported (never silent).
    assert len(errors) >= 2
    assert len(errors) == client.attempts - 2
    assert all(not f.fatal and f.exception is failure for f in errors)


@pytest.mark.asyncio
async def test_stt_audio_during_error_reconnect_is_replayed(fast_timings):
    failing = ScriptedSession(("sleep", 0.03), ("raise", server_close()))
    recovered = ScriptedSession()
    client = ScriptedLiveClient(failing, Delayed(0.05, recovered))

    _, _, errors = await run_stt(client, [SleepFrame(sleep=0.06), audio(5), SleepFrame(sleep=0.3)])

    # The stream error itself is still reported once.
    assert len(errors) == 1
    assert not errors[0].fatal
    assert recovered.audio == audio(5).audio


# --- REV-78: TTS response without audio ---------------------------------------


def tts_chunk(*, data=None, finish_reason=None, block_reason=None, text=None):
    if data is None and finish_reason is None and text is None:
        candidates = None
    else:
        part = SimpleNamespace(
            inline_data=SimpleNamespace(data=data) if data is not None else None, text=text
        )
        candidates = [
            SimpleNamespace(content=SimpleNamespace(parts=[part]), finish_reason=finish_reason)
        ]
    feedback = SimpleNamespace(block_reason=block_reason) if block_reason else None
    return SimpleNamespace(candidates=candidates, prompt_feedback=feedback)


async def run_tts(mock_client_class, chunks):
    mock_client = MagicMock()
    mock_client_class.return_value = mock_client

    async def stream(*args, **kwargs):
        async def gen():
            for chunk in chunks:
                yield chunk

        return gen()

    mock_client.aio.models.generate_content_stream = stream
    tts = GeminiTTSService(
        api_key=FAKE_KEY,
        settings=GeminiTTSService.Settings(model="gemini-2.5-flash-preview-tts", voice="Kore"),
    )
    _, up = await run_test(tts, frames_to_send=[TTSSpeakFrame(text="Hello.")])
    return [f for f in up if isinstance(f, ErrorFrame)]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "chunks, finish_reason, block_reason",
    [
        ([tts_chunk(block_reason="PROHIBITED_CONTENT")], None, "PROHIBITED_CONTENT"),
        ([tts_chunk(text="", finish_reason="OTHER")], "OTHER", None),
        ([], None, None),
    ],
    ids=["prompt-blocked", "finished-without-audio", "empty-stream"],
)
@patch("google.genai.Client")
async def test_tts_response_without_audio_pushes_non_fatal_error(
    mock_client_class, chunks, finish_reason, block_reason
):
    errors = await run_tts(mock_client_class, chunks)

    assert len(errors) == 1
    assert not errors[0].fatal
    assert isinstance(errors[0].exception, GeminiTTSNoAudioError)
    assert errors[0].exception.finish_reason == finish_reason
    assert errors[0].exception.block_reason == block_reason


@pytest.mark.asyncio
@patch("google.genai.Client")
async def test_tts_response_with_audio_is_not_an_error(mock_client_class):
    errors = await run_tts(
        mock_client_class, [tts_chunk(data=b"\x00\x01" * 800, finish_reason="STOP")]
    )
    assert errors == []


# --- REV-28: File API key in header, proxy env honored ------------------------


class Captured(Exception):
    """Raised by the transport double once the outgoing request is recorded."""


async def capture_file_api_request(make_call):
    import aiohttp

    captured = {}

    async def fake_request(self, method, url, *, params=None, headers=None, **kwargs):
        merged = dict(self.headers)
        merged.update(headers or {})
        captured.update(
            method=method,
            url=str(url),
            params=dict(params or {}),
            headers=merged,
            trust_env=self.trust_env,
        )
        raise Captured()

    with patch.object(aiohttp.ClientSession, "_request", fake_request):
        with pytest.raises(Captured):
            await make_call()
    return captured


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["upload", "get", "list", "delete"])
async def test_file_api_sends_key_in_header_and_honors_proxy_env(operation, tmp_path):
    api = GeminiFileAPI(api_key=FAKE_KEY)
    upload = tmp_path / "notes.txt"
    upload.write_text("hello")
    calls = {
        "upload": lambda: api.upload_file(str(upload)),
        "get": lambda: api.get_file("files/abc123"),
        "list": lambda: api.list_files(page_size=5),
        "delete": lambda: api.delete_file("files/abc123"),
    }

    captured = await capture_file_api_request(calls[operation])

    assert FAKE_KEY not in captured["url"]
    assert "key=" not in captured["url"]
    assert "key" not in captured["params"]
    assert FAKE_KEY not in str(captured["params"])
    assert captured["url"].split("/")[2] == "generativelanguage.googleapis.com"
    assert captured["headers"]["x-goog-api-key"] == FAKE_KEY
    assert captured["trust_env"] is True
