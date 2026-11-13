#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for the Meta Muse realtime STT service against a local fake server."""

import asyncio
import json

import aiohttp
import pytest
import pytest_asyncio
from aiohttp import web

import pipecat.services.meta.stt as muse
from pipecat.frames.frames import InterimTranscriptionFrame, TranscriptionFrame
from pipecat.services.meta.stt import CHUNK_BYTES, MuseSTTError, MuseSTTService

API_KEY = "muse-test-key-0123456789abcdef"


class FakeMuse:
    """A Muse-like WebSocket server whose behavior per connection is scripted."""

    def __init__(self):
        self.connections = 0
        self.handshakes: list[dict] = []
        self.audio_bytes: list[int] = []
        self.script = None  # async (ws, index) -> None

    async def handler(self, request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        index = self.connections
        self.connections += 1
        msg = await ws.receive()
        self.handshakes.append(json.loads(msg.data))
        await self.script(ws, index)
        return ws


async def accept(ws):
    await ws.send_str(json.dumps({"sessionId": "s"}))


async def drain_audio(ws, fake, seconds=0.3):
    try:
        async with asyncio.timeout(seconds):
            async for m in ws:
                if m.type == aiohttp.WSMsgType.BINARY:
                    fake.audio_bytes.append(len(m.data))
    except TimeoutError:
        pass


@pytest_asyncio.fixture
async def fake(monkeypatch):
    server = FakeMuse()
    app = web.Application()
    app.router.add_get("/", server.handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    monkeypatch.setattr(muse, "MUSE_ASR_URL", f"ws://127.0.0.1:{port}/")
    monkeypatch.setattr(muse, "RECONNECT_INTERVAL_SECONDS", 0.05)
    yield server
    await runner.cleanup()


def make_service(**kwargs):
    service = MuseSTTService(api_key=API_KEY, proxy=None, **kwargs)
    service._sample_rate = 24000
    service._user_id = "user"
    frames, errors, events = [], [], []

    async def push_frame(frame, direction=None):
        frames.append(frame)

    async def push_error(error_msg, exception=None, fatal=False):
        errors.append((error_msg, exception, fatal))

    async def call_event_handler(name, *args):
        events.append(name)

    async def noop(*a, **k):
        pass

    def create_task(coro, name=None):
        return asyncio.create_task(coro)

    async def cancel_task(task, timeout=None):
        task.cancel()
        try:
            await task
        except BaseException:
            pass

    service.push_frame = push_frame
    service.push_error = push_error
    service._call_event_handler = call_event_handler
    service.stop_processing_metrics = noop
    service._handle_transcription = noop
    service.create_task = create_task
    service.cancel_task = cancel_task
    return service, frames, errors, events


async def run_service(service):
    service._http = aiohttp.ClientSession()
    await service._connect()


async def stop_service(service):
    service._stopped = True
    await service._shutdown()


async def wait_for(cond, timeout=2.0):
    async with asyncio.timeout(timeout):
        while not cond():
            await asyncio.sleep(0.01)


@pytest.mark.asyncio
async def test_handshake_carries_bearer_token_and_pcm_24khz(fake):
    async def script(ws, i):
        await accept(ws)
        await drain_audio(ws, fake)

    fake.script = script
    service, *_ = make_service()
    await run_service(service)
    await wait_for(lambda: fake.handshakes)
    await stop_service(service)

    hs = fake.handshakes[0]
    assert hs["authorization"] == {"accessToken": f"Bearer {API_KEY}"}
    assert hs["audioEncoding"] == "PCM_24KHZ"
    assert hs["mode"] == "ENDPOINTING"
    assert hs["partialMode"] == "CUMULATIVE"
    assert hs["languageBias"] == ["Mandarin Chinese", "English"]
    assert hs["model"] == "muse-voice-transcribe-1.0"


@pytest.mark.asyncio
async def test_final_comes_only_from_speech_complete_and_is_converted(fake):
    async def script(ws, i):
        await accept(ws)
        for event in [
            {"type": "speechStart", "turnId": 1},
            {"type": "transcript", "turnId": 1, "transcript": "请问"},
            {"type": "transcript", "turnId": 1, "transcript": "请问"},
            {"type": "transcript", "turnId": 1, "transcript": "请问营业时间"},
            {"type": "speechEnd", "turnId": 1},
            {"type": "speechComplete", "turnId": 1, "transcript": "请问你们营业时间？"},
            {"type": "audioProgress", "audioProcessedMs": 900},
        ]:
            await ws.send_str(json.dumps(event))
        await drain_audio(ws, fake)

    fake.script = script
    service, frames, errors, _ = make_service()
    await run_service(service)
    await wait_for(lambda: any(isinstance(f, TranscriptionFrame) for f in frames))
    await stop_service(service)

    interims = [f.text for f in frames if isinstance(f, InterimTranscriptionFrame)]
    finals = [f for f in frames if isinstance(f, TranscriptionFrame)]
    assert interims == ["請問", "請問營業時間"]
    assert len(finals) == 1
    assert finals[0].text == "請問你們營業時間？"
    assert finals[0].finalized is True
    assert errors == []


def test_conversion_is_character_level_and_keeps_ascii():
    service, *_ = make_service()
    assert service._convert("分机 1234，找林会计，email abc@x.com") == (
        "分機 1234，找林會計，email abc@x.com"
    )
    # Character-level only: no phrase substitution (软件 would become 軟體 with s2twp).
    assert service._convert("软件") == "軟件"


def test_output_script_none_passes_text_through():
    service, *_ = make_service(settings=MuseSTTService.Settings(output_script="none"))
    assert service._convert("请问") == "请问"


def test_empty_api_key_and_unknown_script_are_rejected():
    with pytest.raises(ValueError):
        MuseSTTService(api_key=" ", proxy=None)
    with pytest.raises(ValueError):
        MuseSTTService(
            api_key=API_KEY, proxy=None, settings=MuseSTTService.Settings(output_script="x")
        )


@pytest.mark.asyncio
async def test_rejected_handshake_reports_non_fatal_errors_and_keeps_retrying(fake):
    async def script(ws, i):
        await ws.close(code=1008)

    fake.script = script
    service, frames, errors, _ = make_service()
    await run_service(service)
    await wait_for(lambda: len(errors) >= 3)
    await stop_service(service)

    assert fake.connections >= 3
    for msg, exc, fatal in errors:
        assert fatal is False
        assert isinstance(exc, MuseSTTError)
        assert API_KEY not in msg and API_KEY not in str(exc)
        assert "Bearer" not in msg


@pytest.mark.asyncio
async def test_server_error_event_is_reported(fake):
    async def script(ws, i):
        await ws.send_str(json.dumps({"type": "error", "message": "invalid token"}))
        await ws.close()

    fake.script = script
    service, _, errors, _ = make_service()
    await run_service(service)
    await wait_for(lambda: errors)
    await stop_service(service)
    assert "handshake" in errors[0][0]
    assert "invalid token" not in errors[0][0]


@pytest.mark.asyncio
async def test_server_close_1000_mid_stream_is_an_error_and_turns_reset(fake):
    async def script(ws, i):
        await accept(ws)
        if i == 0:
            await ws.send_str(
                json.dumps({"type": "speechComplete", "turnId": 1, "transcript": "第一句"})
            )
            await ws.close(code=1000)
        else:
            # A new stream may reuse turn ids from the previous one.
            await ws.send_str(
                json.dumps({"type": "speechComplete", "turnId": 1, "transcript": "第二句"})
            )
            await drain_audio(ws, fake)

    fake.script = script
    service, frames, errors, events = make_service()
    await run_service(service)
    await wait_for(lambda: len([f for f in frames if isinstance(f, TranscriptionFrame)]) == 2)
    await stop_service(service)

    assert [f.text for f in frames if isinstance(f, TranscriptionFrame)] == ["第一句", "第二句"]
    assert len(errors) == 1 and errors[0][2] is False
    assert errors[0][1].close_code == 1000
    assert events.count("on_connected") == 2


@pytest.mark.asyncio
async def test_no_reconnect_after_stop(fake):
    async def script(ws, i):
        await ws.close(code=1011)

    fake.script = script
    service, _, errors, _ = make_service()
    await run_service(service)
    await wait_for(lambda: errors)
    await stop_service(service)
    seen = fake.connections
    await asyncio.sleep(0.3)
    assert fake.connections == seen


@pytest.mark.asyncio
async def test_setup_timeout_counts_as_failure(fake, monkeypatch):
    monkeypatch.setattr(muse, "SETUP_TIMEOUT_SECONDS", 0.2)

    async def script(ws, i):
        await asyncio.sleep(1.0)  # never answers the handshake

    fake.script = script
    service, _, errors, _ = make_service()
    await run_service(service)
    await wait_for(lambda: errors)
    await stop_service(service)
    assert errors[0][1].phase == "setup"


@pytest.mark.asyncio
async def test_audio_is_resampled_and_chunked_to_80ms(fake):
    async def script(ws, i):
        await accept(ws)
        await drain_audio(ws, fake, seconds=1.0)

    fake.script = script
    service, *_ = make_service()
    service._sample_rate = 16000
    await run_service(service)
    await wait_for(lambda: service._ws is not None)
    one_second_16k = b"\x01\x00" * 16000
    async for _ in service.run_stt(one_second_16k):
        pass
    await wait_for(lambda: sum(fake.audio_bytes) >= CHUNK_BYTES * 10)
    await stop_service(service)
    assert all(n == CHUNK_BYTES for n in fake.audio_bytes)
    assert CHUNK_BYTES == 3840  # 80 ms at 24 kHz, 16-bit mono


@pytest.mark.asyncio
async def test_audio_without_a_stream_is_capped(fake):
    service, *_ = make_service()
    for _ in range(200):  # 16 s of audio
        service._buffer_audio(b"\0" * CHUNK_BYTES)
    assert service._pending_audio_bytes <= int(muse.PENDING_AUDIO_MAX_SECONDS * 24000 * 2)


def test_proxy_resolution_uses_https_proxy_and_honors_no_proxy(monkeypatch):
    for var in ("HTTPS_PROXY", "https_proxy", "NO_PROXY", "no_proxy", "WSS_PROXY", "wss_proxy"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("HTTPS_PROXY", "http://egress-proxy-ai:3128")
    assert muse.resolve_https_proxy("wss://api.meta.ai/v1/asr/realtime") == (
        "http://egress-proxy-ai:3128"
    )
    monkeypatch.setenv("NO_PROXY", "api.meta.ai")
    assert muse.resolve_https_proxy("wss://api.meta.ai/v1/asr/realtime") is None


@pytest.mark.asyncio
async def test_connection_goes_through_explicit_proxy(monkeypatch):
    """With only HTTPS_PROXY set, the CONNECT reaches the proxy (aiohttp trust_env would not)."""
    connects = []

    async def proxy_handler(reader, writer):
        line = await reader.readline()
        connects.append(line.decode().strip())
        writer.write(b"HTTP/1.1 403 Forbidden\r\nContent-Length: 0\r\n\r\n")
        await writer.drain()
        writer.close()

    server = await asyncio.start_server(proxy_handler, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    for var in ("NO_PROXY", "no_proxy", "WSS_PROXY", "wss_proxy", "https_proxy"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("HTTPS_PROXY", f"http://127.0.0.1:{port}")

    service = MuseSTTService(api_key=API_KEY)
    service._http = aiohttp.ClientSession()
    try:
        with pytest.raises(MuseSTTError) as info:
            await service._open_stream()
    finally:
        await service._http.close()
        server.close()
    assert connects and connects[0].startswith("CONNECT api.meta.ai:443")
    assert info.value.phase == "connect"
    assert API_KEY not in str(info.value)
