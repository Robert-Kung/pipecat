#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#
# The wire protocol handling (handshake fields, event types, close-code
# semantics) is adapted from livekit-plugins-meta 1.8.11 ``stt.py``,
# Copyright 2023 LiveKit, Inc., licensed under the Apache License 2.0
# (http://www.apache.org/licenses/LICENSE-2.0). Modified: rewritten as a
# Pipecat STTService with a different failure/reconnect model, explicit proxy
# resolution, and no environment-variable credential fallback.
#

"""Meta Muse realtime speech-to-text service for Pipecat.

``MuseSTTService`` streams 24 kHz PCM audio to Muse realtime ASR over a
WebSocket and pushes interim and final transcription frames.
"""

import asyncio
import json
import urllib.request
from collections import deque
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlsplit

import aiohttp
from loguru import logger

from pipecat.audio.utils import create_stream_resampler
from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    InterimTranscriptionFrame,
    StartFrame,
    TranscriptionFrame,
)
from pipecat.services.settings import NOT_GIVEN, STTSettings, is_given
from pipecat.services.settings import _NotGiven as NotGiven
from pipecat.services.stt_service import STTService
from pipecat.utils.time import time_now_iso8601
from pipecat.utils.tracing.service_decorators import traced_stt

MUSE_ASR_URL = "wss://api.meta.ai/v1/asr/realtime"
DEFAULT_MODEL = "muse-voice-transcribe-1.0"

# Muse realtime ASR only accepts 16-bit mono PCM at 24 kHz.
MUSE_SAMPLE_RATE = 24_000
# Audio is sent in 80 ms messages.
CHUNK_BYTES = int(MUSE_SAMPLE_RATE * 2 * 0.08)
MAX_MESSAGE_BYTES = 1024 * 1024

# Failure model mirrors GeminiSTTService so the application can count
# consecutive STT errors the same way for both: one non-fatal ErrorFrame per
# failure, a retry every RECONNECT_INTERVAL_SECONDS, no retry cap.
RECONNECT_INTERVAL_SECONDS = 1.0
# Opening the socket plus the server's handshake reply.
SETUP_TIMEOUT_SECONDS = 5.0
# Most recent audio kept while no connection can accept it.
PENDING_AUDIO_MAX_SECONDS = 5.0
# Speech end to final transcript, measured against the live service.
MUSE_TTFS_P99 = 1.0

OUTPUT_SCRIPTS = ("zh-TW", "none")


class MuseSTTError(Exception):
    """A Muse connection or stream failure.

    The message never contains the handshake or credentials: only the phase
    that failed, the exception type underneath, and the WebSocket close code.
    """

    def __init__(self, phase: str, *, cause: str | None = None, close_code: int | None = None):
        """Initialize the error.

        Args:
            phase: What failed (e.g. ``"connect"``, ``"handshake"``, ``"stream"``).
            cause: Name of the underlying exception type, if any.
            close_code: WebSocket close code, if the server closed the socket.
        """
        self.phase = phase
        self.cause = cause
        self.close_code = close_code
        detail = []
        if cause:
            detail.append(cause)
        if close_code is not None:
            detail.append(f"close_code={close_code}")
        suffix = f" ({', '.join(detail)})" if detail else ""
        super().__init__(f"Muse ASR {phase} failed{suffix}")


@dataclass
class MuseSTTSettings(STTSettings):
    """Settings for MuseSTTService.

    Parameters:
        language_bias: Muse language names to bias recognition toward
            (e.g. ``["Mandarin Chinese", "English"]``).
        keywords: Terms to bias recognition toward.
        output_script: ``"zh-TW"`` converts Simplified Chinese output to
            Traditional characters (character-level, no phrase substitution);
            ``"none"`` passes the text through unchanged.
    """

    language_bias: list[str] | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    keywords: list[str] | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    output_script: str | NotGiven = field(default_factory=lambda: NOT_GIVEN)


def resolve_https_proxy(url: str) -> str | None:
    """Return the HTTPS proxy for ``url`` from the environment, honoring NO_PROXY.

    aiohttp's ``trust_env`` looks proxies up by URL scheme, so a ``wss://`` URL
    only matches ``WSS_PROXY``; deployments set ``HTTPS_PROXY``. The proxy is
    therefore resolved here and passed explicitly.
    """
    host = urlsplit(url).hostname or ""
    if urllib.request.proxy_bypass(host):
        return None
    return urllib.request.getproxies().get("https")


class MuseSTTService(STTService):
    """Streaming speech-to-text using Meta Muse realtime ASR.

    Muse detects utterance boundaries itself (``ENDPOINTING`` mode). Interim
    results come from ``transcript`` events; each utterance is finalized by a
    ``speechComplete`` event, which is pushed as one ``TranscriptionFrame``
    with ``finalized=True``.

    Failures: a failed connection or handshake, a server ``error`` event, a
    setup that takes longer than ``SETUP_TIMEOUT_SECONDS``, and any close of
    the socket by the server (the service never ends the stream itself while
    running) each push one non-fatal ``ErrorFrame`` and are retried after
    ``RECONNECT_INTERVAL_SECONDS`` with no cap. Errors are non-fatal so the
    application can transfer the call before ending it. Once stopped or
    cancelled the service never reconnects. Audio received while no connection
    is usable (up to ``PENDING_AUDIO_MAX_SECONDS``) is sent to the next one.

    Event handlers available:

    - on_connected: Called when a stream is ready to receive audio.
    - on_disconnected: Called when a ready stream closes.
    - on_connection_error: Called with a description of each failure.
    """

    Settings = MuseSTTSettings
    _settings: Settings

    def __init__(
        self,
        *,
        api_key: str,
        sample_rate: int | None = None,
        settings: Settings | None = None,
        ttfs_p99_latency: float | None = MUSE_TTFS_P99,
        proxy: str | None | NotGiven = NOT_GIVEN,
        **kwargs,
    ):
        """Initialize the Muse STT service.

        Args:
            api_key: Muse API key. Required; no environment fallback.
            sample_rate: Input sample rate. If None, taken from the start frame.
            settings: Runtime-updatable settings. Defaults to
                ``muse-voice-transcribe-1.0`` biased to Mandarin Chinese and
                English, with Traditional Chinese output.
            ttfs_p99_latency: P99 latency from speech end to final transcript.
            proxy: HTTPS proxy URL for the WebSocket. Defaults to the proxy
                resolved from ``HTTPS_PROXY``/``NO_PROXY``; ``None`` connects
                directly.
            **kwargs: Additional arguments passed to the parent STTService.

        Raises:
            ValueError: If ``api_key`` is empty or ``output_script`` is unknown.
        """
        if not api_key or not api_key.strip():
            raise ValueError("MuseSTTService requires a non-empty api_key")

        default_settings = self.Settings(
            model=DEFAULT_MODEL,
            language=None,
            language_bias=["Mandarin Chinese", "English"],
            keywords=None,
            output_script="zh-TW",
        )
        if settings is not None:
            default_settings.apply_update(settings)
        if default_settings.output_script not in OUTPUT_SCRIPTS:
            raise ValueError(f"output_script must be one of {OUTPUT_SCRIPTS}")

        super().__init__(
            sample_rate=sample_rate,
            ttfs_p99_latency=ttfs_p99_latency,
            settings=default_settings,
            **kwargs,
        )

        self._api_key = api_key.strip()
        self._proxy = proxy if is_given(proxy) else resolve_https_proxy(MUSE_ASR_URL)
        self._converter = self._create_converter(default_settings.output_script)

        self._http: aiohttp.ClientSession | None = None
        self._ws: aiohttp.ClientWebSocketResponse | None = None
        self._receive_task: asyncio.Task | None = None
        self._stopped = False

        self._resampler = create_stream_resampler()
        self._send_buffer = bytearray()
        self._pending_audio: deque[bytes] = deque()
        self._pending_audio_bytes = 0

        self._reset_turns()

    @staticmethod
    def _create_converter(output_script: str):
        if output_script == "none":
            return None
        try:
            import opencc
        except ModuleNotFoundError as e:
            logger.error('Traditional Chinese output needs `uv add "pipecat-ai[meta]"`.')
            raise ImportError(f"Missing module: {e}") from e
        return opencc.OpenCC("s2tw")

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate processing metrics.

        Returns:
            True.
        """
        return True

    async def start(self, frame: StartFrame):
        """Start the service and connect.

        Args:
            frame: The start frame.
        """
        await super().start(frame)
        self._http = aiohttp.ClientSession()
        await self._connect()

    async def stop(self, frame: EndFrame):
        """Stop the service.

        Args:
            frame: The end frame.
        """
        self._stopped = True
        await super().stop(frame)
        await self._shutdown()

    async def cancel(self, frame: CancelFrame):
        """Cancel the service.

        Args:
            frame: The cancel frame.
        """
        self._stopped = True
        await super().cancel(frame)
        await self._shutdown()

    async def cleanup(self):
        """Release resources."""
        self._stopped = True
        await super().cleanup()
        await self._shutdown()

    async def _shutdown(self):
        await self._disconnect()
        self._clear_pending_audio()
        if self._http is not None:
            http, self._http = self._http, None
            await http.close()

    # -- audio --------------------------------------------------------------

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame | None, None]:
        """Send audio to Muse; results arrive through the receive loop.

        Args:
            audio: 16-bit mono PCM at the pipeline's input sample rate.

        Yields:
            None.
        """
        if self.sample_rate != MUSE_SAMPLE_RATE:
            audio = await self._resampler.resample(audio, self.sample_rate, MUSE_SAMPLE_RATE)
        self._send_buffer.extend(audio)
        while len(self._send_buffer) >= CHUNK_BYTES:
            chunk = bytes(self._send_buffer[:CHUNK_BYTES])
            del self._send_buffer[:CHUNK_BYTES]
            await self._send_chunk(chunk)
        yield None

    async def _send_chunk(self, chunk: bytes):
        ws = self._ws
        if ws is None or ws.closed:
            self._buffer_audio(chunk)
            return
        try:
            await ws.send_bytes(chunk)
        except Exception as e:
            # The receive loop sees the closed socket and reports the failure.
            logger.warning(f"{self}: audio send failed ({type(e).__name__}), closing the stream")
            self._buffer_audio(chunk)
            await ws.close()

    def _buffer_audio(self, chunk: bytes):
        if self._stopped:
            return
        self._pending_audio.append(chunk)
        self._pending_audio_bytes += len(chunk)
        max_bytes = int(PENDING_AUDIO_MAX_SECONDS * MUSE_SAMPLE_RATE * 2)
        while self._pending_audio_bytes > max_bytes and self._pending_audio:
            self._pending_audio_bytes -= len(self._pending_audio.popleft())

    def _clear_pending_audio(self):
        self._pending_audio.clear()
        self._pending_audio_bytes = 0

    # -- connection -----------------------------------------------------------

    def _handshake(self) -> dict[str, Any]:
        handshake: dict[str, Any] = {
            "mode": "ENDPOINTING",
            "authorization": {"accessToken": f"Bearer {self._api_key}"},
            "audioEncoding": "PCM_24KHZ",
            "model": self._settings.model,
            "partialMode": "CUMULATIVE",
            "emitAudioProgress": True,
        }
        keywords = self._settings.keywords
        if is_given(keywords) and keywords:
            handshake["keywords"] = list(keywords)
        language_bias = self._settings.language_bias
        if is_given(language_bias) and language_bias:
            handshake["languageBias"] = list(language_bias)
        return handshake

    async def _open_stream(self) -> aiohttp.ClientWebSocketResponse:
        """Open the socket and complete the handshake within SETUP_TIMEOUT_SECONDS.

        Raises:
            MuseSTTError: If the connection, handshake, or timeout fails.
        """
        assert self._http is not None
        ws: aiohttp.ClientWebSocketResponse | None = None
        try:
            async with asyncio.timeout(SETUP_TIMEOUT_SECONDS):
                try:
                    ws = await self._http.ws_connect(
                        MUSE_ASR_URL, proxy=self._proxy, max_msg_size=MAX_MESSAGE_BYTES
                    )
                except aiohttp.WSServerHandshakeError as e:
                    raise MuseSTTError("connect", cause=f"HTTP {e.status}") from None
                except aiohttp.ClientError as e:
                    raise MuseSTTError("connect", cause=type(e).__name__) from None
                await ws.send_str(json.dumps(self._handshake(), separators=(",", ":")))
                msg = await ws.receive()
                if msg.type != aiohttp.WSMsgType.TEXT:
                    raise MuseSTTError("handshake", close_code=ws.close_code)
                reply = self._parse(msg.data, phase="handshake")
                if reply.get("type") == "error" or not reply.get("sessionId"):
                    raise MuseSTTError("handshake", cause="rejected", close_code=ws.close_code)
        except TimeoutError:
            await self._close_quietly(ws)
            raise MuseSTTError("setup", cause="timeout") from None
        except BaseException:
            await self._close_quietly(ws)
            raise
        return ws

    @staticmethod
    def _parse(data: Any, *, phase: str) -> dict[str, Any]:
        try:
            message = json.loads(data)
        except (TypeError, ValueError):
            raise MuseSTTError(phase, cause="invalid JSON") from None
        if not isinstance(message, dict):
            raise MuseSTTError(phase, cause="invalid message")
        return message

    @staticmethod
    async def _close_quietly(ws: aiohttp.ClientWebSocketResponse | None):
        if ws is None:
            return
        try:
            await ws.close()
        except Exception:
            pass

    async def _activate(self, ws: aiohttp.ClientWebSocketResponse):
        """Send buffered audio, then make ``ws`` the stream audio goes to."""
        try:
            while self._pending_audio:
                chunk = self._pending_audio.popleft()
                self._pending_audio_bytes -= len(chunk)
                await ws.send_bytes(chunk)
        except Exception:
            await self._close_quietly(ws)
            raise
        self._ws = ws
        self._reset_turns()
        await self._call_event_handler("on_connected")
        logger.debug(f"{self}: connected to Muse")

    async def _connect(self):
        if self._stopped or self._receive_task:
            return
        retry_delay = 0.0
        try:
            await self._activate(await self._open_stream())
        except Exception as e:
            await self._report_failure(e)
            retry_delay = RECONNECT_INTERVAL_SECONDS
        self._receive_task = self.create_task(self._receive_handler(retry_delay))

    async def _disconnect(self):
        if self._receive_task:
            task, self._receive_task = self._receive_task, None
            await self.cancel_task(task, timeout=1.0)
        await self._close_stream()

    async def _close_stream(self):
        ws, self._ws = self._ws, None
        if ws is None:
            return
        await self._close_quietly(ws)
        await self._call_event_handler("on_disconnected")

    async def _receive_handler(self, retry_delay: float = 0.0):
        """Read the stream, reconnecting after every failure until stopped."""
        while not self._stopped:
            try:
                if retry_delay:
                    await asyncio.sleep(retry_delay)
                    retry_delay = 0.0
                    if self._stopped:
                        return
                if self._ws is None:
                    await self._activate(await self._open_stream())
                await self._receive_from(self._ws)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                await self._close_stream()
                await self._report_failure(e)
                retry_delay = RECONNECT_INTERVAL_SECONDS

    async def _receive_from(self, ws: aiohttp.ClientWebSocketResponse):
        """Handle server messages; raises when the stream ends for any reason."""
        while True:
            msg = await ws.receive()
            if msg.type == aiohttp.WSMsgType.TEXT:
                await self._handle_message(self._parse(msg.data, phase="stream"))
            elif msg.type in (
                aiohttp.WSMsgType.CLOSE,
                aiohttp.WSMsgType.CLOSING,
                aiohttp.WSMsgType.CLOSED,
            ):
                raise MuseSTTError("stream", cause="closed by server", close_code=ws.close_code)
            elif msg.type == aiohttp.WSMsgType.ERROR:
                raise MuseSTTError("stream", cause="socket error", close_code=ws.close_code)

    async def _report_failure(self, error: Exception):
        if self._stopped:
            return
        if not isinstance(error, MuseSTTError):
            # Never echo arbitrary exception text: only its type.
            error = MuseSTTError("stream", cause=type(error).__name__)
        logger.warning(f"{self}: {error}, retrying in {RECONNECT_INTERVAL_SECONDS}s")
        await self._call_event_handler("on_connection_error", str(error))
        await self.push_error(error_msg=f"Muse STT connection error: {error}", exception=error)

    # -- transcripts ----------------------------------------------------------

    def _reset_turns(self):
        """Forget turn state; turn ids are only meaningful within one stream."""
        self._completed_turns: set[Any] = set()
        self._interim_text: dict[Any, str] = {}
        self._active_turn: Any = None

    def _convert(self, text: str) -> str:
        return self._converter.convert(text) if self._converter else text

    async def _handle_message(self, message: dict[str, Any]):
        event = message.get("type")
        if event == "error":
            raise MuseSTTError("stream", cause="server error")
        if event == "speechStart":
            self._active_turn = message.get("turnId")
        elif event == "transcript":
            await self._handle_interim(message)
        elif event == "speechComplete":
            await self._handle_final(message)
        # speechEnd only marks the end of speech; audioProgress is usage info.

    async def _handle_interim(self, message: dict[str, Any]):
        text = message.get("transcript")
        if not isinstance(text, str) or not text:
            return
        turn = message.get("turnId", self._active_turn)
        if turn in self._completed_turns or self._interim_text.get(turn) == text:
            return
        self._interim_text[turn] = text
        await self.push_frame(
            InterimTranscriptionFrame(
                self._convert(text), self._user_id, time_now_iso8601(), None, result=message
            )
        )

    async def _handle_final(self, message: dict[str, Any]):
        turn = message.get("turnId", self._active_turn)
        if turn in self._completed_turns:
            return
        self._completed_turns.add(turn)
        self._interim_text.pop(turn, None)
        text = message.get("transcript")
        if not isinstance(text, str) or not text.strip():
            return
        text = self._convert(text)
        await self.push_frame(
            TranscriptionFrame(
                text, self._user_id, time_now_iso8601(), None, result=message, finalized=True
            )
        )
        await self._handle_transcription(text, True, None)
        await self.stop_processing_metrics()

    @traced_stt
    async def _handle_transcription(self, transcript: str, is_final: bool, language=None):
        """Handle a transcription result with tracing."""
        pass
