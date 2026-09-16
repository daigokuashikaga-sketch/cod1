"""Audio output.

The TTS backends synthesise PCM; something has to put it on the speakers, and
that something has to be interruptible from another thread -- otherwise barge-in
is a lie. Playback is therefore its own seam rather than a detail of the TTS
backend: ``play()`` returns immediately, ``stop()`` cuts the current utterance.
"""

from __future__ import annotations

import threading
from typing import Any, Protocol, runtime_checkable

DEFAULT_SAMPLE_RATE = 24_000  # Kokoro's output rate
CHUNK_FRAMES = 1024


@runtime_checkable
class AudioPlayer(Protocol):
    name: str

    def play(self, pcm: bytes, sample_rate: int = DEFAULT_SAMPLE_RATE) -> None: ...

    def stop(self) -> None: ...

    @property
    def is_playing(self) -> bool: ...


class NullPlayer:
    """Plays nothing, remembers everything. The default, and what the tests use."""

    name = "null"

    def __init__(self) -> None:
        self.played: list[bytes] = []
        self.stops = 0

    def play(self, pcm: bytes, sample_rate: int = DEFAULT_SAMPLE_RATE) -> None:
        self.played.append(pcm)

    def stop(self) -> None:
        self.stops += 1

    @property
    def is_playing(self) -> bool:
        return False


class SoundDevicePlayer:
    """Streams int16 PCM through PortAudio on a worker thread.

    The worker writes in small chunks and checks a cancel flag between them, so
    ``stop()`` takes effect within a chunk rather than at the end of a sentence.
    """

    name = "sounddevice"

    def __init__(self, device: int | str | None = None, module: Any | None = None) -> None:
        if module is None:
            try:
                import sounddevice  # optional dependency, imported on demand
            except ImportError as exc:  # pragma: no cover - depends on the host
                raise RuntimeError("audio output needs: pip install 'jarvis[stt]'") from exc
            module = sounddevice
        self._sd = module
        self._device = device
        self._cancel = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.RLock()

    def play(self, pcm: bytes, sample_rate: int = DEFAULT_SAMPLE_RATE) -> None:
        if not pcm:
            return
        self.stop()
        self._cancel.clear()
        with self._lock:
            self._thread = threading.Thread(
                target=self._stream, args=(pcm, sample_rate), name="jarvis-speak", daemon=True
            )
            self._thread.start()

    def _stream(self, pcm: bytes, sample_rate: int) -> None:
        stream = self._sd.RawOutputStream(
            samplerate=sample_rate, channels=1, dtype="int16", device=self._device
        )
        stream.start()
        try:
            step = CHUNK_FRAMES * 2  # int16 mono
            for offset in range(0, len(pcm), step):
                if self._cancel.is_set():
                    return
                stream.write(pcm[offset : offset + step])
        finally:
            stream.stop()
            stream.close()

    def stop(self) -> None:
        self._cancel.set()
        with self._lock:
            thread = self._thread
            self._thread = None
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=1.0)

    @property
    def is_playing(self) -> bool:
        with self._lock:
            return self._thread is not None and self._thread.is_alive()


def build_player(name: str) -> AudioPlayer:
    if name in {"null", "none", ""}:
        return NullPlayer()
    if name == "sounddevice":
        try:
            return SoundDevicePlayer()
        except RuntimeError:
            return NullPlayer()
    raise ValueError(f"unknown audio player: {name}")
