"""The microphone pump: raw frames in, whole utterances out.

This is the piece that turns the STT and wake-word backends into an actual voice
loop. It reads fixed-size frames from an :class:`AudioSource`, gates them through
a VAD, glues the speech frames into an utterance, and transcribes once -- never
per frame.

Segmentation rules, all in frames so they are testable without a clock:

* ``start_frames`` consecutive speech frames open an utterance (ignores a cough);
* ``pre_roll_ms`` of preceding audio is prepended so the first word is not clipped;
* ``silence_hangover_ms`` of quiet closes it (survives the pause mid-sentence);
* shorter than ``min_speech_ms`` of actual speech is discarded as a noise burst;
* longer than ``max_utterance_s`` is cut off, so a stuck stream cannot grow forever.
"""

from __future__ import annotations

import threading
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import Any, Protocol, runtime_checkable

from ..core.events import EventBus
from .stt import STTBackend, Transcript
from .vad import VoiceActivityDetector

BYTES_PER_SAMPLE = 2  # int16 mono


@runtime_checkable
class AudioSource(Protocol):
    """A microphone, a file, or a test fixture. Returns ``None`` when exhausted."""

    name: str
    sample_rate: int

    def read(self, size: int) -> bytes | None: ...

    def close(self) -> None: ...


class NullAudioSource:
    """No microphone. The default, so a headless install never blocks on audio."""

    name = "null"

    def __init__(self, sample_rate: int = 16_000) -> None:
        self.sample_rate = sample_rate

    def read(self, size: int) -> bytes | None:
        return None

    def close(self) -> None:
        return None


class SyntheticAudioSource:
    """Replays a fixed list of frames, then reports end-of-stream. Used by tests."""

    def __init__(
        self, frames: Sequence[bytes], sample_rate: int = 16_000, loop: bool = False
    ) -> None:
        self.name = "synthetic"
        self.sample_rate = sample_rate
        self._frames = list(frames)
        self._loop = loop
        self._index = 0
        self.closed = False

    def read(self, size: int) -> bytes | None:
        if self._index >= len(self._frames):
            if not self._loop or not self._frames:
                return None
            self._index = 0
        frame = self._frames[self._index]
        self._index += 1
        return frame

    def close(self) -> None:
        self.closed = True


class SoundDeviceSource:
    """A real microphone via ``sounddevice`` (PortAudio), int16 mono."""

    name = "sounddevice"

    def __init__(self, sample_rate: int = 16_000, device: int | str | None = None) -> None:
        try:
            import sounddevice  # optional dependency, imported on demand
        except ImportError as exc:  # pragma: no cover - depends on the host
            raise RuntimeError("microphone input needs: pip install 'jarvis[stt]'") from exc
        self.sample_rate = sample_rate
        self._stream = sounddevice.RawInputStream(
            samplerate=sample_rate, channels=1, dtype="int16", device=device, blocksize=0
        )
        self._stream.start()

    def read(self, size: int) -> bytes | None:  # pragma: no cover - needs a microphone
        frames = size // BYTES_PER_SAMPLE
        data, overflowed = self._stream.read(frames)
        del overflowed  # a dropped frame is not worth interrupting the user over
        return bytes(data)

    def close(self) -> None:  # pragma: no cover - needs a microphone
        self._stream.stop()
        self._stream.close()


def build_audio_source(name: str, sample_rate: int = 16_000) -> AudioSource:
    if name in {"null", "none", ""}:
        return NullAudioSource(sample_rate)
    if name == "sounddevice":
        try:
            return SoundDeviceSource(sample_rate)
        except RuntimeError:
            return NullAudioSource(sample_rate)
    raise ValueError(f"unknown audio source: {name}")


@dataclass(frozen=True)
class ListenerSettings:
    sample_rate: int = 16_000
    frame_ms: int = 30
    start_frames: int = 3
    pre_roll_ms: int = 300
    silence_hangover_ms: int = 700
    min_speech_ms: int = 300
    max_utterance_s: float = 15.0

    @property
    def frame_bytes(self) -> int:
        return int(self.sample_rate * self.frame_ms / 1000) * BYTES_PER_SAMPLE

    @property
    def pre_roll_frames(self) -> int:
        return max(0, self.pre_roll_ms // self.frame_ms)

    @property
    def hangover_frames(self) -> int:
        return max(1, self.silence_hangover_ms // self.frame_ms)

    @property
    def max_frames(self) -> int:
        return max(1, int(self.max_utterance_s * 1000 / self.frame_ms))


class VoiceListener:
    """Reads frames, segments utterances, transcribes them, hands them on.

    ``poll_once`` processes exactly one frame and is the whole state machine, so
    the tests drive it directly -- no threads, no sleeps, no microphone.
    """

    def __init__(
        self,
        source: AudioSource,
        vad: VoiceActivityDetector,
        stt: STTBackend,
        on_utterance: Callable[[Transcript], Any] | None = None,
        on_speech_start: Callable[[], Any] | None = None,
        bus: EventBus | None = None,
        settings: ListenerSettings | None = None,
        is_open: Callable[[], bool] | None = None,
    ) -> None:
        self.source = source
        self.vad = vad
        self.stt = stt
        self.on_utterance = on_utterance
        self.on_speech_start = on_speech_start
        self.bus = bus
        self.settings = settings or ListenerSettings()
        # Half duplex: the orchestrator closes the mic while Jarvis speaks.
        self.is_open = is_open
        self.exhausted = False

        self._pre_roll: deque[bytes] = deque(maxlen=self.settings.pre_roll_frames)
        self._buffer: list[bytes] = []
        self._active = False
        self._speech_run = 0
        self._silence_run = 0
        self._speech_frames = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def capturing(self) -> bool:
        """True while an utterance is open, i.e. the user is mid-sentence."""
        return self._active

    # -- the state machine -------------------------------------------------

    def poll_once(self) -> Transcript | None:
        """Consume one frame. Returns a transcript only when an utterance ends."""
        frame = self.source.read(self.settings.frame_bytes)
        if frame is None or not frame:
            self.exhausted = True
            return self._finalize("stream ended") if self._active else None

        if self.is_open is not None and not self.is_open():
            # Drain the device but ignore what it heard: this is Jarvis's own
            # voice coming back through the speakers. Feeding it to the VAD
            # would also drag its noise floor up for minutes afterwards.
            if self._active:
                self._reset_segment()
                self._publish("dropped", reason="microphone closed while speaking")
            return None

        speech = self.vad.is_speech(frame, self.settings.sample_rate)

        if not self._active:
            self._pre_roll.append(frame)
            self._speech_run = self._speech_run + 1 if speech else 0
            if self._speech_run >= self.settings.start_frames:
                self._open()
            return None

        self._buffer.append(frame)
        if speech:
            self._speech_frames += 1
            self._silence_run = 0
        else:
            self._silence_run += 1

        if self._silence_run >= self.settings.hangover_frames:
            return self._finalize("silence")
        if len(self._buffer) >= self.settings.max_frames:
            return self._finalize("max length")
        return None

    def _open(self) -> None:
        self._active = True
        self._buffer = list(self._pre_roll)  # keep the syllables that opened it
        self._speech_frames = self._speech_run
        self._silence_run = 0
        self._pre_roll.clear()
        self._publish("speech-start")
        if self.on_speech_start is not None:
            self.on_speech_start()

    def _finalize(self, reason: str) -> Transcript | None:
        audio = b"".join(self._buffer)
        speech_ms = self._speech_frames * self.settings.frame_ms
        self._reset_segment()

        if speech_ms < self.settings.min_speech_ms:
            self._publish("discarded", reason=f"only {speech_ms}ms of speech")
            return None

        transcript = self.stt.transcribe(audio, self.settings.sample_rate)
        if transcript and not transcript.duration_s:
            # Backends that do not report duration still get a truthful one.
            seconds = len(audio) / (BYTES_PER_SAMPLE * self.settings.sample_rate)
            transcript = replace(transcript, duration_s=seconds)
        if not transcript:
            self._publish("empty", reason="transcriber returned nothing")
            return None

        self._publish("utterance", text=transcript.text, reason=reason, speech_ms=speech_ms)
        return transcript

    def _reset_segment(self) -> None:
        self._active = False
        self._buffer = []
        self._speech_run = 0
        self._silence_run = 0
        self._speech_frames = 0
        self._pre_roll.clear()

    def _publish(self, status: str, **payload: Any) -> None:
        if self.bus is not None:
            self.bus.publish("voice", status=status, **payload)

    # -- background loop ---------------------------------------------------

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        if self.running:
            return
        self._stop.clear()
        self.exhausted = False
        self._thread = threading.Thread(target=self.run, name="jarvis-listen", daemon=True)
        self._thread.start()

    def run(self) -> None:
        """Pump frames until stopped or the source runs dry."""
        while not self._stop.is_set():
            try:
                transcript = self.poll_once()
            except Exception as exc:  # a glitching device must not kill the loop
                self._publish("error", reason=f"{type(exc).__name__}: {exc}")
                break
            if self.exhausted:
                break
            if transcript is not None and self.on_utterance is not None:
                self.on_utterance(transcript)

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        self.source.close()
