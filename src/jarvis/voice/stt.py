"""Speech to text.

``faster-whisper`` and ``whisper.cpp`` share the same Whisper weights, so the
choice is about hardware, not accuracy: faster-whisper on NVIDIA, whisper.cpp on
Apple Silicon. Only the faster-whisper adapter ships here; the protocol makes
adding the other one a single file.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class Transcript:
    text: str
    language: str = "en"
    confidence: float = 1.0
    duration_s: float = 0.0

    def __bool__(self) -> bool:
        return bool(self.text.strip())


@runtime_checkable
class STTBackend(Protocol):
    name: str
    available: bool

    def transcribe(self, audio: bytes, sample_rate: int = 16_000) -> Transcript: ...


class NullSTT:
    """No microphone. Returns empty transcripts."""

    name = "null"
    available = False

    def transcribe(self, audio: bytes, sample_rate: int = 16_000) -> Transcript:
        return Transcript("")


class ScriptedSTT:
    """Returns a fixed sequence of utterances. Used by tests and ``jarvis demo``."""

    name = "scripted"
    available = True

    def __init__(self, utterances: Sequence[str]) -> None:
        self._queue = list(utterances)

    def transcribe(self, audio: bytes, sample_rate: int = 16_000) -> Transcript:
        if not self._queue:
            return Transcript("")
        return Transcript(self._queue.pop(0))


class FasterWhisperSTT:
    """Local Whisper via CTranslate2. Free, fast on GPU, usable on CPU."""

    name = "faster_whisper"

    def __init__(
        self,
        model: str = "small.en",
        device: str = "auto",
        compute_type: str = "int8",
        model_obj: Any | None = None,
    ) -> None:
        if model_obj is not None:
            self._model = model_obj
            self.available = True
            return
        try:
            from faster_whisper import WhisperModel  # optional dependency
        except ImportError as exc:  # pragma: no cover - depends on the host
            raise RuntimeError("STT needs: pip install 'jarvis[stt]'") from exc
        self._model = WhisperModel(model, device=device, compute_type=compute_type)
        self.available = True

    def transcribe(
        self, audio: bytes, sample_rate: int = 16_000
    ) -> Transcript:  # pragma: no cover - needs the model
        samples = _pcm16_to_float32(audio)
        segments, info = self._model.transcribe(samples, beam_size=1, language="en")
        text = " ".join(segment.text.strip() for segment in segments).strip()
        return Transcript(
            text=text,
            language=getattr(info, "language", "en"),
            confidence=float(getattr(info, "language_probability", 1.0) or 1.0),
            duration_s=len(audio) / (2 * sample_rate),
        )


def _pcm16_to_float32(audio: bytes) -> list[float]:
    import array  # local: only this path needs it

    values = array.array("h")
    values.frombytes(audio[: len(audio) - (len(audio) % 2)])
    return [v / 32768.0 for v in values]


def build_stt(name: str, model: str = "small.en") -> STTBackend:
    if name in {"null", "none", ""}:
        return NullSTT()
    if name == "faster_whisper":
        try:
            return FasterWhisperSTT(model=model)
        except RuntimeError:
            return NullSTT()
    raise ValueError(f"unknown STT backend: {name}")
