"""Wake-word detection.

Two implementations: ``openWakeWord`` for audio (it ships a "hey jarvis" model),
and a text matcher for typed input and tests. Both honour the same suppression
rule -- ignore detections while Jarvis is speaking, or it will wake itself.
"""

from __future__ import annotations

import re
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class WakeWordDetector(Protocol):
    name: str
    phrase: str

    def detect(self, audio: bytes, sample_rate: int = 16_000) -> float: ...


class TextWakeWord:
    """Matches the wake phrase in transcribed text; strips it from the utterance."""

    name = "text"

    def __init__(self, phrase: str = "hey jarvis") -> None:
        self.phrase = phrase.lower().strip()
        escaped = r"\s+".join(re.escape(word) for word in self.phrase.split())
        self._pattern = re.compile(rf"\b{escaped}\b[\s,.:!-]*", re.IGNORECASE)

    def detect(self, audio: bytes, sample_rate: int = 16_000) -> float:
        return 0.0

    def matches(self, text: str) -> bool:
        return bool(self._pattern.search(text or ""))

    def strip_phrase(self, text: str) -> str:
        """Return the command with the wake phrase removed."""
        return self._pattern.sub("", text or "", count=1).strip()


class NullWakeWord:
    """Always-listening: every utterance counts, no wake phrase needed."""

    name = "null"

    def __init__(self, phrase: str = "") -> None:
        self.phrase = phrase

    def detect(self, audio: bytes, sample_rate: int = 16_000) -> float:
        return 1.0


class OpenWakeWord:
    """Audio wake word via openWakeWord. Bundles a 'hey jarvis' model."""

    name = "openwakeword"

    def __init__(
        self, phrase: str = "hey jarvis", threshold: float = 0.5, model: Any | None = None
    ) -> None:
        self.phrase = phrase
        self.threshold = threshold
        if model is not None:
            self._model = model
            return
        try:
            from openwakeword.model import Model  # optional dependency
        except ImportError as exc:  # pragma: no cover - depends on the host
            raise RuntimeError("wake word needs: pip install 'jarvis[stt]'") from exc
        self._model = Model(wakeword_models=[phrase.replace(" ", "_")])

    def detect(
        self, audio: bytes, sample_rate: int = 16_000
    ) -> float:  # pragma: no cover - needs the model
        scores = self._model.predict(audio)
        return float(max(scores.values())) if scores else 0.0


def build_wake_word(phrase: str, enabled: bool = True, backend: str = "text") -> WakeWordDetector:
    if not enabled:
        return NullWakeWord(phrase)
    if backend == "text":
        return TextWakeWord(phrase)
    if backend == "openwakeword":
        try:
            return OpenWakeWord(phrase)
        except RuntimeError:
            return TextWakeWord(phrase)
    raise ValueError(f"unknown wake-word backend: {backend}")
