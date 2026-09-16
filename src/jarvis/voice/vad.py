"""Voice activity detection: the gate that decides when audio is worth transcribing.

Running Whisper on every 30ms of room tone would burn CPU all day and transcribe
keyboard clatter as words. A VAD is the same idea as the screen loop's change
detector -- a cheap local filter in front of an expensive step.

``EnergyVAD`` is the stdlib default: RMS against an adaptive noise floor. It is
crude but it costs microseconds and needs no model. ``SileroVAD`` wraps the real
thing (~2MB ONNX) when the ``stt`` extra is installed.
"""

from __future__ import annotations

import array
import math
import sys
from typing import Any, Protocol, runtime_checkable

INT16_MAX = 32768.0


@runtime_checkable
class VoiceActivityDetector(Protocol):
    name: str

    def is_speech(self, frame: bytes, sample_rate: int = 16_000) -> bool: ...

    def reset(self) -> None: ...


def rms(frame: bytes) -> float:
    """Root-mean-square amplitude of a little-endian int16 frame, 0.0-1.0."""
    samples = array.array("h")
    samples.frombytes(frame[: len(frame) - (len(frame) % 2)])
    if sys.byteorder == "big":  # the wire format is little-endian everywhere we read it
        samples.byteswap()
    if not samples:
        return 0.0
    total = math.fsum(float(sample) * float(sample) for sample in samples)
    return math.sqrt(total / len(samples)) / INT16_MAX


class EnergyVAD:
    """RMS against an adaptive noise floor.

    The floor adapts asymmetrically: it drops quickly when the room goes quiet
    and rises slowly when it does not. That asymmetry is the whole trick. Only
    adapting on frames judged "not speech" deadlocks -- a fan or an air
    conditioner sits above the threshold forever, so every frame looks like
    speech and the floor never learns. Rising slowly instead means steady noise
    is absorbed within a few seconds, while a genuine utterance (a second or
    two) barely moves the bar and still registers.
    """

    name = "energy"

    def __init__(
        self,
        threshold_ratio: float = 3.0,
        absolute_floor: float = 0.006,
        adapt_down: float = 0.08,
        adapt_up: float = 0.005,
        initial_noise: float = 0.002,
    ) -> None:
        self.threshold_ratio = threshold_ratio
        self.absolute_floor = absolute_floor
        self.adapt_down = adapt_down
        self.adapt_up = adapt_up
        self._initial_noise = initial_noise
        self._noise = initial_noise

    @property
    def noise_floor(self) -> float:
        return self._noise

    @property
    def threshold(self) -> float:
        return max(self.absolute_floor, self._noise * self.threshold_ratio)

    def is_speech(self, frame: bytes, sample_rate: int = 16_000) -> bool:
        level = rms(frame)
        speech = level > self.threshold
        adapt = self.adapt_down if level < self._noise else self.adapt_up
        self._noise = (1.0 - adapt) * self._noise + adapt * level
        return speech

    def reset(self) -> None:
        self._noise = self._initial_noise


class AlwaysOnVAD:
    """Treats every frame as speech. For push-to-talk and for tests."""

    name = "always"

    def is_speech(self, frame: bytes, sample_rate: int = 16_000) -> bool:
        return True

    def reset(self) -> None:
        return None


class SileroVAD:
    """Silero VAD via its ONNX model. Far better than energy in a noisy room."""

    name = "silero"

    def __init__(self, threshold: float = 0.5, model: Any | None = None) -> None:
        self.threshold = threshold
        if model is not None:
            self._model = model
            return
        try:
            from silero_vad import load_silero_vad  # optional dependency
        except ImportError as exc:  # pragma: no cover - depends on the host
            raise RuntimeError("Silero VAD needs: pip install 'jarvis[stt]'") from exc
        self._model = load_silero_vad(onnx=True)

    def is_speech(
        self, frame: bytes, sample_rate: int = 16_000
    ) -> bool:  # pragma: no cover - needs the model
        import torch  # optional dependency, pulled in by silero-vad

        samples = array.array("h")
        samples.frombytes(frame[: len(frame) - (len(frame) % 2)])
        if sys.byteorder == "big":
            samples.byteswap()
        tensor = torch.tensor([s / INT16_MAX for s in samples], dtype=torch.float32)
        return float(self._model(tensor, sample_rate).item()) >= self.threshold

    def reset(self) -> None:  # pragma: no cover - needs the model
        reset = getattr(self._model, "reset_states", None)
        if reset is not None:
            reset()


def build_vad(name: str, threshold: float = 0.5) -> VoiceActivityDetector:
    """Build a detector, degrading to :class:`EnergyVAD` when Silero is unavailable."""
    if name in {"energy", "", "null"}:
        return EnergyVAD()
    if name == "always":
        return AlwaysOnVAD()
    if name == "silero":
        try:
            return SileroVAD(threshold=threshold)
        except RuntimeError:
            return EnergyVAD()
    raise ValueError(f"unknown VAD backend: {name}")
