"""Text to speech.

Kokoro-82M is the default because it is Apache-2.0, runs on CPU and costs
nothing; ElevenLabs is there for when expressiveness matters more than money.
Both are optional -- the null backend just records what would have been said,
which is exactly what the tests and headless runs need.
"""

from __future__ import annotations

import os
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class TTSBackend(Protocol):
    name: str
    available: bool

    def speak(self, text: str) -> bytes: ...

    def stop(self) -> None: ...


class NullTTS:
    """Silent. Keeps a transcript of everything it was asked to say."""

    name = "null"
    available = True

    def __init__(self) -> None:
        self.spoken: list[str] = []

    def speak(self, text: str) -> bytes:
        self.spoken.append(text)
        return b""

    def stop(self) -> None:
        return None


class KokoroTTS:
    """Local neural TTS. ~300MB of weights, no API key, no per-character bill."""

    name = "kokoro"

    def __init__(self, voice: str = "af_heart", pipeline: Any | None = None) -> None:
        self.voice = voice
        if pipeline is not None:
            self._pipeline = pipeline
            self.available = True
            return
        try:
            from kokoro import KPipeline  # optional dependency
        except ImportError as exc:  # pragma: no cover - depends on the host
            raise RuntimeError("Kokoro TTS needs: pip install 'jarvis[tts]'") from exc
        self._pipeline = KPipeline(lang_code="a")
        self.available = True

    def speak(self, text: str) -> bytes:  # pragma: no cover - needs the model
        chunks = [
            audio.tobytes() if hasattr(audio, "tobytes") else bytes(audio)
            for _, _, audio in self._pipeline(text, voice=self.voice)
        ]
        return b"".join(chunks)

    def stop(self) -> None:  # pragma: no cover - needs the model
        return None


class ElevenLabsTTS:
    """Hosted premium TTS. Bills per character -- keep it for the reply, not the idle chatter."""

    name = "elevenlabs"

    def __init__(
        self, voice: str = "Rachel", api_key: str | None = None, client: Any | None = None
    ) -> None:
        self.voice = voice
        if client is not None:
            self._client = client
            self.available = True
            return
        key = api_key or os.environ.get("ELEVENLABS_API_KEY")
        if not key:
            raise RuntimeError("ELEVENLABS_API_KEY is not set")
        try:
            from elevenlabs.client import ElevenLabs  # optional dependency
        except ImportError as exc:  # pragma: no cover - depends on the host
            raise RuntimeError("ElevenLabs TTS needs: pip install elevenlabs") from exc
        self._client = ElevenLabs(api_key=key)
        self.available = True

    def speak(self, text: str) -> bytes:  # pragma: no cover - needs the API
        audio = self._client.text_to_speech.convert(
            voice_id=self.voice,
            model_id="eleven_flash_v2_5",
            text=text,
            # Raw PCM, so AudioPlayer can stream and interrupt it; mp3 could not be.
            output_format="pcm_24000",
        )
        return b"".join(audio) if not isinstance(audio, bytes) else audio

    def stop(self) -> None:  # pragma: no cover - needs the API
        return None


def build_tts(name: str, voice: str = "af_heart") -> TTSBackend:
    if name in {"null", "none", ""}:
        return NullTTS()
    if name == "kokoro":
        try:
            return KokoroTTS(voice=voice)
        except RuntimeError:
            return NullTTS()
    if name == "elevenlabs":
        try:
            return ElevenLabsTTS(voice=voice)
        except RuntimeError:
            return NullTTS()
    raise ValueError(f"unknown TTS backend: {name}")
