"""Voice: speech-to-text, text-to-speech, and wake-word detection.

Every backend here is optional and behind a protocol. The ``null`` backends are
real implementations of "do nothing", so a headless install still runs.
"""

from .stt import STTBackend, Transcript, build_stt
from .tts import TTSBackend, build_tts
from .wakeword import WakeWordDetector, build_wake_word

__all__ = [
    "STTBackend",
    "TTSBackend",
    "Transcript",
    "WakeWordDetector",
    "build_stt",
    "build_tts",
    "build_wake_word",
]
