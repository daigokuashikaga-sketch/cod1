"""Voice: speech-to-text, text-to-speech, and wake-word detection.

Every backend here is optional and behind a protocol. The ``null`` backends are
real implementations of "do nothing", so a headless install still runs.
"""

from .audio import AudioSource, VoiceListener, build_audio_source
from .playback import AudioPlayer, build_player
from .stt import STTBackend, Transcript, build_stt
from .tts import TTSBackend, build_tts
from .vad import VoiceActivityDetector, build_vad
from .wakeword import WakeWordDetector, build_wake_word

__all__ = [
    "AudioPlayer",
    "AudioSource",
    "STTBackend",
    "TTSBackend",
    "Transcript",
    "VoiceActivityDetector",
    "VoiceListener",
    "WakeWordDetector",
    "build_audio_source",
    "build_player",
    "build_stt",
    "build_tts",
    "build_vad",
    "build_wake_word",
]
