from __future__ import annotations

import pytest

from jarvis.voice.stt import FasterWhisperSTT, NullSTT, ScriptedSTT, Transcript, build_stt
from jarvis.voice.tts import KokoroTTS, NullTTS, build_tts
from jarvis.voice.wakeword import NullWakeWord, TextWakeWord, build_wake_word


def test_transcript_is_falsy_when_empty() -> None:
    assert not Transcript("")
    assert not Transcript("   ")
    assert Transcript("hello")


def test_scripted_stt_drains_its_queue() -> None:
    stt = ScriptedSTT(["one", "two"])
    assert stt.transcribe(b"").text == "one"
    assert stt.transcribe(b"").text == "two"
    assert stt.transcribe(b"").text == ""


def test_null_backends_are_usable_everywhere() -> None:
    assert build_stt("null").transcribe(b"").text == ""
    tts = build_tts("null")
    tts.speak("hello")
    tts.stop()
    assert tts.spoken == ["hello"]


def test_missing_optional_backends_degrade_instead_of_crashing() -> None:
    # faster-whisper and kokoro are optional; without them we fall back to null
    # rather than refusing to start.
    assert isinstance(build_stt("faster_whisper"), NullSTT | FasterWhisperSTT)
    assert isinstance(build_tts("kokoro"), NullTTS | KokoroTTS)


def test_unknown_backends_are_rejected() -> None:
    with pytest.raises(ValueError, match="unknown STT backend"):
        build_stt("telepathy")
    with pytest.raises(ValueError, match="unknown TTS backend"):
        build_tts("telepathy")
    with pytest.raises(ValueError, match="unknown wake-word backend"):
        build_wake_word("hey jarvis", backend="telepathy")


@pytest.mark.parametrize(
    "utterance, expected",
    [
        ("hey jarvis what time is it", "what time is it"),
        ("Hey Jarvis, what time is it?", "what time is it?"),
        ("HEY   JARVIS: open the logs", "open the logs"),
        ("okay, hey jarvis - stop", "okay, stop"),
    ],
)
def test_wake_word_matching_is_forgiving(utterance: str, expected: str) -> None:
    detector = TextWakeWord("hey jarvis")
    assert detector.matches(utterance)
    assert detector.strip_phrase(utterance) == expected


@pytest.mark.parametrize("utterance", ["hey there", "jarvis", "heyjarvis", ""])
def test_near_misses_do_not_wake_it(utterance: str) -> None:
    assert not TextWakeWord("hey jarvis").matches(utterance)


def test_a_custom_phrase_works() -> None:
    detector = TextWakeWord("computer")
    assert detector.matches("Computer, status report")
    assert detector.strip_phrase("Computer, status report") == "status report"


def test_disabled_wake_word_returns_the_always_on_detector() -> None:
    assert isinstance(build_wake_word("hey jarvis", enabled=False), NullWakeWord)
    assert build_wake_word("hey jarvis", enabled=False).detect(b"") == 1.0
