"""The microphone pump: VAD, utterance segmentation, and interruptible playback."""

from __future__ import annotations

import threading
from typing import Any

import pytest

from conftest import audio_silence, audio_speech
from jarvis.core.events import EventBus
from jarvis.voice.audio import (
    ListenerSettings,
    NullAudioSource,
    SyntheticAudioSource,
    VoiceListener,
    build_audio_source,
)
from jarvis.voice.playback import SoundDevicePlayer, build_player
from jarvis.voice.stt import ScriptedSTT, Transcript
from jarvis.voice.vad import AlwaysOnVAD, EnergyVAD, build_vad, rms

SETTINGS = ListenerSettings(
    sample_rate=16_000,
    frame_ms=30,
    start_frames=3,
    pre_roll_ms=120,
    silence_hangover_ms=300,
    min_speech_ms=180,
    max_utterance_s=1.5,
)


silence = audio_silence
speech = audio_speech


class RecordingSTT:
    """Returns a fixed transcript and remembers exactly what audio it was given."""

    name, available = "recording", True

    def __init__(self, text: str = "hello") -> None:
        self.text = text
        self.audio: list[bytes] = []

    def transcribe(self, audio: bytes, sample_rate: int = 16_000) -> Transcript:
        self.audio.append(audio)
        return Transcript(self.text)


def listener(frames, stt=None, bus=None, settings=SETTINGS, **kwargs: Any) -> VoiceListener:
    return VoiceListener(
        SyntheticAudioSource(frames),
        EnergyVAD(),
        stt or ScriptedSTT(["first", "second", "third"]),
        bus=bus,
        settings=settings,
        **kwargs,
    )


def drain(pump: VoiceListener) -> list[Transcript]:
    heard: list[Transcript] = []
    while not pump.exhausted:
        transcript = pump.poll_once()
        if transcript is not None:
            heard.append(transcript)
    return heard


# -- voice activity detection ---------------------------------------------


def test_rms_of_silence_is_zero_and_of_speech_is_not() -> None:
    assert rms(silence()[0]) == 0.0
    assert rms(speech()[0]) > 0.2
    assert rms(b"") == 0.0


def test_energy_vad_separates_speech_from_room_tone() -> None:
    vad = EnergyVAD()
    assert vad.is_speech(silence()[0]) is False
    assert vad.is_speech(speech()[0]) is True


def test_the_noise_floor_adapts_to_a_noisy_room() -> None:
    vad = EnergyVAD()
    hiss = speech(1, level=900)[0]  # a fan, not a voice
    assert vad.is_speech(hiss) is True  # at first it looks like speech
    for _ in range(200):  # ~6 seconds of it
        vad.is_speech(hiss)
    assert vad.is_speech(hiss) is False  # the floor has risen to meet it
    assert vad.is_speech(speech()[0]) is True  # real speech still gets through


def test_a_normal_utterance_does_not_raise_the_floor_over_itself() -> None:
    vad = EnergyVAD()
    voice = speech(1, level=6000)[0]
    for _ in range(70):  # ~2 seconds of continuous talking
        assert vad.is_speech(voice) is True


def test_reset_restores_the_initial_floor() -> None:
    vad = EnergyVAD()
    for _ in range(200):
        vad.is_speech(speech(1, level=900)[0])
    raised = vad.noise_floor
    vad.reset()
    assert vad.noise_floor < raised


def test_vad_backends_build_and_degrade() -> None:
    assert isinstance(build_vad("energy"), EnergyVAD)
    assert isinstance(build_vad("always"), AlwaysOnVAD)
    assert isinstance(build_vad("silero"), EnergyVAD | type(build_vad("silero")))
    with pytest.raises(ValueError, match="unknown VAD backend"):
        build_vad("telepathy")


# -- segmentation ----------------------------------------------------------


def test_a_single_utterance_is_transcribed_once() -> None:
    stt = RecordingSTT("hey jarvis")
    heard = drain(listener(silence(5) + speech(15) + silence(15), stt=stt))
    assert [t.text for t in heard] == ["hey jarvis"]
    assert len(stt.audio) == 1  # one transcription for the whole utterance


def test_two_utterances_are_separated_by_silence() -> None:
    frames = silence(5) + speech(12) + silence(15) + speech(12) + silence(15)
    assert [t.text for t in drain(listener(frames))] == ["first", "second"]


def test_a_pause_mid_sentence_does_not_split_the_utterance() -> None:
    # 5 frames of quiet = 150ms, inside the 300ms hangover.
    frames = silence(5) + speech(10) + silence(5) + speech(10) + silence(15)
    assert [t.text for t in drain(listener(frames))] == ["first"]


def test_a_brief_noise_burst_is_discarded() -> None:
    stt = RecordingSTT()
    bus = EventBus()
    seen: list[dict] = []
    bus.subscribe(lambda e: seen.append(e.to_dict()))
    # 4 speech frames = 120ms, under the 180ms minimum.
    assert drain(listener(silence(5) + speech(4) + silence(15), stt=stt, bus=bus)) == []
    assert stt.audio == []  # never reached the transcriber
    assert any(event.get("status") == "discarded" for event in seen)


def test_one_stray_frame_never_opens_an_utterance() -> None:
    stt = RecordingSTT()
    assert drain(listener(silence(5) + speech(1) + silence(20), stt=stt)) == []
    assert stt.audio == []


def test_pre_roll_keeps_the_start_of_the_word() -> None:
    stt = RecordingSTT()
    drain(listener(silence(10) + speech(15) + silence(15), stt=stt))
    frames_sent = len(stt.audio[0]) // SETTINGS.frame_bytes
    # 15 speech frames + hangover + up to 4 frames of pre-roll.
    assert frames_sent > 15 + SETTINGS.hangover_frames


def test_a_long_monologue_is_cut_off() -> None:
    settings = ListenerSettings(frame_ms=30, start_frames=3, max_utterance_s=0.3)
    heard = drain(listener(silence(2) + speech(60), settings=settings))
    assert heard  # cut and transcribed rather than buffered forever
    assert len(heard[0].text) > 0


def test_an_open_utterance_is_flushed_when_the_stream_ends() -> None:
    assert [t.text for t in drain(listener(silence(3) + speech(12)))] == ["first"]


def test_an_empty_transcription_yields_nothing() -> None:
    bus = EventBus()
    seen: list[dict] = []
    bus.subscribe(lambda e: seen.append(e.to_dict()))
    heard = drain(listener(silence(5) + speech(15) + silence(15), stt=ScriptedSTT([""]), bus=bus))
    assert heard == []
    assert any(event.get("status") == "empty" for event in seen)


def test_duration_is_filled_in_when_the_backend_omits_it() -> None:
    heard = drain(listener(silence(5) + speech(15) + silence(15), stt=RecordingSTT()))
    assert heard[0].duration_s > 0.3


def test_speech_start_fires_once_per_utterance() -> None:
    starts: list[int] = []
    frames = silence(5) + speech(12) + silence(15) + speech(12) + silence(15)
    drain(listener(frames, on_speech_start=lambda: starts.append(1)))
    assert len(starts) == 2


def test_capturing_reports_whether_the_user_is_mid_sentence() -> None:
    pump = listener(silence(5) + speech(15) + silence(15))
    for _ in range(8):
        pump.poll_once()
    assert pump.capturing is True
    drain(pump)
    assert pump.capturing is False


def test_the_background_loop_delivers_utterances_and_stops() -> None:
    heard: list[str] = []
    pump = listener(
        silence(5) + speech(15) + silence(15), on_utterance=lambda t: heard.append(t.text)
    )
    pump.start()
    for _ in range(200):
        if not pump.running:
            break
        threading.Event().wait(0.01)
    pump.stop()
    assert heard == ["first"]
    assert pump.source.closed is True


def test_a_failing_source_ends_the_loop_with_an_event() -> None:
    class BrokenSource:
        name, sample_rate = "broken", 16_000

        def read(self, size: int) -> bytes:
            raise OSError("device disappeared")

        def close(self) -> None: ...

    bus = EventBus()
    seen: list[dict] = []
    bus.subscribe(lambda e: seen.append(e.to_dict()))
    pump = VoiceListener(BrokenSource(), EnergyVAD(), ScriptedSTT(["x"]), bus=bus)
    pump.run()  # must return rather than raise
    assert any(event.get("status") == "error" for event in seen)


def test_null_source_means_no_audio() -> None:
    assert isinstance(build_audio_source("null"), NullAudioSource)
    assert build_audio_source("null").read(960) is None
    with pytest.raises(ValueError, match="unknown audio source"):
        build_audio_source("telepathy")


# -- playback --------------------------------------------------------------


def test_null_player_records_instead_of_playing() -> None:
    player = build_player("null")
    player.play(b"\x00\x01")
    player.stop()
    assert player.played == [b"\x00\x01"]
    assert player.stops == 1
    assert player.is_playing is False


def test_unknown_players_are_rejected() -> None:
    with pytest.raises(ValueError, match="unknown audio player"):
        build_player("gramophone")


class FakeSoundDevice:
    """Minimal stand-in for the sounddevice module."""

    def __init__(self, on_write=None) -> None:
        self.on_write = on_write
        self.streams: list[FakeStream] = []

    def RawOutputStream(self, **kwargs: Any) -> FakeStream:  # mirrors the real sounddevice API
        stream = FakeStream(self.on_write)
        self.streams.append(stream)
        return stream


class FakeStream:
    def __init__(self, on_write=None) -> None:
        self.writes: list[bytes] = []
        self.closed = threading.Event()
        self._on_write = on_write

    def start(self) -> None: ...

    def write(self, data: bytes) -> None:
        self.writes.append(data)
        if self._on_write is not None:
            self._on_write(len(self.writes))

    def stop(self) -> None: ...

    def close(self) -> None:
        self.closed.set()


def test_playback_streams_pcm_in_chunks() -> None:
    fake = FakeSoundDevice()
    player = SoundDevicePlayer(module=fake)
    player.play(b"\x00\x01" * 4096)
    assert fake.streams[0].closed.wait(5)
    assert len(fake.streams[0].writes) == 4  # 4096 frames / 1024 per chunk
    assert b"".join(fake.streams[0].writes) == b"\x00\x01" * 4096


def test_stop_cuts_playback_mid_utterance() -> None:
    player: SoundDevicePlayer | None = None

    def interrupt(written: int) -> None:
        if written == 2:
            player.stop()  # the barge-in path, from inside the playback thread

    fake = FakeSoundDevice(on_write=interrupt)
    player = SoundDevicePlayer(module=fake)
    player.play(b"\x00\x01" * 40_960)
    assert fake.streams[0].closed.wait(5)
    assert len(fake.streams[0].writes) == 2  # the rest was never written
    assert player.is_playing is False


def test_playing_nothing_is_a_noop() -> None:
    fake = FakeSoundDevice()
    SoundDevicePlayer(module=fake).play(b"")
    assert fake.streams == []


def test_a_new_utterance_replaces_the_previous_one() -> None:
    fake = FakeSoundDevice()
    player = SoundDevicePlayer(module=fake)
    player.play(b"\x00\x01" * 1024)
    assert fake.streams[0].closed.wait(5)
    player.play(b"\x02\x03" * 1024)
    assert fake.streams[1].closed.wait(5)
    assert len(fake.streams) == 2
