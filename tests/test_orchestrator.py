"""The rules that only exist once the subsystems are wired together."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pytest

from conftest import make_frame
from jarvis.core.agent import Agent
from jarvis.core.config import Config
from jarvis.core.events import EventBus
from jarvis.core.llm import Completion, EchoBackend, ScriptedBackend
from jarvis.core.orchestrator import Jarvis
from jarvis.core.state_machine import ProactiveGate, State, StateMachine
from jarvis.memory.store import MemoryStore
from jarvis.perception.change import ChangeDetector
from jarvis.perception.screen import SyntheticCapture
from jarvis.perception.vision import ScreenWatcher
from jarvis.voice.stt import NullSTT
from jarvis.voice.tts import NullTTS
from jarvis.voice.wakeword import NullWakeWord, TextWakeWord


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def build(
    config: Config,
    agent_replies=("ok",),
    vision_replies=("a terminal",),
    frames=None,
    clock: FakeClock | None = None,
    wake_word=None,
) -> tuple[Jarvis, NullTTS, FakeClock]:
    clock = clock or FakeClock()
    bus = EventBus()
    memory = MemoryStore(":memory:")
    frames = frames or [make_frame(seed=1), make_frame(seed=2)]
    watcher = ScreenWatcher(
        config.vision,
        SyntheticCapture(frames),
        ScriptedBackend(list(vision_replies), fallback="a terminal"),
        memory,
        bus,
        detector=ChangeDetector(threshold=config.vision.change_threshold, max_interval_s=None),
        title_provider=lambda: "Terminal",
    )
    agent = Agent(
        config.agent,
        ScriptedBackend(list(agent_replies), fallback="ok"),
        memory,
        bus,
        screen_describer=watcher.describe_now,
    )
    tts = NullTTS()
    jarvis = Jarvis(
        config=config,
        memory=memory,
        agent=agent,
        watcher=watcher,
        tts=tts,
        stt=NullSTT(),
        wake_word=wake_word or TextWakeWord(config.voice.wake_word),
        bus=bus,
        state=StateMachine(bus, clock=clock),
        gate=ProactiveGate(
            config.proactive, clock=clock, wall_clock=lambda: datetime(2026, 9, 16, 14, 0)
        ),
        clock=clock,
    )
    return jarvis, tts, clock


# -- conversation ----------------------------------------------------------


def test_a_turn_runs_the_fsm_and_ends_idle(config: Config) -> None:
    jarvis, tts, _ = build(config, agent_replies=("Noted.",))
    states: list[str] = []
    jarvis.bus.subscribe(lambda e: states.append(e.payload["state"]) if e.kind == "state" else None)
    assert jarvis.handle_text("hello") == "Noted."
    assert states == ["thinking", "speaking", "idle"]
    assert tts.spoken == ["Noted."]
    assert jarvis.state.state is State.IDLE
    jarvis.close()


def test_empty_input_is_ignored(config: Config) -> None:
    jarvis, tts, _ = build(config)
    assert jarvis.handle_text("   ") == ""
    assert tts.spoken == []
    jarvis.close()


def test_speak_false_still_answers(config: Config) -> None:
    jarvis, tts, _ = build(config, agent_replies=("quiet answer",))
    assert jarvis.handle_text("hi", speak=False) == "quiet answer"
    assert tts.spoken == []
    jarvis.close()


def test_a_paused_jarvis_says_nothing(config: Config) -> None:
    jarvis, tts, _ = build(config)
    jarvis.pause()
    assert jarvis.handle_text("hello") == ""
    assert tts.spoken == []
    jarvis.resume()
    assert jarvis.handle_text("hello") == "ok"
    jarvis.close()


def test_an_agent_failure_leaves_the_fsm_recovered(config: Config) -> None:
    class BrokenBackend:
        name = "broken"

        def complete(self, *args: Any, **kwargs: Any) -> Completion:
            raise RuntimeError("upstream is down")

    jarvis, _, _ = build(config)
    jarvis.agent.backend = BrokenBackend()
    with pytest.raises(RuntimeError):
        jarvis.handle_text("hello")
    assert jarvis.state.state is State.IDLE
    jarvis.close()


def test_a_tts_failure_does_not_lose_the_reply(config: Config) -> None:
    class BrokenTTS:
        name, available = "broken", True

        def speak(self, text: str) -> bytes:
            raise OSError("no audio device")

        def stop(self) -> None: ...

    jarvis, _, _ = build(config, agent_replies=("still answered",))
    jarvis.tts = BrokenTTS()
    assert jarvis.handle_text("hello") == "still answered"
    assert jarvis.state.state is State.IDLE
    jarvis.close()


# -- wake word -------------------------------------------------------------


def test_utterances_without_the_wake_word_are_ignored(config: Config) -> None:
    jarvis, tts, _ = build(config)
    assert jarvis.handle_utterance("just muttering to myself") is None
    assert tts.spoken == []
    jarvis.close()


def test_the_wake_word_is_stripped_before_reasoning(config: Config) -> None:
    jarvis, _, _ = build(config, agent_replies=("2pm.",))
    assert jarvis.handle_utterance("Hey Jarvis, what time is the meeting?") == "2pm."
    assert jarvis.memory.recent_turns()[0].content == "what time is the meeting?"
    jarvis.close()


def test_a_bare_wake_word_still_reaches_the_agent(config: Config) -> None:
    jarvis, _, _ = build(config, agent_replies=("Yes?",))
    assert jarvis.handle_utterance("hey jarvis") == "Yes?"
    jarvis.close()


def test_jarvis_does_not_wake_itself_while_speaking(config: Config) -> None:
    jarvis, _, _ = build(config)
    jarvis.state.to(State.THINKING)
    jarvis.state.to(State.SPEAKING)
    assert jarvis.handle_utterance("hey jarvis, are you there?") is None
    jarvis.close()


def test_suppression_can_be_switched_off_for_barge_in(config: Config) -> None:
    config.voice.suppress_while_speaking = False
    jarvis, _, _ = build(config, agent_replies=("interrupted",))
    jarvis.state.to(State.THINKING)
    jarvis.state.to(State.SPEAKING)
    assert jarvis.handle_utterance("hey jarvis, stop") == "interrupted"
    jarvis.close()


def test_a_null_wake_word_means_always_listening(config: Config) -> None:
    jarvis, _, _ = build(config, agent_replies=("heard you",), wake_word=NullWakeWord())
    assert jarvis.handle_utterance("no wake word here") == "heard you"
    jarvis.close()


# -- proactive speech ------------------------------------------------------


def proactive_config(config: Config) -> Config:
    config.vision.enabled = True
    config.vision.max_edge_px = 64
    config.proactive.enabled = True
    config.proactive.idle_threshold_s = 30.0
    config.proactive.cooldown_s = 300.0
    return config


def test_an_unremarkable_frame_never_interrupts(config: Config) -> None:
    jarvis, tts, clock = build(proactive_config(config), vision_replies=("a terminal",))
    clock.advance(60)
    jarvis.vision_tick()
    assert tts.spoken == []
    jarvis.close()


def test_a_noteworthy_frame_speaks_once_the_user_is_idle(config: Config) -> None:
    jarvis, tts, clock = build(
        proactive_config(config),
        agent_replies=("Your build just went red.",),
        vision_replies=("NOTE: the build failed",),
    )
    clock.advance(60)
    jarvis.vision_tick()
    assert tts.spoken == ["Your build just went red."]
    assert jarvis.memory.recent_turns()[-1].content == "Your build just went red."
    assert jarvis.state.state is State.IDLE
    jarvis.close()


def test_a_busy_user_is_not_interrupted(config: Config) -> None:
    jarvis, tts, clock = build(
        proactive_config(config),
        agent_replies=("would have said something",),
        vision_replies=("NOTE: the build failed",),
    )
    clock.advance(60)
    jarvis.gate.note_user_activity()  # they just typed
    reasons: list[str] = []
    jarvis.bus.subscribe(
        lambda e: reasons.append(e.payload.get("reason", "")) if e.kind == "proactive" else None
    )
    jarvis.vision_tick()
    assert tts.spoken == []
    assert any("idle" in reason for reason in reasons)
    jarvis.close()


def test_the_model_can_decline_to_speak(config: Config) -> None:
    jarvis, tts, clock = build(
        proactive_config(config),
        agent_replies=("SILENCE",),
        vision_replies=("NOTE: a notification appeared",),
    )
    clock.advance(60)
    jarvis.vision_tick()
    assert tts.spoken == []
    assert jarvis.state.state is State.IDLE
    jarvis.close()


def test_the_cooldown_limits_proactive_chatter(config: Config) -> None:
    jarvis, tts, clock = build(
        proactive_config(config),
        agent_replies=("first remark", "second remark"),
        vision_replies=("NOTE: one", "NOTE: two"),
        frames=[make_frame(seed=1), make_frame(seed=2)],
    )
    clock.advance(60)
    jarvis.vision_tick()
    clock.advance(60)  # inside the 300s cooldown
    jarvis.vision_tick()
    assert tts.spoken == ["first remark"]
    jarvis.close()


def test_proactive_remarks_are_skipped_while_paused(config: Config) -> None:
    jarvis, tts, clock = build(
        proactive_config(config),
        agent_replies=("would have spoken",),
        vision_replies=("NOTE: something",),
    )
    clock.advance(60)
    jarvis.pause()
    jarvis.vision_tick()
    assert tts.spoken == []
    jarvis.close()


def test_a_proactive_agent_failure_is_contained(config: Config) -> None:
    class BrokenBackend:
        name = "broken"

        def complete(self, *args: Any, **kwargs: Any) -> Completion:
            raise RuntimeError("upstream is down")

    jarvis, tts, clock = build(proactive_config(config), vision_replies=("NOTE: something",))
    jarvis.agent.backend = BrokenBackend()
    clock.advance(60)
    jarvis.vision_tick()  # must not raise
    assert tts.spoken == []
    assert jarvis.state.state is State.IDLE
    jarvis.close()


# -- lifecycle -------------------------------------------------------------


def test_status_reports_what_the_orb_needs(config: Config) -> None:
    jarvis, _, clock = build(proactive_config(config), agent_replies=("hi",))
    jarvis.handle_text("hello")
    clock.advance(5)
    status = jarvis.status_dict()
    assert status["state"] == "idle"
    assert status["turns"] == 2
    assert status["vision_enabled"] is True
    assert status["idle_for"] == pytest.approx(5.0)
    jarvis.close()


def test_from_config_builds_a_working_offline_stack(config: Config) -> None:
    config.agent.backend = "echo"
    with Jarvis.from_config(config, capture=SyntheticCapture([make_frame(seed=1)])) as jarvis:
        assert jarvis.handle_text("hello") == "[echo] hello"
        assert isinstance(jarvis.agent.backend, EchoBackend)
        assert jarvis.vision_tick().status == "disabled"  # vision stays off by default


def test_start_and_stop_are_safe_when_vision_is_off(config: Config) -> None:
    jarvis, _, _ = build(config)
    jarvis.start()
    jarvis.stop()
    jarvis.close()
