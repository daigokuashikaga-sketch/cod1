from __future__ import annotations

from datetime import datetime

import pytest

from jarvis.core.config import ProactiveConfig
from jarvis.core.events import EventBus
from jarvis.core.state_machine import (
    InvalidTransition,
    ProactiveGate,
    State,
    StateMachine,
)


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def test_happy_path_turn() -> None:
    sm = StateMachine()
    assert sm.state is State.IDLE
    sm.to(State.LISTENING)
    sm.to(State.THINKING)
    sm.to(State.SPEAKING)
    sm.to(State.IDLE)
    assert sm.state is State.IDLE


def test_illegal_transition_raises_and_try_to_does_not() -> None:
    sm = StateMachine()
    with pytest.raises(InvalidTransition):
        sm.to(State.SPEAKING)  # cannot speak without thinking first
    assert sm.state is State.IDLE
    assert sm.try_to(State.SPEAKING) is False
    assert sm.state is State.IDLE


def test_transition_to_same_state_is_a_noop() -> None:
    sm = StateMachine()
    assert sm.to(State.IDLE) is State.IDLE


def test_barge_in_moves_speaking_to_listening() -> None:
    sm = StateMachine()
    sm.to(State.THINKING)
    sm.to(State.SPEAKING)
    assert sm.try_to(State.LISTENING, reason="barge-in") is True


def test_cancel_from_any_busy_state() -> None:
    for busy in (State.LISTENING, State.THINKING, State.SPEAKING):
        sm = StateMachine()
        sm.to(State.THINKING) if busy is not State.LISTENING else None
        sm.try_to(busy)
        assert sm.cancel() is True
        assert sm.state is State.CANCELLED
        assert sm.try_to(State.IDLE) is True


def test_transitions_publish_events() -> None:
    bus = EventBus()
    seen: list[dict] = []
    bus.subscribe(lambda event: seen.append(event.to_dict()))
    sm = StateMachine(bus)
    sm.to(State.THINKING, reason="user turn")
    assert seen[-1]["state"] == "thinking"
    assert seen[-1]["previous"] == "idle"
    assert seen[-1]["reason"] == "user turn"


def test_time_in_state_uses_the_injected_clock() -> None:
    clock = FakeClock()
    sm = StateMachine(clock=clock)
    clock.advance(12.0)
    assert sm.time_in_state() == pytest.approx(12.0)
    sm.to(State.THINKING)
    assert sm.time_in_state() == pytest.approx(0.0)


def gate(clock: FakeClock, **kwargs: object) -> ProactiveGate:
    defaults = {
        "enabled": True,
        "idle_threshold_s": 30.0,
        "cooldown_s": 300.0,
        "max_per_hour": 6,
    }
    defaults.update(kwargs)
    return ProactiveGate(
        ProactiveConfig(**defaults),
        clock=clock,
        wall_clock=lambda: datetime(2026, 9, 16, 14, 0),
    )


def test_gate_is_closed_until_the_idle_threshold() -> None:
    clock = FakeClock()
    g = gate(clock)
    assert not g.check(State.IDLE)
    clock.advance(31)
    assert g.check(State.IDLE)


def test_gate_only_opens_from_idle() -> None:
    clock = FakeClock()
    g = gate(clock)
    clock.advance(60)
    for busy in (State.LISTENING, State.THINKING, State.SPEAKING, State.PAUSED):
        assert not g.check(busy)
    assert g.check(State.IDLE)


def test_user_activity_resets_the_idle_timer() -> None:
    clock = FakeClock()
    g = gate(clock)
    clock.advance(60)
    g.note_user_activity()
    assert not g.check(State.IDLE)
    assert "idle" in g.check(State.IDLE).reason


def test_cooldown_blocks_a_second_remark() -> None:
    clock = FakeClock()
    g = gate(clock)
    clock.advance(60)
    assert g.check(State.IDLE)
    g.note_spoke()
    clock.advance(60)
    decision = g.check(State.IDLE)
    assert not decision and "cooldown" in decision.reason
    clock.advance(300)
    assert g.check(State.IDLE)


def test_hourly_cap_is_enforced_even_after_the_cooldown() -> None:
    clock = FakeClock()
    g = gate(clock, cooldown_s=1.0, max_per_hour=3)
    clock.advance(60)
    for _ in range(3):
        assert g.check(State.IDLE)
        g.note_spoke()
        clock.advance(10)
    decision = g.check(State.IDLE)
    assert not decision and "hourly cap" in decision.reason
    clock.advance(3600)  # the window rolls forward
    assert g.check(State.IDLE)


def test_quiet_hours_wrap_past_midnight() -> None:
    clock = FakeClock()
    night = ProactiveGate(
        ProactiveConfig(enabled=True, idle_threshold_s=0.0, quiet_hours=(22, 8)),
        clock=clock,
        wall_clock=lambda: datetime(2026, 9, 16, 23, 30),
    )
    assert not night.check(State.IDLE)

    morning = ProactiveGate(
        ProactiveConfig(enabled=True, idle_threshold_s=0.0, quiet_hours=(22, 8)),
        clock=clock,
        wall_clock=lambda: datetime(2026, 9, 16, 9, 0),
    )
    assert morning.check(State.IDLE)


def test_disabled_gate_never_opens() -> None:
    clock = FakeClock()
    g = gate(clock, enabled=False)
    clock.advance(10_000)
    assert not g.check(State.IDLE)
