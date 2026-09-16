"""The finite state machine that decides what Jarvis is doing, and whether it may speak.

Two separable concerns live here:

``StateMachine``   -- legal transitions between IDLE/LISTENING/THINKING/SPEAKING/...
``ProactiveGate``  -- whether an unprompted remark is allowed *right now*.

Proactive speech is only ever permitted from IDLE, after an idleness threshold,
subject to a cooldown, an hourly cap and quiet hours. That combination is what
keeps a screen-watching companion from becoming a nuisance.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from .config import ProactiveConfig
from .events import EventBus


class State(StrEnum):
    IDLE = "idle"
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"
    CANCELLED = "cancelled"
    ERROR = "error"
    PAUSED = "paused"


# Explicit transition table: anything not listed is a bug, not a surprise.
TRANSITIONS: dict[State, frozenset[State]] = {
    State.IDLE: frozenset({State.LISTENING, State.THINKING, State.PAUSED, State.ERROR}),
    State.LISTENING: frozenset(
        {State.THINKING, State.IDLE, State.CANCELLED, State.PAUSED, State.ERROR}
    ),
    State.THINKING: frozenset(
        {State.SPEAKING, State.IDLE, State.CANCELLED, State.PAUSED, State.ERROR}
    ),
    # SPEAKING -> LISTENING is barge-in: the user talks over Jarvis.
    State.SPEAKING: frozenset(
        {State.IDLE, State.LISTENING, State.CANCELLED, State.PAUSED, State.ERROR}
    ),
    State.CANCELLED: frozenset({State.IDLE, State.LISTENING, State.PAUSED, State.ERROR}),
    State.ERROR: frozenset({State.IDLE, State.PAUSED}),
    State.PAUSED: frozenset({State.IDLE}),
}

# States during which an unprompted remark would interrupt something.
BUSY_STATES = frozenset({State.LISTENING, State.THINKING, State.SPEAKING})


class InvalidTransition(RuntimeError):
    def __init__(self, source: State, target: State) -> None:
        super().__init__(f"illegal transition {source.value} -> {target.value}")
        self.source = source
        self.target = target


class StateMachine:
    def __init__(
        self,
        bus: EventBus | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._state = State.IDLE
        self._bus = bus
        self._clock = clock
        self._since = clock()
        self._lock = threading.RLock()

    @property
    def state(self) -> State:
        with self._lock:
            return self._state

    @property
    def is_busy(self) -> bool:
        return self.state in BUSY_STATES

    def time_in_state(self) -> float:
        with self._lock:
            return self._clock() - self._since

    def can(self, target: State) -> bool:
        with self._lock:
            return target is self._state or target in TRANSITIONS[self._state]

    def to(self, target: State, reason: str = "") -> State:
        """Transition, raising :class:`InvalidTransition` if the move is not legal."""
        with self._lock:
            source = self._state
            if target is source:
                return source
            if target not in TRANSITIONS[source]:
                raise InvalidTransition(source, target)
            self._state = target
            self._since = self._clock()
        if self._bus is not None:
            self._bus.publish("state", state=target.value, previous=source.value, reason=reason)
        return target

    def try_to(self, target: State, reason: str = "") -> bool:
        """Transition if legal; return False instead of raising when it is not."""
        try:
            self.to(target, reason=reason)
        except InvalidTransition:
            return False
        return True

    def cancel(self, reason: str = "user interrupt") -> bool:
        """Abort the current turn. Used for barge-in and for Ctrl-C."""
        return self.try_to(State.CANCELLED, reason=reason)


@dataclass(frozen=True)
class GateDecision:
    allowed: bool
    reason: str

    def __bool__(self) -> bool:  # lets callers write `if gate.check(...):`
        return self.allowed


class ProactiveGate:
    """Decides whether Jarvis may say something nobody asked for."""

    def __init__(
        self,
        config: ProactiveConfig,
        clock: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], datetime] = datetime.now,
    ) -> None:
        self.config = config
        self._clock = clock
        self._wall_clock = wall_clock
        self._last_spoke_at: float | None = None
        self._last_user_activity_at: float = clock()
        self._recent: deque[float] = deque()

    def note_user_activity(self) -> None:
        """Call on any keystroke, mouse move or utterance: it resets the idle timer."""
        self._last_user_activity_at = self._clock()

    def note_spoke(self) -> None:
        """Call after a proactive remark actually goes out."""
        now = self._clock()
        self._last_spoke_at = now
        self._recent.append(now)
        self._trim(now)

    def idle_for(self) -> float:
        return self._clock() - self._last_user_activity_at

    def _trim(self, now: float) -> None:
        while self._recent and now - self._recent[0] > 3600.0:
            self._recent.popleft()

    def _in_quiet_hours(self) -> bool:
        window = self.config.quiet_hours
        if not window:
            return False
        start, end = window
        hour = self._wall_clock().hour
        if start == end:
            return False
        if start < end:
            return start <= hour < end
        return hour >= start or hour < end  # window wraps past midnight

    def check(self, state: State) -> GateDecision:
        if not self.config.enabled:
            return GateDecision(False, "proactive speech disabled")
        if state is State.PAUSED:
            return GateDecision(False, "paused")
        if state is not State.IDLE:
            return GateDecision(False, f"not idle (state={state.value})")
        if self._in_quiet_hours():
            return GateDecision(False, "quiet hours")

        now = self._clock()
        idle = now - self._last_user_activity_at
        if idle < self.config.idle_threshold_s:
            return GateDecision(
                False, f"idle {idle:.0f}s < threshold {self.config.idle_threshold_s:.0f}s"
            )
        if self._last_spoke_at is not None:
            since = now - self._last_spoke_at
            if since < self.config.cooldown_s:
                return GateDecision(
                    False, f"cooldown {since:.0f}s < {self.config.cooldown_s:.0f}s"
                )
        self._trim(now)
        if len(self._recent) >= self.config.max_per_hour:
            return GateDecision(False, f"hourly cap reached ({self.config.max_per_hour})")
        return GateDecision(True, "ok")
