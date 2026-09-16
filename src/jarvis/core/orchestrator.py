"""The orchestrator: one object that owns the subsystems and the rules between them.

Everything above this line is independently testable; everything below it is
wiring. The rules that live here, and only here, are:

* the FSM moves IDLE -> THINKING -> SPEAKING -> IDLE for every turn;
* the wake word gates spoken input, and is ignored while Jarvis is speaking;
* the screen loop may only interrupt from IDLE, via :class:`ProactiveGate`.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ..memory.store import MemoryStore
from ..perception.change import ChangeDetector
from ..perception.screen import ScreenCapture, build_capture
from ..perception.vision import ScreenWatcher, TickResult
from ..voice.stt import STTBackend, build_stt
from ..voice.tts import TTSBackend, build_tts
from ..voice.wakeword import TextWakeWord, WakeWordDetector, build_wake_word
from .agent import Agent
from .config import Config
from .events import EventBus
from .llm import build_backend
from .state_machine import ProactiveGate, State, StateMachine

PROACTIVE_PROMPT = (
    "You just noticed this on the user's screen, unprompted: {summary}\n\n"
    "If it is worth interrupting them for, say one short sentence to them now. "
    "If it is not, reply with exactly: SILENCE"
)
SILENCE = "SILENCE"


@dataclass
class Status:
    state: str
    idle_for: float
    turns: int
    spend_today: float
    vision_enabled: bool
    proactive_enabled: bool
    last_observation: str | None


class Jarvis:
    def __init__(
        self,
        config: Config,
        memory: MemoryStore,
        agent: Agent,
        watcher: ScreenWatcher,
        tts: TTSBackend,
        stt: STTBackend,
        wake_word: WakeWordDetector,
        bus: EventBus,
        state: StateMachine | None = None,
        gate: ProactiveGate | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.config = config
        self.memory = memory
        self.agent = agent
        self.watcher = watcher
        self.tts = tts
        self.stt = stt
        self.wake_word = wake_word
        self.bus = bus
        self.state = state or StateMachine(bus, clock=clock)
        self.gate = gate or ProactiveGate(config.proactive, clock=clock)
        self._clock = clock
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []

    # -- construction ------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        config: Config,
        bus: EventBus | None = None,
        capture: ScreenCapture | None = None,
    ) -> Jarvis:
        bus = bus or EventBus()
        memory = MemoryStore(
            config.memory.db_path,
            embedder=config.memory.embedder,
            embedding_dim=config.memory.embedding_dim,
        )
        backend = build_backend(config.agent.backend, config.anthropic_api_key)
        vision_backend = build_backend(
            "auto" if config.vision.enabled else "echo", config.anthropic_api_key
        )
        watcher = ScreenWatcher(
            config.vision,
            capture or build_capture("auto" if config.vision.enabled else "null"),
            vision_backend,
            memory,
            bus,
            detector=ChangeDetector(threshold=config.vision.change_threshold),
        )
        agent = Agent(
            config.agent, backend, memory, bus, screen_describer=watcher.describe_now
        )
        return cls(
            config=config,
            memory=memory,
            agent=agent,
            watcher=watcher,
            tts=build_tts(config.voice.tts_backend, config.voice.tts_voice),
            stt=build_stt(config.voice.stt_backend, config.voice.stt_model),
            wake_word=build_wake_word(config.voice.wake_word, config.voice.wake_word_enabled),
            bus=bus,
        )

    # -- conversation ------------------------------------------------------

    def handle_text(self, text: str, speak: bool = True) -> str:
        """A direct, explicitly addressed turn. Always answered."""
        text = text.strip()
        if not text:
            return ""
        self.gate.note_user_activity()
        if self.state.state is State.PAUSED:
            return ""
        self.state.try_to(State.THINKING, reason="user turn")
        self.bus.publish("user", text=text)
        try:
            reply = self.agent.respond(text)
        except Exception as exc:
            self.state.try_to(State.ERROR, reason=str(exc))
            self.bus.publish("error", where="agent", detail=f"{type(exc).__name__}: {exc}")
            self.state.try_to(State.IDLE, reason="recovered")
            raise
        self._deliver(reply.text, speak=speak)
        return reply.text

    def handle_utterance(self, text: str, speak: bool = True) -> str | None:
        """A spoken utterance. Returns ``None`` when the wake word gate rejects it."""
        text = (text or "").strip()
        if not text:
            return None
        if self.config.voice.suppress_while_speaking and self.state.state is State.SPEAKING:
            self.bus.publish("wake", accepted=False, reason="suppressed while speaking")
            return None
        if isinstance(self.wake_word, TextWakeWord):
            if not self.wake_word.matches(text):
                self.gate.note_user_activity()
                self.bus.publish("wake", accepted=False, reason="no wake word")
                return None
            text = self.wake_word.strip_phrase(text) or text
        self.bus.publish("wake", accepted=True, reason="wake word")
        self.state.try_to(State.LISTENING, reason="utterance")
        return self.handle_text(text, speak=speak)

    def _deliver(self, text: str, speak: bool = True) -> None:
        if not text:
            self.state.try_to(State.IDLE, reason="empty reply")
            return
        self.state.try_to(State.SPEAKING, reason="reply")
        if speak:
            try:
                self.tts.speak(text)
            except Exception as exc:
                self.bus.publish("error", where="tts", detail=f"{type(exc).__name__}: {exc}")
        self.state.try_to(State.IDLE, reason="done speaking")

    # -- perception --------------------------------------------------------

    def vision_tick(self) -> TickResult:
        """One perception pass, plus the proactive decision that may follow it."""
        result = self.watcher.tick()
        observation = result.observation
        if observation is None or not observation.noteworthy:
            return result

        decision = self.gate.check(self.state.state)
        if not decision.allowed:
            self.bus.publish("proactive", spoken=False, reason=decision.reason)
            return result

        self.state.try_to(State.THINKING, reason="proactive")
        try:
            reply = self.agent.respond(
                PROACTIVE_PROMPT.format(summary=observation.summary),
                persist=False,
                source="proactive",
            )
        except Exception as exc:
            self.state.try_to(State.ERROR, reason=str(exc))
            self.bus.publish("error", where="proactive", detail=f"{type(exc).__name__}: {exc}")
            self.state.try_to(State.IDLE, reason="recovered")
            return result

        text = reply.text.strip()
        if not text or text.upper().startswith(SILENCE):
            self.bus.publish("proactive", spoken=False, reason="model chose silence")
            self.state.try_to(State.IDLE, reason="proactive silence")
            return result

        self.memory.add_turn("assistant", text, source="proactive")
        self.bus.publish("proactive", spoken=True, text=text)
        self.gate.note_spoke()
        self._deliver(text)
        return result

    # -- lifecycle ---------------------------------------------------------

    def pause(self) -> None:
        """Stop looking and stop talking until resumed. The privacy panic button."""
        self.state.try_to(State.PAUSED, reason="paused")
        self.bus.publish("paused", paused=True)

    def resume(self) -> None:
        self.state.try_to(State.IDLE, reason="resumed")
        self.gate.note_user_activity()
        self.bus.publish("paused", paused=False)

    def start(self) -> None:
        """Start the background perception loop (no-op when vision is disabled)."""
        if not self.config.vision.enabled or self._threads:
            return
        self._stop.clear()
        thread = threading.Thread(target=self._vision_loop, name="jarvis-vision", daemon=True)
        thread.start()
        self._threads.append(thread)

    def _vision_loop(self) -> None:
        interval = max(1.0, self.config.vision.capture_interval_s)
        while not self._stop.wait(interval):
            if self.state.state is State.PAUSED:
                continue
            try:
                self.vision_tick()
            except Exception as exc:  # the loop outlives any single failure
                self.bus.publish("error", where="vision-loop", detail=str(exc))

    def stop(self) -> None:
        self._stop.set()
        for thread in self._threads:
            thread.join(timeout=2.0)
        self._threads.clear()
        self.watcher.capture.close()

    def close(self) -> None:
        self.stop()
        self.memory.close()

    def __enter__(self) -> Jarvis:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- introspection -----------------------------------------------------

    def status(self) -> Status:
        last = self.watcher.last_observation
        return Status(
            state=self.state.state.value,
            idle_for=self.gate.idle_for(),
            turns=self.memory.turn_count(),
            spend_today=self.memory.spend_today(),
            vision_enabled=self.config.vision.enabled,
            proactive_enabled=self.config.proactive.enabled,
            last_observation=last.summary if last else None,
        )

    def status_dict(self) -> dict[str, Any]:
        return vars(self.status())
