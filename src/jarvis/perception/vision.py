"""The screen-watching loop: capture, filter, gate, and only then pay Claude.

Order matters, and it is cheapest-first:

  1. capture          -- microseconds, local
  2. privacy filter   -- never send a password manager to a cloud API
  3. change detection -- local hash; most frames stop here
  4. budget guard     -- refuse to exceed today's cap
  5. Claude vision    -- the only step that costs money
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ..core.config import VisionConfig
from ..core.events import EventBus
from ..core.llm import Completion, Image, LLMBackend, Message
from ..memory.store import MemoryStore
from .change import ChangeDetector
from .costs import image_cost_usd
from .frame import Frame
from .screen import PrivacyFilter, ScreenCapture, active_window_title

VISION_SYSTEM = (
    "You are the eyes of a desktop companion. You are shown one screenshot. "
    "Reply with a single sentence describing what the user is doing -- the app, the "
    "task, and anything obviously wrong (an error, a failing test, a stuck build). "
    "If, and only if, you see something the user would genuinely want flagged right "
    "now, start your reply with 'NOTE: '. Most frames are unremarkable; do not "
    "invent significance. Never transcribe personal data, credentials or message "
    "contents."
)


@dataclass(frozen=True)
class Observation:
    summary: str
    noteworthy: bool
    width: int
    height: int
    cost_usd: float
    input_tokens: int
    output_tokens: int
    model: str
    at: float


@dataclass(frozen=True)
class TickResult:
    """Why this tick did or did not cost money."""

    status: str  # disabled | no-frame | blocked | unchanged | over-budget | observed | error
    detail: str = ""
    observation: Observation | None = None
    distance: int | None = None

    @property
    def escalated(self) -> bool:
        return self.observation is not None


class ScreenWatcher:
    def __init__(
        self,
        config: VisionConfig,
        capture: ScreenCapture,
        backend: LLMBackend,
        memory: MemoryStore,
        bus: EventBus | None = None,
        detector: ChangeDetector | None = None,
        privacy: PrivacyFilter | None = None,
        title_provider: Callable[[], str | None] = active_window_title,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.config = config
        self.capture = capture
        self.backend = backend
        self.memory = memory
        self.bus = bus
        self.detector = detector or ChangeDetector(threshold=config.change_threshold)
        self.privacy = privacy or PrivacyFilter(
            denylist=config.title_denylist, allowlist=config.title_allowlist
        )
        self.title_provider = title_provider
        self.clock = clock
        self.last_observation: Observation | None = None

    # -- public API --------------------------------------------------------

    def tick(self, force: bool = False) -> TickResult:
        """One pass of the loop. Call this on a timer; it is cheap unless it isn't."""
        if not self.config.enabled and not force:
            return self._result("disabled", "vision is switched off")

        title = self._safe_title()
        decision = self.privacy.check(title)
        if not decision.allowed:
            self.detector.reset()  # do not let a blocked frame anchor the hash
            return self._result("blocked", decision.reason)

        frame = self.capture.grab(self.config.monitor)
        if frame is None:
            return self._result("no-frame", "capture backend returned nothing")

        scaled = frame.resized(self.config.max_edge_px)
        change = self.detector.evaluate(scaled)
        if not change.changed and not force:
            return self._result("unchanged", change.reason, distance=change.distance)

        estimated = image_cost_usd(scaled.width, scaled.height, self.config.model)
        spent = self.memory.spend_today(self.config.model)
        if spent + estimated > self.config.daily_budget_usd:
            return self._result(
                "over-budget",
                f"today's spend ${spent:.2f} + ${estimated:.4f} exceeds "
                f"${self.config.daily_budget_usd:.2f}",
                distance=change.distance,
            )

        try:
            observation = self._describe(scaled, title)
        except Exception as exc:  # a vision outage must not kill the loop
            return self._result("error", f"{type(exc).__name__}: {exc}", distance=change.distance)

        self.last_observation = observation
        self.memory.record_observation(observation.summary, app=title)
        self.memory.record_usage(
            observation.model,
            observation.input_tokens,
            observation.output_tokens,
            observation.cost_usd,
        )
        return self._result(
            "observed", change.reason, observation=observation, distance=change.distance
        )

    def describe_now(self) -> str:
        """Force one description, bypassing the change gate but not privacy or budget.

        This is what the agent's ``look_at_screen`` tool calls.
        """
        result = self.tick(force=True)
        if result.observation is not None:
            return result.observation.summary
        return f"could not look at the screen ({result.status}: {result.detail})"

    # -- internals ---------------------------------------------------------

    def _describe(self, frame: Frame, title: str | None) -> Observation:
        context = f"Active window: {title}" if title else "Active window: unknown"
        completion: Completion = self.backend.complete(
            VISION_SYSTEM,
            [Message("user", context, images=(Image(frame.to_png()),))],
            model=self.config.model,
            max_tokens=self.config.max_tokens,
            temperature=1.0,
            cache_system=True,
        )
        text = completion.text.strip()
        noteworthy = text.upper().startswith("NOTE:")
        if noteworthy:
            text = text[len("NOTE:") :].strip()
        cost = completion.cost_usd or image_cost_usd(frame.width, frame.height, self.config.model)
        return Observation(
            summary=text,
            noteworthy=noteworthy,
            width=frame.width,
            height=frame.height,
            cost_usd=cost,
            input_tokens=completion.input_tokens,
            output_tokens=completion.output_tokens,
            model=completion.model,
            at=self.clock(),
        )

    def _safe_title(self) -> str | None:
        try:
            return self.title_provider()
        except Exception:
            return None

    def _result(self, status: str, detail: str = "", **extra: Any) -> TickResult:
        result = TickResult(status=status, detail=detail, **extra)
        if self.bus is not None and status not in {"unchanged", "disabled"}:
            self.bus.publish(
                "vision",
                status=status,
                detail=detail,
                summary=result.observation.summary if result.observation else None,
                cost_usd=result.observation.cost_usd if result.observation else 0.0,
            )
        return result
