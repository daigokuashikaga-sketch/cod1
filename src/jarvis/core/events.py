"""A tiny synchronous event bus.

Every subsystem publishes here and nothing subscribes to another subsystem
directly, so the UI, the logger and the tests can all watch the same stream.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Event:
    kind: str
    payload: dict[str, Any] = field(default_factory=dict)
    ts: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "ts": self.ts, **self.payload}


Subscriber = Callable[[Event], None]


class EventBus:
    """Fan-out with a bounded replay buffer, safe to publish to from any thread."""

    def __init__(self, history: int = 200) -> None:
        self._subscribers: list[Subscriber] = []
        self._history: deque[Event] = deque(maxlen=history)
        self._lock = threading.RLock()

    def subscribe(self, subscriber: Subscriber) -> Callable[[], None]:
        with self._lock:
            self._subscribers.append(subscriber)

        def unsubscribe() -> None:
            with self._lock:
                if subscriber in self._subscribers:
                    self._subscribers.remove(subscriber)

        return unsubscribe

    def publish(self, kind: str, **payload: Any) -> Event:
        event = Event(kind=kind, payload=payload)
        with self._lock:
            self._history.append(event)
            subscribers = list(self._subscribers)
        for subscriber in subscribers:
            # One bad subscriber must not take down the perception loop.
            try:
                subscriber(event)
            except Exception:  # one bad subscriber must not break the loop
                continue
        return event

    def replay(self) -> Iterable[Event]:
        with self._lock:
            return list(self._history)
