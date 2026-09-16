"""The screen watcher: the order of the gates, and who pays for what."""

from __future__ import annotations

from typing import Any

from conftest import make_frame
from jarvis.core.config import VisionConfig
from jarvis.core.events import EventBus
from jarvis.core.llm import Completion, ScriptedBackend
from jarvis.memory.store import MemoryStore
from jarvis.perception.change import ChangeDetector
from jarvis.perception.screen import NullCapture, SyntheticCapture
from jarvis.perception.vision import ScreenWatcher


class CountingBackend(ScriptedBackend):
    """A scripted backend that also records how often it was actually paid to look."""

    def complete(self, system, messages, **kwargs: Any) -> Completion:
        completion = super().complete(system, messages, **kwargs)
        return Completion(
            text=completion.text,
            model=kwargs.get("model", "test"),
            input_tokens=1800,
            output_tokens=40,
            cost_usd=0.002,
        )


def watcher(
    memory: MemoryStore,
    frames,
    replies=("a terminal",),
    title: str | None = "Terminal",
    bus: EventBus | None = None,
    **config_kwargs: Any,
) -> ScreenWatcher:
    config_kwargs.setdefault("max_edge_px", 64)
    config = VisionConfig(enabled=True, **config_kwargs)
    return ScreenWatcher(
        config,
        SyntheticCapture(frames),
        CountingBackend(list(replies), fallback="a terminal"),
        memory,
        bus,
        detector=ChangeDetector(threshold=config.change_threshold, max_interval_s=None),
        title_provider=lambda: title,
    )


def test_disabled_vision_never_captures(memory: MemoryStore) -> None:
    w = watcher(memory, [make_frame(seed=1)])
    w.config.enabled = False
    assert w.tick().status == "disabled"
    assert len(w.backend.calls) == 0


def test_a_changed_frame_is_described_and_billed(memory: MemoryStore) -> None:
    w = watcher(memory, [make_frame(seed=1)], replies=("a terminal running tests",))
    result = w.tick()
    assert result.status == "observed"
    assert result.observation.summary == "a terminal running tests"
    assert result.observation.noteworthy is False
    assert memory.spend_today(w.config.model) == 0.002
    assert memory.recent_observations()[0][1] == "a terminal running tests"


def test_an_unchanged_frame_costs_nothing(memory: MemoryStore) -> None:
    frame = make_frame(seed=1)
    w = watcher(memory, [frame, frame])
    assert w.tick().status == "observed"
    assert w.tick().status == "unchanged"
    assert len(w.backend.calls) == 1  # the second frame never reached the model


def test_note_prefix_marks_a_frame_as_noteworthy(memory: MemoryStore) -> None:
    w = watcher(memory, [make_frame(seed=1)], replies=("NOTE: the build is red",))
    observation = w.tick().observation
    assert observation.noteworthy is True
    assert observation.summary == "the build is red"  # the marker is stripped


def test_sensitive_windows_are_never_captured(memory: MemoryStore) -> None:
    w = watcher(memory, [make_frame(seed=1)], title="1Password")
    result = w.tick()
    assert result.status == "blocked"
    assert "denylisted" in result.detail
    assert len(w.backend.calls) == 0


def test_a_blocked_frame_does_not_become_the_baseline(memory: MemoryStore) -> None:
    """After a blocked frame the next allowed frame must still be escalated."""
    frame = make_frame(seed=1)
    w = watcher(memory, [frame, frame])
    assert w.tick().status == "observed"
    w.title_provider = lambda: "1Password"
    assert w.tick().status == "blocked"
    w.title_provider = lambda: "Terminal"
    w.capture = SyntheticCapture([frame])
    assert w.tick().status == "observed"


def test_the_daily_budget_stops_the_loop(memory: MemoryStore) -> None:
    w = watcher(memory, [make_frame(seed=1), make_frame(seed=2)], daily_budget_usd=0.0005)
    assert w.tick().status == "observed"
    result = w.tick()
    assert result.status == "over-budget"
    assert "exceeds" in result.detail
    assert len(w.backend.calls) == 1


def test_no_frame_when_the_backend_has_no_display(memory: MemoryStore) -> None:
    w = watcher(memory, [make_frame(seed=1)])
    w.capture = NullCapture()
    assert w.tick().status == "no-frame"


def test_a_vision_outage_is_reported_not_raised(memory: MemoryStore) -> None:
    class BrokenBackend:
        name = "broken"

        def complete(self, *args: Any, **kwargs: Any) -> Completion:
            raise ConnectionError("api down")

    w = watcher(memory, [make_frame(seed=1)])
    w.backend = BrokenBackend()
    result = w.tick()
    assert result.status == "error"
    assert "ConnectionError" in result.detail


def test_a_broken_title_provider_does_not_break_capture(memory: MemoryStore) -> None:
    def explode() -> str:
        raise OSError("no window server")

    w = watcher(memory, [make_frame(seed=1)])
    w.title_provider = explode
    assert w.tick().status == "observed"


def test_describe_now_bypasses_the_change_gate(memory: MemoryStore) -> None:
    frame = make_frame(seed=1)
    w = watcher(memory, [frame, frame], replies=("a terminal", "still a terminal"))
    w.tick()
    assert w.describe_now() == "still a terminal"


def test_describe_now_still_respects_privacy(memory: MemoryStore) -> None:
    w = watcher(memory, [make_frame(seed=1)], title="Bitwarden")
    assert "could not look at the screen" in w.describe_now()


def test_frames_are_downscaled_before_being_sent(memory: MemoryStore) -> None:
    w = watcher(memory, [make_frame(256, 128, seed=3)], max_edge_px=64)
    observation = w.tick().observation
    assert (observation.width, observation.height) == (64, 32)


def test_events_are_published_for_billable_ticks(memory: MemoryStore) -> None:
    bus = EventBus()
    seen: list[dict] = []
    bus.subscribe(lambda e: seen.append(e.to_dict()))
    frame = make_frame(seed=1)
    w = watcher(memory, [frame, frame], bus=bus)
    w.tick()
    w.tick()  # unchanged: deliberately silent, to keep the event feed useful
    kinds = [e for e in seen if e["kind"] == "vision"]
    assert len(kinds) == 1
    assert kinds[0]["status"] == "observed"
