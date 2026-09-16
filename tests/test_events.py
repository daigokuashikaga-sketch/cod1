from __future__ import annotations

import threading

from jarvis.core.events import EventBus


def test_subscribers_receive_published_events() -> None:
    bus = EventBus()
    seen: list[dict] = []
    bus.subscribe(lambda event: seen.append(event.to_dict()))
    bus.publish("state", state="idle")
    assert seen == [{"kind": "state", "ts": seen[0]["ts"], "state": "idle"}]


def test_unsubscribe_stops_delivery() -> None:
    bus = EventBus()
    seen: list[str] = []
    unsubscribe = bus.subscribe(lambda event: seen.append(event.kind))
    bus.publish("a")
    unsubscribe()
    bus.publish("b")
    assert seen == ["a"]
    unsubscribe()  # idempotent


def test_a_failing_subscriber_does_not_break_the_others() -> None:
    bus = EventBus()
    seen: list[str] = []

    def explode(event) -> None:
        raise RuntimeError("bad subscriber")

    bus.subscribe(explode)
    bus.subscribe(lambda event: seen.append(event.kind))
    bus.publish("state", state="idle")
    assert seen == ["state"]


def test_history_is_bounded_and_replayable() -> None:
    bus = EventBus(history=3)
    for i in range(5):
        bus.publish("tick", i=i)
    replayed = list(bus.replay())
    assert [event.payload["i"] for event in replayed] == [2, 3, 4]


def test_publishing_is_thread_safe() -> None:
    bus = EventBus(history=1000)
    seen: list[str] = []
    lock = threading.Lock()

    def collect(event) -> None:
        with lock:
            seen.append(event.kind)

    bus.subscribe(collect)
    threads = [
        threading.Thread(target=lambda: [bus.publish("tick") for _ in range(50)])
        for _ in range(4)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(seen) == 200
