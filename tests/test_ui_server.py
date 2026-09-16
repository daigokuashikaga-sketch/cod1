"""The local HTTP/SSE bridge the orb talks to."""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from collections.abc import Iterator

import pytest

from conftest import make_frame
from jarvis.core.config import Config
from jarvis.core.events import EventBus
from jarvis.core.orchestrator import Jarvis
from jarvis.perception.screen import SyntheticCapture
from jarvis.ui.server import UIServer


@pytest.fixture
def server(config: Config) -> Iterator[tuple[UIServer, Jarvis]]:
    config.agent.backend = "echo"
    bus = EventBus()
    jarvis = Jarvis.from_config(config, bus=bus, capture=SyntheticCapture([make_frame(seed=1)]))
    ui = UIServer(jarvis, bus, host="127.0.0.1", port=0)
    ui.start()
    try:
        yield ui, jarvis
    finally:
        ui.stop()
        jarvis.close()


def get(ui: UIServer, path: str) -> tuple[int, bytes]:
    with urllib.request.urlopen(ui.url.rstrip("/") + path, timeout=5) as response:
        return response.status, response.read()


def post(ui: UIServer, path: str, payload: dict | None = None) -> tuple[int, dict]:
    request = urllib.request.Request(
        ui.url.rstrip("/") + path,
        data=json.dumps(payload or {}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def test_it_serves_the_orb_page(server) -> None:
    ui, _ = server
    status, body = get(ui, "/")
    assert status == 200
    assert b"<canvas id=\"orb\"" in body
    assert get(ui, "/orb.js")[0] == 200


def test_status_endpoint_mirrors_the_core(server) -> None:
    ui, jarvis = server
    jarvis.handle_text("hello")
    _, body = get(ui, "/api/status")
    payload = json.loads(body)
    assert payload["state"] == "idle"
    assert payload["turns"] == 2


def test_posting_a_message_gets_a_reply(server) -> None:
    ui, jarvis = server
    status, payload = post(ui, "/api/message", {"text": "hello"})
    assert status == 200
    assert payload["reply"] == "[echo] hello"
    assert jarvis.memory.turn_count() == 2


def test_empty_messages_are_rejected(server) -> None:
    ui, _ = server
    status, payload = post(ui, "/api/message", {"text": "  "})
    assert status == 400
    assert "error" in payload


def test_pause_and_resume_endpoints(server) -> None:
    ui, jarvis = server
    assert post(ui, "/api/pause")[1]["state"] == "paused"
    assert post(ui, "/api/resume")[1]["state"] == "idle"
    assert jarvis.state.state.value == "idle"


def test_unknown_routes_are_404(server) -> None:
    ui, _ = server
    with pytest.raises(urllib.error.HTTPError) as exc:
        get(ui, "/nope")
    assert exc.value.code == 404


def test_core_failures_become_500s(server) -> None:
    ui, jarvis = server

    class BrokenBackend:
        name = "broken"

        def complete(self, *args, **kwargs):
            raise RuntimeError("upstream is down")

    jarvis.agent.backend = BrokenBackend()
    status, payload = post(ui, "/api/message", {"text": "hello"})
    assert status == 500
    assert "upstream is down" in payload["error"]


def test_events_stream_reaches_the_browser(server) -> None:
    ui, jarvis = server
    received: list[dict] = []
    ready = threading.Event()

    def reader() -> None:
        with urllib.request.urlopen(ui.url.rstrip("/") + "/api/events", timeout=10) as response:
            ready.set()
            for raw in response:
                line = raw.decode().strip()
                if line.startswith("data: "):
                    received.append(json.loads(line[6:]))
                    if any(event.get("kind") == "reply" for event in received):
                        return

    thread = threading.Thread(target=reader, daemon=True)
    thread.start()
    assert ready.wait(5)
    jarvis.handle_text("hello")
    thread.join(timeout=5)

    kinds = [event["kind"] for event in received]
    assert "state" in kinds  # the stream opens with the current state
    assert "reply" in kinds


def test_stopping_the_server_releases_the_port(config: Config) -> None:
    config.agent.backend = "echo"
    jarvis = Jarvis.from_config(config, capture=SyntheticCapture([make_frame(seed=1)]))
    ui = UIServer(jarvis, jarvis.bus, port=0)
    ui.start()
    port = ui.actual_port
    ui.stop()
    jarvis.close()
    with pytest.raises(urllib.error.URLError):
        urllib.request.urlopen(f"http://127.0.0.1:{port}/api/status", timeout=2)
