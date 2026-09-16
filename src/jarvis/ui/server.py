"""A local HTTP + Server-Sent-Events bridge between the Python core and the orb.

Stdlib only, bound to loopback. SSE rather than WebSockets because it is a
one-way state feed with a three-line client and no dependency; the handful of
inbound actions go over plain POSTs.

Point a Tauri or Electron shell at this server to get a transparent,
always-on-top orb; open it in a browser to get the same thing in a tab.
"""

from __future__ import annotations

import json
import queue
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..core.events import Event, EventBus

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..core.orchestrator import Jarvis

WEB_ROOT = Path(__file__).parent / "web"
MAX_QUEUE = 256


class UIServer:
    """Serves the orb page and streams core events to it."""

    def __init__(
        self,
        jarvis: Jarvis,
        bus: EventBus | None = None,
        host: str = "127.0.0.1",
        port: int = 8765,
    ) -> None:
        self.jarvis = jarvis
        self.bus = bus or jarvis.bus
        self.host = host
        self.port = port
        self._clients: list[queue.Queue[Event]] = []
        self._lock = threading.Lock()
        self._unsubscribe = self.bus.subscribe(self._fan_out)
        self._server = ThreadingHTTPServer((host, port), _make_handler(self))
        self._server.daemon_threads = True
        self._thread: threading.Thread | None = None

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.actual_port}/"

    @property
    def actual_port(self) -> int:
        return int(self._server.server_address[1])

    # -- event fan-out -----------------------------------------------------

    def _fan_out(self, event: Event) -> None:
        with self._lock:
            clients = list(self._clients)
        for client in clients:
            try:
                client.put_nowait(event)
            except queue.Full:
                continue  # a stalled browser tab must not block the core

    def register(self) -> queue.Queue[Event]:
        client: queue.Queue[Event] = queue.Queue(maxsize=MAX_QUEUE)
        with self._lock:
            self._clients.append(client)
        return client

    def unregister(self, client: queue.Queue[Event]) -> None:
        with self._lock:
            if client in self._clients:
                self._clients.remove(client)

    # -- lifecycle ---------------------------------------------------------

    def serve_forever(self) -> None:
        self._server.serve_forever()

    def start(self) -> str:
        self._thread = threading.Thread(
            target=self.serve_forever, name="jarvis-ui", daemon=True
        )
        self._thread.start()
        return self.url

    def stop(self) -> None:
        self._unsubscribe()
        self._server.shutdown()
        self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None


def _make_handler(ui: UIServer) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "Jarvis/0.1"
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt: str, *args: Any) -> None:  # keep the console clean
            return

        # -- helpers -------------------------------------------------------

        def _send(self, code: int, body: bytes, content_type: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, payload: dict[str, Any], code: int = 200) -> None:
            self._send(code, json.dumps(payload).encode("utf-8"), "application/json")

        def _read_json(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length") or 0)
            if not length:
                return {}
            try:
                return json.loads(self.rfile.read(length) or b"{}")
            except json.JSONDecodeError:
                return {}

        # -- routes --------------------------------------------------------

        def do_GET(self) -> None:  # BaseHTTPRequestHandler naming
            path = self.path.split("?", 1)[0]
            if path in {"/", "/index.html"}:
                self._send(200, (WEB_ROOT / "index.html").read_bytes(), "text/html; charset=utf-8")
                return
            if path == "/orb.js":
                self._send(
                    200, (WEB_ROOT / "orb.js").read_bytes(), "text/javascript; charset=utf-8"
                )
                return
            if path == "/api/status":
                self._json(ui.jarvis.status_dict())
                return
            if path == "/api/events":
                self._stream()
                return
            self._json({"error": "not found"}, code=404)

        def do_POST(self) -> None:  # BaseHTTPRequestHandler naming
            path = self.path.split("?", 1)[0]
            if path == "/api/message":
                text = str(self._read_json().get("text", "")).strip()
                if not text:
                    self._json({"error": "empty message"}, code=400)
                    return
                try:
                    reply = ui.jarvis.handle_text(text)
                except Exception as exc:
                    self._json({"error": f"{type(exc).__name__}: {exc}"}, code=500)
                    return
                self._json({"reply": reply, "state": ui.jarvis.state.state.value})
                return
            if path in {"/api/pause", "/api/resume"}:
                (ui.jarvis.pause if path.endswith("pause") else ui.jarvis.resume)()
                self._json({"state": ui.jarvis.state.state.value})
                return
            self._json({"error": "not found"}, code=404)

        def _stream(self) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "keep-alive")
            self.end_headers()
            client = ui.register()
            try:
                self._write_event({"kind": "state", "state": ui.jarvis.state.state.value})
                for event in ui.bus.replay():
                    self._write_event(event.to_dict())
                while True:
                    try:
                        event = client.get(timeout=15.0)
                    except queue.Empty:
                        self.wfile.write(b": keep-alive\n\n")  # stops proxies idling us out
                        self.wfile.flush()
                        continue
                    self._write_event(event.to_dict())
            except (BrokenPipeError, ConnectionResetError):
                return
            finally:
                ui.unregister(client)

        def _write_event(self, payload: dict[str, Any]) -> None:
            self.wfile.write(f"data: {json.dumps(payload)}\n\n".encode())
            self.wfile.flush()

    return Handler
