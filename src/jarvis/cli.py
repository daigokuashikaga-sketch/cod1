"""Command line entry points.

``jarvis demo`` is the one to run first: it exercises the whole pipeline --
memory, FSM, change detection, proactive gating, the orb feed -- with synthetic
frames and the offline echo backend, so it needs no API key and no display.
"""

from __future__ import annotations

import argparse
import array
import json
import random
import sys
import threading
import time
from collections.abc import Sequence
from pathlib import Path

from .core.config import Config, load_config
from .core.events import Event, EventBus
from .core.llm import ScriptedBackend, build_backend
from .core.orchestrator import SILENCE, Jarvis
from .memory.store import MemoryStore
from .perception.costs import PRICING, image_tokens, project_monthly
from .perception.frame import Frame
from .perception.screen import SyntheticCapture, build_capture
from .voice.audio import SyntheticAudioSource
from .voice.stt import ScriptedSTT

BANNER = r"""
   _  __ _ _ ___   __ ___ ___
  | |/ _` | '_/ V / (_-</ -_)    jarvis {version}
 _/ |\__,_|_|  \_/ \___/\___|    type /help for commands, /quit to leave
|__/
"""


def _config(args: argparse.Namespace) -> Config:
    return load_config(args.config)


def _print_event(event: Event) -> None:
    if event.kind == "voice" and event.payload.get("status") == "utterance":
        print(f"\n\033[32myou (heard)>\033[0m {event.payload.get('text', '')}")
    elif event.kind == "bargein":
        print("  \033[2m[interrupted]\033[0m")
    elif event.kind == "reply" and event.payload.get("text"):
        print(f"\033[36mjarvis>\033[0m {event.payload['text']}")
    elif event.kind == "vision" and event.payload.get("summary"):
        print(f"  \033[2m[saw] {event.payload['summary']}\033[0m")
    elif event.kind == "proactive" and event.payload.get("spoken"):
        print(f"\n\033[36mjarvis>\033[0m {event.payload.get('text', '')}")
    elif event.kind == "error":
        print(f"  \033[31m[{event.payload.get('where')}] {event.payload.get('detail')}\033[0m")


# -- commands --------------------------------------------------------------


def cmd_chat(args: argparse.Namespace) -> int:
    from . import __version__

    config = _config(args)
    bus = EventBus()
    if args.verbose:
        bus.subscribe(_print_event)
    with Jarvis.from_config(config, bus=bus) as jarvis:
        jarvis.start()
        print(BANNER.format(version=__version__))
        print(
            f"  backend={jarvis.agent.backend.name} model={config.agent.model} "
            f"vision={'on' if config.vision.enabled else 'off'} "
            f"proactive={'on' if config.proactive.enabled else 'off'}\n"
        )
        while True:
            try:
                line = input("\033[32myou>\033[0m ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not line:
                continue
            if line in {"/quit", "/exit"}:
                break
            if line == "/help":
                print(
                    "  /status  current state, spend, turn count\n"
                    "  /facts   what Jarvis remembers about you\n"
                    "  /look    describe the screen right now\n"
                    "  /pause   stop watching and speaking   /resume\n"
                    "  /quit    leave"
                )
                continue
            if line == "/status":
                print(json.dumps(jarvis.status_dict(), indent=2, default=str))
                continue
            if line == "/facts":
                for fact in jarvis.memory.facts():
                    print(f"  {fact.key}: {fact.value}")
                continue
            if line == "/look":
                print(f"  {jarvis.watcher.describe_now()}")
                continue
            if line == "/pause":
                jarvis.pause()
                print("  paused")
                continue
            if line == "/resume":
                jarvis.resume()
                print("  resumed")
                continue
            reply = jarvis.handle_text(line)
            print(f"\033[36mjarvis>\033[0m {reply}\n")
    return 0


def cmd_ui(args: argparse.Namespace) -> int:
    from .ui.server import UIServer

    config = _config(args)
    bus = EventBus()
    with Jarvis.from_config(config, bus=bus) as jarvis:
        jarvis.start()
        server = UIServer(
            jarvis,
            bus,
            host=args.host or config.ui.host,
            port=args.port if args.port is not None else config.ui.port,
        )
        print(f"orb running at {server.url}  (ctrl-c to stop)")
        if args.open:
            import webbrowser

            webbrowser.open(server.url)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nstopping")
        finally:
            server.stop()
    return 0


def cmd_listen(args: argparse.Namespace) -> int:
    """Open the microphone and talk to Jarvis hands-free."""
    config = _config(args)
    if args.audio:
        config.voice.audio_backend = args.audio
    if args.player:
        config.voice.player_backend = args.player
    if config.voice.audio_backend == "null":
        config.voice.audio_backend = "sounddevice"
    if config.voice.player_backend == "null":
        config.voice.player_backend = "sounddevice"

    bus = EventBus()
    bus.subscribe(_print_event)
    with Jarvis.from_config(config, bus=bus) as jarvis:
        if not jarvis.start_listening():
            print(
                "jarvis: no microphone available.\n"
                "  Install the voice extra (pip install 'jarvis[stt]') and check that an\n"
                "  input device exists. `jarvis doctor` shows what is configured.",
                file=sys.stderr,
            )
            return 1
        jarvis.start()
        wake = config.voice.wake_word if config.voice.wake_word_enabled else "(always listening)"
        print(
            f"listening - say \"{wake}\" to talk"
            f"  [stt={jarvis.stt.name} tts={jarvis.tts.name} vad={config.voice.vad_backend}]"
            "  (ctrl-c to stop)"
        )
        stop = threading.Event()
        try:
            while not stop.wait(1.0):
                pass
        except KeyboardInterrupt:
            print("\nstopped")
    return 0


def cmd_watch(args: argparse.Namespace) -> int:
    """Run the perception loop in the foreground and print every decision."""
    config = _config(args)
    config.vision.enabled = True
    bus = EventBus()
    bus.subscribe(_print_event)
    with Jarvis.from_config(config, bus=bus) as jarvis:
        print(
            f"watching monitor {config.vision.monitor} every "
            f"{config.vision.capture_interval_s:g}s, budget "
            f"${config.vision.daily_budget_usd:.2f}/day (ctrl-c to stop)"
        )
        try:
            while True:
                result = jarvis.vision_tick()
                stamp = time.strftime("%H:%M:%S")
                distance = "" if result.distance is None else f" d={result.distance}"
                print(f"{stamp} {result.status}{distance} {result.detail}")
                time.sleep(max(1.0, config.vision.capture_interval_s))
        except KeyboardInterrupt:
            print("\nstopped")
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    """End-to-end run on synthetic frames with the offline backend. No API key needed."""
    config = load_config(args.config)
    config.agent.backend = "echo"
    config.memory.db_path = ":memory:"
    config.vision.enabled = True
    config.vision.daily_budget_usd = 0.50
    config.proactive.enabled = True
    config.proactive.idle_threshold_s = 0.0
    config.proactive.cooldown_s = 0.0

    rng = random.Random(7)
    def noise() -> Frame:
        return Frame(64, 48, bytes(rng.randrange(256) for _ in range(64 * 48 * 3)))

    still = noise()
    frames = [still, still, noise(), still, noise()]

    bus = EventBus()
    events: list[str] = []

    pending: list[str] = []

    def watch(event: Event) -> None:
        # Buffered rather than printed, so proactive lines land under the frame
        # that triggered them instead of ahead of it.
        events.append(event.kind)
        if event.kind == "proactive":
            pending.append(
                f"    jarvis (unprompted)> {event.payload.get('text', '')}"
                if event.payload.get("spoken")
                else f"    (stayed quiet: {event.payload.get('reason')})"
            )

    bus.subscribe(watch)
    with Jarvis.from_config(config, bus=bus, capture=SyntheticCapture(frames)) as jarvis:
        jarvis.watcher.title_provider = lambda: "Terminal"
        # Canned assistant replies so the demo reads like a real session offline.
        jarvis.agent.backend = ScriptedBackend(
            [
                "Noted - neovim.",
                "You use neovim.",
                "Three tests just went red in tests/test_memory.py.",
            ],
            fallback=SILENCE,
        )
        jarvis.stt = ScriptedSTT(
            ["hey jarvis, what is my editor?", "just mumbling to myself"]
        )
        # Canned vision replies so the demo also exercises the proactive path.
        jarvis.watcher.backend = ScriptedBackend(
            [
                "a terminal running the test suite",
                "NOTE: three tests are failing in tests/test_memory.py",
                "the same terminal, scrolled",
                "an editor with the failing test open",
            ],
            fallback="a terminal",
        )
        print("-- conversation --")
        print("you>    my editor is neovim")
        print(f"jarvis> {jarvis.handle_text('my editor is neovim')}")
        jarvis.memory.remember("editor", "neovim")

        print("\n-- voice (synthetic microphone, no audio device needed) --")
        heard: list[str] = []
        bus.subscribe(
            lambda e: heard.append(e.payload.get("text", ""))
            if e.kind == "voice" and e.payload.get("status") == "utterance"
            else None
        )
        # Two utterances separated by silence; only the first says the wake word.
        mic_frames = (
            _audio(0, 5) + _audio(9000, 20) + _audio(0, 30) + _audio(9000, 20) + _audio(0, 30)
        )
        jarvis.build_listener(SyntheticAudioSource(mic_frames)).run()
        for utterance in heard:
            print(f"  heard>  {utterance}")
        print(f"  jarvis> {jarvis.memory.recent_turns()[-1].content}")
        print("  (the second utterance had no wake word, so it was ignored)")

        print("\n-- perception (5 synthetic frames) --")
        for i in range(len(frames)):
            result = jarvis.vision_tick()
            print(f"  frame {i + 1}: {result.status:<11} {result.detail}")
            while pending:
                print(pending.pop(0))

        print("\n-- budget guard --")
        jarvis.config.vision.daily_budget_usd = 0.0
        jarvis.watcher.detector.reset()
        print(f"  with a $0 daily cap: {jarvis.vision_tick().status}")

        print("\n-- state --")
        print(json.dumps(jarvis.status_dict(), indent=2, default=str))
        print(f"\nevents seen: {', '.join(sorted(set(events)))}")
    return 0


def _audio(level: int, frames: int, samples: int = 480) -> list[bytes]:
    """Synthetic int16 mono frames: silence at level 0, speech above it."""
    wave = array.array("h", [level if i % 2 else -level for i in range(samples)])
    return [wave.tobytes()] * frames


def cmd_estimate(args: argparse.Namespace) -> int:
    width, height = (int(p) for p in args.resolution.lower().split("x"))
    print(f"frame {width}x{height}, {args.hours:g}h/day, {args.days} days\n")
    header = f"{'model':<28}{'tokens/img':>11}{'escalated':>11}{'$/month':>10}"
    print(header)
    print("-" * len(header))
    for model in PRICING:
        for rate in (1.0, args.change_rate):
            projection = project_monthly(
                width, height, args.interval, args.hours, args.days, rate, model
            )
            label = model if rate == 1.0 else f"  ^ with {rate:.0%} change gate"
            print(
                f"{label:<28}{image_tokens(width, height, model):>11}"
                f"{projection.escalated:>11}{projection.total_usd:>10.2f}"
            )
            if args.change_rate >= 1.0:
                break
    print(
        "\nPrices move; re-check them against the live pricing page before "
        "trusting a monthly figure."
    )
    return 0


def cmd_memory(args: argparse.Namespace) -> int:
    config = _config(args)
    with MemoryStore(
        config.memory.db_path,
        embedder=config.memory.embedder,
        embedding_dim=config.memory.embedding_dim,
    ) as memory:
        if args.remember:
            key, _, value = args.remember.partition("=")
            if not value:
                print("use --remember key=value", file=sys.stderr)
                return 2
            memory.remember(key.strip(), value.strip())
            print(f"remembered {key.strip()}")
            return 0
        if args.forget:
            print("forgotten" if memory.forget(args.forget) else "no such fact")
            return 0
        if args.search:
            for hit in memory.recall(args.search, limit=10):
                print(f"  {hit.score:.2f} ({hit.kind}) {hit.text}")
            return 0
        print(f"db: {config.memory.db_path}")
        print(f"turns: {memory.turn_count()}  sessions: {', '.join(memory.sessions()) or '-'}")
        print(f"spend today: ${memory.spend_today():.4f}")
        facts = memory.facts()
        if facts:
            print("facts:")
            for fact in facts:
                print(f"  {fact.key}: {fact.value}")
        for ts, summary, app, _ in memory.recent_observations(5):
            when = time.strftime("%H:%M:%S", time.localtime(ts))
            print(f"  [{when}] {app or '?'}: {summary}")
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    config = _config(args)
    rows: list[tuple[str, str, str]] = []

    key = "set" if config.anthropic_api_key else "MISSING"
    rows.append(("ANTHROPIC_API_KEY", key, "required for Claude reasoning and vision"))

    backend = build_backend("auto", config.anthropic_api_key)
    rows.append(("llm backend", backend.name, f"configured: {config.agent.backend}"))

    capture = build_capture("auto")
    rows.append((
        "screen capture",
        type(capture).__name__,
        "install 'jarvis[screen]' and run with a display for real capture",
    ))
    capture.close()

    for label, module, extra in (
        ("stt", "faster_whisper", "jarvis[stt]"),
        ("tts", "kokoro", "jarvis[tts]"),
        ("audio in/out", "sounddevice", "jarvis[stt]"),
        ("vad", "silero_vad", "jarvis[stt]"),
        ("wake word", "openwakeword", "jarvis[stt]"),
        ("vectors", "numpy", "jarvis[vectors]"),
    ):
        try:
            __import__(module)
            rows.append((label, "available", module))
        except ImportError:
            rows.append((label, "not installed", f"pip install '{extra}'"))

    db = Path(config.memory.db_path)
    rows.append(("memory db", "present" if db.exists() else "will be created", str(db)))

    width = max(len(row[0]) for row in rows)
    for name, value, note in rows:
        flag = "\033[33m!\033[0m" if value in {"MISSING", "not installed"} else "\033[32m+\033[0m"
        print(f" {flag} {name:<{width}}  {value:<16} {note}")
    return 0


# -- parser ----------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="jarvis", description=__doc__)
    parser.add_argument("--config", help="path to config.toml")
    sub = parser.add_subparsers(dest="command", required=True)

    chat = sub.add_parser("chat", help="interactive text conversation")
    chat.add_argument("-v", "--verbose", action="store_true", help="print perception events")
    chat.set_defaults(func=cmd_chat)

    ui = sub.add_parser("ui", help="serve the orb overlay")
    ui.add_argument("--host")
    ui.add_argument("--port", type=int)
    ui.add_argument("--open", action="store_true", help="open a browser window")
    ui.set_defaults(func=cmd_ui)

    listen = sub.add_parser("listen", help="hands-free voice conversation")
    listen.add_argument("--audio", help="audio input backend (default: sounddevice)")
    listen.add_argument("--player", help="audio output backend (default: sounddevice)")
    listen.set_defaults(func=cmd_listen)

    watch = sub.add_parser("watch", help="run the screen loop in the foreground")
    watch.set_defaults(func=cmd_watch)

    demo = sub.add_parser("demo", help="offline end-to-end demo, no API key required")
    demo.set_defaults(func=cmd_demo)

    estimate = sub.add_parser("estimate", help="project the monthly vision bill")
    estimate.add_argument("--resolution", default="1920x1080")
    estimate.add_argument("--interval", type=float, default=10.0, help="seconds between frames")
    estimate.add_argument("--hours", type=float, default=8.0)
    estimate.add_argument("--days", type=int, default=30)
    estimate.add_argument(
        "--change-rate",
        type=float,
        default=0.12,
        help="fraction of frames the local change detector escalates",
    )
    estimate.set_defaults(func=cmd_estimate)

    memory = sub.add_parser("memory", help="inspect or edit what Jarvis remembers")
    memory.add_argument("--remember", metavar="KEY=VALUE")
    memory.add_argument("--forget", metavar="KEY")
    memory.add_argument("--search", metavar="QUERY")
    memory.set_defaults(func=cmd_memory)

    doctor = sub.add_parser("doctor", help="report which backends are actually available")
    doctor.set_defaults(func=cmd_doctor)
    return parser


MISSING_BACKEND_HINT = (
    "  Put ANTHROPIC_API_KEY in your environment (see .env.example), or set\n"
    '  agent.backend = "echo" in config.toml to use the offline stand-in.\n'
    "  Run `jarvis doctor` to see what is configured."
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except RuntimeError as exc:
        # Missing keys and missing optional libraries are configuration problems,
        # not crashes: say what to do instead of printing a traceback.
        print(f"jarvis: {exc}", file=sys.stderr)
        print(MISSING_BACKEND_HINT, file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
