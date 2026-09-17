# CLAUDE.md

Jarvis is a desktop-resident AI companion: it watches the screen, listens for a
wake word, remembers across restarts, and speaks up on its own when it has
something worth saying. The Python core is the source of truth; the orb UI only
renders state.

## Commands

```bash
uv sync --extra dev          # or: pip install -e '.[dev]'
pytest                       # whole suite, offline, no API key
ruff check src tests         # lint
mypy                         # types
python -m jarvis run         # everything at once: orb + screen loop + microphone
python -m jarvis demo        # end-to-end run on synthetic frames, no API key
python -m jarvis chat        # interactive text session
python -m jarvis listen      # hands-free voice conversation (needs a microphone)
python -m jarvis ui --open   # orb overlay at http://127.0.0.1:8765
(cd desktop/src-tauri && cargo run)   # the same orb as a desktop overlay window
python -m jarvis watch       # foreground screen loop, prints every decision
python -m jarvis estimate    # project the monthly vision bill
python -m jarvis doctor      # which backends are actually installed
```

## Layout

```
src/jarvis/
  core/         config, event bus, FSM + proactive gate, LLM backends, agent, orchestrator
  perception/   frame maths, change detection, cost arithmetic, capture, screen watcher
  memory/       SQLite store, embeddings, recall
  voice/        mic pump (audio, vad), STT, TTS, playback, wake word -- all optional
  ui/           local HTTP/SSE server + the dependency-free orb page
  cli.py        argparse entry points
desktop/        Tauri v2 shell: transparent always-on-top window + click-through
tests/          stdlib-only pytest suite; no network, no display, no API key
docs/           architecture, costs, configuration
```

## Rules

- Keep the core importable with **zero third-party dependencies**. Every heavy
  library (anthropic, mss, faster-whisper, kokoro, numpy) is an optional extra,
  imported lazily inside the class that needs it, with a `null` fallback so a
  bare install still runs.
- Every backend is a `Protocol` in its own module with at least one offline
  implementation (`EchoBackend`, `ScriptedBackend`, `NullTTS`, `SyntheticCapture`).
  Tests use those, never the network.
- The cheap gates run before the expensive one, in this order: privacy filter,
  change detection, budget guard, then the Claude vision call. Never reorder them.
- Proactive speech only ever leaves `ProactiveGate.check()`. Do not add a second
  path that lets Jarvis interrupt the user.
- The voice loop mirrors the screen loop: a cheap local gate (VAD) in front of
  the expensive step (STT), one transcription per utterance, never per frame.
  Segmentation is counted in frames so it is testable without a clock.
- Half duplex is the default: the mic is closed while Jarvis speaks, so it
  cannot hear itself and interrupt its own reply. Barge-in needs
  `voice.duplex = "full"`, which needs headphones.
- Prompt caching is a byte-exact prefix match. Anything that changes per turn
  goes in `SystemPrompt.volatile`, never `static`, and a breakpoint is only sent
  when the prefix clears the model's minimum (see `is_cacheable`).
- Money: anything that can spend it must be off by default, capped by
  `vision.daily_budget_usd`, and recorded through `MemoryStore.record_usage`.
- Privacy: screenshots are downscaled before leaving the machine, sensitive
  window titles are never captured, and `jarvis.pause()` stops everything.
- Inject clocks (`clock=time.monotonic`) rather than sleeping in tests.
- Write tests that assert behaviour ("an unchanged frame costs nothing"), not
  implementation details.

## Conventions

- Python 3.11+, `from __future__ import annotations`, full type hints.
- Line length 100, ruff rules `E,F,I,UP,B,SIM,RUF`.
- Dataclasses for values, `Protocol` for seams, no inheritance for reuse.
- Comments explain *why* (especially cost and privacy trade-offs), not *what*.
- Events go on the bus (`bus.publish("vision", ...)`); subsystems never call
  each other directly. The orchestrator owns all wiring.
- The desktop shell stays thin: window behaviour only, no product logic. Anything
  it must agree with the core about (port, orb geometry) gets a test in
  `tests/test_desktop_shell.py` rather than a comment.

## Where to add things

| Task | Place |
|---|---|
| New TTS/STT engine | `voice/tts.py` / `voice/stt.py`, then `build_*` |
| New VAD or audio device | `voice/vad.py` / `voice/audio.py`, then `build_*` |
| New capture method | `perception/screen.py`, implement `ScreenCapture` |
| New agent tool | `TOOLS` + `Agent._dispatch` in `core/agent.py` |
| New UI signal | `bus.publish(...)`, then a case in `ui/web/orb.js` |
| Window behaviour | `desktop/src-tauri/src/main.rs` (+ a drift test) |
| New model pricing | `PRICING` in `perception/costs.py` |
