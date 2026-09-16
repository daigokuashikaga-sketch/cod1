# Architecture

## The shape of it

```
core/orchestrator.py   wires everything; owns the rules between subsystems
core/state_machine.py  IDLE/LISTENING/THINKING/SPEAKING/CANCELLED/ERROR/PAUSED + ProactiveGate
core/agent.py          persona, memory injection, tool round-trip
core/llm.py            Anthropic / Echo / Scripted backends behind one protocol
core/events.py         synchronous fan-out bus with a replay buffer
core/config.py         dataclasses <- config.toml <- JARVIS_* env vars

perception/frame.py    Frame: resize, grayscale, PNG encode -- all pure stdlib
perception/change.py   dHash + Hamming distance; the cheap pre-filter
perception/costs.py    Anthropic image-token arithmetic and monthly projections
perception/screen.py   capture backends (mss / null / synthetic) + PrivacyFilter
perception/vision.py   ScreenWatcher: the gated capture -> Claude pipeline

memory/store.py        SQLite: turns, facts, observations, usage
memory/embeddings.py   hashing embedder + cosine; swap in a real model if you want

voice/stt.py           faster-whisper / null / scripted
voice/tts.py           Kokoro / ElevenLabs / null
voice/wakeword.py      openWakeWord / text matcher / always-on

ui/server.py           stdlib HTTP + SSE bridge, loopback only
ui/web/                the orb: one HTML file, one JS file, no build step
```

## Three decisions worth explaining

### 1. The core has no required dependencies

`pip install jarvis` pulls nothing. Every heavy library is an optional extra,
imported inside the class that needs it, and every backend has a working offline
implementation. Three consequences, all good: the test suite runs anywhere in
seconds, a missing library degrades to `null` instead of a crash at startup, and
you can develop the FSM, memory and UI on a laptop with no GPU and no API key.

The cost is a little hand-written image code (`Frame.resized`, `Frame.to_png`).
That is a few hundred lines against a hard dependency on Pillow for something
the vision loop does a handful of times a minute.

### 2. Cheap gates run before expensive ones, always

```
capture → privacy → change detection → budget → Claude vision
```

Each stage can only reject. The ordering is the whole cost and privacy story: a
password manager window is rejected before any pixels are hashed, an unchanged
frame is rejected before any money is spent, and the budget guard is checked
against *recorded* spend so a crash-loop cannot outrun it.

`ScreenWatcher.tick()` returns a `TickResult` whose `status` names exactly which
gate stopped it (`blocked`, `unchanged`, `over-budget`, `observed`, …). That is
what `jarvis watch` prints and what the tests assert on.

### 3. There is exactly one path to unprompted speech

`ProactiveGate.check(state)` is the only thing that can authorise Jarvis to talk
without being addressed, and it demands all of:

- proactive speech enabled in config;
- the FSM in `IDLE` (never mid-turn, never while speaking, never paused);
- user idle longer than `idle_threshold_s`;
- `cooldown_s` elapsed since the last remark;
- fewer than `max_per_hour` remarks in the trailing hour;
- the wall clock outside `quiet_hours` (which may wrap past midnight).

Even then the model gets a final veto: the proactive prompt tells it to reply
`SILENCE` if the observation is not worth an interruption, and that reply is
never spoken or logged.

This is the difference between a companion and a nuisance, so it lives in one
small, heavily tested class rather than being spread across the loop.

## State machine

```
        ┌──────────────────── PAUSED ◄──── (any state)
        ▼
      IDLE ──► LISTENING ──► THINKING ──► SPEAKING ──► IDLE
        ▲          │             │            │
        └──────────┴── CANCELLED ┴────────────┘   (barge-in / Ctrl-C)
                        │
                      ERROR ──► IDLE
```

Transitions not in the table raise `InvalidTransition`; `try_to()` is the
non-raising variant used in the loops. `SPEAKING → LISTENING` exists on purpose:
that is the user talking over Jarvis.

## Data flow for one spoken turn

```
mic ─► VAD ─► STT ─► wake word ─► orchestrator.handle_utterance
                                        │ (FSM: LISTENING → THINKING)
                                        ▼
                        agent.respond ─► memory.recall + facts → system prompt
                                        │
                                        ▼  (tool round-trips: remember/recall/look_at_screen)
                                    Claude Messages API
                                        │
                        memory.add_turn ◄┘   bus.publish("reply")
                                        │ (FSM: SPEAKING)
                                        ▼
                                    TTS ─► speakers        orb.js repaints
```

## Memory

One SQLite file holds four tables: `turns` (the chat log, with an embedding per
turn), `facts` (durable key/value the agent can write through its `remember`
tool), `observations` (what the screen loop saw) and `usage` (per-day, per-model
token and dollar totals — this is what the budget guard reads).

Recall is hybrid: cosine similarity over stored vectors, falling back to keyword
overlap when embeddings are disabled, with facts weighted slightly above old
chatter. The default embedder is a stdlib hashing vectoriser — no model
download, deterministic, good enough for lexical recall. Swap in a real
embedding model by implementing the `Embedder` protocol; stored rows record
which embedder produced them.

## The orb

The Python core is the source of truth. The UI subscribes to the event bus over
SSE and repaints; it never decides anything. That means the same core can drive
a browser tab, a Tauri window, or nothing at all, and the tests can assert on
events instead of pixels.

`ui/web/orb.js` is canvas 2D on purpose: no build step, no CDN, works offline,
and it renders correctly inside a transparent always-on-top window. Each FSM
state maps to a colour, pulse speed and spin rate, eased between states so
transitions glide.

To make it a real desktop overlay, point a Tauri v2 (or Electron) shell at
`http://127.0.0.1:8765` with a transparent, always-on-top, click-through window.
Tauri is the better default in 2026 — 5–15 MB binaries and 30–80 MB RAM against
Electron's 80–150 MB and 100–300 MB — because it uses the system webview.

## What is not built yet

- **The microphone pump.** `voice/stt.py` and `voice/wakeword.py` implement the
  backends and `handle_utterance` implements the gating, but nothing yet reads
  an audio device, chunks it through VAD and feeds the transcriber. That is a
  `sounddevice` loop in a thread calling `jarvis.handle_utterance`.
- **Barge-in.** The FSM allows `SPEAKING → LISTENING` and `TTSBackend.stop()`
  exists, but no one calls it mid-utterance yet.
- **A packaged desktop shell.** The orb runs in a browser today.
- **Live2D / VRM avatars.** If you want a character rather than an orb, fork
  Open-LLM-VTuber or Project AIRI instead of reimplementing it here.
- **A local vision pre-filter.** For heavy use, a small local VLM between change
  detection and Claude would cut escalations further.
