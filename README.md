# Jarvis

A desktop-resident AI companion built on Claude: it watches your screen, listens
for a wake word, remembers what matters across restarts, speaks up on its own
when something is worth saying — and shows all of it as a glowing orb.

```
$ python -m jarvis demo          # no API key, no display, no microphone needed
-- conversation --
you>    my editor is neovim
jarvis> Noted - neovim.

-- voice (synthetic microphone, no audio device needed) --
  heard>  hey jarvis, what is my editor?
  heard>  just mumbling to myself
  jarvis> You use neovim.
  (the second utterance had no wake word, so it was ignored)

-- perception (5 synthetic frames) --
  frame 1: observed    first frame
  frame 2: unchanged   distance 0 < 8
  frame 3: observed    distance 28 >= 8
    jarvis (unprompted)> Three tests just went red in tests/test_memory.py.
  frame 4: observed    distance 28 >= 8
  frame 5: observed    distance 32 >= 8

-- budget guard --
  with a $0 daily cap: over-budget
```

## What it does

| Capability | How |
|---|---|
| **Sees your screen** | `mss` capture → downscale → perceptual hash → Claude vision *only* when something changed |
| **Hears you** | a VAD-gated microphone loop: one transcription per utterance, behind a `hey jarvis` wake word |
| **Talks back** | Kokoro-82M locally (free) or ElevenLabs — with interruptible playback, so you can talk over it |
| **Remembers** | SQLite chat log + durable facts + embedding recall, all in one file |
| **Speaks up on its own** | An FSM plus an idle/cooldown/hourly-cap/quiet-hours gate |
| **Shows itself** | A canvas orb that glows, pulses and spins with the core's state |
| **Sits on your desktop** | A Tauri shell: frameless, transparent, always-on-top, click-through |

Everything heavy is optional. The core imports with **no third-party
dependencies at all**, and every backend has an offline stand-in, so you can
develop and test the whole thing on a headless box with no API key.

## Install

```bash
uv sync                      # core only
uv sync --extra all          # + Claude, screen capture, voice, vectors
uv sync --extra dev          # + pytest, ruff, mypy
# or: pip install -e '.[all,dev]'

cp .env.example .env         # put your ANTHROPIC_API_KEY here
cp config.example.toml config.toml
```

## Use

```bash
python -m jarvis run         # everything at once: orb + screen loop + microphone
python -m jarvis doctor      # what is actually installed and configured
python -m jarvis demo        # full pipeline, offline
python -m jarvis chat        # text conversation (/help for commands)
python -m jarvis listen      # hands-free voice: wake word in, speech out
python -m jarvis ui --open   # the orb, at http://127.0.0.1:8765
python -m jarvis watch       # foreground screen loop, prints every decision
python -m jarvis estimate    # what the screen loop would cost you per month
python -m jarvis memory --remember "editor=neovim"

cd desktop/src-tauri && cargo run   # the orb as a desktop overlay (see desktop/README.md)
```

Vision and proactive speech are **off by default**. Turn them on in
`config.toml` once you have read [docs/COSTS.md](docs/COSTS.md).

## What it costs to think

Conversation is cheap, but only if prompt caching actually works — and caching
fails silently in two different ways. A cache entry is a byte-exact prefix
match, so Jarvis splits the system prompt: the persona is the cached half, and
the clock, your remembered facts and recalled context all sit *after* the
breakpoint. And below a model-dependent minimum (512 tokens on Opus 5, 1,024 on
Sonnet 5, 4,096 on Haiku 4.5) the API accepts the marker, caches nothing, and
says nothing — so Jarvis estimates the prefix and declines to ask for caching
that cannot happen. `jarvis doctor` tells you which case you are in:

```
 + prompt cache   no-op   static prefix ~336 tokens < 1024 for claude-sonnet-5
```

That is the shipped default: caching starts paying only once your static half
grows (a long persona, house rules, a style guide). `jarvis memory` prints the
read/write counters so you can check rather than assume.

## How it stays cheap

Continuously sending screenshots to a vision model is what runs up a bill — at
one 1080p frame every 10 seconds, 8 hours a day, that is roughly **$224/month on
Haiku and over $1,500 on Opus**. So the screen loop pays for a frame only after
it has failed to reject it locally:

```
capture (µs) → privacy filter (µs) → dHash change detection (ms) → budget guard → Claude
```

A local difference hash rejects every frame that looks like the last one, which
on real desktop use is most of them. `jarvis estimate` will do the arithmetic
for your own resolution, interval and change rate:

```
model                        tokens/img  escalated   $/month
------------------------------------------------------------
claude-haiku-4-5-20251001          1844      86400    224.12
  ^ with 12% change gate           1844      10368     26.89
```

Downscaling is the other big lever (cost scales with pixel area), and
`vision.daily_budget_usd` is a hard stop that refuses the call once today's
recorded spend would exceed it.

## How it stays polite

Proactive speech has exactly one path into the world, `ProactiveGate.check()`,
and it requires *all* of: proactive speech enabled, the FSM in `IDLE`, the user
idle past a threshold, a cooldown elapsed, an hourly cap not reached, and the
clock outside quiet hours. Then the model still gets to answer `SILENCE`.

## How it hears you

The microphone loop is the screen loop rotated ninety degrees: a cheap local
gate in front of an expensive step.

```
frames → VAD (µs) → segmentation → one STT call per utterance → wake word → agent
```

Nothing is transcribed per frame. An utterance opens after three consecutive
speech frames (so a cough does not), keeps 300ms of pre-roll (so the first
syllable is not clipped), closes after 700ms of quiet (so a pause mid-sentence
does not split it), and is discarded untranscribed if it turns out to be under
300ms of actual speech.

**It does not hear itself.** By default the microphone is closed while Jarvis is
speaking, plus a short tail for the speakers to settle. Without that, a
companion on speakers hears its own voice, decides someone is talking, and
interrupts its own reply — the one bug that makes a voice assistant unusable in
a room. The cost is that barge-in needs `voice.duplex = "full"`, which in turn
needs headphones; then talking over a reply cuts it off mid-sentence, because
the listener fires `on_speech_start` before any transcription and the
orchestrator stops playback and moves `SPEAKING → LISTENING`.

## How it stays private

Screenshots are downscaled before they leave the machine; windows whose title
matches the denylist (password managers, banking, authenticators) are never
captured at all; an allowlist flips the default to deny; and `pause` stops
capture and speech immediately. A blocked frame never becomes the change
detector's baseline, so nothing leaks by comparison either.

## On the desktop

`jarvis ui` is enough to use the orb in a browser tab. `desktop/` wraps the same
page in a [Tauri v2](https://tauri.app) window that floats above everything,
has no frame, and lets clicks pass through to whatever is behind it — except on
the orb's own disc.

The shell stays thin on purpose: it points a webview at the core and owns only
what a browser cannot do (floating, click-through, optionally starting and
stopping the core). Navigation is pinned to the core's origin, because a
frameless always-on-top window with no address bar should never be able to show
an arbitrary page.

```bash
jarvis ui                          # terminal 1: the core
cd desktop/src-tauri && cargo run  # terminal 2: the window
```

See [desktop/README.md](desktop/README.md) — including what has and has not been
verified.

## Architecture

```
          ┌───────────── orchestrator (core/orchestrator.py) ─────────────┐
          │  the only place subsystems are wired to each other            │
          └───┬─────────────┬──────────────┬─────────────┬────────────────┘
        state machine    agent          watcher        voice
        + proactive   (core/agent)   (perception/)   (voice/)
            gate           │              │             │
                           └── memory (SQLite + embeddings) ──┘
                                          │
                                     event bus ──→ UI server (SSE) ──→ orb
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the reasoning,
[docs/CONFIG.md](docs/CONFIG.md) for every setting, and
[CLAUDE.md](CLAUDE.md) for the rules this codebase is maintained by.

## Development

```bash
pytest                       # 227 tests, offline, ~6s
ruff check src tests
mypy
```

Tests never touch the network, a display, a microphone or an API key: they use
`EchoBackend`/`ScriptedBackend`, `SyntheticCapture`, `SyntheticAudioSource`,
`NullTTS` and injected clocks. If a change needs a real service to be tested, the seam is in the wrong
place.

## Status

All five phases of the build plan are implemented: scaffold, text+memory, the
voice loop (microphone → VAD → STT → wake word → TTS → interruptible playback,
with barge-in), gated screen vision, and the orb UI.

What is *not* done: the desktop shell compiles and passes clippy but has never
been launched on a real display, so its window behaviour is unverified; true
acoustic echo cancellation (half duplex avoids the problem instead of solving
it); streaming partial transcripts; and Live2D/VRM avatars — see
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md#what-is-not-built-yet).

MIT licensed.
