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
python -m jarvis doctor      # what is actually installed and configured
python -m jarvis demo        # full pipeline, offline
python -m jarvis chat        # text conversation (/help for commands)
python -m jarvis listen      # hands-free voice: wake word in, speech out
python -m jarvis ui --open   # the orb, at http://127.0.0.1:8765
python -m jarvis watch       # foreground screen loop, prints every decision
python -m jarvis estimate    # what the screen loop would cost you per month
python -m jarvis memory --remember "editor=neovim"
```

Vision and proactive speech are **off by default**. Turn them on in
`config.toml` once you have read [docs/COSTS.md](docs/COSTS.md).

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

Talking over a reply stops it: the listener fires `on_speech_start` the moment
the VAD hears you — before any transcription — and the orchestrator cuts
playback and moves `SPEAKING → LISTENING`.

## How it stays private

Screenshots are downscaled before they leave the machine; windows whose title
matches the denylist (password managers, banking, authenticators) are never
captured at all; an allowlist flips the default to deny; and `pause` stops
capture and speech immediately. A blocked frame never becomes the change
detector's baseline, so nothing leaks by comparison either.

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
pytest                       # 191 tests, offline, ~5s
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

What is *not* done: a packaged desktop shell (the orb runs in a browser),
acoustic echo cancellation, streaming partial transcripts, and Live2D/VRM
avatars — see
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md#what-is-not-built-yet).

MIT licensed.
