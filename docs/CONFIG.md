# Configuration

Settings resolve in this order, last wins:

1. dataclass defaults in `src/jarvis/core/config.py`
2. `config.toml` (cwd) or `~/.config/jarvis/config.toml`, or `--config PATH`
3. environment variables `JARVIS_<SECTION>__<FIELD>` (note the double underscore)

Secrets are only ever read from the environment: `ANTHROPIC_API_KEY`,
`ELEVENLABS_API_KEY`. Never put them in `config.toml`.

```bash
JARVIS_VISION__ENABLED=true JARVIS_VISION__CAPTURE_INTERVAL_S=30 jarvis watch
```

Lists accept a TOML array or a comma-separated string; `none` clears an optional
value (`JARVIS_PROACTIVE__QUIET_HOURS=none`). Unknown sections and keys in a
config *file* are errors, so a typo fails loudly instead of being ignored.

## `[agent]`

| Key | Default | Meaning |
|---|---|---|
| `backend` | `anthropic` | `anthropic` or `echo` (offline stand-in) |
| `model` | `claude-sonnet-5` | Model for conversation |
| `max_tokens` | `1024` | Reply cap |
| `temperature` | `1.0` | |
| `persona` | (see source) | The system prompt. Static, so it is cacheable |
| `history_turns` | `20` | Turns replayed into each request |
| `cache_system_prompt` | `true` | Mark the persona block with `cache_control` |

## `[vision]` — the only section that can cost real money

| Key | Default | Meaning |
|---|---|---|
| `enabled` | `false` | Master switch for screen watching |
| `monitor` | `1` | Which monitor (`mss` numbering; 1 is primary) |
| `capture_interval_s` | `10.0` | Seconds between captures |
| `max_edge_px` | `1024` | Downscale before sending; biggest cost lever |
| `model` | `claude-haiku-4-5-20251001` | Use the cheap model here |
| `max_tokens` | `300` | Description length cap |
| `change_threshold` | `8` | dHash distance 0–64 to count as changed |
| `daily_budget_usd` | `1.00` | Hard stop, checked against recorded spend |
| `title_denylist` | password managers, banking, authenticators | Never captured |
| `title_allowlist` | `[]` | If non-empty, *only* these are captured |

## `[proactive]` — when Jarvis may speak unprompted

| Key | Default | Meaning |
|---|---|---|
| `enabled` | `false` | Master switch |
| `idle_threshold_s` | `45.0` | User must be idle this long first |
| `cooldown_s` | `300.0` | Minimum gap between remarks |
| `max_per_hour` | `6` | Hard cap per rolling hour |
| `quiet_hours` | `[22, 8]` | Local-time window of silence; may wrap midnight |

## `[voice]`

| Key | Default | Meaning |
|---|---|---|
| `stt_backend` | `null` | `faster_whisper` with the `stt` extra |
| `stt_model` | `small.en` | Whisper model name |
| `tts_backend` | `null` | `kokoro` (free, local) or `elevenlabs` (paid) |
| `tts_voice` | `af_heart` | Backend-specific voice id |
| `wake_word` | `hey jarvis` | Phrase required to address Jarvis |
| `wake_word_enabled` | `true` | `false` = always listening |
| `suppress_while_speaking` | `true` | Stops Jarvis waking itself |
| `audio_backend` | `null` | `sounddevice` opens a microphone |
| `player_backend` | `null` | `sounddevice` plays replies aloud |
| `sample_rate` | `16000` | Microphone rate; Whisper wants 16kHz |
| `frame_ms` | `30` | VAD frame size |
| `vad_backend` | `energy` | `energy` (stdlib), `silero`, or `always` |
| `vad_threshold` | `0.5` | Silero only; EnergyVAD adapts to the room |
| `start_frames` | `3` | Consecutive speech frames that open an utterance |
| `pre_roll_ms` | `300` | Audio kept from before it opened, so words aren't clipped |
| `silence_hangover_ms` | `700` | Quiet needed to close an utterance |
| `min_speech_ms` | `300` | Shorter bursts are discarded untranscribed |
| `max_utterance_s` | `15.0` | Hard cut, so a stuck stream can't grow forever |
| `barge_in` | `true` | Talking over a reply stops playback and reopens the mic |
| `tts_sample_rate` | `24000` | Kokoro's output rate; must match your TTS backend |

## `[memory]`

| Key | Default | Meaning |
|---|---|---|
| `db_path` | `data/jarvis.sqlite3` | One file: turns, facts, observations, usage |
| `recall_limit` | `5` | Items injected into the prompt |
| `embedder` | `hashing` | `hashing` (stdlib) or `none` (keyword only) |
| `embedding_dim` | `256` | Changing this invalidates stored vectors |

## `[ui]`

| Key | Default | Meaning |
|---|---|---|
| `host` | `127.0.0.1` | Keep it on loopback |
| `port` | `8765` | |
| `auto_open` | `false` | Open a browser on start |

## Recipes

**Free and offline** — no API key, nothing leaves the machine:

```toml
[agent]
backend = "echo"
[vision]
enabled = false
```

**Cheap always-on watching** — a few dollars a month:

```toml
[vision]
enabled = true
model = "claude-haiku-4-5-20251001"
capture_interval_s = 30.0
max_edge_px = 1024
change_threshold = 10
daily_budget_usd = 0.25
```

**Hands-free voice** — nothing leaves the machine except the reasoning call:

```toml
[voice]
audio_backend = "sounddevice"
player_backend = "sounddevice"
stt_backend = "faster_whisper"
tts_backend = "kokoro"
vad_backend = "silero"
```

**Maximum privacy** — only ever looks at your editor and terminal:

```toml
[vision]
enabled = true
title_allowlist = ["visual studio code", "terminal", "iterm"]
```
