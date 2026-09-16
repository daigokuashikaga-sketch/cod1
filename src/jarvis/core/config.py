"""Configuration: dataclass defaults <- config.toml <- environment variables.

Environment overrides use the ``JARVIS_<SECTION>__<FIELD>`` convention, e.g.
``JARVIS_VISION__CAPTURE_INTERVAL_S=30``. Secrets only ever come from the
environment; they are never read from, or written to, config.toml.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

DEFAULT_CONFIG_PATHS = (
    Path("config.toml"),
    Path.home() / ".config" / "jarvis" / "config.toml",
)


@dataclass
class AgentConfig:
    """The reasoning loop: which model answers the user, and with what persona."""

    backend: str = "anthropic"  # "anthropic" | "echo"
    model: str = "claude-sonnet-5"
    max_tokens: int = 1024
    temperature: float = 1.0
    persona_name: str = "Jarvis"
    persona: str = (
        "You are Jarvis, a calm, dry-witted desktop companion. You share the user's "
        "screen and voice. Be concise: two sentences unless asked for more. Never "
        "narrate what you are about to do, just do it. When you have nothing useful "
        "to add, say nothing rather than filling silence."
    )
    history_turns: int = 20
    cache_system_prompt: bool = True


@dataclass
class VisionConfig:
    """The screen-watching loop. This is the only part of Jarvis that costs real money."""

    enabled: bool = False
    monitor: int = 1
    capture_interval_s: float = 10.0
    max_edge_px: int = 1024  # downscale before billing; see docs/COSTS.md
    model: str = "claude-haiku-4-5-20251001"
    max_tokens: int = 300
    # dHash distance (0-64) above which a frame counts as "meaningfully changed".
    change_threshold: int = 8
    # Hard stop so a runaway loop cannot empty your account.
    daily_budget_usd: float = 1.00
    # Windows whose title matches any of these are never captured (case-insensitive).
    title_denylist: tuple[str, ...] = (
        "1password",
        "bitwarden",
        "keepass",
        "lastpass",
        "banking",
        "authenticator",
    )
    # If non-empty, ONLY windows whose title matches one of these are captured.
    title_allowlist: tuple[str, ...] = ()


@dataclass
class ProactiveConfig:
    """When Jarvis is allowed to speak without being spoken to."""

    enabled: bool = False
    # Seconds of user idleness before proactive speech is permitted.
    idle_threshold_s: float = 45.0
    # Minimum gap between two proactive remarks.
    cooldown_s: float = 300.0
    # Hard cap per rolling hour, regardless of cooldown.
    max_per_hour: int = 6
    # Local-time hours [start, end) during which Jarvis stays quiet. 22-8 = overnight.
    quiet_hours: tuple[int, int] | None = (22, 8)


@dataclass
class VoiceConfig:
    """Speech in and speech out. Local backends are free; hosted ones are not."""

    stt_backend: str = "null"  # "null" | "faster_whisper"
    stt_model: str = "small.en"
    tts_backend: str = "null"  # "null" | "kokoro" | "elevenlabs"
    tts_voice: str = "af_heart"
    wake_word: str = "hey jarvis"
    wake_word_enabled: bool = True
    # Ignore wake-word detections while Jarvis is talking, so it cannot trigger itself.
    suppress_while_speaking: bool = True


@dataclass
class MemoryConfig:
    """Durable state: chat log, learned facts, semantic recall."""

    db_path: str = "data/jarvis.sqlite3"
    recall_limit: int = 5
    embedder: str = "hashing"  # "hashing" (stdlib, offline) | "none"
    embedding_dim: int = 256


@dataclass
class UIConfig:
    """The orb overlay. The Python core is the source of truth; the UI just renders state."""

    host: str = "127.0.0.1"
    port: int = 8765
    auto_open: bool = False


@dataclass
class Config:
    agent: AgentConfig = field(default_factory=AgentConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    proactive: ProactiveConfig = field(default_factory=ProactiveConfig)
    voice: VoiceConfig = field(default_factory=VoiceConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    ui: UIConfig = field(default_factory=UIConfig)

    @property
    def anthropic_api_key(self) -> str | None:
        return os.environ.get("ANTHROPIC_API_KEY") or None

    @property
    def elevenlabs_api_key(self) -> str | None:
        return os.environ.get("ELEVENLABS_API_KEY") or None


def _coerce(value: Any, target_type: Any) -> Any:
    """Coerce a TOML/env scalar into the type the dataclass field declares."""
    if target_type is bool:
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in {"1", "true", "yes", "on"}
    if target_type is int:
        return int(value)
    if target_type is float:
        return float(value)
    if target_type is str:
        return str(value)
    return value


def _field_type(section: Any, name: str) -> Any:
    for f in fields(section):
        if f.name == name:
            return f.type
    return None


def _apply(section: Any, values: dict[str, Any]) -> None:
    known = {f.name for f in fields(section)}
    for key, raw in values.items():
        key = key.lower()
        if key not in known:
            raise ValueError(f"unknown config key: {type(section).__name__}.{key}")
        setattr(section, key, _convert(raw, getattr(section, key), str(_field_type(section, key))))


def _convert(value: Any, current: Any, declared: str) -> Any:
    """Turn a TOML or env scalar into the shape the field expects."""
    optional = "None" in declared
    if isinstance(value, str) and optional and value.strip().lower() in {"", "none", "null"}:
        return None

    wants_tuple = isinstance(current, tuple) or "tuple" in declared
    if wants_tuple:
        items = (
            [p.strip() for p in value.split(",") if p.strip()]
            if isinstance(value, str)
            else list(value)
        )
        if "int" in declared:
            return tuple(int(item) for item in items)
        return tuple(str(item) for item in items)

    for kind in (bool, int, float, str):
        if isinstance(current, kind) or declared.startswith(kind.__name__):
            return _coerce(value, kind)
    return value


def load_config(path: str | Path | None = None, env: dict[str, str] | None = None) -> Config:
    """Build a :class:`Config` from defaults, then a TOML file, then the environment."""
    cfg = Config()
    env = dict(os.environ if env is None else env)

    candidates = [Path(path)] if path else list(DEFAULT_CONFIG_PATHS)
    for candidate in candidates:
        if candidate.is_file():
            data = tomllib.loads(candidate.read_text(encoding="utf-8"))
            for section_name, values in data.items():
                section = getattr(cfg, section_name.lower(), None)
                if section is None or not is_dataclass(section):
                    raise ValueError(f"unknown config section: [{section_name}]")
                _apply(section, values)
            break

    section_names = {f.name for f in fields(cfg)}
    for key, value in env.items():
        if not key.startswith("JARVIS_") or "__" not in key:
            continue
        section_name, _, field_name = key[len("JARVIS_") :].partition("__")
        section_name = section_name.lower()
        if section_name not in section_names:
            continue
        _apply(getattr(cfg, section_name), {field_name.lower(): value})

    return cfg
