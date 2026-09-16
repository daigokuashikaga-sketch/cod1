from __future__ import annotations

import pytest

from jarvis.core.config import Config, load_config


def test_defaults_are_conservative() -> None:
    cfg = Config()
    # Anything that costs money or watches the screen is off until asked for.
    assert cfg.vision.enabled is False
    assert cfg.proactive.enabled is False
    assert cfg.voice.stt_backend == "null"
    assert cfg.vision.daily_budget_usd > 0


def test_toml_file_overrides_defaults(tmp_path) -> None:
    path = tmp_path / "config.toml"
    path.write_text(
        """
        [agent]
        model = "claude-haiku-4-5-20251001"
        history_turns = 4

        [vision]
        enabled = true
        capture_interval_s = 30
        title_denylist = ["vault"]

        [proactive]
        quiet_hours = [23, 7]
        """,
        encoding="utf-8",
    )
    cfg = load_config(path, env={})
    assert cfg.agent.model == "claude-haiku-4-5-20251001"
    assert cfg.agent.history_turns == 4
    assert cfg.vision.enabled is True
    assert cfg.vision.capture_interval_s == 30.0
    assert cfg.vision.title_denylist == ("vault",)
    assert cfg.proactive.quiet_hours == (23, 7)


def test_env_beats_file(tmp_path) -> None:
    path = tmp_path / "config.toml"
    path.write_text("[vision]\nenabled = false\n", encoding="utf-8")
    cfg = load_config(
        path, env={"JARVIS_VISION__ENABLED": "true", "JARVIS_AGENT__MAX_TOKENS": "77"}
    )
    assert cfg.vision.enabled is True
    assert cfg.agent.max_tokens == 77


def test_env_parses_lists_and_none() -> None:
    cfg = load_config(
        env={
            "JARVIS_VISION__TITLE_ALLOWLIST": "code, terminal",
            "JARVIS_PROACTIVE__QUIET_HOURS": "none",
        }
    )
    assert cfg.vision.title_allowlist == ("code", "terminal")
    assert cfg.proactive.quiet_hours is None


def test_unknown_keys_and_sections_are_rejected(tmp_path) -> None:
    bad_section = tmp_path / "a.toml"
    bad_section.write_text("[nope]\nx = 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown config section"):
        load_config(bad_section, env={})

    bad_key = tmp_path / "b.toml"
    bad_key.write_text("[vision]\nnope = 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown config key"):
        load_config(bad_key, env={})


def test_unknown_env_sections_are_ignored() -> None:
    load_config(env={"JARVIS_NOPE__X": "1", "NOT_JARVIS": "x", "JARVIS_NOSEPARATOR": "y"})


def test_api_keys_come_from_the_environment(monkeypatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert Config().anthropic_api_key is None
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    assert Config().anthropic_api_key == "sk-test"
