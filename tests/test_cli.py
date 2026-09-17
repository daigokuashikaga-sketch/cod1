"""The CLI surface. These run the real commands, offline."""

from __future__ import annotations

import pytest

from jarvis.cli import main


def test_demo_runs_the_whole_pipeline_offline(capsys) -> None:
    assert main(["demo"]) == 0
    out = capsys.readouterr().out
    assert "-- conversation --" in out
    assert "observed" in out and "unchanged" in out  # the change gate did its job
    assert "jarvis (unprompted)>" in out  # and the proactive path fired
    assert "over-budget" in out  # and the budget guard held


def test_demo_exercises_the_microphone_path(capsys) -> None:
    assert main(["demo"]) == 0
    out = capsys.readouterr().out
    assert "heard>  hey jarvis, what is my editor?" in out
    assert "no wake word" in out


def test_run_starts_every_subsystem_it_can(capsys, tmp_path) -> None:
    config = tmp_path / "config.toml"
    config.write_text(
        f'[agent]\nbackend = "echo"\n\n[memory]\ndb_path = "{tmp_path / "db.sqlite3"}"\n',
        encoding="utf-8",
    )
    argv = ["--config", str(config), "run", "--no-ui", "--no-voice", "--duration", "0.1"]
    assert main(argv) == 0
    out = capsys.readouterr().out
    assert "jarvis running:" in out
    assert "screen off" in out  # vision is off by default and says so
    assert "mic off" in out


def test_listen_explains_a_missing_microphone(capsys, tmp_path) -> None:
    config = tmp_path / "config.toml"
    config.write_text(
        f'[agent]\nbackend = "echo"\n\n[memory]\ndb_path = "{tmp_path / "db.sqlite3"}"\n',
        encoding="utf-8",
    )
    assert main(["--config", str(config), "listen", "--audio", "null"]) == 1
    assert "no microphone available" in capsys.readouterr().err


def test_estimate_prints_a_projection_per_model(capsys) -> None:
    assert main(["estimate", "--resolution", "1920x1080", "--interval", "10"]) == 0
    out = capsys.readouterr().out
    assert "claude-haiku-4-5-20251001" in out
    assert "claude-opus-5" in out
    assert "change gate" in out


def test_estimate_respects_a_full_change_rate(capsys) -> None:
    assert main(["estimate", "--change-rate", "1.0"]) == 0
    assert "change gate" not in capsys.readouterr().out


def test_doctor_reports_backend_availability(capsys, monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    config = tmp_path / "config.toml"
    config.write_text(f'[memory]\ndb_path = "{tmp_path / "db.sqlite3"}"\n', encoding="utf-8")
    assert main(["--config", str(config), "doctor"]) == 0
    out = capsys.readouterr().out
    assert "ANTHROPIC_API_KEY" in out and "MISSING" in out
    assert "screen capture" in out


def test_memory_command_round_trips_a_fact(capsys, tmp_path) -> None:
    config = tmp_path / "config.toml"
    config.write_text(f'[memory]\ndb_path = "{tmp_path / "db.sqlite3"}"\n', encoding="utf-8")
    argv = ["--config", str(config), "memory"]

    assert main([*argv, "--remember", "editor=neovim"]) == 0
    assert main([*argv, "--search", "editor"]) == 0
    assert "neovim" in capsys.readouterr().out

    assert main(argv) == 0
    assert "editor: neovim" in capsys.readouterr().out

    assert main([*argv, "--forget", "editor"]) == 0
    assert "forgotten" in capsys.readouterr().out


def test_memory_remember_requires_a_pair(capsys, tmp_path) -> None:
    config = tmp_path / "config.toml"
    config.write_text(f'[memory]\ndb_path = "{tmp_path / "db.sqlite3"}"\n', encoding="utf-8")
    assert main(["--config", str(config), "memory", "--remember", "editor"]) == 2


def test_a_missing_api_key_is_explained_not_traced(capsys, monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    config = tmp_path / "config.toml"
    config.write_text(
        f'[agent]\nbackend = "anthropic"\n\n[memory]\ndb_path = "{tmp_path / "db.sqlite3"}"\n',
        encoding="utf-8",
    )
    assert main(["--config", str(config), "chat"]) == 1
    err = capsys.readouterr().err
    assert "ANTHROPIC_API_KEY is not set" in err
    assert "jarvis doctor" in err


def test_a_command_is_required(capsys) -> None:
    with pytest.raises(SystemExit):
        main([])
