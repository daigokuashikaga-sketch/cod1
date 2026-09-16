"""The Tauri shell's contract with the Python core.

The shell cannot be compiled or launched from the test suite, so these tests
guard the seams where the two halves agree with each other: the port, the
window flags that make it an overlay at all, the URL the bootstrap page waits
on, and the hit-test radius that has to match what orb.js draws.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from jarvis.core.config import UIConfig

DESKTOP = Path(__file__).resolve().parents[1] / "desktop"
TAURI = DESKTOP / "src-tauri"
MAIN_RS = (TAURI / "src" / "main.rs").read_text(encoding="utf-8")
BOOTSTRAP = (DESKTOP / "web" / "index.html").read_text(encoding="utf-8")
WEB = Path(__file__).resolve().parents[1] / "src" / "jarvis" / "ui" / "web"


@pytest.fixture(scope="module")
def config() -> dict:
    return json.loads((TAURI / "tauri.conf.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def window(config: dict) -> dict:
    return config["app"]["windows"][0]


def test_the_window_is_an_overlay_not_an_app_window(window: dict) -> None:
    assert window["label"] == "main"
    assert window["transparent"] is True
    assert window["decorations"] is False
    assert window["alwaysOnTop"] is True
    assert window["skipTaskbar"] is True
    assert window["resizable"] is False


def test_the_window_is_built_in_code_so_navigation_can_be_gated(window: dict) -> None:
    # create:false hands window creation to setup(), which is the only place
    # on_navigation can be attached in Tauri v2.
    assert window["create"] is False
    assert "WebviewWindowBuilder::from_config" in MAIN_RS
    assert "on_navigation" in MAIN_RS


def test_navigation_is_limited_to_the_core_and_bundled_assets() -> None:
    assert "fn navigation_allowed" in MAIN_RS
    assert "url.origin() == core.origin()" in MAIN_RS


def test_the_shell_and_the_core_agree_on_the_port() -> None:
    match = re.search(r'DEFAULT_CORE_URL: &str = "([^"]+)"', MAIN_RS)
    assert match, "the shell must declare the core URL it attaches to"
    assert match.group(1) == f"http://{UIConfig().host}:{UIConfig().port}"


def test_the_bootstrap_page_waits_for_the_core_then_hands_over() -> None:
    assert "/api/status" in BOOTSTRAP  # polls until the core answers
    assert "?mode=orb" in BOOTSTRAP  # then loads the orb-only view


def test_the_orb_page_has_an_overlay_mode() -> None:
    orb_js = (WEB / "orb.js").read_text(encoding="utf-8")
    index = (WEB / "index.html").read_text(encoding="utf-8")
    assert "mode') === 'orb'" in orb_js
    assert "body.overlay" in index
    assert "body.overlay #orb { width: 100%; height: 100%; }" in index


def test_the_hit_test_radius_covers_the_drawn_orb() -> None:
    """The shell hit-tests a circle; orb.js draws rings out to 0.42 of the window."""
    match = re.search(r"ORB_RADIUS_RATIO: f64 = ([0-9.]+)", MAIN_RS)
    assert match
    ratio = float(match.group(1))
    assert 0.42 <= ratio <= 0.5, "must cover the rings without swallowing the corners"


def test_spawning_the_core_is_opt_in() -> None:
    # Attaching to a running core is the default; the shell only owns the
    # process when JARVIS_CORE_CMD says so.
    assert 'std::env::var("JARVIS_CORE_CMD").ok()?' in MAIN_RS
    assert "child.kill()" in MAIN_RS  # and stops it again on exit


def test_bundled_icons_exist_and_are_real_images(config: dict) -> None:
    icons = config["bundle"]["icon"]
    assert icons, "the bundle needs icons"
    for relative in icons:
        path = TAURI / relative
        assert path.is_file(), f"missing icon: {relative}"
        magic = path.read_bytes()[:8]
        if path.suffix == ".png":
            assert magic == b"\x89PNG\r\n\x1a\n"
        else:
            assert magic[:4] == b"\x00\x00\x01\x00"  # ICO header


def test_icons_are_reproducible_from_the_generator() -> None:
    assert (DESKTOP / "scripts" / "make_icons.py").is_file()


def test_capabilities_cover_the_window(config: dict) -> None:
    capability = json.loads(
        (TAURI / "capabilities" / "default.json").read_text(encoding="utf-8")
    )
    assert capability["windows"] == ["main"]
    assert capability["permissions"] == ["core:default"]  # the orb needs nothing else
