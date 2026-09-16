"""Screen capture backends and the privacy filter that sits in front of them.

``MSSCapture`` is the real one. ``NullCapture`` and ``SyntheticCapture`` let the
whole perception loop run (and be tested) on a headless machine with no display.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from .frame import Frame


@runtime_checkable
class ScreenCapture(Protocol):
    """Anything that can hand back the current screen as a :class:`Frame`."""

    def grab(self, monitor: int = 1) -> Frame | None: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class PrivacyDecision:
    allowed: bool
    reason: str

    def __bool__(self) -> bool:
        return self.allowed


class PrivacyFilter:
    """Blocks capture of sensitive windows.

    A denylist stops the obvious offenders (password managers, banking, 2FA). An
    allowlist, when non-empty, flips the default to deny -- the safer mode, and
    the one to use if you ever leave this running unattended.
    """

    def __init__(
        self,
        denylist: Sequence[str] = (),
        allowlist: Sequence[str] = (),
    ) -> None:
        self.denylist = tuple(item.lower() for item in denylist if item)
        self.allowlist = tuple(item.lower() for item in allowlist if item)

    def check(self, window_title: str | None) -> PrivacyDecision:
        title = (window_title or "").lower()
        if self.allowlist:
            if not title:
                return PrivacyDecision(False, "allowlist set but active window is unknown")
            for allowed in self.allowlist:
                if allowed in title:
                    return PrivacyDecision(True, f"allowlisted ({allowed})")
            return PrivacyDecision(False, "window not on allowlist")
        for denied in self.denylist:
            if denied and denied in title:
                return PrivacyDecision(False, f"denylisted ({denied})")
        return PrivacyDecision(True, "ok")


class NullCapture:
    """Captures nothing. The default on machines without a display."""

    def grab(self, monitor: int = 1) -> Frame | None:
        return None

    def close(self) -> None:
        return None


class SyntheticCapture:
    """Replays a fixed list of frames. Used by the tests and by ``jarvis demo``."""

    def __init__(self, frames: Sequence[Frame], loop: bool = True) -> None:
        if not frames:
            raise ValueError("SyntheticCapture needs at least one frame")
        self._frames = list(frames)
        self._loop = loop
        self._index = 0

    def grab(self, monitor: int = 1) -> Frame | None:
        if self._index >= len(self._frames):
            if not self._loop:
                return None
            self._index = 0
        frame = self._frames[self._index]
        self._index += 1
        return frame

    def close(self) -> None:
        return None


class MSSCapture:
    """Real screen capture via ``mss`` (fast, cross-platform, MIT)."""

    def __init__(self) -> None:
        try:
            import mss  # optional dependency, imported on demand
        except ImportError as exc:  # pragma: no cover - depends on the host
            raise RuntimeError(
                "screen capture needs the 'screen' extra: pip install 'jarvis[screen]'"
            ) from exc
        self._sct = mss.mss()

    def grab(self, monitor: int = 1) -> Frame | None:  # pragma: no cover - needs a display
        monitors = self._sct.monitors
        if monitor >= len(monitors):
            monitor = 0 if len(monitors) == 1 else 1
        shot = self._sct.grab(monitors[monitor])
        # mss hands back BGRA; Frame wants tightly packed RGB.
        bgra = shot.raw
        rgb = bytearray(shot.width * shot.height * 3)
        rgb[0::3] = bgra[2::4]
        rgb[1::3] = bgra[1::4]
        rgb[2::3] = bgra[0::4]
        return Frame(shot.width, shot.height, bytes(rgb), source=f"monitor{monitor}")

    def close(self) -> None:  # pragma: no cover - needs a display
        self._sct.close()


def active_window_title() -> str | None:
    """Best-effort active-window title, used by :class:`PrivacyFilter`.

    Returns ``None`` when the platform hook is unavailable; with an allowlist
    configured that is treated as "deny", which is the intended failure mode.
    """
    import platform  # only needed on this path
    import shutil
    import subprocess

    system = platform.system()
    try:
        if system == "Linux" and shutil.which("xdotool"):
            out = subprocess.run(
                ["xdotool", "getactivewindow", "getwindowname"],
                capture_output=True,
                text=True,
                timeout=1.0,
                check=False,
            )
            return out.stdout.strip() or None
        if system == "Darwin" and shutil.which("osascript"):
            script = (
                'tell application "System Events" to get name of first application '
                "process whose frontmost is true"
            )
            out = subprocess.run(
                ["osascript", "-e", script],
                capture_output=True,
                text=True,
                timeout=1.0,
                check=False,
            )
            return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None
    return None


def build_capture(backend: str = "auto") -> ScreenCapture:
    """Pick a capture backend; fall back to :class:`NullCapture` when unavailable."""
    if backend == "null":
        return NullCapture()
    if backend in {"auto", "mss"}:
        try:
            return MSSCapture()
        except RuntimeError:
            if backend == "mss":
                raise
            return NullCapture()
    raise ValueError(f"unknown capture backend: {backend}")


TitleProvider = Callable[[], str | None]
