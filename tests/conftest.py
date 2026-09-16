"""Shared fixtures. Everything here is stdlib-only so the suite runs anywhere."""

from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:  # belt and braces alongside pyproject's pythonpath
    sys.path.insert(0, str(SRC))

from jarvis.core.config import Config  # noqa: E402
from jarvis.core.events import EventBus  # noqa: E402
from jarvis.memory.store import MemoryStore  # noqa: E402
from jarvis.perception.frame import Frame  # noqa: E402


def make_frame(width: int = 32, height: int = 24, seed: int = 0, fill: int | None = None) -> Frame:
    """A deterministic frame: solid colour when ``fill`` is given, else pseudo-random."""
    if fill is not None:
        return Frame(width, height, bytes([fill]) * (width * height * 3))
    rng = random.Random(seed)
    return Frame(width, height, bytes(rng.randrange(256) for _ in range(width * height * 3)))


@pytest.fixture
def memory() -> MemoryStore:
    store = MemoryStore(":memory:")
    yield store
    store.close()


@pytest.fixture
def bus() -> EventBus:
    return EventBus()


@pytest.fixture
def config(tmp_path: Path) -> Config:
    cfg = Config()
    cfg.memory.db_path = str(tmp_path / "jarvis.sqlite3")
    cfg.agent.backend = "echo"
    return cfg
