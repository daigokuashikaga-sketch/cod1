"""Local change detection: the cheap pre-filter that keeps the vision bill survivable.

Most consecutive screen frames are identical or near-identical. Hashing a frame
locally costs microseconds; sending it to a vision model costs money. So every
frame is hashed, and only frames whose hash moved far enough are escalated.
"""

from __future__ import annotations

from dataclasses import dataclass

from .frame import Frame

HASH_SIDE = 8  # dHash compares 9x8 luma samples -> 64 bits


def dhash(frame: Frame, side: int = HASH_SIDE) -> int:
    """Difference hash: 1 bit per horizontally-adjacent pixel pair of a tiny thumbnail."""
    rows = frame.grayscale(side + 1, side)
    bits = 0
    for row in rows:
        for x in range(side):
            bits = (bits << 1) | int(row[x] < row[x + 1])
    return bits


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


@dataclass(frozen=True)
class ChangeResult:
    changed: bool
    distance: int
    hash_value: int
    reason: str

    def __bool__(self) -> bool:
        return self.changed


class ChangeDetector:
    """Stateful gate: feed it frames, it tells you which ones are worth paying for.

    ``threshold`` is a dHash Hamming distance in [0, 64]. Roughly:
      0-3    noise, a blinking cursor, a clock tick
      4-8    scrolling, a dialog opening
      9+     a different window or a different app entirely

    ``max_interval_s`` forces a periodic escalation even on a static screen, so
    Jarvis still has fresh context after the user has been reading for a while.
    """

    def __init__(self, threshold: int = 8, max_interval_s: float | None = 300.0) -> None:
        if not 0 <= threshold <= 64:
            raise ValueError("threshold must be within the 64-bit dHash range 0..64")
        self.threshold = threshold
        self.max_interval_s = max_interval_s
        self._last_hash: int | None = None
        self._last_escalated_at: float | None = None

    @property
    def last_hash(self) -> int | None:
        return self._last_hash

    def reset(self) -> None:
        self._last_hash = None
        self._last_escalated_at = None

    def evaluate(self, frame: Frame) -> ChangeResult:
        value = dhash(frame)
        if self._last_hash is None:
            self._accept(value, frame.captured_at)
            return ChangeResult(True, 64, value, "first frame")

        distance = hamming(value, self._last_hash)
        self._last_hash = value

        if distance >= self.threshold:
            self._accept(value, frame.captured_at)
            return ChangeResult(True, distance, value, f"distance {distance} >= {self.threshold}")

        if self.max_interval_s is not None and self._last_escalated_at is not None:
            elapsed = frame.captured_at - self._last_escalated_at
            if elapsed >= self.max_interval_s:
                self._accept(value, frame.captured_at)
                return ChangeResult(
                    True, distance, value, f"refresh after {elapsed:.0f}s without change"
                )

        return ChangeResult(False, distance, value, f"distance {distance} < {self.threshold}")

    def _accept(self, value: int, at: float) -> None:
        self._last_hash = value
        self._last_escalated_at = at
