"""A captured screen frame, plus the pure-stdlib image maths the rest of the loop needs.

Deliberately dependency-free: downscaling, grayscale and PNG encoding are all
implemented here so the capture -> change-detection path runs on a bare Python
install. Pillow and numpy, when present, are only used as faster paths.
"""

from __future__ import annotations

import struct
import time
import zlib
from dataclasses import dataclass, field

_R_WEIGHT, _G_WEIGHT, _B_WEIGHT = 299, 587, 114  # ITU-R BT.601 luma, x1000


@dataclass(frozen=True)
class Frame:
    """An RGB frame. ``pixels`` is tightly packed, 3 bytes per pixel, row-major."""

    width: int
    height: int
    pixels: bytes
    captured_at: float = field(default_factory=time.time)
    source: str = "screen"

    def __post_init__(self) -> None:
        expected = self.width * self.height * 3
        if len(self.pixels) != expected:
            raise ValueError(
                f"pixel buffer is {len(self.pixels)} bytes, expected {expected} "
                f"for {self.width}x{self.height} RGB"
            )

    @property
    def long_edge(self) -> int:
        return max(self.width, self.height)

    def scale_for(self, max_edge: int) -> float:
        """The factor that brings the long edge down to ``max_edge`` (never upscales)."""
        if max_edge <= 0 or self.long_edge <= max_edge:
            return 1.0
        return max_edge / self.long_edge

    def resized(self, max_edge: int) -> Frame:
        """Nearest-neighbour downscale so the long edge is at most ``max_edge``.

        Nearest-neighbour is the right trade here: it is cheap, and the frame is
        being sent to a vision model or hashed, not printed.
        """
        scale = self.scale_for(max_edge)
        if scale >= 1.0:
            return self
        new_w = max(1, int(self.width * scale))
        new_h = max(1, int(self.height * scale))
        return self.resample(new_w, new_h)

    def resample(self, new_w: int, new_h: int) -> Frame:
        src, w = self.pixels, self.width
        x_map = [min(self.width - 1, x * self.width // new_w) * 3 for x in range(new_w)]
        out = bytearray(new_w * new_h * 3)
        pos = 0
        for y in range(new_h):
            row_start = min(self.height - 1, y * self.height // new_h) * w * 3
            for xo in x_map:
                i = row_start + xo
                out[pos : pos + 3] = src[i : i + 3]
                pos += 3
        return Frame(new_w, new_h, bytes(out), captured_at=self.captured_at, source=self.source)

    def grayscale(self, new_w: int | None = None, new_h: int | None = None) -> list[list[int]]:
        """Luma matrix, optionally resampled first. Rows outer, columns inner."""
        frame = self if new_w is None or new_h is None else self.resample(new_w, new_h)
        px = frame.pixels
        rows: list[list[int]] = []
        for y in range(frame.height):
            base = y * frame.width * 3
            row = []
            for x in range(frame.width):
                i = base + x * 3
                row.append(
                    (px[i] * _R_WEIGHT + px[i + 1] * _G_WEIGHT + px[i + 2] * _B_WEIGHT) // 1000
                )
            rows.append(row)
        return rows

    def to_png(self) -> bytes:
        """Encode as a PNG. Used to hand frames to the Anthropic vision API."""
        raw = bytearray()
        stride = self.width * 3
        for y in range(self.height):
            raw.append(0)  # filter type 0 (None)
            raw.extend(self.pixels[y * stride : (y + 1) * stride])

        def chunk(tag: bytes, data: bytes) -> bytes:
            return (
                struct.pack(">I", len(data))
                + tag
                + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
            )

        header = struct.pack(">IIBBBBB", self.width, self.height, 8, 2, 0, 0, 0)
        return (
            b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", header)
            + chunk(b"IDAT", zlib.compress(bytes(raw), 6))
            + chunk(b"IEND", b"")
        )
