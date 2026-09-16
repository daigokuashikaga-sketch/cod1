#!/usr/bin/env python3
"""Generate the orb app icons with no image library, the way the core does.

Tauri wants a PNG set plus a Windows .ico at bundle time. Rather than commit
opaque binaries nobody can regenerate, draw them here: a glowing cyan orb on
transparency, same palette as ui/web/orb.js.

    python desktop/scripts/make_icons.py
"""

from __future__ import annotations

import math
import struct
import zlib
from pathlib import Path

ICON_DIR = Path(__file__).resolve().parents[1] / "src-tauri" / "icons"
SIZES = {"32x32.png": 32, "128x128.png": 128, "128x128@2x.png": 256, "icon.png": 512}
ICO_SIZES = (16, 32, 48, 64, 256)

CORE = (140, 230, 255)
RING = (60, 140, 220)


def orb_rgba(size: int) -> bytes:
    """Radial glow with a brighter ring, premultiplied into straight RGBA."""
    out = bytearray(size * size * 4)
    centre = (size - 1) / 2.0
    radius = size * 0.46
    for y in range(size):
        for x in range(size):
            dx, dy = x - centre, y - centre
            distance = math.hypot(dx, dy) / radius
            i = (y * size + x) * 4
            if distance > 1.0:
                continue
            # Core fades out toward the rim; a ring sits just inside the edge.
            core = max(0.0, 1.0 - distance**1.6)
            ring = math.exp(-((distance - 0.82) ** 2) / 0.006)
            level = min(1.0, core + ring * 0.9)
            red = CORE[0] * core + RING[0] * ring
            green = CORE[1] * core + RING[1] * ring
            blue = CORE[2] * core + RING[2] * ring
            alpha = min(1.0, level * 1.15) * min(1.0, (1.0 - distance) * 8.0)
            out[i] = min(255, int(red))
            out[i + 1] = min(255, int(green))
            out[i + 2] = min(255, int(blue))
            out[i + 3] = int(alpha * 255)
    return bytes(out)


def png(size: int, rgba: bytes) -> bytes:
    raw = bytearray()
    stride = size * 4
    for y in range(size):
        raw.append(0)  # filter type 0 (None)
        raw.extend(rgba[y * stride : (y + 1) * stride])

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    header = struct.pack(">IIBBBBB", size, size, 8, 6, 0, 0, 0)  # colour type 6 = RGBA
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(bytes(raw), 9))
        + chunk(b"IEND", b"")
    )


def ico(images: dict[int, bytes]) -> bytes:
    """An .ico is a small header plus embedded PNGs, which Windows accepts."""
    count = len(images)
    header = struct.pack("<HHH", 0, 1, count)
    offset = 6 + 16 * count
    entries, payload = bytearray(), bytearray()
    for size, data in sorted(images.items()):
        entries += struct.pack(
            "<BBBBHHII", size if size < 256 else 0, size if size < 256 else 0,
            0, 0, 1, 32, len(data), offset
        )
        payload += data
        offset += len(data)
    return header + bytes(entries) + bytes(payload)


def main() -> None:
    ICON_DIR.mkdir(parents=True, exist_ok=True)
    for name, size in SIZES.items():
        (ICON_DIR / name).write_bytes(png(size, orb_rgba(size)))
        print(f"  {name} ({size}x{size})")
    (ICON_DIR / "icon.ico").write_bytes(ico({s: png(s, orb_rgba(s)) for s in ICO_SIZES}))
    print("  icon.ico")
    print(f"written to {ICON_DIR}")


if __name__ == "__main__":
    main()
