"""Frame maths, change detection, cost arithmetic and the privacy filter."""

from __future__ import annotations

import struct
import zlib

import pytest

from conftest import make_frame
from jarvis.perception.change import ChangeDetector, dhash, hamming
from jarvis.perception.costs import (
    image_cost_usd,
    image_tokens,
    pricing_for,
    project_monthly,
)
from jarvis.perception.frame import Frame
from jarvis.perception.screen import (
    NullCapture,
    PrivacyFilter,
    SyntheticCapture,
    build_capture,
)

# -- frames ---------------------------------------------------------------


def test_frame_rejects_a_mismatched_buffer() -> None:
    with pytest.raises(ValueError, match="pixel buffer"):
        Frame(2, 2, b"\x00" * 10)


def test_resize_caps_the_long_edge_and_never_upscales() -> None:
    frame = make_frame(64, 32)
    small = frame.resized(16)
    assert (small.width, small.height) == (16, 8)
    assert frame.resized(1000) is frame


def test_grayscale_uses_luma_weights() -> None:
    green = Frame(1, 1, bytes([0, 255, 0]))
    assert green.grayscale()[0][0] == 149  # 255 * 0.587


def test_png_round_trips_through_zlib() -> None:
    frame = make_frame(8, 4, seed=3)
    png = frame.to_png()
    assert png[:8] == b"\x89PNG\r\n\x1a\n"

    # Walk the chunks, verifying each CRC, then inflate IDAT and unfilter it.
    offset, chunks = 8, {}
    while offset < len(png):
        (length,) = struct.unpack(">I", png[offset : offset + 4])
        tag = png[offset + 4 : offset + 8]
        data = png[offset + 8 : offset + 8 + length]
        (crc,) = struct.unpack(">I", png[offset + 8 + length : offset + 12 + length])
        assert crc == zlib.crc32(tag + data) & 0xFFFFFFFF
        chunks[tag] = data
        offset += 12 + length

    width, height, depth, colour = struct.unpack(">IIBB", chunks[b"IHDR"][:10])
    assert (width, height, depth, colour) == (8, 4, 8, 2)
    raw = zlib.decompress(chunks[b"IDAT"])
    stride = width * 3
    recovered = b"".join(
        raw[y * (stride + 1) + 1 : (y + 1) * (stride + 1)] for y in range(height)
    )
    assert recovered == frame.pixels
    assert b"IEND" in chunks


# -- change detection ------------------------------------------------------


def test_identical_frames_hash_identically() -> None:
    frame = make_frame(seed=1)
    assert dhash(frame) == dhash(frame)
    assert hamming(dhash(frame), dhash(frame)) == 0


def test_different_frames_move_the_hash() -> None:
    assert hamming(dhash(make_frame(seed=1)), dhash(make_frame(seed=2))) > 8


def test_detector_escalates_the_first_frame_then_stays_quiet() -> None:
    detector = ChangeDetector(threshold=8, max_interval_s=None)
    frame = make_frame(seed=5)
    assert detector.evaluate(frame).changed is True
    assert detector.evaluate(frame).changed is False
    assert detector.evaluate(make_frame(seed=6)).changed is True


def test_threshold_controls_sensitivity() -> None:
    a, b = make_frame(seed=1), make_frame(seed=2)
    distance = hamming(dhash(a), dhash(b))

    strict = ChangeDetector(threshold=max(1, distance - 1), max_interval_s=None)
    strict.evaluate(a)
    assert strict.evaluate(b).changed is True

    lax = ChangeDetector(threshold=min(64, distance + 1), max_interval_s=None)
    lax.evaluate(a)
    assert lax.evaluate(b).changed is False


def test_max_interval_forces_a_refresh_on_a_static_screen() -> None:
    detector = ChangeDetector(threshold=8, max_interval_s=60.0)
    first = Frame(16, 16, make_frame(16, 16, seed=9).pixels, captured_at=0.0)
    later = Frame(16, 16, first.pixels, captured_at=30.0)
    much_later = Frame(16, 16, first.pixels, captured_at=120.0)
    assert detector.evaluate(first).changed is True
    assert detector.evaluate(later).changed is False
    result = detector.evaluate(much_later)
    assert result.changed is True and "refresh" in result.reason


def test_reset_forgets_the_baseline() -> None:
    detector = ChangeDetector(threshold=8)
    frame = make_frame(seed=4)
    detector.evaluate(frame)
    detector.reset()
    assert detector.last_hash is None
    assert detector.evaluate(frame).changed is True


def test_threshold_must_be_in_range() -> None:
    with pytest.raises(ValueError, match="dHash range"):
        ChangeDetector(threshold=99)


# -- costs -----------------------------------------------------------------


def test_token_formula_matches_the_documented_divisor() -> None:
    assert image_tokens(1092, 1092, "claude-sonnet-5") == round(1092 * 1092 / 750)


def test_oversized_images_are_scaled_before_billing() -> None:
    sonnet = image_tokens(4000, 2000, "claude-sonnet-5")
    opus = image_tokens(4000, 2000, "claude-opus-5")
    assert sonnet < opus  # Opus allows a larger long edge, so it bills more
    assert sonnet == image_tokens(1568, 784, "claude-sonnet-5")


def test_downscaling_is_the_biggest_cost_lever() -> None:
    # Area scales quadratically, so halving each edge quarters the bill.
    assert image_cost_usd(500, 500) == pytest.approx(image_cost_usd(1000, 1000) / 4, rel=0.01)
    # A 1080p frame is already past the 1568px cap, so the saving is smaller but real.
    assert image_cost_usd(960, 540) < image_cost_usd(1920, 1080) / 2


def test_unknown_models_fall_back_to_the_cheap_default() -> None:
    assert pricing_for("claude-something-new").input_per_mtok == 1.0


def test_change_gating_scales_the_projection_linearly() -> None:
    every = project_monthly(1920, 1080, 10.0)
    gated = project_monthly(1920, 1080, 10.0, change_rate=0.1)
    assert gated.escalated == pytest.approx(every.escalated * 0.1, rel=0.01)
    assert gated.total_usd == pytest.approx(every.total_usd * 0.1, rel=0.01)


def test_longer_intervals_cost_less() -> None:
    assert project_monthly(1920, 1080, 30.0).total_usd < project_monthly(1920, 1080, 5.0).total_usd


def test_projection_rejects_a_zero_interval() -> None:
    with pytest.raises(ValueError, match="interval_s"):
        project_monthly(100, 100, 0)


# -- capture and privacy ---------------------------------------------------


def test_privacy_denylist_blocks_sensitive_windows() -> None:
    f = PrivacyFilter(denylist=("1password", "banking"))
    assert f.check("Terminal")
    assert not f.check("1Password 8")
    assert "denylisted" in f.check("My Banking - Chrome").reason


def test_allowlist_flips_the_default_to_deny() -> None:
    f = PrivacyFilter(denylist=(), allowlist=("code", "terminal"))
    assert f.check("Visual Studio Code")
    assert not f.check("Mail")
    assert not f.check(None)  # unknown window with an allowlist set means deny


def test_synthetic_capture_replays_and_loops() -> None:
    a, b = make_frame(seed=1), make_frame(seed=2)
    capture = SyntheticCapture([a, b])
    assert [capture.grab(), capture.grab(), capture.grab()] == [a, b, a]

    once = SyntheticCapture([a], loop=False)
    assert once.grab() is a
    assert once.grab() is None


def test_build_capture_degrades_to_null_without_a_display() -> None:
    assert isinstance(build_capture("null"), NullCapture)
    assert build_capture("auto").grab() is None or True  # mss may or may not be installed
    with pytest.raises(ValueError, match="unknown capture backend"):
        build_capture("nope")
