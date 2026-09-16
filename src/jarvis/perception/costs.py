"""Vision cost arithmetic, so the screen loop can refuse to bankrupt you.

Anthropic bills images by tokens, approximately ``width * height / 750``, after
scaling the image down so its long edge fits the model's limit. Prices below are
USD per million tokens and move over time -- re-check them against the live
pricing page before trusting a monthly projection.
"""

from __future__ import annotations

from dataclasses import dataclass

TOKENS_PER_PIXEL_DIVISOR = 750


@dataclass(frozen=True)
class ModelPricing:
    name: str
    input_per_mtok: float
    output_per_mtok: float
    max_image_edge_px: int = 1568


PRICING: dict[str, ModelPricing] = {
    "claude-haiku-4-5-20251001": ModelPricing("claude-haiku-4-5-20251001", 1.0, 5.0, 1568),
    "claude-sonnet-5": ModelPricing("claude-sonnet-5", 2.0, 10.0, 1568),
    "claude-opus-5": ModelPricing("claude-opus-5", 5.0, 25.0, 2576),
}
DEFAULT_PRICING = PRICING["claude-haiku-4-5-20251001"]


def pricing_for(model: str) -> ModelPricing:
    if model in PRICING:
        return PRICING[model]
    for key, value in PRICING.items():
        if model.startswith(key.split("-2025")[0]):
            return value
    return DEFAULT_PRICING


def image_tokens(width: int, height: int, model: str = DEFAULT_PRICING.name) -> int:
    """Billed input tokens for one image, accounting for the model's downscale cap."""
    limit = pricing_for(model).max_image_edge_px
    long_edge = max(width, height)
    if long_edge > limit:
        scale = limit / long_edge
        width, height = int(width * scale), int(height * scale)
    return max(1, round(width * height / TOKENS_PER_PIXEL_DIVISOR))


def image_cost_usd(width: int, height: int, model: str = DEFAULT_PRICING.name) -> float:
    return image_tokens(width, height, model) * pricing_for(model).input_per_mtok / 1_000_000


def call_cost_usd(
    input_tokens: int, output_tokens: int, model: str = DEFAULT_PRICING.name
) -> float:
    p = pricing_for(model)
    return (input_tokens * p.input_per_mtok + output_tokens * p.output_per_mtok) / 1_000_000


@dataclass(frozen=True)
class Projection:
    model: str
    frames: int
    escalated: int
    per_image_usd: float
    total_usd: float


def project_monthly(
    width: int,
    height: int,
    interval_s: float,
    hours_per_day: float = 8.0,
    days: int = 30,
    change_rate: float = 1.0,
    model: str = DEFAULT_PRICING.name,
    output_tokens: int = 150,
) -> Projection:
    """Project a month of always-on capture.

    ``change_rate`` is the fraction of frames the local detector actually
    escalates. Measured on real desktop use it is usually well under 0.2, which
    is precisely why the pre-filter exists.
    """
    if interval_s <= 0:
        raise ValueError("interval_s must be positive")
    frames = int(hours_per_day * 3600 / interval_s * days)
    escalated = int(frames * max(0.0, min(1.0, change_rate)))
    per_image = image_cost_usd(width, height, model)
    per_call = per_image + call_cost_usd(0, output_tokens, model)
    return Projection(model, frames, escalated, per_image, escalated * per_call)
