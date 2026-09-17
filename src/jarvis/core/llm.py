"""LLM backends.

``AnthropicBackend`` talks to the Claude Messages API (lazy import, so the rest
of Jarvis runs without the SDK installed). ``EchoBackend`` and
``ScriptedBackend`` are deterministic offline stand-ins used by the tests,
``jarvis demo`` and any machine without an API key. All three speak the same
small vocabulary: text in, text plus token counts out, with optional images and
tools.
"""

from __future__ import annotations

import base64
import json
import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from ..perception.costs import call_cost_usd, is_cacheable


@dataclass(frozen=True)
class Image:
    data: bytes
    media_type: str = "image/png"

    def to_block(self) -> dict[str, Any]:
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": self.media_type,
                "data": base64.b64encode(self.data).decode("ascii"),
            },
        }


@dataclass(frozen=True)
class Message:
    """One turn. ``blocks``, when set, is passed to the API verbatim.

    Raw blocks are how tool_use / tool_result round-trips are represented
    without the rest of Jarvis having to know the wire format.
    """

    role: str  # "user" | "assistant"
    content: str = ""
    images: tuple[Image, ...] = ()
    blocks: tuple[dict[str, Any], ...] = ()

    def to_api(self) -> dict[str, Any]:
        if self.blocks:
            return {"role": self.role, "content": [dict(b) for b in self.blocks]}
        if not self.images:
            return {"role": self.role, "content": self.content}
        image_blocks: list[dict[str, Any]] = [image.to_block() for image in self.images]
        if self.content:
            image_blocks.append({"type": "text", "text": self.content})
        return {"role": self.role, "content": image_blocks}


@dataclass(frozen=True)
class SystemPrompt:
    """A system prompt split at the cache boundary.

    Caching is a prefix match, so everything before the breakpoint must be
    byte-identical between requests. ``static`` is the persona: long, frozen,
    worth caching. ``volatile`` is what changes every turn -- the clock,
    remembered facts, recalled context -- and must come *after* the breakpoint,
    or it invalidates the entry it is sitting in and the cache never hits.
    """

    static: str = ""
    volatile: str = ""

    def text(self) -> str:
        return "\n\n".join(part for part in (self.static, self.volatile) if part)

    def __str__(self) -> str:
        return self.text()


def as_system_prompt(system: SystemPrompt | str) -> SystemPrompt:
    return system if isinstance(system, SystemPrompt) else SystemPrompt(static=system)


@dataclass(frozen=True)
class Completion:
    text: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    stop_reason: str | None = None
    tool_calls: tuple[dict[str, Any], ...] = ()
    raw: Any = field(default=None, repr=False)


@runtime_checkable
class LLMBackend(Protocol):
    name: str

    def complete(
        self,
        system: SystemPrompt | str,
        messages: Sequence[Message],
        *,
        model: str,
        max_tokens: int = 1024,
        temperature: float = 1.0,
        tools: Sequence[dict[str, Any]] | None = None,
        cache_system: bool = False,
    ) -> Completion: ...


class EchoBackend:
    """Offline backend. Deterministic, free, and obviously not intelligent.

    It exists so that every layer above it -- the FSM, memory, the orb UI, the
    perception loop -- can be developed and tested end to end with no API key.
    """

    name = "echo"

    def __init__(self, prefix: str = "[echo]") -> None:
        self.prefix = prefix
        self.calls: list[dict[str, Any]] = []

    def complete(
        self,
        system: SystemPrompt | str,
        messages: Sequence[Message],
        *,
        model: str = "echo",
        max_tokens: int = 1024,
        temperature: float = 1.0,
        tools: Sequence[dict[str, Any]] | None = None,
        cache_system: bool = False,
    ) -> Completion:
        self.calls.append({"system": system, "messages": list(messages), "model": model})
        last = next((m for m in reversed(messages) if m.role == "user"), None)
        body = last.content if last else ""
        images = sum(len(m.images) for m in messages)
        if images:
            body = f"{body} (+{images} image{'s' if images > 1 else ''})".strip()
        text = f"{self.prefix} {body}".strip()
        return Completion(
            text=text,
            model=model,
            input_tokens=len(body.split()),
            output_tokens=len(text.split()),
            cost_usd=0.0,
            stop_reason="end_turn",
        )


class ScriptedBackend:
    """Replays canned replies in order. For tests and demos of a specific flow."""

    name = "scripted"

    def __init__(self, replies: Sequence[str], fallback: str = "") -> None:
        self._replies = list(replies)
        self.fallback = fallback
        self.calls: list[list[Message]] = []

    def complete(
        self,
        system: SystemPrompt | str,
        messages: Sequence[Message],
        *,
        model: str = "scripted",
        max_tokens: int = 1024,
        temperature: float = 1.0,
        tools: Sequence[dict[str, Any]] | None = None,
        cache_system: bool = False,
    ) -> Completion:
        self.calls.append(list(messages))
        text = self._replies.pop(0) if self._replies else self.fallback
        return Completion(
            text=text,
            model=model,
            input_tokens=0,
            output_tokens=len(text.split()),
            cost_usd=0.0,
            stop_reason="end_turn",
        )


class AnthropicBackend:
    """Claude via the Messages API.

    The system prompt is marked with ``cache_control`` when ``cache_system`` is
    set: the persona block is static and long, so caching it is close to free
    money. Note that a screenshot changes every frame and therefore can never be
    cached -- that is why the change detector, not caching, is the cost lever.
    """

    name = "anthropic"

    def __init__(self, api_key: str | None = None, client: Any | None = None) -> None:
        if client is not None:
            self._client = client
            return
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set")
        try:
            import anthropic  # optional dependency, imported on demand
        except ImportError as exc:  # pragma: no cover - depends on the host
            raise RuntimeError(
                "the Anthropic backend needs: pip install 'jarvis[claude]'"
            ) from exc
        self._client = anthropic.Anthropic(api_key=key)

    def complete(
        self,
        system: SystemPrompt | str,
        messages: Sequence[Message],
        *,
        model: str,
        max_tokens: int = 1024,
        temperature: float = 1.0,
        tools: Sequence[dict[str, Any]] | None = None,
        cache_system: bool = False,
    ) -> Completion:
        prompt = as_system_prompt(system)
        system_blocks: list[dict[str, Any]] = [{"type": "text", "text": prompt.static}]
        if cache_system and self.would_cache(prompt.static, model, tools):
            system_blocks[0]["cache_control"] = {"type": "ephemeral"}
        if prompt.volatile:
            # After the breakpoint on purpose: this is the part that changes.
            system_blocks.append({"type": "text", "text": prompt.volatile})

        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "system": system_blocks,
            "messages": [m.to_api() for m in messages],
        }
        if tools:
            kwargs["tools"] = list(tools)

        response = self._client.messages.create(**kwargs)
        return _completion_from_response(response, model)

    @staticmethod
    def would_cache(
        static: str, model: str, tools: Sequence[dict[str, Any]] | None = None
    ) -> bool:
        """Whether a breakpoint here would create a real cache entry.

        The cacheable prefix is everything the API renders before the
        breakpoint -- tools first, then the static system block -- so the tool
        definitions count toward the model's minimum.
        """
        prefix = json.dumps(list(tools)) if tools else ""
        return is_cacheable(prefix + static, model)


def _completion_from_response(response: Any, model: str) -> Completion:
    text_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    for block in getattr(response, "content", []) or []:
        block_type = getattr(block, "type", None)
        if block_type == "text":
            text_parts.append(getattr(block, "text", ""))
        elif block_type == "tool_use":
            tool_calls.append(
                {
                    "id": getattr(block, "id", ""),
                    "name": getattr(block, "name", ""),
                    "input": getattr(block, "input", {}) or {},
                }
            )

    usage = getattr(response, "usage", None)
    input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
    output_tokens = int(getattr(usage, "output_tokens", 0) or 0)
    cache_read = int(getattr(usage, "cache_read_input_tokens", 0) or 0)
    cache_write = int(getattr(usage, "cache_creation_input_tokens", 0) or 0)
    return Completion(
        text="\n".join(part for part in text_parts if part).strip(),
        model=getattr(response, "model", model) or model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=call_cost_usd(input_tokens, output_tokens, model, cache_read, cache_write),
        cache_read_tokens=cache_read,
        cache_write_tokens=cache_write,
        stop_reason=getattr(response, "stop_reason", None),
        tool_calls=tuple(tool_calls),
        raw=response,
    )


def build_backend(name: str, api_key: str | None = None) -> LLMBackend:
    """Build a backend, degrading to :class:`EchoBackend` when Claude is unreachable."""
    if name == "echo":
        return EchoBackend()
    if name == "scripted":
        return ScriptedBackend([])
    if name == "anthropic":
        return AnthropicBackend(api_key=api_key)
    if name == "auto":
        try:
            return AnthropicBackend(api_key=api_key)
        except RuntimeError:
            return EchoBackend()
    raise ValueError(f"unknown LLM backend: {name}")
