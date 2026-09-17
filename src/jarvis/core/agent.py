"""The reasoning loop: persona, memory injection, and the tool round-trip.

The agent owns *what Jarvis says*. It does not own the microphone, the speaker,
the screen or the orb -- the orchestrator wires those in. Keeping it that narrow
means the whole conversational core can be exercised in a unit test.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from ..memory.store import MemoryStore, format_context
from .config import AgentConfig
from .events import EventBus
from .llm import Completion, Image, LLMBackend, Message, SystemPrompt

TOOLS: list[dict[str, Any]] = [
    {
        "name": "remember",
        "description": (
            "Store a durable fact about the user or their setup, so it survives restarts. "
            "Use for stable preferences and identity, not for passing chatter."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "key": {"type": "string", "description": "Short stable key, e.g. 'editor'."},
                "value": {"type": "string", "description": "The fact itself."},
            },
            "required": ["key", "value"],
        },
    },
    {
        "name": "recall",
        "description": "Search durable facts and past conversation for context you lack.",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
    {
        "name": "forget",
        "description": "Delete a stored fact by key, when the user says it is wrong or stale.",
        "input_schema": {
            "type": "object",
            "properties": {"key": {"type": "string"}},
            "required": ["key"],
        },
    },
    {
        "name": "look_at_screen",
        "description": (
            "Capture and describe what is on screen right now. Only call this when the "
            "user's request actually depends on what they are looking at: it costs money."
        ),
        "input_schema": {"type": "object", "properties": {}},
    },
]

MAX_TOOL_ROUNDS = 4


@dataclass
class Reply:
    text: str
    completion: Completion
    tool_rounds: int = 0

    @property
    def cost_usd(self) -> float:
        return self.completion.cost_usd


class Agent:
    def __init__(
        self,
        config: AgentConfig,
        backend: LLMBackend,
        memory: MemoryStore,
        bus: EventBus | None = None,
        screen_describer: Callable[[], str] | None = None,
        now: Callable[[], datetime] = datetime.now,
    ) -> None:
        self.config = config
        self.backend = backend
        self.memory = memory
        self.bus = bus
        self.screen_describer = screen_describer
        self.now = now

    # -- prompt assembly ---------------------------------------------------

    def system_prompt(self, query: str = "") -> SystemPrompt:
        """Persona in the cacheable half, everything that moves in the other.

        The split is not cosmetic. A cache entry is a byte-exact prefix match,
        so putting the clock or the user's facts before the breakpoint would
        invalidate the entry on every single turn: you would pay the 1.25x
        write surcharge forever and never read a hit.
        """
        volatile: list[str] = [f"Right now it is {self.now():%A %d %B %Y, %H:%M} local time."]

        facts = self.memory.facts()
        if facts:
            rendered = "\n".join(f"- {fact.key}: {fact.value}" for fact in facts)
            volatile.append(f"What you know about the user:\n{rendered}")

        if query:
            recalled = self.memory.recall(query, limit=self.memory_recall_limit)
            if recalled:
                volatile.append(f"Possibly relevant from earlier:\n{format_context(recalled)}")

        observations = self.memory.recent_observations(limit=3)
        if observations:
            rendered = "\n".join(f"- {summary}" for _, summary, _, _ in observations)
            volatile.append(f"Recent things you noticed on screen:\n{rendered}")

        return SystemPrompt(static=self.config.persona, volatile="\n\n".join(volatile))

    @property
    def memory_recall_limit(self) -> int:
        return 5

    def _history(self) -> list[Message]:
        turns = self.memory.recent_turns(limit=self.config.history_turns)
        return [Message(role=t.role, content=t.content) for t in turns if t.role != "system"]

    # -- the loop ----------------------------------------------------------

    def respond(
        self,
        user_text: str,
        images: Sequence[Image] = (),
        persist: bool = True,
        source: str = "user",
    ) -> Reply:
        """One user turn in, one assistant turn out, tools resolved in between."""
        history = self._history()
        if persist:
            self.memory.add_turn("user", user_text, source=source)
        messages = [*history, Message("user", user_text, images=tuple(images))]
        system = self.system_prompt(user_text)

        rounds = 0
        completion = self._call(system, messages)
        while completion.tool_calls and rounds < MAX_TOOL_ROUNDS:
            rounds += 1
            messages.append(
                Message("assistant", blocks=tuple(self._assistant_blocks(completion)))
            )
            results = [self._run_tool(call) for call in completion.tool_calls]
            messages.append(Message("user", blocks=tuple(results)))
            completion = self._call(system, messages)

        text = completion.text.strip()
        if persist and text:
            self.memory.add_turn("assistant", text, model=completion.model)
        if completion.cost_usd:
            self.memory.record_usage(
                completion.model,
                completion.input_tokens,
                completion.output_tokens,
                completion.cost_usd,
                cache_read_tokens=completion.cache_read_tokens,
                cache_write_tokens=completion.cache_write_tokens,
            )
        if self.bus is not None:
            self.bus.publish(
                "reply", text=text, model=completion.model, cost_usd=completion.cost_usd
            )
        return Reply(text=text, completion=completion, tool_rounds=rounds)

    def _call(self, system: SystemPrompt, messages: Sequence[Message]) -> Completion:
        return self.backend.complete(
            system,
            messages,
            model=self.config.model,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            tools=TOOLS,
            cache_system=self.config.cache_system_prompt,
        )

    @staticmethod
    def _assistant_blocks(completion: Completion) -> list[dict[str, Any]]:
        blocks: list[dict[str, Any]] = []
        if completion.text:
            blocks.append({"type": "text", "text": completion.text})
        for call in completion.tool_calls:
            blocks.append(
                {
                    "type": "tool_use",
                    "id": call["id"],
                    "name": call["name"],
                    "input": call["input"],
                }
            )
        return blocks

    def _run_tool(self, call: dict[str, Any]) -> dict[str, Any]:
        name = call.get("name", "")
        args = call.get("input", {}) or {}
        try:
            result = self._dispatch(name, args)
            is_error = False
        except Exception as exc:  # surfaced back to the model, not crashed on
            result = f"{type(exc).__name__}: {exc}"
            is_error = True
        if self.bus is not None:
            self.bus.publish("tool", name=name, args=args, error=is_error)
        return {
            "type": "tool_result",
            "tool_use_id": call.get("id", ""),
            "content": result if isinstance(result, str) else json.dumps(result),
            "is_error": is_error,
        }

    def _dispatch(self, name: str, args: dict[str, Any]) -> str:
        if name == "remember":
            fact = self.memory.remember(str(args["key"]), str(args["value"]), source="agent")
            return f"stored {fact.key} = {fact.value}"
        if name == "recall":
            hits = self.memory.recall(str(args.get("query", "")), limit=5)
            return format_context(hits) or "nothing relevant found"
        if name == "forget":
            return "forgotten" if self.memory.forget(str(args["key"])) else "no such fact"
        if name == "look_at_screen":
            if self.screen_describer is None:
                return "screen vision is not enabled right now"
            return self.screen_describer()
        raise ValueError(f"unknown tool: {name}")
