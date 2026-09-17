"""Prompt caching: does the breakpoint create an entry, and does anything read it?

Caching is a byte-exact prefix match, and the API accepts a `cache_control`
marker on a prefix too short to cache without saying so. Both failure modes are
silent and both cost money, so they get tests.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, ClassVar

import pytest

from jarvis.core.agent import TOOLS, Agent
from jarvis.core.config import AgentConfig
from jarvis.core.llm import AnthropicBackend, Completion, Message, SystemPrompt
from jarvis.memory.store import MemoryStore
from jarvis.perception.costs import call_cost_usd, is_cacheable

LONG_PERSONA = "You are Jarvis. " * 500  # comfortably over every model's minimum


class FakeResponse:
    def __init__(self, **usage: int) -> None:
        self.content = [type("Block", (), {"type": "text", "text": "hello"})()]
        self.model = "claude-sonnet-5"
        self.stop_reason = "end_turn"
        self.usage = type("Usage", (), usage)()


class FakeMessages:
    DEFAULT_USAGE: ClassVar[dict[str, int]] = {"input_tokens": 100, "output_tokens": 20}

    def __init__(self, usage: dict[str, int] | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        # An explicit {} means "a response whose usage reports nothing".
        self.usage = self.DEFAULT_USAGE if usage is None else usage

    def create(self, **kwargs: Any) -> FakeResponse:
        self.calls.append(kwargs)
        return FakeResponse(**self.usage)


class FakeClient:
    def __init__(self, usage: dict[str, int] | None = None) -> None:
        self.messages = FakeMessages(usage)


def backend(usage: dict[str, int] | None = None) -> tuple[AnthropicBackend, FakeMessages]:
    client = FakeClient(usage)
    return AnthropicBackend(client=client), client.messages


# -- the split -------------------------------------------------------------


def test_the_prompt_is_sent_as_two_blocks() -> None:
    api, sent = backend()
    api.complete(
        SystemPrompt(static=LONG_PERSONA, volatile="it is 09:00"),
        [Message("user", "hi")],
        model="claude-sonnet-5",
        cache_system=True,
    )
    blocks = sent.calls[0]["system"]
    assert [block["text"] for block in blocks] == [LONG_PERSONA, "it is 09:00"]


def test_only_the_static_block_carries_the_breakpoint() -> None:
    api, sent = backend()
    api.complete(
        SystemPrompt(static=LONG_PERSONA, volatile="volatile"),
        [Message("user", "hi")],
        model="claude-sonnet-5",
        cache_system=True,
    )
    static, volatile = sent.calls[0]["system"]
    assert static["cache_control"] == {"type": "ephemeral"}
    assert "cache_control" not in volatile  # everything after the breakpoint


def test_a_prefix_too_short_to_cache_is_not_marked() -> None:
    """The API would accept the marker, cache nothing, and charge 1.25x to do it."""
    api, sent = backend()
    api.complete(
        SystemPrompt(static="You are Jarvis."),
        [Message("user", "hi")],
        model="claude-sonnet-5",
        cache_system=True,
    )
    assert "cache_control" not in sent.calls[0]["system"][0]


def test_tool_definitions_count_toward_the_minimum() -> None:
    """Tools render before the system prompt, so they are part of the prefix."""
    persona = "You are Jarvis. " * 200  # ~800 tokens: short of Sonnet's 1024 alone
    assert not is_cacheable(persona, "claude-sonnet-5")
    assert AnthropicBackend.would_cache(persona, "claude-sonnet-5", TOOLS)


def test_the_shipped_persona_is_too_short_to_cache() -> None:
    """Worth knowing rather than wondering: the default prompt never caches.

    Persona (~67 tokens) plus the tool definitions (~270) is nowhere near the
    512-4096 a model needs before an entry exists. Caching only starts paying
    once the static half is substantial -- a long persona, house rules, a style
    guide. Until then the code correctly declines to ask for it.
    """
    persona = AgentConfig().persona
    for model in ("claude-opus-5", "claude-sonnet-5", "claude-haiku-4-5-20251001"):
        assert not AnthropicBackend.would_cache(persona, model, TOOLS)


def test_the_minimum_is_per_model_and_not_monotonic() -> None:
    persona = "You are Jarvis. " * 200  # ~800 tokens, plus ~270 of tools
    assert AnthropicBackend.would_cache(persona, "claude-opus-5", TOOLS)  # minimum 512
    assert not AnthropicBackend.would_cache(persona, "claude-haiku-4-5-20251001", TOOLS)  # 4096


def test_caching_can_be_switched_off() -> None:
    api, sent = backend()
    api.complete(
        SystemPrompt(static=LONG_PERSONA),
        [Message("user", "hi")],
        model="claude-sonnet-5",
        cache_system=False,
    )
    assert "cache_control" not in sent.calls[0]["system"][0]


def test_an_empty_volatile_half_sends_one_block() -> None:
    api, sent = backend()
    api.complete(SystemPrompt(static=LONG_PERSONA), [Message("user", "hi")], model="claude-opus-5")
    assert len(sent.calls[0]["system"]) == 1


def test_a_plain_string_system_prompt_still_works() -> None:
    api, sent = backend()
    api.complete("just a string", [Message("user", "hi")], model="claude-opus-5")
    assert sent.calls[0]["system"][0]["text"] == "just a string"


# -- the invalidators ------------------------------------------------------


def clock() -> datetime:
    return datetime(2026, 9, 17, 9, 30)


def test_the_cached_half_is_byte_identical_across_turns(memory: MemoryStore) -> None:
    """The whole point: whatever changes must not move the cached prefix."""
    from jarvis.core.llm import ScriptedBackend

    api = ScriptedBackend([], fallback="ok")
    agent = Agent(AgentConfig(persona=LONG_PERSONA), api, memory, now=clock)

    first = agent.system_prompt("question one")
    memory.remember("editor", "neovim")
    memory.record_observation("a terminal")
    second = agent.system_prompt("a completely different question")

    assert first.static == second.static
    assert first.volatile != second.volatile  # and the change landed after it


def test_the_clock_lives_after_the_breakpoint(memory: MemoryStore) -> None:
    """A timestamp in the cached block is the classic silent cache killer."""
    from jarvis.core.llm import ScriptedBackend

    agent = Agent(AgentConfig(), ScriptedBackend([], fallback="ok"), memory, now=clock)
    prompt = agent.system_prompt()
    assert "09:30" in prompt.volatile
    assert "09:30" not in prompt.static


def test_the_agent_knows_what_time_it_is(memory: MemoryStore) -> None:
    from jarvis.core.llm import ScriptedBackend

    agent = Agent(AgentConfig(), ScriptedBackend([], fallback="ok"), memory, now=clock)
    assert "Thursday 17 September 2026" in agent.system_prompt().volatile


# -- the accounting --------------------------------------------------------


def test_cache_reads_are_billed_at_a_tenth() -> None:
    uncached = call_cost_usd(1000, 0, "claude-sonnet-5")
    cached = call_cost_usd(0, 0, "claude-sonnet-5", cache_read_tokens=1000)
    assert cached == pytest.approx(uncached * 0.1)


def test_cache_writes_carry_a_surcharge() -> None:
    plain = call_cost_usd(1000, 0, "claude-sonnet-5")
    written = call_cost_usd(0, 0, "claude-sonnet-5", cache_write_tokens=1000)
    assert written == pytest.approx(plain * 1.25)


def test_two_requests_break_even_on_a_cached_prefix() -> None:
    uncached_twice = call_cost_usd(2000, 0, "claude-sonnet-5")
    write_then_read = call_cost_usd(0, 0, "claude-sonnet-5", cache_write_tokens=1000) + (
        call_cost_usd(0, 0, "claude-sonnet-5", cache_read_tokens=1000)
    )
    assert write_then_read < uncached_twice


def test_usage_counters_reach_memory(memory: MemoryStore) -> None:
    api, _ = backend(
        {
            "input_tokens": 50,
            "output_tokens": 20,
            "cache_read_input_tokens": 900,
            "cache_creation_input_tokens": 0,
        }
    )
    agent = Agent(AgentConfig(model="claude-sonnet-5"), api, memory, now=clock)
    reply = agent.respond("hello")
    assert reply.completion.cache_read_tokens == 900
    assert memory.cache_stats() == {"reads": 900, "writes": 0, "calls": 1}


def test_a_completion_without_cache_fields_is_handled(memory: MemoryStore) -> None:
    api, _ = backend({"input_tokens": 50, "output_tokens": 20})
    agent = Agent(AgentConfig(model="claude-sonnet-5"), api, memory, now=clock)
    assert agent.respond("hello").completion.cache_read_tokens == 0


def test_an_old_database_gains_the_cache_columns(tmp_path) -> None:
    """An existing jarvis.sqlite3 must survive the upgrade that added them."""
    import sqlite3

    path = tmp_path / "old.sqlite3"
    legacy = sqlite3.connect(path)
    legacy.execute(
        "CREATE TABLE usage (day TEXT NOT NULL, model TEXT NOT NULL, calls INTEGER NOT NULL"
        " DEFAULT 0, input_tokens INTEGER NOT NULL DEFAULT 0, output_tokens INTEGER NOT NULL"
        " DEFAULT 0, cost_usd REAL NOT NULL DEFAULT 0.0, PRIMARY KEY (day, model))"
    )
    legacy.execute("INSERT INTO usage VALUES ('2026-01-01', 'claude-sonnet-5', 3, 10, 5, 0.5)")
    legacy.commit()
    legacy.close()

    with MemoryStore(path) as store:
        store.record_usage("claude-sonnet-5", 10, 5, 0.01, cache_read_tokens=700)
        assert store.cache_stats()["reads"] == 700
        assert any(row["cost_usd"] == 0.5 for row in store.usage_summary())  # old row intact


def test_completion_from_a_response_without_usage() -> None:
    api, _ = backend({})
    completion: Completion = api.complete(
        SystemPrompt(static="x"), [Message("user", "hi")], model="claude-sonnet-5"
    )
    assert completion.input_tokens == 0 and completion.cost_usd == 0.0
