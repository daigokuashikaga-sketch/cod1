from __future__ import annotations

from typing import Any

from jarvis.core.agent import Agent
from jarvis.core.config import AgentConfig
from jarvis.core.events import EventBus
from jarvis.core.llm import Completion, EchoBackend, Image, Message, ScriptedBackend
from jarvis.memory.store import MemoryStore


class ToolThenTextBackend:
    """Asks for one tool call, then answers with text. Records what it was sent."""

    name = "tool-then-text"

    def __init__(self, tool_name: str, tool_input: dict[str, Any]) -> None:
        self.tool_name = tool_name
        self.tool_input = tool_input
        self.systems: list[str] = []
        self.transcripts: list[list[Message]] = []
        self._served = False

    def complete(self, system: str, messages, **kwargs: Any) -> Completion:
        self.systems.append(system)
        self.transcripts.append(list(messages))
        if not self._served:
            self._served = True
            return Completion(
                text="",
                model="test",
                tool_calls=({"id": "t1", "name": self.tool_name, "input": self.tool_input},),
                stop_reason="tool_use",
            )
        return Completion(text="done", model="test", stop_reason="end_turn")


def agent(memory: MemoryStore, backend: Any, **kwargs: Any) -> Agent:
    return Agent(AgentConfig(model="test-model"), backend, memory, **kwargs)


def test_reply_is_persisted_with_the_user_turn(memory: MemoryStore) -> None:
    a = agent(memory, EchoBackend())
    reply = a.respond("hello")
    assert reply.text == "[echo] hello"
    assert [t.role for t in memory.recent_turns()] == ["user", "assistant"]


def test_history_is_replayed_to_the_model(memory: MemoryStore) -> None:
    backend = ScriptedBackend(["one", "two"])
    a = agent(memory, backend)
    a.respond("first question")
    a.respond("second question")
    second_call = backend.calls[1]
    assert [m.content for m in second_call] == [
        "first question",
        "one",
        "second question",
    ]


def test_history_is_trimmed_to_the_configured_window(memory: MemoryStore) -> None:
    backend = ScriptedBackend([], fallback="ok")
    config = AgentConfig(model="test-model", history_turns=2)
    a = Agent(config, backend, memory)
    for i in range(5):
        a.respond(f"q{i}")
    assert len(backend.calls[-1]) == 3  # 2 remembered turns + the new question


def test_persist_false_keeps_the_log_clean(memory: MemoryStore) -> None:
    agent(memory, EchoBackend()).respond("proactive musing", persist=False)
    assert memory.turn_count() == 0


def test_facts_are_injected_into_the_system_prompt(memory: MemoryStore) -> None:
    memory.remember("editor", "neovim")
    prompt = agent(memory, EchoBackend()).system_prompt()
    assert "editor: neovim" in prompt


def test_recalled_context_is_injected_for_the_current_question(memory: MemoryStore) -> None:
    memory.add_turn("user", "the staging deploy script keeps failing")
    prompt = agent(memory, EchoBackend()).system_prompt("deploy script")
    assert "Possibly relevant" in prompt
    assert "deploy" in prompt


def test_recent_observations_reach_the_prompt(memory: MemoryStore) -> None:
    memory.record_observation("a failing test run in the terminal")
    assert "failing test run" in agent(memory, EchoBackend()).system_prompt()


def test_remember_tool_writes_to_memory(memory: MemoryStore) -> None:
    backend = ToolThenTextBackend("remember", {"key": "editor", "value": "neovim"})
    reply = agent(memory, backend).respond("I use neovim")
    assert reply.tool_rounds == 1
    assert reply.text == "done"
    assert memory.get_fact("editor").value == "neovim"


def test_tool_results_are_sent_back_to_the_model(memory: MemoryStore) -> None:
    backend = ToolThenTextBackend("remember", {"key": "city", "value": "Tokyo"})
    agent(memory, backend).respond("I live in Tokyo")
    follow_up = backend.transcripts[1]
    assert follow_up[-2].blocks[0]["type"] == "tool_use"
    result = follow_up[-1].blocks[0]
    assert result["type"] == "tool_result" and result["is_error"] is False


def test_forget_tool(memory: MemoryStore) -> None:
    memory.remember("city", "Tokyo")
    agent(memory, ToolThenTextBackend("forget", {"key": "city"})).respond("forget my city")
    assert memory.get_fact("city") is None


def test_look_at_screen_tool_uses_the_injected_describer(memory: MemoryStore) -> None:
    backend = ToolThenTextBackend("look_at_screen", {})
    a = agent(memory, backend, screen_describer=lambda: "a terminal with failing tests")
    a.respond("what am I looking at?")
    assert "a terminal with failing tests" in backend.transcripts[1][-1].blocks[0]["content"]


def test_look_at_screen_degrades_when_vision_is_off(memory: MemoryStore) -> None:
    backend = ToolThenTextBackend("look_at_screen", {})
    agent(memory, backend).respond("what am I looking at?")
    assert "not enabled" in backend.transcripts[1][-1].blocks[0]["content"]


def test_a_failing_tool_is_reported_to_the_model_not_raised(memory: MemoryStore) -> None:
    backend = ToolThenTextBackend("remember", {})  # missing required arguments
    reply = agent(memory, backend).respond("remember something")
    result = backend.transcripts[1][-1].blocks[0]
    assert result["is_error"] is True
    assert "KeyError" in result["content"]
    assert reply.text == "done"


def test_unknown_tools_are_reported_as_errors(memory: MemoryStore) -> None:
    backend = ToolThenTextBackend("teleport", {})
    agent(memory, backend).respond("teleport me")
    assert "unknown tool" in backend.transcripts[1][-1].blocks[0]["content"]


def test_tool_rounds_are_bounded(memory: MemoryStore) -> None:
    class AlwaysTools:
        name = "always"

        def __init__(self) -> None:
            self.calls = 0

        def complete(self, system, messages, **kwargs) -> Completion:
            self.calls += 1
            return Completion(
                text="",
                model="test",
                tool_calls=({"id": f"t{self.calls}", "name": "recall", "input": {"query": "x"}},),
                stop_reason="tool_use",
            )

    backend = AlwaysTools()
    agent(memory, backend).respond("loop forever")
    assert backend.calls == 5  # the initial call plus MAX_TOOL_ROUNDS


def test_usage_is_recorded_when_the_backend_reports_cost(memory: MemoryStore) -> None:
    class CostlyBackend:
        name = "costly"

        def complete(self, system, messages, **kwargs) -> Completion:
            return Completion("hi", "claude-sonnet-5", 1000, 100, cost_usd=0.003)

    agent(memory, CostlyBackend()).respond("hello")
    assert memory.spend_today("claude-sonnet-5") == 0.003


def test_replies_are_published_on_the_bus(memory: MemoryStore) -> None:
    bus = EventBus()
    seen: list[str] = []
    bus.subscribe(lambda e: seen.append(e.kind))
    agent(memory, EchoBackend(), bus=bus).respond("hello")
    assert "reply" in seen


def test_images_reach_the_backend(memory: MemoryStore) -> None:
    backend = ScriptedBackend(["seen"])
    agent(memory, backend).respond("what is this", images=[Image(b"\x89PNG")])
    assert backend.calls[0][-1].images


def test_system_prompt_is_cacheable_and_stable(memory: MemoryStore) -> None:
    a = agent(memory, EchoBackend())
    assert a.system_prompt().startswith(a.config.persona)
