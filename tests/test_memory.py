from __future__ import annotations

from jarvis.memory.embeddings import HashingEmbedder, NullEmbedder, cosine, pack, unpack
from jarvis.memory.store import MemoryStore, format_context


def test_turns_round_trip_oldest_first(memory: MemoryStore) -> None:
    memory.add_turn("user", "first")
    memory.add_turn("assistant", "second")
    turns = memory.recent_turns(limit=10)
    assert [t.content for t in turns] == ["first", "second"]
    assert memory.turn_count() == 2


def test_recent_turns_is_capped(memory: MemoryStore) -> None:
    for i in range(30):
        memory.add_turn("user", f"line {i}")
    turns = memory.recent_turns(limit=5)
    assert len(turns) == 5
    assert turns[-1].content == "line 29"  # newest last, ready for the prompt


def test_turn_metadata_survives(memory: MemoryStore) -> None:
    memory.add_turn("assistant", "hello", model="claude-sonnet-5", source="proactive")
    assert memory.recent_turns()[0].meta["source"] == "proactive"


def test_facts_upsert_rather_than_duplicate(memory: MemoryStore) -> None:
    memory.remember("editor", "vim")
    memory.remember("Editor", "neovim")  # keys are normalised
    facts = memory.facts()
    assert len(facts) == 1
    assert facts[0].value == "neovim"
    assert memory.get_fact("editor").value == "neovim"


def test_forget_reports_whether_anything_went(memory: MemoryStore) -> None:
    memory.remember("city", "Tokyo")
    assert memory.forget("city") is True
    assert memory.forget("city") is False
    assert memory.get_fact("city") is None


def test_recall_finds_semantically_related_turns(memory: MemoryStore) -> None:
    memory.add_turn("user", "the staging deploy script keeps failing")
    memory.add_turn("user", "my cat is called Pepper")
    hits = memory.recall("why does the staging deploy fail")
    assert hits
    assert "deploy" in hits[0].text
    assert all("Pepper" not in hit.text for hit in hits)


def test_recall_prefers_facts_over_chatter(memory: MemoryStore) -> None:
    memory.add_turn("user", "my editor is probably neovim these days")
    memory.remember("editor", "neovim")
    hits = memory.recall("which editor do I use")
    assert hits[0].kind == "fact"


def test_recall_of_an_empty_query_is_empty(memory: MemoryStore) -> None:
    memory.add_turn("user", "anything")
    assert memory.recall("   ") == []


def test_keyword_fallback_when_embeddings_are_disabled() -> None:
    with MemoryStore(":memory:", embedder="none") as store:
        store.add_turn("user", "the deploy script keeps failing")
        hits = store.recall("deploy script")
        assert hits and "deploy" in hits[0].text


def test_usage_accumulates_per_model(memory: MemoryStore) -> None:
    memory.record_usage("claude-haiku-4-5-20251001", 1800, 100, 0.002)
    memory.record_usage("claude-haiku-4-5-20251001", 1800, 100, 0.002)
    memory.record_usage("claude-sonnet-5", 1000, 50, 0.010)
    assert memory.spend_today("claude-haiku-4-5-20251001") == 0.004
    assert memory.spend_today() == 0.014
    assert len(memory.usage_summary()) == 2


def test_observations_are_logged_newest_first(memory: MemoryStore) -> None:
    memory.record_observation("a terminal", app="Terminal")
    memory.record_observation("an editor", app="Code")
    observations = memory.recent_observations(limit=2)
    assert observations[0][1] == "an editor"
    assert observations[0][2] == "Code"


def test_state_survives_a_restart(tmp_path) -> None:
    path = tmp_path / "jarvis.sqlite3"
    with MemoryStore(path) as store:
        store.remember("name", "Daigo")
        store.add_turn("user", "hello")
    with MemoryStore(path) as store:
        assert store.get_fact("name").value == "Daigo"
        assert store.turn_count() == 1
        assert store.sessions() == ["default"]


def test_sessions_are_isolated(tmp_path) -> None:
    path = tmp_path / "jarvis.sqlite3"
    with MemoryStore(path, session="work") as work:
        work.add_turn("user", "work thing")
    with MemoryStore(path, session="home") as home:
        home.add_turn("user", "home thing")
        assert home.turn_count() == 1
        assert set(home.sessions()) == {"work", "home"}
        assert home.recent_turns()[0].content == "home thing"


def test_hashing_embedder_is_deterministic_and_normalised() -> None:
    embedder = HashingEmbedder(dim=64)
    a = embedder.embed("deploy script failing")
    assert a == embedder.embed("deploy script failing")
    assert abs(sum(v * v for v in a) - 1.0) < 1e-6
    assert cosine(a, embedder.embed("the deploy script is failing")) > cosine(
        a, embedder.embed("dinner reservation at eight")
    )


def test_embedding_pack_round_trips() -> None:
    values = HashingEmbedder(dim=32).embed("hello world")
    restored = unpack(pack(values))
    assert len(restored) == 32
    assert cosine(values, restored) > 0.999


def test_null_embedder_yields_no_vector() -> None:
    assert NullEmbedder().embed("anything") == []
    assert cosine([], [1.0]) == 0.0


def test_format_context_is_prompt_shaped(memory: MemoryStore) -> None:
    memory.remember("editor", "neovim")
    rendered = format_context(memory.recall("editor"))
    assert rendered.startswith("- (fact)")
