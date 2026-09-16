"""Embeddings for semantic recall.

The default ``HashingEmbedder`` is a stdlib hashing vectoriser: no model
download, no GPU, no API call, deterministic across runs. It captures lexical
overlap rather than deep semantics, which is enough for "what did we say about
the deploy script" -- and it keeps the memory subsystem testable offline.

Swap in a real embedding model by implementing :class:`Embedder`; stored vectors
carry the embedder name so a mismatch can be detected and re-indexed.
"""

from __future__ import annotations

import array
import hashlib
import math
import re
from typing import Protocol, runtime_checkable

_TOKEN = re.compile(r"[a-z0-9']+")
_STOPWORD_TEXT = (
    "a an and are as at be but by for from has have i in is it its of on or that the to was were "
    "what when where which who will with you your me my we us do does did not no yes"
)
_STOPWORDS = frozenset(_STOPWORD_TEXT.split())


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOPWORDS and len(t) > 1]


@runtime_checkable
class Embedder(Protocol):
    name: str
    dim: int

    def embed(self, text: str) -> list[float]: ...


class HashingEmbedder:
    """Signed hashing trick + sublinear term weighting, L2-normalised."""

    name = "hashing"

    def __init__(self, dim: int = 256) -> None:
        if dim < 8:
            raise ValueError("embedding dim must be at least 8")
        self.dim = dim

    def embed(self, text: str) -> list[float]:
        vector = array.array("d", [0.0]) * self.dim
        counts: dict[str, int] = {}
        for token in tokenize(text):
            counts[token] = counts.get(token, 0) + 1
        for token, count in counts.items():
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "big") % self.dim
            sign = 1.0 if digest[4] & 1 else -1.0
            vector[index] += sign * (1.0 + math.log(count))
        norm = math.sqrt(sum(v * v for v in vector))
        if norm == 0.0:
            return [0.0] * self.dim
        return [v / norm for v in vector]


class NullEmbedder:
    """Disables semantic recall; the store falls back to keyword search."""

    name = "none"
    dim = 0

    def embed(self, text: str) -> list[float]:
        return []


def cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    # Vectors from the embedders above are already normalised, but do not assume it.
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def pack(vector: list[float]) -> bytes:
    return array.array("f", vector).tobytes()


def unpack(blob: bytes) -> list[float]:
    values = array.array("f")
    values.frombytes(blob)
    return list(values)


def build_embedder(name: str, dim: int = 256) -> Embedder:
    if name == "hashing":
        return HashingEmbedder(dim)
    if name in {"none", "null", ""}:
        return NullEmbedder()
    raise ValueError(f"unknown embedder: {name}")
