"""LLM のストリーム出力を、音声合成に投げられる文単位へ切り出す。

全文を待たずに1文目から合成・再生できるので、最初の発話までの体感待ち時間が
大きく縮む（数秒 → 1秒前後）。ここは純粋なロジックなのでテストしやすい。
"""

from __future__ import annotations

SENTENCE_ENDINGS = "。！？!?\n"

# 短すぎる断片（「はい。」など）は単独で合成せず次の文とまとめる。
# ただし1文目だけは速さを優先してそのまま出す。
DEFAULT_MIN_LENGTH = 8


class SentenceChunker:
    def __init__(self, min_length: int = DEFAULT_MIN_LENGTH) -> None:
        self.min_length = min_length
        self._buffer = ""
        self._emitted = 0

    def feed(self, text: str) -> list[str]:
        """ストリームの断片を追加し、確定した文のリストを返す。"""
        self._buffer += text
        sentences: list[str] = []

        while True:
            end = self._next_boundary()
            if end is None:
                break
            candidate = self._buffer[:end].strip()
            self._buffer = self._buffer[end:]
            if candidate:
                sentences.append(candidate)
                self._emitted += 1

        return sentences

    def flush(self) -> str | None:
        """バッファに残った未確定分を吐き出す（ストリーム終了時に呼ぶ）。"""
        remaining = self._buffer.strip()
        self._buffer = ""
        if not remaining:
            return None
        self._emitted += 1
        return remaining

    def _next_boundary(self) -> int | None:
        """十分な長さの文が確定していれば、その終端位置を返す。

        短すぎる文（「はい。」など）で切らず、次の句点まで読み進めて
        まとめる。まだ確定していなければ None。
        """
        threshold = 1 if self._emitted == 0 else self.min_length
        start = 0
        while True:
            index = self._find_ending(start)
            if index is None:
                return None
            end = index + 1
            if len(self._buffer[:end].strip()) >= threshold:
                return end
            start = end

    def _find_ending(self, start: int = 0) -> int | None:
        for i in range(start, len(self._buffer)):
            if self._buffer[i] in SENTENCE_ENDINGS:
                return i
        return None
