"""Anthropic API で応答を生成する。

    python -m voice_agent.llm.anthropic_client "こんにちは"

APIキーは .env の ANTHROPIC_API_KEY から読む（SDK が環境変数を解決する）。
音声対話なので thinking は使わず、ストリーミングで先頭から文字を受け取る。
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterator

from voice_agent.config import ANTHROPIC, AnthropicConfig

# 履歴として保持する最大メッセージ数（user/assistant 合計）
MAX_HISTORY_MESSAGES = 20


class ResponseGenerator:
    """会話履歴を持ち、ユーザー発話に対する応答をストリーミングで返す。"""

    def __init__(self, config: AnthropicConfig = ANTHROPIC, client=None) -> None:
        self.config = config
        self._client = client
        self.history: list[dict] = []

    @property
    def client(self):
        if self._client is None:
            import anthropic

            if self.config.api_key() is None:
                raise SystemExit(
                    "ANTHROPIC_API_KEY が設定されていません。"
                    ".env.example を .env にコピーしてキーを設定してください。"
                )
            self._client = anthropic.Anthropic()
        return self._client

    def reset(self) -> None:
        self.history.clear()

    def _trim(self) -> None:
        if len(self.history) > MAX_HISTORY_MESSAGES:
            # 先頭が user になるよう偶数個だけ残す
            self.history[:] = self.history[-MAX_HISTORY_MESSAGES:]
            while self.history and self.history[0]["role"] != "user":
                self.history.pop(0)

    def stream_reply(self, user_text: str) -> Iterator[str]:
        """応答を細切れのテキストで順次返す。完了後に履歴へ追加する。"""
        self.history.append({"role": "user", "content": user_text})
        self._trim()

        chunks: list[str] = []
        with self.client.messages.stream(
            model=self.config.model,
            max_tokens=self.config.max_tokens,
            system=self.config.system_prompt,
            # ストリーム中に履歴へ追記するのでコピーを渡す
            messages=list(self.history),
        ) as stream:
            for text in stream.text_stream:
                chunks.append(text)
                yield text

        self.history.append({"role": "assistant", "content": "".join(chunks)})

    def reply(self, user_text: str) -> str:
        """応答をまとめて1つの文字列で返す。"""
        return "".join(self.stream_reply(user_text))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Anthropic API で応答を生成する")
    parser.add_argument("text", nargs="?", help="ユーザー発話。省略すると標準入力から読む")
    args = parser.parse_args(argv)

    user_text = args.text if args.text is not None else sys.stdin.read().strip()
    if not user_text:
        print("入力が空です。", file=sys.stderr)
        return 1

    generator = ResponseGenerator()
    for chunk in generator.stream_reply(user_text):
        print(chunk, end="", flush=True)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
