"""設定。値は .env から読み、コードにキーを埋め込まない。"""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()

# 音声フォーマット（録音・認識で共通）
SAMPLE_RATE = 16_000


def _env(name: str, default: str) -> str:
    value = os.getenv(name)
    return default if value is None or value == "" else value


@dataclass(frozen=True)
class WhisperConfig:
    model: str = _env("WHISPER_MODEL", "large-v3")
    device: str = _env("WHISPER_DEVICE", "cpu")
    compute_type: str = _env("WHISPER_COMPUTE_TYPE", "int8")
    language: str = "ja"


@dataclass(frozen=True)
class AnthropicConfig:
    model: str = _env("ANTHROPIC_MODEL", "claude-sonnet-4-6")
    # 音声で読み上げる前提なので短く返させる
    max_tokens: int = int(_env("ANTHROPIC_MAX_TOKENS", "1024"))
    system_prompt: str = _env(
        "ANTHROPIC_SYSTEM_PROMPT",
        "あなたは音声で会話するアシスタントです。"
        "返答は読み上げられるため、1〜3文の短い話し言葉で答えてください。"
        "箇条書き・記号・マークダウンは使わず、地の文だけで話してください。",
    )

    @staticmethod
    def api_key() -> str | None:
        """SDK が環境変数から解決するので通常は不要。事前チェック用。"""
        return os.getenv("ANTHROPIC_API_KEY")


@dataclass(frozen=True)
class VoicevoxConfig:
    base_url: str = _env("VOICEVOX_URL", "http://localhost:50021")
    speaker: int = int(_env("VOICEVOX_SPEAKER", "3"))
    speed_scale: float = float(_env("VOICEVOX_SPEED_SCALE", "1.0"))


WHISPER = WhisperConfig()
ANTHROPIC = AnthropicConfig()
VOICEVOX = VoicevoxConfig()
