# Voice Agent Project

## 構成
- ASR: faster-whisper (large-v3, language="ja")
- LLM: Anthropic API (claude-sonnet-4-6)
- TTS: VOICEVOX (ローカルサーバー, http://localhost:50021)

## 規約
- Python 3.11, venv使用
- 環境変数はpython-dotenvで.envから読む
- APIキーはハードコードしない
