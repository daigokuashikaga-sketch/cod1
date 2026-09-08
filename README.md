# 音声対話エージェント

faster-whisper（音声認識）+ Anthropic API（応答生成）+ VOICEVOX（音声合成）による
日本語の音声対話エージェント。

## セットアップ

```bash
# 1. venv を作って依存を入れる
python3.11 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. 環境変数を用意する
cp .env.example .env
# .env を開いて ANTHROPIC_API_KEY を設定する

# 3. VOICEVOX ENGINE を起動する（別ターミナル）
docker run --rm -p 50021:50021 voicevox/voicevox_engine:cpu-ubuntu20.04-latest
```

Linux では sounddevice が PortAudio を必要とします。

```bash
sudo apt install libportaudio2
```

macOS / Windows は wheel に同梱されているため追加インストールは不要です。

## 使い方

### 対話ループ（全部つないだもの）

```bash
python -m voice_agent.agent
```

話しかけると、録音 → 文字起こし → 応答生成 → 音声合成 → 再生が回ります。
「終了」「さようなら」と言うか Ctrl+C で終わります。

応答は 1 文が確定するたびに合成して再生キューへ送るので、1 文目を喋りながら
2 文目を合成する形になり、最初の発話までの待ち時間が短くなります。

### 個別に動かす

```bash
# 録音
python -m voice_agent.audio.record --list-devices
python -m voice_agent.audio.record -o recordings/input.wav          # 無音で自動停止
python -m voice_agent.audio.record -o recordings/input.wav --seconds 5
python -m voice_agent.audio.record --threshold 500 --silence-duration 1.5

# 文字起こし
python -m voice_agent.stt.whisper recordings/input.wav
python -m voice_agent.stt.whisper recordings/input.wav --model medium   # 軽いモデルで試す

# 応答生成
python -m voice_agent.llm.anthropic_client "こんにちは"

# 音声合成
python -m voice_agent.tts.voicevox --list-speakers
python -m voice_agent.tts.voicevox "こんにちは、テストです。"
python -m voice_agent.tts.voicevox "保存だけする" -o out.wav
```

### テスト

```bash
python -m unittest discover -s tests -t . -v
```

音声デバイス・API キー・VOICEVOX なしで動く範囲（無音検出、文分割、WAV 入出力、
VOICEVOX クライアント、応答生成の履歴管理）を検証します。

## 構成

```
voice_agent/
├── config.py             # .env から読む設定（APIキーはコードに書かない）
├── agent.py              # 対話ループ
├── audio/
│   ├── record.py         # マイク入力 → WAV
│   └── playback.py       # WAV の再生とキュー
├── stt/
│   └── whisper.py        # faster-whisper で文字起こし
├── llm/
│   └── anthropic_client.py  # Anthropic API で応答生成（ストリーミング）
└── tts/
    ├── chunker.py        # ストリーム出力を文単位に切り出す
    └── voicevox.py       # VOICEVOX で音声合成
```

データの流れ:

```
マイク → 無音検出 → WAV
  → faster-whisper → テキスト
  → Anthropic API（ストリーミング）→ 文単位に分割
  → VOICEVOX → 再生キュー → スピーカー
```

## 設定

すべて `.env` で上書きできます（既定値は `.env.example` を参照）。

| 変数 | 既定値 | 用途 |
|---|---|---|
| `ANTHROPIC_API_KEY` | （必須） | Anthropic API キー |
| `ANTHROPIC_MODEL` | `claude-sonnet-4-6` | 応答生成に使うモデル |
| `ANTHROPIC_MAX_TOKENS` | `1024` | 読み上げ前提なので短め |
| `VOICEVOX_URL` | `http://localhost:50021` | ENGINE のエンドポイント |
| `VOICEVOX_SPEAKER` | `3` | 話者ID（`--list-speakers` で確認） |
| `VOICEVOX_SPEED_SCALE` | `1.0` | 読み上げ速度 |
| `WHISPER_MODEL` | `large-v3` | 重い場合は `medium` / `small` |
| `WHISPER_DEVICE` | `cpu` | GPU があれば `cuda` |
| `WHISPER_COMPUTE_TYPE` | `int8` | `cuda` なら `float16` |

## ロードマップ

- [x] 1. requirements.txt と venv のセットアップ
- [x] 2. マイク入力を録音して WAV に保存
- [x] 3. faster-whisper で WAV を文字起こし
- [x] 4. 文字起こし結果を Anthropic API に投げて応答を得る
- [x] 5. VOICEVOX で音声合成して再生
- [x] 6. 1〜5 を繋いだ対話ループ
- [ ] 7. 認識状況を表示する画面
- [ ] 8. 割り込み（発話中に話しかけたら止まる）
