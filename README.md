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
```

Linux では sounddevice が PortAudio を必要とします。

```bash
sudo apt install libportaudio2
```

macOS / Windows は wheel に同梱されているため追加インストールは不要です。

## 使い方

### 録音

```bash
# 入力デバイスの確認
python -m voice_agent.audio.record --list-devices

# 無音を検出したら自動で停止（既定）
python -m voice_agent.audio.record -o recordings/input.wav

# 5秒固定で録音
python -m voice_agent.audio.record -o recordings/input.wav --seconds 5

# 環境ノイズが多くて止まらない / すぐ止まる場合はしきい値を調整
python -m voice_agent.audio.record --threshold 500 --silence-duration 1.5
```

出力は 16kHz / モノラル / 16bit PCM の WAV で、faster-whisper にそのまま渡せます。

### テスト

```bash
python -m unittest discover -s tests -t . -v
```

音声デバイスに依存しないロジック（無音検出・WAV 書き出し）のみを検証します。

## 構成

```
voice_agent/
└── audio/
    └── record.py     # マイク入力 → WAV        （step 2 ✅）
scripts/
└── list_devices.py   # デバイス一覧の確認
tests/
└── test_record.py
```

## ロードマップ

- [x] 1. requirements.txt と venv のセットアップ
- [x] 2. マイク入力を録音して WAV に保存
- [ ] 3. faster-whisper で WAV を文字起こし
- [ ] 4. 文字起こし結果を Anthropic API に投げて応答を得る
- [ ] 5. VOICEVOX で音声合成して再生
- [ ] 6. 1〜5 を繋いだ対話ループ
- [ ] 7. 認識状況を表示する画面
