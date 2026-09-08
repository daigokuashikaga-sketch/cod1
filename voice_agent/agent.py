"""録音 → 文字起こし → 応答生成 → 音声合成 → 再生 の対話ループ。

    python -m voice_agent.agent

1文が確定するたびに合成して再生キューへ送るので、「1文目を喋りながら
2文目を合成する」形になり、最初の発話までの待ち時間が短い。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from voice_agent.audio.playback import PlaybackQueue
from voice_agent.audio.record import record_until_silence, save_wav
from voice_agent.config import ANTHROPIC, VOICEVOX
from voice_agent.llm.anthropic_client import ResponseGenerator
from voice_agent.stt.whisper import Transcriber
from voice_agent.tts.chunker import SentenceChunker
from voice_agent.tts.voicevox import VoicevoxClient, VoicevoxError

EXIT_PHRASES = ("終了", "さようなら", "バイバイ", "またね")
LAST_INPUT_PATH = Path("recordings/last_input.wav")


def preflight(voicevox: VoicevoxClient) -> bool:
    """起動前に必要なものが揃っているか確認する。"""
    ok = True

    if ANTHROPIC.api_key() is None:
        print("✗ ANTHROPIC_API_KEY が未設定です（.env を確認してください）", file=sys.stderr)
        ok = False
    else:
        print(f"✓ Anthropic API ({ANTHROPIC.model})")

    if not voicevox.is_available():
        print(f"✗ VOICEVOX に接続できません: {VOICEVOX.base_url}", file=sys.stderr)
        print("  docker run --rm -p 50021:50021 voicevox/voicevox_engine:cpu-ubuntu20.04-latest", file=sys.stderr)
        ok = False
    else:
        print(f"✓ VOICEVOX ({VOICEVOX.base_url}, speaker={VOICEVOX.speaker})")

    return ok


def run(device: int | None = None, threshold: float | None = None) -> int:
    voicevox = VoicevoxClient()
    if not preflight(voicevox):
        return 1

    transcriber = Transcriber()
    generator = ResponseGenerator()
    playback = PlaybackQueue()

    # モデルのロードを先に済ませておく（初回の応答が遅くならないように）
    _ = transcriber.model

    print("\n準備完了。話しかけてください。終了するには Ctrl+C。\n")

    try:
        while True:
            audio = record_until_silence(
                device=device,
                **({"threshold": threshold} if threshold is not None else {}),
            )
            if audio.size == 0:
                continue

            save_wav(LAST_INPUT_PATH, audio)

            started = time.perf_counter()
            user_text = transcriber.transcribe(LAST_INPUT_PATH)
            stt_elapsed = time.perf_counter() - started
            if not user_text:
                print("（聞き取れませんでした）\n")
                continue

            print(f"あなた: {user_text}  [認識 {stt_elapsed:.1f}秒]")

            if any(phrase in user_text for phrase in EXIT_PHRASES):
                voicevox_speak_safely(voicevox, playback, "またお話ししましょう。")
                playback.wait()
                print("終了します。")
                return 0

            print("エージェント: ", end="", flush=True)
            chunker = SentenceChunker()
            first_audio_at: float | None = None
            started = time.perf_counter()

            for delta in generator.stream_reply(user_text):
                print(delta, end="", flush=True)
                for sentence in chunker.feed(delta):
                    if voicevox_speak_safely(voicevox, playback, sentence) and first_audio_at is None:
                        first_audio_at = time.perf_counter() - started

            tail = chunker.flush()
            if tail:
                if voicevox_speak_safely(voicevox, playback, tail) and first_audio_at is None:
                    first_audio_at = time.perf_counter() - started

            if first_audio_at is not None:
                print(f"\n  [最初の音声まで {first_audio_at:.1f}秒]")
            else:
                print()

            playback.wait()
            print()

    except KeyboardInterrupt:
        print("\n終了します。")
        return 0
    finally:
        playback.stop()
        playback.close()


def voicevox_speak_safely(voicevox: VoicevoxClient, playback: PlaybackQueue, text: str) -> bool:
    """合成に失敗しても対話を止めない。成功したら True。"""
    try:
        playback.put(voicevox.synthesize(text))
        return True
    except (VoicevoxError, ValueError) as exc:
        print(f"\n[warn] 音声合成に失敗しました: {exc}", file=sys.stderr)
        return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="音声対話エージェント")
    parser.add_argument("-d", "--device", type=int, default=None, help="入力デバイス番号")
    parser.add_argument("--threshold", type=float, default=None, help="無音判定のRMSしきい値")
    args = parser.parse_args(argv)
    return run(device=args.device, threshold=args.threshold)


if __name__ == "__main__":
    raise SystemExit(main())
