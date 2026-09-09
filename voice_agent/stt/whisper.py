"""faster-whisper で WAV ファイルを文字起こしする。

    python -m voice_agent.stt.whisper recordings/input.wav

モデルのロードは数秒〜数十秒かかるため、対話ループでは Transcriber を
1回だけ作って使い回す。large-v3 は初回に約3GBのダウンロードが走る。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from voice_agent.config import WHISPER, WhisperConfig


class Transcriber:
    """faster-whisper のラッパ。モデルは初回利用時にロードする。"""

    def __init__(self, config: WhisperConfig = WHISPER) -> None:
        self.config = config
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from faster_whisper import WhisperModel

            print(
                f"Whisper モデルを読み込み中: {self.config.model} "
                f"({self.config.device}/{self.config.compute_type})",
                file=sys.stderr,
            )
            started = time.perf_counter()
            self._model = WhisperModel(
                self.config.model,
                device=self.config.device,
                compute_type=self.config.compute_type,
            )
            print(f"読み込み完了 ({time.perf_counter() - started:.1f}秒)", file=sys.stderr)
        return self._model

    def transcribe(self, wav_path: str | Path) -> str:
        """WAV を文字起こしして、認識テキストを1つの文字列で返す。"""
        wav_path = Path(wav_path)
        if not wav_path.exists():
            raise FileNotFoundError(wav_path)

        segments, _info = self.model.transcribe(
            str(wav_path),
            language=self.config.language,
            beam_size=5,
            vad_filter=True,  # 無音・雑音区間を捨てて幻聴（ハルシネーション）を減らす
        )
        # segments はジェネレータ。ここで初めて実際の推論が走る。
        return "".join(segment.text for segment in segments).strip()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="WAV を faster-whisper で文字起こしする")
    parser.add_argument("wav", help="入力 WAV ファイル")
    parser.add_argument("--model", default=None, help="モデル名で上書き（例: medium, small）")
    args = parser.parse_args(argv)

    config = WHISPER if args.model is None else WhisperConfig(model=args.model)
    transcriber = Transcriber(config)

    started = time.perf_counter()
    text = transcriber.transcribe(args.wav)
    elapsed = time.perf_counter() - started

    if not text:
        print("（認識結果は空でした）", file=sys.stderr)
        return 1
    print(text)
    print(f"[{elapsed:.2f}秒]", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
