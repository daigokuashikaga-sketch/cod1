"""マイク入力を録音して WAV ファイルに保存する。

使い方:
    # デバイス一覧を確認
    python -m voice_agent.audio.record --list-devices

    # 無音を検出したら自動で止める（既定）
    python -m voice_agent.audio.record -o recordings/input.wav

    # 秒数を固定して録音
    python -m voice_agent.audio.record -o recordings/input.wav --seconds 5

faster-whisper がそのまま受け取れるよう 16kHz / モノラル / 16bit PCM で保存する。
sounddevice は PortAudio に依存するため、import は関数の中で行う
（音声デバイスのない環境でも下の純粋関数と単体テストは動く）。
"""

from __future__ import annotations

import argparse
import queue
import sys
import wave
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16_000
CHANNELS = 1
DTYPE = "int16"
BLOCK_SECONDS = 0.03  # 1ブロック30ms

# 無音判定のしきい値（int16 の RMS）。環境ノイズが多いなら上げる。
DEFAULT_SILENCE_THRESHOLD = 300.0
# この秒数だけ連続で無音なら録音を終了する
DEFAULT_SILENCE_DURATION = 1.0
# 発話が始まる前にこの秒数を過ぎたら諦める
DEFAULT_START_TIMEOUT = 10.0


def rms(block: np.ndarray) -> float:
    """ブロックの音量（RMS）を返す。空ブロックは 0。"""
    if block.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(block.astype(np.float64) ** 2)))


class SilenceDetector:
    """「発話が始まり、そのあと一定時間無音になったら終わり」を判定する。

    sounddevice に依存しないので単体テストできる。
    """

    def __init__(
        self,
        threshold: float = DEFAULT_SILENCE_THRESHOLD,
        silence_duration: float = DEFAULT_SILENCE_DURATION,
        start_timeout: float = DEFAULT_START_TIMEOUT,
        block_seconds: float = BLOCK_SECONDS,
    ) -> None:
        self.threshold = threshold
        self.silence_duration = silence_duration
        self.start_timeout = start_timeout
        self.block_seconds = block_seconds
        self.started = False
        self._silent_seconds = 0.0
        self._elapsed_seconds = 0.0

    def feed(self, block: np.ndarray) -> bool:
        """1ブロック分を食わせる。録音を終了すべきなら True を返す。"""
        self._elapsed_seconds += self.block_seconds
        loud = rms(block) >= self.threshold

        if not self.started:
            if loud:
                self.started = True
            elif self._elapsed_seconds >= self.start_timeout:
                return True  # 発話が始まらないままタイムアウト
            return False

        if loud:
            self._silent_seconds = 0.0
        else:
            self._silent_seconds += self.block_seconds
        return self._silent_seconds >= self.silence_duration


def save_wav(path: str | Path, audio: np.ndarray, samplerate: int = SAMPLE_RATE) -> Path:
    """int16 のモノラル配列を WAV として書き出す。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)  # int16
        wf.setframerate(samplerate)
        wf.writeframes(audio.astype(np.int16).tobytes())
    return path


def _import_sounddevice():
    """sounddevice を読み込む。PortAudio が無い環境では手順を示して終了する。"""
    try:
        import sounddevice as sd
    except OSError as exc:  # PortAudio が見つからない
        raise SystemExit(
            f"音声デバイスを初期化できません: {exc}\n"
            "Linux では `sudo apt install libportaudio2` が必要です。"
        ) from exc
    return sd


def list_devices() -> None:
    sd = _import_sounddevice()

    print(sd.query_devices())


def _open_stream(device: int | None):
    sd = _import_sounddevice()

    blocksize = int(SAMPLE_RATE * BLOCK_SECONDS)
    q: queue.Queue[np.ndarray] = queue.Queue()

    def callback(indata, frames, time_info, status):  # noqa: ARG001
        if status:
            print(f"[warn] {status}", file=sys.stderr)
        q.put(indata.copy().reshape(-1))

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype=DTYPE,
        blocksize=blocksize,
        device=device,
        callback=callback,
    )
    return stream, q


def record_fixed(seconds: float, device: int | None = None) -> np.ndarray:
    """指定秒数だけ録音する。"""
    stream, q = _open_stream(device)
    blocks: list[np.ndarray] = []
    total_blocks = int(seconds / BLOCK_SECONDS)
    with stream:
        print(f"録音中... ({seconds:.1f}秒)")
        for _ in range(total_blocks):
            blocks.append(q.get())
    return np.concatenate(blocks) if blocks else np.zeros(0, dtype=np.int16)


def record_until_silence(
    device: int | None = None,
    threshold: float = DEFAULT_SILENCE_THRESHOLD,
    silence_duration: float = DEFAULT_SILENCE_DURATION,
    start_timeout: float = DEFAULT_START_TIMEOUT,
    max_seconds: float = 30.0,
) -> np.ndarray:
    """発話を検出して録音し、無音が続いたら自動で止める。"""
    stream, q = _open_stream(device)
    detector = SilenceDetector(threshold, silence_duration, start_timeout)
    blocks: list[np.ndarray] = []
    max_blocks = int(max_seconds / BLOCK_SECONDS)

    with stream:
        print("話しかけてください... (Ctrl+C で中断)")
        speaking_notified = False
        for _ in range(max_blocks):
            block = q.get()
            blocks.append(block)
            done = detector.feed(block)
            if detector.started and not speaking_notified:
                print("発話を検出、録音中...")
                speaking_notified = True
            if done:
                break

    if not detector.started:
        print("発話が検出されませんでした。", file=sys.stderr)
        return np.zeros(0, dtype=np.int16)
    return np.concatenate(blocks)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="マイク入力を録音して WAV に保存する")
    parser.add_argument("-o", "--output", default="recordings/input.wav", help="出力先の WAV パス")
    parser.add_argument("-d", "--device", type=int, default=None, help="入力デバイス番号")
    parser.add_argument("--seconds", type=float, default=None, help="固定秒数で録音する")
    parser.add_argument("--threshold", type=float, default=DEFAULT_SILENCE_THRESHOLD, help="無音判定のRMSしきい値")
    parser.add_argument(
        "--silence-duration", type=float, default=DEFAULT_SILENCE_DURATION, help="この秒数の無音で録音を終える"
    )
    parser.add_argument("--max-seconds", type=float, default=30.0, help="自動停止モードでの最大録音秒数")
    parser.add_argument("--list-devices", action="store_true", help="入出力デバイス一覧を表示して終了")
    args = parser.parse_args(argv)

    if args.list_devices:
        list_devices()
        return 0

    try:
        if args.seconds is not None:
            audio = record_fixed(args.seconds, device=args.device)
        else:
            audio = record_until_silence(
                device=args.device,
                threshold=args.threshold,
                silence_duration=args.silence_duration,
                max_seconds=args.max_seconds,
            )
    except KeyboardInterrupt:
        print("\n中断しました。", file=sys.stderr)
        return 130

    if audio.size == 0:
        return 1

    path = save_wav(args.output, audio)
    print(f"保存しました: {path} ({audio.size / SAMPLE_RATE:.2f}秒)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
