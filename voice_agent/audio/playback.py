"""合成された WAV バイト列を再生する。

再生はワーカースレッドのキューで直列化する。こうすると「1文目を再生しながら
2文目を合成する」という重ね合わせができ、文と文の間が詰まる。
"""

from __future__ import annotations

import io
import queue
import threading
import wave

import numpy as np


def wav_bytes_to_array(data: bytes) -> tuple[np.ndarray, int]:
    """WAV バイト列を (int16 配列, サンプリングレート) に変換する。"""
    with wave.open(io.BytesIO(data), "rb") as wf:
        samplerate = wf.getframerate()
        channels = wf.getnchannels()
        frames = wf.readframes(wf.getnframes())
    audio = np.frombuffer(frames, dtype=np.int16)
    if channels > 1:
        audio = audio.reshape(-1, channels)
    return audio, samplerate


def play_wav_bytes(data: bytes) -> None:
    """WAV バイト列を再生し、再生し終わるまで待つ。"""
    from voice_agent.audio.record import _import_sounddevice

    sd = _import_sounddevice()
    audio, samplerate = wav_bytes_to_array(data)
    sd.play(audio, samplerate)
    sd.wait()


class PlaybackQueue:
    """投入された順に WAV を再生し続けるワーカースレッド。"""

    def __init__(self) -> None:
        self._queue: queue.Queue[bytes | None] = queue.Queue()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._idle = threading.Event()
        self._idle.set()
        self._thread.start()

    def put(self, data: bytes) -> None:
        self._idle.clear()
        self._queue.put(data)

    def wait(self) -> None:
        """キューが空になり、再生も終わるまで待つ。"""
        self._queue.join()
        self._idle.wait()

    def stop(self) -> None:
        """再生中の音を止め、キューを捨てる（割り込み用）。"""
        while True:
            try:
                self._queue.get_nowait()
                self._queue.task_done()
            except queue.Empty:
                break
        from voice_agent.audio.record import _import_sounddevice

        _import_sounddevice().stop()

    def close(self) -> None:
        self._queue.put(None)
        self._thread.join(timeout=5)

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                self._queue.task_done()
                return
            try:
                play_wav_bytes(item)
            except Exception as exc:  # 再生に失敗しても対話は続ける
                print(f"[warn] 再生に失敗しました: {exc}")
            finally:
                self._queue.task_done()
                if self._queue.empty():
                    self._idle.set()
