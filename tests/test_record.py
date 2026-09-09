"""録音ロジックのうち、音声デバイスに依存しない部分のテスト。

    python -m unittest discover -s tests -v
"""

import unittest
import wave
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from voice_agent.audio.record import (
    BLOCK_SECONDS,
    SAMPLE_RATE,
    SilenceDetector,
    rms,
    save_wav,
)

BLOCK_LEN = int(SAMPLE_RATE * BLOCK_SECONDS)


def silent_block() -> np.ndarray:
    return np.zeros(BLOCK_LEN, dtype=np.int16)


def loud_block(amplitude: int = 5000) -> np.ndarray:
    return np.full(BLOCK_LEN, amplitude, dtype=np.int16)


class TestRms(unittest.TestCase):
    def test_empty_block_is_zero(self):
        self.assertEqual(rms(np.zeros(0, dtype=np.int16)), 0.0)

    def test_silence_is_zero(self):
        self.assertEqual(rms(silent_block()), 0.0)

    def test_constant_amplitude(self):
        self.assertAlmostEqual(rms(loud_block(1000)), 1000.0)


class TestSilenceDetector(unittest.TestCase):
    def test_silence_before_speech_does_not_stop(self):
        d = SilenceDetector(threshold=300, silence_duration=0.3, start_timeout=10.0)
        for _ in range(20):
            self.assertFalse(d.feed(silent_block()))
        self.assertFalse(d.started)

    def test_stops_after_silence_following_speech(self):
        d = SilenceDetector(threshold=300, silence_duration=0.3, start_timeout=10.0)
        for _ in range(10):
            self.assertFalse(d.feed(loud_block()))
        self.assertTrue(d.started)

        # 0.3秒 = 10ブロック分の無音で停止する
        stopped_at = None
        for i in range(1, 21):
            if d.feed(silent_block()):
                stopped_at = i
                break
        self.assertEqual(stopped_at, 10)

    def test_short_pause_does_not_stop(self):
        d = SilenceDetector(threshold=300, silence_duration=0.3, start_timeout=10.0)
        d.feed(loud_block())
        for _ in range(5):  # 0.15秒の間
            self.assertFalse(d.feed(silent_block()))
        self.assertFalse(d.feed(loud_block()))  # 再び話し始めたのでリセット
        for _ in range(9):
            self.assertFalse(d.feed(silent_block()))
        self.assertTrue(d.feed(silent_block()))

    def test_start_timeout(self):
        d = SilenceDetector(threshold=300, silence_duration=0.3, start_timeout=0.3)
        for _ in range(9):
            self.assertFalse(d.feed(silent_block()))
        self.assertTrue(d.feed(silent_block()))


class TestSaveWav(unittest.TestCase):
    def test_writes_16k_mono_16bit(self):
        audio = np.concatenate([loud_block(), silent_block()])
        with TemporaryDirectory() as tmp:
            path = save_wav(Path(tmp) / "nested" / "out.wav", audio)
            self.assertTrue(path.exists())
            with wave.open(str(path), "rb") as wf:
                self.assertEqual(wf.getnchannels(), 1)
                self.assertEqual(wf.getsampwidth(), 2)
                self.assertEqual(wf.getframerate(), SAMPLE_RATE)
                self.assertEqual(wf.getnframes(), audio.size)
                restored = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
        np.testing.assert_array_equal(restored, audio)


if __name__ == "__main__":
    unittest.main()
