"""WAV バイト列のデコードのテスト（再生はしない）。"""

import io
import unittest
import wave

import numpy as np

from voice_agent.audio.playback import wav_bytes_to_array


def make_wav(audio: np.ndarray, samplerate: int = 24_000, channels: int = 1) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(samplerate)
        wf.writeframes(audio.astype(np.int16).tobytes())
    return buf.getvalue()


class TestWavBytesToArray(unittest.TestCase):
    def test_roundtrip_mono(self):
        original = np.array([0, 1000, -1000, 32767, -32768], dtype=np.int16)
        audio, samplerate = wav_bytes_to_array(make_wav(original))
        self.assertEqual(samplerate, 24_000)
        np.testing.assert_array_equal(audio, original)

    def test_stereo_is_reshaped(self):
        original = np.array([1, 2, 3, 4, 5, 6], dtype=np.int16)
        audio, _ = wav_bytes_to_array(make_wav(original, channels=2))
        self.assertEqual(audio.shape, (3, 2))


if __name__ == "__main__":
    unittest.main()
