"""VOICEVOX クライアントのテスト（ダミー session を注入し、通信はしない）。"""

import unittest

from voice_agent.config import VoicevoxConfig
from voice_agent.tts.voicevox import VoicevoxClient, VoicevoxError


class FakeResponse:
    def __init__(self, json_data=None, content=b"", status_code=200):
        self._json = json_data
        self.content = content
        self.status_code = status_code

    def json(self):
        return self._json

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class FakeSession:
    def __init__(self, *, fail=False):
        self.fail = fail
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append(("GET", url, kwargs))
        if self.fail:
            raise ConnectionError("接続拒否")
        return FakeResponse(json_data=[], status_code=200)

    def post(self, url, **kwargs):
        self.calls.append(("POST", url, kwargs))
        if self.fail:
            raise ConnectionError("接続拒否")
        if url.endswith("/audio_query"):
            return FakeResponse(json_data={"speedScale": 1.0, "accent_phrases": []})
        return FakeResponse(content=b"RIFF....WAVE")


CONFIG = VoicevoxConfig(base_url="http://localhost:50021", speaker=3, speed_scale=1.2)


class TestVoicevoxClient(unittest.TestCase):
    def test_synthesize_calls_audio_query_then_synthesis(self):
        session = FakeSession()
        client = VoicevoxClient(CONFIG, session=session)

        data = client.synthesize("こんにちは")

        self.assertEqual(data, b"RIFF....WAVE")
        urls = [url for _method, url, _kwargs in session.calls]
        self.assertEqual(
            urls,
            ["http://localhost:50021/audio_query", "http://localhost:50021/synthesis"],
        )

    def test_speaker_and_text_are_passed(self):
        session = FakeSession()
        VoicevoxClient(CONFIG, session=session).synthesize("こんにちは")

        _method, _url, query_kwargs = session.calls[0]
        self.assertEqual(query_kwargs["params"], {"text": "こんにちは", "speaker": 3})

    def test_speed_scale_is_applied_to_query(self):
        session = FakeSession()
        VoicevoxClient(CONFIG, session=session).synthesize("こんにちは")

        _method, _url, synthesis_kwargs = session.calls[1]
        self.assertEqual(synthesis_kwargs["json"]["speedScale"], 1.2)
        self.assertEqual(synthesis_kwargs["params"], {"speaker": 3})

    def test_empty_text_raises_value_error(self):
        client = VoicevoxClient(CONFIG, session=FakeSession())
        with self.assertRaises(ValueError):
            client.synthesize("   ")

    def test_connection_failure_raises_voicevox_error(self):
        client = VoicevoxClient(CONFIG, session=FakeSession(fail=True))
        with self.assertRaises(VoicevoxError):
            client.synthesize("こんにちは")

    def test_is_available_false_when_unreachable(self):
        self.assertFalse(VoicevoxClient(CONFIG, session=FakeSession(fail=True)).is_available())
        self.assertTrue(VoicevoxClient(CONFIG, session=FakeSession()).is_available())


if __name__ == "__main__":
    unittest.main()
