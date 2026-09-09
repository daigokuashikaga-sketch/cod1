"""VOICEVOX のローカル API で音声合成する。

    python -m voice_agent.tts.voicevox "こんにちは、テストです。"
    python -m voice_agent.tts.voicevox "保存だけする" -o out.wav
    python -m voice_agent.tts.voicevox --list-speakers

VOICEVOX ENGINE を先に起動しておくこと:

    docker run --rm -p 50021:50021 voicevox/voicevox_engine:cpu-ubuntu20.04-latest
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from voice_agent.config import VOICEVOX, VoicevoxConfig

TIMEOUT_SECONDS = 30


class VoicevoxError(RuntimeError):
    pass


class VoicevoxClient:
    """VOICEVOX ENGINE の HTTP クライアント。

    session を差し替えられるようにしてあるので、テストではダミーを渡せる。
    """

    def __init__(self, config: VoicevoxConfig = VOICEVOX, session=None) -> None:
        self.config = config
        if session is None:
            import requests

            session = requests.Session()
        self.session = session

    def is_available(self) -> bool:
        try:
            response = self.session.get(f"{self.config.base_url}/version", timeout=3)
            return response.status_code == 200
        except Exception:
            return False

    def speakers(self) -> list[dict]:
        response = self.session.get(f"{self.config.base_url}/speakers", timeout=TIMEOUT_SECONDS)
        response.raise_for_status()
        return response.json()

    def audio_query(self, text: str) -> dict:
        response = self.session.post(
            f"{self.config.base_url}/audio_query",
            params={"text": text, "speaker": self.config.speaker},
            timeout=TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        query = response.json()
        query["speedScale"] = self.config.speed_scale
        return query

    def synthesize(self, text: str) -> bytes:
        """テキストを WAV バイト列に変換する。"""
        if not text.strip():
            raise ValueError("空のテキストは合成できません")
        try:
            query = self.audio_query(text)
            response = self.session.post(
                f"{self.config.base_url}/synthesis",
                params={"speaker": self.config.speaker},
                json=query,
                timeout=TIMEOUT_SECONDS,
            )
            response.raise_for_status()
        except Exception as exc:
            raise VoicevoxError(
                f"VOICEVOX ({self.config.base_url}) への接続に失敗しました: {exc}\n"
                "ENGINE が起動しているか確認してください。"
            ) from exc
        return response.content

    def speak(self, text: str) -> None:
        """合成して即座に再生する（再生し終わるまで戻らない）。"""
        from voice_agent.audio.playback import play_wav_bytes

        play_wav_bytes(self.synthesize(text))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="VOICEVOX で音声合成する")
    parser.add_argument("text", nargs="?", help="読み上げるテキスト")
    parser.add_argument("-o", "--output", default=None, help="再生せず WAV に保存する")
    parser.add_argument("-s", "--speaker", type=int, default=None, help="話者IDで上書き")
    parser.add_argument("--list-speakers", action="store_true", help="話者一覧を表示して終了")
    args = parser.parse_args(argv)

    config = VOICEVOX if args.speaker is None else VoicevoxConfig(speaker=args.speaker)
    client = VoicevoxClient(config)

    if args.list_speakers:
        if not client.is_available():
            print(f"VOICEVOX に接続できません: {config.base_url}", file=sys.stderr)
            return 1
        for speaker in client.speakers():
            for style in speaker["styles"]:
                print(f"{style['id']:>4}  {speaker['name']} ({style['name']})")
        return 0

    if not args.text:
        parser.error("text か --list-speakers のどちらかを指定してください")

    try:
        data = client.synthesize(args.text)
    except VoicevoxError as exc:
        print(exc, file=sys.stderr)
        return 1

    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        print(f"保存しました: {path} ({len(data) / 1024:.1f} KB)")
        return 0

    from voice_agent.audio.playback import play_wav_bytes

    play_wav_bytes(data)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
