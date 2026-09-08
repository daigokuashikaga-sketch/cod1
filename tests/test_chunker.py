"""文分割ロジックのテスト。"""

import unittest

from voice_agent.tts.chunker import SentenceChunker


class TestSentenceChunker(unittest.TestCase):
    def test_emits_first_sentence_immediately(self):
        c = SentenceChunker()
        # 1文目は短くてもすぐ出す（最初の発話を早くするため）
        self.assertEqual(c.feed("はい。"), ["はい。"])

    def test_short_later_fragment_is_merged(self):
        c = SentenceChunker(min_length=8)
        c.feed("今日はいい天気ですね。")
        self.assertEqual(c.feed("はい。"), [])
        self.assertEqual(c.feed("散歩に行きましょう。"), ["はい。散歩に行きましょう。"])

    def test_splits_on_multiple_endings(self):
        c = SentenceChunker(min_length=1)
        sentences = c.feed("こんにちは。元気ですか？ 今日は暑いですね！")
        self.assertEqual(sentences, ["こんにちは。", "元気ですか？", "今日は暑いですね！"])

    def test_accumulates_across_deltas(self):
        c = SentenceChunker()
        self.assertEqual(c.feed("こんに"), [])
        self.assertEqual(c.feed("ちは"), [])
        self.assertEqual(c.feed("。"), ["こんにちは。"])

    def test_flush_returns_remainder(self):
        c = SentenceChunker()
        c.feed("こんにちは。")
        c.feed("句点のない末尾")
        self.assertEqual(c.flush(), "句点のない末尾")

    def test_flush_on_empty_buffer(self):
        c = SentenceChunker()
        self.assertIsNone(c.flush())
        c.feed("こんにちは。")
        self.assertIsNone(c.flush())

    def test_no_text_is_lost(self):
        c = SentenceChunker(min_length=8)
        deltas = ["こんにちは", "。今日は", "いい天気ですね！", "はい。", "そうですね。"]
        collected = []
        for delta in deltas:
            collected.extend(c.feed(delta))
        tail = c.flush()
        if tail:
            collected.append(tail)
        self.assertEqual("".join(collected), "".join(deltas))


if __name__ == "__main__":
    unittest.main()
