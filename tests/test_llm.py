"""応答生成のテスト（ダミー client を注入し、API は呼ばない）。"""

import unittest

from voice_agent.config import AnthropicConfig
from voice_agent.llm.anthropic_client import MAX_HISTORY_MESSAGES, ResponseGenerator


class FakeStream:
    def __init__(self, texts):
        self.text_stream = iter(texts)

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


class FakeMessages:
    def __init__(self, texts):
        self.texts = texts
        self.calls = []

    def stream(self, **kwargs):
        self.calls.append(kwargs)
        return FakeStream(self.texts)


class FakeClient:
    def __init__(self, texts=("こんにちは", "。元気ですか？")):
        self.messages = FakeMessages(list(texts))


CONFIG = AnthropicConfig(model="claude-sonnet-4-6", max_tokens=256, system_prompt="短く答えて")


class TestResponseGenerator(unittest.TestCase):
    def test_stream_reply_yields_deltas(self):
        generator = ResponseGenerator(CONFIG, client=FakeClient())
        self.assertEqual(list(generator.stream_reply("やあ")), ["こんにちは", "。元気ですか？"])

    def test_reply_joins_deltas(self):
        generator = ResponseGenerator(CONFIG, client=FakeClient())
        self.assertEqual(generator.reply("やあ"), "こんにちは。元気ですか？")

    def test_history_records_both_turns(self):
        generator = ResponseGenerator(CONFIG, client=FakeClient())
        generator.reply("やあ")
        self.assertEqual(
            generator.history,
            [
                {"role": "user", "content": "やあ"},
                {"role": "assistant", "content": "こんにちは。元気ですか？"},
            ],
        )

    def test_history_is_sent_on_the_next_turn(self):
        client = FakeClient()
        generator = ResponseGenerator(CONFIG, client=client)
        generator.reply("1回目")
        generator.reply("2回目")

        second_call = client.messages.calls[1]
        self.assertEqual(len(second_call["messages"]), 3)
        self.assertEqual(second_call["messages"][-1], {"role": "user", "content": "2回目"})

    def test_config_is_passed_to_the_api(self):
        client = FakeClient()
        ResponseGenerator(CONFIG, client=client).reply("やあ")

        call = client.messages.calls[0]
        self.assertEqual(call["model"], "claude-sonnet-4-6")
        self.assertEqual(call["max_tokens"], 256)
        self.assertEqual(call["system"], "短く答えて")

    def test_history_is_trimmed_and_starts_with_user(self):
        generator = ResponseGenerator(CONFIG, client=FakeClient())
        for i in range(MAX_HISTORY_MESSAGES):
            generator.reply(f"発話{i}")

        self.assertLessEqual(len(generator.history), MAX_HISTORY_MESSAGES + 1)
        self.assertEqual(generator.history[0]["role"], "user")

    def test_reset_clears_history(self):
        generator = ResponseGenerator(CONFIG, client=FakeClient())
        generator.reply("やあ")
        generator.reset()
        self.assertEqual(generator.history, [])


if __name__ == "__main__":
    unittest.main()
