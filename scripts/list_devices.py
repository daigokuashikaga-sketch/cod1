"""利用可能な音声入出力デバイスを一覧表示する。

    python scripts/list_devices.py
"""

import sounddevice as sd

if __name__ == "__main__":
    print(sd.query_devices())
    print()
    print("default (input, output):", sd.default.device)
