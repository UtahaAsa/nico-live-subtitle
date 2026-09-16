from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import AppConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Windows 系统音频日英识别与中文翻译悬浮字幕"
    )
    parser.add_argument("--config", type=Path, help="JSON 配置文件路径")
    parser.add_argument(
        "--list-devices", action="store_true", help="列出音频回环设备后退出"
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        config = AppConfig.load(args.config)
    except (OSError, ValueError) as error:
        print(f"配置加载失败：{error}", file=sys.stderr)
        return 2

    if args.list_devices:
        return _print_devices()

    from PySide6 import QtWidgets

    from .ui import OverlayWindow

    application = QtWidgets.QApplication(sys.argv)
    application.setApplicationName("Nico Live Subtitle")
    application.setQuitOnLastWindowClosed(True)
    window = OverlayWindow(config)
    application.aboutToQuit.connect(window.stop_pipeline)
    window.show()
    return application.exec()


def _print_devices() -> int:
    from .audio import list_loopback_devices

    try:
        devices = list_loopback_devices()
    except Exception as error:
        print(f"读取音频设备失败：{error}", file=sys.stderr)
        return 1
    if not devices:
        print("没有找到 WASAPI 回环设备")
        return 1
    for device in devices:
        marker = "*" if device.is_default else " "
        print(f"{marker} {device.name}\n  {device.id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
