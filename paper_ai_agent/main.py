"""
命令行启动入口。

用法示例：
  python main.py "请阅读 sample.pdf 并总结核心观点"
  python main.py --interactive
  python main.py
"""

from __future__ import annotations

import argparse
import sys

from agent_core import PaperAgent, LOG_SEPARATOR


def print_usage() -> None:
    """打印中文用法说明。"""
    print("用法：")
    print("  python main.py \"你的问题\"     单次提问")
    print("  python main.py --interactive   交互对话")
    print("  python main.py                 数字菜单选择模式")


def print_mode_menu() -> None:
    """打印数字选项菜单。"""
    print(f"\n{LOG_SEPARATOR}")
    print("请选择运行模式：")
    print("  1 — 单次提问")
    print("  2 — 交互对话")
    print("  0 — 退出")
    print(LOG_SEPARATOR)


def read_mode_choice() -> str:
    """读取并校验菜单选项（仅接受数字）。"""
    while True:
        choice = input("\n请输入选项（0 / 1 / 2）：").strip()
        if choice in {"0", "1", "2"}:
            return choice
        print("无效选项，请输入 0、1 或 2。")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="课程论文 AI 助手 — 智谱 GLM + 本地 PDF + SerpAPI",
        add_help=False,
    )
    parser.add_argument(
        "query",
        nargs="?",
        default="",
        help="单次提问内容",
    )
    parser.add_argument(
        "-i",
        "--interactive",
        action="store_true",
        help="进入交互对话模式",
    )
    parser.add_argument(
        "-h",
        "--help",
        action="store_true",
        help="显示用法说明",
    )
    return parser


def run_interactive(agent: PaperAgent) -> None:
    """交互模式：持续读取用户输入并调用智能体。"""
    print(f"\n{LOG_SEPARATOR}")
    print("进入交互模式。输入 0 退出。")
    print(LOG_SEPARATOR)

    while True:
        try:
            user_input = input("\n你：").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见。")
            break

        if not user_input:
            continue
        if user_input == "0":
            print("再见。")
            break

        agent.run(user_input)


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.help:
        print_usage()
        return 0

    if not args.query and not args.interactive:
        print_mode_menu()
        choice = read_mode_choice()
        if choice == "0":
            print("再见。")
            return 0
        if choice == "2":
            args.interactive = True
        else:
            try:
                args.query = input("\n请输入你的问题：").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n再见。")
                return 0
            if not args.query:
                print("未输入问题，已退出。")
                return 1

    try:
        agent = PaperAgent()
    except (EnvironmentError, FileNotFoundError, ValueError) as exc:
        print(f"启动失败：{exc}", file=sys.stderr)
        return 1

    if args.interactive:
        run_interactive(agent)
        return 0

    agent.run(args.query)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
