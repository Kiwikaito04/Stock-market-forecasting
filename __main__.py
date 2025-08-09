import os
import argparse
import subprocess
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
venv_python = sys.executable

SCRIPTS = {
    "intra_3_lstm": "Intraday-240,3-LSTM.py",
    "intra_3_rf":   "Intraday-240,3-RF.py",
    "intra_1_lstm": "Intraday-240,1-LSTM.py",
    "intra_1_rf":   "Intraday-240,1-RF.py",
    "next_1_lstm":  "NextDay-240,1-LSTM.py",
    "next_1_rf":    "NextDay-240,1-RF.py",
}

GROUPS = {
    "paper": ["intra_3_lstm", "intra_3_rf", "next_1_lstm", "next_1_rf"],
}

def main():
    parser = argparse.ArgumentParser(
        description="🎯 Chạy các pipeline LSTM/RF",
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        "--scripts", "-s", nargs="+", choices=SCRIPTS.keys(),
        help="🧩 Chọn script cụ thể để chạy:\n  ví dụ: -s intra_3_lstm intra_3_rf"
    )

    parser.add_argument(
        "--all", "-a", action="store_true",
        help="🚀 Chạy tất cả các script"
    )

    parser.add_argument(
        "--group", "-g", choices=GROUPS.keys(),
        help="📦 Chạy nhóm script định nghĩa sẵn:\n  ví dụ: -g paper"
    )

    args = parser.parse_args()

    selected = set()

    if args.all:
        selected.update(SCRIPTS.keys())

    if args.group:
        selected.update(GROUPS[args.group])

    if args.scripts:
        selected.update(args.scripts)

    if not selected:
        print("⚠️  Bạn chưa chọn script nào để chạy.")
        print("👉 Gợi ý:")
        print("   • Chạy tất cả:         python __main__.py --all")
        print("   • Chạy nhóm 'paper':   python __main__.py --group paper")
        print("   • Chạy cụ thể:         python __main__.py --scripts intra_3_lstm intra_3_rf")
        sys.exit(1)

    for key in selected:
        script = SCRIPTS[key]
        print(f"\n🔧 Đang chạy: {script} ...\n")
        subprocess.run([venv_python, script], cwd=BASE_DIR)


if __name__ == "__main__":
    main()
