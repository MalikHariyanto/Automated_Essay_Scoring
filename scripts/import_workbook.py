"""Impor Excel privat ke CSV untuk loader AES yang sama di CLI dan notebook."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from aes.workbook import import_workbook, TEACHER_SHEET


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workbook', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--sheet', default=TEACHER_SHEET)
    args = parser.parse_args()
    report = import_workbook(args.workbook, args.output, args.sheet)
    print(json.dumps({k: v for k, v in report.items() if k != 'ai_summary'}, ensure_ascii=False, indent=2))
