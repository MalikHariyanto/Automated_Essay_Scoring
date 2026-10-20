"""Ubah ZIP/folder hasil lama menjadi lima laporan tanpa model atau training."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from aes.teacher_score.presentation import build_reports, repack_archive


def main():
    parser=argparse.ArgumentParser()
    source=parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--archive')
    source.add_argument('--folder')
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    result=repack_archive(args.archive,args.output) if args.archive else build_reports(args.folder,args.output)
    print('Selesai. Tidak ada model yang dilatih ulang atau metrik yang dihitung ulang.')
    print('Unduh:',result['archive'])


if __name__=='__main__':main()
