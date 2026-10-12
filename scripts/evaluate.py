"""Pintasan CLI memakai modul yang sama dengan notebook."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from aes.script_entry import run_command
if __name__=='__main__':raise SystemExit(run_command('evaluate'))
