"""Referensi R pada fixture metrik, bukan skor esai penelitian."""
import os,shutil,subprocess,sys
from pathlib import Path
import pytest

def test_ac2_matches_irrcac():
    rscript=os.environ.get('AES_RSCRIPT') or shutil.which('Rscript')
    if not rscript:pytest.skip('R/irrCAC belum tersedia; jalankan setup dan verify_ac2 --require-r')
    root=Path(__file__).resolve().parents[1]
    subprocess.run([sys.executable,str(root/'scripts/verify_ac2.py'),'--require-r','--rscript',rscript],check=True,capture_output=True,text=True)
