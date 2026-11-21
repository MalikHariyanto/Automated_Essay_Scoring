"""Verify the actual ZIP size cap and independently readable parts in E15."""

import hashlib
import io
import json
import math
import random
import re
import string
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import Workbook, load_workbook
from openpyxl.cell import WriteOnlyCell
from openpyxl.styles import Font, PatternFill

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / 'notebooks/multitrait_reset/AES_Reset_E15_S1_S2_S3_TraitGatedConcat.ipynb'


def test_zip_parts_keep_all_workbook_rows_under_size_cap(tmp_path):
    namespace = dict(Path=Path, Workbook=Workbook, WriteOnlyCell=WriteOnlyCell,
                     Font=Font, PatternFill=PatternFill, np=np, pd=pd, re=re, math=math,
                     json=json, zipfile=zipfile,
                     digest=lambda path: hashlib.sha256(path.read_bytes()).hexdigest())
    notebook = json.loads(NOTEBOOK.read_text())
    for cell in notebook['cells']:
        if cell['metadata'].get('aes_multitrait_role') in ('definition:excel_value', 'definition:write_excel'):
            exec(cell['source'], namespace)

    rng = random.Random(42)
    rows = [(i, ''.join(rng.choices(string.ascii_letters, k=2400))) for i in range(53)]
    groups = {'02_Vektor': {'Rows': pd.DataFrame(rows, columns=['i', 'text'])},
              '05_Riwayat': {'Summary': pd.DataFrame({'count': [53]})}}
    limit = 80_000
    archives = namespace['export_excel_zip'](tmp_path / 'hasil.zip', groups, {}, max_bytes=limit)
    assert len(archives) > 1
    restored = {}
    totals = []
    for path, byte_count, inventory in archives:
        assert path.stat().st_size == byte_count < limit
        with zipfile.ZipFile(path) as archive:
            assert archive.testzip() is None
            manifest = json.loads(archive.read(f'{path.stem}/manifest.json'))
            assert manifest['workbooks'] == inventory
            for entry in inventory:
                content = archive.read(f"{path.stem}/{entry['file']}")
                assert hashlib.sha256(content).hexdigest() == entry['sha256']
                assert len(content) == entry['bytes'] < limit
                book = load_workbook(io.BytesIO(content), read_only=True)
                try:
                    for table in entry['tables']:
                        values = list(book[table['name']].values)[1:]
                        assert len(values) == table['rows']
                        if table['name'] == 'Rows':
                            assert table['row_start'] not in restored
                            restored[table['row_start']] = values
                        else:
                            totals.extend(values)
                finally:
                    book.close()
    position = 0
    for start, values in sorted(restored.items()):
        assert start == position
        assert values == rows[start:start + len(values)]
        position += len(values)
    assert position == len(rows) and totals == [(53,)]
