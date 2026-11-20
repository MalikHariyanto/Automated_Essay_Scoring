"""Technical checks for the reset E5 notebook and independently readable Excel parts."""

import ast
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
import pytest
from openpyxl import Workbook, load_workbook
from openpyxl.cell import WriteOnlyCell
from openpyxl.styles import Font, PatternFill

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / 'notebooks/multitrait_reset'
E2 = NOTEBOOKS / 'AES_Reset_E2_S3_MultiTrait_TraitHeads.ipynb'
E5 = NOTEBOOKS / 'AES_Reset_E5_S3_MultiTrait_WeightedLoss.ipynb'


def cells(path):
    return {
        cell['metadata']['aes_multitrait_role']: ''.join(cell['source'])
        for cell in json.loads(path.read_text())['cells']
        if cell['cell_type'] == 'code'
    }


def test_model_split_evaluation_and_training_settings_follow_e2():
    control, current = cells(E2), cells(E5)
    for role in (
        'definition:SingleScoreModel', 'definition:Engine', 'definition:optimizer_groups',
        'definition:make_split', 'definition:trait_metrics', 'metrics', 'test_definition',
    ):
        assert current[role] == control[role].replace('E2 reset', 'E5 reset'), role
    control_parameters = ast.parse(control['hyperparameters'])
    current_parameters = ast.parse(current['hyperparameters'])
    assert ast.dump(current_parameters.body[-1]) == ast.dump(
        ast.parse('LOSS_WEIGHT_MAX_RATIO = 2.0').body[0]
    )
    current_parameters.body.pop()
    assert ast.dump(current_parameters) == ast.dump(control_parameters)
    copy = ROOT / 'eksperimen/03_tiga_input_multi_trait/notebooks' / E5.name
    assert E5.read_bytes() == copy.read_bytes()


def test_class_weights_have_unit_mean_per_trait_and_cap_the_ratio():
    source = cells(E5)
    namespace = {'np': np, 'math': math}
    exec(source['columns'], namespace)
    exec(source['definition:fit'], namespace)
    labels = np.array([[2, 2, 2, 2]] * 9 + [[1, 1, 2, 1]], dtype=np.float32)
    weights, audit = namespace['training_loss_weights'](labels, 2.)
    np.testing.assert_allclose(weights.mean(axis=0), np.ones(4), rtol=1e-6)
    np.testing.assert_allclose(weights[-1] / weights[0], [2., 2., 1., 2.])
    assert sum(row['n_training'] for row in audit if row['trait'] == 'organisasi') == len(labels)
    unseen = next(row for row in audit if row['trait'] == 'isi' and row['skor'] == 4)
    assert unseen == {'trait': 'isi', 'skor': 4, 'n_training': 0, 'bobot_loss': None}
    uniform, _ = namespace['training_loss_weights'](labels, 1.)
    np.testing.assert_array_equal(uniform, np.ones_like(labels))


def test_archive_splits_large_excel_and_keeps_every_row_and_literal_string(tmp_path):
    source = cells(E5)
    namespace = dict(
        Path=Path, Workbook=Workbook, WriteOnlyCell=WriteOnlyCell, Font=Font,
        PatternFill=PatternFill, np=np, pd=pd, re=re, math=math, json=json,
        zipfile=zipfile, digest=lambda path: hashlib.sha256(path.read_bytes()).hexdigest(),
    )
    for role in ('definition:excel_value', 'definition:write_excel'):
        exec(source[role], namespace)

    rng = random.Random(42)
    rows = [(i, '=' + ''.join(rng.choices(string.ascii_letters, k=1800))) for i in range(42)]
    frame = pd.DataFrame(rows, columns=['i', 'text'])
    small = pd.DataFrame({'count': [42]})
    target = tmp_path / 'hasil.zip'
    inventory = namespace['export_excel_zip'](
        target, {'01_Data': {'Rows': frame, 'Summary': small}}, {'scope': 'fixture'}, max_bytes=15_000
    )
    assert len(inventory) > 2
    restored = []
    summary_count = 0
    with zipfile.ZipFile(target) as archive:
        assert archive.testzip() is None
        assert json.loads(archive.read('hasil/manifest.json'))['workbooks'] == inventory
        for entry in inventory:
            member = f"hasil/{entry['file']}"
            content = archive.read(member)
            assert 0 < len(content) == archive.getinfo(member).file_size == entry['bytes'] < 15_000
            assert hashlib.sha256(content).hexdigest() == entry['sha256']
            book = load_workbook(io.BytesIO(content), read_only=True)
            try:
                for table in entry['tables']:
                    values = list(book[table['name']].values)[1:]
                    assert len(values) == table['rows']
                    if table['name'] == 'Rows':
                        assert table['row_start'] == len(restored)
                        assert book['Rows']['B2'].data_type == 's'
                        restored.extend(values)
                    else:
                        summary_count += 1
                        assert values == [(42,)]
            finally:
                book.close()
    assert restored == rows and summary_count == 1

    previous = target.read_bytes()
    with pytest.raises(ValueError, match='tidak dapat dipisah'):
        namespace['export_excel_zip'](target, {'01_Data': {'Rows': frame.iloc[:1]}}, {}, max_bytes=1024)
    assert target.read_bytes() == previous
