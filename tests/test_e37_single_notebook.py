"""Current E37 is one five-fold S3 run; no paired or multi-seed loop."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import nbformat

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tests'))
from test_e36_notebook import execute_full_native_fixture

NAME='AES_300_E37_E30_S3_EncoderLR7e6_Native_Learnable_GPU.ipynb'
NOTEBOOK=ROOT/'notebooks/experiments'/NAME
BASE=ROOT/'notebooks/experiments/AES_300_E36_E30_S3_EncoderLR3e6_Native_Learnable_GPU.ipynb'


def test_single_five_fold_source_matches_e36_except_encoder_lr_and_presentation():
    raw=NOTEBOOK.read_bytes();n=nbformat.reads(raw.decode(),as_version=4);nbformat.validate(n)
    old=nbformat.read(BASE,as_version=4)
    before={c.metadata.aes36_role:c.source for c in old.cells if c.cell_type=='code'}
    after={c.metadata.aes37_role:c.source for c in n.cells if c.cell_type=='code'}
    assert len(after)==len(before)==89
    # All training/math/selection functions retain the five-fold native implementation.
    for role,source in before.items():
        if role.startswith('definition:'):
            # Only experiment names in errors/archive filenames may change here.
            expected=source.replace(BASE.stem,NOTEBOOK.stem).replace('E36','E37')
            assert ast.dump(ast.parse(expected))==ast.dump(ast.parse(after[role])),role
    assert 'ENCODER_LR = 7e-6' in after['hyperparameters']
    assert 'FOLDS = 5' in after['hyperparameters'] and 'EPOCHS = 25' in after['hyperparameters']
    assert 'EARLY_STOPPING_PATIENCE = 5' in after['hyperparameters'] and 'TRAIN_SEED = 42' in after['hyperparameters']
    text='\n'.join(after.values())
    for banned in ('TRAINING_SEEDS','run_paired_cv','paired_study','E30_control','E37_candidate','cosine','%%aes','WORKER_SOURCE'):
        assert banned not in text
    for c in n.cells:
        if c.cell_type=='code':
            assert not c.outputs and c.execution_count is None
            assert not any(line.lstrip().startswith(('%','!')) for line in c.source.splitlines())
            ast.parse(c.source)
    fingerprint=n.metadata.aes_standalone.code_sha256
    assert hashlib.sha256(text.replace(fingerprint,'__NOTEBOOK_CODE_SHA256__').encode()).hexdigest()==fingerprint
    assert n.metadata.aes_experiment.cv_fits==5 and n.metadata.aes_experiment.refits==1
    historical=BASE.read_bytes()
    subprocess.run([sys.executable,'scripts/build_e37_single_notebook.py'],cwd=ROOT,check=True,capture_output=True)
    assert raw==NOTEBOOK.read_bytes()==(ROOT/'aes-learning/notebooks/experiments'/NAME).read_bytes()
    assert historical==BASE.read_bytes()


def test_full_single_notebook_and_fresh_kernel_resume(tmp_path):
    execute_full_native_fixture(tmp_path,NOTEBOOK,'aes37_role',7e-6)
