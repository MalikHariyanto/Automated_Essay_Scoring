"""Eksekusi notebook tanpa repo; data dan encoder fixture, bukan hasil penelitian."""
import ast
import inspect
import json
import os
from pathlib import Path
import sys
import time

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager
import pytest

from test_teacher_score import TechnicalTokenizer,make_book


ROOT=Path(__file__).resolve().parents[1]


def test_notebook_has_no_project_imports_or_git_commands():
    notebook=nbformat.read(ROOT/'notebooks/03_AES_dataset_guru_300.ipynb',as_version=4)
    assert notebook.metadata.aes_standalone.repository_required is False
    for cell in notebook.cells:
        if cell.cell_type!='code':continue
        source=cell.source.removeprefix('%%aes\n')
        for node in ast.walk(ast.parse(source)):
            if isinstance(node,ast.ImportFrom):
                assert not node.module or not (node.module=='aes' or node.module.startswith('aes.'))
            if isinstance(node,ast.Import):
                assert not any(a.name=='aes' or a.name.startswith('aes.') for a in node.names)
            if isinstance(node,ast.Constant) and isinstance(node.value,str):
                assert node.value!='git' and '__file__' not in node.value
        assert 'docs/aes_dataset300_references.json' not in source
    requirements=next(c.source for c in notebook.cells if c.cell_type=='code' and 'REQUIREMENTS_TEXT=' in c.source)
    assert 'pytest==' not in requirements and 'nbclient==' not in requirements and 'nbformat==' not in requirements


def test_standalone_notebook_runs_to_five_excel_reports_without_repository(tmp_path):
    pytest.importorskip('ipykernel',reason='Eksekusi notebook memerlukan ipykernel pada lingkungan pengujian.')
    notebook=nbformat.read(ROOT/'notebooks/03_AES_dataset_guru_300.ipynb',as_version=4)
    isolated=tmp_path/'only_notebook_and_data';isolated.mkdir()
    workspace=isolated/'private_results'
    fixture=isolated/'TECHNICAL_FIXTURE_ONLY.xlsx';make_book(fixture)
    guard='''
import builtins
_original_import=builtins.__import__
def _no_project_import(name,*args,**kwargs):
    if name=='aes' or name.startswith('aes.'):
        raise ImportError('Repository imports are blocked in this standalone test')
    return _original_import(name,*args,**kwargs)
builtins.__import__=_no_project_import
'''
    tiny_encoder='''
# PRIVATE TECHNICAL FIXTURE: random tiny transformers, not pretrained models.
from transformers import RobertaConfig,RobertaModel,XLMRobertaConfig,XLMRobertaModel
def encoder(cfg,role):
    cls,model,dim,heads=(RobertaConfig,RobertaModel,768,12) if role=='roberta' else (XLMRobertaConfig,XLMRobertaModel,1024,16)
    with torch.random.fork_rng():
        torch.manual_seed(170)
        result=model(cls(vocab_size=64,hidden_size=dim,num_hidden_layers=1,num_attention_heads=heads,
                         intermediate_size=16,max_position_embeddings=64,pad_token_id=1,bos_token_id=0,
                         eos_token_id=2,hidden_dropout_prob=0.,attention_probs_dropout_prob=0.),add_pooling_layer=False)
    result.requires_grad_(role=='roberta')
    return result
def tokenizer(*args,**kwargs):return TechnicalTokenizer()
'''
    for cell in notebook.cells:
        if cell.cell_type!='code':continue
        if 'EPOCHS,FOLDS=5,5' in cell.source:
            cell.source=cell.source.replace('EPOCHS,FOLDS=5,5','EPOCHS,FOLDS=1,5').replace('expected_rows=300','expected_rows=24')
            cell.source=cell.source.replace("lesson_version='inline_excel_v1'","lesson_version='TECHNICAL_FIXTURE_ONLY'")
        if "journal.save('01_konfigurasi'" in cell.source:
            cell.source=cell.source.replace("journal.save('01_konfigurasi'","journal.metadata['evidence_scope']='TECHNICAL RANDOM TRANSFORMER FIXTURE ONLY; NOT STUDENT RESULTS'\njournal.save('01_konfigurasi'")
        if 'def encoder(cfg, role):' in cell.source:
            cell.source+='\n'+inspect.getsource(TechnicalTokenizer)+tiny_encoder
        if cell.source.startswith('# Penghubung proses Python:'):
            node=next(n for n in ast.parse(cell.source).body if isinstance(n,ast.Assign))
            cell.source='WORKER_SOURCE='+repr(guard+ast.literal_eval(node.value))+'\n'
    guard_cell=nbformat.v4.new_code_cell(guard);guard_cell.pop('id',None)
    notebook.cells.insert(0,guard_cell)
    notebook_path=isolated/'03_AES_dataset_guru_300.ipynb'
    nbformat.write(notebook,notebook_path)
    kernels=tmp_path/'kernels';spec=kernels/'aes_isolated';spec.mkdir(parents=True)
    (spec/'kernel.json').write_text(json.dumps({'argv':[sys.executable,'-I','-m','ipykernel_launcher','-f','{connection_file}'],
                                              'display_name':'AES technical isolated','language':'python'}))
    # Git tidak tersedia. Kalau notebook mencoba clone/pull, proses langsung gagal.
    bin_folder=tmp_path/'bin';bin_folder.mkdir()
    (bin_folder/'git').write_text('#!/bin/sh\nexit 97\n');(bin_folder/'git').chmod(0o755)
    environment={**os.environ,'AES300_WORKSPACE':str(workspace),'AES300_DATASET':str(fixture),
                 'AES300_PYTHON':sys.executable,'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1',
                 'PATH':str(bin_folder)+os.pathsep+os.environ['PATH'],'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2',
                 'IPYTHONDIR':str(tmp_path/'ipython')}
    environment.pop('PYTHONPATH',None);environment.pop('AES300_AUDIT_ONLY',None)
    manager=KernelManager(kernel_name='aes_isolated',kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(kernels)]))
    start=time.time()
    try:
        NotebookClient(notebook,timeout=360,km=manager,resources={'metadata':{'path':str(isolated)}}).execute(env=environment)
    finally:
        if manager.has_kernel:manager.shutdown_kernel(now=True)
    nbformat.write(notebook,isolated/'TECHNICAL_executed.ipynb')
    pointer=json.loads((workspace/'outputs/dataset300_download.json').read_text())
    assert pointer['excel_count']==5
    report=json.loads((Path(pointer['run_folder'])/'final_report.json').read_text())
    assert [r['Skenario'] for r in report['rows']]==['S1','S2','S3']
    assert len(report['cv'])==15
    assert report['test_evaluations_per_scenario']==1
    assert not list(workspace.rglob('*.py'))
    assert not (workspace/'src').exists() and not (workspace/'docs').exists() and not (workspace/'.git').exists()
    proof={'scope':'TECHNICAL FIXTURE ONLY; NOT STUDENT SCORES','code_cells_executed':sum(c.cell_type=='code' and c.execution_count is not None for c in notebook.cells),
           'production_code_cells':sum(c.cell_type=='code' for c in notebook.cells)-1,'fixture_guard_cells':1,
           'elapsed_seconds':round(time.time()-start,2),'notebook_and_dataset_only':True,'project_imports_blocked':True,
           'git_unavailable':True,'private_python_files_created':False,'output':pointer}
    (isolated/'standalone_proof.json').write_text(json.dumps(proof,indent=2))
    print(json.dumps(proof))
