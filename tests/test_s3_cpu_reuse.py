"""Production CPU configuration and reuse; only technical synthetic data."""
import ast
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import nbformat
import pytest
import torch

import test_s3_gpu_reuse as reuse
from test_s3_gpu_reuse import source_run

ROOT=Path(__file__).resolve().parents[1]
NOTEBOOK=ROOT/'notebooks/experiments/AES_300_E6_S3_LR5e6_CPU_Reuse.ipynb'
GPU=ROOT/'notebooks/experiments/AES_300_E6_S3_LR5e6_GPU_Reuse.ipynb'


def test_cpu_artifact_and_algorithm_equal_gpu():
    n=nbformat.read(NOTEBOOK,as_version=4);nbformat.validate(n)
    assert n.metadata.aes_experiment.base_sha256==hashlib.sha256(GPU.read_bytes()).hexdigest()
    assert n.metadata.aes_experiment.device=='cpu' and 'accelerator' not in n.metadata
    assert (ROOT/'aes-learning/notebooks/experiments'/NOTEBOOK.name).read_bytes()==NOTEBOOK.read_bytes()
    def functions(notebook):
        result={}
        for c in notebook.cells:
            if c.cell_type!='code':continue
            assert c.outputs==[] and c.execution_count is None
            for node in ast.parse(c.source.removeprefix('%%aes\n')).body:
                if isinstance(node,(ast.FunctionDef,ast.ClassDef)):result[node.name]=ast.dump(node,include_attributes=False)
        return result
    cpu=functions(n);gpu=functions(nbformat.read(GPU,as_version=4))
    # Environment folder differs, all experiment/reuse/training methods match.
    for name in ('SingleScoreModel','Engine','optimizer_groups','fit','train_all','inspect_control_source',
                 'prepare_reused_controls','load_model','make_split','metrics','score_answer','export_final_features'):
        assert cpu[name]==gpu[name],name
    before=NOTEBOOK.read_bytes()
    subprocess.run([sys.executable,str(ROOT/'scripts/build_s3_lr_cpu_reuse_notebook.py')],check=True,capture_output=True)
    assert NOTEBOOK.read_bytes()==before


@pytest.mark.parametrize('cuda_available',[False,True])
def test_cpu_config_even_when_environment_requests_gpu(source_run,monkeypatch,cuda_available):
    ns=reuse.runtime(NOTEBOOK)
    monkeypatch.setenv('AES300_DATASET',str(source_run[1]));monkeypatch.setenv('AES300_RESULTS',str(source_run[0]/'target'))
    monkeypatch.setenv('AES300_REQUIRE_GPU','1');monkeypatch.delenv('AES300_AUDIT_ONLY',raising=False)
    monkeypatch.setattr(torch.cuda,'is_available',lambda:cuda_available)
    code=next(c.source for c in nbformat.read(NOTEBOOK,as_version=4).cells if c.cell_type=='code' and 'ROBERTA_CHECKPOINT=' in c.source)
    exec(code.removeprefix('%%aes\n'),ns)
    assert ns['REQUIRE_GPU'] is False and ns['cfg']['device']=='cpu'
    assert ns['device_name'](ns['cfg'])=='cpu' and ns['cfg']['s3_encoder_lr']==5e-6


def test_reused_controls_and_only_s3_training(source_run,tmp_path,monkeypatch):
    monkeypatch.setattr(reuse,'NOTEBOOK',NOTEBOOK)
    reuse.test_only_s3_fit_runs_source_and_source_s3_remain_unchanged(source_run,tmp_path)


def test_gpu_metadata_checkpoint_loaded_on_cpu(source_run,tmp_path):
    # Simulate device metadata of a completed GPU source; weights remain a
    # random technical CPU fixture, not proof of actual GPU execution.
    root,path,old,bundle,manifest,bge,source,report=source_run
    gpu_cfg={**old,'device':'cuda'}
    semantic={k:v for k,v in gpu_cfg.items() if k not in ('dataset','output','trace_position','vector_source','new_answer','focus_word')}
    key=hashlib.sha256(json.dumps({'source':bundle.source_sha256,'config':semantic,
        'code':'49f80d493aa531197b6a01e57fcb80e51ff18fb4f2ecda09abfc1f37814f1da7'},sort_keys=True).encode()).hexdigest()[:16]
    copied=tmp_path/'gpu_metadata_source'/key;shutil.copytree(source,copied)
    (copied/'effective_config.json').write_text(json.dumps(gpu_cfg))
    ns=reuse.runtime(NOTEBOOK);report=json.loads((copied/'final_report.json').read_text())
    for scenario in ('S1','S2'):
        file=copied/scenario/'final.pt'
        payload=torch.load(file,weights_only=True,map_location='cpu');payload['config']=gpu_cfg;torch.save(payload,file)
        checksum=ns['digest'](file);report['checkpoints'][scenario]=checksum
        result=json.loads((copied/scenario/'test_result.json').read_text());result['checkpoint_sha256']=checksum
        (copied/scenario/'test_result.json').write_text(json.dumps(result))
    (copied/'final_report.json').write_text(json.dumps(report))
    cfg=ns['config']({**old,'s3_encoder_lr':5e-6,'device':'cpu','reuse_run':str(copied)})
    plan=ns['inspect_control_source'](copied,bundle,manifest,cfg)
    assert plan['source_device']=='cuda' and set(plan['selected_epochs'])=={'S1','S2'}


def test_full_cpu_reuse_notebook_without_device_override(source_run,tmp_path,monkeypatch):
    monkeypatch.setattr(reuse,'NOTEBOOK',NOTEBOOK)
    # Existing execution harness substitutes only dataset/model/epoch fixtures.
    # Its GPU->CPU replacements are no-ops: this artifact already selects CPU.
    reuse.test_notebook_executes_all_reports_without_control_training(source_run,tmp_path)
