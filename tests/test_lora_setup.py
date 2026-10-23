"""Reproduce broken torchvision and protect inherited runtime torch; synthetic fixtures."""
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import venv
import zipfile

import nbformat
import pytest
from packaging.requirements import Requirement

ROOT=Path(__file__).resolve().parents[1]
NOTEBOOK=ROOT/'notebooks/experiments/AES_300_S3_LoRA_MLP_GPU.ipynb'


def setup_namespace(root):
    import argparse
    import importlib.util
    spec=importlib.util.spec_from_file_location('lora_setup_subprocess',subprocess.__file__)
    separate_subprocess=importlib.util.module_from_spec(spec);spec.loader.exec_module(separate_subprocess)
    n=nbformat.read(NOTEBOOK,as_version=4)
    ns={'ROOT':root,'Path':Path,'sys':sys,'os':os,'json':json,'argparse':argparse,
        'hashlib':hashlib,'venv':venv,'Requirement':Requirement,'subprocess':separate_subprocess}
    requirements=next(c.source for c in n.cells if c.cell_type=='code' and 'REQUIREMENTS_TEXT=' in c.source)
    node=next(node for node in ast.parse(requirements).body if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='REQUIREMENTS_TEXT' for t in node.targets))
    (root/'requirements.dataset300.txt').write_text(ast.literal_eval(node.value))
    setup=next(c.source for c in n.cells if c.cell_type=='code' and 'def prepare_environment(' in c.source)
    exec(compile(setup,'<actual-notebook-setup>','exec'),ns)
    return ns


def broken_vision(root):
    fake=root/'optional_vision';fake.mkdir()
    package=fake/'torchvision';package.mkdir()
    # Actual torch dispatcher error, not a hardcoded replacement traceback.
    (package/'__init__.py').write_text('''import torch
@torch.library.register_fake("torchvision::nms")
def _technical_missing_nms(dets,scores,iou_threshold):
    return dets
''')
    meta=fake/'torchvision-0.20.1.dist-info';meta.mkdir()
    (meta/'METADATA').write_text('Metadata-Version: 2.1\nName: torchvision\nVersion: 0.20.1\n')
    return fake


def test_reproduce_missing_nms_then_actual_peft_forward_backward_without_vision(tmp_path):
    ns=setup_namespace(tmp_path);fake=broken_vision(tmp_path)
    env={**os.environ,'PYTHONPATH':str(fake),'HF_HOME':str(tmp_path/'hf'),'USE_TF':'0','USE_FLAX':'0'}
    original=subprocess.run([sys.executable,'-c','from peft import LoraConfig,get_peft_model'],env=env,capture_output=True,text=True)
    assert original.returncode!=0 and 'operator torchvision::nms does not exist' in original.stderr
    success=subprocess.run([sys.executable,'-c',ns['IMPORTS']+'''
import sys
assert 'torchvision' not in sys.modules
from transformers import RobertaConfig,RobertaModel
model=get_peft_model(RobertaModel(RobertaConfig(vocab_size=32,hidden_size=32,num_hidden_layers=1,
    num_attention_heads=4,intermediate_size=48),add_pooling_layer=False),
    LoraConfig(r=4,lora_alpha=8,target_modules=['query','value'],bias='none'))
value=model(input_ids=torch.tensor([[0,4,5,2]])).last_hidden_state
value[:,:,0].square().mean().backward()
assert any(p.grad is not None and p.grad.abs().sum()>0 for n,p in model.named_parameters() if 'lora_B' in n)
assert 'torchvision' not in sys.modules
print('TECHNICAL_LORA_FORWARD_BACKWARD_OK')
'''],env=env,capture_output=True,text=True)
    assert success.returncode==0,success.stderr
    assert 'TECHNICAL_LORA_FORWARD_BACKWARD_OK' in success.stdout


def test_resolves_missing_dependencies_in_rounds_without_installing_torch(tmp_path,monkeypatch):
    ns=setup_namespace(tmp_path)
    folder=ns['environment_folder'](tmp_path,sys.version_info)
    assert folder.name.endswith('_text_lora')
    (folder/'bin').mkdir(parents=True);(folder/'bin/python').touch();(folder/'pyvenv.cfg').write_text('fixture')
    calls=[];inspections=0
    def run(args,**kwargs):
        nonlocal inspections
        calls.append(args)
        if len(args)>2 and args[2]==ns['INSPECT']:
            pending=[['peft==0.14.0'],['huggingface-hub<1.0,>=0.25.0'],[]][min(inspections,2)]
            inspections+=1
            data={'python':'3.13','runtime_version':list(sys.version_info[:2]),'versions':{'torch':'2.6.0+cu124'},
                  'torch_present':True,'dependency_issues':pending,'install_requirements':pending}
            return SimpleNamespace(returncode=0,stdout=json.dumps(data),stderr='')
        if len(args)>2 and args[2]==ns['IMPORTS']:
            return SimpleNamespace(returncode=0,stdout=json.dumps({'torch':'2.6.0+cu124','imports_verified':True,'cuda_available':True}),stderr='')
        assert '--no-deps' in args and '--ignore-installed' in args
        assert all(Requirement(raw).name!='torch' for raw in args[args.index('--no-input')+1:])
        return SimpleNamespace(returncode=0,stdout='',stderr='')
    monkeypatch.setattr(ns['subprocess'],'run',run)
    result=ns['prepare_environment']([])
    assert result['torch']=='2.6.0+cu124' and result['torch_reused']
    assert len([c for c in calls if 'install' in c])==2


@pytest.mark.parametrize('torch_version,torch_present', [('2.6.0+cu124',False),('2.6.0+cu124',True)])
def test_no_cpu_fallback_or_torch_replacement(tmp_path,monkeypatch,torch_version,torch_present):
    ns=setup_namespace(tmp_path)
    folder=ns['environment_folder'](tmp_path,sys.version_info)
    (folder/'bin').mkdir(parents=True);(folder/'bin/python').touch();(folder/'pyvenv.cfg').write_text('fixture')
    def run(args,**kwargs):
        assert 'install' not in args
        value={'python':'3.13','runtime_version':list(sys.version_info[:2]),'versions':{'torch':torch_version},
               'torch_present':torch_present,'dependency_issues':['torch changed'], 'install_requirements':['torch>=2.7']}
        return SimpleNamespace(returncode=0,stdout=json.dumps(value),stderr='')
    monkeypatch.setattr(ns['subprocess'],'run',run)
    with pytest.raises(ns['SetupError'],match='PyTorch'):ns['prepare_environment']([])


def test_real_offline_private_install_reuses_runtime_torch(tmp_path,monkeypatch):
    ns=setup_namespace(tmp_path)
    (tmp_path/'requirements.dataset300.txt').write_text('aes-lora-setup-fixture==1.0\n')
    folder=ns['environment_folder'](tmp_path,sys.version_info)
    venv.EnvBuilder(with_pip=False,system_site_packages=True).create(folder)
    site=folder/'lib'/f'python{sys.version_info.major}.{sys.version_info.minor}'/'site-packages'
    parent=[p for p in sys.path if p.endswith(('site-packages','dist-packages'))]
    (site/'runtime_fixture.pth').write_text('\n'.join(parent)+'\n')
    wheels=tmp_path/'wheels';wheels.mkdir()
    with zipfile.ZipFile(wheels/'aes_lora_setup_fixture-1.0-py3-none-any.whl','w') as wheel:
        entries={'aes_lora_setup_fixture/__init__.py':"VALUE='TECHNICAL_FIXTURE_ONLY'\n",
                 'aes_lora_setup_fixture-1.0.dist-info/METADATA':'Metadata-Version: 2.1\nName: aes-lora-setup-fixture\nVersion: 1.0\nRequires-Dist: torch>=2.6.0\n',
                 'aes_lora_setup_fixture-1.0.dist-info/WHEEL':'Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n'}
        entries['aes_lora_setup_fixture-1.0.dist-info/RECORD']=''.join(p+',,\n' for p in entries)+'aes_lora_setup_fixture-1.0.dist-info/RECORD,,\n'
        for path,data in entries.items():wheel.writestr(path,data)
    monkeypatch.setenv('PIP_NO_INDEX','1');monkeypatch.setenv('PIP_FIND_LINKS',str(wheels))
    fake=broken_vision(tmp_path);monkeypatch.setenv('PYTHONPATH',str(fake))
    result=ns['prepare_environment']([])
    assert result['imports_verified'] and result['torch_reused'] and result['packages_action']=='installed'
    assert result['transformers_backend']=='pytorch_text_only'
    assert not (site/'torch').exists() and not list(site.glob('torch-*.dist-info'))
    log=Path(result['log_file']).read_text()
    assert '--no-deps' in log and "Can't uninstall" not in log


def test_import_failure_shows_original_cause_not_only_log_path(tmp_path,monkeypatch):
    ns=setup_namespace(tmp_path)
    def run(args,**kwargs):
        if args[2]==ns['INSPECT']:
            value={'python':'3.13','runtime_version':list(sys.version_info[:2]),'versions':{'torch':'2.6.0'},
                   'torch_present':True,'dependency_issues':[],'install_requirements':[]}
            return SimpleNamespace(returncode=0,stdout=json.dumps(value),stderr='')
        return SimpleNamespace(returncode=1,stdout='',stderr='RuntimeError: TECHNICAL_IMPORT_FAILURE')
    monkeypatch.setattr(ns['subprocess'],'run',run)
    with pytest.raises(ns['SetupError'],match='TECHNICAL_IMPORT_FAILURE'):
        ns['prepare_environment'](['--verify-python',str(tmp_path/'worker')])
    assert not (tmp_path/'outputs/dataset300_setup.json').exists()
