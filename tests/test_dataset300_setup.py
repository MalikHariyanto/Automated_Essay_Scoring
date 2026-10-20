"""Regresi setup Colab: Python3.13 dan preservasi torch bawaan runtime."""
import importlib.util
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace
import venv
import zipfile

import pytest


def setup_module_source():
    path = Path(__file__).resolve().parents[1]/'scripts/setup_dataset300.py'
    spec = importlib.util.spec_from_file_location('dataset300_setup_under_test',path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('version,folder', [((3,12,14),'dataset300_env'),((3,13,5),'dataset300_env_py313')])
def test_supported_runtime_and_existing_torch_preserved(tmp_path,monkeypatch,version,folder):
    module = setup_module_source()
    environment = tmp_path/'outputs'/folder
    (environment/'bin').mkdir(parents=True)
    (environment/'bin/python').touch()
    (environment/'pyvenv.cfg').write_text('fixture only')
    shutil.copy(Path(__file__).resolve().parents[1]/'requirements.dataset300.txt',tmp_path)
    monkeypatch.setattr(module,'ROOT',tmp_path)
    monkeypatch.setattr(module.sys,'version_info',version)
    calls=[]
    def successful(args,**kwargs):
        calls.append(args)
        if args[2]==module.INSPECT:
            result={'python':'.'.join(map(str,version)),'runtime_version':list(version[:2]),
                    'versions':{'torch':'2.6.0+cu124'},'torch_present':True,
                    'dependency_issues':[],'dependency_count':53}
        elif args[2]==module.IMPORTS:
            result={'torch':'2.6.0+cu124','cuda_available':True,'device':'FIXTURE GPU',
                    'checkpoint_loaded':False,'imports_verified':True}
        else:pytest.fail('Pustaka yang cocok tidak boleh dipasang ulang')
        return SimpleNamespace(returncode=0,stdout=json.dumps(result),stderr='')
    monkeypatch.setattr(module.subprocess,'run',successful)
    module.main([])
    ready=json.loads((tmp_path/'outputs/dataset300_setup.json').read_text())
    assert ready['python'] == str(environment/'bin/python')
    assert ready['runtime_version'] == list(version[:2]) and ready['imports_verified']
    assert not any('--index-url' in call for call in calls), 'Torch bawaan tidak dipasang ulang/diganti CPU'
    assert ready['torch_reused'] and ready['packages_action']=='reused' and ready['cuda_available']
    assert len(calls)==3 and all('-c' in call for call in calls)
    selected=module.active_requirements(tmp_path/'requirements.dataset300.txt',version)
    assert [x for x in selected if x.startswith('transformers')]==[
        'transformers=='+('4.44.2' if version[:2]==(3,12) else '4.48.3')]


def test_unsupported_runtime_stops_before_installation(tmp_path,monkeypatch):
    module=setup_module_source()
    monkeypatch.setattr(module,'ROOT',tmp_path)
    monkeypatch.setattr(module.sys,'version_info',(3,14,0))
    def forbidden(*a,**k):
        pytest.fail('Runtime tidak didukung tidak boleh memasang paket')
    monkeypatch.setattr(module.subprocess,'run',forbidden)
    with pytest.raises(RuntimeError,match='3.12 atau 3.13'):
        module.main([])
    assert not (tmp_path/'outputs').exists()


def test_installs_private_overlay_without_uninstalling_or_replacing_torch(tmp_path,monkeypatch):
    """Pip nyata, wheel fixture lokal: versi bawaan 2.0 tetap ada; worker memakai 1.0."""
    module=setup_module_source();monkeypatch.setattr(module,'ROOT',tmp_path)
    (tmp_path/'requirements.dataset300.txt').write_text('aes-setup-fixture==1.0\n')
    environment=module.environment_folder(tmp_path,sys.version_info)
    venv.EnvBuilder(with_pip=False,system_site_packages=True).create(environment)
    site=environment/'lib'/f'python{sys.version_info.major}.{sys.version_info.minor}'/'site-packages'
    inherited=tmp_path/'runtime_packages';inherited.mkdir()
    (inherited/'aes_setup_fixture').mkdir()
    (inherited/'aes_setup_fixture/__init__.py').write_text("VERSION='2.0'\n")
    metadata=inherited/'aes_setup_fixture-2.0.dist-info';metadata.mkdir()
    original='Metadata-Version: 2.1\nName: aes-setup-fixture\nVersion: 2.0\n'
    (metadata/'METADATA').write_text(original)
    # Meniru runtime Colab: torch dan dependensinya tersedia dari luar worker.
    parent_sites=[p for p in sys.path if p.endswith(('site-packages','dist-packages'))]
    (site/'runtime_fixture.pth').write_text(str(inherited)+'\n'+'\n'.join(parent_sites)+'\n')
    wheels=tmp_path/'wheels';wheels.mkdir()
    with zipfile.ZipFile(wheels/'aes_setup_fixture-1.0-py3-none-any.whl','w') as wheel:
        entries={'aes_setup_fixture/__init__.py':"VERSION='1.0'\n",
                 'aes_setup_fixture-1.0.dist-info/METADATA':'Metadata-Version: 2.1\nName: aes-setup-fixture\nVersion: 1.0\n',
                 'aes_setup_fixture-1.0.dist-info/WHEEL':'Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n'}
        entries['aes_setup_fixture-1.0.dist-info/RECORD']=''.join(name+',,\n' for name in entries)+'aes_setup_fixture-1.0.dist-info/RECORD,,\n'
        for name,text in entries.items():wheel.writestr(name,text)
    monkeypatch.setenv('PIP_NO_INDEX','1');monkeypatch.setenv('PIP_FIND_LINKS',str(wheels))
    result=module.main([])
    assert result['imports_verified'] and result['torch_reused']
    assert result['versions']['aes-setup-fixture']=='1.0' and result['packages_action']=='installed'
    assert (metadata/'METADATA').read_text()==original
    log=Path(result['log_file']).read_text()
    assert "Can't uninstall" not in log and 'Attempting uninstall' not in log


@pytest.mark.parametrize('failure',['import','dependencies'])
def test_failed_check_does_not_publish_stale_ready_status(tmp_path,monkeypatch,failure):
    module=setup_module_source();monkeypatch.setattr(module,'ROOT',tmp_path)
    shutil.copy(Path(__file__).resolve().parents[1]/'requirements.dataset300.txt',tmp_path)
    outputs=tmp_path/'outputs';outputs.mkdir()
    (outputs/'dataset300_setup.json').write_text('{"imports_verified":true}')
    calls=[]
    def run(args,**kwargs):
        calls.append(args)
        if args[2]==module.INSPECT:
            value={'python':'3.13','runtime_version':list(sys.version_info[:2]),'versions':{'torch':'2.6.0'},
                   'torch_present':True,'dependency_issues':['required version missing'] if failure=='dependencies' else [],'dependency_count':53}
            return SimpleNamespace(returncode=0,stdout=json.dumps(value),stderr='')
        return SimpleNamespace(returncode=1,stdout='',stderr='FIXTURE import error: <script>not executed</script>')
    monkeypatch.setattr(module.subprocess,'run',run)
    with pytest.raises(module.SetupError):module.main(['--verify-python',str(tmp_path/'worker/bin/python')])
    assert not (outputs/'dataset300_setup.json').exists()
    assert 'FIXTURE import error' in (outputs/'dataset300_setup.log').read_text() if failure=='import' else len(calls)==1
    assert all('install' not in call for call in calls)


def test_existing_torch_import_failure_is_not_replaced_by_cpu(tmp_path,monkeypatch):
    module=setup_module_source();monkeypatch.setattr(module,'ROOT',tmp_path)
    shutil.copy(Path(__file__).resolve().parents[1]/'requirements.dataset300.txt',tmp_path)
    environment=module.environment_folder(tmp_path,sys.version_info)
    (environment/'bin').mkdir(parents=True);(environment/'bin/python').touch()
    (environment/'pyvenv.cfg').write_text('fixture')
    calls=[]
    def run(args,**kwargs):
        calls.append(args)
        if args[2]==module.INSPECT:
            value={'python':'3.13','runtime_version':list(sys.version_info[:2]),'versions':{'torch':'2.6.0+cu124'},
                   'torch_present':True,'dependency_issues':[],'dependency_count':53}
            return SimpleNamespace(returncode=0,stdout=json.dumps(value),stderr='')
        return SimpleNamespace(returncode=1,stdout='',stderr='FIXTURE broken CUDA library')
    monkeypatch.setattr(module.subprocess,'run',run)
    with pytest.raises(module.SetupError):module.main([])
    assert all('--index-url' not in call and 'install' not in call for call in calls)
