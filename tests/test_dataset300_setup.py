"""Regresi setup Colab: Python3.13 dan preservasi torch bawaan runtime."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

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
    monkeypatch.setattr(module,'ROOT',tmp_path)
    monkeypatch.setattr(module.sys,'version_info',version)
    calls=[]
    def successful(args,**kwargs):
        calls.append(args)
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(module.subprocess,'run',successful)
    module.main()
    ready=json.loads((tmp_path/'outputs/dataset300_setup.json').read_text())
    assert ready['python'] == str(environment/'bin/python')
    assert ready['runtime_version'] == list(version[:2]) and ready['imports_verified']
    assert not any('--index-url' in call for call in calls), 'Torch bawaan tidak dipasang ulang/diganti CPU'
    assert any('-r' in call and str(tmp_path/'requirements.dataset300.txt') in call for call in calls)


def test_unsupported_runtime_stops_before_installation(tmp_path,monkeypatch):
    module=setup_module_source()
    monkeypatch.setattr(module,'ROOT',tmp_path)
    monkeypatch.setattr(module.sys,'version_info',(3,14,0))
    def forbidden(*a,**k):
        pytest.fail('Runtime tidak didukung tidak boleh memasang paket')
    monkeypatch.setattr(module.subprocess,'run',forbidden)
    with pytest.raises(RuntimeError,match='3.12 atau 3.13'):
        module.main()
    assert not (tmp_path/'outputs').exists()
