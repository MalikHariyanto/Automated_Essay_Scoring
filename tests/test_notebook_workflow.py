"""Fixture teknis: setup/runtime dan worker notebook, bukan skor penelitian."""
import importlib.util
import json
from pathlib import Path
import sys
import os
import hashlib
import types
import subprocess

import pytest
from requests.exceptions import RequestException
from aes.io import sha256


def script_module(name):
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(name, root / 'scripts' / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_python_313_rejected_before_environment_or_download(tmp_path, monkeypatch):
    setup = script_module('setup_notebook')
    with monkeypatch.context() as patch:
        patch.setattr(sys, 'version_info', (3, 13, 0))
        with pytest.raises(RuntimeError, match='Python 3.12'):
            setup.setup(tmp_path, tmp_path / 'env')
    assert not (tmp_path / 'env').exists()


def test_environment_creation_works_when_ensurepip_is_unusable(tmp_path, monkeypatch):
    setup = script_module('setup_notebook')
    def unusable_ensurepip(*args, **kwargs):
        raise AssertionError('Fixture teknis: ensurepip tidak tersedia')
    monkeypatch.setattr(setup.venv.EnvBuilder, '_setup_pip', unusable_ensurepip)
    environment = tmp_path / 'without_pip'
    interpreter = setup.create_environment(environment)
    assert interpreter.is_file()
    assert 'include-system-site-packages = false' in (environment / 'pyvenv.cfg').read_text()
    missing_pip = subprocess.run([str(interpreter), '-m', 'pip', '--version'],
                                 capture_output=True, text=True)
    assert missing_pip.returncode != 0
    # Kondisi parsial setelah kegagalan lama: interpreter ada, pip belum ada.
    sentinel = environment / 'keep_existing_file.txt'
    sentinel.write_text('Fixture teknis: jangan hapus lingkungan yang sudah ada.')
    assert setup.create_environment(environment) == interpreter
    assert sentinel.read_text().startswith('Fixture teknis')


def test_bootstrap_pip_failure_is_not_reported_as_setup_success(tmp_path, monkeypatch):
    setup = script_module('setup_notebook')
    interpreter = tmp_path / 'env/bin/python'
    calls = []
    def failure(command, **kwargs):
        calls.append(command)
        if '--version' in command:
            return subprocess.CompletedProcess(command, 1, '', 'No module named pip')
        raise subprocess.CalledProcessError(1, command)
    monkeypatch.setattr(setup.subprocess, 'run', failure)
    with pytest.raises(subprocess.CalledProcessError):
        setup.ensure_pip(interpreter, tmp_path, {})
    assert len(calls) == 2
    assert '--python' in calls[1] and str(interpreter) in calls[1]
    assert 'ensurepip' not in calls[1]


def test_draft_worker_runs_inspection_and_skips_learning_without_mutating_inputs(technical_source):
    _, _, cfg = technical_source
    worker = script_module('notebook_step')
    original = {key: sha256(cfg[key]) for key in ['csv', 'rubric']}
    worker.audit_step(cfg)
    worker.supervised_step(cfg, 'learning')
    output = Path(cfg['output'])
    preview = json.loads((output / 'notebook_preview.json').read_text())
    result = json.loads((output / 'notebook_supervised.json').read_text())
    assert result['status'] == 'skip_data_ineligible'
    assert result['teacher_training_run'] is False
    assert any('skor guru' in reason for reason in result['blockers'])
    assert preview['rubric']['confirmed'] is False
    assert preview['essay']['ready_for_training'] == 'false'
    assert all(preview['essay']['score_' + name] == '' for name in
               ['isi', 'organisasi', 'kebahasaan', 'mekanik'])
    assert {key: sha256(cfg[key]) for key in original} == original


def test_worker_reports_network_blocker_but_does_not_hide_model_bug(tmp_path, monkeypatch):
    worker = script_module('notebook_step')
    cfg = {'output': str(tmp_path)}
    def network_error(config):
        raise RequestException('Fixture teknis: jaringan tidak tersedia')
    monkeypatch.setattr(worker, 'inspect_models', network_error)
    worker.models_step(cfg)
    status = json.loads((tmp_path / 'notebook_models_status.json').read_text())
    assert status['status'] == 'checkpoint_access_blocked'
    assert 'RequestException' in status['reason']
    def shape_error(config):
        raise RuntimeError('Fixture teknis: shape output salah')
    monkeypatch.setattr(worker, 'inspect_models', shape_error)
    with pytest.raises(RuntimeError, match='shape output salah'):
        worker.models_step(cfg)


def test_colab_input_cell_uploads_to_private_folder_without_replacing_existing_rubric(
        technical_source, tmp_path, monkeypatch):
    # Tiruan dialog upload; data hanya fixture teknis, bukan unggahan siswa.
    _, _, cfg = technical_source
    source_root = Path(__file__).resolve().parents[1]
    root = tmp_path / 'notebook_checkout'
    (root / 'configs').mkdir(parents=True)
    (root / 'configs/default.json').write_bytes((source_root / 'configs/default.json').read_bytes())
    private = root / 'data/source'
    private.mkdir(parents=True)
    existing_rubric = private / 'rubrik_multitrait_DRAF.json'
    existing_rubric.write_bytes(Path(cfg['rubric']).read_bytes())
    original_hash = sha256(existing_rubric)
    csv_bytes = Path(cfg['csv']).read_bytes()
    def upload():
        assert Path.cwd() == private
        name = 'dataset_aes_multitrait_DRAF.csv'
        Path(name).write_bytes(csv_bytes)
        return {name: csv_bytes}
    google = types.ModuleType('google')
    colab = types.ModuleType('google.colab')
    colab.files = types.SimpleNamespace(upload=upload)
    monkeypatch.setitem(sys.modules, 'google', google)
    monkeypatch.setitem(sys.modules, 'google.colab', colab)
    monkeypatch.chdir(root)
    output = root / 'outputs'
    output.mkdir()
    namespace = dict(ROOT=root, OUTPUT=output, IN_COLAB=True, MODE='inspect',
                     Path=Path, os=os, json=json, hashlib=hashlib)
    nb = json.loads((source_root / 'notebooks/01_belajar_alur_AES.ipynb').read_text())
    cell = next(''.join(c['source']) for c in nb['cells']
                if c['cell_type'] == 'code' and ''.join(c['source']).startswith('CSV_PATH ='))
    exec(compile(cell, 'notebook_input_cell', 'exec'), namespace)
    assert (private / 'dataset_aes_multitrait_DRAF.csv').read_bytes() == csv_bytes
    assert sha256(existing_rubric) == original_hash
    assert not (root / 'dataset_aes_multitrait_DRAF.csv').exists()
    assert Path.cwd() == root
    effective = json.loads((output / 'notebook_config.json').read_text())
    assert effective['csv'] == str(private / 'dataset_aes_multitrait_DRAF.csv')


def first_cell_source():
    root = Path(__file__).resolve().parents[1]
    nb = json.loads((root/'notebooks/01_belajar_alur_AES.ipynb').read_text())
    return next(''.join(cell['source']) for cell in nb['cells']
                if cell['cell_type']=='code')


def test_git_timeout_stops_first_cell_and_preserves_existing_files(tmp_path, monkeypatch, capsys):
    root = tmp_path/'checkout';root.mkdir()
    sentinel = root/'keep.txt';sentinel.write_text('Fixture: file pengguna tidak dihapus.')
    monkeypatch.setenv('AES_NOTEBOOK_ROOT', str(root))
    actual_is_dir = Path.is_dir
    monkeypatch.setattr(Path, 'is_dir', lambda p: True if str(p)=='/content' else actual_is_dir(p))
    calls = []
    def stalled_git(command, **kwargs):
        calls.append(command)
        assert kwargs['env']['GIT_TERMINAL_PROMPT']=='0'
        assert kwargs['stdin']==subprocess.DEVNULL
        if 'status' in command:
            return subprocess.CompletedProcess(command,0,stdout='')
        if 'branch' in command:
            return subprocess.CompletedProcess(command,0,stdout='codex/aes-learning\n')
        assert 'fetch' in command and kwargs['timeout']==120
        raise subprocess.TimeoutExpired(command, kwargs['timeout'])
    monkeypatch.setattr(subprocess,'run',stalled_git)
    with pytest.raises(RuntimeError,match='melewati batas waktu'):
        exec(compile(first_cell_source(),'notebook_first_cell','exec'),{})
    assert sentinel.read_text()=='Fixture: file pengguna tidak dihapus.'
    assert not any('merge' in command for command in calls)
    assert not (root/'outputs').exists()
    captured=capsys.readouterr().out
    assert 'Mulai cell 1' in captured and 'Mengambil pembaruan dari GitHub' in captured
    assert 'CELL 1 SELESAI' not in captured


def test_first_cell_reports_stages_and_uses_shallow_clone(tmp_path, monkeypatch, capsys):
    root=tmp_path/'new_checkout'
    monkeypatch.setenv('AES_NOTEBOOK_ROOT',str(root))
    actual_is_dir=Path.is_dir
    monkeypatch.setattr(Path,'is_dir',lambda p: True if str(p)=='/content' else actual_is_dir(p))
    previous=Path.cwd()
    def clone(command,**kwargs):
        assert 'clone' in command and '--depth' in command
        assert kwargs['timeout']==120 and kwargs['env']['GIT_TERMINAL_PROMPT']=='0'
        (root/'scripts').mkdir(parents=True)
        (root/'scripts/setup_notebook.py').write_text('# Fixture: tidak menjalankan setup.')
        return subprocess.CompletedProcess(command,0)
    monkeypatch.setattr(subprocess,'run',clone)
    try:
        namespace={}
        exec(compile(first_cell_source(),'notebook_first_cell','exec'),namespace)
        assert namespace['INPUT_FORMAT']=='xlsx'
        captured=capsys.readouterr().out
        assert 'Mengambil source dari GitHub' in captured and 'CELL 1 SELESAI' in captured
        assert not (root/'outputs/notebook_env').exists()
    finally:
        os.chdir(previous)
