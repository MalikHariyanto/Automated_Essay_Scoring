"""Setup CPU terisolasi untuk notebook; tidak mengubah Python bawaan Colab."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import venv


PROBE = """
import json, sys
from importlib.metadata import version
import numpy, pandas, scipy, sklearn, torch
from transformers import AutoTokenizer, AutoModel
from FlagEmbedding import BGEM3FlagModel
print(json.dumps({
    'python': sys.version.split()[0], 'interpreter': sys.executable,
    'versions': {p: version(p) for p in
        ['numpy','scipy','scikit-learn','torch','transformers','FlagEmbedding']},
    'imports_ok': True, 'checkpoint_loaded': False,
    'cuda_available': torch.cuda.is_available()
}))
"""


def check_runtime(version_info):
    if tuple(version_info[:2]) != (3, 12):
        raise RuntimeError(
            'Setup notebook ini diuji dengan Python 3.12. Pada Colab pilih '
            'Runtime > Change runtime type > Runtime version 2026.07, CPU. '
            'Setup dihentikan sebelum mengunduh paket.'
        )


def create_environment(environment):
    """Tanpa ensurepip: distro Colab dapat menyediakan venv tanpa bundel pip."""
    environment = Path(environment).resolve()
    if environment.exists() and not (environment / 'pyvenv.cfg').is_file():
        raise RuntimeError(f'{environment} bukan virtual environment; tidak ditimpa.')
    interpreter = environment / 'bin/python'
    if not interpreter.exists():
        venv.EnvBuilder(with_pip=False, system_site_packages=False).create(environment)
    return interpreter


def ensure_pip(interpreter, root, child_env):
    """Pulihkan juga venv parsial: pip --python tidak memerlukan pip target."""
    probe = subprocess.run([str(interpreter), '-m', 'pip', '--version'],
                           cwd=root, env=child_env, capture_output=True, text=True)
    if probe.returncode:
        print('Pip target belum siap; bootstrap melalui pip induk --python.', flush=True)
        # Resmi sejak pip 22.3. Tidak mengunduh/menjalankan get-pip.py atau memakai
        # ensurepip, dan tidak memasang paket ke Python sistem Colab.
        subprocess.run([
            sys.executable, '-m', 'pip', '--python', str(interpreter), 'install',
            '--disable-pip-version-check', '--only-binary=:all:', 'pip==25.0.1'
        ], cwd=root, env=child_env, check=True)
    subprocess.run([str(interpreter), '-m', 'pip', '--version'],
                   cwd=root, env=child_env, check=True)


def setup(root, environment):
    check_runtime(sys.version_info)
    root, environment = Path(root).resolve(), Path(environment).resolve()
    lock = root / 'requirements.lock.txt'
    interpreter = create_environment(environment)
    marker = environment / 'aes_setup.json'
    fingerprint = hashlib.sha256(lock.read_bytes()).hexdigest()
    child_env = os.environ.copy()
    child_env.pop('PYTHONPATH', None)
    child_env.pop('PYTHONHOME', None)
    child_env['HF_HOME'] = str(root / 'outputs/cache/huggingface')
    child_env['PIP_CACHE_DIR'] = str(root / 'outputs/cache/pip')
    child_env['OMP_NUM_THREADS'] = '2'
    child_env['MKL_NUM_THREADS'] = '2'
    ensure_pip(interpreter, root, child_env)
    previous = json.loads(marker.read_text()) if marker.exists() else {}
    if previous.get('lock_sha256') != fingerprint:
        subprocess.run([
            str(interpreter), '-m', 'pip', 'install',
            '--disable-pip-version-check', '-r', str(lock)
        ], cwd=root, env=child_env, check=True)
    subprocess.run([
        str(interpreter), '-m', 'pip', 'install', '--disable-pip-version-check',
        '--no-deps', '--no-build-isolation', '-e', str(root)
    ], cwd=root, env=child_env, check=True)
    subprocess.run([str(interpreter), '-m', 'pip', 'check'],
                   cwd=root, env=child_env, check=True)
    result = subprocess.run(
        [str(interpreter), '-c', PROBE], cwd=root, env=child_env,
        check=True, capture_output=True, text=True
    )
    # Error import tetap gagal. Tidak memakai hasil diagnostik sebagai skor esai.
    proof = json.loads(result.stdout.strip().splitlines()[-1])
    proof['lock_sha256'] = fingerprint
    marker.write_text(json.dumps(proof, indent=2), encoding='utf-8')
    print(json.dumps(proof, indent=2))
    print('SETUP OK: proses AES terpisah dari memori notebook Colab.')
    return proof


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument('--environment')
    args = parser.parse_args()
    root = Path(args.root).resolve()
    setup(root, args.environment or root / 'outputs/notebook_env')
