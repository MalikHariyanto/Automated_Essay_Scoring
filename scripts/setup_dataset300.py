"""Setup worker terpisah; pertahankan PyTorch CUDA bawaan Colab."""
import json
import os
from pathlib import Path
import subprocess
import sys
import venv

ROOT = Path(__file__).resolve().parents[1]


def environment_folder(root, version):
    version = tuple(version[:2])
    if version not in ((3, 12), (3, 13)):
        raise RuntimeError('Setup AES mendukung Python 3.12 atau 3.13. Versi lain belum diuji.')
    # Jangan memakai ulang binary/site-packages3.12 pada kernel3.13, atau sebaliknya.
    return Path(root)/'outputs'/('dataset300_env' if version == (3, 12) else 'dataset300_env_py313')


def main():
    folder = environment_folder(ROOT, sys.version_info)
    print(f'Setup AES: Python {sys.version.split()[0]} → {folder}', flush=True)
    if folder.exists() and not (folder/'pyvenv.cfg').is_file():
        raise RuntimeError('Folder target bukan venv; tidak ditimpa.')
    if not (folder/'bin/python').exists():
        venv.EnvBuilder(with_pip=False, system_site_packages=True).create(folder)
    python = str(folder/'bin/python')
    env = {**os.environ, 'HF_HOME': str(ROOT/'outputs/cache/huggingface'),
           'PIP_DISABLE_PIP_VERSION_CHECK': '1'}
    def execute(args, **kwargs):
        return subprocess.run(args, check=True, cwd=ROOT, env=env, **kwargs)
    execute([python, '-c', 'import sys; assert sys.version_info[:2] == '+repr(tuple(sys.version_info[:2]))+
             ', "Venv memakai Python berbeda dari runtime; gunakan folder environment baru."'])
    pip = [sys.executable, '-m', 'pip', '--python', python]
    execute(pip+['install', 'pip==25.1.1'])
    probe = subprocess.run([python, '-c', 'import torch; print(torch.__version__)'], cwd=ROOT, env=env)
    if probe.returncode:
        execute(pip+['install', 'torch==2.6.0', '--index-url', 'https://download.pytorch.org/whl/cpu'])
    execute(pip+['install', '-r', str(ROOT/'requirements.dataset300.txt')])
    # Proses baru: tidak memakai NumPy/Pandas yang mungkin telah diimpor kernel.
    execute([python, '-c', '''
import json, sys
from importlib.metadata import version
import torch, numpy, pandas, scipy, sklearn, openpyxl
from transformers import AutoModel, AutoTokenizer
from packaging.version import Version
if Version(torch.__version__.split('+')[0]) < Version('2.6.0'):
    raise RuntimeError('PyTorch minimal 2.6.0 diperlukan; gunakan runtime Colab yang lebih baru.')
print(json.dumps({'python':sys.version.split()[0], 'torch':torch.__version__,
 'cuda_available':torch.cuda.is_available(), 'checkpoint_loaded':False,
 'versions':{p:version(p) for p in ('numpy','pandas','scipy','scikit-learn','transformers','openpyxl')}}))
'''])
    ready = ROOT/'outputs/dataset300_setup.json'
    ready.write_text(json.dumps({'python': python, 'runtime_version': list(sys.version_info[:2]),
                                 'imports_verified': True}, indent=2)+'\n')
    print('Worker siap:', python, flush=True)


if __name__ == '__main__':
    main()
