"""Periksa kebutuhan AES, pasang overlay privat bila perlu, dan pertahankan torch runtime."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import venv

try:
    from packaging.requirements import Requirement
except ImportError:
    from pip._vendor.packaging.requirements import Requirement

ROOT = Path(__file__).resolve().parents[1]

# Probe berjalan di Python worker, bukan kernel yang mungkin sudah memuat NumPy lama.
INSPECT = r'''
import json, sys
from importlib import metadata, util
try:
    from packaging.requirements import Requirement
except ImportError:
    from pip._vendor.packaging.requirements import Requirement
roots=json.loads(sys.argv[1])+['torch>=2.6.0']
versions={}
for raw in roots:
    name=Requirement(raw).name
    try:versions[name]=metadata.version(name)
    except metadata.PackageNotFoundError:versions[name]=None
queue=[Requirement(raw) for raw in roots];seen=set();issues=[]
while queue:
    req=queue.pop()
    if req.marker and not req.marker.evaluate({'extra':''}):continue
    try:dist=metadata.distribution(req.name)
    except metadata.PackageNotFoundError:
        issues.append(str(req)+' belum tersedia');continue
    if not req.specifier.contains(dist.version,prereleases=True):
        issues.append(str(req)+' diperlukan; tersedia '+dist.version)
    name=dist.metadata['Name'].lower().replace('_','-')
    if name in seen:continue
    seen.add(name)
    queue.extend(Requirement(raw) for raw in (dist.requires or []))
print(json.dumps({'python':sys.version.split()[0],'runtime_version':list(sys.version_info[:2]),
 'versions':versions,'torch_present':util.find_spec('torch') is not None,
 'dependency_issues':sorted(set(issues)),'dependency_count':len(seen)}))
'''

IMPORTS = r'''
import json
import torch, numpy, pandas, scipy, sklearn, openpyxl
from transformers import AutoModel, AutoTokenizer
available=torch.cuda.is_available()
print(json.dumps({'torch':torch.__version__,'cuda_available':available,
 'device':torch.cuda.get_device_name(0) if available else 'CPU',
 'checkpoint_loaded':False,'imports_verified':True}))
'''


def environment_folder(root, version):
    version=tuple(version[:2])
    if version not in ((3,12),(3,13)):
        raise RuntimeError('Setup AES mendukung Python 3.12 atau 3.13. Versi lain belum diuji.')
    return Path(root)/'outputs'/('dataset300_env' if version==(3,12) else 'dataset300_env_py313')


def active_requirements(path, version):
    """Hanya aturan untuk versi Python aktif; tidak mencoba semua versi Transformers."""
    selected=[]
    for line in Path(path).read_text().splitlines():
        line=line.split('#',1)[0].strip()
        if not line:continue
        req=Requirement(line)
        if req.marker and not req.marker.evaluate({'python_version':'.'.join(map(str,version[:2])), 'extra':''}):continue
        selected.append(req.name+str(req.specifier))
    return selected


class SetupError(RuntimeError):
    pass


def main(argv=None):
    parser=argparse.ArgumentParser()
    parser.add_argument('--verify-python',help='Periksa worker yang disediakan pengguna tanpa memasang paket.')
    args=parser.parse_args(argv)
    folder=environment_folder(ROOT,sys.version_info)
    outputs=ROOT/'outputs';outputs.mkdir(parents=True,exist_ok=True)
    ready=outputs/'dataset300_setup.json';ready.unlink(missing_ok=True)
    logfile=outputs/'dataset300_setup.log'
    requirements=active_requirements(ROOT/'requirements.dataset300.txt',sys.version_info)
    env={**os.environ,'HF_HOME':str(outputs/'cache/huggingface'),'PIP_DISABLE_PIP_VERSION_CHECK':'1'}
    with logfile.open('w') as log:
        def execute(command,stage):
            log.write(stage+'\n'+json.dumps(command)+'\n');log.flush()
            process=subprocess.run(command,cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
            log.write(process.stdout or '');log.write(process.stderr or '');log.flush()
            if process.returncode:
                raise SetupError(stage+' gagal. Rincian lengkap: '+str(logfile))
            return process.stdout
        def inspect(python):
            result=json.loads(execute([python,'-c',INSPECT,json.dumps(requirements)],'Pemeriksaan versi dan dependensi'))
            if result['runtime_version']!=list(sys.version_info[:2]):
                raise SetupError('Versi Python worker berbeda dari runtime. Gunakan lingkungan untuk versi runtime ini.')
            return result

        print('1/3 · Memeriksa lingkungan dan versi pustaka AES...',flush=True)
        if args.verify_python:
            # abspath menjaga symlink bin/python venv; resolve akan memakai Python global.
            python=str(Path(args.verify_python).absolute())
        else:
            if folder.exists() and not (folder/'pyvenv.cfg').is_file():
                raise SetupError('Folder target bukan lingkungan Python AES; tidak ditimpa.')
            if not (folder/'bin/python').exists():
                # Torch CUDA tetap berasal dari runtime. Paket AES yang berbeda dipasang lokal.
                venv.EnvBuilder(with_pip=False,system_site_packages=True).create(folder)
            python=str(folder/'bin/python')
        before=inspect(python)
        torch_version=before['versions'].get('torch')
        if torch_version and not Requirement('torch>=2.6.0').specifier.contains(torch_version,prereleases=True):
            raise SetupError('PyTorch minimal 2.6.0 diperlukan. Pilih runtime Colab yang lebih baru; torch bawaan tidak diganti.')
        installed=False
        if before['dependency_issues']:
            if args.verify_python:
                raise SetupError('Worker yang dipilih belum memenuhi kebutuhan AES. Rincian: '+str(logfile))
            print('2/3 · Menyiapkan versi pustaka khusus AES; pustaka bawaan tetap disimpan...',flush=True)
            pip=[sys.executable,'-m','pip','--python',python,'install','--ignore-installed','--no-input']
            if not before['torch_present']:
                execute(pip+['torch==2.6.0','--index-url','https://download.pytorch.org/whl/cpu'],
                        'Pemasangan PyTorch CPU karena runtime tidak menyediakan torch')
            selected=outputs/('dataset300_requirements_py'+''.join(map(str,sys.version_info[:2]))+'.txt')
            selected.write_text('\n'.join(requirements)+'\n')
            execute(pip+['-r',str(selected)],'Pemasangan pustaka AES di lingkungan privat')
            installed=True
        else:
            print('2/3 · Versi pustaka sudah cocok; tidak perlu instalasi ulang.',flush=True)
        after=inspect(python)
        if after['dependency_issues']:
            raise SetupError('Dependensi AES belum cocok: '+'; '.join(after['dependency_issues'][:3])+'. Rincian: '+str(logfile))
        # Jangan mengganti torch yang terpasang tetapi rusak dengan wheel CPU secara diam-diam.
        print('3/3 · Memeriksa import dan perangkat perhitungan...',flush=True)
        checked=json.loads(execute([python,'-c',IMPORTS],'Pemeriksaan import AES dan PyTorch'))
        result={**after,**checked,'python_version':after['python'],'python':python,
                'packages_action':'installed' if installed else 'reused','torch_reused':before['torch_present'],
                'log_file':str(logfile),'requirements_sha256':hashlib.sha256((ROOT/'requirements.dataset300.txt').read_bytes()).hexdigest()}
        ready.write_text(json.dumps(result,indent=2)+'\n')
    print('Persiapan AES selesai. Versi, dependensi, dan import sudah diperiksa.',flush=True)
    return result


if __name__=='__main__':
    try:main()
    except SetupError as error:
        print('Setup AES belum berhasil: '+str(error),file=sys.stderr)
        sys.exit(1)
