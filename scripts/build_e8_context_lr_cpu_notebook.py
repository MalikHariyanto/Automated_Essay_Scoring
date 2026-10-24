"""Varian CPU E8; arsitektur, LR, konteks pair dan dua slot tetap."""
import ast
import copy
import hashlib
import json
from pathlib import Path
from build_context_pair_notebook import source, replace_once

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'notebooks/experiments/AES_300_E8_ContextPair_LR5e6_GPU.ipynb'
DESTINATION=BASE.with_name('AES_300_E8_ContextPair_LR5e6_CPU.ipynb')


def build():
    original=BASE.read_bytes();n=copy.deepcopy(json.loads(original))
    old=n['metadata']['aes_standalone']['code_sha256']
    for cell in n['cells']:
        s=source(cell).replace(old,'__NOTEBOOK_CODE_SHA256__').replace('AES_300_E8_ContextPair_LR5e6_GPU','AES_300_E8_ContextPair_LR5e6_CPU')
        s=s.replace("lesson_version='e8_context_pair_lr5e6_gpu_v1'","lesson_version='e8_context_pair_lr5e6_cpu_v1'")
        if cell['cell_type']=='code':cell['outputs']=[];cell['execution_count']=None
        cell['source']=s
    n['cells'][10]['source']=replace_once(source(n['cells'][10]),"'device': 'auto'","'device': 'cpu'")
    s=source(n['cells'][7])
    s=replace_once(s,"'AES300_REQUIRE_GPU':'1' if IN_COLAB else os.environ.get('AES300_REQUIRE_GPU','0')","'AES300_REQUIRE_GPU':'0','CUDA_VISIBLE_DEVICES':''")
    s=replace_once(s,"('Perangkat','GPU: '+ready['device'] if ready['cuda_available'] else 'CPU — GPU belum terdeteksi')","('Perangkat','CPU — dipilih untuk seluruh model; tidak memakai kuota GPU')")
    n['cells'][7]['source']=s
    s=source(n['cells'][12]).replace("device='auto'","device='cpu'")
    start=s.index('# Colab harus mendeteksi GPU');end=s.index('# BGE tidak mendapat learning rate',start)
    s=s[:start]+'''# CPU dipilih eksplisit, tidak memerlukan GPU atau kuota T4.
REQUIRE_GPU=False
torch.set_num_threads(2)
gpu=check_gpu(require_gpu=False)
print('Perangkat training: CPU | float32 tanpa AMP | thread:',torch.get_num_threads(),flush=True)
'''+s[end:]
    n['cells'][12]['source']=s
    s=source(n['cells'][0]).replace('· GPU\n','· CPU\n')
    start=s.index('**GPU wajib:**');end=s.index('\n\n**Model:**',start)
    s=s[:start]+'''**CPU:** Runtime → Change runtime type → CPU, lalu jalankan dari atas. Tidak memerlukan kuota GPU. Model memakai float32 tanpa AMP dan dua thread. Unggah fix_dataset_300.xlsx jika diminta. Kode tersedia langsung dalam notebook. Training CPU jauh lebih lama daripada T4; epoch dan metode tidak dipangkas.

**Run CPU terpisah:** tidak mengambil checkpoint/hasil GPU atau eksperimen sebelumnya. S1/S2/S3 dilatih baru, satu seed. Melanjutkan progres berlaku untuk run CPU ini dengan sumber, data dan pengaturan yang sama. Folder GPU dan hasil lama tidak perlu dihapus.''' +s[end:]
    s=s.replace('GPU/versi/seed yang berbeda','Perangkat/versi/seed yang berbeda')
    s=s.replace('E6 memakai jawaban saja dengan LR5e-6.','E6 memakai jawaban saja dengan LR5e-6. CPU dan GPU dapat memberi hasil berbeda karena presisi/implementasi numerik.')
    n['cells'][0]['source']=s
    for cell in n['cells']:
        if cell['cell_type']=='markdown':
            cell['source']=source(cell).replace('PyTorch runtime dipertahankan agar akses GPU yang tersedia tidak diganti wheel CPU.','PyTorch yang cocok dipertahankan; model tetap memakai CPU sesuai konfigurasi.')
    codes=[source(c).removeprefix('%%aes\n') for c in n['cells'] if c['cell_type']=='code']
    for code in codes:ast.parse(code)
    fingerprint=hashlib.sha256('\n'.join(codes).encode()).hexdigest()
    for cell in n['cells']:cell['source']=source(cell).replace('__NOTEBOOK_CODE_SHA256__',fingerprint)
    n['metadata']['aes_standalone']['code_sha256']=fingerprint
    n['metadata'].pop('accelerator',None);n['metadata']['colab']['name']=DESTINATION.name
    n['metadata']['aes_experiment'].update(name='E8_CONTEXT_PAIR_LR5E6_CPU',device='cpu',precision='float32',cpu_threads=2,
        gpu_notebook_sha256=hashlib.sha256(original).hexdigest(),drive_root='AES_300_E8_ContextPair_LR5e6_CPU_Privat')
    DESTINATION.write_text(json.dumps(n,ensure_ascii=False,indent=1)+'\n')
    (ROOT/'aes-learning/notebooks/experiments'/DESTINATION.name).write_bytes(DESTINATION.read_bytes())
    assert BASE.read_bytes()==original
    print(json.dumps({'notebook':str(DESTINATION.relative_to(ROOT)),'code_sha256':fingerprint,'gpu_unchanged':True}))


if __name__=='__main__':build()
