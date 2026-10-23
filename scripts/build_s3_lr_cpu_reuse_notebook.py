"""E6 CPU dengan reuse S1/S2, diturunkan dari versi GPU tanpa mengubah metode."""
import ast
import copy
import hashlib
import json
from pathlib import Path

from build_s3_lr_cpu_notebook import replace_once,source

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'notebooks/experiments/AES_300_E6_S3_LR5e6_GPU_Reuse.ipynb'
DESTINATION=ROOT/'notebooks/experiments/AES_300_E6_S3_LR5e6_CPU_Reuse.ipynb'


def build():
    original=BASE.read_bytes();notebook=copy.deepcopy(json.loads(original))
    old_hash=notebook['metadata']['aes_standalone']['code_sha256']
    for cell in notebook['cells']:
        text=source(cell)
        text=text.replace('AES_300_E6_S3_LR5e6_GPU_Reuse','AES_300_E6_S3_LR5e6_CPU_Reuse')
        if cell['cell_type']=='code':
            text=text.replace(old_hash,'__NOTEBOOK_CODE_SHA256__')
            text=text.replace("lesson_version='e6_s3_lr5e6_gpu_reuse_v1'","lesson_version='e6_s3_lr5e6_cpu_reuse_v1'")
            if 'def prepare_environment(' in text:text=text.replace('_text_e6_gpu_reuse','_text_e6_cpu_reuse')
            if 'DEFAULTS = {' in text:text=replace_once(text,"'device': 'cuda'","'device': 'cpu'")
            if 'ROBERTA_CHECKPOINT=' in text:
                text=replace_once(text,"device='cuda'","device='cpu'")
                text=replace_once(text,"REQUIRE_GPU=True\ngpu=check_gpu(require_gpu=RUN_MODELS and REQUIRE_GPU)\nprint('Perangkat eksperimen S3:',gpu['name'],'— CUDA, hanya S3 dilatih baru.',flush=True)",
                    "REQUIRE_GPU=False\nprint('Perangkat eksperimen S3: CPU — float32, hanya S3 dilatih baru.',flush=True)")
                text=text.replace('Latihan baru di GPU','Latihan baru di CPU')
            if "PYTHON = os.environ.get('AES300_PYTHON'" in text:
                text=replace_once(text,"'AES300_REQUIRE_GPU':'1'","'AES300_REQUIRE_GPU':'0'")
                text=replace_once(text,"('Perangkat','GPU: '+ready['device'] if ready['cuda_available'] else 'GPU belum terdeteksi')", "('Perangkat','CPU — dipilih eksplisit; GPU tidak diperlukan')")
            cell['outputs'],cell['execution_count']=[],None
        else:
            if text.startswith('# AES 300'):
                text=text.replace('· GPU · Reuse S1/S2','· CPU · Reuse S1/S2')
                text=replace_once(text,'**GPU:** Runtime → Change runtime type → T4 GPU, lalu jalankan dari atas. CPU tidak dipilih otomatis.',
                    '**CPU:** Runtime → Change runtime type → Hardware accelerator: None, lalu jalankan dari atas. CPU dipilih eksplisit meskipun GPU tersedia. Training menggunakan float32, tanpa AMP dan tanpa kewajiban GPU. CPU dapat jauh lebih lama daripada T4.')
                text += '\n\n**Sumber dari GPU boleh:** model final S1/S2 dimuat dengan map_location=cpu dan dapat dipakai ulang pada CPU setelah seluruh pemeriksaan cocok. Versi Torch/Transformers/NumPy masih harus sama sesuai pemeriksaan sumber. Progres epoch S3 GPU tidak dipaksa dilanjutkan ke CPU; versi ini memulai S3 pada folder CPU baru. Untuk perbandingan LR yang terkontrol, catat perbedaan GPU AMP dan CPU float32.\n'
            else:
                text=text.replace('GPU wajib untuk S3 baru; S1/S2 diperiksa dan dipakai ulang dari run selesai.', 'CPU eksplisit untuk S3 baru; S1/S2 diperiksa dan dipakai ulang dari run selesai.')
                text=text.replace('Training S3 memakai AMP GPU seperti acuan; S1/S2 tidak dilatih ulang.', 'Training S3 memakai CPU float32 tanpa AMP; S1/S2 tidak dilatih ulang.')
        cell['source']=text;cell['metadata']={};cell.pop('attachments',None)
    normalized=[source(c).removeprefix('%%aes\n') for c in notebook['cells'] if c['cell_type']=='code']
    for code in normalized:ast.parse(code)
    fingerprint=hashlib.sha256('\n'.join(normalized).encode()).hexdigest()
    for cell in notebook['cells']:cell['source']=source(cell).replace('__NOTEBOOK_CODE_SHA256__',fingerprint)
    notebook['metadata']['aes_standalone']['code_sha256']=fingerprint
    notebook['metadata'].pop('accelerator',None)
    notebook['metadata']['colab'].pop('gpuType',None)
    notebook['metadata']['colab']['name']=DESTINATION.name
    notebook['metadata']['aes_experiment']={**notebook['metadata']['aes_experiment'],
        'name':'E6_S3_LR5E6_CPU_REUSE','base_sha256':hashlib.sha256(original).hexdigest(),'device':'cpu',
        'drive_root':'AES_300_E6_S3_LR5e6_CPU_Reuse_Privat'}
    DESTINATION.write_text(json.dumps(notebook,ensure_ascii=False,indent=1)+'\n')
    (ROOT/'aes-learning/notebooks/experiments'/DESTINATION.name).write_bytes(DESTINATION.read_bytes())
    assert BASE.read_bytes()==original
    print(json.dumps({'notebook':str(DESTINATION.relative_to(ROOT)),'code_sha256':fingerprint,'gpu_source_unchanged':True}))
    return DESTINATION


if __name__=='__main__':build()
