"""E21/E22 web Colab GPU, hasil lokal tanpa mount/autosave Google Drive."""
import ast
import copy
import hashlib
import json
from pathlib import Path
from build_e21_e22_notebooks import build as build_cpu, NAMES
from build_context_pair_notebook import source,replace_once

ROOT=Path(__file__).resolve().parents[1]


def build(experiment):
    base=build_cpu(experiment);raw=base.read_bytes();n=copy.deepcopy(json.loads(raw))
    previous=n['metadata']['aes_standalone']['code_sha256']
    name=NAMES[experiment].replace('_CPU','_GPU')
    for cell in n['cells']:
        s=source(cell).replace(previous,'__NOTEBOOK_CODE_SHA256__')
        s=s.replace(NAMES[experiment],name)
        if cell['cell_type']=='code':
            s=s.replace("'AES300_REQUIRE_GPU':'0','CUDA_VISIBLE_DEVICES':'',", "'AES300_REQUIRE_GPU':'1',")
            s=s.replace("('Perangkat','CPU — dipilih untuk seluruh model; tidak memakai kuota GPU')",
                        "('Perangkat','GPU Colab — pilih T4 GPU; CUDA diperiksa sebelum model dimuat')")
            s=s.replace("'device': 'cpu'", "'device': 'cuda'")
            s=s.replace("device='cpu',new_answer=", "device='cuda',new_answer=")
            s=s.replace("lesson_version='three_inputs_format_ablation_local_cpu_v1'",
                        "lesson_version='three_inputs_format_ablation_local_gpu_v1'")
            s=s.replace('# CPU dipilih eksplisit, tidak memerlukan GPU atau kuota T4.',
                        '# GPU wajib tersedia; training tidak beralih diam-diam ke CPU.')
            s=s.replace('REQUIRE_GPU=False','REQUIRE_GPU=True')
            s=s.replace('gpu=check_gpu(require_gpu=False)','gpu=check_gpu(require_gpu=True)')
            s=s.replace("print('Perangkat training: CPU | float32 tanpa AMP | thread:',torch.get_num_threads(),flush=True)",
                        "print('Perangkat training:',gpu['name'],'| CUDA aktif | AMP fp16/bf16 sesuai GPU',flush=True)")
            cell['execution_count']=None;cell['outputs']=[]
        else:
            s=s.replace('· CPU ·','· GPU ·')
            s=s.replace('CPU float32 dua thread.',
                        'GPU CUDA; AMP fp16 pada T4, bf16 bila didukung GPU. Thread CPU tetap dua.')
            s=s.replace('CPU penuh bisa lama;', 'GPU harus tersedia;')
            s=s.replace('**Web Colab biasa:** buka notebook lalu **Runtime → Run all**.',
                        '**Web Colab biasa:** pilih **Runtime → Change runtime type → T4 GPU**, lalu **Runtime → Run all**.')
        cell['source']=s
    codes=[source(c).removeprefix('%%aes\n') for c in n['cells'] if c['cell_type']=='code']
    for s in codes:ast.parse(s)
    fingerprint=hashlib.sha256('\n'.join(codes).encode()).hexdigest()
    for c in n['cells']:c['source']=source(c).replace('__NOTEBOOK_CODE_SHA256__',fingerprint)
    n['metadata']['aes_standalone']['code_sha256']=fingerprint
    n['metadata']['colab']['name']=name+'.ipynb';n['metadata']['accelerator']='GPU'
    n['metadata']['aes_experiment'].update(device='cuda',precision='AMP fp16/bf16',
        base_cpu_notebook=base.name,base_cpu_sha256=hashlib.sha256(raw).hexdigest())
    target=base.with_name(name+'.ipynb');target.write_text(json.dumps(n,ensure_ascii=False,indent=1)+'\n')
    (ROOT/'aes-learning/notebooks/experiments'/target.name).write_bytes(target.read_bytes())
    print(json.dumps({'experiment':experiment,'notebook':str(target.relative_to(ROOT)),
                      'code_sha256':fingerprint,'device':'cuda','drive_autosave':False}))
    return target


if __name__=='__main__':
    for experiment in NAMES:build(experiment)
