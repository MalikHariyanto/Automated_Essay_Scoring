"""Eksperimen CPU kandidat B: hanya LR encoder S3 berubah dari MLP acuan."""
import ast
import copy
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'notebooks/experiments/AES_300_S3_MLP.ipynb'
SETUP_SOURCE=ROOT/'notebooks/experiments/AES_300_S3_LoRA_MLP_GPU.ipynb'
DESTINATION=ROOT/'notebooks/experiments/AES_300_S3_MLP_LR5e6_CPU.ipynb'


def source(cell):
    return cell['source'] if isinstance(cell['source'],str) else ''.join(cell['source'])


def replace_once(text,old,new):
    if text.count(old)!=1:
        raise ValueError('Anchor harus unik: '+old[:100])
    return text.replace(old,new)


def definitions(text):
    text=text.removeprefix('%%aes\n')
    return {node.name:ast.get_source_segment(text,node) for node in ast.parse(text).body
            if isinstance(node,(ast.FunctionDef,ast.ClassDef))}


def build():
    original=BASE.read_bytes();setup_original=SETUP_SOURCE.read_bytes()
    notebook=copy.deepcopy(json.loads(original))
    old_hash=notebook['metadata']['aes_standalone']['code_sha256']
    setup_notebook=json.loads(setup_original)
    setup=next(source(c) for c in setup_notebook['cells'] if c['cell_type']=='code' and 'def prepare_environment(' in source(c))
    # Hanya ambil perbaikan installer/isolasi teks; tidak mengambil model LoRA.
    setup=replace_once(setup,'from peft import LoraConfig, get_peft_model\n','')
    setup=setup.replace('_text_lora','_text_s3_lr_cpu').replace('PEFT','pustaka AES')
    setup=setup.replace('Pilih runtime Colab GPU yang baru','Pilih runtime Colab yang baru')
    setup=setup.replace('Gunakan runtime GPU yang kompatibel','Gunakan runtime Colab yang kompatibel')
    backend=definitions((ROOT/'src/aes/teacher_score/lora_text_runtime.py').read_text())['configure_text_only_transformers']
    manual=definitions((ROOT/'src/aes/teacher_score/manual_checkpoints.py').read_text())
    for cell in notebook['cells']:
        text=source(cell)
        if cell['cell_type']=='code':
            if 'def prepare_environment(' in text:
                text=setup
            text=text.replace(old_hash,'__NOTEBOOK_CODE_SHA256__')
            text=text.replace("lesson_version='s3_mlp_account2_v1'","lesson_version='s3_mlp_lr5e6_cpu_v1'")
            if text.startswith('%%aes\nimport os, sys'):
                text=replace_once(text,'from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup',
                    backend+'\n\nconfigure_text_only_transformers()\nfrom transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup')
            if 'DEFAULTS = {' in text:
                text=replace_once(text,"'encoder_lr': 1e-5,", "'encoder_lr': 1e-5, 's3_encoder_lr': 5e-6,")
                text=replace_once(text,"'bge_batch': 4, 'device': 'auto'", "'bge_batch': 4, 'device': 'cpu'")
                text=replace_once(text,"('encoder_lr', 'projection_lr'", "('encoder_lr', 's3_encoder_lr', 'projection_lr'")
            if 'ROBERTA_CHECKPOINT=' in text:
                text=replace_once(text,'ENCODER_LR=1e-5',"""ENCODER_LR=1e-5  # S1 mengikuti acuan.
S3_ENCODER_LR=5e-6  # Kandidat B: hanya encoder RoBERTa S3 diperlambat.""")
                text=replace_once(text,'encoder_lr=ENCODER_LR,projection_lr=', 'encoder_lr=ENCODER_LR,s3_encoder_lr=S3_ENCODER_LR,projection_lr=')
                text=replace_once(text,"device='auto'","device='cpu'")
                text=text[:text.index('# Colab harus mendeteksi GPU')]+"""# CPU dipilih eksplisit meskipun runtime kebetulan menyediakan GPU.
REQUIRE_GPU=False
print('Perangkat aktif: CPU. Training memakai float32; GPU tidak dipakai.',flush=True)
show(pd.DataFrame([
    {'Skenario':'S1','LR Encoder RoBERTa':ENCODER_LR,'LR Proyeksi BGE':None,'LR Head':HEAD_LR},
    {'Skenario':'S2','LR Encoder RoBERTa':None,'LR Proyeksi BGE':None,'LR Head':HEAD_LR},
    {'Skenario':'S3','LR Encoder RoBERTa':S3_ENCODER_LR,'LR Proyeksi BGE':PROJECTION_LR,'LR Head':HEAD_LR}
]),'Learning rate dasar setiap skenario; encoder BGE tetap frozen')
"""
            if "PYTHON = os.environ.get('AES300_PYTHON'" in text:
                text=replace_once(text,"'AES300_REQUIRE_GPU':'1' if IN_COLAB else os.environ.get('AES300_REQUIRE_GPU','0')", "'AES300_REQUIRE_GPU':'0'")
                text=replace_once(text,"('Perangkat','GPU: '+ready['device'] if ready['cuda_available'] else 'CPU — GPU belum terdeteksi')", "('Perangkat','CPU — dipilih untuk seluruh eksperimen')")
            if 'def save_training_progress(' in text:
                parts=definitions(text)
                text='%%aes\n# Progres tiap epoch tetap disimpan; penghapusan checkpoint manual.\n\n'+'\n\n\n'.join(manual[name] if name=='save_training_progress' else value for name,value in parts.items() if name!='clear_training_progress')+'\n'
                text=replace_once(text,"rates = {'roberta': cfg['encoder_lr'],", "rates = {'roberta': cfg['s3_encoder_lr'] if model.scenario=='S3' else cfg['encoder_lr'],")
            if 'def train_all(' in text:
                text='\n'.join(line for line in text.splitlines() if not line.lstrip().startswith('clear_training_progress('))+'\n'
            if "settings={'epochs'" in text:
                text=replace_once(text,"'encoder_lr':'Learning rate encoder'", "'encoder_lr':'Learning rate encoder RoBERTa S1','s3_encoder_lr':'Learning rate encoder RoBERTa S3'")
            cell['outputs'],cell['execution_count']=[],None
        else:
            if text.startswith('# AES 300'):
                text='''# AES 300 · S3 MLP kandidat B · LR RoBERTa 5e-6 · CPU

Eksperimen terpisah dari MLP acuan, menggunakan jawaban siswa saja. Tidak memakai LoRA, konteks pair atau fitur linguistik tambahan. **Satu perubahan hyperparameter: LR encoder RoBERTa S3 dari 1e-5 menjadi 5e-6.** LR encoder S1 tetap 1e-5; proyeksi BGE dan head MLP tetap 1e-4. CPU dipilih eksplisit untuk semua forward/training/prediksi walaupun GPU tersedia.

S1: RoBERTa fine-tuned + linear 768→1. S2: BGE-M3 frozen + MLP. S3: RoBERTa fine-tuned → LayerNorm, BGE-M3 frozen → proyeksi 256 + LayerNorm, concat 1024 → MLP 256→GELU→dropout→1. Tidak membekukan RoBERTa S3. Arsitektur, fungsi loss dan metrik mengikuti acuan.

**Urutan:** setup/Drive → pilih fix_dataset_300.xlsx → audit/cleaning → split 80/20 → tokenisasi/vektor → model → 5-fold CV pada development → refit → test → prediksi → vektor sesudah training → lima Excel. Satu skor guru 0–100, maksimum 25 epoch per fold, patience 5, microbatch 1/batch efektif 8, warmup 10% dan scheduler linear tetap. RMSE validasi memilih epoch terbaik; test tidak dipakai memilih LR. Notebook tetap menjalankan tiga skenario agar laporan lengkap seperti acuan.

**Menjalankan di CPU:** Runtime → Change runtime type → Hardware accelerator: None, lalu jalankan dari atas. Tidak perlu T4. Training full fine-tuning CPU dapat jauh lebih lama daripada GPU; epoch/batch tidak dipangkas diam-diam. Setup memakai lingkungan privat teks, mempertahankan PyTorch runtime dan tidak memerlukan PEFT. Python 3.12/3.13 didukung sesuai versi pustaka acuan.

**Drive terpisah:** MyDrive/AES_300_S3_MLP_LR5e6_CPU_Privat. Progres disimpan setiap epoch. Tidak ada auto-delete atau Drive API cleanup; checkpoint lama dihapus manual. Pertahankan final.pt dan JSON hasil; untuk tahap aktif pertahankan dua snapshot yang dirujuk JSON progres. Jalankan notebook yang sama dari atas setelah runtime terputus untuk melanjutkan progres yang cocok. Checkpoint GPU/eksperimen lama tidak dipaksa dipakai ulang.

**Hasil:** lima Excel mengikuti acuan, termasuk token/ID, vektor jawaban/kata/kalimat, nilai guru/prediksi/pembulatan, riwayat training dan RMSE/Pearson/QWK/Gwet AC2. LR khusus S3 tampil di konfigurasi layar dan laporan pelatihan. Seluruh kode tersedia langsung di cell; tidak clone/import proyek GitHub. Dataset, output privat dan bobot tidak ditanam di notebook publik.

**Batas perbandingan:** acuan GPU memakai mixed precision ketika training, sedangkan versi CPU memakai float32. Perbedaan hasil antarperangkat tidak dapat seluruhnya dikaitkan dengan LR. Untuk mengisolasi pengaruh LR, konfirmasikan acuan 1e-5 dan kandidat 5e-6 pada perangkat/seed/split yang sama. Ini kandidat yang diuji, belum menjamin RMSE lebih kecil daripada acuan 6,0516 poin CV.
'''
            if text.startswith('## 3 '):
                text+='\n\nCPU dipilih melalui device=cpu; S3_ENCODER_LR hanya mengubah RoBERTa S3, bukan S1.\n'
            if text.startswith('## 11 '):
                text+='\n\noptimizer_groups memilih s3_encoder_lr hanya ketika scenario=S3. Proyeksi/head mengikuti LR acuan. Training CPU menggunakan float32 dan tidak mengaktifkan AMP.\n'
            if text.startswith('## 12 '):
                text+='\n\nSimpan setiap epoch tanpa penghapusan otomatis. CPU membutuhkan waktu lebih lama; checkpoint menyimpan optimizer, scheduler, riwayat dan keadaan acak untuk resume.\n'
        text=text.replace('AES_300_MLP_Privat','AES_300_S3_MLP_LR5e6_CPU_Privat')
        text=text.replace('/content/AES_300_MLP','/content/AES_300_S3_MLP_LR5e6_CPU')
        text=text.replace('outputs/AES_300_MLP','outputs/AES_300_S3_MLP_LR5e6_CPU')
        cell['source']=text;cell['metadata']={};cell.pop('attachments',None)
    normalized=[source(c).removeprefix('%%aes\n') for c in notebook['cells'] if c['cell_type']=='code']
    for code in normalized:ast.parse(code)
    fingerprint=hashlib.sha256('\n'.join(normalized).encode()).hexdigest()
    for c in notebook['cells']:c['source']=source(c).replace('__NOTEBOOK_CODE_SHA256__',fingerprint)
    notebook['metadata']['aes_standalone']['code_sha256']=fingerprint
    notebook['metadata']['colab']['name']=DESTINATION.name
    notebook['metadata'].pop('accelerator',None)
    notebook['metadata']['colab'].pop('gpuType',None)
    notebook['metadata']['aes_experiment']={'name':'S3_MLP_LR5E6_CPU','base_sha256':hashlib.sha256(original).hexdigest(),
        'setup_source_sha256':hashlib.sha256(setup_original).hexdigest(),'device':'cpu','input':'answer_only',
        's3_encoder_lr':5e-6,'s1_encoder_lr':1e-5,'projection_lr':1e-4,'head_lr':1e-4,
        'automatic_checkpoint_deletion':False,'drive_root':'AES_300_S3_MLP_LR5e6_CPU_Privat'}
    DESTINATION.parent.mkdir(parents=True,exist_ok=True)
    DESTINATION.write_text(json.dumps(notebook,ensure_ascii=False,indent=1)+'\n')
    mirror=ROOT/'aes-learning/notebooks/experiments'/DESTINATION.name
    mirror.parent.mkdir(parents=True,exist_ok=True);mirror.write_bytes(DESTINATION.read_bytes())
    assert BASE.read_bytes()==original and SETUP_SOURCE.read_bytes()==setup_original
    print(json.dumps({'notebook':str(DESTINATION.relative_to(ROOT)),'code_sha256':fingerprint,'base_unchanged':True}))
    return DESTINATION


if __name__=='__main__':
    build()
