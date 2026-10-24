"""E8 GPU: konteks pair E4 + LR encoder RoBERTa S3 5e-6; E7 tetap terpisah."""
import ast
import copy
import hashlib
import json
from pathlib import Path
from build_context_pair_notebook import source, replace_once, definitions

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'notebooks/experiments/AES_300_S3_ContextPair_MLP.ipynb'
SETUP=ROOT/'notebooks/experiments/AES_300_E6_S3_LR5e6_GPU_Reuse.ipynb'
SLOTS=ROOT/'notebooks/experiments/AES_300_E6_3Seed_GPU.ipynb'
DESTINATION=BASE.with_name('AES_300_E8_ContextPair_LR5e6_GPU.ipynb')


def build():
    original=BASE.read_bytes();n=copy.deepcopy(json.loads(original))
    setup=json.loads(SETUP.read_bytes());slots=json.loads(SLOTS.read_bytes())
    old_hash=n['metadata']['aes_standalone']['code_sha256']
    # Perbaikan lingkungan teks saja; tidak mengambil LoRA atau reuse kontrol E6.
    for index in (5,9):n['cells'][index]['source']=source(setup['cells'][index])
    for cell in n['cells']:
        s=source(cell).replace(old_hash,'__NOTEBOOK_CODE_SHA256__')
        # Worker impor membawa identitas sumber setup; kunci ulang ke kode E8, bukan E6.
        s=s.replace(setup['metadata']['aes_standalone']['code_sha256'],'__NOTEBOOK_CODE_SHA256__')
        s=s.replace('AES_300_S3_ContextPair_MLP','AES_300_E8_ContextPair_LR5e6_GPU')
        s=s.replace("lesson_version='s3_context_pair_mlp_v1'","lesson_version='e8_context_pair_lr5e6_gpu_v1'")
        if cell['cell_type']=='code':
            cell['outputs']=[];cell['execution_count']=None
            if 'DEFAULTS = {' in s:
                s=replace_once(s,"'encoder_lr': 1e-5,","'encoder_lr': 1e-5, 's3_encoder_lr': 5e-6,")
                s=replace_once(s,"('encoder_lr', 'projection_lr'","('encoder_lr', 's3_encoder_lr', 'projection_lr'")
            if 'ROBERTA_CHECKPOINT=' in s:
                s=replace_once(s,'ENCODER_LR=1e-5',"ENCODER_LR=1e-5  # S1 tetap mengikuti pengaturan sebelumnya.\nS3_ENCODER_LR=5e-6  # E8: langkah encoder S3 diperkecil, input tetap konteks pair.")
                s=replace_once(s,'encoder_lr=ENCODER_LR,projection_lr=','encoder_lr=ENCODER_LR,s3_encoder_lr=S3_ENCODER_LR,projection_lr=')
                s+='''
# BGE tidak mendapat learning rate karena encodernya frozen.
show(pd.DataFrame([
 {'Skenario':'S1','Input RoBERTa':'Jawaban saja','LR encoder':ENCODER_LR,'LR head':HEAD_LR},
 {'Skenario':'S2','Input RoBERTa':'Tidak dipakai; BGE jawaban saja','LR encoder':None,'LR head':HEAD_LR},
 {'Skenario':'S3','Input RoBERTa':'Soal/referensi + jawaban','LR encoder':S3_ENCODER_LR,'LR head':HEAD_LR}
]),'E8: learning rate dasar; proyeksi S3 tetap 1e-4')
'''
            if 'def optimizer_groups(' in s:
                s=replace_once(s,"rates = {'roberta': cfg['encoder_lr'],","rates = {'roberta': cfg['s3_encoder_lr'] if model.scenario=='S3' else cfg['encoder_lr'],")
                old=definitions(s)['save_training_progress']
                new=definitions(source(slots['cells'][36]))['save_training_progress']
                s=replace_once(s,old,new)
            if "'encoder_lr':'Learning rate encoder'" in s:
                s=replace_once(s,"'encoder_lr':'Learning rate encoder'","'encoder_lr':'Learning rate encoder RoBERTa S1','s3_encoder_lr':'Learning rate encoder RoBERTa S3'")
            s=s.replace('Hasil_AES_Ringkas.zip','Hasil_AES_E8_ContextPair_LR5e6_GPU.zip')
        cell['source']=s;cell['metadata']={};cell.pop('attachments',None)
    n['cells'][0]['source']='''# AES 300 · Eksperimen 8 · Konteks pair + MLP + LR S3 5e-6 · GPU

E8 menguji hipotesis: apakah LR encoder RoBERTa S3 yang lebih kecil membantu fusion MLP dengan konteks pair? Pembanding utama adalah konteks pair sebelumnya (E4, LR1e-5); E6 memakai jawaban saja dengan LR5e-6. E7 merupakan eksperimen tiga seed yang terpisah. E8 belum mempunyai hasil dan tidak menjamin RMSE turun.

**GPU wajib:** Runtime → Change runtime type → T4 GPU, lalu jalankan dari atas. Semua kode sudah tersedia di notebook; tidak clone proyek. Unggah fix_dataset_300.xlsx jika diminta. Dataset satu skor guru0–100; tidak membuat empat skor rubrik dari satu nilai.

**Model:** S1 jawaban → RoBERTa full fine-tuning → dropout → linear768→1. S2 jawaban → BGE-M3 frozen → MLP1024→256→1. S3 pasangan soal/referensi + jawaban → RoBERTa full fine-tuning → LayerNorm768; BGE tetap membaca jawaban saja → proyeksi256 + LayerNorm; concat1024 → MLP256 → GELU → dropout → linear1. Tidak memakai LoRA dan tidak membekukan RoBERTa S3.

**Learning rate:** encoder S1=1e-5; encoder S3=5e-6; proyeksi/head=1e-4. Anggaran konteks192token termasuk prefix, soal maksimal64token; window RoBERTa512 termasuk4special pasangan, overlap128. Seluruh token jawaban dicakup lewat window. Token konteks yang dipotong dilaporkan. Soal/referensi yang dipakai harus tersedia pada waktu prediksi; skor guru dan email tidak masuk encoder.

**Latihan:** seed42, split_seed2026, development/test80/20, lima fold pada development, maksimal25epoch/fold, patience5, microbatch1 dan batch efektif8, warmup10% dan scheduler linear. S1/S2/S3 dilatih baru: 15pelatihan fold +3refit, satu seed. Tidak membaca hasil eksperimen lama. Refit memakai ceil median epoch terpilih CV; test tidak memilih LR/epoch. Setelah selesai, notebook yang sama melanjutkan atau memakai hasil E8 yang cocok.

**Drive baru:** MyDrive/AES_300_E8_ContextPair_LR5e6_GPU_Privat. Autosave dua slot A/B per fold/refit; slot lama ditimpa setelah slot terbaru lain tersedia sebagai cadangan. Tidak memakai Drive API, auto-delete, atau pembersihan Sampah. Model final dan JSON tetap disimpan. Slot seluruh fold tetap memerlukan ruang Drive; mekanisme ini tidak menghapus checkpoint eksperimen lama atau versi file di Drive. Pertahankan notebook/pengaturan yang sama untuk resume setelah sesi terputus.

**Hasil:** token pasangan dan jawaban, vektor awal/akhir serta kata/kalimat, riwayat latihan, nilai guru/prediksi/pembulatan, dan RMSE/Pearson/QWK/GwetAC2. Cell terakhir mengunduh lima Excel dalam Hasil_AES_E8_ContextPair_LR5e6_GPU.zip. Untuk jawaban baru, isi NEW_QUESTION, NEW_REFERENCE dan NEW_ANSWER.

**Batas kesimpulan:** arsitektur dan anggaran token tetap mengikuti konteks pair E4; hanya LR encoder S3 yang diubah sebagai kandidat metode. Pengaturan lingkungan/penyimpanan diperbaiki agar notebook dapat dijalankan. GPU/versi/seed yang berbeda tetap dapat memengaruhi hasil; bandingkan CV terlebih dahulu dan ulangi pada lingkungan sama untuk mengisolasi efek LR. Data uji telah dilihat selama pengembangan, sehingga hasil lanjutan bersifat eksploratif.
'''
    for cell in n['cells'][1:]:
        if cell['cell_type']=='markdown':
            s=source(cell).replace('Semua checkpoint tetap disimpan','Dua slot checkpoint per fold/refit disimpan')
            s=s.replace('Encoder RoBERTa memakai 1e-5','Encoder RoBERTa S1 memakai1e-5 dan S3 memakai5e-6')
            s=s.replace('Learning rate kandidat A memakai 0,00001 pada encoder RoBERTa dan 0,0001 pada proyeksi/head.',
                        'LR encoder RoBERTa S1=1e-5 dan S3=5e-6; proyeksi/head tetap1e-4.')
            s=s.replace('**Learning rate kandidat A:** RoBERTa 0,00001 (`1e-5`), head dan proyeksi 0,0001 (`1e-4`).',
                        '**Learning rate E8:** RoBERTa S1=1e-5, RoBERTa S3=5e-6; head dan proyeksi=1e-4.')
            s=s.replace('Hanya head S3 yang berubah; S1/S2, learning rate kandidat A, 25 epoch maksimal, early stopping lima epoch, split 80/20, dan 5-fold CV tetap.',
                        'Head MLP mengikuti eksperimen konteks pair sebelumnya. E8 mengubah LR encoder S3 menjadi5e-6; S1/S2, 25epoch maksimal, patience5, split80/20 dan5fold tetap.')
            s=s.replace('Checkpoint setiap epoch dan fold tetap disimpan untuk dibersihkan manual.',
                        'Autosave memakai dua slot tetap A/B per fold/refit; tidak ada penghapusan file Drive otomatis.')
            s+='\n\nE8 mempertahankan alur konteks pair; LR encoder S3=5e-6. Autosave memakai dua slot tanpa auto-delete.\n' if s.startswith('## 11 ') else ''
            cell['source']=s
    codes=[source(c).removeprefix('%%aes\n') for c in n['cells'] if c['cell_type']=='code']
    for code in codes:ast.parse(code)
    fingerprint=hashlib.sha256('\n'.join(codes).encode()).hexdigest()
    for cell in n['cells']:cell['source']=source(cell).replace('__NOTEBOOK_CODE_SHA256__',fingerprint)
    n['metadata']['aes_standalone']['code_sha256']=fingerprint
    n['metadata']['accelerator']='GPU';n['metadata']['colab']['name']=DESTINATION.name
    n['metadata']['aes_experiment'].update(name='E8_CONTEXT_PAIR_LR5E6_GPU',base_sha256=hashlib.sha256(original).hexdigest(),
        setup_source_sha256=hashlib.sha256(SETUP.read_bytes()).hexdigest(),s1_encoder_lr=1e-5,s3_encoder_lr=5e-6,
        head_lr=1e-4,projection_lr=1e-4,training_seed=42,split_seed=2026,reused_controls=[],
        checkpoint_strategy='two_slots_in_place',drive_root='AES_300_E8_ContextPair_LR5e6_GPU_Privat')
    DESTINATION.write_text(json.dumps(n,ensure_ascii=False,indent=1)+'\n')
    (ROOT/'aes-learning/notebooks/experiments'/DESTINATION.name).write_bytes(DESTINATION.read_bytes())
    assert BASE.read_bytes()==original
    print(json.dumps({'notebook':str(DESTINATION.relative_to(ROOT)),'code_sha256':fingerprint,'context_base_unchanged':True}))
    return DESTINATION


if __name__=='__main__':build()
