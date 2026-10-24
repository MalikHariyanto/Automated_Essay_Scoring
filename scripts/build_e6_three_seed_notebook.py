"""E6 GPU tiga seed: seluruh S1/S2/S3 dilatih baru, tanpa mengubah notebook acuan."""
import ast
import copy
import hashlib
import json
from pathlib import Path
from build_s3_lr_cpu_notebook import replace_once,source

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'notebooks/experiments/AES_300_S3_MLP_LR5e6_CPU.ipynb'
DESTINATION=ROOT/'notebooks/experiments/AES_300_E6_3Seed_GPU.ipynb'


def build():
    original=BASE.read_bytes();n=copy.deepcopy(json.loads(original));old_hash=n['metadata']['aes_standalone']['code_sha256']
    for cell in n['cells']:
        s=source(cell).replace('AES_300_S3_MLP_LR5e6_CPU','AES_300_E6_3Seed_GPU').replace(old_hash,'__NOTEBOOK_CODE_SHA256__')
        if cell['cell_type']=='code':cell['outputs']=[];cell['execution_count']=None
        cell['source']=s;cell['metadata']={};cell.pop('attachments',None)
    n['cells'][0]['source']='''# E6 · Tiga seed · S1, S2, S3 dilatih ulang · GPU

Tiga seed training **42, 123, 2026**; pembagian 80/20 dan lima fold dikunci dengan split_seed=2026. **45 pelatihan fold + 9 refit**, seluruhnya model baru dari pretrained/head baru; tidak mengambil model S1/S2/S3 eksperimen lama. S1 full fine-tuning RoBERTa + linear; S2 BGE frozen + MLP; S3 RoBERTa full fine-tuning + BGE frozen + fusion MLP. Jawaban saja, tanpa LoRA atau konteks pair.

**LR tetap E6:** encoder RoBERTa S1=1e-5; encoder RoBERTa S3=5e-6; proyeksi/head=1e-4. Maksimal 25 epoch per fold, patience 5, microbatch 1/batch efektif 8, warmup 10% dan scheduler linear. Seed run42 memakai seed43–47 pada fold1–5, seperti aturan E6; refit memakai42. Aturan yang sama berlaku untuk123 dan2026.

**GPU wajib:** Runtime → Change runtime type → T4 GPU. Jalankan dari atas. Folder baru **MyDrive/AES_300_E6_3Seed_GPU_Privat**; hasil lama tidak perlu dihapus untuk mulai baru. RUN_TAG=ulangan_01 menandai eksperimen ini. Untuk benar-benar mengulang dari nol setelah selesai, ganti RUN_TAG; jangan menghapus checkpoint acuan yang masih dibutuhkan. Restart sesi dengan RUN_TAG dan pengaturan sama akan melanjutkan progres eksperimen ini.

**Penyimpanan:** autosave dua slot A/B per fold/refit. Penulisan menimpa slot yang lebih lama secara langsung; slot terbaru lain tetap cadangan. Tidak membuat satu file besar untuk setiap epoch; tidak menjalankan auto-delete, Drive API, atau pembersihan Sampah. Model final, JSON, dan sumber lama tidak dihapus. Semua fold yang selesai tetap memiliki hasil dan slot cadangannya, jadi ruang untuk tiga seed masih diperlukan.

**Audit hasil berubah:** hash kode fungsi aktual, hash arsitektur, hash bobot awal head/proyeksi, sampel bobot encoder, seed fold, GPU, versi pustaka, presisi dan dua forward eval dicatat. Arsitektur boleh tetap sama sementara inisialisasi head, urutan batch dan dropout berubah karena seed. CUDA/AMP/versi pustaka dapat memberi variasi tambahan. Deterministic algorithms diaktifkan dengan warn_only, TF32 dimatikan, workspace cuBLAS dikunci; ini membantu pengulangan, bukan jaminan identik bit demi bit lintas perangkat/versi. Dua forward eval menguji pembacaan model yang sama, bukan mengulang seluruh training.

**Hasil akhir:** satu Ringkasan_3_Seed.xlsx (CV/test mean dan SD antar-seed,45 fold, guru/prediksi9model, audit) + lima Excel per seed. Vektor pretrained/tokenisasi/BGE dihitung bersama sekali; tahap inspeksi sebelum training bukan head awal khusus tiap seed. Vektor sesudah training dan prediksi berasal dari model final masing-masing seed. Semua3seed dilaporkan; tidak memilih seed terbaik dari test, tidak menggabungkan prediksi tiga seed sebagai ensemble. Test yang sudah dilihat sebelumnya tetap eksploratif.
'''
    for i in (10,12):n['cells'][i]['source']=source(n['cells'][i]).replace("'device': 'cpu'","'device': 'cuda'").replace("device='cpu'","device='cuda'")
    s=source(n['cells'][7]).replace("'AES300_REQUIRE_GPU':'0'","'AES300_REQUIRE_GPU':'1', 'CUBLAS_WORKSPACE_CONFIG':':4096:8', 'PYTHONHASHSEED':'0'")
    s=s.replace("('Perangkat','CPU — dipilih untuk seluruh eksperimen')","('Perangkat','GPU: '+ready['device'] if ready['cuda_available'] else 'GPU belum terdeteksi')")
    n['cells'][7]['source']=s
    s=source(n['cells'][12]).replace("lesson_version='s3_mlp_lr5e6_cpu_v1'","lesson_version='e6_three_seed_gpu_v1',seed=TRAINING_SEEDS[0],split_seed=SPLIT_SEED,training_seeds=TRAINING_SEEDS,run_tag=RUN_TAG,checkpoint_strategy='two_slots'")
    s=replace_once(s,"NEW_ANSWER=''", """# Ganti RUN_TAG bila ingin eksperimen baru; tetap sama untuk resume sesi terputus.
TRAINING_SEEDS=[42,123,2026]
SPLIT_SEED=2026  # Tidak ikut berubah ketika seed training berganti.
RUN_TAG='ulangan_01'
NEW_ANSWER=''""")
    s=s[:s.index('# CPU dipilih eksplisit')]+'''# GPU wajib agar eksperimen tidak diam-diam berjalan di CPU.
REQUIRE_GPU=True
gpu=check_gpu(require_gpu=RUN_MODELS and REQUIRE_GPU)
print('GPU aktif:',gpu['name'],'| tiga seed:',TRAINING_SEEDS,'| split tetap:',SPLIT_SEED,flush=True)
show(pd.DataFrame([
 {'Skenario':'S1','Encoder':'RoBERTa full fine-tuning','LR encoder':'1e-5','LR head':'1e-4'},
 {'Skenario':'S2','Encoder':'BGE-M3 frozen','LR encoder':'Tidak dilatih','LR head':'1e-4'},
 {'Skenario':'S3','Encoder':'RoBERTa full fine-tuning + BGE frozen','LR encoder':'5e-6','LR head/proyeksi':'1e-4'}
]),'Learning rate E6 tetap; semua model baru pada tiap seed')
'''
    n['cells'][12]['source']=s
    s=source(n['cells'][31])+'''\n# Menahan variasi perhitungan perangkat tanpa mengganti arsitektur model.
_original_seed_everything=seed_everything
def seed_everything(seed):
    _original_seed_everything(seed)
    torch.use_deterministic_algorithms(True,warn_only=True)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
'''
    # Hindari fungsi pembungkus tersembunyi saat mengekstrak runtime untuk pengujian.
    s=s[:s.index('# Menahan variasi')]
    s=replace_once(s,"    torch.backends.cudnn.benchmark = False", """    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True,warn_only=True)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False""")
    n['cells'][31]['source']=s
    s=source(n['cells'][36]);a=s.index('def save_training_progress(');b=s.index('\ndef load_training_progress',a)
    s=s[:a]+'''def save_training_progress(path,payload):
    """Dua nama tetap, tidak menghapus file Drive atau membuat file per epoch."""
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    previous=json.loads(path.read_text())['snapshots'] if path.exists() else []
    slot='A' if payload['epoch']%2 else 'B'
    target=path.with_name(f'{path.stem}_slot_{slot}.pt')
    # Tulis hanya slot yang lebih lama. Bila terputus, slot terbaru lain masih utuh.
    torch.save(payload,target)
    checksum=digest(target)
    snapshots=[{'file':target.name,'sha256':checksum}]+[r for r in previous if r['file']!=target.name][:1]
    dump(path,{'scope':SCOPE,'snapshots':snapshots})

'''+s[b:]
    s=replace_once(s,"    groups = optimizer_groups(model, cfg)", """    initial_audit=initialization_audit(model,engine,training,cfg,seed)
    if checkpoint is not None:
        audit_path=Path(checkpoint).with_name(Path(checkpoint).stem+'_audit.json')
        dump(audit_path,initial_audit)
    groups = optimizer_groups(model, cfg)""")
    s=replace_once(s,'    return model, engine, info',"    info['reproducibility']=initial_audit\n    return model, engine, info")
    n['cells'][36]['source']=s
    helpers=(ROOT/'scripts/multi_seed_helpers.py').read_text();n['cells'][38]['source']=source(n['cells'][38])+'\n'+helpers
    s=source(n['cells'][39]);s=s[:s.index('\nif RUN_TRAINING:')]+'''
# Seluruh seed dijalankan tanpa mengambil model acuan lama.
seed_runs=[]
if RUN_TRAINING:
    seed_runs=run_three_seeds(bundle,manifest,bge,cfg,output,journal)
    seed_summary=summarize_seed_runs(seed_runs,bundle)
    show(seed_summary['CV per Seed'],'Rata-rata lima fold pada tiap seed',rows=9)
    show(seed_summary['Rerata CV 3 Seed'],'Rerata dan SD antar tiga seed — validasi',rows=3)
    show(seed_summary['Uji per Seed'],'Evaluasi data uji tiap seed; semua hasil dilaporkan',rows=9)
    show(seed_summary['Rerata Uji 3 Seed'],'Rerata dan SD antar tiga seed — data uji',rows=3)
'''
    n['cells'][39]['source']=s
    for index,condition,title in [(41,'if RUN_TRAINING:','export_seed_example'),(43,'if RUN_TRAINING and RUN_FINAL_FEATURES:','export_seed_vectors')]:
        s=source(n['cells'][index]);a=s.index('\n'+condition);body=s[a+len('\n'+condition)+1:]
        # Tubuh if lama sudah menjorok empat spasi, cocok untuk fungsi baru.
        s=s[:a]+f'\ndef {title}(run):\n    cfg,output,journal=run["cfg"],run["output"],run["journal"]\n'+body
        s+='\nfor run in seed_runs:\n'
        s+=f'    {title}(run)\n' if index==41 else f'    if RUN_FINAL_FEATURES:{title}(run)\n'
        n['cells'][index]['source']=s
    n['cells'][46]['source']='''# Menyusun satu paket seluruh seed, tanpa melatih atau menguji ulang model.
aes_cell('',"""
if not seed_runs:raise RuntimeError('Jalankan bagian training tiga seed terlebih dahulu.')
summary_folder=output/'hasil_3_seed';summary_folder.mkdir(parents=True,exist_ok=True)
summary_frames=summarize_seed_runs(seed_runs,bundle)
summary_frames={'Panduan':pd.DataFrame([
 {'Bagian':'Rerata CV 3 Seed','Cara baca':'Rerata dan SD dari tiga rata-rata CV; setiap seed memiliki lima fold yang sama.'},
 {'Bagian':'Rerata Uji 3 Seed','Cara baca':'Rerata dan SD dari tiga model final per skenario, bukan ensemble.'},
 {'Bagian':'Audit Model dan Seed','Cara baca':'Hash arsitektur sama = struktur sama. Hash awal head dapat berbeda karena seed.'},
 {'Bagian':'Guru dan Prediksi','Cara baca':'Prediksi mentah; RMSE/Pearson memakai ini. Pembulatan hanya untuk QWK/AC2.'},
 {'Bagian':'Pengulangan','Cara baca':'Seluruh seed dilaporkan, tidak memilih pemenang dari data uji.'},
 {'Bagian':'Penyimpanan','Cara baca':'Dua slot per fold/refit, tanpa auto-delete Drive. Folder eksperimen lama tidak disentuh.'}
]),**summary_frames}
summary_path=summary_folder/'Ringkasan_3_Seed.xlsx'
write_excel(summary_path,summary_frames,{'scope':SCOPE,'stage':'Tiga seed, seluruh skenario dilatih baru','split_sha256':manifest['split_sha256']})
exports=[summary_path]
for run in seed_runs:
    result=build_reports(run['journal'].folder,summary_folder/f'seed_{run["seed"]}',verbose=False)
    exports.extend(Path(p) for p in result['files'])
archive=summary_folder/'Hasil_E6_3_Seed_GPU.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as package:
    for path in exports:package.write(path,arcname=str(path.relative_to(summary_folder)))
dump(ROOT/'outputs/dataset300_download.json',{'archive':str(archive),'files':[str(p.relative_to(summary_folder)) for p in exports]})
""")
download=json.loads((ROOT/'outputs/dataset300_download.json').read_text())
print('Selesai: Ringkasan_3_Seed.xlsx dan lima Excel untuk masing-masing seed.')
print('Folder hasil:',Path(download['archive']).parent)
print('Mulai dari Ringkasan_3_Seed.xlsx → Rerata CV 3 Seed dan Audit Model dan Seed.')
if IN_COLAB:
    from google.colab import files
    files.download(download['archive'])
else:print('Paket:',download['archive'])
'''
    for cell in n['cells']:
        if cell['cell_type']=='markdown' and not source(cell).startswith('# E6'):
            s=source(cell)
            s=s.replace('CPU dipilih melalui device=cpu; S3_ENCODER_LR hanya mengubah RoBERTa S3, bukan S1.','GPU wajib; semua skenario dilatih untuk tiga seed. LR E6 tetap.')
            s=s.replace('Training CPU menggunakan float32 dan tidak mengaktifkan AMP.','Training GPU memakai AMP sesuai dukungan perangkat; presisi dicatat pada audit.')
            s=s.replace('Simpan setiap epoch tanpa penghapusan otomatis. CPU membutuhkan waktu lebih lama; checkpoint menyimpan optimizer, scheduler, riwayat dan keadaan acak untuk resume.','Autosave dua slot setiap fold/refit; tidak ada auto-delete/Drive API. Progres dapat dilanjutkan dengan RUN_TAG dan lingkungan sama.')
            if s.startswith('## 12 '):s+='\n\nBagian ini menjalankan seluruh tiga seed secara berurutan. S1/S2 tidak dipakai ulang dari folder eksperimen lain.\n'
            if s.startswith('## 15 '):s+='\n\nPaket akhir mencakup Ringkasan_3_Seed.xlsx + tiga folder berisi lima laporan Excel masing-masing.\n'
            cell['source']=s
    codes=[source(c).removeprefix('%%aes\n') for c in n['cells'] if c['cell_type']=='code']
    for s in codes:ast.parse(s)
    fingerprint=hashlib.sha256('\n'.join(codes).encode()).hexdigest()
    for c in n['cells']:c['source']=source(c).replace('__NOTEBOOK_CODE_SHA256__',fingerprint)
    n['metadata']['aes_standalone']['code_sha256']=fingerprint;n['metadata']['accelerator']='GPU';n['metadata']['colab']['name']=DESTINATION.name
    n['metadata']['aes_experiment']={'name':'E6_3SEED_GPU_ALL_SCENARIOS','base_sha256':hashlib.sha256(original).hexdigest(),
        'training_seeds':[42,123,2026],'split_seed':2026,'device':'cuda','reused_controls':[],
        's1_encoder_lr':1e-5,'s3_encoder_lr':5e-6,'head_lr':1e-4,'projection_lr':1e-4,
        'checkpoint_strategy':'two_slots_in_place','automatic_checkpoint_deletion':False,
        'drive_root':'AES_300_E6_3Seed_GPU_Privat'}
    DESTINATION.write_text(json.dumps(n,ensure_ascii=False,indent=1)+'\n')
    (ROOT/'aes-learning/notebooks/experiments'/DESTINATION.name).write_bytes(DESTINATION.read_bytes())
    assert BASE.read_bytes()==original
    print(json.dumps({'notebook':str(DESTINATION.relative_to(ROOT)),'code_sha256':fingerprint,'base_unchanged':True}))
    return DESTINATION


if __name__=='__main__':build()
