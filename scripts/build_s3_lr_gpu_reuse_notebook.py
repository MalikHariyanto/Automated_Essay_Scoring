"""E6 GPU: S3 kandidat B; reuse kontrol S1/S2 selesai tanpa menghapus run lama."""
import ast
import copy
import hashlib
import json
from pathlib import Path

from build_s3_lr_cpu_notebook import definitions,replace_once,source

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'notebooks/experiments/AES_300_S3_MLP_LR5e6_CPU.ipynb'
DESTINATION=ROOT/'notebooks/experiments/AES_300_E6_S3_LR5e6_GPU_Reuse.ipynb'


def build():
    original=BASE.read_bytes();notebook=copy.deepcopy(json.loads(original))
    old_hash=notebook['metadata']['aes_standalone']['code_sha256']
    helper_source=(ROOT/'src/aes/teacher_score/reuse_controls.py').read_text()
    helpers=definitions(helper_source)
    assignment=next(n for n in ast.parse(helper_source).body if isinstance(n,ast.Assign)
                    and any(isinstance(t,ast.Name) and t.id=='REUSE_SOURCE_PROFILES' for t in n.targets))
    profile=ast.get_source_segment(helper_source,assignment)
    for cell in notebook['cells']:
        text=source(cell)
        text=text.replace('AES_300_S3_MLP_LR5e6_CPU_Privat','AES_300_E6_S3_LR5e6_GPU_Reuse_Privat')
        text=text.replace('/content/AES_300_S3_MLP_LR5e6_CPU','/content/AES_300_E6_S3_LR5e6_GPU_Reuse')
        text=text.replace('outputs/AES_300_S3_MLP_LR5e6_CPU','outputs/AES_300_E6_S3_LR5e6_GPU_Reuse')
        if cell['cell_type']=='code':
            text=text.replace(old_hash,'__NOTEBOOK_CODE_SHA256__')
            if 'def prepare_environment(' in text:
                text=text.replace('_text_s3_lr_cpu','_text_e6_gpu_reuse')
            text=text.replace("lesson_version='s3_mlp_lr5e6_cpu_v1'","lesson_version='e6_s3_lr5e6_gpu_reuse_v1'")
            if 'DEFAULTS = {' in text:
                text=replace_once(text,"'device': 'cpu'","'device': 'cuda', 'reuse_run': ''")
            if 'ROBERTA_CHECKPOINT=' in text:
                text=replace_once(text,"NEW_ANSWER=''", """# Kosong: cari otomatis sumber kompatibel, prioritaskan folder MLP acuan lalu LoRA.
# Bila folder Drive berbeda, isi PATH folder hash yang berisi S1, S2 dan final_report.json.
REUSE_RUN=''
NEW_ANSWER=''""")
                text=replace_once(text,"device='cpu',new_answer=NEW_ANSWER", "device='cuda',reuse_run=REUSE_RUN,new_answer=NEW_ANSWER")
                start=text.index('# CPU dipilih eksplisit')
                text=text[:start]+"""REQUIRE_GPU=True
gpu=check_gpu(require_gpu=RUN_MODELS and REQUIRE_GPU)
print('Perangkat eksperimen S3:',gpu['name'],'— CUDA, hanya S3 dilatih baru.',flush=True)
show(pd.DataFrame([
    {'Skenario':'S1','Pelaksanaan':'Pakai hasil selesai yang diverifikasi','LR Encoder':ENCODER_LR,'LR Head':HEAD_LR},
    {'Skenario':'S2','Pelaksanaan':'Pakai hasil selesai yang diverifikasi','LR Encoder':None,'LR Head':HEAD_LR},
    {'Skenario':'S3','Pelaksanaan':'Latihan baru di GPU','LR Encoder':S3_ENCODER_LR,'LR Head':HEAD_LR}
]),'Eksperimen 6: reuse kontrol; kandidat B hanya mengubah LR RoBERTa S3')
"""
            if "PYTHON = os.environ.get('AES300_PYTHON'" in text:
                text=replace_once(text,"('Perangkat','CPU — dipilih untuk seluruh eksperimen')", "('Perangkat','GPU: '+ready['device'] if ready['cuda_available'] else 'GPU belum terdeteksi')")
                text=replace_once(text,"'AES300_REQUIRE_GPU':'0'","'AES300_REQUIRE_GPU':'1'")
            if 'def train_all(' in text:
                text=replace_once(text,'def train_all(',profile+'\n\n'+'\n\n\n'.join(helpers.values())+'\n\ndef train_all(')
                text=replace_once(text,"    final_path = output/'final_report.json'", "    reused=prepare_reused_controls(bundle,manifest,cfg,output)\n    final_path = output/'final_report.json'")
                text=replace_once(text,"            if complete.exists():", """            if scenario in ('S1','S2') and not complete.exists():
                if selected_epochs[scenario]!=reused['selected_epochs'][scenario]:
                    raise ValueError('Pilihan epoch kontrol berubah dari sumber terkunci.')
                dump(complete,{'selection':selection,'source':bundle.source_sha256,'sha256':digest(path),
                               'reused_from':reused['source_run'],'trained_in_this_run':False})
            if complete.exists():""")
                text=replace_once(text,"    dump(final_path, report)", """    report['reused_control_scenarios']=['S1','S2']
    report['control_source_run']=reused['source_run']
    report['test_evaluations_this_run']={'S1':0,'S2':0,'S3':1}
    dump(final_path, report)""")
            if 'report=train_all(' in text:
                text=replace_once(text,'    export_completed_training(report,bundle,cfg,output,journal)', """    reuse=json.loads((output/'s1_s2_reuse.json').read_text())
    origin=pd.DataFrame([
        {'Skenario':'S1','Asal':'Hasil run lama, tidak dilatih/diuji ulang','Folder Sumber':reuse['source_run']},
        {'Skenario':'S2','Asal':'Hasil run lama, tidak dilatih/diuji ulang','Folder Sumber':reuse['source_run']},
        {'Skenario':'S3','Asal':'Kandidat B, dilatih dan diuji pada run ini','Folder Sumber':str(output)}])
    journal.save('08b_asal_model',{'Asal':origin},'Asal checkpoint dan hasil evaluasi setiap skenario')
    show(origin,'Asal model S1/S2/S3')
    export_completed_training(report,bundle,cfg,output,journal)""")
            if 'def evaluation_report(' in text:
                text=replace_once(text,"    settings={'epochs'", """    if source.has('08b_asal_model.xlsx','Asal'):
        sheets['Asal Model']=source.copy('08b_asal_model.xlsx','Asal')
    settings={'epochs'""")
            cell['outputs'],cell['execution_count']=[],None
        else:
            if text.startswith('# AES 300'):
                text='''# AES 300 · Eksperimen 6 · S3 LR 5e-6 · GPU · Reuse S1/S2

**Tidak perlu menghapus file apa pun dari eksperimen lama.** S1/S2 memakai hasil selesai yang diverifikasi dari MLP acuan atau run LoRA; hanya S3 dilatih baru dari pretrained dengan head MLP acuan, LR encoder 5e-6, proyeksi/head 1e-4. Tanpa LoRA, konteks pair atau fitur linguistik tambahan pada S3 baru. S1/S2 tidak menjalankan CV, refit, atau evaluasi test baru; tabel hasilnya berasal dari sumber yang dicatat.

**GPU:** Runtime → Change runtime type → T4 GPU, lalu jalankan dari atas. CPU tidak dipilih otomatis. Parameter S3 lain tetap: maksimal 25 epoch/fold, patience 5, lima fold dalam development 80%, test 20%, microbatch 1/batch efektif 8, warmup dan scheduler linear. Model S3 baru tidak melanjutkan optimizer/bobot S3 lama karena LR/metodenya berbeda. Checkpoint S3 dalam eksperimen ini tetap bisa dilanjutkan setelah runtime terputus dengan konfigurasi/perangkat yang cocok.

**Sumber S1/S2:** REUSE_RUN kosong mencari run lengkap dalam MyDrive/AES_300_MLP_Privat, lalu AES_300_S3_LoRA_MLP_Privat. Di setiap folder, run terbaru dicoba dulu; sumber tidak dipilih berdasarkan skor. Pemeriksaan mencakup identitas kode sumber yang dikenal, dataset, split, konfigurasi S1/S2, versi numerik, lima fold, pilihan epoch, checksum model final/prediksi test dan keutuhan pemuatan model. Run yang tidak cocok ditolak. Jika folder berbeda, isi REUSE_RUN dengan folder hash run lama yang langsung berisi effective_config.json, final_report.json, S1 dan S2. Jika tidak ada sumber cocok, training berhenti dengan alasan dan tidak mengulang S1/S2 diam-diam.

**Folder baru:** MyDrive/AES_300_E6_S3_LR5e6_GPU_Reuse_Privat. Hanya hasil CV S1/S2, model final dan hasil test yang diperlukan disalin ke run baru; checkpoint epoch lama dan seluruh S3 sumber tidak disalin. Sumber tidak ditulis/dihapus. Penyalinan model final memerlukan ruang Drive tambahan. Rencana sumber/checksum dikunci sebelum copy untuk pemulihan jika terputus. Tidak menyalin refit_locked.json atau final_report.json dari run lama yang mengunci ketiga skenario.

Dataset tetap fix_dataset_300.xlsx, satu nilai guru 0–100; hanya jawaban siswa masuk encoder. Alur tokenisasi, vektor, prediksi, empat metrik dan lima Excel mengikuti acuan. Asal S1/S2 dan S3 ditampilkan di layar serta laporan pelatihan. Semua kode runtime tersedia di cell; tidak clone/import proyek atau memerlukan PEFT.

**Penyimpanan:** autosave tiap epoch S3 tetap aktif, tanpa auto-delete atau Drive API cleanup. Pertahankan final.pt dan JSON; pada tahap berjalan pertahankan dua snapshot yang dirujuk JSON progres. S1/S2 lama dan S3 lama tidak perlu dihapus. S1/S2 di laporan ini adalah hasil reuse, bukan hasil training baru. Test lama sudah dilihat, sehingga pemilihan LR harus melalui CV dan hasil pengembangan tetap eksploratif.
'''
            else:
                text=text.replace('CPU dipilih melalui device=cpu; S3_ENCODER_LR hanya mengubah RoBERTa S3, bukan S1.', 'GPU wajib untuk S3 baru; S1/S2 diperiksa dan dipakai ulang dari run selesai.')
                text=text.replace('Training CPU menggunakan float32 dan tidak mengaktifkan AMP.', 'Training S3 memakai AMP GPU seperti acuan; S1/S2 tidak dilatih ulang.')
                text=text.replace('CPU membutuhkan waktu lebih lama; checkpoint menyimpan optimizer, scheduler, riwayat dan keadaan acak untuk resume.', 'Checkpoint S3 menyimpan optimizer, scheduler, riwayat dan keadaan acak untuk resume dalam eksperimen yang sama.')
        cell['source']=text;cell['metadata']={};cell.pop('attachments',None)
    normalized=[source(c).removeprefix('%%aes\n') for c in notebook['cells'] if c['cell_type']=='code']
    for code in normalized:ast.parse(code)
    fingerprint=hashlib.sha256('\n'.join(normalized).encode()).hexdigest()
    for cell in notebook['cells']:cell['source']=source(cell).replace('__NOTEBOOK_CODE_SHA256__',fingerprint)
    notebook['metadata']['aes_standalone']['code_sha256']=fingerprint
    notebook['metadata']['accelerator']='GPU';notebook['metadata']['colab']['name']=DESTINATION.name
    notebook['metadata']['aes_experiment']={'name':'E6_S3_LR5E6_GPU_REUSE','base_sha256':hashlib.sha256(original).hexdigest(),
        'device':'cuda','s3_encoder_lr':5e-6,'reuse':['S1','S2'],'automatic_checkpoint_deletion':False,
        'drive_root':'AES_300_E6_S3_LR5e6_GPU_Reuse_Privat'}
    DESTINATION.write_text(json.dumps(notebook,ensure_ascii=False,indent=1)+'\n')
    mirror=ROOT/'aes-learning/notebooks/experiments'/DESTINATION.name;mirror.write_bytes(DESTINATION.read_bytes())
    assert BASE.read_bytes()==original
    print(json.dumps({'notebook':str(DESTINATION.relative_to(ROOT)),'code_sha256':fingerprint,'base_unchanged':True}))
    return DESTINATION


if __name__=='__main__':build()
