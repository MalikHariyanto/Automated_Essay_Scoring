"""E21/E22 CPU lokal: tiga input seluruh skenario, pasangan sebagai ablation."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import subprocess
from build_context_pair_notebook import source, replace_once, definitions

ROOT=Path(__file__).resolve().parents[1]
PIN='8380355f0db53a1d258f29a2ab17152fd965d85d'
BASE='notebooks/experiments/AES_300_E6_Tiga_Input_CPU.ipynb'
NAMES={'E21':'AES_300_E21_Tiga_Input_CPU', 'E22':'AES_300_E22_Tiga_Input_ContextPair_CPU'}


def build(experiment):
    raw=subprocess.check_output(['git','show',PIN+':'+BASE],cwd=ROOT)
    n=copy.deepcopy(json.loads(raw));previous=n['metadata']['aes_standalone']['code_sha256']
    mode='single_sequence' if experiment=='E21' else 'context_pair'
    name=NAMES[experiment]
    helpers=definitions((ROOT/'scripts/e21_e22_input_helpers.py').read_text())
    helper=helpers['pair_content_windows']
    for index,cell in enumerate(n['cells']):
        s=source(cell).replace(previous,'__NOTEBOOK_CODE_SHA256__')
        s=s.replace('AES_300_E6_Tiga_Input_CPU',name)
        s=s.replace('Hasil_AES_E6_Tiga_Input_CPU.zip','Hasil_'+name+'.zip')
        s=s.replace('e6_three_inputs_all_scenarios_cpu_v1','three_inputs_format_ablation_local_cpu_v1')
        if cell['cell_type']=='code':
            if index==3:
                # Abaikan override persistent lama: tidak me-mount/menulis Drive.
                s=replace_once(s,definitions(s)['prepare_persistent_storage'],'''def prepare_local_storage(workspace, in_colab):
    storage=Path(workspace).expanduser().resolve()
    if storage==Path('/content/drive') or Path('/content/drive') in storage.parents:
        raise ValueError('E21/E22 hanya menyimpan lokal; pilih workspace di luar Google Drive.')
    storage.mkdir(parents=True,exist_ok=True)
    probe=storage/'.aes_write_check';probe.write_text('local write check');probe.unlink()
    return storage
''')
                s=replace_once(s,'ROOT.mkdir(parents=True,exist_ok=True)',
                    "ROOT=ROOT.resolve()\nif ROOT==Path('/content/drive') or Path('/content/drive') in ROOT.parents:\n    raise ValueError('Workspace harus lokal; E21/E22 tidak menulis Google Drive.')\nROOT.mkdir(parents=True,exist_ok=True)")
                s+='\n'+helpers['resolve_dataset_upload']+'\n'
            if index==7:
                s=s.replace('prepare_persistent_storage(ROOT,IN_COLAB)','prepare_local_storage(ROOT,IN_COLAB)')
                s=s.replace("print('Progres disimpan otomatis setiap epoch. Saat sesi terputus, jalankan kembali notebook yang sama dari atas.')",
                    "print('Checkpoint hanya lokal. Bisa dilanjutkan selama folder runtime masih ada; runtime dihapus berarti file lokal hilang. Unduh hasil sebelum menutup sesi.')")
                start=s.index("DATASET_NAME='fix_dataset_300.xlsx'")
                end=s.index('def display_aes_table(event):',start)
                s=s[:start]+'''# Web Colab biasa: Choose Files muncul jika dataset belum diunggah.
DATA_PATH=resolve_dataset_upload(PERSISTENT_ROOT,DATASET_FILE or os.environ.get('AES300_DATASET',''),browser_upload=IN_COLAB)
'''+s[end:]
                s="DATASET_FILE=''  # Biasanya otomatis dari /content; isi path lengkap jika berbeda.\n"+s
            if index==10:
                s=s.replace("'encoder_lr': 1e-5", "'encoder_lr': 5e-6")
                s=replace_once(s,"'input_policy':'three_texts_all_scenarios_v1',",
                    "'input_policy':'three_texts_all_scenarios_v1', 'input_format':'"+mode+"',")
                s=replace_once(s,'    return result',
                    "    if result['input_format'] not in ('single_sequence','context_pair'):\n        raise ValueError('Format input tidak sah.')\n    return result")
            if index==11:
                s=replace_once(s,"'response_text':'Jawaban Siswa'","'input_format':'Format Input','response_text':'Jawaban Siswa'")
            if index==12:
                s=s.replace('ENCODER_LR=1e-5  # S1 tetap mengikuti pengaturan sebelumnya.',
                            'ENCODER_LR=5e-6  # LR encoder RoBERTa S1 dan S3 disamakan.')
                s=s.replace('S3_ENCODER_LR=5e-6  # E6 tiga input: LR encoder S3 tetap 5e-6.',
                            'S3_ENCODER_LR=ENCODER_LR')
                s=replace_once(s,"NEW_QUESTION=''", "INPUT_FORMAT='"+mode+"'\nNEW_QUESTION=''")
                s=s.replace('device=\'cpu\',new_answer=', "input_format=INPUT_FORMAT,device='cpu',new_answer=")
                s=s.replace('E6 tiga input: S1/S2/S3 dilatih baru; proyeksi S3 tetap 1e-4',
                            'Tiga input seluruh skenario; LR encoder 5e-6; proyeksi/head 1e-4')
                s+="\nprint('Format input:',cfg['input_format'],'| penyimpanan: disk runtime lokal, tanpa Google Drive',flush=True)\n"
            if index==20:
                s=replace_once(s,definitions(s)['pair_content_windows'],helper)
            if index==26:
                s=replace_once(s,"'policy':'three_inputs_pair_mean_CLS_then_L2_v1'",
                              "'policy':'three_inputs_mean_CLS_then_L2_v2', 'input_format':cfg['input_format']")
                s=s.replace("'feature_source':'soal + jawaban_referensi + response_text; no teacher score',",
                            "'feature_source':'soal + jawaban_referensi + response_text; no teacher score', 'input_format':cfg['input_format'],")
            if index==28:
                s=s.replace("item['input_policy']='soal/referensi berpasangan dengan jawaban' if paired else 'jawaban siswa saja'",
                            "item['input_policy']='soal + referensi + jawaban; '+cfg['input_format']")
            if index==36:
                s=s.replace('Dua nama tetap, tidak menghapus file Drive atau membuat file per epoch.',
                            'Dua slot pada disk runtime lokal; tidak membuat satu file per epoch.')
                s=s.replace('tersimpan otomatis.', 'tersimpan di disk lokal.')
            if index==45:
                s=replace_once(s,"settings={'epochs':'Maksimum epoch per fold'",
                    "settings={'input_format':'Format input tiga teks','epochs':'Maksimum epoch per fold'")
            if index==46:
                s=replace_once(s,"if IN_COLAB:\n    from google.colab import files\n    files.download(download['archive'])\nelse:print('Lokasi file:',download['archive'])",
                    """print('Lokasi ZIP pada runtime:',download['archive'])
print('Web Colab: panel Files/Berkas → cari ZIP tersebut → menu tiga titik → Download.')
DOWNLOAD_WITH_COLAB_BROWSER=True  # Unduh ZIP hasil otomatis pada web Colab.
if IN_COLAB and DOWNLOAD_WITH_COLAB_BROWSER:
    from google.colab import files
    files.download(download['archive'])""")
            s=s.replace('Token pasangan aktual: soal/referensi + jawaban untuk seluruh skenario',
                        'Token aktual tiga input seluruh skenario; lihat kolom input_format pada cakupan')
            s=s.replace('input sama untuk S1 dan S3','input sama untuk S1 dan S3; format mengikuti konfigurasi')
            s=s.replace('S1/S3 sama-sama membaca pasangan soal/referensi + jawaban.',
                        'S1/S3 membaca soal + referensi + jawaban dengan format eksperimen yang sama.')
            s=s.replace('konteks diulang per window','soal/referensi diulang per window')
            cell['outputs']=[];cell['execution_count']=None
        cell['source']=s;cell['metadata']={};cell['id']=f'aes-cell-{index:02d}';cell.pop('attachments',None)
    n['cells'][0]['source']=intro(experiment,mode,name)
    for cell in n['cells'][1:]:
        if cell['cell_type']=='markdown':
            s=source(cell).replace('E6 tiga input',experiment+' tiga input')
            cell['source']=s.replace('Semua skenario membaca soal + jawaban referensi + jawaban siswa.',
                'Semua skenario membaca soal + referensi + jawaban; format '+mode+'. Hasil hanya pada disk lokal.')
    codes=[source(c).removeprefix('%%aes\n') for c in n['cells'] if c['cell_type']=='code']
    for code in codes:ast.parse(code)
    fingerprint=hashlib.sha256('\n'.join(codes).encode()).hexdigest()
    for cell in n['cells']:cell['source']=source(cell).replace('__NOTEBOOK_CODE_SHA256__',fingerprint)
    n['metadata']['aes_standalone']['code_sha256']=fingerprint
    n['nbformat_minor']=5
    n['metadata']['colab']['name']=name+'.ipynb';n['metadata'].pop('accelerator',None)
    n['metadata']['aes_experiment']={'name':experiment,'device':'cpu','precision':'float32',
        'base_commit':PIN,'base_path':BASE,'base_sha256':hashlib.sha256(raw).hexdigest(),
        'input_columns':['soal','response_text','jawaban_referensi'],'target':'rater',
        'input_format':mode,'s1':'roberta_three_inputs_full_finetuning_linear',
        's2':'bge_three_inputs_frozen_mlp','s3':'three_inputs_fusion_mlp',
        's1_encoder_lr':5e-6,'s3_encoder_lr':5e-6,'projection_lr':1e-4,'head_lr':1e-4,
        'reused_controls':[],'checkpoint_strategy':'two_slots_local_runtime_only',
        'automatic_checkpoint_deletion':False,'drive_mount':False,'drive_autosave':False}
    target=ROOT/'notebooks/experiments'/(name+'.ipynb')
    target.write_text(json.dumps(n,ensure_ascii=False,indent=1)+'\n')
    (ROOT/'aes-learning/notebooks/experiments'/target.name).write_bytes(target.read_bytes())
    print(json.dumps({'experiment':experiment,'notebook':str(target.relative_to(ROOT)),
                      'code_sha256':fingerprint,'blocks':len(codes),'input_format':mode}))
    return target


def intro(experiment,mode,name):
    description=('Satu urutan teks: soal + referensi + penanda jawaban + jawaban siswa.' if mode=='single_sequence'
                 else 'Pasangan tokenizer: teks A = soal + referensi; teks B = penanda jawaban + jawaban siswa.')
    return f'''# AES 300 · {experiment} · tiga input · CPU · tanpa simpan Drive

**X:** `soal`, `jawaban_referensi`, `response_text`, tersedia untuk SELURUH S1/S2/S3. **y:** `rater` (satu skor guru 0–100), tidak masuk tokenizer. Email/duplikat hanya untuk kelompok pencegah kebocoran; bukan fitur encoder.

**Perbedaan E21/E22:** {description} Kedua eksperimen memuat fungsi yang sama; format input adalah satu-satunya perubahan metode antarkeduanya. Anggaran soal64/konteks192, penanda jawaban, seluruh token jawaban, starts/overlap dan ukuran window disamakan. E21 menggunakan special token satu urutan; E22 memakai separator pasangan resmi tokenizer. RoBERTa/XLM-R tidak memakai token-type A/B seperti BERT; pasangan dibedakan oleh special separator, posisi dan perhatian antartoken. Tidak menambah encoder atau target baru.

**Model mengikuti E6:** S1 RoBERTa full fine-tuning → mean CLS768 → dropout → linear1; S2 BGE-M3 frozen → mean CLS1024 → L2 → MLP256/GELU/dropout/linear1; S3 LayerNorm RoBERTa768 + proyeksi BGE1024→256/LayerNorm → concat1024 → MLP256/GELU/dropout/linear1. S1 tetap linear sesuai pilihan pengguna. Semua model dilatih ulang, tanpa mengambil hasil E6 lama.

**Training:** LR encoder RoBERTa S1/S3=5e-6; head/proyeksi=1e-4; BGE frozen. CPU float32 dua thread. Seed42/split_seed2026, 80/20, lima fold development, maksimal25epoch/fold, early stopping patience5, microbatch1/batch efektif8, warmup10%, AdamW/scheduler linear. Refit memakai ceil median epoch terbaik CV. MSE target0–1; RMSE/Pearson pada poin0–100 dan QWK/GwetAC2 pada salinan ordinal. CPU penuh bisa lama; tidak menjanjikan RMSE lebih baik.

**Tanpa Google Drive:** tidak mount, autentikasi, Drive API atau autosave Drive. Dataset, model, checkpoint dua slot dan Excel disimpan HANYA di folder lokal `{name}`. Jalankan ulang dari atas untuk melanjutkan jika folder lokal masih ada. Runtime Colab dihapus/diganti berarti file lokal hilang; unduh hasil sebelum sesi berakhir. Notebook tidak menghapus hasil eksperimen lain dan tidak memulihkan otomatis dari Drive.

**Web Colab biasa:** buka notebook lalu **Runtime → Run all**. Pada cell setup klik **Choose Files**, pilih `fix_dataset_300.xlsx` dari komputer dan tunggu upload selesai. Jika file sudah ada di runtime, notebook langsung memakainya. Cell terakhir mengunduh ZIP lima Excel lengkap. Tidak perlu ekstensi/editor lain atau menghubungkan Drive. Jika tombol upload terlihat abu-abu dari output sesi lama, jalankan ulang cell setup di sesi browser saat ini. Untuk prediksi baru isi NEW_QUESTION/NEW_REFERENCE/NEW_ANSWER; `rater` baru tidak diperlukan. ZIP laporan tidak berisi bobot model: `S1/final.pt`, `S2/final.pt`, `S3/final.pt` tersedia di folder run lokal dan dapat diunduh manual jika dibutuhkan lagi.

Pemotongan soal/referensi dicatat; seluruh token jawaban mendapat window. Batas sama menjaga perbandingan format. Ini generalisasi jawaban pada soal yang tersedia, bukan uji soal baru. Data uji lama sudah pernah dilihat selama pengembangan; hasil lanjutan bersifat eksploratif, pemilihan metode tetap melalui development/CV. Notebook publik tidak menyertakan data/skor individual atau hasil penelitian pengganti.
'''


if __name__=='__main__':
    for experiment in NAMES:build(experiment)
