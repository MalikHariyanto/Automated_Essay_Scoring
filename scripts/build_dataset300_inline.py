"""Build03: actual code in cells, numeric output, complete private Excel exports."""
import ast
import json
from pathlib import Path
import re
import textwrap

ROOT = Path(__file__).resolve().parents[1]

COMMENTS = {
    'result.requires_grad_(': 'RoBERTa boleh belajar; bobot encoder BGE tidak diperbarui.',
    'ids = tokenizer.encode(': 'Tokenizer mengubah teks menjadi ID subword, bukan menghitung skor.',
    'self.head = nn.Linear(768, 1)': 'S1: 768 koordinat RoBERTa dipetakan ke satu prediksi kontinu.',
    'self.head = nn.Sequential(': 'S2: MLP belajar memberi skor di atas fitur BGE yang tetap.',
    'self.projection = nn.Sequential(': 'S3: padatkan BGE1024 menjadi256 sebelum digabung.',
    "features['concat'] = torch.cat(": 'Fusion menyambung RoBERTa768 dan proyeksiBGE256 menjadi1024.',
    'cls = self.roberta(': 'Ambil hidden state token pertama(CLS) untuk setiapwindow.',
    'essay = mean_by_essay(': 'Rata-rata CLS window menjadi satu vektor setiapjawaban.',
    'enc = [p for name, p in model.named_parameters()': 'Pilih bobot encoder RoBERTa yang menerima gradien.',
    'head = [p for name, p in model.named_parameters()': 'Pilih bobot head/proyeksi; kecepatannya bisa berbeda dari encoder.',
    'optimizer = torch.optim.AdamW(': 'AdamW memperbarui kelompok parameter sesuai learning rate masing-masing.',
    'targets = torch.as_tensor(': 'Normalisasi skor guru ke0–1 hanya untuk menghitung loss.',
    'scaler.scale(loss*': 'Hitung gradien dan bobot microbatch sesuai ukuran batch efektif aktual.',
    'scaler.step(optimizer)': 'Perbarui bobot; GradScaler bisa melewati langkah yang overflow pada AMP.',
    "if row['validation']['RMSE'] < best_rmse:": 'Pilih epoch hanya dari validation fold, bukan dari finaltest.',
    'clipped = np.clip(': 'Buat SALINAN skor sah untuk agreement; prediksi mentah tetap utuh.',
    'qwk = cohen_kappa_score(': 'QWK memakai jarak kategori kuadratik, bukan vektor embedding.',
    "result['AC2'] = gwet_ac2(": 'AC2 memakai kategori yang sama dengan QWK dan koreksi peluang berbeda.',
}


def source(path, names):
    raw = (ROOT/path).read_text(); lines = raw.splitlines(); found = []
    for node in ast.parse(raw).body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names:
            first = min([node.lineno]+[x.lineno for x in getattr(node,'decorator_list',[])])
            block = '\n'.join(lines[first-1:node.end_lineno])
            # Imports relatif diganti nama yang sudah didefinisikan di cell sebelumnya.
            annotated = []
            for line in block.splitlines():
                if line.lstrip().startswith('from .'):
                    continue
                for prefix, explanation in COMMENTS.items():
                    if line.lstrip().startswith(prefix):
                        annotated.append(line[:len(line)-len(line.lstrip())]+'# '+explanation)
                annotated.append(line)
            found.append('\n'.join(annotated))
    if len(found) != len(names):
        raise ValueError('Definisi source hilang: '+str(names))
    return '\n\n\n'.join(found)


def build():
    cells = []
    def md(text):
        if '## 12 ·' in text:
            text='''## 12 · Pelatihan dan evaluasi tiga skenario

            Model dilatih dengan lima fold pada data development. Epoch terbaik dipilih dari RMSE validasi, kemudian ketiga model dilatih ulang pada development sebelum final test dibuka. Cara ini tetap sama seperti versi sebelumnya.

            Riwayat seluruh fold digabung per model pada **05_Riwayat_Pelatihan.xlsx**. Prediksi dan empat metrik tersedia pada **04_Prediksi_dan_Evaluasi.xlsx**. File rincian per fold tetap disimpan oleh program, tetapi tidak dimasukkan ke paket utama.'''
        elif '## 14 ·' in text:
            text='''## 14 · Lihat vektor setelah model belajar

            Tahap penghitungan tetap mengambil fitur model final setelah evaluasi. Untuk laporan utama, cukup gunakan vektor RoBERTa sebelum/sesudah latihan, BGE-M3, dan gabungan fusion pada **02_Vektor_Jawaban.xlsx**.

            Vektor kata dan kalimat contoh tersedia pada **03_Vektor_Kata_dan_Kalimat.xlsx**. Angka lapisan internal seperti dropout atau aktivasi tidak perlu dilampirkan ke dosen; rincian itu tetap disimpan secara privat oleh program.'''
        elif '## 15 ·' in text:
            text='''## 15 · Unduh lima file hasil utama

            Paket akhir **Hasil_AES_Ringkas.zip** berisi:
            1. **Data dan Tokenisasi** — jawaban, nilai guru, pembagian data, serta token.
            2. **Vektor Jawaban** — seluruh komponen vektor per jawaban, sebelum/sesudah latihan dan fusion.
            3. **Vektor Kata dan Kalimat** — satu contoh jawaban dan perbandingan kata berulang.
            4. **Prediksi dan Evaluasi** — buka sheet **Perbandingan Model** terlebih dahulu.
            5. **Riwayat Pelatihan** — fold dan epoch digabung per model.

            Tiap file diawali sheet **Panduan**. Hasil yang belum dihitung tidak diisi angka pengganti. Paket parsial hanya memuat file yang datanya sudah tersedia.

            **Sudah punya ZIP versi lama?** Gunakan [notebook merapikan hasil](https://colab.research.google.com/github/MalikHariyanto/Automated_Essay_Scoring/blob/codex/aes-learning/notebooks/04_Rapikan_Hasil_AES.ipynb). Cukup upload ZIP lama, tanpa dataset, checkpoint, atau training ulang.'''
        replacements={'01_konfigurasi.xlsx':'05_Riwayat_Pelatihan.xlsx','01_rujukan_paper.xlsx':'05_Riwayat_Pelatihan.xlsx',
            '02_dataset_dan_cleaning.xlsx':'01_Data_dan_Tokenisasi.xlsx','03_split_dan_cv.xlsx':'01_Data_dan_Tokenisasi.xlsx',
            '04_tokenisasi.xlsx':'01_Data_dan_Tokenisasi.xlsx','05_roberta_pretrained.xlsx':'02_Vektor_Jawaban.xlsx',
            '06_bge_frozen.xlsx':'02_Vektor_Jawaban.xlsx','07_vektor_kata_kalimat_pretrained.xlsx':'03_Vektor_Kata_dan_Kalimat.xlsx',
            '08_arsitektur.xlsx':'05_Riwayat_Pelatihan.xlsx','13_prediksi_satu_jawaban.xlsx':'04_Prediksi_dan_Evaluasi.xlsx'}
        for old,new in replacements.items():text=text.replace(old,new)
        text=text.replace('**Tiap tahap penghitungan menyimpan Excel** dengan `Info`/provenance dan seluruh koordinat per kolom.',
                          '**Hasil akhir dirangkum dalam lima Excel** dengan nama sheet dan kolom yang mudah dibaca. Rincian setiap tahap tetap tersimpan secara privat.')
        text = re.sub(r'([A-Za-z]{3,})(\d)',r'\1 \2',text)
        text = re.sub(r'(\d)([A-Za-z]{3,})',r'\1 \2',text)
        for before,after in {'seluruhjawaban':'seluruh jawaban','tiapfold':'tiap fold','setiapfold':'setiap fold',
                             'setiapwindow':'setiap window','setiapepoch':'setiap epoch','semua skenario':'semua skenario',
                             'fullfine-tuned':'fully fine-tuned','modeleval':'model eval','checkpointfinal':'checkpoint final',
                             'nilaiguru':'nilai guru','teksbaru':'teks baru','melatihulang':'melatih ulang',
                             'out-of-range':'out-of-range','dataframe':'DataFrame','tidaktercakup':'tidak tercakup',
                             'skor/model':'skor/model','kelasXI':'kelas XI','finaltest':'final test',
                             'sebelumtest':'sebelum test','ditest':'di test','earlystopping':'early stopping',
                             'batchlain':'batch lain','metriklain':'metrik lain','fiturmanual':'fitur manual',
                             'fileunggahan':'file unggahan','bukanskor':'bukan skor','barubukan':'baru bukan',
                             'selesai.':'selesai.','testbaru':'test baru','Dropoutnonaktif':'Dropout nonaktif',
                             'bobot/cache':'bobot/cache','republik':'repo publik','repopublik':'repo publik',
                             'outputfusion':'output fusion','untukinspeksi':'untuk inspeksi','untukpelaporan':'untuk pelaporan'}.items():
            text=text.replace(before,after)
        cells.append(dict(cell_type='markdown',metadata={},source=textwrap.dedent(text).strip()+'\n'))
    def code(text, worker=True):
        cells.append(dict(cell_type='code',metadata={},execution_count=None,outputs=[],
                          source=('%%aes\n' if worker else '')+textwrap.dedent(text).strip()+'\n'))
    def defs(path,names,comment):
        code('# '+comment+'\n'+source(path,names))
    md('''
    # AES guru300 · RoBERTa, BGE-M3, fusion → prediksi dan evaluasi

    Urutan mengikuti **BGE-M3 [AES].ipynb** unggahan: instalasi/impor → Excel/dataset → split/tokenisasi → embedding → arsitektur → loss/optimizer → training → prediksi/evaluasi. **Kode proses ada di cell dengan komentar `#`; output tidak lagi mencetak source kode.**

    **S1** RoBERTa Indonesia fine-tuned. **S2** BGE-M3 frozen+MLP. **S3** fusion RoBERTa fine-tuned dan BGE frozen. Ini mempertahankan rancangan sebelumnya; notebook BGE unggahan sendiri memperbarui seluruh BGE. Dataset hanya mempunyai satu `rater`, jadi setiap model menghasilkan **satu skor**, bukan empat label rubrik yang tidak ada.

    Gunakan GPU Colab, Python3.12/3.13. Jalankan berurutan. Default5epoch/fold,5fold,15CVfit+3refit; proses bisa melewati sesi gratis. **Tiap tahap penghitungan menyimpan Excel** dengan `Info`/provenance dan seluruh koordinat per kolom. Preview notebook hanya cuplikan. Seluruh jawaban: embedding per jawaban; satu jawaban contoh: vektor token/kata/kalimat lengkap. Definisi fungsi menyiapkan proses, bukan menghasilkan skor dengan sendirinya.

    File dataset, Excel, checkpoint dan notebook terisi tetap privat. GitHub menyimpan source/notebook tanpa output.
    ''')
    md('## 1 · Ambil source terbaru\nSource publik diambil; dataset tidak ditanam di GitHub.')
    code('''
    import os, sys, json, subprocess, html, secrets, atexit
    from pathlib import Path
    from IPython.display import display, HTML
    REPO_URL = 'https://github.com/MalikHariyanto/Automated_Essay_Scoring.git'
    BRANCH = 'codex/aes-learning'
    IN_COLAB = 'google.colab' in sys.modules or Path('/content').is_dir()
    ROOT = Path('/content/Automated_Essay_Scoring') if IN_COLAB else Path.cwd()
    if not IN_COLAB:
        while not (ROOT/'src/aes').is_dir() and ROOT != ROOT.parent:
            ROOT = ROOT.parent
    if not (ROOT/'.git').exists():
        if ROOT.exists() and any(ROOT.iterdir()):
            raise RuntimeError('Folder target berisi file lain; kode tidak ditimpa.')
        subprocess.run(['git','clone','--branch',BRANCH,'--single-branch',REPO_URL,str(ROOT)],check=True)
    elif IN_COLAB:
        branch = subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()
        dirty = subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()
        if branch != BRANCH or dirty:
            raise RuntimeError('Checkout mempunyai branch/perubahan lain; simpan dahulu atau pakai runtime baru.')
        subprocess.run(['git','pull','--ff-only','origin',BRANCH],cwd=ROOT,check=True)
    print('Source:',subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
    print('Python kernel:',sys.version.split()[0])
    ''',False)
    md('''
    ## 2 · Instalasi dan pemilih Excel

    Setup mempertahankan torch GPU runtime. Python worker numerik terpisah menghindari konflik NumPy kernel Colab. Awalan **`%%aes` menjalankan kode cell pada worker yang sama**, sehingga variabel bertahan antar-cell. Bukan perintah menampilkan kode. Upload sekali per runtime; bila file sudah ada tidak diminta lagi.
    ''')
    code(r'''
    PYTHON = os.environ.get('AES300_PYTHON','')
    if not PYTHON:
        process = subprocess.Popen([sys.executable,str(ROOT/'scripts/setup_dataset300.py')],cwd=ROOT,
                                   stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        log=[]
        for line in process.stdout:
            print(line,end='',flush=True);log.append(line)
        if process.wait():
            raise RuntimeError('Setup gagal: '+''.join(log[-15:]))
        ready=json.loads((ROOT/'outputs/dataset300_setup.json').read_text())
        if ready['runtime_version'] != list(sys.version_info[:2]) or not ready['imports_verified']:
            raise RuntimeError('Metadata setup tidak cocok runtime aktif.')
        PYTHON=ready['python']
    DATASET_NAME='fix_dataset_300.xlsx'
    PRIVATE_INPUT=ROOT/'data/dataset300';PRIVATE_INPUT.mkdir(parents=True,exist_ok=True)
    DATA_PATH=Path(os.environ.get('AES300_DATASET',str(PRIVATE_INPUT/DATASET_NAME))).resolve()
    if not DATA_PATH.exists():
        if not IN_COLAB:
            raise FileNotFoundError(str(DATA_PATH))
        from google.colab import files
        previous=Path.cwd()
        try:
            os.chdir(PRIVATE_INPUT);uploaded=files.upload()
        finally:
            os.chdir(previous)
        if DATASET_NAME not in uploaded:
            raise ValueError('Upload '+DATASET_NAME)
    def display_aes_table(event):
        esc=lambda value:html.escape(str(value))
        th='background:#172035!important;color:#fff!important;padding:9px;border:1px solid #64748b'
        td='background:#f8fafc!important;color:#172035!important;padding:9px;border:1px solid #cbd5e1'
        body='<table><thead><tr>'+''.join('<th style="'+th+'">'+esc(x)+'</th>' for x in event['columns'])+'</tr></thead><tbody>'
        body+=''.join('<tr>'+''.join('<td style="'+td+'">'+esc(x)+'</td>' for x in row)+'</tr>' for row in event['rows'])
        print(event['title'])
        print('Cuplikan:',len(event['rows']),'dari',event['shape'][0],'baris. Data lengkap tersedia di Excel akhir.')
        display(HTML('<div style="max-height:420px;overflow:auto;background:#f8fafc;color:#172035">'+body+'</tbody></table></div>'))
    if '_aes_process' in globals() and _aes_process.poll() is None:
        _aes_process.terminate()
    _aes_prefix='__AES_'+secrets.token_hex(12)+'__'
    child_env={**os.environ,'PYTHONPATH':str(ROOT/'src'),'HF_HOME':str(ROOT/'outputs/cache/huggingface'),
               'AES300_DATASET':str(DATA_PATH),'TOKENIZERS_PARALLELISM':'false',
               'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','PYTHONUNBUFFERED':'1'}
    _aes_process=subprocess.Popen([PYTHON,str(ROOT/'scripts/aes_lesson_worker.py'),'--prefix',_aes_prefix],
                                  cwd=ROOT,env=child_env,stdin=subprocess.PIPE,stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT,text=True,bufsize=1)
    while True:
        line=_aes_process.stdout.readline()
        if not line:
            raise RuntimeError('Worker gagal mulai; lihat log setup.')
        if line.startswith(_aes_prefix):
            print('Worker siap:',json.loads(line[len(_aes_prefix):])['python']);break
        print(line,end='')
    def aes_cell(line,cell):
        if _aes_process.poll() is not None:
            raise RuntimeError('Worker berhenti; jalankan setup dan cell berikutnya dari awal.')
        _aes_process.stdin.write(json.dumps({'code':cell,'filename':'<notebook-cell>'})+'\n')
        _aes_process.stdin.flush()
        while True:
            output=_aes_process.stdout.readline()
            if not output:
                raise RuntimeError('Worker terhenti; jika RAM penuh pilih runtime lebih besar.')
            if not output.startswith(_aes_prefix):
                print(output,end='',flush=True);continue
            event=json.loads(output[len(_aes_prefix):])
            if event['kind']=='done':break
            if event['kind']=='error':raise RuntimeError(event['traceback'])
            if event['kind']=='table':display_aes_table(event)
    get_ipython().register_magic_function(aes_cell,'cell','aes')
    atexit.register(lambda:_aes_process.terminate() if _aes_process.poll() is None else None)
    print('Dataset siap. Jalankan cell %%aes berurutan.')
    ''',False)
    md('''
    ## 3 · Impor dan konfigurasi sebelum eksperimen

    Sama seperti file unggahan: impor pustaka, pilih device, tentukan epoch/batch. Default5epoch/fold mengikuti rancangan sebelumnya, bukan20epoch file unggahan. **Output:**`01_konfigurasi.xlsx` dan `01_rujukan_paper.xlsx`. Rujukan dipilih dari koleksi pengguna; tahun/versi dan batas penerapan dicatat, bukan klaim semua paper memakai model identik.
    ''')
    code('''
    import os, sys, json, math, hashlib, random, warnings, re, itertools, unicodedata, zipfile, subprocess, gc, logging
    from collections import Counter
    from pathlib import Path
    from importlib.metadata import version
    import numpy as np
    import pandas as pd
    import torch
    from torch import nn
    from torch.nn.functional import normalize
    from torch.utils.data import DataLoader
    from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
    from scipy.stats import pearsonr
    from sklearn.metrics import cohen_kappa_score
    from sklearn.model_selection import GroupKFold, GroupShuffleSplit
    from openpyxl import load_workbook
    from aes.cleaning import clean
    from aes.teacher_score import SCOPE
    from aes.teacher_score.data import DEFAULTS, REQUIRED, Bundle, config, digest, dump, read_dataset, run_folder
    from aes.teacher_score.excel import ExcelJournal, vector_frame
    SCENARIOS=('S1','S2','S3')

    # Ubah parameter DI SINI sebelum latihan; jangan tuning dari hasil test.
    EPOCHS,FOLDS=5,5
    MICRO_BATCH,EFFECTIVE_BATCH=1,8
    NEW_ANSWER=''  # Kosong: satu jawaban held-out untuk contoh scoring/vektor.
    AUDIT_ONLY=os.environ.get('AES300_AUDIT_ONLY')=='1'
    RUN_MODELS=RUN_TRAINING=RUN_WORD_VECTORS=RUN_FINAL_FEATURES=not AUDIT_ONLY
    cfg=config(dict(dataset=os.environ['AES300_DATASET'],output=str(ROOT/'outputs/dataset300_inline'),
                    sheet='Sheet1',expected_rows=300,score_min=0,score_max=100,
                    epochs=EPOCHS,folds=FOLDS,micro_batch=MICRO_BATCH,effective_batch=EFFECTIVE_BATCH,
                    device='auto',new_answer=NEW_ANSWER,vector_source='pretrained',lesson_version='inline_excel_v1',
                    runtime_versions={p:version(p) for p in ('torch','transformers','numpy')}))
    bundle=read_dataset(cfg)  # Periksa enam header, label, duplikat dan kelompok email.
    output=run_folder(cfg,bundle)
    journal=ExcelJournal(output/'excel',{'scope':SCOPE,'dataset_sha256':bundle.source_sha256,
                         'run_folder':str(output),'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                         'privacy':'PRIVAT; bukan untuk repo publik'})
    journal.save('01_konfigurasi',{'Parameter':pd.DataFrame([{'name':k,'value':v} for k,v in cfg.items()]),
                                 'Versi':pd.DataFrame([{'library':p,'version':version(p)} for p in ('torch','transformers','numpy','pandas','scikit-learn','openpyxl')])},
                 'Konfigurasi aktual; belum ada hasil training')
    references=json.loads((ROOT/'docs/aes_dataset300_references.json').read_text())
    journal.save('01_rujukan_paper',{'Rujukan':pd.DataFrame(references)},'Rujukan koleksi pengguna dan batas penerapan')
    show(pd.DataFrame(references)[['id','title','used_for']],'Rujukan proses')
    print('Device:', 'cuda' if torch.cuda.is_available() else 'cpu')
    ''')
    md('''
    ## 4 · Baca Excel dan cleaning

    Hanya `response_text` masuk encoder. Soal/kunci/email/skor tidak menjadi fitur. Cleaning mempertahankan ejaan, kapital, tanda baca/paragraf; sumber tidak ditimpa. Dataset satu nilai tidak memberi empat label rubrik; file bertopik metodologi penelitian ini juga tidak membuktikan populasi SMAkelasXI.

    **Output:** dataset asli, audit, teks sebelum/sesudah dan target loss `(rater−0)/100`→`02_dataset_dan_cleaning.xlsx`. Rujukan m01 menempatkan pemahaman/persiapan data sebelum modeling.
    ''')
    code('''
    df=pd.read_excel(cfg['dataset'],sheet_name=cfg['sheet'])
    df=df.loc[:,~df.columns.astype(str).str.startswith('Unnamed:')]
    cleaning=pd.DataFrame([{'row_id':r['row_id'],'teks_asli':r['raw_text'],'teks_model':r['text'],
                           'berubah':r['raw_text']!=r['text'],'skor_guru':r['score'],
                           'target_loss':(r['score']-cfg['score_min'])/(cfg['score_max']-cfg['score_min'])} for r in bundle.rows])
    journal.save('02_dataset_dan_cleaning',{'Dataset_asli':df,'Audit':pd.DataFrame([bundle.summary]),
                                          'Sebelum_sesudah':cleaning},'Pembacaan/cleaning aktual; sumber utuh')
    show(cleaning[['row_id','teks_model','skor_guru']],'Contoh jawaban dan nilai guru')
    print('Jumlah:',len(df),'| target:',bundle.summary['target_columns'],'| output model:1')
    ''')
    md('''
    ## 5 · Split80/20 dan5-fold validasi

    Email/duplikat tidak menyeberang split. Kandidat dipilih mendekati80/20 berdasarkan jumlah baris saja, tanpa melihat skor/model. Lima fold berada di development; tidak diperlukan split validasi ketiga. **Output:**`03_split_dan_cv.xlsx`, setiapbaris/kelompok/fold. Ini menguji jawaban kelompok baru pada pertanyaan yang sama, bukan unseen-prompt. Rujukan m09 membahas dependensi fold, bukan kewajiban universal5fold.
    ''')
    code(source('src/aes/teacher_score/data.py',['make_split'])+'''

manifest=make_split(bundle,cfg)
dump(output/'split_manifest.json',manifest)
roles={i:'development' for i in manifest['development']}
roles.update({i:'test' for i in manifest['test']})
split_rows=pd.DataFrame([{'row_id':r['row_id'],'role':roles[i],'connected_group':int(bundle.groups[i])} for i,r in enumerate(bundle.rows)])
fold_rows=pd.DataFrame([{'fold':f['fold'],'row_id':bundle.rows[i]['row_id'],'role':role} for f in manifest['folds'] for role,key in [('train','train'),('validation','validation')] for i in f[key]])
journal.save('03_split_dan_cv',{'Split_setiap_baris':split_rows,'Fold_setiap_baris':fold_rows,
                              'Ringkasan':pd.DataFrame([{k:v for k,v in manifest.items() if k not in ('folds','development','test')}])},'Split dan5fold sama untukS1/S2/S3')
show(split_rows.groupby('role').size().rename('rows').reset_index(),'Pembagian aktual')
trace_index=manifest['test'][cfg['trace_position']]
trace_text=cfg.get('new_answer','').strip() or bundle.rows[trace_index]['text']
print('Contoh:',bundle.rows[trace_index]['row_id'],'| bukan untuk memilih epoch.')
''')
    md('''
    ## 6 · Tokenizer dan window yang benar-benar dipakai

    RoBERTa512=510isi+2special, overlap128, seluruh token akhir tercakup. BGE memakai tokenizer berbeda, batas eksperimen1024 termasukspecial; truncation dilaporkan. Dasar tokenizer/encoder: m04/p14; batas/overlap ini keputusan proyek. **Output:**token,ID,attentionmask,padding,window/coverage→`04_tokenisasi.xlsx`.
    ''')
    defs('src/aes/windows.py',['content_windows','batch_windows','mean_by_essay'],'Buat window, tambahkan special sekali, mean CLS secara differentiable.')
    defs('src/aes/teacher_score/models.py',['device_name','release','ExpectedPoolerNotice','encoder','tokenizer'],'Checkpoint terkunci; RoBERTa requires_grad=True, BGE frozen.')
    code(source('src/aes/teacher_score/lesson.py',['export_tokens'])+'''

if RUN_MODELS:
    token_roberta,token_bge=export_tokens(bundle,cfg,journal)
    show(token_roberta,'Token RoBERTa');show(token_bge,'Token BGE')
else:
    print('AUDIT ONLY: belum tokenisasi encoder; tidak ada angka pengganti.')
''')
    md('''
    ## 7 · RoBERTa pretrained → embedding Excel

    `last_hidden_state[:,0,:]` adalah CLS tiapwindow; mean antarwindow menjadi fitur jawaban. **Output:**seluruhjawaban×768 dan setiapCLSwindow→`05_roberta_pretrained.xlsx`. Ini SEBELUM fine-tuning; bukan nilai siswa dan bukan similarity yang otomatis terkalibrasi. Semua koordinat tersimpan, preview hanya sedikitkolom.
    ''')
    code(source('src/aes/teacher_score/lesson.py',['roberta_pretrained_vectors'])+'''

if RUN_MODELS:
    roberta_before=roberta_pretrained_vectors(bundle,cfg,journal)
    show(vector_frame(roberta_before,[r['row_id'] for r in bundle.rows]),'RoBERTa pretrained [N,768]')
''')
    md('''
    ## 8 · BGE-M3 frozen → embedding Excel

    **Output:**seluruhjawaban×1024→`06_bge_frozen.xlsx`. Dense CLS+L2 mengikuti p14§3.2, `e=norm(H[0])` (PDF gabungan3135). Vektor belum menjadi skor; head belajar memetakannya. Tidak memakai sparse/ColBERT atau melatih contrastive retrieval ulang.
    ''')
    code(source('src/aes/teacher_score/models.py',['bge_features'])+'''

if RUN_MODELS:
    bge=bge_features(bundle,cfg,output)
    journal.save('06_bge_frozen',{'Vektor_jawaban_1024':vector_frame(bge.numpy(),[r['row_id'] for r in bundle.rows]),
                                'Encoding':pd.DataFrame([json.loads((output/'bge_encoding.json').read_text())])},
                 'BGE dense frozen; semua koordinat',source=cfg['bge'],pooling='CLS then L2')
    show(vector_frame(bge.numpy(),[r['row_id'] for r in bundle.rows]),'BGE frozen [N,1024]')
''')
    md('''
    ## 9 · Vektor token/kata/kalimat satu jawaban → Excel

    Lookup awal dibanding hidden state kontekstual terakhir. Kata=mean subword beririsan span; overlap dirata-rata pertoken dahulu. Kalimat dibaca terpisah memakaiCLS. Kata BGE di luar batas ditandai tidaktercakup/NaN(sel kosong), bukan0. Segmentasi tanda baca sederhana bisa keliru pada singkatan.

    **Output:**semua koordinat,metadata,ID/span,coverage,perbandingan kata berulang→`07_vektor_kata_kalimat_pretrained.xlsx`. ID sama pada checkpoint sama→lookup sama; konteks/posisi dapat mengubah vektor akhir. Bandingkan pengulangan di encoder yang sama; jangan cosine langsung ruang768 dengan1024.
    ''')
    code("WORD = re.compile(r\"\\b\\w+(?:[-’']\\w+)*\\b\", re.UNICODE)\n"+source('src/aes/teacher_score/vectors.py',['sentence_spans','token_vectors','word_vectors','cosine','compare_repetitions','vector_lab'])+'\n'+source('src/aes/teacher_score/lesson.py',['export_word_sentence_vectors'])+'''

if RUN_WORD_VECTORS:
    before_words=vector_lab(trace_text,{**cfg,'vector_source':'pretrained'},output)
    export_word_sentence_vectors(before_words,journal,'07_vektor_kata_kalimat_pretrained')
    for role in ('roberta','bge'):
        preview=pd.DataFrame(before_words[role]['words'],columns=['word_index','word','covered','contextual_first5'])
        preview=preview.rename(columns={'contextual_first5':'Lima Komponen Pertama'})
        show(preview,('RoBERTa' if role=='roberta' else 'BGE-M3')+' — contoh vektor kata')
        from aes.teacher_score.presentation import repeated_word_rows
        pairs=list(repeated_word_rows(before_words[role]['repeated_words'],role))
        show(pd.DataFrame(pairs,columns=['Model','Kata','Posisi 1','Posisi 2','ID Sama','Vektor Awal Sama','Vektor Akhir Sama','Kemiripan Akhir','Apa Artinya']),
             'Perbandingan kata yang berulang',columns=9)
''')
    md('''
    ## 10 · ModelS1/S2/S3 dan forward

    S1:RoBERTa768→linear1. S2:BGE1024→MLP256→1. S3:RoBERTa768+BGE diproyeksikan256→concat1024→linear1. `forward` di cell benar-benar dipakai saattraining. Bobot head baru belum layak menjadi skor.

    p10 mendukung konsep menggabungkan fitur, tetapi memakai RoBERTa+fiturmanual, bukan replikasi RoBERTa+BGE ini. BGE dalam unggahan fullyfine-tuned; di sini S2/S3frozen sesuai rancangan sebelumnya, head/proyeksi tetapbelajar. **Output spesifikasi:**`08_arsitektur.xlsx`, jelas diberi statusDESIGN, bukan angka hasilmodel.
    ''')
    defs('src/aes/teacher_score/models.py',['SingleScoreModel','Engine'],'Satu head satu nilai; Engine menyiapkan tensor dan mengembalikan satuan0–100.')
    defs('src/aes/teacher_score/experiment.py',['seed_everything','make_model'],'Model baru pada setiapfold/refit, bukan meneruskan bobot foldlain.')
    code('''
    specifications=pd.DataFrame([
        {'scenario':'S1','encoder':'RoBERTa fine-tuned','vector':768,'head':'Linear768→1','output':1},
        {'scenario':'S2','encoder':'BGE frozen','vector':1024,'head':'MLP1024→256→1','output':1},
        {'scenario':'S3','encoder':'RoBERTa fine-tuned+BGE frozen','vector':'768+1024','head':'proyeksi256;concat1024→1','output':1}])
    journal.save('08_arsitektur',{'Spesifikasi':specifications},'Spesifikasi model, bukan prediksi',status='DESIGN / CONFIGURATION')
    show(specifications,'Arsitektur dan bagian yang belajar')
    ''')
    md('''
    ## 11 · Empat metrik, loss, optimizer dan batch

    RMSE/Pearson memakai prediksi mentah0–100. QWK/AC2 kuadratik memakai salinan clip0–100 lalu integerterdekat(tie.5kebawah),101kategori dipertahankan. Undefined=kosong+alasan. AC2 sensitif kategoridan bisa tinggi saat QWK buruk; baca bersama metriklain(m06). Rumus AC2 juga diverifikasi dengan sourceirrCAC; bab rumus tidak tersedia dalam cuplikan bukum08.

    `fit` di bawah adalah fungsi latihan sebenarnya. RoBERTaLR2e−5; head/proyeksi1e−3; AdamW,MSE,warmup10%,clipgrad1. Microbatch1 diakumulasi menjadi efektif8. Target0–1 hanya untukloss, RMSE tetap poinasli. **Output sesudahtraining:**riwayatepoch/loss/metrik/epochpilihan→Excel tiapfold/refit.
    ''')
    defs('src/aes/metrics.py',['gwet_ac2'],'Agreement kuadratik Gwet, bukan sekadar korelasi.')
    defs('src/aes/teacher_score/models.py',['metrics'],'Empat metrik tanpa menimpa prediksi mentah.')
    defs('src/aes/teacher_score/experiment.py',['fit'],'backward menghitung gradien; optimizer.step memperbarui parameter.')
    md('''
    ## 12 · TrainingS1→S2→S3, refit, finaltest

    CV development memilih epoch minimumRMSE. Refit memakai ceil median epoch5fold. Semua konfigurasi/refit selesai sebelumtest, tanpa earlystopping/tuning ditest. p08 juga membedakan fungsi validation dantesting; bentuk5fold ini keputusan proyek, bukan menyalin split70/15/15paper. Fold yang berbagi data tidak independen(m09).

    **Output:**`09_S1_fold_1.xlsx` sampai`09_S3_fold_5.xlsx`(riwayat setiapepoch,prediksi/metrikvalidation), `10_S1_refit.xlsx`dst., `11_S1_test.xlsx`dst., `12_evaluasi_tiga_skenario.xlsx`. Tiapfold selesai langsung disimpan. Resume hasil cocok sumber/config/code memakai test tersimpan.
    ''')
    defs('src/aes/teacher_score/experiment.py',['save_model','load_model','train_all'],'CV semua skenario, refitsemua, lalu finaltest dengan checkpoint terkunci.')
    code(source('src/aes/teacher_score/lesson.py',['ordinal_copy','prediction_frame','export_training_event','export_completed_training'])+'''

if RUN_TRAINING:
    if not RUN_MODELS:raise RuntimeError('Aktifkan RUN_MODELS sebelumtraining.')
    def on_training_stage(kind,scenario,result):
        export_training_event(kind,scenario,result,bundle,cfg,journal)
    report=train_all(bundle,manifest,bge,cfg,output,observer=on_training_stage)
    export_completed_training(report,bundle,cfg,output,journal)
    show(pd.DataFrame(report['cv']),'Hasil validasi silang (lima fold)',rows=15)
    final_table=pd.DataFrame(report['rows'])[['Skenario','RMSE','Pearson','QWK','AC2']]
    show(final_table,'Perbandingan model pada data uji',rows=3)
else:
    print('Training belumdijalankan. Tidak ada angka penelitian pengganti.')
''')
    md('''
    ## 13 · Prediksi satu jawaban dengan3model final

    Sama seperti akhir fileunggahan: tampilkan prediksi. `NEW_ANSWER`kosong memakai contohheld-out; teksbaru tidak memerlukan nilaiguru dan tidak melatihulang. **Output:**3skormentah,kategorisalinan,out-of-range→`13_prediksi_satu_jawaban.xlsx`.
    ''')
    code(source('src/aes/teacher_score/experiment.py',['score_answer'])+'''

if RUN_TRAINING:
    prediction_rows=score_answer(trace_text,cfg,output)
    journal.save('13_prediksi_satu_jawaban',{'Input_privat':pd.DataFrame([{'text':trace_text}]),
                                           'Tiga_skenario':pd.DataFrame(prediction_rows)},'Prediksi checkpointfinal; bukan trainingulang')
    show(pd.DataFrame(prediction_rows),'Prediksi skor0–100 ketigaskenario',columns=6)
''')
    md('''
    ## 14 · Fitur SESUDAHtraining dan outputfusion → Excel

    Setelah finaltest, ekstrak fitur modeleval:RoBERTa768,BGE1024,proyeksi256sebelum/sesudahLayerNorm,concat1024,hiddenMLP/GELU/dropouteval,outputhead. Hanya komponen yang ada pada skenario diekspor. Dropout nonaktif saatinspeksi, berbeda daritraining.

    **Output:**`14_S1_fitur_setelah_training.xlsx`,S2,S3. Cocokkan`row_id`dengan05untuk membandingkan sebelum/sesudah. BGE tetapfrozen. Semua jawaban untukinspeksi, bukan evaluasitestbaru. Vektor kata/kalimat contohsesudahfine-tuning→`15_S1_...xlsx`dan`15_S3_...xlsx`.
    ''')
    code(source('src/aes/teacher_score/lesson.py',['export_final_features'])+'''

if RUN_TRAINING and RUN_FINAL_FEATURES:
    print('Mengambil vektor jawaban dari ketiga model yang sudah dilatih...')
    export_final_features(bundle,bge,cfg,output,journal)
    print('Vektor',len(bundle.rows),'jawaban siap disajikan dalam 02_Vektor_Jawaban.xlsx.')
    if RUN_WORD_VECTORS:
        for scenario in ('S1','S3'):
            after_words=vector_lab(trace_text,{**cfg,'vector_source':scenario},output)
            export_word_sentence_vectors(after_words,journal,f'15_{scenario}_vektor_kata_setelah_training')
''')
    md('''
    ## 15 · Indeks output dan unduhExcel privat

    `00_indeks_output.xlsx`mendaftar workbook/shape yang benar-benar selesai. Tidak ada angka dummy untuk tahap yang belumjalan. Semua koordinat vektor perkolom, sumber/revision/pooling/statusdi`Info`. Arsip hanyaExcel, bukanbobot/cache. Janganupload fileberisi teks/nilai ke repopublik. Download atau simpanfolderoutput keDriveprivat sebelumruntime dihapus.
    ''')
    code('''
    from aes.teacher_score.presentation import build_reports
    summary=build_reports(journal.folder,output/'hasil_ringkas')
    dump(ROOT/'outputs/dataset300_download.json',{'archive':summary['archive'],'run_folder':str(output),'excel_count':len(summary['files'])})
    print('Selesai. Paket unduhan berisi',len(summary['files']),'Excel utama.')
    if any(Path(path).name.startswith('04_') for path in summary['files']):
        print('Untuk melihat hasil akhir, buka 04_Prediksi_dan_Evaluasi.xlsx → Perbandingan Model.')
    else:
        print('Prediksi dan evaluasi belum tersedia. Paket ini hanya memuat tahap yang sudah selesai.')
    ''')
    code('''
    download=json.loads((ROOT/'outputs/dataset300_download.json').read_text())
    print('Fileprivat:',download['archive'])
    if IN_COLAB:
        from google.colab import files
        files.download(download['archive'])
    ''',False)
    nb=dict(cells=cells,metadata={'kernelspec':{'display_name':'Python3','language':'python','name':'python3'},
             'language_info':{'name':'python'},'colab':{'name':'03_AES_dataset_guru_300.ipynb'}},nbformat=4,nbformat_minor=4)
    content=json.dumps(nb,ensure_ascii=False,indent=1)+'\n'
    for directory in ('notebooks','aes-learning/notebooks'):
        (ROOT/directory/'03_AES_dataset_guru_300.ipynb').write_text(content,encoding='utf-8')
    return nb


if __name__=='__main__':
    print('Notebookinline:',len(build()['cells']),'cell;2mirror identik tanpaoutput.')
