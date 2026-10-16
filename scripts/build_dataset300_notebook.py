"""Bangun notebook publik tanpa output; tidak membaca dataset privat."""
import json
from pathlib import Path
import textwrap

ROOT = Path(__file__).resolve().parents[1]


def build():
    cells = []
    def md(text):
        cells.append({'cell_type': 'markdown', 'metadata': {}, 'source': textwrap.dedent(text).strip()+'\n'})
    def code(text):
        cells.append({'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [],
                      'source': textwrap.dedent(text).strip()+'\n'})

    md('''
    # AES dengan dataset guru 300: S1 → S2 → S3 → evaluasi → prediksi

    **Notebook terbaru untuk `fix_dataset_300.xlsx`.** Jalankan dari atas. Urutan mengikuti notebook BGE-M3 unggahan: instalasi → impor/input → konfigurasi/dataset → model → optimizer/metrik → training → evaluasi/prediksi. Ditambahkan audit, holdout kelompok, 5-fold CV, dan laboratorium vektor.

    **Cakupan baru: satu skor guru.** Dataset menyediakan `rater`, bukan empat label isi/organisasi/kebahasaan/mekanik. Output tiap skenario `[N,1]`. Notebook lama empat dimensi tetap tersedia dan tidak diubah.

    Pilih **Runtime → Change runtime type → GPU (T4 atau lebih besar), Python 3.12**. BGE-M3 membutuhkan unduhan besar dan RAM; 15 latihan CV + 3 refit dapat melewati satu sesi gratis. Hasil fold disimpan untuk melanjutkan konfigurasi yang sama. CPU bisa dipakai tetapi sangat lambat. Gunakan runtime baru untuk menghindari paket kernel lama.

    File siswa, email, output terisi, dan checkpoint **tetap privat**. GitHub hanya menyimpan kode dan notebook kosong. Notebook ini akan menjalankan training sungguhan; angka penelitian baru muncul setelah proses itu selesai, bukan dari contoh simulasi.
    ''')
    md('''
    ## 1 · Siapkan source dan pilih eksperimen

    Default **5 epoch maksimum per fold**, bukan 20 epoch seperti notebook unggahan. `EPOCHS` dapat diubah **sebelum** eksperimen; jangan memilihnya dari hasil test. Epoch ialah satu putaran seluruh data training pada fold tersebut.

    `MICRO_BATCH=1`: satu jawaban diproses pada satu forward/backward agar hemat memori. Gradien dikumpulkan sampai `EFFECTIVE_BATCH=8` sebelum optimizer memperbarui parameter. Batch terakhir tetap diberi bobot berdasarkan jumlah jawaban aktual.
    ''')
    code('''
    import os, sys, json, subprocess, hashlib, html, zipfile
    from pathlib import Path
    from IPython.display import display, HTML, Code

    REPO_URL = 'https://github.com/MalikHariyanto/Automated_Essay_Scoring.git'
    BRANCH = 'codex/aes-learning'
    IN_COLAB = 'google.colab' in sys.modules or Path('/content').is_dir()
    ROOT = Path('/content/Automated_Essay_Scoring') if IN_COLAB else Path.cwd()
    if not IN_COLAB:
        while not (ROOT/'src/aes').is_dir() and ROOT != ROOT.parent:
            ROOT = ROOT.parent
    if not (ROOT/'.git').exists():
        if ROOT.exists() and any(ROOT.iterdir()):
            raise RuntimeError('Folder target berisi file lain. Pilih folder checkout terpisah.')
        subprocess.run(['git', 'clone', '--branch', BRANCH, '--single-branch', REPO_URL, str(ROOT)], check=True)
    elif IN_COLAB:
        branch = subprocess.check_output(['git','branch','--show-current'], cwd=ROOT, text=True).strip()
        dirty = subprocess.check_output(['git','status','--porcelain'], cwd=ROOT, text=True).strip()
        if branch != BRANCH or dirty:
            raise RuntimeError('Checkout mempunyai perubahan/branch lain. Simpan dahulu dan gunakan runtime baru; kode tidak ditimpa.')
        subprocess.run(['git','pull','--ff-only','origin',BRANCH], cwd=ROOT, check=True)

    DATASET_NAME = 'fix_dataset_300.xlsx'
    SHEET = 'Sheet1'
    SCORE_MIN, SCORE_MAX = 0, 100  # Rentang dinyatakan notebook unggahan; bukan min/max test.
    EPOCHS, FOLDS = 5, 5
    MICRO_BATCH, EFFECTIVE_BATCH = 1, 8
    DEVICE = 'auto'
    RUN_TRAINING = True
    RUN_VECTOR_LAB = True
    VECTOR_SOURCE = 'pretrained'  # Atau S1/S3 setelah training untuk RoBERTa fine-tuned.
    NEW_ANSWER = ''  # Kosong: gunakan satu jawaban held-out untuk demonstrasi scoring/vektor.
    FOCUS_WORD = ''  # Kosong: pilih otomatis kata yang berulang dalam jawaban tersebut.
    # Untuk audit lokal otomatis saja; tidak membuat angka/model pengganti.
    AUDIT_ONLY = os.environ.get('AES300_AUDIT_ONLY') == '1'
    if AUDIT_ONLY:
        RUN_TRAINING = RUN_VECTOR_LAB = False
    print('Root:', ROOT, '| Python:', sys.version.split()[0], '| Mode:', 'AUDIT ONLY' if AUDIT_ONLY else 'training guru')
    print('Commit source:', subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip())
    ''')
    md('''
    ## 2 · Instalasi terpisah dan impor

    Worker adalah proses Python baru yang mengerjakan model. Ini mencegah konflik NumPy yang sudah terlanjur diimpor kernel Colab. Setup mempertahankan PyTorch GPU bawaan runtime; tidak menggantinya dengan wheel CPU. Checkpoint asli baru diunduh pada tahap encoder.
    ''')
    code('''
    PYTHON = os.environ.get('AES300_PYTHON', '')
    if not PYTHON:
        subprocess.run([sys.executable, str(ROOT/'scripts/setup_dataset300.py')], cwd=ROOT, check=True)
        PYTHON = str(ROOT/'outputs/dataset300_env/bin/python')
    CHILD_ENV = {**os.environ, 'PYTHONPATH': str(ROOT/'src'),
                 'HF_HOME': str(ROOT/'outputs/cache/huggingface'),
                 'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2', 'TOKENIZERS_PARALLELISM': 'false'}
    subprocess.run([PYTHON, '-c',
        "import torch,numpy,pandas,sklearn,openpyxl,transformers; print('Import OK; torch',torch.__version__,'CUDA',torch.cuda.is_available())"],
        cwd=ROOT, env=CHILD_ENV, check=True)
    ''')
    md('''
    ## 3 · Baca dataset asli

    Upload Excel yang dikirim tadi. Notebook mengharapkan **300 baris**, tab `Sheet1`, dan enam header: `Email address`, `response_number`, `response_text`, `soal`, `jawaban_referensi`, `rater`. Nilai/teks kosong menyebabkan error; tidak diisi atau diberi label otomatis.

    Input model hanya `response_text`. Email dipakai untuk kelompok pembagian data; pertanyaan dan jawaban referensi tidak dikirim ke encoder pada eksperimen ini.
    ''')
    code('''
    PRIVATE_INPUT = ROOT/'data/dataset300'
    PRIVATE_INPUT.mkdir(parents=True, exist_ok=True)
    DATA_PATH = Path(os.environ.get('AES300_DATASET', str(PRIVATE_INPUT/DATASET_NAME))).resolve()
    if not DATA_PATH.exists():
        if not IN_COLAB:
            raise FileNotFoundError('Letakkan Excel asli di '+str(DATA_PATH))
        from google.colab import files
        previous = Path.cwd()
        try:
            os.chdir(PRIVATE_INPUT)
            uploaded = files.upload()
        finally:
            os.chdir(previous)
        if DATASET_NAME not in uploaded:
            raise ValueError('Upload file dengan nama '+DATASET_NAME)
    OUTPUT = ROOT/'outputs/dataset300'
    OUTPUT.mkdir(parents=True, exist_ok=True)
    CFG = dict(dataset=str(DATA_PATH), output=str(OUTPUT), sheet=SHEET, expected_rows=300,
               score_min=SCORE_MIN, score_max=SCORE_MAX, epochs=EPOCHS, folds=FOLDS,
               micro_batch=MICRO_BATCH, effective_batch=EFFECTIVE_BATCH,
               device=DEVICE, vector_source=VECTOR_SOURCE, new_answer=NEW_ANSWER, focus_word=FOCUS_WORD)
    CONFIG_PATH = OUTPUT/'notebook_config.json'
    CONFIG_PATH.write_text(json.dumps(CFG, ensure_ascii=False, indent=2))

    def step(name):
        print('Mulai tahap:', name, flush=True)
        subprocess.run([PYTHON, str(ROOT/'scripts/dataset300_step.py'), name, '--config', str(CONFIG_PATH)],
                       cwd=ROOT, env=CHILD_ENV, check=True)

    def load_result(name):
        pointer = json.loads((OUTPUT/'active_run.json').read_text())
        return json.loads((Path(pointer['folder'])/name).read_text())

    def table(rows):
        if not rows:
            print('Tidak ada baris untuk kondisi ini.')
            return
        keys = list(dict.fromkeys(k for row in rows for k in row))
        render = lambda x: html.escape(json.dumps(x, ensure_ascii=False) if isinstance(x,(list,dict)) else str(x))
        content = '<table><thead><tr>'+''.join('<th>'+render(k)+'</th>' for k in keys)+'</tr></thead><tbody>'
        content += ''.join('<tr>'+''.join('<td>'+render(row.get(k,''))+'</td>' for k in keys)+'</tr>' for row in rows)
        display(HTML('<div style="overflow:auto;max-height:480px">'+content+'</tbody></table></div>'))

    def show_code(relative, start, stop=None):
        source = (ROOT/relative).read_text()
        begin = source.index(start)
        end = source.index(stop, begin+len(start)) if stop else len(source)
        display(Code(source[begin:end], language='python'))
    print('File privat tersedia; SHA256:', hashlib.sha256(DATA_PATH.read_bytes()).hexdigest())
    ''')
    md('''
    ## 4 · Audit label: bisakah dibuat empat rubrik?

    **Belum bisa dilatih/dievaluasi secara sah untuk empat dimensi dari satu nilai ini.** Linear layer empat output mudah dibuat, tetapi satu nilai guru tidak memberi tahu kontribusi isi, organisasi, bahasa, dan mekanik masing-masing. Menyalin nilai menjadi empat atau membagi rata tidak menambah informasi.

    Solusi untuk penelitian empat dimensi: definisikan deskriptor rubrik dan minta penilai manusia memberi empat label terpisah pada setiap jawaban; sebaiknya ada pemeriksaan kesepakatan antarpenilai. Label AI hanya boleh menjadi usulan terpisah, bukan diam-diam disebut skor guru.

    Eksperimen ini belajar **satu skor yang sudah tersedia**. Dataset membahas metodologi penelitian; file saja tidak membuktikan respondennya siswa kelas XI. Rentang 0–100 mengikuti notebook unggahan, sedangkan skor teramati bukan batas rubrik baru.
    ''')
    code('''
    step('audit')
    audit = load_result('dataset_audit.json')
    table([audit])
    preview = load_result('preview.json')
    display(HTML('<h4>Satu jawaban held-out · hanya di runtime privat</h4><pre style="white-space:pre-wrap;max-height:280px;overflow:auto">'+html.escape(preview['text'])+'</pre>'))
    print('Skor guru contoh:', preview['teacher_score'], '| Tidak digunakan untuk memilih epoch/model.')
    ''')
    md('''
    ## 5 · Pembagian data dan pencegahan kebocoran

    Target pembagian **80% baris development dan 20% test**, dengan kelompok tetap utuh. Sebelum latihan, dari 128 kandidat GroupShuffleSplit deterministik dipilih proporsi baris terdekat target berdasarkan ukuran kelompok saja, **tanpa melihat nilai guru atau hasil model**. Di development, lakukan **5-fold GroupKFold**: empat fold melatih dan satu memvalidasi, bergiliran. Tidak diperlukan set validasi ketiga di luar CV. Fold validasi memilih epoch; test hanya mengukur hasil final.

    Email yang sama tidak menyeberang split. Jawaban identik setelah normalisasi audit juga menyatukan kelompok email yang berbeda, agar duplikat tidak bocor. Kelompok tidak dipecah demi memaksa rasio; bila 80:20 tidak memungkinkan, tampilkan rasio terdekat yang ditemukan. Semua skenario memakai manifest yang sama. Ini mengukur kelompok baru pada kumpulan pertanyaan yang sama, bukan kemampuan pada pertanyaan baru.
    ''')
    code('''
    split = load_result('split_manifest.json')
    table([{k:split[k] for k in ('development_rows','test_rows','development_groups','test_groups','actual_test_row_fraction','split_sha256')}])
    table([dict(fold=f['fold'], training_rows=len(f['train']), validation_rows=len(f['validation']), test_rows=split['test_rows']) for f in split['folds']])
    show_code('src/aes/teacher_score/data.py', 'def make_split', 'def run_folder')
    ''')
    md('''
    ## 6 · Tokenisasi, window, dan label loss

    Token bukan selalu satu kata: tokenizer dapat memecah kata menjadi subword. RoBERTa punya maksimum 512 token termasuk `<s>` dan `</s>` (**510 token isi + 2 special**). Jawaban panjang memakai window dengan overlap 128 token; CLS setiap window dirata-ratakan. Token terakhir tetap tercakup.

    BGE memakai tokenizer sendiri dan batas eksperimen **1024 token termasuk special**, walaupun checkpoint mendukung hingga 8192. Pemotongan dilaporkan agar tidak tersembunyi. Jumlah token RoBERTa tidak bisa dipakai untuk menebak jumlah token BGE.

    Untuk MSE training saja, label guru diubah `(nilai−0)/100`. Setelah forward, output dikembalikan ke poin 0–100 sebelum RMSE, Pearson, dan scoring. RMSE bukan dibatasi 0–1. Skor mentah dapat keluar rentang dan tetap dilaporkan.
    ''')
    code('''
    show_code('src/aes/windows.py', 'def content_windows')
    ''')
    md('''
    ## 7 · Model S1, S2, S3: bagian mana yang berubah?

    | Skenario | Encoder dan proses | Head satu nilai | Yang belajar |
    |---|---|---|---|
    | S1 | RoBERTa Indonesia → mean CLS window 768 | dropout → linear 768→1 | RoBERTa + head |
    | S2 | BGE-M3 → CLS 1024 → normalisasi L2 | MLP 1024→256→1 | MLP saja; BGE frozen |
    | S3 | RoBERTa 768 + BGE 1024 diproyeksikan ke 256 | gabung 768+256=1024 → linear 1 | RoBERTa, proyeksi, head; BGE frozen |

    **Fine-tuned:** parameter encoder boleh mendapat gradien dan diperbarui optimizer. **Frozen:** bobot encoder tetap; MLP/proyeksi setelahnya masih bisa belajar. BGE diekstrak sekali dalam `eval()` dan `no_grad()` lalu disimpan sebagai cache privat. Ini berbeda dari notebook BGE unggahan yang memperbarui seluruh BGE; pilihan S2/S3 mengikuti skenario frozen yang sebelumnya lo tentukan.

    RoBERTa menggunakan `flax-community/indonesian-roberta-base`; BGE menggunakan `BAAI/bge-m3`. Revision kedua checkpoint dikunci dalam konfigurasi efektif supaya percobaan bisa ditelusuri.
    ''')
    code('''
    show_code('src/aes/teacher_score/models.py', 'def encoder', 'def tokenizer')
    show_code('src/aes/teacher_score/models.py', 'class SingleScoreModel', 'class Engine')
    ''')
    md('''
    ## 8 · Empat metrik yang diminta

    **RMSE ↓**: besar kesalahan dalam poin skor asli; selisih besar dihukum lebih kuat. **Pearson ↑**: kesamaan arah/peringkat linear, bukan ketepatan nilai. Prediksi selalu 10 poin terlalu tinggi masih bisa mempunyai Pearson tinggi.

    **QWK ↑** dan **Gwet AC2 kuadratik ↑**: kesepakatan kategori dengan penalti kuadratik untuk jarak skor, memakai koreksi peluang yang berbeda. Untuk keduanya saja, salinan prediksi dipotong ke 0–100 lalu dibulatkan ke integer terdekat; tepat `.5` memilih yang lebih rendah. Skor mentah tidak ditimpa. Semua **101 kategori 0…100** dipertahankan, termasuk kategori tidak teramati. Mengganti daftar kategori mengubah definisi evaluasi, terutama AC2; AC2 tinggi sendiri bukan bukti model bagus. Baca keempat metrik bersama.

    Nilai tidak terdefinisi (misalnya Pearson saat prediksi konstan) ditulis `None` dengan alasan, bukan dipalsukan menjadi 0. Tidak ada tolerance/adjacent agreement tambahan pada eksperimen ini.
    ''')
    code('''
    show_code('src/aes/teacher_score/models.py', 'def metrics', 'def bge_features')
    ''')
    md('''
    ## 9 · Optimizer, learning rate, batch, dan pemilihan epoch

    Encoder RoBERTa: LR `2e-5`; head/proyeksi: LR `1e-3`, optimizer AdamW, MSE, warmup 10%, gradient clip 1. Scheduler mengurangi LR sampai akhir latihan. `requires_grad` yang memilih bobot boleh belajar; pengelompokan optimizer mengatur kecepatan belajarnya.

    Per fold, simpan epoch dengan RMSE validasi terkecil. Untuk refit seluruh development, gunakan **ceil(median epoch terbaik lima fold)**. Aturan ini ditetapkan sebelum melihat test. Ketiga skenario diselesaikan dan konfigurasi dikunci sebelum test dibuka. Tabel CV dipakai membandingkan skenario; hindari tuning lagi berdasarkan tabel test.
    ''')
    code('''
    show_code('src/aes/teacher_score/experiment.py', 'def fit', 'def save_model')
    ''')
    md('''
    ## 10 · Ekstraksi BGE frozen

    Fitur BGE hanya bergantung pada teks dan checkpoint pretrained; label tidak dipakai di sini. Karena tidak ada transformasi yang di-fit pada label/data gabungan, fitur frozen boleh dihitung untuk semua jawaban, sementara pemilihan dan training tetap memakai indeks development. Hash sumber, teks bersih, revision, dan batas token memisahkan cache.
    ''')
    code('''
    if RUN_TRAINING:
        step('encode')
        table([load_result('bge_encoding.json')])
    else:
        print('Training dinonaktifkan: checkpoint/fit tidak dijalankan pada tahap ini.')
    ''')
    md('''
    ## 11 · Latih S1 → S2 → S3 dengan 5-fold CV, lalu refit

    Cell ini bisa lama: 3 × 5 fold + 3 refit. Progress menampilkan skenario, fold, epoch, MSE training, dan RMSE validasi. Jika runtime terputus, hasil fold yang sudah lengkap dapat dilanjutkan dari folder privat yang sama. Penyimpanan Colab `/content` hilang saat runtime dihapus: bila ingin resume antarsesi, pindahkan `OUTPUT` pada cell input ke folder Drive privat sebelum mulai.

    Folder eksperimen dibedakan hash dataset, konfigurasi, dan source model. Mengulang cell setelah selesai memakai report/checkpoint yang cocok; tidak menguji test lagi. Jangan mengganti konfigurasi sesudah melihat test untuk mengejar angka terbaik. Jika ingin pengembangan baru, gunakan protokol evaluasi baru yang jelas.
    ''')
    code('''
    if RUN_TRAINING:
        step('train')
    else:
        print('BELUM DILATIH pada eksekusi ini. Tidak ada angka penelitian pengganti.')
    ''')
    md('''
    ## 12 · Evaluasi: bedakan CV dan test akhir

    Tabel pertama menunjukkan fold development untuk memilih skenario; ringkasan mean ± SD menghitung variasi antar-fold (bukan interval kepercayaan). Tabel kedua menunjukkan tiga model final pada **holdout yang sama**. QWK/AC2/Pearson lebih tinggi umumnya lebih baik; RMSE lebih rendah lebih baik. Hasil buruk tetap ditampilkan apa adanya.
    ''')
    code('''
    RUN_FOLDER = Path(json.loads((OUTPUT/'active_run.json').read_text())['folder'])
    if (RUN_FOLDER/'final_report.json').exists():
        report = load_result('final_report.json')
        print('CV development — dipakai untuk pemilihan:')
        table(report['cv'])
        import statistics
        summary = []
        for scenario in ('S1','S2','S3'):
            row = {'Skenario':scenario}
            for metric in ('QWK','AC2','RMSE','Pearson'):
                values = [r[metric] for r in report['cv'] if r['Skenario']==scenario and r[metric] is not None]
                row[metric+' mean ± SD'] = (f'{statistics.mean(values):.4f} ± {statistics.stdev(values):.4f}' if len(values)>1 else 'undefined/terlalu sedikit fold')
            summary.append(row)
        table(summary)
        print('TEST AKHIR — tidak dipakai memilih epoch/hiperparameter:')
        table(report['rows'])
        print('Epoch refit:', report['selected_epochs'], '| Baris test:', report['test_rows'])
        print('Label: satu skor guru, bukan empat rubrik.')
    else:
        print('Belum ada hasil training lengkap. Tabel tidak diisi dengan nilai contoh.')
    ''')
    md('''
    ## 13 · Scoring/prediksi satu jawaban dengan ketiga model final

    `NEW_ANSWER` kosong berarti contoh held-out yang diaudit tadi. Isi teks baru di konfigurasi sebelum menjalankan ulang jika ingin scoring jawaban lain. Proses prediksi tidak melatih model lagi dan tidak memerlukan skor guru baru. Skor mentah dan salinan kategori ditampilkan terpisah; keluaran ini satu skor, bukan empat dimensi.
    ''')
    code('''
    if RUN_TRAINING and (RUN_FOLDER/'final_report.json').exists():
        step('predict')
        table(load_result('one_answer_prediction.json')['rows'])
    else:
        print('Prediksi menunggu tiga model final yang sudah dilatih.')
    ''')
    md('''
    ## 14 · Satu jawaban → vektor per token, per kata, per kalimat

    Default `VECTOR_SOURCE='pretrained'`: bandingkan sifat encoder asli sebelum fine-tuning AES. Pilih `S1` atau `S3` untuk menggunakan RoBERTa hasil latihan; BGE tetap pretrained frozen. Keduanya `eval()`/tanpa gradien agar dropout tidak menambahkan acak.

    **Vektor awal token** adalah lookup dari tabel embedding berdasarkan ID token, sebelum konteks/posisi. **Vektor kata awal** merata-ratakan subword kata tersebut. **Vektor kata kontekstual** merata-ratakan subword dari hidden state terakhir setelah membaca jawaban (overlap window dirata-ratakan per token dahulu). Penentuan kata memakai offset karakter; fast/slow tokenizer harus menghasilkan ID identik, kalau tidak proses berhenti.

    **Vektor kalimat** dibuat dengan membaca setiap kalimat secara terpisah, mengambil CLS (mean CLS window untuk RoBERTa, CLS + L2 untuk BGE). Ini bukan rata-rata vektor kata. Pemisah kalimat berbasis tanda baca sederhana; singkatan mungkin terpisah. BGE yang melewati batas token diberi laporan coverage; vektor kata yang tidak tercakup adalah `NaN`, bukan angka karangan.

    RoBERTa mempunyai 768 koordinat, BGE 1024. **Jangan menghitung cosine langsung antarmodel**: dimensinya dan ruang koordinatnya berbeda. Bandingkan kemunculan kata berulang di dalam model masing-masing.
    ''')
    code('''
    show_code('src/aes/teacher_score/vectors.py', 'def token_vectors', 'def word_vectors')
    if RUN_VECTOR_LAB:
        step('vectors')
        vector_report = load_result('vectors/'+VECTOR_SOURCE+'/report.json')
        for role in ('roberta','bge'):
            result = vector_report[role]
            print(role.upper(), '| sumber:', result['source'])
            print('Shape kata:',result['word_shape'],'| kalimat:',result['sentence_shape'],'| token tak tercakup:',result['uncovered_tokens'])
            print('Token/ID (cuplikan):'); table(result['tokens'][:20])
            print('Vektor kata (5 koordinat pertama):'); table(result['words'][:12])
            print('Vektor kalimat (5 koordinat pertama):'); table(result['sentences'])
            print('Perbandingan kata berulang:'); table(result['repeated_words'])
        print(vector_report['interpretation'])
    else:
        print('Laboratorium vektor dinonaktifkan; tidak ada hasil encoder yang diklaim.')
    ''')
    md('''
    ## 15 · Cara membaca hasil kata berulang dan menyimpan vektor lengkap

    `same_token_ids=True` dan selisih vektor awal 0 berarti subword sama mengambil isi tabel embedding yang sama. `contextual_max_abs_difference>0` membuktikan vektor akhir berubah pada dua kemunculan tersebut; itu bisa dipengaruhi konteks **dan posisi**. Bila ID berbeda, ejaan yang terlihat sama belum tentu mempunyai tokenisasi sama (misalnya penanda spasi `Ġ`, huruf besar, atau pemecahan subword).

    Cosine dekat 1 berarti arah vektor mirip, tidak otomatis identik. Kolom selisih maksimum memeriksa beda koordinat; `allclose` memakai toleransi numerik `1e-6`. Ini bukan tolerance agreement skor AES. Tidak adanya kata berulang akan menghasilkan tabel kosong dengan penjelasan; tidak dibuat jawaban siswa pengganti.

    File NPZ menyimpan **semua koordinat**, bukan hanya lima yang ditampilkan. Keys: `token_initial`, `token_contextual`, `token_ids`, `token_offsets`, `token_coverage`, `word_initial`, `word_contextual`, `sentence_cls`. CSV metadata menghubungkan baris vektor dengan kata/kalimat. Semuanya privat.
    ''')
    code(r'''
    if RUN_VECTOR_LAB:
        print('Contoh kode membaca vektor lengkap pada worker:')
        snippet = ['import numpy as np', 'from pathlib import Path',
                   'folder = Path('+repr(str(RUN_FOLDER/'vectors'/VECTOR_SOURCE))+')',
                   "for model in ('roberta','bge'):",
                   "    with np.load(folder/(model+'_vectors.npz'), allow_pickle=False) as data:",
                   "        print(model, data['word_contextual'].shape)",
                   "        print(data['word_contextual'][0])  # seluruh koordinat kata pertama"]
        display(Code('\n'.join(snippet), language='python'))
        archive = OUTPUT/'hasil_vektor_privat.zip'
        with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as target:
            for file in (RUN_FOLDER/'vectors'/VECTOR_SOURCE).iterdir():
                if file.is_file():
                    target.write(file, arcname=file.name)
        print('Arsip vektor privat:', archive)
        if IN_COLAB:
            from google.colab import files
            files.download(str(archive))
    ''')
    md('''
    ## 16 · Simpan hasil evaluasi privat

    Arsip berikut mencakup report, konfigurasi, prediksi test, dan riwayat CV; dapat mengandung nilai individual. Jangan unggah ke repo publik. Bobot model besar dan cache tidak dimasukkan. Model final tetap di folder `S1/final.pt`, `S2/final.pt`, `S3/final.pt`; salin ke penyimpanan privat jika ingin digunakan lagi.

    Source penting: `src/aes/teacher_score/data.py` (input/split), `models.py` (encoder/head/metrik), `experiment.py` (training/CV/prediksi), `vectors.py` (per-kata/per-kalimat). Dokumentasi lengkap: `docs/dataset_guru_300.md`. Angka unit test adalah bukti mekanisme program, bukan akurasi AES.
    ''')
    code('''
    if (RUN_FOLDER/'final_report.json').exists():
        archive = OUTPUT/'hasil_evaluasi_privat.zip'
        with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as target:
            for file in RUN_FOLDER.rglob('*'):
                if file.is_file() and file.suffix in ('.json','.csv') and 'vectors' not in file.parts:
                    target.write(file, arcname=str(file.relative_to(RUN_FOLDER)))
        print('Arsip evaluasi privat:', archive)
        if IN_COLAB:
            from google.colab import files
            files.download(str(archive))
    else:
        print('Audit selesai; belum ada hasil evaluasi penelitian untuk diunduh.')
    ''')
    notebook = {'cells': cells, 'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                 'language_info': {'name': 'python', 'version': '3.12'}, 'colab': {'name': '03_AES_dataset_guru_300.ipynb'}},
                'nbformat': 4, 'nbformat_minor': 4}
    content = json.dumps(notebook, ensure_ascii=False, indent=1)+'\n'
    for folder in ('notebooks', 'aes-learning/notebooks'):
        path = ROOT/folder/'03_AES_dataset_guru_300.ipynb'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding='utf-8')
    return notebook


if __name__ == '__main__':
    notebook = build()
    print('Notebook publik tanpa output:', len(notebook['cells']), 'cell; dua salinan identik.')
