"""Build E35 as readable native Python cells, preserving the E30 training method."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
PIN = '7f3363643e6bf402a293de069bcfc23d85b19c89'
BASE = 'notebooks/experiments/AES_300_E30_E26_BiGRU_CoAttention_Pooling_GPU.ipynb'
NAME = 'AES_300_E35_E30_LayersLR5e5_Native_Learnable_GPU'
PLACEHOLDER = '__NOTEBOOK_CODE_SHA256__'

SETUP = '''# Jalankan pada runtime Colab baru, sebelum mengimpor pustaka numerik/model.
import sys
import subprocess
from importlib import metadata, util
from packaging.requirements import Requirement

# Pertahankan Torch/CUDA, NumPy, pandas, SciPy dan sklearn bawaan Colab.
# Mengganti pustaka numerik yang sudah dimuat kernel dapat menyebabkan konflik ABI.
CORE_REQUIREMENTS = {
    'torch': 'torch>=2.6', 'numpy': 'numpy>=1.26,<3',
    'pandas': 'pandas>=2,<3', 'scipy': 'scipy>=1.10,<2',
    'scikit-learn': 'scikit-learn>=1.2,<2',
}
for package, requirement in CORE_REQUIREMENTS.items():
    try:
        installed = metadata.version(package)
    except metadata.PackageNotFoundError as error:
        raise RuntimeError('Gunakan runtime Google Colab baru; pustaka dasar belum ada: '+package) from error
    if not Requirement(requirement).specifier.contains(installed):
        raise RuntimeError(requirement+' diperlukan; runtime ini memiliki '+installed+'. Pilih runtime Colab baru.')

# Tokenizers 0.19 tidak menyediakan wheel Python 3.13; tetap satu versi per runtime.
TRANSFORMERS_REQUIREMENT = 'transformers==4.48.3' if sys.version_info >= (3, 13) else 'transformers==4.44.2'
EXTRA_REQUIREMENTS = [TRANSFORMERS_REQUIREMENT, 'openpyxl>=3.1,<4']
to_install = []
for requirement in EXTRA_REQUIREMENTS:
    req = Requirement(requirement)
    try:
        installed = metadata.version(req.name)
    except metadata.PackageNotFoundError:
        installed = None
    if installed is None or not req.specifier.contains(installed):
        if req.name in sys.modules:
            raise RuntimeError(req.name+' sudah diimpor dengan versi berbeda. Restart session, lalu Run all dari atas.')
        to_install.append(requirement)
if to_install:
    # Kendala ini mencegah pip mengganti pustaka numerik yang digunakan kernel.
    from pathlib import Path
    constraints = Path('/tmp/aes35_native_constraints.txt')
    constraints.write_text(''.join(p+'=='+metadata.version(p)+'\\n' for p in CORE_REQUIREMENTS))
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--quiet',
                           '--constraint', str(constraints), *to_install])
print('Setup selesai. Semua cell berikutnya berjalan langsung di kernel notebook ini.')
print('Python:', sys.version.split()[0])
'''

MODEL_NAMES = '''# NAMA MODEL PRETRAINED — ubah di sini jika membuat eksperimen lain.
ROBERTA_CHECKPOINT = 'flax-community/indonesian-roberta-base'
BGE_CHECKPOINT = 'BAAI/bge-m3'

# Revision mengunci bobot dan tokenizer yang sama seperti E30.
ROBERTA_REVISION = 'cc5ecee7de8532880ad08d802292a6f42865bc83'
BGE_REVISION = '5617a9f61b028005a4858fdac845db406aefb181'

print('RoBERTa:', ROBERTA_CHECKPOINT, '| encoder ikut fine-tuning')
print('BGE-M3:', BGE_CHECKPOINT, '| encoder frozen; proyeksi/head tetap belajar')
'''

COLUMNS = '''# X terdiri atas TIGA kolom teks. Y berisi SATU skor guru.
QUESTION_COLUMN = 'soal'
REFERENCE_COLUMN = 'jawaban_referensi'
ANSWER_COLUMN = 'response_text'
TARGET_COLUMN = 'rater'
INPUT_COLUMNS = [QUESTION_COLUMN, REFERENCE_COLUMN, ANSWER_COLUMN]

# Kedua kolom ini hanya metadata/kelompok pemisahan data; tidak masuk encoder.
GROUP_COLUMN = 'Email address'
RESPONSE_NUMBER_COLUMN = 'response_number'
REQUIRED = [GROUP_COLUMN, RESPONSE_NUMBER_COLUMN, *INPUT_COLUMNS, TARGET_COLUMN]
if len(set(REQUIRED)) != len(REQUIRED):
    raise ValueError('Nama kolom input, target dan metadata harus berbeda.')

# Kosongkan untuk menggunakan pemilih file upload Colab.
DATASET_FILE = os.environ.get('AES300_DATASET', '')
SHEET_NAME = 'Sheet1'
EXPECTED_ROWS = 300
print('Kolom X:', INPUT_COLUMNS)
print('Kolom Y:', TARGET_COLUMN, '| skor guru 0–100')
'''

HYPERPARAMETERS = '''# HIPERPARAMETER PENELITIAN — resep E35, satu perubahan LR dari E30.
EPOCHS = 25
FOLDS = 5
EARLY_STOPPING_PATIENCE = 5
TRAIN_SEED = 42
SPLIT_SEED = 2026
TEST_FRACTION = 0.20
MICRO_BATCH = 1
EFFECTIVE_BATCH = 8

ENCODER_LR = 5e-6             # RoBERTa S1 dan S3.
S1_HEAD_LR = 1e-4             # Linear S1.
S2_MLP_LR = 1e-4              # MLP S2 tetap kontrol E30.
S3_MLP_LR = 1e-4              # MLP S3 tetap nilai asli E30.
S3_LAYERS_LR = 5e-5           # SATU perubahan: sequence_pool.* + roberta_norm.*.
PROJECTION_LR = 1e-4          # Proyeksi BGE projection.* tidak ikut diturunkan.

BIGRU_HIDDEN_PER_DIRECTION = 64   # Dua arah menghasilkan128fitur/token.
DROPOUT = 0.1                   # Dropout hidden MLP, mengikuti E30.
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.10
GRADIENT_CLIP = 1.0

ROBERTA_MAX_TOKENS = 512
BGE_MAX_TOKENS = 1024
WINDOW_OVERLAP = 128
CONTEXT_MAX_TOKENS = 192
QUESTION_MAX_TOKENS = 64
DEVICE = 'cuda'
REQUIRE_GPU = True
'''

FLAGS = '''# KONTROL EKSEKUSI — Run all menjalankan seluruh resep default.
# Untuk membaca/audit dataset saja: ubah keempat RUN_... menjadi False.
RUN_MODELS = True
RUN_TRAINING = True
RUN_WORD_VECTORS = True
RUN_FINAL_FEATURES = True
DOWNLOAD_WITH_COLAB_BROWSER = True

# OPSIONAL: prediksi jawaban baru setelah training; isi ketiganya bersama-sama.
# Jika kosong, bagian contoh menggunakan satu jawaban test yang sudah dipisahkan.
NEW_QUESTION = ''
NEW_REFERENCE = ''
NEW_ANSWER = ''
'''

CONFIG_APPLY = '''# cfg hanya mengumpulkan pilihan di atas agar dapat diteruskan ke fungsi Python.
cfg = config(dict(
    dataset=str(DATASET_PATH), output=str(ROOT/'outputs'),
    sheet=SHEET_NAME, expected_rows=EXPECTED_ROWS,
    score_min=0, score_max=100, seed=TRAIN_SEED, split_seed=SPLIT_SEED,
    test_fraction=TEST_FRACTION, epochs=EPOCHS, folds=FOLDS,
    micro_batch=MICRO_BATCH, effective_batch=EFFECTIVE_BATCH,
    early_stopping_patience=EARLY_STOPPING_PATIENCE,
    encoder_lr=ENCODER_LR, s3_encoder_lr=ENCODER_LR,
    head_lr=S1_HEAD_LR, s2_mlp_lr=S2_MLP_LR, s3_mlp_lr=S3_MLP_LR,
    s3_layers_lr=S3_LAYERS_LR, projection_lr=PROJECTION_LR,
    dropout=DROPOUT, weight_decay=WEIGHT_DECAY,
    warmup_ratio=WARMUP_RATIO, gradient_clip=GRADIENT_CLIP,
    roberta={'checkpoint':ROBERTA_CHECKPOINT, 'revision':ROBERTA_REVISION},
    bge={'checkpoint':BGE_CHECKPOINT, 'revision':BGE_REVISION},
    roberta_max_length=ROBERTA_MAX_TOKENS, bge_max_length=BGE_MAX_TOKENS,
    roberta_overlap=WINDOW_OVERLAP, bge_overlap=WINDOW_OVERLAP,
    s3_pooling='bigru_coattention_pooling', s3_sequence_hidden=BIGRU_HIDDEN_PER_DIRECTION,
    input_format='single_sequence', device=DEVICE,
    context_max_tokens=CONTEXT_MAX_TOKENS, question_max_tokens=QUESTION_MAX_TOKENS,
    new_answer=NEW_ANSWER, new_question=NEW_QUESTION, new_reference=NEW_REFERENCE,
    vector_source='pretrained', lesson_version='e35_native_layers_lr5e5_v1',
    source_columns={'question':QUESTION_COLUMN, 'reference':REFERENCE_COLUMN,
                    'answer':ANSWER_COLUMN, 'target':TARGET_COLUMN,
                    'group':GROUP_COLUMN, 'response_number':RESPONSE_NUMBER_COLUMN},
    runtime_versions={p:version(p) for p in ('torch','transformers','numpy','pandas','scikit-learn')},
))
torch.set_num_threads(2)
gpu = check_gpu(require_gpu=REQUIRE_GPU)
print('Perangkat:', gpu['name'])
display(pd.DataFrame([
    {'Kelompok':'RoBERTa S1/S3', 'LR':ENCODER_LR},
    {'Kelompok':'Linear S1', 'LR':S1_HEAD_LR},
    {'Kelompok':'MLP S2', 'LR':S2_MLP_LR},
    {'Kelompok':'MLP S3', 'LR':S3_MLP_LR},
    {'Kelompok':'BiGRU/co-attention/pooling/norm S3', 'LR':S3_LAYERS_LR},
    {'Kelompok':'Proyeksi BGE S3', 'LR':PROJECTION_LR},
]))
'''

REFERENCES = '''Rujukan menjelaskan komponen; angka LR/hidden size/epoch pada E35 adalah pilihan ablasi proyek, bukan nilai optimal yang diwajibkan paper.

| Komponen | Sumber dan batas penerapan |
|---|---|
| RoBERTa | [Liu dkk.,2019](https://arxiv.org/abs/1907.11692); checkpoint Indonesia dikunci terpisah. |
| BGE-M3 | [Chen dkk.,2024](https://arxiv.org/abs/2402.03216); model embedding retrieval, digunakan sebagai fitur AES frozen. |
| GRU | [Cho dkk.,2014](https://arxiv.org/abs/1406.1078); implementasi dua arah di sini memakai `nn.GRU`. |
| BiGRU/attention untuk AES | [PaSTO-GNN,Alhothali2026](https://doi.org/10.3389/frai.2026.1842542), [Dong dkk.,2017](https://aclanthology.org/K17-1017/); proyek mengadaptasi token dan komponen berbeda. |
| Co-attention | [Zhang & Litman,2018](https://aclanthology.org/W18-0549/); paper tingkat kalimat, E35 mengadaptasi token respons–referensi. |
| LayerNorm / GELU | [Ba dkk.,2016](https://arxiv.org/abs/1607.06450), [Hendrycks & Gimpel,2016](https://arxiv.org/abs/1606.08415). |
| AdamW | [Loshchilov & Hutter](https://arxiv.org/abs/1711.05101); resep decay0,01 di sini mengikuti kontrol E30. |
'''


def build():
    raw = subprocess.check_output(['git', 'show', PIN+':'+BASE], cwd=ROOT)
    old = json.loads(raw)
    previous = old['metadata']['aes_standalone']['code_sha256']

    def original(index):
        s = old['cells'][index]['source']
        s = s if isinstance(s, str) else ''.join(s)
        return s.removeprefix('%%aes\n').replace(previous, PLACEHOLDER).replace(Path(BASE).stem, NAME)

    cells = []

    def md(title, body='', anchor=None):
        heading = (f'<a id="{anchor}"></a>\n' if anchor else '')+title+'\n\n'+body.strip()+'\n'
        cells.append({'cell_type':'markdown', 'metadata':{}, 'source':heading})

    def code(s, role):
        ast.parse(s)
        cells.append({'cell_type':'code', 'metadata':{'aes35_role':role},
                      'source':s.strip()+'\n', 'outputs':[], 'execution_count':None})

    def parts(s):
        lines = s.splitlines()
        result = []
        for node in ast.parse(s).body:
            start = min([node.lineno, *[d.lineno for d in getattr(node, 'decorator_list', [])]])-1
            result.append((node, '\n'.join(lines[start:node.end_lineno])))
        return result

    def definitions(index, descriptions=None):
        """Emit complete named functions/classes as real, inspectable Python cells."""
        for node, text in parts(original(index)):
            if not isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                continue
            description = (descriptions or {}).get(node.name,
                'Fungsi pendukung ini dipanggil oleh bagian eksekusi di bawah; menjalankan definisi belum menjalankan training.')
            md('### `'+node.name+'`', description)
            code(text, 'definition:'+node.name)

    def actions(index):
        return '\n\n'.join(text for node,text in parts(original(index))
                           if not isinstance(node,(ast.FunctionDef,ast.ClassDef)))

    md('# E35 · E30 dengan LR layer S3 5e-5 — Python biasa', '''
Notebook ini berjalan langsung di kernel Colab: tidak ada cell magic, worker tersembunyi, clone repo, mount Drive, atau file Python pendamping.

**Satu perubahan yang memengaruhi resep training dari E30:** LR `sequence_pool.*` dan `roberta_norm.*` menjadi5e-5. MLP S3 tetap1e-4, encoder RoBERTa5e-6, proyeksi BGE1e-4. BGE frozen; prediktor akhir MLP. Semua S1/S2/S3 dilatih baru.

**Cara menjalankan:** pilih T4 GPU, lalu Run all dan upload `fix_dataset_300.xlsx`. Seluruh hasil tersimpan lokal; unduh ZIP sebelum runtime berakhir. Empat tombol `RUN_...` dapat dimatikan bila hanya ingin membaca dataset.

**Peta belajar:** [nama model](#nama-model), [kolom X/Y](#kolom-xy), [hyperparameter](#hyperparameter), [X/y aktual](#xy-aktual), [arsitektur](#arsitektur), [training](#training), [evaluasi](#evaluasi), [unduh hasil](#unduh).

5fold CV, maksimal25epoch, early stopping patience5; refit memakai ceil median best epoch dan test terpisah. Tidak ada hasil penelitian E35 sebelum notebook benar-benar dilatih. Fungsi/kolom pemeriksaan cosine dihapus; vektor dan perbandingan komponen tetap tersedia.
''')
    md('## 1 · Setup pustaka pada kernel Colab', 'Jalankan sebelum bagian import. Pustaka numerik/CUDA bawaan dipertahankan; tidak ada lingkungan Python kedua.')
    code(SETUP, 'dependency_setup')
    md('## 2 · Import dan ruang variabel notebook', 'Semua fungsi dan variabel berada pada kernel yang sama. `SingleScoreModel` adalah kelas model; `ROBERTA_CHECKPOINT` nanti adalah nama bobot pretrained yang dimuat kelas tersebut.')
    imports = original(9).replace('# Hanya worker AES yang diubah; paket Colab dan operator torch tetap utuh.',
                                '# Backend teks dikonfigurasi pada kernel notebook; tidak memuat operator vision.')
    imports = imports.replace("# Hanya worker AES", "# Kernel notebook")
    code(imports+'\nfrom IPython.display import display, HTML\n', 'imports')
    md('## 3 · Nama model pretrained', 'Cari dan ubah nama checkpoint di cell ini. Revision mempertahankan versi model/tokenizer yang sama seperti kontrol E30.', 'nama-model')
    code(MODEL_NAMES, 'model_names')
    md('## 4 · Kolom input X, target y, dan file dataset', 'X memiliki tiga fitur teks; `rater` hanya y. Email/nomor respons hanya metadata. Mengubah nama kolom di sini benar-benar mengubah pembacaan Excel, bukan hanya label tampilan.', 'kolom-xy')
    code(COLUMNS, 'input_columns')
    md('## 5 · Hyperparameter E35', 'MLP tetap1e-4. Kelompok layer S3 mencakup BiGRU, proyeksi referensi, matching, pooling, proyeksi ringkasan, dan norm RoBERTa. Proyeksi BGE berada pada kelompok berbeda. Jangan memilih angka dari hasil test.', 'hyperparameter')
    code(HYPERPARAMETERS, 'hyperparameters')
    md('## 6 · Bagian yang dijalankan dan contoh jawaban baru', 'Jika Run all dengan nilai bawaan, notebook menjalankan training. Prediksi baru memerlukan ketiga teks, tanpa nilai guru.')
    code(FLAGS, 'runtime_flags')
    md('## 7 · Lokasi lokal dan upload Excel', 'Kode pemilih file berada langsung di sini. File sumber disalin, tidak ditimpa. Workspace dan checkpoint E35 terpisah dari E30/E33/E34.')
    storage_node = next(text for node,text in parts(original(3))
                        if isinstance(node,ast.FunctionDef) and node.name=='resolve_dataset_upload')
    storage = f'''IN_COLAB = 'google.colab' in sys.modules or Path('/content').is_dir()
default_workspace = '/content/{NAME}' if IN_COLAB else str(Path.cwd()/'outputs/{NAME}')
ROOT = Path(os.environ.get('AES300_WORKSPACE', default_workspace)).expanduser().resolve()
if ROOT == Path('/content/drive') or Path('/content/drive') in ROOT.parents:
    raise ValueError('Pilih penyimpanan runtime lokal, bukan Google Drive.')
ROOT.mkdir(parents=True, exist_ok=True)
'''
    code(storage+'\n'+storage_node+'''\n
DATASET_PATH = resolve_dataset_upload(ROOT, configured=DATASET_FILE, browser_upload=IN_COLAB)
print('Dataset:', DATASET_PATH)
print('Workspace:', ROOT)
''', 'storage_and_upload')

    md('## 8 · Pemeriksaan konfigurasi dan perangkat', 'Nilai bawaan di bawah berasal dari variabel yang sudah terlihat di bagian3–6; nilai itu dikumpulkan ke dictionary `cfg`.')
    defaults = original(10)
    mapping = {
        "'sheet': 'Sheet1', 'expected_rows': 300":"'sheet': SHEET_NAME, 'expected_rows': EXPECTED_ROWS",
        "'test_fraction': .2, 'folds': 5, 'seed': 42, 'split_seed': 2026":"'test_fraction': TEST_FRACTION, 'folds': FOLDS, 'seed': TRAIN_SEED, 'split_seed': SPLIT_SEED",
        "'epochs': 25, 'early_stopping_patience': 5, 'effective_batch': 8, 'micro_batch': 1":"'epochs': EPOCHS, 'early_stopping_patience': EARLY_STOPPING_PATIENCE, 'effective_batch': EFFECTIVE_BATCH, 'micro_batch': MICRO_BATCH",
        "'encoder_lr': 5e-6, 's3_encoder_lr': 5e-6, 'projection_lr': 1e-4, 'head_lr': 1e-4, 'weight_decay': .01":"'encoder_lr': ENCODER_LR, 's3_encoder_lr': ENCODER_LR, 'projection_lr': PROJECTION_LR, 'head_lr': S1_HEAD_LR, 'weight_decay': WEIGHT_DECAY",
        "'warmup_ratio': .1, 'gradient_clip': 1., 'dropout': .1":"'warmup_ratio': WARMUP_RATIO, 'gradient_clip': GRADIENT_CLIP, 'dropout': DROPOUT",
        "'roberta_max_length': 512, 'roberta_overlap': 128":"'roberta_max_length': ROBERTA_MAX_TOKENS, 'roberta_overlap': WINDOW_OVERLAP",
        "'bge_max_length': 1024, 'bge_overlap': 128, 'bge_batch': 4, 'device': 'cuda'":"'bge_max_length': BGE_MAX_TOKENS, 'bge_overlap': WINDOW_OVERLAP, 'bge_batch': 4, 'device': DEVICE",
        "'checkpoint': 'flax-community/indonesian-roberta-base'":"'checkpoint': ROBERTA_CHECKPOINT",
        "'revision': 'cc5ecee7de8532880ad08d802292a6f42865bc83'":"'revision': ROBERTA_REVISION",
        "'checkpoint': 'BAAI/bge-m3'":"'checkpoint': BGE_CHECKPOINT",
        "'revision': '5617a9f61b028005a4858fdac845db406aefb181'":"'revision': BGE_REVISION",
        "'context_max_tokens': 192, 'question_max_tokens': 64":"'context_max_tokens': CONTEXT_MAX_TOKENS, 'question_max_tokens': QUESTION_MAX_TOKENS",
        "'s3_sequence_hidden': 64":"'s3_sequence_hidden': BIGRU_HIDDEN_PER_DIRECTION",
        "'s2_mlp_lr': 0.0001, 's3_mlp_lr': 0.0001, 's3_layers_lr': 1e-4":"'s2_mlp_lr': S2_MLP_LR, 's3_mlp_lr': S3_MLP_LR, 's3_layers_lr': S3_LAYERS_LR",
    }
    for a,b in mapping.items():
        assert a in defaults, a
        defaults = defaults.replace(a,b)
    defaults = re.sub(r"REQUIRED = \[.*?\]\n", '', defaults, count=1, flags=re.S)
    code(defaults, 'configuration_defaults')

    # Preserve component comparison diagnostics, removing the unused similarity calculation.
    presentation = original(11)
    presentation = presentation.replace("          'initial_cosine':'Kemiripan Sebelum Konteks (Cosine)',\n", '')
    presentation = presentation.replace("          'contextual_cosine':'Kemiripan Setelah Konteks (Cosine)',\n", '')
    presentation = presentation.replace("          'Kemiripan Vektor Akhir':'Kemiripan Setelah Konteks (Cosine)','Apa Artinya':'Penjelasan'}", "          'Apa Artinya':'Penjelasan'}")
    presentation = presentation.replace("                    'Kemiripan Setelah Konteks (Cosine)','Penjelasan']", "                    'Penjelasan']")
    presentation = '\n'.join(l for l in presentation.splitlines() if "'Cosine mengukur" not in l)
    presentation = presentation.replace("row.get('contextual_cosine'),explanation]", 'explanation]')
    code(presentation, 'presentation_helpers')
    code('''def show(frame, title='', rows=8, columns=8):
    frame = frame if isinstance(frame, pd.DataFrame) else pd.DataFrame(frame)
    view = preview_table(frame.columns, frame.head(rows).to_dict('records'), title,
                         row_count=len(frame), max_rows=rows, max_columns=columns)
    display(HTML(render_table(view)))
''', 'definition:show')
    code(CONFIG_APPLY, 'configuration_apply')

    md('## 9 · Membaca dan memeriksa dataset', 'Teks dibersihkan dari karakter kontrol/Unicode; tidak melakukan stemming atau menulis ulang jawaban. Label kosong ditolak. Kelompok email/duplikat disimpan untuk menjaga pemisahan data.')
    data = original(14)
    for a,b in {"raw['rater']":"raw[TARGET_COLUMN]", "raw['response_text']":"raw[ANSWER_COLUMN]",
                "raw['soal']":"raw[QUESTION_COLUMN]", "raw['jawaban_referensi']":"raw[REFERENCE_COLUMN]",
                "raw['Email address']":"raw[GROUP_COLUMN]", "raw['response_number']":"raw[RESPONSE_NUMBER_COLUMN]",
                "('soal','jawaban_referensi')":"(QUESTION_COLUMN,REFERENCE_COLUMN)",
                "'target_columns': ['rater']":"'target_columns': [TARGET_COLUMN]",
                "'input_features': ['soal','response_text','jawaban_referensi']":"'input_features': list(INPUT_COLUMNS)"}.items():
        assert a in data,a
        data = data.replace(a,b)
    code(data, 'dataset_definitions')
    md('## 10 · Pencatatan Excel', 'Setiap tahap mencatat sumber dan hasil aktual. Ini pencatatan, bukan training tambahan.')
    definitions(15)
    md('## 11 · Variabel X dan y yang benar-benar dibaca', '''
`X` menampilkan ketiga fitur teks setelah cleaning. `y` menampilkan nilai guru. Fungsi training membawa data yang sama di dalam `bundle`, bersama ID dan kelompok split: `bundle.texts` berisi kolom jawaban, `contexts_from_bundle(bundle)` berisi soal/referensi, dan `bundle.y` adalah target. Tidak ada variabel di worker lain.
''', 'xy-aktual')
    read = original(16)
    read += '''

# INILAH X DAN Y — satu baris X dan satu nilai y untuk setiap jawaban.
X = pd.DataFrame({
    QUESTION_COLUMN: [row['question'] for row in bundle.rows],
    REFERENCE_COLUMN: [row['reference'] for row in bundle.rows],
    ANSWER_COLUMN: [row['text'] for row in bundle.rows],
}, columns=INPUT_COLUMNS)
y = pd.Series(bundle.y, index=X.index, name=TARGET_COLUMN)
assert TARGET_COLUMN not in X.columns and len(X) == len(y)
print('X:', X.shape, '| y:', y.shape)
display(X.head(3))
display(y.head(3).to_frame())
'''
    code(read, 'dataset_execute_xy')

    md('## 12 · Pisahkan development/test dan lima fold', 'Fold hanya pada development. Email/duplikat tidak menyeberang. Pemilihan split berdasarkan jumlah baris, tanpa membaca label. Test tidak memilih LR atau epoch.')
    definitions(18, {'make_split':'Menghasilkan daftar indeks development, test dan5pasangan train/validation yang sama untuk S1/S2/S3.',
                     'contexts_from_bundle':'Mengambil pasangan soal/referensi dari data X; tidak membaca nilai guru.'})
    code(actions(18), 'split_execute')
    md('## 13 · Tokenisasi, window dan posisi input', 'Tiga teks disusun dalam single_sequence. Posisi respons dan referensi ditandai terpisah. Mean antar-window tetap differentiable untuk jalur RoBERTa yang dilatih.')
    descriptions = {
        'mean_by_essay':'Merata-ratakan representasi antar-window milik jawaban yang sama; ini mean pooling antar-window, berbeda dari attention pooling token.',
        'pair_content_windows':'Menyusun token soal/referensi/jawaban, menyimpan mask posisi asli dan memastikan seluruh token jawaban tercakup.',
        'paired_token_vectors':'Inspeksi embedding lookup dan hidden state Transformer. Mode tanpa gradien, tidak memperbarui bobot.',
        'paired_pretrained_values':'Membuat fitur dari encoder pada tiga teks. BGE memakai rata-rata CLS window lalu normalisasi L2.',
    }
    definitions(20, descriptions)
    md('## 14 · Memuat tokenizer dan encoder pretrained', 'Nama checkpoint berasal dari bagian3. `encoder()` memuat objek model; `tokenizer()` memuat pemecah teks. Fungsi loader ini tidak langsung menjalankan supervised training.')
    definitions(21, {'encoder':'AutoModel.from_pretrained memuat bobot yang dikunci revision. RoBERTa trainable; BGE frozen. Dimensi768/1024 diperiksa.',
                     'tokenizer':'AutoTokenizer.from_pretrained memuat tokenizer milik checkpoint yang sama; skor guru tidak ditokenisasi.'})
    md('## 15 · Ekspor token dan cakupan konteks', 'Periksa token dari tiga input, token yang terpotong, dan posisi yang dipakai pooling.')
    definitions(22)
    code(actions(22), 'token_export_execute')
    md('## 16 · Vektor RoBERTa sebelum supervised training', 'Bahan pembanding fitur awal/akhir. Angka ini bukan prediksi penelitian sebelum model dilatih.')
    definitions(24)
    code(actions(24), 'pretrained_roberta_execute')
    md('## 17 · Fitur BGE-M3 frozen', 'Fitur1024 disimpan dalam cache berdasarkan teks/konteks/resep tokenizer, tanpa target. S2/S3 memakai fitur yang sama; layer setelah BGE tetap dilatih.')
    definitions(26)
    code(actions(26), 'bge_execute')

    md('## 18 · Inspeksi vektor kata dan kalimat', 'Tetap menyimpan seluruh komponen vektor, posisi token, dan pemeriksaan selisih komponen kata berulang. Hasil inspeksi bukan fitur tambahan ke model atau loss.')
    word = original(28)
    word = re.sub(r'\ndef cosine\(a, b\):\n.*?\n\n', '\n', word, count=1, flags=re.S)
    word = '\n'.join(l for l in word.splitlines() if not any(x in l for x in ["'initial_cosine':", "'contextual_cosine':"]))
    word = word.replace('jangan menghitung cosine langsung antarmodel.', 'bandingkan vektor hanya di ruang encoder yang sama.')
    for node,text in parts(word):
        if isinstance(node,(ast.FunctionDef,ast.ClassDef)):
            md('### `'+node.name+'`', 'Inspeksi memakai vektor aktual. Pemanggilan eksekusinya berada setelah semua fungsi bagian ini didefinisikan.')
            code(text,'definition:'+node.name)
        elif not isinstance(node,ast.If):
            code(text,'word_helpers')
    code('\n'.join(text for node,text in parts(word) if isinstance(node,ast.If)), 'word_execute')

    md('## 19 · Arsitektur S1/S2/S3', '''
S1: RoBERTa→linear. S2: BGE frozen→MLP. S3: token respons RoBERTa→BiGRU128; referensi RoBERTa→proyeksi128; co-attention→matching128→attention pooling→proyeksi768→LayerNorm. BGE1024→proyeksi256+LayerNorm. Concat1024→MLP256→satu skor.

BiGRU, co-attention, attention pooling dan MLP ditulis sebagai kelas/fungsi yang dapat dibaca langsung di bawah. Jangan menganggap nama checkpoint pretrained sama dengan nama kelas Python.
''', 'arsitektur')
    descriptions = {
        'answer_token_mask':'Mask memisahkan token jawaban dari soal/referensi/prefix/padding berdasarkan posisi tokenizer.',
        'AnswerBiGRUAttention':'Komponen dasar E26: GRU64perarah→128; additive attention128→64→1 merangkum token; proyeksi128→768. [GRU](https://arxiv.org/abs/1406.1078), [attention AES](https://aclanthology.org/K17-1017/).',
        'AnswerBiGRUCoAttentionPool':'Komponen E30/E35: scaled dot product/√128 dan softmax dua arah mencocokkan respons–referensi; matching640→128, lalu attention pooling. [Rujukan](https://aclanthology.org/W18-0549/) diadaptasi pada token.',
        'SingleScoreModel':'KELAS MODEL SKOR. Cari `self.head` untuk MLP, `self.projection` untuk proyeksi BGE, `self.sequence_pool` untuk BiGRU/co-attention/pooling, dan `forward()` untuk jalur fitur→skor.',
        'Engine':'Menyiapkan tensor dari teks/konteks, menjalankan model dan mengembalikan prediksi ke satuan nilai guru. Target tidak diberikan ke forward().',
    }
    definitions(30, descriptions)
    md('## 20 · Inisialisasi model dan spesifikasi', 'Setiap fold/refit membuat model baru. Seed direset pada awal fit. Ini tidak melanjutkan bobot dari fold sebelumnya.')
    definitions(31)
    code(original(32), 'architecture_export')
    md('## 21 · Metrik evaluasi', 'RMSE/Pearson memakai prediksi kontinu. QWK/AC2 memakai salinan kategori0–100, pembulatan tie ke bawah. Angka agreement bukan persentase jawaban tepat.')
    definitions(34)
    definitions(35)
    md('## 22 · Optimizer, loss MSE, early stopping dan checkpoint', '''
`optimizer_groups()` menentukan LR setiap kelompok tepat sekali. `fit()` membaca batch→forward→MSE→backward→gradient clipping→AdamW→scheduler. Target loss0–1; validasi memakai RMSE poin. Patience5 dihitung terhadap rekor RMSE validasi, lalu bobot terbaik dipulihkan. Lihat kode `loss = ...` di fungsi fit untuk peran y.
''', 'training')
    definitions(36, {'optimizer_groups':'S3 head1e-4, sequence_pool/roberta_norm5e-5, proyeksi BGE1e-4, encoder5e-6. S1/S2 tetap resep kontrol E30.',
                     'fit':'SATU latihan fold/refit. Fungsi ini menyimpan checkpoint progres beserta RNG untuk resume lokal; parameter dipilih melalui validation, bukan test.'})
    md('## 23 · Lima fold, refit dan final test', 'Setiap skenario memiliki5model CV baru dan1refit. Epoch refit=ceil median epoch terbaik5fold. Data test dievaluasi sesudah pemilihan model.', 'evaluasi')
    definitions(38)
    md('## 24 · Menyimpan dan menjalankan training', 'Fungsi ekspor di bawah mencatat hasil tahap. Cell terakhir bagian ini benar-benar memanggil `train_all(...)`; matikan RUN_TRAINING sebelum Run all jika hanya ingin audit data.')
    definitions(39)
    code(actions(39), 'training_execute')
    md('## 25 · Prediksi tiga teks baru', 'Gunakan NEW_QUESTION/NEW_REFERENCE/NEW_ANSWER dari bagian6. Jika dibiarkan kosong, notebook menampilkan contoh test yang sudah dipisahkan. Tidak ada training tambahan.')
    definitions(41)
    code(actions(41), 'prediction_execute')
    md('## 26 · Fitur model setelah training', 'Inspeksi checkpoint final memakai eval/no_grad.300baris fitur bukan300data test. Bobot attention dan proyeksi/MLP intermediate diekspor untuk dipelajari.')
    definitions(43)
    code(actions(43), 'final_features_execute')
    md('## 27 · Susun lima workbook hasil', 'Masing-masing fungsi laporan membaca artefak sesi yang sudah ada. Tidak melatih ulang dan tidak menimpa skor guru. Rumus/metode tetap sumber E30.')
    reports = original(45).replace("['Batas perbandingan','Jangan menghitung cosine langsung antarruang RoBERTa dan BGE. Vektor bukan skor siswa.']", "['Batas perbandingan','Bandingkan komponen hanya dalam ruang encoder yang sama. Vektor bukan skor siswa.']")
    for node,text in parts(reports):
        md('### `'+node.name+'`', 'Pembaca/penulis laporan aktual; data individual dan vektor disimpan hanya pada runtime/ZIP pribadi.')
        code(text,'definition:'+node.name)
    md('## 28 · Buat ZIP dan unduh hasil', 'ZIP berisi5Excel; checkpoint model tidak otomatis dimasukkan. Simpan ZIP sebelum menghapus runtime. File hasil mengandung jawaban/nilai pribadi, bukan source publik.', 'unduh')
    download = original(46)
    tree = ast.parse(download)
    call = tree.body[0].value
    assert isinstance(call, ast.Call) and isinstance(call.func,ast.Name) and call.func.id=='aes_cell'
    direct = ast.literal_eval(call.args[1])
    rest = '\n'.join(download.splitlines()[tree.body[0].end_lineno:])
    rest = rest.replace('DOWNLOAD_WITH_COLAB_BROWSER=True  # Unduh ZIP hasil otomatis pada web Colab.\n', '')
    code(direct+'\n'+rest, 'download_execute')
    md('## Rujukan dan batas kesimpulan', REFERENCES)

    fingerprint = hashlib.sha256('\n'.join(c['source'] for c in cells if c['cell_type']=='code').encode()).hexdigest()
    for c in cells:
        c['source'] = c['source'].replace(PLACEHOLDER, fingerprint)
    n = {'nbformat':4, 'nbformat_minor':5, 'metadata':copy.deepcopy(old['metadata']), 'cells':cells}
    n['metadata']['colab'] = {'name':NAME+'.ipynb','provenance':[]}
    n['metadata']['aes_standalone'] = {'code_sha256':fingerprint,'native_kernel':True,
        'base_commit':PIN,'base_path':BASE,'base_sha256':hashlib.sha256(raw).hexdigest()}
    n['metadata']['aes_experiment'].update(name='E35',base_commit=PIN,base_path=BASE,
        base_sha256=hashlib.sha256(raw).hexdigest(),s3='E30 + LR pooling_norm S3 5e-5',
        s2_mlp_lr=1e-4,s3_mlp_lr=1e-4,s3_layers_lr=5e-5,new_layer_lr=5e-5,
        native_kernel=True,reused_controls=[],inspection_cosine=False,
        treatment='s3_layers_lr only; native cells and removed inspection similarity do not change the prediction architecture')
    for i,c in enumerate(cells):
        c['id'] = 'e35-'+str(i).zfill(3)+'-'+hashlib.sha256(c['source'].encode()).hexdigest()[:8]
    codes = '\n'.join(c['source'] for c in cells if c['cell_type']=='code')
    for bad in ['%%aes','aes_cell(', 'WORKER_SOURCE', 'register_cell_magic', 'cosine']:
        assert bad not in codes, bad
    for c in cells:
        if c['cell_type'] == 'code':
            assert not any(isinstance(v,ast.Call) and isinstance(v.func,ast.Name)
                           and v.func.id in ('exec','eval') for v in ast.walk(ast.parse(c['source'])))
    output = json.dumps(n,ensure_ascii=False,indent=1)+'\n'
    for directory in ['notebooks/experiments','aes-learning/notebooks/experiments']:
        (ROOT/directory/(NAME+'.ipynb')).write_text(output)
    print(json.dumps({'name':NAME,'cells':len(cells),'code_cells':sum(c['cell_type']=='code' for c in cells),'code_sha256':fingerprint}))
    return n


if __name__ == '__main__':
    build()
