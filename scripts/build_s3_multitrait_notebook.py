"""Build the self-contained four-trait Colab notebook from the native E43 cells."""
import ast
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = 'AES_300_E43_E42_S3_MaskedMeanPooling_Native_Learnable_GPU.ipynb'
NAME = 'AES_S3_MultiTrait_3Sekolah.ipynb'
E2_NAME = 'AES_E2_S3_MultiTrait_TraitHeads_3Sekolah.ipynb'
E3_NAME = 'AES_E3_S3_MultiTrait_Dropout02_3Sekolah.ipynb'
E4_NAME = 'AES_E4_S3_MultiTrait_Dropout02_Sigmoid_3Sekolah.ipynb'
E5_NAME = 'AES_E5_S3_MultiTrait_InterTraitAttention_3Sekolah.ipynb'
EXPERIMENTS = Path('notebooks/experiments')
PLACEHOLDER = '__NOTEBOOK_CODE_SHA256__'


def replace(source, old, new):
    if source.count(old) != 1:
        raise ValueError(f'Expected one E43 source anchor: {old!r}')
    return source.replace(old, new)


def dataset_source(source):
    source = replace(source, 'header = [str(x).strip() if x is not None else None for x in next(data)]',
                     'header = [str(x).strip() if x is not None else None for x in next(data, ())]')
    source = source.replace('Enam header wajib', 'Header wajib')
    start = source.index('            if any(raw[k]')
    end = source.index('            for key in (QUESTION_COLUMN', start)
    source = source[:start] + '''            missing = [k for k in REQUIRED if raw[k] is None or not str(raw[k]).strip()]
            if missing:
                raise ValueError(f'Baris {excel_row}: kolom wajib kosong: '+', '.join(missing))
            scores = []
            for _, column, low, high in TRAITS:
                score = raw[column]
                if (isinstance(score, bool) or not isinstance(score, (int, float))
                    or not math.isfinite(score) or int(score) != score or not low <= score <= high):
                    raise ValueError(f'Baris {excel_row}: {column} harus bilangan bulat {low}–{high}.')
                scores.append(int(score))
''' + source[end:]
    source = replace(source, 'if not isinstance(raw[ANSWER_COLUMN], str):',
                     'if not isinstance(raw[ANSWER_COLUMN], str) or not clean(raw[ANSWER_COLUMN]).strip():')
    source = replace(source, "'text': clean(raw[ANSWER_COLUMN]), 'raw_text': raw[ANSWER_COLUMN],",
                     "'text': clean(raw[ANSWER_COLUMN]),")
    source = replace(source, "'score': int(score)", "'scores': scores")
    start = source.index("    y = np.asarray([r['score']")
    end = source.index('    return Bundle', start)
    source = source[:start] + '''    y = np.asarray([row.pop('scores') for row in rows], dtype=np.float32)
    summary = {
        'scope': SCOPE, 'rows': len(rows), 'target_columns': list(TARGET_COLUMNS),
        'model_outputs': len(TRAITS), 'categories': RUBRIC['traits'],
        'email_groups': len(email_first), 'connected_groups': len(roots),
        'questions': len({row['response_number'] for row in rows}),
        'duplicate_sets': sum(len(indices) > 1 for indices in duplicate_sets.values()),
        'source_sha256': source_hash, 'input_features': list(INPUT_COLUMNS),
    }
    for row in rows:
        del row['email_group']
''' + source[end:]
    return source


def model_source(source):
    source = source.replace('SingleScoreModel', 'MultiTraitModel')
    source = replace(source, 'nn.Linear(256, 1))', 'nn.Linear(256, len(TRAITS)))')
    return replace(source, """if prediction.ndim != 2 or prediction.shape[1] != 1:
            raise ValueError('S3 harus menghasilkan satu nilai untuk setiap jawaban.')""",
                   """if prediction.shape != (n_essays, len(TRAITS)):
            raise ValueError('S3 harus menghasilkan empat skor per jawaban.')""")


def trait_heads_source(source):
    source = model_source(source)
    source = replace(source, '        self.head = nn.Sequential(nn.Linear(1024, 256), nn.GELU(),\n'
                     '                                  nn.Dropout(s3_mlp_dropout), nn.Linear(256, len(TRAITS)))',
                     '        # Pakai urutan inisialisasi E1 untuk menjaga bobot awal encoder dan pooling.\n'
                     '        nn.Linear(1024, 256)\n'
                     '        nn.Linear(256, len(TRAITS))')
    source = replace(source, '        self.scalar_mix = ScalarMix(scalar_mix_layers, gamma_initial)',
                     """        self.scalar_mix = ScalarMix(scalar_mix_layers, gamma_initial)
        # Inisialisasi head di CPU tanpa menggeser urutan pengacakan training bersama.
        with torch.random.fork_rng(devices=[]):
            self.head = nn.ModuleDict({
                name: nn.Sequential(
                    nn.Linear(1024, 64), nn.GELU(), nn.Dropout(s3_mlp_dropout), nn.Linear(64, 1),
                )
                for name, *_ in TRAITS
            })""")
    return replace(source, "        prediction = self.head(features['concat'])",
                   """        # Setiap head membaca vektor yang sama dan menghasilkan satu kolom skor.
        prediction = torch.cat(
            [self.head[name](features['concat']) for name, *_ in TRAITS], dim=1,
        )""")


def engine_source(source):
    source = replace(source, 'self.forward(ids).reshape(-1).float()', 'self.forward(ids).float()')
    return source[:source.index('        # Kembalikan skala asli')] + (
        '        # Kembalikan setiap keluaran ke satuan poin trait.\n'
        '        return scaled * SCORE_SPAN + SCORE_MIN\n'
    )


def fit_source(source):
    source = replace(source, ', seed, checkpoint=None):', ', seed):')
    source = source[:source.index('    start_epoch = 1')] + source[source.index('    for epoch in range('):]
    source = replace(source, 'range(start_epoch, epochs+1)', 'range(1, epochs+1)')
    start = source.index('        if checkpoint is not None:')
    source = source[:start] + source[source.index('        if early_stopped:', start):]
    source = replace(source, """                # Ubah nilai guru dari skala 0–100 menjadi 0–1 hanya untuk menghitung loss.
                targets = torch.as_tensor((bundle.y[ids]-cfg['score_min'])/(cfg['score_max']-cfg['score_min']),
                                          dtype=torch.float32, device=device).reshape(-1, 1)""",
                     """                # Normalisasi per trait memberi bobot sama pada empat dimensi loss.
                targets = torch.as_tensor((bundle.y[ids] - SCORE_MIN) / SCORE_SPAN,
                                          dtype=torch.float32, device=device)""")
    return source.replace('[B,1]', '[B,4]').replace(
        'metrics(bundle.y[validation], predicted, cfg)', 'metrics(bundle.y[validation], predicted)')


def config_source(source):
    source = source.replace("SCOPE = 'teacher_single_score_dataset300'\n", '')
    source = replace(source, """    'sheet': SHEET_NAME, 'expected_rows': EXPECTED_ROWS,
    # Rentang dinyatakan notebook unggahan, bukan diambil dari min/max test.
    'score_min': 0, 'score_max': 100,""", "    'sheet': SHEET_NAME, 'expected_rows': None,")
    start = source.index("    if result['score_min']")
    source = source[:start] + source[source.index("    if not 0 < result['test_fraction']", start):]
    return source


def build(experiment='E1'):
    notebook_name = {'E1': NAME, 'E2': E2_NAME, 'E3': E3_NAME, 'E4': E4_NAME, 'E5': E5_NAME}[experiment]
    trait_heads = experiment == 'E2'
    raw = (ROOT / EXPERIMENTS / BASE).read_bytes()
    base = json.loads(raw)
    source_by_role = {c['metadata']['aes43_role']: ''.join(c['source'])
                      for c in base['cells'] if c['cell_type'] == 'code'}
    old_hash = base['metadata']['aes_standalone']['code_sha256']
    workflow = (ROOT / 'scripts/s3_multitrait_native.py').read_text()
    sections = dict(re.findall(r'^# %% (\w+)\n(.*?)(?=^# %% |\Z)', workflow, re.M | re.S))
    cells = []

    def markdown(source):
        cells.append({'cell_type': 'markdown', 'metadata': {}, 'source': source})

    def code(role, source):
        cells.append({'cell_type': 'code', 'metadata': {'aes_multitrait_role': role},
                      'source': source.rstrip() + '\n', 'outputs': [], 'execution_count': None})

    def reuse(role, edit=lambda source: source):
        source = source_by_role[role].replace(old_hash, PLACEHOLDER)
        code(role, edit(source))

    def definition(name, text, role=None, edit=lambda source: source):
        """Place the explanation immediately above its native Python definition."""
        markdown(f'### `{name}`\n\n{text}')
        reuse(role or f'definition:{name}', edit)

    markdown("""# E43 S3 Multi-Trait · Native Learnable Code

Pilih **GPU** di Google Colab dan jalankan cell berurutan. Saat diminta, hubungkan Google Drive. Pada sesi pertama, unggah `data_cleaned_3sekolah.xlsx` agar notebook menyimpannya di `MyDrive/AES/datasets/semi_final/`. Sesi berikutnya langsung memakai salinan Drive setelah memeriksa SHA-256 workbook. Folder ini menjadi lokasi dataset semi-final untuk eksperimen berikutnya.

- **Masukan encoder:** pertanyaan, kunci jawaban, jawaban. Email dipakai untuk pembagian kelompok. Jawaban duplikat juga dikelompokkan.
- **Empat keluaran:** isi (1–4), organisasi (1–2), kebahasaan (1–2), mekanik (1–2). Rentang awal mengikuti kategori pada sampel. Cocokkan rentang di cell Pemetaan dengan rubrik resmi sebelum run penelitian. Target diambil langsung dari empat kolom rubrik.
- **Satu resep E43:** RoBERTa pretrained dilatih ulang dengan LR 5e-6, LR layer lain 1e-4, Scalar Mix empat layer + gamma, BiGRU, co-attention satu head, masked mean, dan BGE-M3 frozen. MSE merata-ratakan error empat target pada skala 0–1.
- **Validasi E43** memakai lima fold development. Epoch terbaik tiap fold memakai RMSE pada skala skor asli. Untuk memilih satu epoch bagi model bersama, hitung akar dari rata-rata kuadrat error seluruh pasangan skor jawaban × trait. Median epoch terbaik yang dibulatkan ke atas dipakai untuk satu refit pada seluruh development. Evaluasi test dilakukan setelah refit. Seed tetap mengikuti E43 asli.
- **Cara baca kode:** heading kecil menjelaskan fungsi dan kelas. Heading yang menyebut tahap seperti **Hitung fitur BGE**, **Jalankan lima fold**, dan **Evaluasi test** masing-masing punya cell yang benar-benar menjalankan proses serta menampilkan hasilnya.
- **Hasil** berupa Excel per trait, prediksi validasi dan test, serta checkpoint final. RMSE/Pearson memakai prediksi mentah, QWK/Gwet AC2 memakai kategori terdekat. Excel dan checkpoint tetap di runtime. Unduhan akhir mengirim Excel ke komputer. Hasil fold selesai bisa dipakai ulang selama filesystem runtime masih tersedia. Satu fit yang terputus dijalankan ulang.
""")
    markdown('## 1 · Setup pustaka pada kernel Colab')
    reuse('dependency_setup')
    markdown('## 2 · Import dan ruang variabel notebook')
    reuse('imports')
    markdown('## 3 · Nama model pretrained')
    reuse('model_names')
    markdown('## 4 · Pemetaan input dan empat target')
    code('columns', sections['columns'])
    markdown('## 5 · Hyperparameter E43')
    reuse('hyperparameters', lambda s: s.replace('HIPERPARAMETER PENELITIAN — E30 MLP', 'HIPERPARAMETER BASELINE E43')
          .replace('REQUIRE_GPU = True\n', ''))
    definition('config', 'Periksa konfigurasi model, ukuran batch, dan anggaran token.',
               'configuration_defaults', config_source)
    definition('device_name', 'Tentukan perangkat untuk encoder dan training.')

    markdown('## 6 · Dataset semi-final di Google Drive')
    markdown('### `read_dataset` dan `Bundle`\n\nWorkbook dibaca sebagai tiga kolom teks dan matriks empat skor. Email hanya dipakai untuk membentuk kelompok split.')
    reuse('dataset_definitions', dataset_source)
    code('storage', sections['storage'])
    markdown('## 7 · Baca workbook dan periksa X serta y')
    code('prepare', sections['prepare'])
    markdown('## 8 · Pisahkan development, test, dan lima fold')
    definition('make_split', 'Seluruh jawaban seorang siswa dan kelompok duplikat tetap bersama. Lima fold dibuat dari development.')
    definition('contexts_from_bundle', 'Ambil pasangan pertanyaan dan referensi yang mengiringi setiap jawaban.')
    code('split_execute', sections['split_execute'])

    markdown('## 9 · Tokenisasi, window, dan posisi input')
    for name, lesson in (
        ('mean_by_essay', 'Rata-ratakan vektor window yang berasal dari jawaban yang sama.'),
        ('context_token_ids', 'Ubah soal dan referensi jadi ID token sesuai batas konteks.'),
        ('reference_context_positions', 'Tandai posisi token referensi untuk co-attention.'),
        ('pair_content_windows', 'Rangkai tiga teks dan bagi jawaban panjang menjadi window yang bertumpang tindih.'),
        ('pair_batch_windows', 'Tambahkan padding dan simpan pemetaan window ke jawaban.'),
    ):
        definition(name, lesson)

    markdown('## 10 · Memuat tokenizer dan encoder pretrained')
    definition('release', 'Bebaskan alokasi yang selesai dipakai sebelum berpindah tahap.')
    reuse('definition:ExpectedPoolerNotice')
    definition('encoder', 'Muat bobot pretrained. Jalur BGE dibekukan dan RoBERTa disiapkan untuk fine-tuning.')
    definition('tokenizer', 'Muat tokenizer dari repositori dan revisi yang cocok dengan encoder.')
    markdown('### Cek window pada satu jawaban development\n\nTabel menampilkan panjang input dan jumlah token yang ditandai untuk referensi serta jawaban.')
    code('token_preview', sections['token_preview'])

    markdown('## 11 · Fitur BGE-M3 frozen')
    definition('paired_pretrained_values', 'Gabungkan vektor dari semua window menjadi satu vektor per jawaban.')
    definition('bge_features', 'Pakai cache bila teks dan konfigurasi tokenizer sama; vektor berdimensi 1024.',
               edit=lambda s: s.replace('soal + jawaban_referensi + response_text; no teacher score',
                                        'Pertanyaan + Kunci_Jawaban + Jawaban'))
    markdown('### Hitung fitur BGE untuk seluruh jawaban\n\nCell ini menampilkan progres encoding dan ukuran matriks vektor yang akan dipakai pada CV dan refit.')
    code('bge_execute', sections['bge_execute'])

    markdown('## 12 · Learnable Code — arsitektur S3 E43\n\nRoBERTa → Scalar Mix 4 + gamma → BiGRU dan co-attention dua arah → masked mean. Vektor RoBERTa 768 digabung dengan proyeksi BGE 256. MLP bersama menghasilkan empat skor.')
    definition('answer_token_mask', 'Pilih posisi token jawaban yang valid.')
    definition('check_answer_pool_inputs', 'Periksa bentuk tensor dan keberadaan token valid sebelum pooling.')
    definition('AnswerBiGRUAttention', 'Kelas dasar menyediakan BiGRU dan urutan inisialisasi yang dipakai E43.')
    definition('reference_token_mask', 'Pilih posisi token referensi untuk jalur pencocokan.')
    definition('AnswerBiGRUCoAttentionMeanPool', 'Cocokkan token jawaban dan referensi, lalu rata-ratakan fitur token jawaban valid. Bobot matching co-attention bergantung pada representasi token.')
    definition('ScalarMix` dan `MultiTraitModel', 'Scalar Mix mempelajari bobot empat layer terakhir dan gamma. MLP mengubah fitur gabungan 1024 menjadi empat skor.',
               'definition:SingleScoreModel', model_source)
    definition('Engine', 'Siapkan input batch dan kembalikan prediksi dari skala normalisasi ke poin tiap trait.',
               edit=engine_source)

    markdown('## 13 · Inisialisasi model dan spesifikasi')
    definition('seed_everything', 'Atur pengacakan training pada awal setiap fit mengikuti seed E43.')
    definition('make_model', 'Buat model baru untuk setiap fold dan refit.',
               edit=lambda s: s.replace('SingleScoreModel', 'MultiTraitModel'))
    code('architecture', sections['architecture'])

    markdown('## 14 · Metrik evaluasi\n\nRMSE dan Pearson memakai prediksi mentah. QWK dan Gwet AC2 memakai kategori terdekat.')
    original_metrics = (ROOT / 'src/aes/metrics.py').read_text().split('\ndef selection_monitor(')[0]
    code('definition:trait_metrics', original_metrics)
    markdown('### `metrics` dan `predictions_array`\n\nPeriksa bentuk empat skor, lalu hitung metrik per trait. Pemilihan epoch memakai RMSE dari seluruh skor validasi pada skala asli.')
    code('metrics', sections['metrics'])

    markdown('## 15 · Learnable Code — optimizer, loss, dan satu fit')
    definition('optimizer_groups', 'Kelompokkan parameter dengan LR RoBERTa 5e-6 dan LR komponen lain 1e-4.')
    definition('fit', 'Latih satu model dengan AdamW dan MSE dari empat target skala 0–1. Saat CV, pilih epoch dengan RMSE validasi terkecil.',
               edit=fit_source)

    markdown('## 16 · Validasi silang pada development')
    markdown('### `run_cv`\n\nLatih lima model E43 pada lima fold. Ambil median epoch terbaik lalu bulatkan ke atas untuk menentukan durasi refit.')
    code('cv_definition', sections['cv_definition'])
    markdown('### Jalankan lima fold\n\nOutput berisi progres per fold, epoch terbaik tiap fold, dan ringkasan CV.')
    code('cv_execute', sections['cv_execute'])

    markdown('## 17 · Refit E43 pada seluruh development')
    definition('save_model', 'Simpan bobot dan metadata dari model yang selesai dilatih.')
    markdown('### `refit_checkpoint`\n\nLatih sekali pada seluruh development menggunakan jumlah epoch dari CV, lalu simpan checkpoint.')
    code('final_model', sections['final_model'])
    markdown('### Jalankan refit dan simpan checkpoint\n\nOutput menunjukkan bobot Scalar Mix final dan lokasi checkpoint.')
    code('refit_execute', sections['refit_execute'])

    markdown('## 18 · Evaluasi test\n\nBaca checkpoint dari tahap refit. Tabel output berisi RMSE, Pearson, QWK, dan Gwet AC2 untuk setiap trait.')
    markdown('### `evaluate_test`\n\nPrediksi data test dan simpan hasilnya agar bisa dibaca ulang saat cell diulang.')
    code('test_definition', sections['test_definition'])
    markdown('### `trait_rows`, `prediction_rows`, dan `export_results`\n\nBentuk tabel evaluasi dan prediksi yang digunakan di layar dan workbook.')
    reuse('definition:excel_value')
    reuse('definition:write_excel')
    code('report', sections['report'])
    markdown('### Jalankan evaluasi test')
    code('test_execute', sections['test_execute'])

    markdown('## 19 · Susun Excel hasil\n\nGabungkan hasil CV, prediksi out-of-fold, hasil test, dan riwayat training yang sudah tersedia.')
    code('export_execute', sections['export_execute'])
    markdown('## 20 · Unduh Excel')
    code('download', sections['download'])

    if experiment != 'E1':
        # Semua eksperimen berbagi data/split/cache BGE. Config dan hash notebook
        # membedakan folder run, checkpoint, serta hasil prediksi.
        workflow_labels = ('prepare', 'cv_definition', 'final_model', 'report', 'export_execute')
        for cell in cells:
            role = cell['metadata'].get('aes_multitrait_role')
            if cell['cell_type'] == 'markdown':
                cell['source'] = (cell['source']
                    .replace('# E43 S3 Multi-Trait', f'# {experiment} S3 Multi-Trait')
                    .replace('**Satu resep E43:**', '**Backbone E1 (resep E43):**')
                    .replace('Validasi E43', f'Validasi {experiment}')
                    .replace('Hyperparameter E43', f'Hyperparameter {experiment}')
                    .replace('arsitektur S3 E43', f'arsitektur S3 {experiment}')
                    .replace('lima model E43', f'lima model {experiment}')
                    .replace('Refit E43', f'Refit {experiment}'))
                if trait_heads:
                    cell['source'] = (cell['source']
                        .replace('MLP bersama menghasilkan empat skor.',
                                 'Empat MLP khusus trait masing-masing menghasilkan satu skor.')
                        .replace('MLP mengubah fitur gabungan 1024 menjadi empat skor.',
                                 'Setiap head mengubah vektor 1024 menjadi satu skor melalui 64 unit tersembunyi.'))
            elif experiment in ('E3', 'E4', 'E5') and role == 'hyperparameters':
                cell['source'] = replace(
                    cell['source'],
                    'S3_MLP_DROPOUT = 0.1             # Kembali ke E30; bukan dropout0.2 E38.',
                    ('S3_MLP_DROPOUT = 0.2             # E3: dropout MLP E1 dinaikkan dari 0,1 ke 0,2.'
                     if experiment == 'E3' else
                     f'S3_MLP_DROPOUT = 0.2             # {experiment}: dropout MLP 0,2 mengikuti E3.'),
                )
            elif role in workflow_labels:
                cell['source'] = cell['source'].replace('E43', experiment)
            elif trait_heads and role == 'architecture':
                cell['source'] = replace(cell['source'], "('Concat → MLP', f'1024 → 256 → {len(TRAITS)}'",
                                         "('Empat head khusus trait', '4 × (1024 → 64 → 1)'")
            elif trait_heads and role == 'definition:SingleScoreModel':
                cell['source'] = trait_heads_source(source_by_role[role])
    if trait_heads:
        cells[0]['source'] += """
**Perbandingan E1 → E2.** E1 adalah notebook multi-trait dari E43 dengan head bersama `1024 → 256 → 4`. E2 memakai empat head `1024 → 64 → 1`, satu per trait. Semua head membaca vektor gabungan yang sama dan urutan keluaran tetap isi, organisasi, kebahasaan, mekanik. Dataset semi-final, split siswa, seed, LR, dropout, MSE, dan RMSE validasi mengikuti E1.
"""

    if experiment == 'E3':
        cells[0]['source'] += """
**Perbandingan E1 → E3.** E3 memakai head bersama E1 `1024 → 256 → 4` dengan dropout MLP 0,2. E1 memakai 0,1. Dataset semi-final, split siswa, seed, LR, MSE, dan RMSE validasi skala asli mengikuti E1. Bandingkan RMSE OOF gabungan dan per trait pada lima fold yang sama.

Nilai 0,2 adalah kandidat yang diambil dari [konfigurasi PMAES](https://github.com/gdufsnlp/PMAES/blob/HEAD/main_PMAES_many2many.py#L43-L50). [Paper PMAES](https://aclanthology.org/2023.acl-long.83/) memakai arsitektur dan protokol berbeda, sehingga pengaruh dropout ini diuji kembali pada S3.
"""

    if experiment == 'E4':
        for cell in cells:
            role = cell['metadata'].get('aes_multitrait_role')
            if role == 'definition:SingleScoreModel':
                cell['source'] = replace(
                    cell['source'],
                    'nn.Dropout(s3_mlp_dropout), nn.Linear(256, len(TRAITS)))',
                    'nn.Dropout(s3_mlp_dropout), nn.Linear(256, len(TRAITS)), nn.Sigmoid())',
                )
            elif role == 'prepare':
                cell['source'] = replace(cell['source'], "'experiment': 'E4',",
                                         "'experiment': 'E4', 'output_activation': 'sigmoid',")
            elif role == 'architecture':
                cell['source'] = replace(cell['source'], "f'1024 → 256 → {len(TRAITS)}'",
                                         "f'1024 → 256 → {len(TRAITS)} → sigmoid'")
            elif cell['cell_type'] == 'markdown':
                cell['source'] = (cell['source']
                    .replace('MLP bersama menghasilkan empat skor.',
                             'MLP bersama menghasilkan empat skor melalui sigmoid.')
                    .replace('MLP mengubah fitur gabungan 1024 menjadi empat skor.',
                             'MLP mengubah fitur gabungan 1024 menjadi empat skor normalisasi melalui sigmoid.'))
        cells[0]['source'] += """
**Perbandingan E3 → E4.** E3 adalah acuan aktif dengan dropout MLP 0,2. E4 menambahkan sigmoid setelah layer linear terakhir pada head bersama `1024 → 256 → 4`. Sigmoid diterapkan saat training dan inferensi. Tiap keluaran berada pada rentang 0–1, lalu dikembalikan ke skala isi 1–4, organisasi 1–2, kebahasaan 1–2, mekanik 1–2. Loss tetap MSE target normalisasi; pemilihan epoch tetap memakai RMSE validasi skala asli.

Keluaran sigmoid digunakan dalam [implementasi CTS](https://github.com/robert1ridley/cross-prompt-trait-scoring/blob/HEAD/models/CTS.py) dan [ProTACT](https://github.com/doheejin/ProTACT/blob/HEAD/models/ProTACT.py). E4 menguji adaptasi aktivasi ini pada S3. Checkpoint dan hasil E4 mempunyai folder run tersendiri. Bandingkan RMSE validasi dan hasil per trait dengan E3.
"""

    if experiment == 'E5':
        module = (ROOT / 'scripts/trait_attention_native.py').read_text()
        index = next(i for i, cell in enumerate(cells)
                     if cell['metadata'].get('aes_multitrait_role') == 'definition:SingleScoreModel')
        cells[index-1:index-1] = [
            {'cell_type': 'markdown', 'metadata': {},
             'source': '### `TraitAttentionHead`\n\nEmpat MLP 1024→64 membentuk representasi trait. Dot-product attention memilih informasi dari tiga trait lain, lalu Linear128→1 memberi skor tiap trait.'},
            {'cell_type': 'code', 'metadata': {'aes_multitrait_role': 'definition:TraitAttentionHead'},
             'source': module, 'outputs': [], 'execution_count': None},
        ]
        for cell in cells:
            role = cell['metadata'].get('aes_multitrait_role')
            if role == 'definition:SingleScoreModel':
                source = trait_heads_source(source_by_role[role])
                start = source.index('            self.head = nn.ModuleDict({')
                end = source.index('            })', start) + len('            })')
                source = source[:start] + (
                    '            self.head = TraitAttentionHead([name for name, *_ in TRAITS], s3_mlp_dropout)'
                ) + source[end:]
                start = source.index('        # Setiap head membaca vektor yang sama')
                end = source.index('        if prediction.shape', start)
                source = source[:start] + "        prediction = self.head(features['concat'])\n" + source[end:]
                cell['source'] = source
            elif role == 'prepare':
                cell['source'] = replace(
                    cell['source'], "'experiment': 'E5',",
                    "'experiment': 'E5', 'head_type': 'inter_trait_attention', 'trait_hidden': 64,",
                )
            elif role == 'architecture':
                cell['source'] = replace(
                    cell['source'], "('Concat → MLP', f'1024 → 256 → {len(TRAITS)}'",
                    "('Representasi trait → attention → skor', '4 × (1024→64), attention antartrait, 128→1 per trait'",
                )
            elif cell['cell_type'] == 'markdown':
                cell['source'] = (cell['source']
                    .replace('MLP bersama menghasilkan empat skor.',
                             'Empat representasi trait saling bertukar informasi lewat attention dan menghasilkan empat skor.')
                    .replace('MLP mengubah fitur gabungan 1024 menjadi empat skor.',
                             'TraitAttentionHead mengubah fitur gabungan 1024 menjadi empat skor melalui attention antartait.'))
        cells[0]['source'] += """
**E5 dapat dijalankan paralel dengan E4 di runtime GPU Colab terpisah.** E3 adalah acuan keduanya. E4 menguji sigmoid. E5 menguji empat representasi trait 64 dimensi dan dot-product attention antartrait, dengan keluaran linear dan dropout 0,2. Total parameter head E5 adalah 262.916, dibanding 263.428 pada E3. Dataset semi-final, split, seed, LR, MSE, dan RMSE skala asli mengikuti E3.

E5 mengadaptasi [attention antartait dari CTS](https://github.com/robert1ridley/cross-prompt-trait-scoring/blob/HEAD/models/CTS.py). CTS membentuk representasi dari LSTM per trait. E5 membentuknya dari fusion S3 lewat MLP 1024→64. Dimensi 64 dan dropout 0,2 adalah rancangan proyek mengikuti anggaran head E2 dan baseline E3. Setiap trait mengambil weighted sum dari tiga trait lain sebelum prediksi. Empat nilai acuan dipakai untuk loss dan evaluasi. Forward head hanya menerima fitur teks.

Hubungkan Drive pada kedua runtime agar keduanya membaca workbook semi-final yang sama. Hasil dan checkpoint dipisahkan melalui konfigurasi eksperimen dan hash notebook.
"""

    fingerprint = hashlib.sha256('\n'.join(c['source'] for c in cells if c['cell_type'] == 'code').encode()).hexdigest()
    for index, cell in enumerate(cells):
        cell['source'] = cell['source'].replace(PLACEHOLDER, fingerprint)
        cell['id'] = f's3-multitrait-{index:03d}'
        if cell['cell_type'] == 'code':
            ast.parse(cell['source'])
    notebook = {
        'cells': cells, 'nbformat': 4, 'nbformat_minor': 5,
        'metadata': {'colab': {'name': notebook_name, 'provenance': []}, 'accelerator': 'GPU',
                     'kernelspec': base['metadata']['kernelspec'],
                     'aes_standalone': {'code_sha256': fingerprint, 'base': BASE,
                                        'base_sha256': hashlib.sha256(raw).hexdigest()}},
    }
    serialized = json.dumps(notebook, ensure_ascii=False, indent=1) + '\n'
    for folder in (ROOT / EXPERIMENTS, ROOT / 'aes-learning' / EXPERIMENTS):
        (folder / notebook_name).write_text(serialized)
    print(f'{notebook_name} · {len(cells)} cells · {fingerprint}')


if __name__ == '__main__':
    build()
    build('E2')
    build('E3')
    build('E4')
    build('E5')
