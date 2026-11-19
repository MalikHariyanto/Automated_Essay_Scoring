"""Build the self-contained four-trait Colab notebook from the native E43 cells."""
import ast
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = 'AES_300_E43_E42_S3_MaskedMeanPooling_Native_Learnable_GPU.ipynb'
NAME = 'AES_S3_MultiTrait_3Sekolah.ipynb'
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
    return (source.replace('[B,1]', '[B,4]').replace('best_rmse', 'best_macro_nrmse')
            .replace('metrics(bundle.y[validation], predicted, cfg)', 'metrics(bundle.y[validation], predicted)')
            .replace("row['validation']['RMSE']", "row['validation']['macro_nRMSE']")
            .replace('row["validation"]["RMSE"]', 'row["validation"]["macro_nRMSE"]')
            .replace('RMSE validasi', 'macro nRMSE validasi').replace(':.4f} poin', ':.4f}')
            .replace('best_validation_rmse', 'best_validation_macro_nrmse')
            .replace('minimum validation RMSE in original score units', 'minimum validation macro nRMSE'))


def config_source(source):
    source = source.replace("SCOPE = 'teacher_single_score_dataset300'\n", '')
    source = replace(source, """    'sheet': SHEET_NAME, 'expected_rows': EXPECTED_ROWS,
    # Rentang dinyatakan notebook unggahan, bukan diambil dari min/max test.
    'score_min': 0, 'score_max': 100,""", "    'sheet': SHEET_NAME, 'expected_rows': None,")
    start = source.index("    if result['score_min']")
    source = source[:start] + source[source.index("    if not 0 < result['test_fraction']", start):]
    return source


def build():
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

    markdown("""# E43 S3 multi-trait · Dataset semi-final tiga sekolah

Pilih **GPU** di Google Colab dan jalankan cell berurutan. Saat diminta, hubungkan Google Drive. Pada sesi pertama, unggah `data_cleaned_3sekolah.xlsx` agar notebook menyimpannya di `MyDrive/AES/datasets/semi_final/`. Sesi berikutnya langsung memakai salinan Drive setelah memeriksa SHA-256 workbook. Folder ini menjadi lokasi dataset semi-final untuk eksperimen berikutnya.

- **Masukan encoder:** pertanyaan, kunci jawaban, jawaban. Email dipakai untuk pembagian kelompok. Jawaban duplikat juga dikelompokkan.
- **Empat keluaran:** isi (1–4), organisasi (1–2), kebahasaan (1–2), mekanik (1–2). Rentang awal mengikuti kategori pada sampel. Cocokkan rentang di cell Pemetaan dengan rubrik resmi sebelum run penelitian. Target diambil langsung dari empat kolom rubrik.
- **Satu resep E43:** RoBERTa pretrained dilatih ulang dengan LR 5e-6, LR layer lain 1e-4, Scalar Mix empat layer + gamma, BiGRU, co-attention satu head, masked mean, dan BGE-M3 frozen. MSE merata-ratakan error empat target pada skala 0–1.
- **Validasi E43** memakai lima fold development. Epoch terbaik tiap fold memakai macro nRMSE, yaitu rata-rata RMSE per trait dibagi lebar skala masing-masing. Median epoch terbaik yang dibulatkan ke atas dipakai untuk satu refit pada seluruh development. Evaluasi test dilakukan setelah refit. Seed tetap mengikuti E43 asli.
- **Hasil** berupa Excel per trait, prediksi validasi dan test, serta checkpoint final. RMSE/Pearson memakai prediksi mentah, QWK/Gwet AC2 memakai kategori terdekat. Excel dan checkpoint tetap di runtime. Unduhan akhir mengirim Excel ke komputer. Hasil fold selesai bisa dipakai ulang selama filesystem runtime masih tersedia. Satu fit yang terputus dijalankan ulang.
""")
    markdown('## 1 · Setup, kolom, dan hyperparameter E43')
    reuse('dependency_setup')
    reuse('imports')
    reuse('model_names')
    code('columns', sections['columns'])
    reuse('hyperparameters', lambda s: s.replace('HIPERPARAMETER PENELITIAN — E30 MLP', 'HIPERPARAMETER BASELINE E43')
          .replace('REQUIRE_GPU = True\n', ''))
    reuse('configuration_defaults', config_source)
    reuse('definition:device_name')
    reuse('dataset_definitions', dataset_source)

    markdown('## 2 · Dataset semi-final di Drive dan split\n\nFolder Drive dipakai ulang di runtime Colab baru. Split menjaga seluruh jawaban per akun dan kelompok duplikat tetap bersama.')
    code('storage', sections['storage'])
    reuse('definition:make_split')
    reuse('definition:contexts_from_bundle')
    code('prepare', sections['prepare'])

    markdown('## 3 · Tokenisasi dan fitur BGE-M3\n\nFitur BGE dibuat dari tiga teks. Encoder BGE frozen dan fiturnya dipakai di lima fold dan refit.')
    for role in ('mean_by_essay', 'context_token_ids', 'reference_context_positions', 'pair_content_windows',
                 'pair_batch_windows', 'paired_pretrained_values', 'release', 'ExpectedPoolerNotice',
                 'encoder', 'tokenizer', 'bge_features'):
        reuse(f'definition:{role}', lambda s: s.replace('soal + jawaban_referensi + response_text; no teacher score',
                                                       'Pertanyaan + Kunci_Jawaban + Jawaban'))

    markdown('## 4 · Model S3\n\nRoBERTa → Scalar Mix 4 + gamma → BiGRU/co-attention → masked mean. Gabungkan fitur RoBERTa 768 dan proyeksi BGE 256. MLP bersama menghasilkan empat skor.')
    for role in ('answer_token_mask', 'check_answer_pool_inputs', 'AnswerBiGRUAttention', 'reference_token_mask'):
        reuse(f'definition:{role}')
    reuse('definition:AnswerBiGRUCoAttentionMeanPool')
    reuse('definition:SingleScoreModel', model_source)
    reuse('definition:Engine', engine_source)
    reuse('definition:seed_everything')
    reuse('definition:make_model', lambda s: s.replace('SingleScoreModel', 'MultiTraitModel'))

    markdown('## 5 · Training E43\n\nEarly stopping memakai macro nRMSE validasi. Refit memakai median epoch terbaik lima fold, dibulatkan ke atas.')
    original_metrics = (ROOT / 'src/aes/metrics.py').read_text().split('\ndef selection_monitor(')[0]
    code('definition:trait_metrics', original_metrics)
    code('metrics', sections['metrics'])
    reuse('definition:optimizer_groups')
    reuse('definition:fit', fit_source)
    reuse('definition:save_model')
    code('experiment', sections['experiment'])

    markdown('## 6 · Hasil dan unduhan\n\n`Ringkasan CV` berisi hasil lima fold E43. Sheet OOF berisi prediksi setiap jawaban development dari fold validasinya. Sheet test berisi hasil E43 setelah refit.')
    reuse('definition:excel_value')
    reuse('definition:write_excel')
    code('report', sections['report'])
    code('execute', sections['execute'])
    code('download', sections['download'])

    fingerprint = hashlib.sha256('\n'.join(c['source'] for c in cells if c['cell_type'] == 'code').encode()).hexdigest()
    for index, cell in enumerate(cells):
        cell['source'] = cell['source'].replace(PLACEHOLDER, fingerprint)
        cell['id'] = f's3-multitrait-{index:03d}'
        if cell['cell_type'] == 'code':
            ast.parse(cell['source'])
    notebook = {
        'cells': cells, 'nbformat': 4, 'nbformat_minor': 5,
        'metadata': {'colab': {'name': NAME, 'provenance': []}, 'accelerator': 'GPU',
                     'kernelspec': base['metadata']['kernelspec'],
                     'aes_standalone': {'code_sha256': fingerprint, 'base': BASE,
                                        'base_sha256': hashlib.sha256(raw).hexdigest()}},
    }
    serialized = json.dumps(notebook, ensure_ascii=False, indent=1) + '\n'
    for folder in (ROOT / EXPERIMENTS, ROOT / 'aes-learning' / EXPERIMENTS):
        (folder / NAME).write_text(serialized)
    print(f'{NAME} · {len(cells)} cells · {fingerprint}')


if __name__ == '__main__':
    build()
