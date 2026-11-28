"""Publish aggregate AES checkpoints from private ZIPs and rebuild the evidence workbook."""
import argparse
import csv
import hashlib
import io
import json
import re
import zipfile
from pathlib import Path

from openpyxl import Workbook, load_workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / 'docs/skripsi'
GROUP = ROOT / 'eksperimen/03_tiga_input_multi_trait'
ARCHIVE = GROUP / 'output/checkpoints'
BOOK = DOCS / 'Register_Bukti_AES_MultiTrait.xlsx'
COMMON = (
    'DATA ROBERTA BGEM3 FUSION HEAD ATTENTION REL MSE OPTIM LR DROP BATCH STOP SPLIT '
    'SEED WORG WISI CHECK ENS CUES MET TPE GATE'
).split()
CONFIG_GROUPS = {
    'DATA': 'expected_rows dataset_version traits',
    'ROBERTA': 'roberta',
    'BGEM3': 'bge bge_batch',
    'FUSION': 's3_fusion fusion_variant',
    'HEAD': 'head_type trait_hidden',
    'EMA31': 'ema_maximum_decay ema_offset inference_weights',
    'TPE': 'hpo_recipe_trial hpo_recipe_archive_sha256',
    'E43PATH': 'scalar_mix_layers scalar_mix_gamma_initial bigru_hidden scalar_mix_per_trait',
    'REL': 'trait_corr_threshold trait_relation_weight',
    'OPTIM': 'weight_decay warmup_ratio gradient_clip',
    'LR': 'encoder_lr s3_encoder_lr projection_lr head_lr s3_mlp_lr s3_layers_lr',
    'DROP': 'dropout s3_mlp_dropout',
    'BATCH': 'micro_batch effective_batch roberta_max_length roberta_overlap bge_max_length bge_overlap context_max_tokens question_max_tokens',
    'STOP': 'epochs early_stopping_patience',
    'SPLIT': 'folds test_fraction split_seed split_candidates',
    'SEED': 'seed',
    'WORG': 'organization_loss_ratio',
    'WISI': 'isi_loss_ratio',
    'CHECK': 'checkpoint_rmse_margin',
    'CUES': 'task_evidence',
}
CONFIG = {key: decision for decision, keys in CONFIG_GROUPS.items() for key in keys.split()}
SETTINGS = {
    'EPOCHS': 'epochs', 'EARLY_STOPPING_PATIENCE': 'early_stopping_patience',
    'TRAIN_SEED': 'seed', 'TRAIN_SEEDS': 'seed', 'SPLIT_SEED': 'split_seed', 'FOLDS': 'folds',
    'TEST_FRACTION': 'test_fraction', 'MICRO_BATCH': 'micro_batch', 'EFFECTIVE_BATCH': 'effective_batch',
    'ENCODER_LR': 's3_encoder_lr', 'S3_MLP_LR': 's3_mlp_lr', 'S3_LAYERS_LR': 's3_layers_lr',
    'PROJECTION_LR': 'projection_lr', 'DROPOUT': 'dropout', 'S3_MLP_DROPOUT': 's3_mlp_dropout',
    'WEIGHT_DECAY': 'weight_decay', 'WARMUP_RATIO': 'warmup_ratio', 'GRADIENT_CLIP': 'gradient_clip',
    'ROBERTA_MAX_TOKENS': 'roberta_max_length', 'BGE_MAX_TOKENS': 'bge_max_length',
    'WINDOW_OVERLAP': 'roberta_overlap', 'CONTEXT_MAX_TOKENS': 'context_max_tokens',
    'QUESTION_MAX_TOKENS': 'question_max_tokens', 'ORGANIZATION_LOSS_RATIO': 'organization_loss_ratio',
    'CHECKPOINT_RMSE_MARGIN': 'checkpoint_rmse_margin',
}
# Export only identified tab/column combinations. Never copy any individual prediction sheet.
METRIC_COLUMNS = 'fold pilihan skenario epoch epoch_terpilih n trait RMSE Pearson QWK macro_QWK proporsi_di_luar_skala'.split() + ['Gwet AC2']
LOW_COLUMNS = ['fold', 'pilihan', 'skenario', 'bagian', 'trait', 'skor_terendah', 'n_label_terendah',
               'n_terdeteksi', 'proporsi_terdeteksi', 'jumlah label 1', '1 benar', '1 salah jadi 2',
               '2 salah jadi 1', 'skor di atas 1 salah jadi 1']
HPO_COLUMNS = ('trial status RMSE encoder_lr aux_lr mlp_dropout weight_decay warmup_ratio best_epoch '
               'elapsed_seconds anchor retry_of seed mean_cv_RMSE epoch_refit pooled_OOF_RMSE macro_QWK').split()
TABS = {
    'Ringkasan CV': 'skenario fold_selesai jumlah_fold jawaban_OOF mean_cv_RMSE epoch_refit'.split(),
    'E25 vs E23 per Trait': ['trait', 'metrik', 'E25 1x (B)', 'E23 2x (B)', 'delta E23 minus E25'],
    'E25 vs E23 Ringkasan': ['konfigurasi', 'RMSE rata-rata fold', 'macro QWK rata-rata fold',
                            'label isi 1', 'isi 1 terdeteksi', 'label 2–4 ke 1'],
    'E27 vs E25 per Trait': ['trait', 'metrik', 'E27 E43 (B)', 'E25 kontrol (B)', 'delta E27 minus E25'],
    'E27 vs E25 Ringkasan': ['konfigurasi', 'RMSE rata-rata fold', 'macro QWK rata-rata fold',
                            'label isi 1', 'isi 1 terdeteksi', 'label 2–4 ke 1'],
    'E28 vs E25 per Trait': ['trait', 'metrik', 'E28 ScalarMix (B)', 'E25 kontrol (B)', 'delta E28 minus E25'],
    'E28 vs E25 Ringkasan': ['konfigurasi', 'RMSE rata-rata fold', 'macro QWK rata-rata fold',
                            'label isi 1', 'isi 1 terdeteksi', 'label 2–4 ke 1'],
    'ScalarMix Checkpoint B': ['fold', 'trait', 'Indeks Layer RoBERTa', 'Layer ke-', 'Logit', 'Bobot Softmax', 'Bobot Awal', 'Gamma'],
    'E29 vs E28 per Trait': ['trait', 'metrik', 'E29 (B)', 'Pembanding (B)', 'delta E29 minus pembanding'],
    'E29 vs E25 per Trait': ['trait', 'metrik', 'E29 (B)', 'Pembanding (B)', 'delta E29 minus pembanding'],
    'E30 vs E29 per Trait': ['trait', 'metrik', 'E30 (B)', 'Pembanding (B)', 'delta E30 minus pembanding'],
    'E30 vs E28 per Trait': ['trait', 'metrik', 'E30 (B)', 'Pembanding (B)', 'delta E30 minus pembanding'],
    'E30 vs E25 per Trait': ['trait', 'metrik', 'E30 (B)', 'Pembanding (B)', 'delta E30 minus pembanding'],
    'E31 vs E29 per Trait': ['trait', 'metrik', 'E31 (B)', 'Pembanding (B)', 'delta E31 minus pembanding'],
    'E31 vs E28 per Trait': ['trait', 'metrik', 'E31 (B)', 'Pembanding (B)', 'delta E31 minus pembanding'],
    'E31 vs E25 per Trait': ['trait', 'metrik', 'E31 (B)', 'Pembanding (B)', 'delta E31 minus pembanding'],
    'E31 vs E29 per Fold': ['fold', 'E31 epoch', 'E29 epoch', 'E31 RMSE', 'E29 RMSE',
                            'E31 macro QWK', 'E29 macro QWK', 'delta RMSE', 'delta macro QWK'],
    'Diagnostik EMA': ['fold', 'epoch', 'RMSE online', 'RMSE EMA', 'update EMA', 'decay EMA', 'checkpoint A', 'checkpoint B'],
    'Progres Fold': 'fold RMSE epoch_terpilih epoch_dijalankan menit_training'.split(),
    'Riwayat Training': (
        'fold skenario epoch validation_RMSE train_mse_scaled train_weighted_mse_scaled train_relation_loss '
        'train_total_loss best_epoch_so_far epochs_without_improvement optimizer_steps active_lr_steps amp_skipped_steps validation_online_RMSE ema_updates ema_decay'
    ).split(),
    'Isi 1 Latih dan Validasi B': ['fold', 'bagian', 'jumlah jawaban', 'label 1',
                                 'label 1 terdeteksi', 'label 2–4 salah jadi 1'],
    'Bobot Fitur Isi B': ['fold', 'lokasi_sekolah', 'pelaku', 'kejadian', 'waktu', 'rumah_tanpa_sekolah'],
}
TABS.update({name: METRIC_COLUMNS for name in (
    'CV per Fold dan Trait', 'CV OOF per Trait', 'Test per Trait', 'Matriks Metrik CV dan Test',
    'Pilihan Checkpoint', 'A B per Fold dan Trait', 'OOF A B per Trait', 'Ringkasan Test A B',
    'Test A B per Trait', 'Metrik Validasi per Epoch',
)})
TABS['Matriks Metrik CV dan Test'] = METRIC_COLUMNS + ['bagian']
TABS.update({name: LOW_COLUMNS for name in (
    'Skor Terendah', 'Kesalahan Skor Rendah', 'Kesalahan Skor Rendah OOF', 'Kesalahan Skor Rendah Test',
)})
TABS.update({name: ['fold', 'skor', 'n_training', 'bobot_loss']
             for name in ('Bobot Loss Organisasi', 'Bobot Loss Isi')})
TABS.update({name: HPO_COLUMNS for name in ('HPO Trials', 'HPO Konfirmasi', 'HPO Stabilitas')})
TABS.update({f'CM_{split}_{trait}': ['aktual', 'prediksi_1', 'prediksi_2', 'prediksi_3', 'prediksi_4']
             for split in ('OOF', 'Test') for trait in ('isi', 'organisasi', 'kebahasaan', 'mekanik')})
RUN_FIELDS = (
    'rows connected_groups development_rows test_rows split_sha256 notebook_code_sha256 notebook_code_id '
    'source_sha256 fold_selesai selection_scope cv_is_post_selection test_previously_inspected_in_research '
    'test_dipakai_untuk_seleksi_epoch test_dievaluasi test_ensemble kriteria_early_stopping e23_seed44_source_sha256 e25_seed44_source_sha256 e28_source_sha256 e29_source_sha256 inference_weights'
).split()


def sha(data):
    return hashlib.sha256(data).hexdigest()


def aggregate_tables(book):
    exported = {}
    for name, fields in TABS.items():
        if name not in book:
            continue
        rows = iter(book[name].values)
        header = next(rows)
        selected = [(i, value) for i, value in enumerate(header) if value in fields]
        if not selected:
            continue
        exported[name] = [{value: row[i] if i < len(row) else None for i, value in selected} for row in rows]
    return exported


def ingest(run_id, path):
    if not re.fullmatch(r'E\d+[A-Z]?(?:[-_][A-Za-z0-9]+)*', run_id):
        raise ValueError(f'ID run tidak sah: {run_id}')
    raw = path.read_bytes()
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        manifest_path, = [n for n in archive.namelist() if Path(n).name == 'manifest.json']
        manifest = json.loads(archive.read(manifest_path))
        tables, info, checksums = {}, {}, {}
        for item in manifest['workbooks']:
            name = item['file']
            if name not in ('04_Prediksi_dan_Evaluasi.xlsx', '05_Riwayat_Pelatihan.xlsx'):
                continue
            path_in_zip, = [n for n in archive.namelist() if Path(n).name == name]
            content = archive.read(path_in_zip)
            if sha(content) != item['sha256'] or len(content) != item['bytes']:
                raise ValueError(f'Workbook gagal verifikasi manifest: {name}')
            checksums[name] = item['sha256']
            book = load_workbook(io.BytesIO(content), read_only=True, data_only=True)
            try:
                if name.startswith('04'):
                    info = dict(list(book['Info'].values)[1:])
                tables.update(aggregate_tables(book))
            finally:
                book.close()
    if len(checksums) != 2 or 'Ringkasan CV' not in tables:
        raise ValueError('Arsip membutuhkan workbook 04–05 yang lengkap.')
    config = json.loads(info['final_config'])
    safe_config = {key: config[key] for key in CONFIG if key in config}
    for encoder in ('roberta', 'bge'):
        if encoder in safe_config:
            safe_config[encoder] = {key: safe_config[encoder][key] for key in ('checkpoint', 'revision')
                                    if key in safe_config[encoder]}
    if int(config['folds']) != tables['Ringkasan CV'][0].get('fold_selesai', config['folds']):
        raise ValueError('Arsip checkpoint harus berasal dari run dengan fold lengkap.')
    payload = {
        'run': run_id, 'experiment': config['experiment'],
        'source': {'name': path.name, 'sha256': sha(raw), 'workbook_sha256': checksums},
        'metadata': {key: info[key] for key in RUN_FIELDS if key in info},
        'config': safe_config, 'tables': tables,
    }
    target = ARCHIVE / f'{run_id}.json'
    content = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    if target.exists() and target.read_text() != content:
        raise ValueError(f'Arsip {run_id} sudah ada. Pakai ID baru untuk hasil yang berbeda.')
    target.write_text(content, encoding='utf-8')


def sheet(book, title, columns, rows, widths):
    tab = book.create_sheet(title)
    tab.append(columns)
    for row in rows:
        tab.append([json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value
                    for value in row])
        for cell in tab[tab.max_row]:
            if isinstance(cell.value, str):
                cell.data_type = 's'
                if cell.value.startswith('https://') and ' ' not in cell.value:
                    cell.hyperlink = cell.value
                    cell.style = 'Hyperlink'
            cell.alignment = Alignment(wrap_text=True, vertical='top')
            if tab.max_row % 2 == 0:
                cell.fill = PatternFill('solid', fgColor='EEF4F9')
    for cell in tab[1]:
        cell.fill = PatternFill('solid', fgColor='173554')
        cell.font = Font(bold=True, color='FFFFFF')
        cell.alignment = Alignment(wrap_text=True, vertical='center')
    tab.row_dimensions[1].height = 30
    for i, width in enumerate(widths, 1):
        tab.column_dimensions[get_column_letter(i)].width = width
    tab.freeze_panes = 'B2'
    tab.auto_filter.ref = tab.dimensions
    return tab


def references(name):
    with (DOCS / 'bukti' / name).open(encoding='utf-8', newline='') as stream:
        headers, *rows = csv.reader(stream, delimiter='\t')
    if any(len(row) != len(headers) for row in rows) or len({row[0] for row in rows}) != len(rows):
        raise ValueError(f'Tabel sumber tidak valid: {name}')
    return headers, rows


def notebook_index():
    notes = {}
    for path in sorted((GROUP / 'notebooks').glob('AES_Reset_E*.ipynb')):
        prefix = re.match(r'AES_Reset_(E\d+[A-Z]?)_', path.name)
        if not prefix:
            continue
        if 'Konfirmasi_Seed' in path.name:
            run_id = prefix[1] + '-' + path.stem.split('_')[-1]
        elif 'E14_S1_S2_Banding' in path.name:
            run_id = 'E14-S1S2'
        else:
            run_id = prefix[1]
        raw = path.read_bytes()
        notes[run_id] = (path.relative_to(ROOT).as_posix(), json.loads(raw), sha(raw))
    return notes


def inventory(notebook):
    results = {}
    for cell in notebook['cells']:
        role = cell.get('metadata', {}).get('aes_multitrait_role', '')
        if role not in ('hyperparameters', 'prepare', 'pilot_settings'):
            continue
        source = ''.join(cell['source'])
        if role == 'pilot_settings':
            for field, value in re.findall(r'^([A-Z][A-Z0-9_]+)\s*=\s*([^#\n]+)', source, re.M):
                if field in ('MODEL_ID', 'MODEL_REVISION', 'DEVICE', 'BATCH', 'MAX_TOKENS', 'AUDIT_THRESHOLD', 'REQUIRED_CRITERIA',
                             'DOC_STRIDE', 'MAX_ANSWER_TOKENS', 'REVIEW_SEED', 'REVIEW_PER_FOLD'):
                    results[field] = (value.strip(), 'pilot_nli', role)
        elif role == 'hyperparameters':
            for field, value in re.findall(r'^([A-Z][A-Z0-9_]+)\s*=\s*([^#\n]+)', source, re.M):
                if field in SETTINGS or re.search('LOSS|DROPOUT|GATE|ATTENTION|WARMUP|CLIP|SEED|LR|BATCH|TOKENS', field):
                    results[field] = (value.strip(), SETTINGS.get(field, field.lower()), role)
        elif role == 'prepare':
            for field, value in re.findall(r"['\"]([a-z_]+)['\"]:\s*([^,\n}]+)", source):
                if field in CONFIG:
                    results[field] = (value.strip(), field, role)
    return results


def project_url(path, branch='codex/aes-referensi-checkpoint'):
    return f'https://github.com/MalikHariyanto/Automated_Essay_Scoring/blob/{branch}/{path}'


def build_workbook():
    book = Workbook()
    book.remove(book.active)
    folders = notebook_index()
    records = {path.stem: json.loads(path.read_text()) for path in ARCHIVE.glob('E*.json')}
    paper_headers, papers = references('paper.tsv')
    decision_headers, decisions = references('keputusan.tsv')
    ids = {row[0] for row in papers}
    for row in decisions:
        if set(row[4].split('; ')) - ids:
            raise ValueError(f'ID rujukan tidak dikenal: {row[0]}')
    sheet(book, 'Mulai', ['Bagian', 'Cara membaca'], [
        ('Cakupan', 'Seri reset multi-trait di katalog repo. Nilai run diambil dari workbook yang tersedia. Konfigurasi historis tanpa ZIP dicatat sebagai deklarasi notebook.'),
        ('Filter literatur', 'Jendela 28 Nov 2021–28 Nov 2026. SJR berbasis Scopus Q1–Q3 atau SINTA 1–2 dan relevansi langsung multi-trait diperiksa terpisah. Bukti SJR adalah level venue; indeks setiap artikel belum dicek lewat Scopus.'),
        ('Bukti SJR', 'Snapshot sjrdata f7ed827, kolom tahun publikasi venue, Source ID dan kategori. Tahun/bidang tercantum pada tiap paper.'),
        ('Nilai lokal', 'Seed, LR, ambang, rasio bobot dan dimensi tidak dibenarkan hanya dengan judul paper. Lihat kolom asal angka, argumen, keluaran lokal dan batas.'),
        ('Checkpoints', 'Tiap JSON menyimpan tabel agregat, checksum workbook/ZIP, konfigurasi efektif, dan jejak versi. OOF dan test dibedakan.'),
        ('Konfirmasi E23', 'E21-B adalah checkpoint pembanding test S3 pada saat register dibuat. Status seed 44 mengikuti tab Peta notebook dan arsip run.'),
        ('Pembaruan', 'Tambah sumber dan keputusan ke TSV, arsipkan hasil agregat dengan builder, periksa data privat dan commit bersama notebook atau hasil terbaru.'),
    ], [25, 110])
    paper_tab = sheet(book, 'Paper dan indeks', paper_headers, papers, [10, 12, 36, 56, 42, 38, 20, 12, 40, 32, 42, 58, 62])
    decision_tab = sheet(book, 'Keputusan dan inisiasi', decision_headers, decisions, [12, 31, 54, 46, 20, 46, 60, 38])
    for tab in (paper_tab, decision_tab):
        for i in range(2, tab.max_row + 1):
            tab.row_dimensions[i].height = 74

    headers, rows = references('checkpoint.tsv')
    tab = sheet(book, 'Keputusan checkpoint', headers, rows, [23, 36, 80, 74])
    for i in range(2, tab.max_row + 1):
        tab.row_dimensions[i].height = 44
    notebooks, configured, actual, by_run, metrics, overview, hpo, low_scores = [], [], [], [], [], [], [], []
    extra = {
        'E17D': ('eksperimen/03_tiga_input_multi_trait/output/E17D_2026-11-25/README.md', 'codex/aes-referensi-checkpoint'),
        'E17E': ('eksperimen/03_tiga_input_multi_trait/notebooks/AES_Reset_E17E_Lanjut_Konfirmasi_dan_Test.ipynb', 'codex/e17e-optuna-gpu'),
        'E18A': ('eksperimen/03_tiga_input_multi_trait/README.md', 'codex/e18a-isi-bge-relevance'),
    }
    for run in sorted(set(folders) | set(records) | set(extra), key=lambda r: (int(re.match(r'E(\d+)', r)[1]), r)):
        archive = records.get(run)
        if run in folders:
            path, notebook, fingerprint = folders[run]
            intro = ''.join(notebook['cells'][0]['source']).splitlines()[0].lstrip('# ')
            link = project_url(path)
            code_hash = notebook.get('metadata', {}).get('aes_standalone', {}).get('code_sha256')
            for field, (value, key, role) in inventory(notebook).items():
                pilot_decision = {'E24B': 'NLI4', 'E24C': 'SPANQA'}.get(run, 'NLI')
                decision = pilot_decision if key == 'pilot_nli' else CONFIG.get(key, 'Lihat kode')
                if run == 'E28' and key in ('scalar_mix_layers', 'scalar_mix_gamma_initial'):
                    decision = 'SMIX4'
                if run == 'E29' and key.startswith('scalar_mix'):
                    decision = 'SMIXTRAIT'
                if run == 'E30' and key in ('hpo_recipe_trial', 'hpo_recipe_archive_sha256', 's3_encoder_lr',
                                           's3_mlp_lr', 's3_layers_lr', 'projection_lr', 'head_lr',
                                           's3_mlp_dropout', 'weight_decay', 'warmup_ratio'):
                    decision = 'TRIAL60TRANSFER'
                if run == 'E25' and key == 'isi_loss_ratio':
                    decision = 'PAIR44'
                configured.append((run, field, value, decision, role, link))
        else:
            default = (None, None)
            path, branch = extra.get(run, default)
            intro, fingerprint, code_hash = run, None, None
            link = project_url(path, branch) if path else None
        status = ('Hasil terarsip' if archive else 'Menunggu ZIP seed 44' if run == 'E23-Seed44'
                  else 'Hasil pilot terarsip' if run == 'E24' and (ARCHIVE / 'E24/Ringkasan_Pilot_NLI.json').exists()
                  else 'Hasil pilot terarsip' if run == 'E24B' and (ARCHIVE / 'E24B/Ringkasan_Pilot_Empat_Unsur.json').exists()
                  else 'Hasil pilot terarsip' if run == 'E24C' and (ARCHIVE / 'E24C/Ringkasan_Pilot_Ekstraksi_Bukti.json').exists()
                  else 'Arsip repo sebelumnya' if run in ('E1','E2','E3','E4','E17D')
                  else 'Dinonaktifkan' if run == 'E18A' else 'Resep tersedia')
        notebooks.append((run, intro, status, link, fingerprint, code_hash))
        if not archive:
            continue
        source = archive['source']
        meta = archive['metadata']
        result_link = project_url((ARCHIVE / f'{run}.json').relative_to(ROOT).as_posix())
        run_code = meta.get('notebook_code_sha256')
        if run == 'E14' and run_code == 'bbf4d90cd070913fd39fa2b86abc3ac2873443929d40e04390d28dd324a756a4':
            link = project_url('notebooks/multitrait_reset/AES_Reset_E14_S1_S2_S3_MultiTrait.ipynb',
                               '2f2690e6a78cf47fd7efcf6db115fd4507a8a18a')
            code_status = 'Cocok dengan revisi historis'
        elif code_hash and run_code:
            code_status = 'Hash cocok' if code_hash == run_code else 'Hash berbeda'
        else:
            code_status = 'Identitas kode dari laporan'
        by_run.append((run, archive['experiment'], source['name'], source['sha256'],
                       meta.get('split_sha256'), meta.get('notebook_code_sha256', meta.get('notebook_code_id')),
                       code_status, result_link, link))
        for key, value in archive['config'].items():
            actual.append((run, key, value, CONFIG[key], result_link))
        tables = archive['tables']
        for name in ('CV OOF per Trait', 'OOF A B per Trait', 'Test per Trait', 'Test A B per Trait'):
            if name == 'CV OOF per Trait' and 'OOF A B per Trait' in tables:
                continue
            for row in tables.get(name, []):
                metrics.append((run, name, *(row.get(key) for key in
                              ('pilihan', 'skenario', 'trait', 'n', 'RMSE', 'Pearson', 'QWK', 'Gwet AC2')), result_link))
        for row in tables.get('Ringkasan CV', []):
            choices = tables.get('Pilihan Checkpoint', [])
            for choice in sorted({r['pilihan'] for r in choices}) if choices else [None]:
                fit = [r for r in choices if r['pilihan'] == choice]
                qwks = [r['macro_QWK'] for r in fit if r.get('macro_QWK') is not None]
                overview.append((run, 'CV · mean fold', choice or row.get('skenario'),
                                 sum(r['RMSE'] for r in fit) / len(fit) if fit else row['mean_cv_RMSE'],
                                 sum(qwks)/len(qwks) if qwks else None, meta.get('development_rows'), result_link))
        for row in tables.get('Ringkasan Test A B', []):
            overview.append((run, 'Test · ensemble', row['pilihan'], row['RMSE'], row['macro_QWK'], row['n'], result_link))
        for name in ('Skor Terendah', 'Kesalahan Skor Rendah OOF',
                     'Kesalahan Skor Rendah Test', 'Isi 1 Latih dan Validasi B'):
            for row in tables.get(name, []):
                low_scores.append((run, name, row.get('pilihan'), row.get('skenario'), row.get('fold'),
                                   row.get('bagian'), row.get('trait', 'isi'),
                                   row.get('jumlah label 1', row.get('label 1', row.get('n_label_terendah'))),
                                   row.get('1 benar', row.get('label 1 terdeteksi', row.get('n_terdeteksi'))),
                                   row.get('label 2–4 salah jadi 1', row.get('skor di atas 1 salah jadi 1')),
                                   row.get('2 salah jadi 1'), result_link))
        for name in ('HPO Trials', 'HPO Konfirmasi', 'HPO Stabilitas'):
            for row in tables.get(name, []):
                keys = ('trial', 'seed', 'status', 'RMSE', 'mean_cv_RMSE', 'macro_QWK', 'encoder_lr',
                        'aux_lr', 'mlp_dropout', 'weight_decay', 'warmup_ratio', 'best_epoch', 'epoch_refit')
                hpo.append((run, name, *(row.get(key) for key in keys)))

    sheet(book, 'Peta notebook', ['Run', 'Rancangan di notebook', 'Status', 'Kode', 'SHA-256 file', 'Hash kode'], notebooks, [18, 69, 26, 34, 67, 67])
    sheet(book, 'Setelan notebook', ['Run', 'Variabel', 'Nilai atau ekspresi kode', 'ID keputusan', 'Cell', 'Kode'], configured, [18, 28, 40, 20, 25, 36])
    sheet(book, 'Konfigurasi run aktual', ['Run', 'Parameter', 'Nilai efektif', 'ID keputusan', 'Arsip'], actual, [18, 28, 66, 20, 36])
    sheet(book, 'Checkpoint dan checksum', ['Run', 'Eksperimen', 'ZIP sumber', 'SHA-256 ZIP', 'SHA-256 split',
          'Identitas kode run', 'Verifikasi kode', 'Arsip JSON', 'Notebook'], by_run, [18, 40, 54, 67, 67, 67, 28, 36, 36])
    sheet(book, 'Ringkasan hasil', ['Run', 'Bagian', 'Checkpoint / skenario', 'RMSE', 'Macro QWK', 'N', 'Arsip'], overview, [18, 26, 16, 16, 16, 12, 36])
    sheet(book, 'Empat metrik per trait', ['Run', 'Bagian', 'Checkpoint', 'Skenario', 'Trait', 'N', 'RMSE', 'Pearson', 'QWK',
          'Gwet AC2', 'Arsip'], metrics, [18, 24, 15, 13, 18, 10, 16, 16, 16, 16, 36])
    sheet(book, 'Deteksi skor rendah', ['Run', 'Tabel', 'Checkpoint', 'Skenario', 'Fold', 'Bagian', 'Trait',
          'N label rendah', 'Terdeteksi', 'Label 2–4 ke 1', 'Label 2 ke 1', 'Arsip'],
          low_scores, [18, 30, 16, 12, 10, 16, 16, 16, 16, 18, 16, 38])
    sheet(book, 'Bukti pencarian HPO', ['Run', 'Tahap', 'Trial', 'Seed', 'Status', 'RMSE pencarian', 'Mean CV RMSE',
          'Macro QWK', 'Encoder LR', 'Aux LR', 'Dropout', 'Weight decay', 'Warmup', 'Best epoch',
          'Epoch refit'], hpo, [16, 21, 12, 12, 16, 20, 20, 16, 18, 18, 16, 16, 16, 16, 16])
    book.save(BOOK)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', nargs=2, action='append', default=[], metavar=('RUN', 'PRIVATE_ZIP'))
    args = parser.parse_args()
    for run, path in args.archive:
        ingest(run, Path(path))
    build_workbook()


if __name__ == '__main__':
    main()
