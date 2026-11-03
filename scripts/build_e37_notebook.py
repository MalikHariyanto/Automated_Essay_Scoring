"""Build E37 paired learning-rate study from pinned E36 native presentation."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
BASE_COMMIT = '349f2e82e876333bf2270d47e7eb86b59e939a55'
BASE_PATH = 'notebooks/experiments/AES_300_E36_E30_S3_EncoderLR3e6_Native_Learnable_GPU.ipynb'
NAME = 'AES_300_E37_E30_S3_EncoderLR5e6_vs7e6_3Seed_Native_Learnable_GPU'
PLACEHOLDER = '__NOTEBOOK_CODE_SHA256__'

TRAINING_EXECUTE = '''if RUN_TRAINING:
    if not RUN_MODELS:raise RuntimeError('Aktifkan RUN_MODELS sebelum training.')
    # TAHAP A: dua LR × tiga seed × lima fold; belum refit atau evaluasi test.
    study_output = output
    study = run_paired_cv(bundle, manifest, bge, cfg, study_output)
    show(pd.DataFrame(study['selection']['per_seed']), 'CV setiap LR dan seed', rows=6)
    show(pd.DataFrame(study['selection']['summary']), 'Pemilihan LR dari CV', rows=2)
    show(pd.DataFrame(study['selection']['paired_differences']), 'Selisih CV kandidat−kontrol; negatif lebih baik', rows=3)
    # TAHAP B: tiga refit/test resep terpilih. Tidak memilih seed berdasarkan test.
    final = evaluate_selected_recipe(study, bundle, manifest, bge)
    primary = final['primary']
    cfg, output, report = primary['cfg'], Path(primary['folder']), primary['report']
    journal.metadata.update(run_folder=str(output), primary_seed=cfg['seed'],
                            selected_encoder_lr=cfg['s3_encoder_lr'], study_output=str(study_output))
    journal.save('01_konfigurasi', {
        'Parameter': pd.DataFrame([{'name': k, 'value': v} for k, v in cfg.items()]),
        'Versi': pd.DataFrame([{'library': p, 'version': version(p)} for p in ('torch','transformers','numpy','pandas','scikit-learn','openpyxl')]),
    }, 'Konfigurasi aktual resep terpilih; tabel standar memakai seed pertama yang ditetapkan sebelum training')
    export_completed_training(report, bundle, cfg, output, journal)
    export_paired_results(study, final, journal)
    show(pd.DataFrame(final['test_rows'])[['recipe','seed','encoder_lr','RMSE','Pearson','QWK','AC2']],
         'Test semua seed resep terpilih; bukan memilih seed terbaik', rows=3)
    show(pd.DataFrame(final['test_summary']), 'Mean dan SD test antar-seed', rows=4)
    print('Tabel Perbandingan Model, vektor, dan contoh prediksi memakai seed pertama:', cfg['seed'])
    print('Ini model tunggal, bukan ensemble. Seluruh seed tetap dilaporkan pada sheet tambahan.')
    comparison = test_score_comparison(report, bundle, cfg, output)
    show(comparison, 'Nilai guru dan prediksi seed utama yang ditentukan sebelumnya', rows=len(comparison), columns=8)
else:
    print('Training belum dijalankan; hasil CV/test belum tersedia.')
'''


def build():
    raw = subprocess.check_output(['git', 'show', BASE_COMMIT+':'+BASE_PATH], cwd=ROOT)
    notebook = copy.deepcopy(json.loads(raw))
    old_hash = notebook['metadata']['aes_standalone']['code_sha256']
    sections = (ROOT/'scripts/e37_native_sections.py').read_text()
    functions = [ast.get_source_segment(sections, n) for n in ast.parse(sections).body
                 if isinstance(n, ast.FunctionDef)]
    new_cells = []
    for cell in notebook['cells']:
        role = cell['metadata'].pop('aes36_role', '')
        if role:cell['metadata']['aes37_role'] = role
        text = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']
        text = text.replace(old_hash, PLACEHOLDER).replace(Path(BASE_PATH).stem, NAME)
        text = text.replace('E36', 'E37').replace('3e-6', '7e-6').replace('lr3e6', 'paired_lr5e6_7e6_3seed')
        if role == 'hyperparameters':
            text = text.replace('TRAIN_SEED = 42', '''TRAINING_SEEDS = (42, 142, 242)  # Sama untuk kontrol/kandidat; tidak dipilih dari test.
TRAIN_SEED = TRAINING_SEEDS[0]  # Seed utama untuk tabel dan vektor, bukan seed terbaik.
CONTROL_ENCODER_LR = 5e-6
CANDIDATE_ENCODER_LR = 7e-6''')
            text = text.replace('ENCODER_LR = 7e-6             # SATU perlakuan: encoder S3; baseline E30 = 5e-6.',
                                'ENCODER_LR = CANDIDATE_ENCODER_LR  # LR dipilih lewat CV dari kedua resep di atas.')
        if role == 'configuration_apply':
            text += '''
# Protokol ditentukan sebelum training dan ikut identitas cache setiap run.
cfg['paired_study'] = {
    'control_encoder_lr': CONTROL_ENCODER_LR,
    'candidate_encoder_lr': CANDIDATE_ENCODER_LR,
    'training_seeds': list(TRAINING_SEEDS),
    'selection': 'mean RMSE CV across all three training seeds; tie keeps control',
    'primary_seed': TRAIN_SEED, 'test_used_for_selection': False,
}
display(pd.DataFrame([{'Resep': label, 'Encoder LR': lr, 'Seed': seed}
    for label, lr in [('Kontrol E30', CONTROL_ENCODER_LR), ('Kandidat E37', CANDIDATE_ENCODER_LR)]
    for seed in TRAINING_SEEDS]))
print('Total: 30 latihan fold CV, lalu 3 refit resep terpilih. BGE frozen dihitung sekali dan dibagi bersama.')
'''
        if role == 'definition:train_all':
            text = text.replace('output, observer=None):', 'output, observer=None, cv_only=False):')
            text = text.replace('if final_path.exists():', 'if final_path.exists() and not cv_only:')
            text = text.replace('            completed.append(result)', '''            if cv_only:
                discard_completed_cv_progress(destination, fold['fold'])
            completed.append(result)''')
            anchor = "    dump(output/'selection_locked.json', selection)"
            assert text.count(anchor) == 1
            text = text.replace(anchor, anchor+'''
    if cv_only:
        report = {'source_sha256': bundle.source_sha256, 'split_sha256': manifest['split_sha256'],
                  'cv': cv_rows, 'selected_epochs': selected_epochs,
                  'test_evaluations_per_scenario': 0, 'stage': 'development CV only'}
        dump(output/'cv_report.json', report)
        return report''')
        if role == 'training_execute':text = TRAINING_EXECUTE
        if role in ('definition:evaluation_report', 'definition:training_report'):
            mapping = ([('Test_per_seed','Test per Seed'), ('Test_mean_SD_seed','Test Mean SD Seed'),
                        ('Prediksi_semua_seed','Prediksi Semua Seed')]
                       if role.endswith('evaluation_report') else
                       [('CV_semua_LR_seed','CV Semua LR Seed'), ('CV_per_seed','CV per Seed'),
                        ('CV_ringkasan_LR','CV Ringkasan LR'), ('Selisih_CV_berpasangan','Selisih CV Berpasangan'),
                        ('Seleksi_terkunci','Seleksi LR Terkunci'), ('Riwayat_semua_LR_seed','Riwayat Semua LR Seed')])
            append = "    for source_sheet, title in "+repr(mapping)+":\n"
            append += "        if source.has('16_paired_lr_seed.xlsx', source_sheet):\n"
            append += "            sheets[title] = source.copy('16_paired_lr_seed.xlsx', source_sheet)\n"
            text = text.replace('    return sheets', append+'    return sheets')
            if role.endswith('training_report'):
                text = text.replace("settings={'s3_mlp_lr'", "settings={'seed':'Seed tabel utama','paired_study':'Protokol LR dan seed','s3_mlp_lr'")
        cell['source'] = text
        if role == 'definition:train_all':
            new_cells.append({'cell_type':'markdown','metadata':{},'source':'''### 23A · Perbandingan LR dan seed yang terlihat

Fungsi berikut menjalankan dua resep pada tiga seed dengan fold yang sama. `select_lr_from_cv()` hanya membaca RMSE validasi; test tidak menjadi argumennya. Mean CV dipakai memilih LR; SD antar-fold dan SD antar-seed dilaporkan terpisah. Kontrol menang jika rerata sama. Ketiga seed resep terpilih dievaluasi; seed pertama selalu menjadi tabel/vektor utama, bukan seed dengan test terbaik.

Checkpoint dua slot untuk fold yang sudah selesai dibuang setelah JSON hasil CV tersimpan. Fold yang terputus tetap dapat dilanjutkan pada runtime yang sama. Jika runtime dihapus, file lokal ikut hilang; ini bukan penyimpanan Google Drive.
'''})
            for function in functions:
                name = ast.parse(function).body[0].name
                new_cells.append({'cell_type':'code','metadata':{'aes37_role':'definition:'+name},
                                  'execution_count':None,'outputs':[],'source':function+'\n'})
        new_cells.append(cell)
    notebook['cells'] = new_cells
    notebook['cells'][0]['source'] = '''# E37 · E30 S3 · LR encoder 5e-6 vs 7e-6 · tiga seed

**Satu faktor yang diuji: learning rate encoder RoBERTa.** Kontrol mengikuti resep E30 (`5e-6`); kandidat E37 memakai `7e-6`. MLP, proyeksi BGE, BiGRU/co-attention/pooling/norm semuanya `1e-4`. BGE-M3 frozen. Fusion concat → MLP, bukan Ridge. Model tetap memakai label `rater` saat training; label tidak masuk input encoder.

**S3 saja**, dengan tiga input `soal`, `jawaban_referensi`, `response_text`. RoBERTa fine-tuned → BiGRU → co-attention respons–referensi → attention pooling → gabung proyeksi BGE frozen → MLP → satu nilai0–100. S1/S2 tidak dijalankan dan tidak dimuat dari eksperimen lama.

**Tahap A:** kedua LR × seed42/142/242 ×5fold yang sama = **30latihan fold**. Maksimal25epoch/fold, early stopping patience5. Mean RMSE CV seluruh seed memilih resep; jika seri, pertahankan kontrol. SD antar-fold berbeda dari SD antar-seed. Ini bukan30split baru.

**Tahap B:** resep yang terpilih dari CV direfit pada240development untuk masing-masing tiga seed. Epoch tiap seed =ceil median best epoch5fold miliknya. Test60jawaban dievaluasi sekali per seed terpilih; semua hasil dilaporkan, tanpa memilih seed test terbaik atau ensemble. Tabel/vektor utama selalu seed42 yang ditetapkan sebelumnya. Test lama telah sering diamati, sehingga hasil tetap eksploratif, bukan konfirmasi pada data baru.

**Format E35/E36 native:** Python langsung per section, tanpa magic/worker tersembunyi, tanpa pemeriksaan cosine, clone repo, atau mount Drive. Output tetap **lima Excel standar**; tab tambahan pada04/05 menyimpan perbandingan seluruh LR/seed. Data pribadi tetap lokal.

Pilih runtime baru **T4 GPU → Run all → upload fix_dataset_300.xlsx**. Enam run CV lebih lama daripada E36. Run ulang pada runtime yang sama melanjutkan checkpoint/hasil cocok hash; runtime yang dihapus menghilangkan file lokal. Tidak ada jaminan kandidat mengalahkan E30.

Peta belajar: [nama model](#nama-model), [kolom X/y](#kolom-xy), [hyperparameter](#hyperparameter), [X/y aktual](#xy-aktual), [arsitektur](#arsitektur), [training](#training), [evaluasi](#evaluasi), [unduh hasil](#unduh).
'''
    for c in notebook['cells']:
        if c['cell_type'] != 'markdown':continue
        c['source'] = c['source'].replace('S3: encoder7e-6;', 'S3: encoder5e-6 (kontrol) atau7e-6 (kandidat);')
        if '## 5 · Hyperparameter' in c['source']:
            c['source'] = '<a id="hyperparameter"></a>\n## 5 · Hyperparameter E37\n\nKontrol encoder5e-6 dibanding kandidat7e-6. Tiga seed42/142/242, fold tetap. Semua kelompok lain1e-4. Nilai LR merupakan awal/puncak sebelum warmup dan scheduler linear. Ubah pengaturan di cell tepat di bawah ini, bukan mencari nilai di tengah model.\n'
        if '## 22 · Optimizer' in c['source']:
            c['source'] = '## 22 · Optimizer, loss MSE dan checkpoint\n\nEncoder memakai LR resep saat ini:5e-6 atau7e-6. MLP, proyeksi BGE dan pooling/norm1e-4. Semua parameter trainable masuk tepat satu kelompok optimizer. fit() tetap mengikuti E30.\n'
    fingerprint = hashlib.sha256('\n'.join(c['source'] for c in notebook['cells'] if c['cell_type']=='code').encode()).hexdigest()
    for i,c in enumerate(notebook['cells']):
        c['source'] = c['source'].replace(PLACEHOLDER, fingerprint)
        c['id'] = f'e37-{i:03d}-'+hashlib.sha256(c['source'].encode()).hexdigest()[:8]
        if c['cell_type']=='code':
            c['outputs']=[];c['execution_count']=None;ast.parse(c['source'])
    notebook['metadata']['colab']['name'] = NAME+'.ipynb'
    notebook['metadata']['aes_standalone']['code_sha256'] = fingerprint
    notebook['metadata']['aes_experiment'].update(
        name='E37', s3='E30 control5e-6 vs candidate7e-6; three paired training seeds',
        encoder_lr=7e-6, s3_encoder_lr=7e-6, control_encoder_lr=5e-6, candidate_encoder_lr=7e-6,
        training_seeds=[42,142,242], primary_seed=42, scenarios=['S3'],
        cv_fits=30, selected_recipe_refits=3, selection_uses_test=False, ensemble=False,
        s1='not run', s2='not run', mlp_lr_scope='S3 model.head.* only',
        automatic_checkpoint_deletion=True,
        checkpoint_deletion_policy='only completed CV-fold progress slots, after durable CV result JSON; final models retained',
        format_commit=BASE_COMMIT, format_path=BASE_PATH, format_sha256=hashlib.sha256(raw).hexdigest(),
        treatment='encoder LR only; paired training seeds with fixed holdout/folds; no best-test-seed selection')
    output = json.dumps(notebook,ensure_ascii=False,indent=1)+'\n'
    for directory in ('notebooks/experiments','aes-learning/notebooks/experiments'):
        (ROOT/directory/(NAME+'.ipynb')).write_text(output)
    print(json.dumps({'name':NAME,'cells':len(notebook['cells']),'code_sha256':fingerprint,'cv_fits':30,'refits':3}))
    return notebook


if __name__ == '__main__':build()
