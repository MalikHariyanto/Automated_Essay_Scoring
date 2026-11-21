"""Build the standalone E14 notebook from the E13 notebook and export layout."""

import hashlib
import json

from build_multitrait_reset_e6_notebook import BASE, GROUP, ROOT, replace
from build_multitrait_reset_e13_notebook import NAME as E13_NAME

NAME = 'AES_Reset_E14_S1_S2_S3_MultiTrait.ipynb'
MARKER = '__RESET_E14_CODE_SHA256__'
INTRO = """# E14 reset · S1, S2, S3 multi-trait · Baseline skripsi dan tuning LR

S1 memakai RoBERTa Indonesia fine-tuned. S2 memakai BGE-M3 frozen.
S3 menggabungkan 768 fitur RoBERTa dengan 256 fitur hasil proyeksi + LayerNorm BGE.
Ketiga skenario memakai MLP dengan satu hidden bersama (256 unit, GELU, dropout 0,1)
dan empat keluaran: isi, organisasi, kebahasaan, mekanik.

Gunakan workbook semi-final yang sama dengan E13. Semua skenario memakai soal,
kunci, jawaban, serta pembagian siswa dan lima fold yang sama. Skor guru menjadi
target MSE yang dinormalisasi per trait. Seleksi memakai rata-rata RMSE per trait
pada validasi dalam skala asli.

Uji paket A/B dari Tabel 3.4 pada seed 42, lalu konfirmasi paket terpilih
pada seed 2026 dan 3407. Refit tiga seed per skenario dan rata-ratakan prediksi
kontinu sebelum menghitung hasil test utama. Total 60 fit development + 9 refit.
Fit yang sudah selesai dapat dipakai ulang pada folder output yang sama.

Pilih T4 GPU. ZIP ringkas 04–05 muncul sesudah evaluasi test, sebelum inspeksi
vektor dan ZIP lengkap 01–05. ZIP lengkap memakai checkpoint S3 seed 42 untuk
ilustrasi vektor final. Hasil siswa tersimpan privat.
"""

HYPERPARAMETERS = """# Tabel 3.4 naskah skripsi; paket BGE tunggal hanya memakai LR head.
EPOCHS = 10
FOLDS = 5
EARLY_STOPPING_PATIENCE = 2
TRAIN_SEED = 42
CONFIRMATION_SEEDS = (2026, 3407)
SPLIT_SEED = 2026
TEST_FRACTION = 0.20
MICRO_BATCH = 1
EFFECTIVE_BATCH = 8

LR_PACKAGES = {
    'A': {'encoder_lr': 2e-5, 'head_lr': 1e-3},
    'B': {'encoder_lr': 1e-5, 'head_lr': 5e-4},
}
DROPOUT = 0.1
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.10
GRADIENT_CLIP = 1.0

ROBERTA_MAX_TOKENS = 512
BGE_MAX_TOKENS = 1024
WINDOW_OVERLAP = 128
CONTEXT_MAX_TOKENS = 192
QUESTION_MAX_TOKENS = 64
DEVICE = 'cuda'
"""

MODEL = """class MultiTraitModel(nn.Module):
    def __init__(self, scenario, roberta, dropout=0.1):
        super().__init__()
        if scenario not in SCENARIOS:
            raise ValueError(f'Skenario tidak dikenal: {scenario}')
        self.scenario, self.roberta = scenario, roberta
        if self.roberta is not None:
            self.roberta.requires_grad_(True)
        if scenario == 'S3':
            self.roberta_norm = nn.LayerNorm(768)
            self.projection = nn.Sequential(nn.Linear(1024, 256), nn.LayerNorm(256))
        self.head = nn.Sequential(
            nn.Linear(768 if scenario == 'S1' else 1024, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, len(TRAITS)),
        )

    def forward(self, windows, mapping, n_essays, bge, return_features=False):
        features = {}
        if self.roberta is not None:
            hidden = self.roberta(**windows).last_hidden_state
            cls = mean_by_essay(hidden[:, 0, :], mapping, n_essays)
            features['roberta'] = cls
        if self.scenario == 'S1':
            representation = cls
        elif self.scenario == 'S2':
            representation = bge
        else:
            projected = self.projection(bge)
            representation = torch.cat([self.roberta_norm(cls), projected], dim=1)
            features.update(bge=bge, projection=projected, concat=representation)
        prediction = self.head(representation)
        return (prediction, features) if return_features else prediction
"""

CV = """def candidate_config(cfg, candidate):
    rates = cfg['lr_packages'][candidate]
    if set(rates) != {'encoder_lr', 'head_lr'} or any(
        not math.isfinite(value) or value <= 0 for value in rates.values()
    ):
        raise ValueError(f'Paket learning rate tidak sah: {candidate}')
    return {**cfg, **rates}


def run_cv_seed(scenario, candidate, seed, bundle, manifest, bge, cfg, output):
    run_cfg = candidate_config(cfg, candidate)
    results = []
    for fold in manifest['folds']:
        number, validation = fold['fold'], fold['validation']
        path = output / 'cv' / scenario / candidate / str(seed) / f'fold_{number}.json'
        identity = {
            'split': manifest['split_sha256'], 'fold': number, 'validation': validation,
            'scenario': scenario, 'candidate': candidate, 'seed': seed,
        }
        print(f'{scenario} · paket {candidate} · seed {seed} · fold {number}', flush=True)
        if path.exists():
            result = json.loads(path.read_text())
            if result['identity'] != identity:
                raise ValueError(f'Identitas cache fold berbeda: {path}')
            print('Hasil fold yang selesai dipakai ulang.', flush=True)
        else:
            model, engine, info = fit(scenario, fold['train'], validation, bundle, bge,
                                      run_cfg, cfg['epochs'], seed)
            prediction = predictions_array(engine.predict(validation), validation)
            result = {'identity': identity, 'prediction': prediction.tolist(), 'training': info}
            dump(path, result)
            del model, engine
            release()
        prediction = predictions_array(result['prediction'], validation)
        evaluation = metrics(bundle.y[validation], prediction)
        if not math.isclose(evaluation['RMSE_macro'],
                            result['training']['best_validation_rmse'], abs_tol=1e-6):
            raise ValueError(f'Prediksi dan epoch terpilih tidak cocok: {path}')
        results.append({**result, 'metrics': evaluation})
    indices = [i for result in results for i in result['identity']['validation']]
    if sorted(indices) != sorted(manifest['development']):
        raise ValueError('Setiap jawaban development harus memiliki tepat satu prediksi OOF per seed.')
    return results


def cv_mean(folds, metric='RMSE_macro'):
    return float(np.mean([result['metrics'][metric] for result in folds]))


def run_cv(bundle, manifest, bge, cfg, output):
    summaries, folds, selected = [], [], {}
    initial = cfg['seed']
    for scenario in SCENARIOS:
        tuning = {
            name: run_cv_seed(scenario, name, initial, bundle, manifest, bge, cfg, output)
            for name in cfg['lr_packages']
        }
        winner = min(tuning, key=lambda name: (round(cv_mean(tuning[name]), 4), name))
        confirmation = {
            seed: run_cv_seed(scenario, winner, seed, bundle, manifest, bge, cfg, output)
            for seed in cfg['confirmation_seeds']
        }
        runs = [(name, initial, results) for name, results in tuning.items()]
        runs.extend((winner, seed, results) for seed, results in confirmation.items())
        chosen_folds = [result for name, _, results in runs if name == winner for result in results]
        epochs = math.ceil(np.median([result['training']['best_epoch'] for result in chosen_folds]))
        selected[scenario] = {'candidate': winner, 'epochs': epochs}
        for name, seed, results in runs:
            rates = cfg['lr_packages'][name]
            summaries.append({
                'eksperimen': 'Reset_E14', 'skenario': scenario, 'kandidat': name, 'seed': seed,
                'tahap': 'tuning' if seed == initial else 'konfirmasi', 'terpilih': name == winner,
                'LR_encoder': rates['encoder_lr'] if scenario != 'S2' else None,
                'LR_head': rates['head_lr'], 'LR_proyeksi': rates['head_lr'] if scenario == 'S3' else None,
                'mean_cv_RMSE': cv_mean(results, 'RMSE'),
                'mean_cv_RMSE_macro': cv_mean(results),
                'sd_cv_RMSE_macro': float(np.std([r['metrics']['RMSE_macro'] for r in results], ddof=1)),
                'epoch_refit': epochs if name == winner else None,
            })
            folds.extend(results)
    dump(output / 'selected_configurations.json', selected)
    return summaries, folds, selected
"""

REFIT = """def refit_checkpoint(scenario, candidate, epochs, seed, bundle, manifest, bge, cfg, output):
    identity = {'scenario': scenario, 'candidate': candidate, 'seed': seed,
                'refit_epochs': epochs, 'split': manifest['split_sha256']}
    output.mkdir(parents=True, exist_ok=True)
    model_path = output / 'final.pt'
    refit_path = output / 'refit_result.json'
    if refit_path.is_file():
        result = json.loads(refit_path.read_text())
        if (result['identity'] != identity or not model_path.is_file()
                or digest(model_path) != result['model_sha256']):
            raise ValueError(f'Metadata refit dan checkpoint final berbeda: {output}')
        print(f'{scenario} · seed {seed}: checkpoint selesai dipakai ulang.', flush=True)
        return result

    print(f'Refit {scenario} · paket {candidate} · seed {seed} · {epochs} epoch.', flush=True)
    run_cfg = candidate_config(cfg, candidate)
    model, engine, info = fit(scenario, manifest['development'], None, bundle, bge,
                              run_cfg, epochs, seed)
    pending = model_path.with_suffix('.pending.pt')
    save_model(pending, model, run_cfg, bundle, manifest, info)
    pending.replace(model_path)
    result = {'identity': identity, 'training': info, 'model_sha256': digest(model_path)}
    dump(refit_path, result)
    del model, engine
    release()
    return result
"""

REPORT = """def cv_keys(identity):
    return {'skenario': identity['scenario'], 'kandidat': identity['candidate'], 'seed': identity['seed']}


def export_results(path, bundle, manifest, cfg, cv_summary, folds, refits, test_runs, inspection=None):
    cv_fold, cv_oof, oof_rows, history, lowest = [], [], [], [], []
    runs = {}
    for result in folds:
        identity = result['identity']
        keys = cv_keys(identity)
        fold, valid = identity['fold'], identity['validation']
        info = result['training']
        cv_fold.extend(trait_rows(result['metrics'], **keys, fold=fold,
                                  epoch_terpilih=info['best_epoch'], n=len(valid)))
        oof_rows.extend(prediction_rows(bundle, valid, result['prediction'], **keys, fold=fold))
        history.extend(history_rows(info, **keys, fold=fold))
        lowest.extend({**keys, 'fold': fold, **row} for row in
                      lowest_score_rows(bundle, valid, result['prediction'], 'fold_validasi'))
        indices, predictions = runs.setdefault(tuple(keys.values()), ([], []))
        indices.extend(valid)
        predictions.extend(result['prediction'])
    for (scenario, candidate, seed), (indices, predictions) in runs.items():
        keys = {'skenario': scenario, 'kandidat': candidate, 'seed': seed}
        cv_oof.extend(trait_rows(metrics(bundle.y[indices], predictions), **keys, n=len(indices)))
        lowest.extend({**keys, **row} for row in lowest_score_rows(bundle, indices, predictions, 'OOF'))

    test = manifest['test']
    test_summary, test_metrics, test_rows = [], [], []
    for keys, prediction in test_runs:
        evaluation = metrics(bundle.y[test], prediction)
        test_summary.append({**keys, 'n': len(test), 'RMSE': evaluation['RMSE'],
                             'RMSE_macro': evaluation['RMSE_macro']})
        test_metrics.extend(trait_rows(evaluation, **keys, n=len(test)))
        test_rows.extend(prediction_rows(bundle, test, prediction, **keys))
        lowest.extend({**keys, **row} for row in lowest_score_rows(bundle, test, prediction, 'test'))
    for result, _ in refits:
        history.extend(history_rows(result['training'], **cv_keys(result['identity']), fold='refit'))

    evaluation = {
        'Ringkasan CV': pd.DataFrame(cv_summary),
        'CV per Fold dan Trait': pd.DataFrame(cv_fold),
        'CV OOF per Trait': pd.DataFrame(cv_oof),
        'Prediksi OOF': pd.DataFrame(oof_rows),
        'Ringkasan Test': pd.DataFrame(test_summary),
        'Test per Trait': pd.DataFrame(test_metrics),
        'Prediksi Test': pd.DataFrame(test_rows),
        'Skor Terendah': pd.DataFrame(lowest),
    }
"""

REFIT_EXECUTE = """if model_stage_ready('Refit model', ('bge', 'cv_summary', 'folds', 'selected')):
    refits = []
    for scenario, choice in selected.items():
        for seed in (cfg['seed'], *cfg['confirmation_seeds']):
            folder = output / 'refit' / scenario / choice['candidate'] / str(seed)
            result = refit_checkpoint(scenario, choice['candidate'], choice['epochs'], seed,
                                      bundle, manifest, bge, cfg, folder)
            refits.append((result, folder))
    display(pd.DataFrame([result['identity'] for result, _ in refits]))
    print(f'{len(refits)} checkpoint final: {output / "refit"}')
"""

TEST_EXECUTE = """if model_stage_ready('Evaluasi test', ('bge', 'refits')):
    test_runs = []
    for scenario in SCENARIOS:
        individual = []
        for result, folder in refits:
            identity = result['identity']
            if identity['scenario'] == scenario:
                run_cfg = candidate_config(cfg, identity['candidate'])
                scores = evaluate_test(result, bundle, manifest, bge, run_cfg, folder)
                individual.append(scores)
                test_runs.append(({**cv_keys(identity), 'jenis_prediksi': 'per_seed'}, scores))
        if len(individual) != 1 + len(cfg['confirmation_seeds']):
            raise ValueError(f'Checkpoint final belum lengkap: {scenario}')
        average = np.mean(individual, axis=0)
        keys = {'skenario': scenario, 'kandidat': selected[scenario]['candidate'],
                'seed': None, 'jenis_prediksi': 'ensemble'}
        test_runs.append((keys, average))
        display(pd.DataFrame(trait_rows(metrics(bundle.y[manifest['test']], average), **keys)))

    quick_path = output / 'Hasil_AES_Reset_E14_Ringkas_04_05.zip'
    export_results(quick_path, bundle, manifest, cfg, cv_summary, folds, refits, test_runs)
    print(f'ZIP ringkas siap diunduh: {quick_path}')
    if 'google.colab' in sys.modules:
        from google.colab import files
        files.download(str(quick_path))
"""


def update_cell(role, source):
    if role == 'configuration_defaults':
        return replace(source, {
            '# Maksimal 25 epoch/fold; berhenti bila RMSE validasi tidak membaik selama patience.':
                '# Refit dan seleksi mengikuti Tabel 3.4 skripsi.',
            "'encoder_lr': ENCODER_LR, 's3_encoder_lr': ENCODER_LR, 'projection_lr': PROJECTION_LR, 'head_lr': 1e-4,":
                "'lr_packages': LR_PACKAGES, 'confirmation_seeds': CONFIRMATION_SEEDS,",
            ", 's3_mlp_dropout': S3_MLP_DROPOUT": "",
            "    's3_mlp_lr': S3_MLP_LR, 's3_layers_lr': S3_LAYERS_LR,\n": "",
            "('encoder_lr', 's3_encoder_lr', 'projection_lr', 'head_lr', 's3_mlp_lr', 's3_layers_lr', 'gradient_clip', 'bge_batch')":
                "('gradient_clip', 'bge_batch')",
            "Learning rate, gradient clip, dan batch BGE": "Gradient clip dan batch BGE",
            "    if not 0 <= result['s3_mlp_dropout'] < 1:\n        raise ValueError('s3_mlp_dropout harus berada pada rentang [0,1).')\n": "",
        })
    if role == 'prepare':
        return replace(source, {
            "'head_type': 'independent_trait_heads', 'trait_hidden': 64":
                "'head_type': 'shared_mlp', 'hidden': 256",
            "'fusion_variant': 'organisasi_prompt_ablation', 'trait_corr_threshold': 0.7, 'trait_relation_weight': 0.0,":
                "'fusion_variant': 'concat',",
        })
    if role == 'definition:Engine':
        start = source.index('    def forward(')
        end = source.index('    def predict(')
        source = source[:start] + """    def forward(self, indices, return_features=False):
        ids = [int(i) for i in indices]
        features = self.bge[ids].to(self.device) if self.model.scenario != 'S1' else None
        windows = mapping = None
        if self.model.scenario != 'S2':
            windows, mapping, _ = pair_batch_windows(
                [self.texts[i] for i in ids], [self.contexts[i] for i in ids],
                self.tokenizer, self.device, self.cfg['roberta_max_length'],
                self.cfg['roberta_overlap'], self.cfg)
        return self.model(windows, mapping, len(ids), features, return_features)

""" + source[end:]
        return replace(source, {
            'Jalankan tiga teks melalui model fusion S3 yang sama.':
                'Bangun input hanya untuk cabang encoder yang dipakai.',
            "        if model.scenario != 'S3':\n            raise ValueError('Engine E43 hanya menjalankan S3.')\n": '',
            "self.tokenizer = roberta_tokenizer or tokenizer(cfg, 'roberta')":
                "self.tokenizer = (roberta_tokenizer or tokenizer(cfg, 'roberta')) if model.scenario != 'S2' else None",
        })
    if role == 'definition:make_model':
        return """def make_model(scenario, cfg):
    roberta = encoder(cfg, 'roberta') if scenario in ('S1', 'S3') else None
    return MultiTraitModel(scenario, roberta, dropout=cfg['dropout'])
"""
    if role == 'architecture':
        return """architecture = pd.DataFrame([
    ('S1', 'RoBERTa fine-tuned', 768, 256, 4),
    ('S2', 'BGE frozen', 1024, 256, 4),
    ('S3', 'RoBERTa fine-tuned + proyeksi BGE frozen (1024→256)', 1024, 256, 4),
], columns=['skenario', 'representasi', 'input_MLP', 'hidden_MLP', 'output'])
display(architecture)
"""
    if role == 'metrics':
        return replace(source, {
            "return {'RMSE': float(np.sqrt(np.mean(residual ** 2))), 'traits': per_trait}":
                """return {'RMSE': float(np.sqrt(np.mean(residual ** 2))),
            'RMSE_macro': float(np.mean([trait['RMSE'] for trait in per_trait.values()])),
            'traits': per_trait}""",
        })
    if role == 'definition:optimizer_groups':
        start = source.index('    rates =')
        end = source.index('    result =', start)
        source = source[:start] + """    rates = {name: cfg['head_lr'] for name in buckets}
    rates['roberta'] = cfg['encoder_lr']
""" + source[end:]
        return source.replace('Lima kelompok optimizer S3', 'Kelompok optimizer per skenario')
    if role == 'definition:fit':
        start = source.index('    pair_indices =')
        end = source.index('    # Buat model baru', start)
        source = source[:start] + "    seed_everything(seed)\n    device = device_name(cfg)\n" + source[end:]
        return replace(source, {
            'loss_total = mse_total = relation_total = 0.': 'mse_total = 0.',
            """                    mse = torch.mean((prediction-targets)**2)
                    relation = relation_error(prediction, targets, pairs)
                    loss = mse + cfg['trait_relation_weight'] * relation""":
                '                    loss = torch.mean((prediction - targets)**2)',
            """                loss_total += loss.item()*len(ids)
                mse_total += mse.item()*len(ids)
                relation_total += relation.item()*len(ids)""":
                '                mse_total += loss.item()*len(ids)',
            """'train_relation_loss': relation_total/len(training),
               'train_total_loss': loss_total/len(training), 'active_trait_pairs': active_pairs,""":
                """'train_relation_loss': 0.0,
               'train_total_loss': mse_total/len(training), 'active_trait_pairs': [],""",
            "if row['validation']['RMSE'] < best_rmse:":
                "if row['validation']['RMSE_macro'] < best_rmse:",
            "best_rmse, best_epoch = row['validation']['RMSE'], epoch":
                "best_rmse, best_epoch = row['validation']['RMSE_macro'], epoch",
            "{row['validation']['RMSE']:.4f}": "{row['validation']['RMSE_macro']:.4f}",
            "RMSE validasi:": "RMSE macro validasi:",
            "minimum validation RMSE in original score units": "minimum mean per-trait validation RMSE in original score units",
        })
    if role == 'cv_execute':
        return """if model_stage_ready('Lima fold CV', ('bge',)):
    cv_summary, folds, selected = run_cv(bundle, manifest, bge, cfg, output)
    display(pd.DataFrame(cv_summary))
"""
    if role == 'test_definition':
        return replace(source, {
            "model = make_model('S3', cfg)": "model = make_model(refit_result['identity']['scenario'], cfg)",
        })
    if role == 'report':
        start = source.index('def export_results(')
        end = source.index('    groups =', start)
        source = source[:start] + REPORT + source[end:]
        return replace(source, {
            "'validation_RMSE': row.get('validation', {}).get('RMSE')}":
                """'validation_RMSE': row.get('validation', {}).get('RMSE'),
             'validation_RMSE_macro': row.get('validation', {}).get('RMSE_macro')}""",
            "'kriteria_early_stopping': 'Akar rata-rata kuadrat error seluruh skor validasi dalam satuan asli.',":
                """'kriteria_early_stopping': 'Rata-rata RMSE empat trait pada validasi dalam satuan asli.',
        'kriteria_paket_lr': 'Rata-rata RMSE macro lima fold seed awal; seri empat desimal memilih A.',
        'jenis_hasil_test_utama': 'ensemble: rata-rata prediksi kontinu tiga seed',
        'inspeksi_model_final': f'S3 seed {cfg["seed"]} sebagai contoh vektor final',""",
        })
    if role == 'final_features':
        start = source.index('    inspection_sheets["Arsitektur"]')
        return replace(source[:start], {
            "('bge', 'refit_result')": "('bge', 'refits', 'inspection_sheets')",
            '    payload = torch.load(output / "final.pt", map_location="cpu", weights_only=True)':
                """    result, s3_folder = next((result, folder) for result, folder in refits
                             if result['identity']['scenario'] == 'S3'
                             and result['identity']['seed'] == cfg['seed'])
    checkpoint = s3_folder / 'final.pt'
    if digest(checkpoint) != result['model_sha256']:
        raise ValueError('Checkpoint S3 untuk inspeksi berubah.')
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)""",
            'Fitur model final untuk inspeksi seluruh jawaban.':
                'Fitur S3 seed awal untuk inspeksi seluruh jawaban.',
            '        output,\n        context=': '        s3_folder,\n        context=',
        }) + '    inspection_sheets["Arsitektur"] = architecture\n'
    if role == 'export_execute':
        return replace(source, {
            "('test_prediction', 'inspection_sheets')": "('test_runs', 'inspection_sheets')",
            'Hasil_AES_Reset_E14_S3_MultiTrait.zip': 'Hasil_AES_Reset_E14_S1_S2_S3_MultiTrait.zip',
            'final_info, test_prediction, inspection=': 'refits, test_runs, inspection=',
        })
    if role == 'download':
        return replace(source, {
            '    from google.colab import files\n    files.download(str(report_path))':
                "    if 'google.colab' in sys.modules:\n        from google.colab import files\n        files.download(str(report_path))",
        })
    return source


def build():
    base = BASE.parent / E13_NAME
    raw = base.read_bytes()
    notebook = json.loads(raw)
    previous_hash = notebook['metadata']['aes_standalone']['code_sha256']
    sections = {
        'hyperparameters': HYPERPARAMETERS,
        'definition:SingleScoreModel': MODEL,
        'cv_definition': CV,
        'final_model': REFIT,
        'refit_execute': REFIT_EXECUTE,
        'test_execute': TEST_EXECUTE,
    }
    for index, cell in enumerate(notebook['cells']):
        source = ''.join(cell['source']).replace(previous_hash, MARKER)
        source = (source.replace('Reset_E13', 'Reset_E14')
                  .replace('E13 reset', 'E14 reset')
                  .replace('reset_e13_s3_multitrait', 'reset_e14_s1_s2_s3_multitrait')
                  .replace('aes_reset_e13_organisasi_prompt_ablation', 'aes_reset_e14_s1_s2_s3')
                  .replace("SCENARIOS = ('S3',)  # Hanya model fusion yang dijalankan.",
                           "SCENARIOS = ('S1', 'S2', 'S3')"))
        role = cell['metadata'].get('aes_multitrait_role')
        if cell['cell_type'] == 'markdown':
            source = (source.replace('### Model fusion multi-trait', '### Model S1, S2, dan fusion S3')
                      .replace('Refit E14 reset pada seluruh development', 'Refit paket terpilih pada seluruh development'))
        cell['source'] = sections[role] if role in sections else update_cell(role, source)
        cell['id'] = f'reset-e14-{index}'

    notebook['cells'][0]['source'] = INTRO
    markdown = {
        60: '## 12 · Model multi-trait untuk S1, S2, dan S3',
        61: 'S1 dan S2 memakai satu encoder. S3 memakai concat RoBERTa dan proyeksi BGE. Setiap skenario mempunyai MLP 256 dengan empat keluaran.',
        82: '### `run_cv`\nUji paket A dan B pada seed awal, lalu konfirmasi paket terpilih pada seed tambahan.',
        84: '### Jalankan lima fold per paket dan seed',
        114: 'Latih setiap skenario dengan paket terpilih sebelum membuka evaluasi test.',
        116: '## 18 · Evaluasi test\nHitung prediksi per seed dan rata-rata prediksi kontinu sebagai hasil utama.',
        127: '## 19 · Inspeksi vektor dan susun ZIP lengkap\nUntuk ilustrasi fitur final, gunakan satu checkpoint S3 dari seed awal. Metrik utama memakai prediksi ensemble.',
    }
    for index, source in markdown.items():
        if notebook['cells'][index]['cell_type'] != 'markdown':
            raise ValueError('Posisi bagian notebook E13 berubah.')
        notebook['cells'][index]['source'] = source

    code = '\n'.join(cell['source'] for cell in notebook['cells'] if cell['cell_type'] == 'code')
    fingerprint = hashlib.sha256(code.encode()).hexdigest()
    for cell in notebook['cells']:
        cell['source'] = cell['source'].replace(MARKER, fingerprint)
        if cell['cell_type'] == 'code':
            compile(cell['source'], cell['id'], 'exec')
    notebook['metadata']['colab']['name'] = NAME
    notebook['metadata']['aes_standalone'] = {
        'code_sha256': fingerprint, 'base': base.relative_to(ROOT).as_posix(),
        'base_sha256': hashlib.sha256(raw).hexdigest(), 'experiment': 'Reset_E14',
    }
    content = json.dumps(notebook, ensure_ascii=False, indent=1) + '\n'
    for folder in (BASE.parent, GROUP / 'notebooks'):
        (folder / NAME).write_text(content, encoding='utf-8')


if __name__ == '__main__':
    build()
