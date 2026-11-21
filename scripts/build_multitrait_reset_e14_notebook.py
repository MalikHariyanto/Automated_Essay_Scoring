"""Build the standalone E14 notebook from the E13 notebook and export layout."""

import hashlib
import json

from build_multitrait_reset_e6_notebook import BASE, GROUP, ROOT, replace
from build_multitrait_reset_e13_notebook import NAME as E13_NAME

NAME = 'AES_Reset_E14_S1_S2_S3_MultiTrait.ipynb'
MARKER = '__RESET_E14_CODE_SHA256__'
INTRO = """# E14 reset · S1, S2, S3 multi-trait · Pembanding concat dengan head bersama

S1 memakai RoBERTa Indonesia fine-tuned. S2 memakai BGE-M3 frozen.
S3 memakai concat fitur RoBERTa 768 dan proyeksi BGE 256.
Setiap skenario memakai MLP dengan satu hidden bersama 256 (GELU, dropout 0,2)
dan empat keluaran: isi, organisasi, kebahasaan, mekanik.

Resep Colab E13 berlaku untuk semua skenario: lima fold kelompok siswa yang sama,
maksimal 25 epoch, patience 5, pemilihan berdasarkan RMSE gabungan dalam satuan
skor asli, dan MSE dengan target ternormalisasi. LR RoBERTa 5e-6, head/proyeksi
1e-4. Seed dasar 42, CV memakai seed dasar + nomor fold seperti E13, dan satu
refit seed 42 per skenario. Total 15 fit CV dan 3 refit.

Pilih T4 GPU dan gunakan workbook semi-final E13. ZIP utama S3 memakai susunan
E13: ringkas 04–05 dan lengkap 01–05. S1 dan S2 memperoleh ZIP ringkas masing-masing
dengan format workbook yang sama. Semua hasil siswa tersimpan privat.
"""

MODEL = """class MultiTraitModel(nn.Module):
    def __init__(self, scenario, roberta, dropout=0.2):
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

CV_EXECUTE = """if model_stage_ready('Lima fold CV', ('bge',)):
    cv_results = {}
    for scenario in SCENARIOS:
        folder = output / scenario
        summary, folds = run_cv(bundle, manifest, bge, cfg, folder, scenario)
        cv_results[scenario] = (summary, folds)
        print(f'{scenario}: ringkasan lima fold')
        display(pd.DataFrame([summary]))
"""

REFIT_EXECUTE = """if model_stage_ready('Refit model', ('bge', 'cv_results')):
    refits = {}
    for scenario, (summary, _) in cv_results.items():
        folder = output / scenario
        result = refit_checkpoint(summary['epoch_refit'], bundle, manifest, bge, cfg, folder, scenario)
        refits[scenario] = result
        display(pd.DataFrame(result['training']['history']))
        print(f'{scenario}: checkpoint final {folder / "final.pt"}')
"""

TEST_EXECUTE = """if model_stage_ready('Evaluasi test', ('bge', 'cv_results', 'refits')):
    test_predictions, quick_paths = {}, {}
    for scenario, refit_result in refits.items():
        folder = output / scenario
        summary, folds = cv_results[scenario]
        prediction = evaluate_test(refit_result, bundle, manifest, bge, cfg, folder)
        print(f'{scenario}: {len(prediction)} jawaban test dievaluasi.')
        display(pd.DataFrame(trait_rows(metrics(bundle.y[manifest['test']], prediction))))
        suffix = '' if scenario == 'S3' else f'{scenario}_'
        quick_path = output / f'Hasil_AES_Reset_E14_{suffix}Ringkas_04_05.zip'
        export_results(quick_path, bundle, manifest, {**cfg, 'scenario': scenario},
                       summary, folds, refit_result['training'], prediction)
        test_predictions[scenario] = prediction
        quick_paths[scenario] = quick_path
        print(f'{scenario}: ZIP ringkas {quick_path}')

    # ZIP lengkap memakai model dan laporan S3, seperti alur E13.
    cv_summary, folds = cv_results['S3']
    refit_result = refits['S3']
    final_info = refit_result['training']
    test_prediction = test_predictions['S3']
    if 'google.colab' in sys.modules:
        from google.colab import files
        files.download(str(quick_paths['S3']))
"""


def update_cell(role, source):
    if role == 'hyperparameters':
        return replace(source, {
            '# HIPERPARAMETER BASELINE E43; Scalar Mix empat layer terakhir.':
                '# Resep training diwarisi dari Colab E13.',
        })
    if role == 'prepare':
        return replace(source, {
            "'head_type': 'independent_trait_heads', 'trait_hidden': 64":
                "'head_type': 'shared_mlp', 'trait_hidden': 256",
            "'fusion_variant': 'organisasi_prompt_ablation'": "'fusion_variant': 'concat'",
        })
    if role == 'definition:SingleScoreModel':
        # Pertahankan diagnostik loss-relasi E13, termasuk riwayat loss ZIP 05.
        diagnostics = source[source.index('def correlated_trait_pairs('):source.index('class MultiTraitModel(')]
        return diagnostics + MODEL
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
    return MultiTraitModel(scenario, roberta, dropout=cfg['s3_mlp_dropout'])
"""
    if role == 'architecture':
        return """display(pd.DataFrame([
    ('S1', 'RoBERTa fine-tuned', 768, 256, 4),
    ('S2', 'BGE frozen', 1024, 256, 4),
    ('S3', 'RoBERTa fine-tuned + proyeksi BGE frozen', 1024, 256, 4),
], columns=['skenario', 'representasi', 'input_MLP', 'hidden_MLP', 'output']))
"""
    if role == 'definition:optimizer_groups':
        return replace(source, {
            'Lima kelompok optimizer S3': 'Kelompok optimizer per skenario',
            "cfg['s3_encoder_lr']": "cfg['s3_encoder_lr'] if model.scenario == 'S3' else cfg['encoder_lr']",
            "cfg['s3_mlp_lr']": "cfg['s3_mlp_lr'] if model.scenario == 'S3' else cfg['head_lr']",
        })
    if role == 'cv_definition':
        return replace(source, {
            'def run_cv(bundle, manifest, bge, cfg, output):':
                'def run_cv(bundle, manifest, bge, cfg, output, scenario):',
            'Validasi silang satu resep E3 pada development.':
                'Validasi silang lima fold dengan resep dan seed E13.',
            "'fold': number, 'validation': validation}":
                "'fold': number, 'validation': validation, 'scenario': scenario}",
            "print(f'E14 reset · fold {number}:": "print(f'E14 reset {scenario} · fold {number}:",
            "print(f'E14 reset · fold {number}/": "print(f'E14 reset {scenario} · fold {number}/",
            "fit('S3', fold['train']": "fit(scenario, fold['train']",
        })
    if role == 'final_model':
        return replace(source, {
            'def refit_checkpoint(epochs, bundle, manifest, bge, cfg, output):':
                'def refit_checkpoint(epochs, bundle, manifest, bge, cfg, output, scenario):',
            "'refit_epochs': epochs, 'split': manifest['split_sha256']}":
                "'refit_epochs': epochs, 'split': manifest['split_sha256'], 'scenario': scenario}",
            "print(f'Refit E14 reset ·": "print(f'Refit E14 reset {scenario} ·",
            "fit('S3', manifest['development']": "fit(scenario, manifest['development']",
        })
    if role == 'test_definition':
        return replace(source, {
            "model = make_model('S3', cfg)": "model = make_model(refit_result['identity']['scenario'], cfg)",
        })
    if role == 'final_features':
        return replace(source, {
            '    payload = torch.load(output / "final.pt", map_location="cpu", weights_only=True)':
                '    s3_folder = output / "S3"\n'
                '    payload = torch.load(s3_folder / "final.pt", map_location="cpu", weights_only=True)',
            '        output,\n        context=': '        s3_folder,\n        context=',
            '"Hidden per trait", "dimensi": 64': '"Hidden bersama", "dimensi": 256',
        })
    if role == 'export_execute':
        return replace(source, {
            'Hasil_AES_Reset_E14_S3_MultiTrait.zip': 'Hasil_AES_Reset_E14_S1_S2_S3_MultiTrait.zip',
            'manifest, cfg, cv_summary': 'manifest, {**cfg, "scenario": "S3"}, cv_summary',
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
        'cv_execute': CV_EXECUTE,
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
            source = source.replace('### Model fusion multi-trait', '### Model S1, S2, dan fusion S3')
        cell['source'] = sections[role] if role in sections else update_cell(role, source)
        cell['id'] = f'reset-e14-{index}'

    notebook['cells'][0]['source'] = INTRO
    markdown = {
        60: '## 12 · Model multi-trait untuk S1, S2, dan S3',
        61: 'S1 dan S2 memakai satu encoder. S3 memakai concat RoBERTa dan proyeksi BGE. Setiap skenario mempunyai MLP 256 dengan empat keluaran.',
        82: '### `run_cv`\nJalankan satu resep E13 pada lima fold yang sama untuk setiap skenario.',
        84: '### Jalankan lima fold per skenario',
        114: 'Refit tiap skenario pada seluruh development sebelum membuka test.',
        116: '## 18 · Evaluasi test\nHitung prediksi dari satu checkpoint final per skenario.',
        127: '## 19 · Inspeksi vektor dan susun ZIP lengkap\nInspeksi vektor final memakai checkpoint S3, mengikuti E13.',
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
