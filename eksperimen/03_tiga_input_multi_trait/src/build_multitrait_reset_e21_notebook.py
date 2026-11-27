"""Build E21, a held-out evaluation of the two frozen E20 checkpoint rules."""

import hashlib
import json

from build_multitrait_reset_e6_notebook import BASE, GROUP, ROOT, replace
from build_multitrait_reset_e20_seed43_notebook import NAME as E20_NAME

NAME = 'AES_Reset_E21_S3_Test_Checkpoint_AB.ipynb'
MARKER = '__RESET_E21_CODE_SHA256__'

INTRO = """# E21 · Evaluasi test S3 dengan checkpoint A dan B

Pilih **T4 GPU**, hubungkan Drive ke workbook E20, lalu klik **Run all**.
E20 menyimpan hasil validasi, sehingga E21 membuat ulang lima fit seed dasar 43.
Fold yang selesai akan langsung tersimpan dan bisa dipakai lagi setelah runtime putus.

- **A** memilih epoch dengan RMSE validasi terendah.
- **B** memilih macro QWK validasi tertinggi dalam margin RMSE 0,01 dari A.
- Lima prediksi mentah test per aturan dirata-ratakan sebelum dikonversi ke kategori.

Label test dipakai untuk evaluasi setelah prediksi dari lima fold lengkap. Lihat
hasil validasi langsung di setiap fold dan rekap test di bagian paling bawah.
"""

CAPTURE = '''
class CandidateWeights:
    """Keep validation candidates that could still win B, on local disk."""

    def __init__(self, directory, margin):
        self.directory = Path(directory)
        self.margin = margin
        self.candidates = {}

    def path(self, epoch):
        return self.directory / f'epoch_{epoch}.pt'

    def observe(self, model, row, best_rmse):
        limit = best_rmse + self.margin + 1e-12
        for epoch, (rmse, _) in list(self.candidates.items()):
            if rmse > limit:
                self.path(epoch).unlink()
                del self.candidates[epoch]

        evaluation = row['validation']
        qwk = mean_defined_qwk(evaluation)
        rmse = evaluation['RMSE']
        if qwk is None or rmse > limit:
            return
        if any(old_rmse <= rmse and old_qwk >= qwk
               for old_rmse, old_qwk in self.candidates.values()):
            return
        for epoch, (old_rmse, old_qwk) in list(self.candidates.items()):
            if rmse <= old_rmse and qwk >= old_qwk:
                self.path(epoch).unlink()
                del self.candidates[epoch]
        torch.save({k: value.detach().cpu() for k, value in model.state_dict().items()},
                   self.path(row['epoch']))
        self.candidates[row['epoch']] = (rmse, qwk)


def fit_and_predict_test(training, validation, seed):
    # File sementara per fold. Cache Drive menyimpan prediksi setelah fold lengkap.
    with TemporaryDirectory(prefix='aes_e21_') as directory:
        weights = CandidateWeights(directory, cfg['checkpoint_rmse_margin'])
        model, engine, info, choices = fit(
            'S3', training, validation, bundle, bge, cfg, cfg['epochs'], seed, weights,
        )
        validation_a = predictions_array(engine.predict(validation), validation)
        test = manifest['test']
        test_a = predictions_array(engine.predict(test), test)
        test_b = test_a
        if choices['B']['epoch'] != choices['A']['epoch']:
            model.load_state_dict(torch.load(
                weights.path(choices['B']['epoch']), map_location='cpu', weights_only=True,
            ))
            validation_b = predictions_array(engine.predict(validation), validation)
            if not np.allclose(validation_b, choices['B']['prediction'], rtol=0, atol=1e-6):
                raise ValueError('Bobot checkpoint B tidak mereproduksi prediksi validasi.')
            test_b = predictions_array(engine.predict(test), test)
        for choice, prediction in (('A', test_a), ('B', test_b)):
            choices[choice]['test_prediction'] = prediction.tolist()
        del engine, model
    release()
    return info, choices, validation_a
'''

TEST_REPORT = '''
def test_tables():
    test = manifest['test']
    if not test or len(test) != len(set(test)) or set(test) & set(manifest['development']):
        raise ValueError('Split test harus unik dan terpisah dari development.')
    per_trait, predictions, low_scores, overview = [], [], [], []
    for choice in ('A', 'B'):
        per_fold = [
            predictions_array(result['checkpoints'][choice]['test_prediction'], test)
            for _, result in sorted(fold_results.items())
        ]
        # Gabungkan skor mentah dulu. Fungsi metrik dan ekspor menerapkan pembulatan E20.
        prediction = np.mean(per_fold, axis=0)
        evaluation = metrics(bundle.y[test], prediction)
        tags = {'pilihan': choice}
        overview.append({**tags, 'RMSE': evaluation['RMSE'],
                         'macro_QWK': mean_defined_qwk(evaluation), 'n': len(test)})
        per_trait.extend(trait_rows(evaluation, **tags, n=len(test)))
        predictions.extend(prediction_rows(bundle, test, prediction, **tags))
        errors = checkpoint_score_errors(test, prediction, **tags)
        for column, trait, row in zip(range(len(TRAITS)), RUBRIC['traits'], errors):
            labels = bundle.y[test, column]
            category = ordinal_prediction(prediction[:, column], trait)
            row['skor di atas 1 salah jadi 1'] = int(np.sum((labels > 1) & (category == 1)))
        low_scores.extend(errors)
    return {
        'Ringkasan Test A B': pd.DataFrame(overview),
        'Test A B per Trait': pd.DataFrame(per_trait),
        'Prediksi Test A B': pd.DataFrame(predictions),
        'Kesalahan Skor Rendah Test': pd.DataFrame(low_scores),
    }
'''


def update(role, source):
    changes = {
        'bge_execute': {
            "    }]))": """    }]))

    # Encode test secara terpisah. Encoder frozen hanya menerima tiga kolom teks.
    test = manifest['test']
    test_bundle = Bundle([bundle.rows[i] for i in test], bundle.y[test],
                         bundle.groups[test], bundle.source_sha256, bundle.summary)
    print(f'Encoding test · {len(test)} jawaban', flush=True)
    test_output = output / 'test_features'
    test_output.mkdir(exist_ok=True)
    bge[test] = bge_features(test_bundle, cfg, test_output)""",
        },
        'definition:fit': {
            'cfg, epochs, seed):': 'cfg, epochs, seed, candidate_weights):',
            "            row.update(best_epoch_so_far": (
                "            candidate_weights.observe(model, row, best_rmse)\n"
                "            row.update(best_epoch_so_far"
            ),
        },
        'fold_helpers': {
            'from time import perf_counter': 'from time import perf_counter\nfrom tempfile import TemporaryDirectory\n\n' + CAPTURE,
            "        'seed': seed,": "        'seed': seed,\n        'test': manifest['test'],",
            """        model, engine, info, checkpoints = fit('S3', training, validation, bundle, bge,
                                  cfg, cfg['epochs'], seed)
        prediction = predictions_array(engine.predict(validation), validation)""": (
                "        info, checkpoints, prediction = fit_and_predict_test(training, validation, seed)"
            ),
            '        del model, engine\n        release()\n': '',
            "    for choice, checkpoint in result['checkpoints'].items():\n        selected": (
                "    for choice, checkpoint in result['checkpoints'].items():\n"
                "        predictions_array(checkpoint['test_prediction'], manifest['test'])\n        selected"
            ),
        },
        'validation_report': {
            'def export_validation(path, final=False):': TEST_REPORT + '\n\ndef export_validation(path, final=False):',
            "    export_excel_zip(path, {": "    if final:\n        worksheets.update(test_tables())\n\n    export_excel_zip(path, {",
            "'test_dievaluasi': False": "'test_dievaluasi': final, 'test_ensemble': 'mean of five raw fold predictions per rule'",
        },
        'final_report': {
            "        print(f'ZIP validasi A/B · {final_path}')": (
                "        print('Evaluasi test dari ensemble lima fold per aturan')\n"
                "        for tab in ('Ringkasan Test A B', 'Test A B per Trait', 'Kesalahan Skor Rendah Test'):\n"
                "            display(sheets[tab].round(4))\n"
                "        print(f'ZIP test A/B · {final_path}')"
            ),
        },
    }
    return replace(source, changes[role]) if role in changes else source


def build():
    base = BASE.parent / E20_NAME
    raw = base.read_bytes()
    notebook = json.loads(raw)
    old_hash = notebook['metadata']['aes_standalone']['code_sha256']
    substitutions = {
        old_hash: MARKER,
        'Reset_E20_Konfirmasi_Seed43': 'Reset_E21_S3_Test_Checkpoint_AB',
        'aes_reset_e20_checkpoint_ab_seed43': 'aes_reset_e21_s3_test_checkpoint_ab',
        'Hasil_AES_Reset_E20_Seed43_Progres_04_05.zip': 'Hasil_AES_Reset_E21_S3_Progres_04_05.zip',
        'Hasil_AES_Reset_E20_Seed43_Validasi_Checkpoint_AB_04_05.zip': 'Hasil_AES_Reset_E21_S3_Test_Checkpoint_AB_04_05.zip',
        '## 17 · Bandingkan checkpoint A dan B': '## 17 · Rekap validasi dan evaluasi test A/B',
        'Rekap prediksi validasi dan unduh ZIP 04–05.': 'Lihat metrik test per trait dan unduh ZIP 04–05.',
    }
    for index, cell in enumerate(notebook['cells']):
        source = ''.join(cell['source'])
        for old, new in substitutions.items():
            source = source.replace(old, new)
        role = cell['metadata'].get('aes_multitrait_role')
        cell['source'] = INTRO if index == 0 else update(role, source)
        cell['id'] = f'reset-e21-{index}'

    code = '\n'.join(cell['source'] for cell in notebook['cells'] if cell['cell_type'] == 'code')
    fingerprint = hashlib.sha256(code.encode()).hexdigest()
    for cell in notebook['cells']:
        cell['source'] = cell['source'].replace(MARKER, fingerprint)
        if cell['cell_type'] == 'code':
            compile(cell['source'], cell['id'], 'exec')
    notebook['metadata']['colab']['name'] = NAME
    notebook['metadata']['aes_standalone'] = {
        'code_sha256': fingerprint, 'base': base.relative_to(ROOT).as_posix(),
        'base_sha256': hashlib.sha256(raw).hexdigest(),
        'experiment': 'Reset_E21_S3_Test_Checkpoint_AB',
    }
    content = json.dumps(notebook, ensure_ascii=False, indent=1) + '\n'
    for folder in (BASE.parent, GROUP / 'notebooks'):
        (folder / NAME).write_text(content, encoding='utf-8')


if __name__ == '__main__':
    build()
