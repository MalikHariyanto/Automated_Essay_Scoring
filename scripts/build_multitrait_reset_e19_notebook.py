"""Build a validation-only E13 ablation with a fixed organization loss ratio."""

import hashlib
import json

from build_multitrait_reset_e6_notebook import BASE, GROUP, ROOT, replace
from build_multitrait_reset_e13_notebook import NAME as E13_NAME
from build_multitrait_reset_e18b_notebook import NAME as E18B_NAME

NAME = 'AES_Reset_E19_E13_Bobot_Organisasi_Validasi.ipynb'
MARKER = '__RESET_E19_CODE_SHA256__'
INTRO = """# E19 · E13 dengan bobot ringan untuk organisasi skor 1

E19 menguji satu perubahan pada E13. Untuk trait organisasi, kesalahan pada
jawaban yang dinilai guru **1** diberi bobot **1,2 kali** bobot skor **2**.
Rata-rata bobot organisasi dari data latih tiap fold disetel ke **1**.
Isi, kebahasaan, dan mekanik tetap memakai bobot **1**.
Ambang kategori organisasi tetap **1,50**.

Pilih **T4 GPU** dan jalankan bagian dari atas. Arsitektur, data, split, seed,
setelan optimizer, maksimal 25 epoch, dan patience 5 mengikuti E13. Setiap fold
menampilkan hasil dan memperbarui ZIP progres 04–05 di Drive. Fold yang telah
selesai bisa dipakai ulang. Run ini berakhir setelah validasi lima fold.

Isi path ZIP ringkas E13 di bagian 5 untuk membandingkan prediksi OOF pada ID
dan fold yang sama. Periksa QWK organisasi, tambahan skor 1 yang dikenali,
tambahan skor 2 yang salah jadi 1, serta dampak pada RMSE dan tiga trait lain.
Output siswa tetap privat.
"""

LOSS_HELPER = """def organization_loss_weights(labels, ratio):
    # Bobot ditentukan hanya dari label organisasi di bagian training fold.
    scores = np.asarray(labels)
    if scores.ndim != 1 or not len(scores) or not np.isin(scores, (1, 2)).all():
        raise ValueError('Label organisasi untuk bobot loss harus berisi skor 1 atau 2.')
    if not math.isfinite(ratio) or ratio < 1:
        raise ValueError('Rasio bobot organisasi harus berhingga dan minimal 1.')

    n_low = int(np.sum(scores == 1))
    n_high = len(scores) - n_low
    average = (ratio * n_low + n_high) / len(scores)
    return [
        {'skor': 1, 'n_training': n_low, 'bobot_loss': ratio / average},
        {'skor': 2, 'n_training': n_high, 'bobot_loss': 1 / average},
    ]


"""


def weighted_fit(source):
    return LOSS_HELPER + replace(source, {
        '    # Buat model baru untuk fold ini, lalu tempatkan di GPU atau CPU yang tersedia.':
            """    org_column = next(i for i, (name, *_) in enumerate(TRAITS) if name == 'organisasi')
    org_weights = organization_loss_weights(bundle.y[training, org_column], cfg['organization_loss_ratio'])
    # Untuk rubrik organisasi 1–2, target ternormalisasi 0 berarti skor asli 1.
    weight_low, weight_high = (row['bobot_loss'] for row in org_weights)
    print(f'Bobot organisasi per skor: {org_weights}', flush=True)
    # Buat model baru untuk fold ini, lalu tempatkan di GPU atau CPU yang tersedia.""",
        'loss_total = mse_total = relation_total = 0.':
            'loss_total = mse_total = weighted_total = relation_total = 0.',
        '                    mse = torch.mean((prediction-targets)**2)':
            """                    squared_error = (prediction - targets).square()
                    mse = squared_error.mean()
                    weights = torch.ones_like(squared_error)
                    weights[:, org_column] = torch.where(
                        targets[:, org_column] == 0, weight_low, weight_high
                    )
                    weighted_mse = (squared_error * weights).mean()""",
        "loss = mse + cfg['trait_relation_weight'] * relation":
            "loss = weighted_mse + cfg['trait_relation_weight'] * relation",
        '                mse_total += mse.item()*len(ids)':
            '                mse_total += mse.item()*len(ids)\n                weighted_total += weighted_mse.item()*len(ids)',
        "        row = {'epoch': epoch, 'train_mse_scaled': mse_total/len(training),":
            """        row = {'epoch': epoch, 'train_mse_scaled': mse_total/len(training),
               'train_weighted_mse_scaled': weighted_total/len(training),""",
        "'parameter_samples_changed': changed, 'BGE_encoder_trained': False,":
            "'parameter_samples_changed': changed, 'BGE_encoder_trained': False,\n            'organization_loss_weights': org_weights,",
    })


def build():
    e13_path = BASE.parent / E13_NAME
    e18b_path = BASE.parent / E18B_NAME
    raw = e18b_path.read_bytes()
    notebook = json.loads(raw)
    old_hash = notebook['metadata']['aes_standalone']['code_sha256']
    original = {
        cell['metadata'].get('aes_multitrait_role'): ''.join(cell['source'])
        for cell in json.loads(e13_path.read_bytes())['cells'] if cell['cell_type'] == 'code'
    }

    for index, cell in enumerate(notebook['cells']):
        source = (''.join(cell['source']).replace(old_hash, MARKER)
                  .replace('Reset_E18B_E13_Trial60', 'Reset_E19_E13_Bobot_Organisasi')
                  .replace('reset_e18b_e13_trial60', 'reset_e19_e13_bobot_organisasi')
                  .replace('aes_reset_e18b_e13_trial60', 'aes_reset_e19_e13_bobot_organisasi')
                  .replace('E18B', 'E19'))
        role = cell['metadata'].get('aes_multitrait_role')
        if index == 0:
            source = INTRO
        elif role == 'hyperparameters':
            source = original[role].replace(
                '# HIPERPARAMETER BASELINE E43; Scalar Mix empat layer terakhir.',
                '# Setelan training E13. Satu perubahan ada pada bobot loss organisasi.',
            )
            source += """
ORGANIZATION_LOSS_RATIO = 1.2
# Opsional: isi path lengkap ZIP ringkas kontrol E13 di Drive.
E13_CONTROL_ZIP = ''
"""
        elif role == 'configuration_defaults':
            source = original[role]
        elif role == 'recipe_preview':
            source = """print(f'Rasio bobot loss organisasi skor 1 terhadap skor 2: {ORGANIZATION_LOSS_RATIO:g}×')
print(f'Maksimum {EPOCHS} epoch per fold · patience {EARLY_STOPPING_PATIENCE} · '
      f'{FOLDS} fold · seed dasar {TRAIN_SEED}')
print('Ambang kategori organisasi tetap 1,50.')"""
        elif role == 'prepare':
            source = replace(source, {
                "'runtime': runtime_information(),":
                    "'runtime': runtime_information(), 'organization_loss_ratio': ORGANIZATION_LOSS_RATIO,",
            })
        elif role == 'definition:fit':
            source = weighted_fit(original[role])
        elif role == 'fold_helpers':
            source = replace(source, {
                '    display(pd.DataFrame(lowest_score_rows(bundle, valid, prediction, \'validasi\')))':
                    """    display(pd.DataFrame(lowest_score_rows(bundle, valid, prediction, 'validasi')))
    display(pd.DataFrame(info['organization_loss_weights']).round(4))""",
            })
        elif role == 'validation_report':
            source = replace(source, {
                "        **bundle.summary, 'experiment': cfg['experiment'], 'trial_sumber': 60,":
                    "        **bundle.summary, 'experiment': cfg['experiment'],\n"
                    "        'organization_loss_ratio': cfg['organization_loss_ratio'],",
                "            'Riwayat Training': pd.DataFrame(history),":
                    """            'Riwayat Training': pd.DataFrame(history),
            'Bobot Loss Organisasi': pd.DataFrame([
                {'fold': number, **row}
                for number, result in sorted(fold_results.items())
                for row in result['training']['organization_loss_weights']
            ]),""",
            })
        cell['source'] = source
        cell['id'] = f'reset-e19-{index}'

    code = '\n'.join(cell['source'] for cell in notebook['cells'] if cell['cell_type'] == 'code')
    fingerprint = hashlib.sha256(code.encode()).hexdigest()
    for cell in notebook['cells']:
        cell['source'] = cell['source'].replace(MARKER, fingerprint)
        if cell['cell_type'] == 'code':
            compile(cell['source'], cell['id'], 'exec')
    notebook['metadata']['colab']['name'] = NAME
    notebook['metadata']['aes_standalone'] = {
        'code_sha256': fingerprint, 'base': e13_path.relative_to(ROOT).as_posix(),
        'base_sha256': hashlib.sha256(e13_path.read_bytes()).hexdigest(),
        'layout_base': e18b_path.relative_to(ROOT).as_posix(),
        'layout_base_sha256': hashlib.sha256(raw).hexdigest(),
        'experiment': 'Reset_E19_E13_Bobot_Organisasi',
    }
    content = json.dumps(notebook, ensure_ascii=False, indent=1) + '\n'
    for folder in (BASE.parent, GROUP / 'notebooks'):
        (folder / NAME).write_text(content, encoding='utf-8')


if __name__ == '__main__':
    build()
