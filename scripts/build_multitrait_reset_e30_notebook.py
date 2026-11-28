"""Build E30 to evaluate the confirmed E17E Trial 60 recipe on E29."""

import hashlib
import json
import re

from build_multitrait_reset_e29_notebook import BASE, GROUP, ROOT, NAME as SOURCE_NAME, replace, snapshot

NAME = 'AES_Reset_E30_S3_ScalarMix_PerTrait_Trial60.ipynb'
EXPERIMENT = 'Reset_E30_S3_ScalarMix_PerTrait_Trial60'
MARKER = '__RESET_E30_CODE_SHA256__'

INTRO = """# E30 · ScalarMix per trait dengan paket Trial 60

Arsitektur mengikuti E29. Paket hyperparameter berasal dari konfigurasi
final E17E Trial 60 yang sudah melalui konfirmasi. Uji ini mengukur transfer
paket tersebut ke arsitektur E29 pada development.

- LR RoBERTa 1,6407433215802577e-5.
- LR proyeksi, head, attention, normalisasi dan ScalarMix 3,345252006826264e-5.
- Dropout head 0,30; weight decay 0,03; warmup 0,10.
- Lima fold, seed dasar 44 (fit 45–49), maksimal 25 epoch dan patience 5.
- Bobot isi 1×, organisasi 1,2× dan aturan A/B mengikuti E29.

Pilih T4, hubungkan Drive ke workbook semi-final, lalu Run all. Hasil muncul
per fold; fold selesai tersimpan. Setelah validasi, lihat perbandingan
RMSE, Pearson, QWK dan Gwet AC2 dengan E29-B, E28-B dan E25-B.
Unduh ZIP 04–05. Tahap ini berakhir di development.

Paket ini adalah hasil pencarian proyek. Nilainya belum terbukti optimal
untuk ScalarMix per trait. Studi HPO baru tidak dijalankan oleh notebook ini.
"""


def trial60():
    path = GROUP / 'output/checkpoints/E17E.json'
    run = json.loads(path.read_text())
    rows = [row for row in run['tables']['HPO Trials'] if row['trial'] == 60 and row['status'] == 'COMPLETE']
    if len(rows) != 1:
        raise ValueError('Trial 60 COMPLETE harus tersedia tepat sekali.')
    row = rows[0]
    config = run['config']
    for key, field in (('s3_encoder_lr', 'encoder_lr'), ('s3_mlp_lr', 'aux_lr'),
                       ('s3_mlp_dropout', 'mlp_dropout'), ('weight_decay', 'weight_decay'),
                       ('warmup_ratio', 'warmup_ratio')):
        if abs(config[key] - row[field]) > 1e-12:
            raise ValueError(f'Konfigurasi final E17E tidak sesuai Trial 60: {key}')
    return row, hashlib.sha256(path.read_bytes()).hexdigest()


def build():
    base = BASE.parent / SOURCE_NAME
    raw = base.read_bytes()
    notebook = json.loads(raw)
    old_hash = notebook['metadata']['aes_standalone']['code_sha256']
    trial, proof = trial60()
    settings = {
        'ENCODER_LR': trial['encoder_lr'],
        'S3_MLP_LR': trial['aux_lr'],
        'S3_LAYERS_LR': trial['aux_lr'],
        'PROJECTION_LR': trial['aux_lr'],
        'S3_MLP_DROPOUT': trial['mlp_dropout'],
        'WEIGHT_DECAY': trial['weight_decay'],
        'WARMUP_RATIO': trial['warmup_ratio'],
    }
    for index, cell in enumerate(notebook['cells']):
        source = ''.join(cell['source']).replace(old_hash, MARKER)
        role = cell['metadata'].get('aes_multitrait_role')
        if index == 0:
            source = INTRO
        elif role == 'hyperparameters':
            for key, value in settings.items():
                source, count = re.subn(rf'^{key}\s*=.*$', f'{key} = {value!r}', source, flags=re.M)
                if count != 1:
                    raise ValueError(f'Hyperparameter harus ada tepat sekali: {key}')
        elif role == 'prepare':
            source = replace(source, {
                'Reset_E29_S3_ScalarMix_PerTrait': EXPERIMENT,
                "'runtime': runtime_information(),": (
                    "'runtime': runtime_information(),\n"
                    f"    'hpo_recipe_trial': 60, 'hpo_recipe_archive_sha256': '{proof}', 'head_lr': S3_MLP_LR,"
                ),
                'output = run_folder(cfg, bundle)': (
                    f"E29_REFERENCE = {snapshot('E29')!r}\n"
                    "if (E29_REFERENCE['split_sha256'] != manifest['split_sha256']\n"
                    "        or E29_REFERENCE['seed'] != cfg['seed']):\n"
                    "    raise ValueError('Split dan seed harus cocok dengan E29.')\n"
                    'output = run_folder(cfg, bundle)'
                ),
            })
        elif role == 'validation_report':
            source = replace(source, {
                "'E29 (B)': value": "'E30 (B)': value",
                "'delta E29 minus pembanding': delta": "'delta E30 minus pembanding': delta",
                "        worksheets['E29 vs E28 per Trait']": (
                    "        worksheets['E30 vs E29 per Trait'] = compare_reference(worksheets, E29_REFERENCE)\n"
                    "        worksheets['E30 vs E28 per Trait']"
                ),
                "worksheets['E29 vs E25 per Trait']": "worksheets['E30 vs E25 per Trait']",
                "'e28_source_sha256': E28_REFERENCE['source_sha256'],": (
                    "'e28_source_sha256': E28_REFERENCE['source_sha256'],\n"
                    "        'e29_source_sha256': E29_REFERENCE['source_sha256'],"
                ),
            })
        elif role == 'final_report':
            source = replace(source, {
                "        print('E29 dibanding E28 · delta = E29 minus pembanding')": (
                    "        print('E30 dibanding E29 · delta = E30 minus pembanding')\n"
                    "        display(sheets['E30 vs E29 per Trait'].round(4))\n"
                    "        print('E30 dibanding E28')"
                ),
                "sheets['E29 vs E28 per Trait']": "sheets['E30 vs E28 per Trait']",
                "print('E29 dibanding E25')": "print('E30 dibanding E25')",
                "sheets['E29 vs E25 per Trait']": "sheets['E30 vs E25 per Trait']",
            })
        source = source.replace('aes_reset_e29_scalar_mix_per_trait', 'aes_reset_e30_pertrait_trial60')
        source = source.replace('Hasil_AES_Reset_E29_Progres', 'Hasil_AES_Reset_E30_Progres')
        source = source.replace('Hasil_AES_Reset_E29_Validasi', 'Hasil_AES_Reset_E30_Validasi')
        cell['source'] = source
        cell['id'] = f'reset-e30-{index}'

    fingerprint = hashlib.sha256('\n'.join(c['source'] for c in notebook['cells']
                                            if c['cell_type'] == 'code').encode()).hexdigest()
    for cell in notebook['cells']:
        cell['source'] = cell['source'].replace(MARKER, fingerprint)
        if cell['cell_type'] == 'code':
            compile(cell['source'], cell['id'], 'exec')
    notebook['metadata']['colab']['name'] = NAME
    notebook['metadata']['aes_standalone'] = {
        'code_sha256': fingerprint, 'base': base.relative_to(ROOT).as_posix(),
        'base_sha256': hashlib.sha256(raw).hexdigest(), 'experiment': EXPERIMENT,
        'recipe_source': 'E17E Trial 60 confirmed', 'recipe_archive_sha256': proof,
    }
    content = json.dumps(notebook, ensure_ascii=False, indent=1) + '\n'
    for folder in (BASE.parent, GROUP / 'notebooks'):
        (folder / NAME).write_text(content, encoding='utf-8')


if __name__ == '__main__':
    build()
