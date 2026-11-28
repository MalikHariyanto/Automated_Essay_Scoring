"""Build E23 with train-fold-only weight for content score 1."""
import hashlib
import json

from build_multitrait_reset_e6_notebook import BASE, GROUP, ROOT, replace
from build_multitrait_reset_e20_seed43_notebook import NAME as BASE_NAME
from build_multitrait_reset_e22_notebook import REPORT, update as update_e22

NAME = 'AES_Reset_E23_S3_Bobot_Isi_2x.ipynb'
MARKER = '__RESET_E23_CODE_SHA256__'
EXPERIMENT = 'Reset_E23_S3_Bobot_Isi_2x'
INTRO = """# E23 · Bobot loss isi skor 1 sebesar 2×

E23 memakai arsitektur dan resep S3 E21 dengan satu intervensi pada loss.
Pada tiap fold, kesalahan isi berlabel 1 berbobot relatif 2× terhadap isi
berlabel 2–4. Bobot dinormalisasi dengan label training fold agar rata-rata 1.
Bobot organisasi, LR, seed dasar 43, ambang kategori, dan split mengikuti E21.

Checkpoint B tetap dipilih dengan macro QWK dalam margin RMSE 0,01 dari A.
Hasil latih dan validasi B muncul setiap fold. Tahap ini berhenti di development.

Pilih T4, hubungkan Drive ke workbook semi-final yang sama, lalu klik Run all.
Fold selesai bisa dipakai ulang. Unduh ZIP 04–05 setelah lima fold lengkap.
"""

AUDIT_REPORT = REPORT.split('def isi_audit_tables',1)[0] + """
def isi_audit_tables():
    comparisons, weights = [], []
    for number, result in sorted(fold_results.items()):
        valid = result['identity']['validation']
        info = result['training']
        prediction = result['checkpoints']['B']['prediction']
        comparisons.append({'fold': number, 'bagian': 'latih', **info['isi_latih_B']})
        comparisons.append({'fold': number, 'bagian': 'validasi', **isi_counts(valid, prediction)})
        weights.extend({'fold': number, **row} for row in info['isi_loss_weights'])
    return pd.DataFrame(comparisons), pd.DataFrame(weights)
"""

def update(role, source):
    if role == 'prepare':
        return replace(source, {"'runtime': runtime_information(),":
                                "'runtime': runtime_information(), 'isi_loss_ratio': 2.0,"})
    if role == 'definition:fit':
        return replace(source, {
            'cfg, epochs, seed):': 'cfg, epochs, seed, candidate_weights):',
            '    # Buat model baru': """    isi_column = next(i for i, (name, *_) in enumerate(TRAITS) if name == 'isi')
    isi_labels = bundle.y[training, isi_column]
    n_low = int(np.sum(isi_labels == 1))
    ratio = cfg['isi_loss_ratio']
    mean_weight = 1.0 + (ratio - 1.0) * n_low / len(training)
    isi_low, isi_other = ratio / mean_weight, 1.0 / mean_weight
    isi_weights = [
        {'skor': '1', 'n_training': n_low, 'bobot_loss': isi_low},
        {'skor': '2–4', 'n_training': len(training) - n_low, 'bobot_loss': isi_other},
    ]
    print(f'Bobot loss isi dari training fold: {isi_weights}', flush=True)
    # Buat model baru""",
            '                    weighted_mse =': """                    weights[:, isi_column] = torch.where(
                        targets[:, isi_column] == 0, isi_low, isi_other
                    )
                    weighted_mse =""",
            '            row.update(best_epoch_so_far': (
                '            candidate_weights.observe(model, row, best_rmse)\n'
                '            row.update(best_epoch_so_far'
            ),
            "'organization_loss_weights': org_weights,": (
                "'organization_loss_weights': org_weights, 'isi_loss_weights': isi_weights,"
            ),
        })
    if role == 'fold_helpers':
        source = update_e22(role, source)
        start = source.index("        info['bobot_fitur_isi_B']")
        end = source.index('        del engine, model', start)
        return (source[:start] + source[end:]).replace('aes_e22_', 'aes_e23_')
    if role == 'validation_report':
        return replace(source, {
            'def export_validation(path, final=False):':
                AUDIT_REPORT + '\n\ndef export_validation(path, final=False):',
            '    export_excel_zip(path, {': """    isi_comparisons, isi_weights = isi_audit_tables()
    worksheets['Isi 1 Latih dan Validasi B'] = isi_comparisons
    export_excel_zip(path, {""",
            "            'Konfigurasi': pd.DataFrame":
                "            'Bobot Loss Isi': isi_weights,\n            'Konfigurasi': pd.DataFrame",
        })
    if role == 'final_report':
        return update_e22(role, source)
    return source


def build():
    path = BASE.parent / BASE_NAME
    raw = path.read_bytes()
    notebook = json.loads(raw)
    old_hash = notebook['metadata']['aes_standalone']['code_sha256']
    substitutions = {
        old_hash: MARKER,
        'Reset_E20_Konfirmasi_Seed43': EXPERIMENT,
        'aes_reset_e20_checkpoint_ab_seed43': 'aes_reset_e23_s3_bobot_isi_2x',
        'Hasil_AES_Reset_E20_Seed43_Progres_04_05.zip': 'Hasil_AES_Reset_E23_Progres_04_05.zip',
        'Hasil_AES_Reset_E20_Seed43_Validasi_Checkpoint_AB_04_05.zip': 'Hasil_AES_Reset_E23_Validasi_04_05.zip',
    }
    for index, cell in enumerate(notebook['cells']):
        source = ''.join(cell['source'])
        for old, new in substitutions.items():
            source = source.replace(old, new)
        cell['source'] = INTRO if index == 0 else update(cell['metadata'].get('aes_multitrait_role'), source)
        cell['id'] = f'reset-e23-{index}'
    code = '\n'.join(c['source'] for c in notebook['cells'] if c['cell_type'] == 'code')
    fingerprint = hashlib.sha256(code.encode()).hexdigest()
    for cell in notebook['cells']:
        cell['source'] = cell['source'].replace(MARKER, fingerprint)
        if cell['cell_type'] == 'code':
            compile(cell['source'], cell['id'], 'exec')
    notebook['metadata']['colab']['name'] = NAME
    notebook['metadata']['aes_standalone'] = {
        'code_sha256': fingerprint, 'base': path.relative_to(ROOT).as_posix(),
        'base_sha256': hashlib.sha256(raw).hexdigest(), 'experiment': EXPERIMENT,
    }
    content = json.dumps(notebook, ensure_ascii=False, indent=1) + '\n'
    for folder in (BASE.parent, GROUP / 'notebooks'):
        (folder / NAME).write_text(content, encoding='utf-8')


if __name__ == '__main__':
    build()
