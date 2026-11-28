"""Build E32: E31 EMA model with an auxiliary ordinal threshold objective."""

import hashlib
import json

from build_multitrait_reset_e31_notebook import BASE, GROUP, ROOT, NAME as SOURCE_NAME, replace
from build_multitrait_reset_e29_notebook import snapshot

NAME = 'AES_Reset_E32_S3_PerTrait_EMA_Ordinal.ipynb'
EXPERIMENT = 'Reset_E32_S3_PerTrait_EMA_Ordinal'
MARKER = '__RESET_E32_CODE_SHA256__'

INTRO = """# E32 · E31 + objective ordinal untuk kategori trait

Fondasi model, EMA, split, seed dasar 44, bobot isi dan organisasi mengikuti E31.
Tambahan satu objective: BCE pada pertanyaan apakah skor melewati tiap batas
kategori, dihitung dari prediksi skor kontinu yang sama dengan MSE.

Untuk isi 1–4, batas asli adalah 1,5; 2,5; 3,5. Trait biner memakai batas 1,5.
Setiap batas dinormalisasi memakai rentang rubrik trait. Logit = (prediksi
normalisasi − batas normalisasi) / 0,10. Loss akhir = weighted MSE + 0,10 ×
ordinal BCE. Rerata BCE dilakukan per batas lalu per trait agar isi tidak
otomatis mendapat bobot tiga kali trait lain.

Koefisien 0,10 dan suhu 0,10 adalah pilot adaptasi lokal, bukan optimum paper.
Head dan metrik tetap memprediksi skor yang sama; threshold evaluasi tetap.
EMA dipakai untuk validasi dan checkpoint A/B. Tabel final membandingkan E32
dengan E31, E28 dan E25, termasuk fold 2, 4 dan 5.

Pilih T4, hubungkan Drive, Run all. Lima fit development, fold selesai bisa
dipakai ulang. Unduh ZIP 04–05. Hasil test tidak dipakai pada tahap ini.
"""

ORDINAL = '''
def ordinal_threshold_loss(prediction, targets, temperature):
    """Rerata BCE per batas dan per trait pada target rubrik 0–1."""
    losses = []
    for index, trait in enumerate(RUBRIC['traits']):
        categories = torch.as_tensor(trait['categories'], dtype=prediction.dtype,
                                     device=prediction.device)
        cuts = (categories[:-1] + categories[1:]) / 2
        cuts = (cuts - trait['min']) / (trait['max'] - trait['min'])
        logits = (prediction[:, index:index + 1] - cuts) / temperature
        labels = (targets[:, index:index + 1] > cuts).to(prediction.dtype)
        losses.append(torch.nn.functional.binary_cross_entropy_with_logits(logits, labels))
    return torch.stack(losses).mean()
'''


def build():
    base = BASE.parent / SOURCE_NAME
    raw = base.read_bytes()
    notebook = json.loads(raw)
    previous_hash = notebook['metadata']['aes_standalone']['code_sha256']
    baseline = json.loads((GROUP / 'output/checkpoints/E31.json').read_text())
    fold_rows = [r for r in baseline['tables']['Pilihan Checkpoint'] if r['pilihan'] == 'B']
    for index, cell in enumerate(notebook['cells']):
        source = ''.join(cell['source']).replace(previous_hash, MARKER)
        role = cell['metadata'].get('aes_multitrait_role')
        # E31 denotes the current run in the base notebook; replace before inserting references.
        source = source.replace('E31', 'E32').replace('e31', 'e32')
        if index == 0:
            source = INTRO
        elif role == 'prepare':
            ref_line = next(line for line in source.splitlines() if line.startswith('E29_REFERENCE ='))
            fold_line = next(line for line in source.splitlines() if line.startswith('E29_FOLD_REFERENCE ='))
            source = replace(source, {
                'Reset_E32_S3_PerTrait_EMA': EXPERIMENT,
                "'runtime': runtime_information(),": (
                    "'runtime': runtime_information(), 'ordinal_loss_weight': 0.10, 'ordinal_temperature': 0.10,"
                ),
                ref_line: f'E31_REFERENCE = {snapshot("E31")!r}',
                fold_line: f'E31_FOLD_REFERENCE = {fold_rows!r}',
            })
            source = source.replace('E29_REFERENCE[', 'E31_REFERENCE[')
            source = source.replace('Split dan seed harus cocok dengan E29.', 'Split dan seed harus cocok dengan E31.')
        elif role == 'definition:fit':
            source = ORDINAL + '\n\n' + replace(source, {
                'loss_total = mse_total = weighted_total = relation_total = 0.':
                    'loss_total = mse_total = weighted_total = relation_total = ordinal_total = 0.',
                "                    loss = weighted_mse + cfg['trait_relation_weight'] * relation": (
                    "                    ordinal = ordinal_threshold_loss(prediction, targets, cfg['ordinal_temperature'])\n"
                    "                    loss = weighted_mse + cfg['trait_relation_weight'] * relation + cfg['ordinal_loss_weight'] * ordinal"
                ),
                '                relation_total += relation.item()*len(ids)': (
                    '                relation_total += relation.item()*len(ids)\n'
                    '                ordinal_total += ordinal.item()*len(ids)'
                ),
                "'train_weighted_mse_scaled': weighted_total/len(training),": (
                    "'train_weighted_mse_scaled': weighted_total/len(training),\n"
                    "               'train_ordinal_bce': ordinal_total/len(training),"
                ),
            })
        elif role in ('validation_report', 'final_report'):
            source = source.replace('E32 vs E29', 'E32 vs E31').replace('E32 dibanding E29', 'E32 dibanding E31')
            source = source.replace('E29_REFERENCE', 'E31_REFERENCE').replace('E29_FOLD_REFERENCE', 'E31_FOLD_REFERENCE')
            source = source.replace('E29 RMSE', 'E31 RMSE').replace('E29 macro QWK', 'E31 macro QWK').replace('E29 epoch', 'E31 epoch')
            source = source.replace("'e29_source_sha256'", "'e31_source_sha256'")
            if role == 'validation_report':
                source = replace(source, {
                    "            'Konfigurasi': pd.DataFrame": (
                        "            'Loss Ordinal': pd.DataFrame([\n"
                        "                {'fold': number, 'epoch': row['epoch'], 'MSE weighted': row['train_weighted_mse_scaled'],\n"
                        "                 'ordinal BCE': row['train_ordinal_bce'],\n"
                        "                 'ordinal contribution': cfg['ordinal_loss_weight'] * row['train_ordinal_bce'],\n"
                        "                 'total loss': row['train_total_loss']}\n"
                        "                for number, result in sorted(fold_results.items())\n"
                        "                for row in result['training']['history']]),\n"
                        "            'Konfigurasi': pd.DataFrame"
                    ),
                })
        source = source.replace('aes_reset_e32_pertrait_ema', 'aes_reset_e32_pertrait_ema_ordinal')
        cell['source'] = source
        cell['id'] = f'reset-e32-{index}'

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
    }
    content = json.dumps(notebook, ensure_ascii=False, indent=1) + '\n'
    for folder in (BASE.parent, GROUP / 'notebooks'):
        (folder / NAME).write_text(content, encoding='utf-8')


if __name__ == '__main__':
    build()
