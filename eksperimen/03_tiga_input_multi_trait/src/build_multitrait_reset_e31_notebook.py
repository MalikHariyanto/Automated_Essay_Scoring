"""Build E31: EMA training and validation on the matched E29 architecture."""

import hashlib
import json

from build_multitrait_reset_e29_notebook import BASE, GROUP, ROOT, NAME as SOURCE_NAME, snapshot, replace

NAME = 'AES_Reset_E31_S3_PerTrait_EMA.ipynb'
EXPERIMENT = 'Reset_E31_S3_PerTrait_EMA'
MARKER = '__RESET_E31_CODE_SHA256__'

INTRO = """# E31 · Bobot EMA untuk menstabilkan S3 E29

Fold 4 memakai seed fit 48 pada run seed dasar 44. Fold ini lebih sulit pada
beberapa run. E31 mempertahankan split dan seed yang sama untuk menguji EMA:
salinan bobot tanpa gradien mengikuti perubahan model online yang dilatih.

Update EMA dilakukan setelah optimizer berhasil dengan LR aktif. Update
pertama menyalin model, selanjutnya decay = min(0,99, (1+t)/(10+t)).
Nilai batas dan ramp mengikuti resep lokal E53. Validasi, early stopping,
checkpoint A/B, serta prediksi akhir fold memakai EMA. Bobot online tetap
menerima gradien. Per-epoch catat RMSE online, EMA, jumlah update dan decay.

Pilih T4, hubungkan Drive, lalu Run all. Lima fit development seed dasar 44
dibandingkan dengan E29-B, E28-B dan E25-B. Lihat perubahan per fold, khususnya
fold 4, serta empat metrik per trait. Fold selesai disimpan untuk pemakaian ulang.
Fit yang terputus dijalankan ulang dari awal fold. Unduh ZIP 04–05 setelah selesai.

EMA menambah satu salinan model pada memori dan satu evaluasi online per epoch.
[Morales-Brotons dkk., TMLR 2024](https://openreview.net/forum?id=2M9CUnYnBA)
mendukung metode EMA secara umum. Manfaat untuk AES multi-trait masih diuji.
"""

EMA = '''
from copy import deepcopy


class WeightEMA:
    """Rata-rata bobot pada setiap update optimizer yang berhasil."""

    def __init__(self, model, maximum_decay, offset):
        self.model = deepcopy(model).eval().requires_grad_(False)
        self.maximum_decay, self.offset = maximum_decay, offset
        self.updates, self.decay = 0, 0.0

    @torch.no_grad()
    def update(self, online):
        self.updates += 1
        self.decay = 0.0 if self.updates == 1 else min(
            self.maximum_decay, (1.0 + self.updates) / (self.offset + self.updates),
        )
        for target, source in zip(self.model.parameters(), online.parameters()):
            if self.updates == 1:
                target.copy_(source)
            else:
                target.lerp_(source.detach(), 1.0 - self.decay)
        # Buffers posisi dan token-type bukan parameter yang dipelajari.
        for target, source in zip(self.model.buffers(), online.buffers()):
            target.copy_(source)
'''

def fold_reference():
    run = json.loads((GROUP / 'output/checkpoints/E29.json').read_text())
    return [r for r in run['tables']['Pilihan Checkpoint'] if r['pilihan'] == 'B']


def build():
    base = BASE.parent / SOURCE_NAME
    raw = base.read_bytes()
    notebook = json.loads(raw)
    old_hash = notebook['metadata']['aes_standalone']['code_sha256']
    for index, cell in enumerate(notebook['cells']):
        source = ''.join(cell['source']).replace(old_hash, MARKER)
        role = cell['metadata'].get('aes_multitrait_role')
        if index == 0:
            source = INTRO
        elif role == 'prepare':
            source = replace(source, {
                'Reset_E29_S3_ScalarMix_PerTrait': EXPERIMENT,
                "'runtime': runtime_information(),": (
                    "'runtime': runtime_information(), 'ema_maximum_decay': 0.99, 'ema_offset': 10,\n"
                    "    'inference_weights': 'ema',"
                ),
                'output = run_folder(cfg, bundle)': (
                    f"E29_REFERENCE = {snapshot('E29')!r}\n"
                    f"E29_FOLD_REFERENCE = {fold_reference()!r}\n"
                    "if E29_REFERENCE['split_sha256'] != manifest['split_sha256'] or E29_REFERENCE['seed'] != cfg['seed']:\n"
                    "    raise ValueError('Split dan seed harus cocok dengan E29.')\n"
                    'output = run_folder(cfg, bundle)'
                ),
            })
        elif role == 'definition:fit':
            source = EMA + '\n\n' + replace(source, {
                '    groups = optimizer_groups(model, cfg)': (
                    "    ema = WeightEMA(model, cfg['ema_maximum_decay'], cfg['ema_offset'])\n"
                    "    ema_engine = Engine(ema.model, bundle.texts, bge, cfg, device,\n"
                    "                        roberta_tokenizer=engine.tokenizer, contexts=engine.contexts)\n"
                    '    groups = optimizer_groups(model, cfg)'
                ),
                '                active_steps += int(lr_active)': (
                    "                active_steps += int(lr_active)\n"
                    "                if lr_active:\n"
                    "                    ema.update(model)"
                ),
                '            predicted = engine.predict(validation)': (
                    "            if ema.updates != active_steps or ema.updates == 0:\n"
                    "                raise RuntimeError('EMA belum mendapat update yang sesuai optimizer.')\n"
                    "            online_prediction = engine.predict(validation)\n"
                    "            row['validation_online_RMSE'] = metrics(bundle.y[validation], online_prediction)['RMSE']\n"
                    "            row['ema_updates'] = ema.updates\n"
                    "            row['ema_decay'] = ema.decay\n"
                    "            predicted = ema_engine.predict(validation)"
                ),
                'v in model.state_dict().items()': 'v in ema.model.state_dict().items()',
                'f" | RMSE validasi: {row[\'validation\'][\'RMSE\']:.4f} poin"':
                    'f" | RMSE EMA: {row[\'validation\'][\'RMSE\']:.4f} | online: {row[\'validation_online_RMSE\']:.4f} poin"',
                '            candidate_weights.observe(model, row, best_rmse)':
                    '            candidate_weights.observe(ema.model, row, best_rmse)',
                '        model.load_state_dict(best_state)': '        ema.model.load_state_dict(best_state)',
                "            'selection': 'minimum validation RMSE in original score units'": (
                    "            'ema_updates_final': ema.updates, 'ema_decay_final': ema.decay,\n"
                    "            'inference_weights': 'ema',\n"
                    "            'selection': 'minimum EMA validation RMSE in original score units'"
                ),
                '    return model, engine, info, choices': (
                    '    return ema.model, ema_engine, info, choices'
                ),
            })
        elif role == 'validation_report':
            source = replace(source, {
                "'E29 (B)': value": "'E31 (B)': value",
                "'delta E29 minus pembanding': delta": "'delta E31 minus pembanding': delta",
                "        worksheets['E29 vs E28 per Trait']": (
                    "        worksheets['E31 vs E29 per Trait'] = compare_reference(worksheets, E29_REFERENCE)\n"
                    "        worksheets['E31 vs E28 per Trait']"
                ),
                "worksheets['E29 vs E25 per Trait']": "worksheets['E31 vs E25 per Trait']",
                "    export_excel_zip(path, {": (
                    "    baseline = pd.DataFrame(E29_FOLD_REFERENCE).rename(columns={\n"
                    "        'RMSE': 'E29 RMSE', 'macro_QWK': 'E29 macro QWK', 'epoch': 'E29 epoch'})\n"
                    "    observed = worksheets['Pilihan Checkpoint'].query(\"pilihan == 'B'\").rename(columns={\n"
                    "        'RMSE': 'E31 RMSE', 'macro_QWK': 'E31 macro QWK', 'epoch': 'E31 epoch'})\n"
                    "    paired = observed.merge(baseline, on='fold', suffixes=('', '_reference'), validate='one_to_one')\n"
                    "    paired['delta RMSE'] = paired['E31 RMSE'] - paired['E29 RMSE']\n"
                    "    paired['delta macro QWK'] = paired['E31 macro QWK'] - paired['E29 macro QWK']\n"
                    "    worksheets['E31 vs E29 per Fold'] = paired.drop(columns=['pilihan_reference'])\n"
                    "    export_excel_zip(path, {"
                ),
                "            'Konfigurasi': pd.DataFrame": (
                    "            'Diagnostik EMA': pd.DataFrame([\n"
                    "                {'fold': number, 'epoch': row['epoch'],\n"
                    "                 'RMSE online': row['validation_online_RMSE'],\n"
                    "                 'RMSE EMA': row['validation']['RMSE'],\n"
                    "                 'update EMA': row['ema_updates'], 'decay EMA': row['ema_decay'],\n"
                    "                 'checkpoint A': row['epoch'] == result['checkpoints']['A']['epoch'],\n"
                    "                 'checkpoint B': row['epoch'] == result['checkpoints']['B']['epoch']}\n"
                    "                for number, result in sorted(fold_results.items())\n"
                    "                for row in result['training']['history']]),\n"
                    "            'Konfigurasi': pd.DataFrame"
                ),
                "'e28_source_sha256': E28_REFERENCE['source_sha256'],": (
                    "'e28_source_sha256': E28_REFERENCE['source_sha256'],\n"
                    "        'e29_source_sha256': E29_REFERENCE['source_sha256'], 'inference_weights': 'ema',"
                ),
            })
        elif role == 'fold_helpers':
            source = replace(source, {
                "    display(pd.DataFrame(options).round(4))": (
                    "    display(pd.DataFrame(options).round(4))\n"
                    "    display(pd.DataFrame([{'epoch': row['epoch'], 'RMSE online': row['validation_online_RMSE'],\n"
                    "                           'RMSE EMA': row['validation']['RMSE'], 'EMA updates': row['ema_updates']}\n"
                    "                          for row in info['history']]).round(4))"
                ),
            })
        elif role == 'final_report':
            source = replace(source, {
                "        print('E29 dibanding E28 · delta = E29 minus pembanding')": (
                    "        print('EMA E31 dibanding E29 · delta = E31 minus pembanding')\n"
                    "        display(sheets['E31 vs E29 per Trait'].round(4))\n"
                    "        display(sheets['E31 vs E29 per Fold'].round(4))\n"
                    "        print('E31 dibanding E28')"
                ),
                "sheets['E29 vs E28 per Trait']": "sheets['E31 vs E28 per Trait']",
                "print('E29 dibanding E25')": "print('E31 dibanding E25')",
                "sheets['E29 vs E25 per Trait']": "sheets['E31 vs E25 per Trait']",
            })
        source = source.replace('aes_reset_e29_scalar_mix_per_trait', 'aes_reset_e31_pertrait_ema')
        source = source.replace('Hasil_AES_Reset_E29_Progres', 'Hasil_AES_Reset_E31_Progres')
        source = source.replace('Hasil_AES_Reset_E29_Validasi', 'Hasil_AES_Reset_E31_Validasi')
        cell['source'] = source
        cell['id'] = f'reset-e31-{index}'

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
