"""Build shared-head E15 with trait-gated S3 and bounded ZIP exports."""

import hashlib
import json

from build_multitrait_reset_e6_notebook import BASE, GROUP, ROOT, replace
from build_multitrait_reset_e14_notebook import NAME as E14_NAME

NAME = 'AES_Reset_E15_S1_S2_S3_TraitGatedConcat.ipynb'
MARKER = '__RESET_E15_CODE_SHA256__'
INTRO = """# E15 reset · S1, S2, S3 multi-trait · Gate per trait

S1 memakai RoBERTa Indonesia dan S2 memakai BGE-M3 frozen. Keduanya memakai
arsitektur E14. S3 menambahkan satu gate per trait sebelum concat, dengan
MLP 256 yang dipakai bersama. Bandingkan S3 dengan concat E14 pada lima fold
yang sama. E15 mempertahankan resep MSE, optimizer, dropout, dan LR E14.

Gate berawal dari 50:50 sehingga pengali kedua cabang adalah 1. Gate dilatih
dengan LR head 1e-4. Ide routing per trait terinspirasi dari
[Wang & Yu (2025)](https://aclanthology.org/2025.findings-emnlp.142/).
Gate dua encoder dan angka LR adalah adaptasi proyek.

Pilih T4 GPU lalu Run all. Unduh ZIP 04–05 masing-masing untuk S1, S2, dan S3.
Setiap ZIP evaluasi berisi metrik per trait pada validasi dan test, prediksi,
dan riwayat training. S3 juga mencatat bobot gate. ZIP tambahan 01–03 berisi
data dan inspeksi vektor S3, serta bisa terbagi menjadi beberapa ZIP.
Ukuran aktual setiap ZIP dicek agar kurang dari 30.000.000 byte. Hasil privat.
"""

DOWNLOAD = """def show_zip_downloads(scenario, archives):
    display(pd.DataFrame([{'skenario': scenario, 'zip': path.name, 'bytes': size}
                          for path, size, _ in archives]))
    if 'google.colab' in sys.modules:
        from google.colab import files
        from ipywidgets import Button, VBox

        def button_for(path):
            button = Button(description=f'Unduh {path.name}', layout={'width': 'auto'})
            button.on_click(lambda _: files.download(str(path)))
            return button

        display(VBox([button_for(path) for path, _, _ in archives]))
"""


def zip_writer(source):
    prefix, marker, _ = source.partition('        # XLSX sudah dikompresi.')
    if not marker:
        raise ValueError('Exporter E14 berubah.')
    prefix = replace(prefix, {'            if size < max_bytes:':
                              '            if size < max_bytes - 65_536:'})
    return (prefix + '        return pack_excel_zip_parts(path, staging, inventory, max_bytes)\n\n\n'
            + (ROOT / 'scripts/reset_e15_zip_parts.py').read_text())


def adapt(role, source):
    if role == 'prepare':
        return replace(source, {"'fusion_variant': 'concat'":
                                "'fusion_variant': 's3_trait_gated_concat', 'gate_lr': S3_MLP_LR"})
    if role == 'definition:SingleScoreModel':
        helpers = (ROOT / 'scripts/reset_e6_trait_gate.py').read_text().split('\n\ndef gate_rows', 1)[1]
        return replace(source, {
            '    def forward(self, windows,': """        if scenario == 'S3':
            with torch.random.fork_rng(devices=[]):
                self.gate = nn.Linear(1024, 2 * len(TRAITS))
                nn.init.zeros_(self.gate.weight)
                nn.init.zeros_(self.gate.bias)

    def forward(self, windows,""",
            '        prediction = self.head(representation)': """            alpha = self.gate(representation).reshape(len(cls), len(TRAITS), 2).float().softmax(dim=-1)
            scales = (2 * alpha).to(representation.dtype)
            roberta, projected = representation.split((768, 256), dim=-1)
            gated = torch.cat([roberta[:, None, :] * scales[:, :, :1],
                               projected[:, None, :] * scales[:, :, 1:]], dim=-1)
            hidden = self.head[1](self.head[0](gated))
            # Share the dropout mask across traits, including during the initial 50:50 forward.
            hidden = hidden * self.head[2](torch.ones_like(hidden[:, :1]))
            prediction = self.head[3](hidden).diagonal(dim1=1, dim2=2)
            features['gate'] = alpha
            return (prediction, features) if return_features else prediction
        prediction = self.head(representation)""",
        }) + '\n\ndef gate_rows' + helpers
    if role == 'definition:optimizer_groups':
        return replace(source, {
            "'pooling_norm': []}": "'pooling_norm': [], 'gate': []}",
            "        elif name.startswith('head.'):":
                "        elif name.startswith('gate.'):\n            group = 'gate'\n        elif name.startswith('head.'):",
            "'pooling_norm': cfg['s3_layers_lr']}":
                "'pooling_norm': cfg['s3_layers_lr'], 'gate': cfg['gate_lr']}",
        })
    if role == 'definition:Engine':
        return replace(source, {
            '    def predict(self, indices):': '    def predict(self, indices, return_gate=False):',
            '        values = []': '        values, gates = [], []',
            '                values.append(self.forward(ids).float().cpu().numpy())': """                if return_gate:
                    prediction, features = self.forward(ids, return_features=True)
                    gates.append(features['gate'].cpu().numpy())
                else:
                    prediction = self.forward(ids)
                values.append(prediction.float().cpu().numpy())""",
            '        return scaled * SCORE_SPAN + SCORE_MIN': """        scores = scaled * SCORE_SPAN + SCORE_MIN
        return (scores, np.concatenate(gates)) if return_gate else scores""",
        })
    if role == 'cv_definition':
        return replace(source, {
            '            prediction = predictions_array(engine.predict(validation), validation)': """            if scenario == 'S3':
                scores, gate = engine.predict(validation, return_gate=True)
            else:
                scores, gate = engine.predict(validation), None
            prediction = predictions_array(scores, validation)""",
            "'prediction': prediction.tolist(), 'training': info":
                "'prediction': prediction.tolist(), 'gate': gate.tolist() if gate is not None else None, 'training': info",
        })
    if role == 'test_definition':
        return replace(source, {
            "return predictions_array(result['prediction'], test)":
                "return predictions_array(result['prediction'], test), result['gate']",
            '    prediction = predictions_array(engine.predict(test), test)': """    if model.scenario == 'S3':
        scores, gate = engine.predict(test, return_gate=True)
    else:
        scores, gate = engine.predict(test), None
    prediction = predictions_array(scores, test)""",
            "'prediction': prediction.tolist()}":
                "'prediction': prediction.tolist(), 'gate': gate.tolist() if gate is not None else None}",
            '\n    return prediction\n': '\n    return prediction, gate\n',
        })
    if role == 'definition:write_excel':
        return zip_writer(source)
    if role == 'report':
        return replace(source, {
            'prediction, inspection=None):': 'prediction, gate=None, inspection=None):',
            "    groups = {": """    if cfg['scenario'] == 'S3' and inspection is None:
        oof_gate = pd.DataFrame([
            row for result in folds
            for row in gate_rows(bundle, result['identity']['validation'], result['gate'],
                                 fold=result['identity']['fold'])
        ])
        test_gate = pd.DataFrame(gate_rows(bundle, test, gate))
        evaluation.update({
            'Gate OOF': oof_gate,
            'Gate Test': test_gate,
            'Ringkasan Gate': pd.DataFrame(gate_summary(oof_gate, 'OOF') + gate_summary(test_gate, 'test')),
        })
    groups = {""",
            '    return export_excel_zip(path, groups, {': """    if inspection is not None:
        groups.pop('04_Prediksi_dan_Evaluasi')
        groups.pop('05_Riwayat_Pelatihan')
    return export_excel_zip(path, groups, {""",
        })
    if role == 'test_execute':
        source = source[:source.index("    if 'google.colab' in sys.modules:")]
        return DOWNLOAD + '\n\n' + replace(source, {
            'test_predictions, quick_paths = {}, {}': 'test_predictions = {}',
            '        prediction = evaluate_test(': '        prediction, gate = evaluate_test(',
            "        suffix = '' if scenario == 'S3' else f'{scenario}_'\n": '',
            'Hasil_AES_Reset_E15_{suffix}Ringkas_04_05.zip': 'Hasil_AES_Reset_E15_{scenario}_Ringkas_04_05.zip',
            "        export_results(quick_path, bundle, manifest, {**cfg, 'scenario': scenario},\n"
            "                       summary, folds, refit_result['training'], prediction)":
                """        scenario_cfg = {**cfg, 'scenario': scenario,
                        'fusion_variant': cfg['fusion_variant'] if scenario == 'S3' else 'single_encoder'}
        archives = export_results(quick_path, bundle, manifest, scenario_cfg,
                                  summary, folds, refit_result['training'], prediction, gate=gate)
        show_zip_downloads(scenario, archives)""",
            "        quick_paths[scenario] = quick_path\n        print(f'{scenario}: ZIP ringkas {quick_path}')\n": '',
            '# ZIP lengkap memakai model dan laporan S3, seperti alur E13.': '# Inspeksi vektor memakai checkpoint S3.',
        })
    if role == 'export_execute':
        return """if 'S3' in RUN_SCENARIOS and model_stage_ready('Ekspor ZIP inspeksi', ('test_prediction', 'inspection_sheets')):
    report_path = output / 'Hasil_AES_Reset_E15_S3_Inspeksi_01_03.zip'
    archives = export_results(report_path, bundle, manifest, {**cfg, 'scenario': 'S3'}, cv_summary, folds,
                              final_info, test_prediction, inspection=inspection_sheets)
    show_zip_downloads('S3 inspeksi', archives)
"""
    if role == 'download':
        return """# Tombol unduhan muncul pada cell evaluasi dan ekspor inspeksi."""
    return source


def build():
    base = BASE.parent / E14_NAME
    raw = base.read_bytes()
    notebook = json.loads(raw)
    previous_hash = notebook['metadata']['aes_standalone']['code_sha256']
    for index, cell in enumerate(notebook['cells']):
        source = ''.join(cell['source']).replace(previous_hash, MARKER)
        source = (source.replace('Reset_E14', 'Reset_E15').replace('E14 reset', 'E15 reset')
                  .replace('reset_e14_s1_s2_s3_multitrait', 'reset_e15_s1_s2_s3_multitrait')
                  .replace('aes_reset_e14_s1_s2_s3', 'aes_reset_e15_trait_gated_concat'))
        if cell['cell_type'] == 'markdown':
            source = (source.replace('susun ZIP lengkap', 'susun ZIP inspeksi 01–03')
                      .replace('Unduh ZIP lengkap', 'Unduh hasil ZIP per bagian'))
        cell['source'] = adapt(cell['metadata'].get('aes_multitrait_role'), source)
        cell['id'] = f'reset-e15-{index}'
    notebook['cells'][0]['source'] = INTRO
    code = '\n'.join(cell['source'] for cell in notebook['cells'] if cell['cell_type'] == 'code')
    fingerprint = hashlib.sha256(code.encode()).hexdigest()
    for cell in notebook['cells']:
        cell['source'] = cell['source'].replace(MARKER, fingerprint)
        if cell['cell_type'] == 'code':
            compile(cell['source'], cell['id'], 'exec')
    notebook['metadata']['colab']['name'] = NAME
    notebook['metadata']['aes_standalone'] = {
        'code_sha256': fingerprint, 'base': base.relative_to(ROOT).as_posix(),
        'base_sha256': hashlib.sha256(raw).hexdigest(), 'experiment': 'Reset_E15',
    }
    content = json.dumps(notebook, ensure_ascii=False, indent=1) + '\n'
    for folder in (BASE.parent, GROUP / 'notebooks'):
        (folder / NAME).write_text(content)


if __name__ == '__main__':
    build()
