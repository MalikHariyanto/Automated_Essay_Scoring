"""Build the standalone reset E6 notebook from the E2 control."""

import ast
import hashlib
import json

from build_multitrait_reset_e5_notebook import BASE, REPORT_HELPERS, ROOT, replace

NAME = 'AES_Reset_E6_S3_MultiTrait_TraitGatedConcat.ipynb'
PLACEHOLDER = '__RESET_E6_CODE_SHA256__'
GROUP = ROOT / 'eksperimen/03_tiga_input_multi_trait'

INTRO = """# E6 reset · S3 multi-trait · Gated concat per trait

E6 memakai E2 sebagai pembanding. Untuk setiap jawaban, empat gate memberi
campuran RoBERTa dan BGE yang bisa berbeda untuk isi, organisasi, kebahasaan,
dan mekanik. Ini adaptasi ide routing berbasis trait dari
[Improving Prompt Generalization for Cross-prompt Essay Trait Scoring from the Scoring-invariance Perspective](https://aclanthology.org/2025.findings-emnlp.142/).
Eksperimen ini menguji gate pada dua encoder S3 dengan loss MSE biasa.

Setiap gate membaca concat 1.024 dimensi lalu menghasilkan dua bobot softmax.
Masing-masing bobot dikali dua sebelum mengatur fitur encoder terkait.
Inisialisasi nol menghasilkan bobot 0,5 dan pengali 1 di kedua cabang,
sehingga prediksi awal sama dengan E2 pada parameter dan seed yang sama.
Gate menambah 8.200 parameter dengan learning rate 1e-4. Resep E2 memakai
RoBERTa 5e-6, head dan proyeksi 1e-4, dropout 0,2 dan split siswa yang sama.

Pilih T4 GPU, hubungkan Drive ke dataset semi-final, lalu jalankan dari atas.
Notebook langsung mengekspor ZIP ringkas 04–05 sesudah test, sebelum inspeksi
vektor dan ZIP lengkap 01–05. Sheet Gate OOF dan Gate Test mencatat bobot
dari forward yang menghasilkan prediksi. Ringkasan Gate menampilkan rerata
dan sebaran tiap pasangan trait dan encoder. Output tetap privat.
"""

REPORT_EDITS = {
    'folds, final_info, prediction):': 'folds, final_info, prediction, test_gate, inspection=None):',
    '    for result in folds:': '    oof_gate_rows = []\n    for result in folds:',
    "        history.extend(history_rows(info, fold=fold))": """        history.extend(history_rows(info, fold=fold))
        oof_gate_rows.extend(gate_rows(bundle, valid, result['gate'], fold=fold))""",
    '    return write_excel(path, {': """    oof_gate = pd.DataFrame(oof_gate_rows)
    test_gate = pd.DataFrame(gate_rows(bundle, test, test_gate))
    evaluation = {""",
    """        'Riwayat Training': pd.DataFrame(history),
        **inspection_sheets,
    }, {""": """        'Skor Terendah': pd.DataFrame(
            lowest_score_rows(bundle, indices, predictions, 'OOF')
            + lowest_score_rows(bundle, test, prediction, 'test')
        ),
        'Gate OOF': oof_gate,
        'Gate Test': test_gate,
        'Ringkasan Gate': pd.DataFrame(
            gate_summary(oof_gate, 'OOF') + gate_summary(test_gate, 'test')
        ),
    }
    groups = {
        '01_Data_dan_Tokenisasi': {},
        '02_Vektor_Jawaban': {},
        '03_Vektor_Kata_dan_Kalimat': {},
        '04_Prediksi_dan_Evaluasi': evaluation,
        '05_Riwayat_Pelatihan': {'Riwayat Training': pd.DataFrame(history)},
    }
    for name, frame in (inspection or {}).items():
        if name.startswith('Kata'):
            workbook = '03_Vektor_Kata_dan_Kalimat'
        elif name.startswith('Final_') or name in ('RoBERTa pretrained', 'BGE frozen'):
            workbook = '02_Vektor_Jawaban'
        elif name in ('Konfigurasi', 'Arsitektur'):
            workbook = '05_Riwayat_Pelatihan'
        else:
            workbook = '01_Data_dan_Tokenisasi'
        groups[workbook][name] = frame
    return export_excel_zip(path, groups, {""",
}

QUICK_EXPORT = """
    quick_path = output / 'Hasil_AES_Reset_E6_Ringkas_04_05.zip'
    export_results(quick_path, bundle, manifest, cfg, cv_summary, folds,
                   final_info, test_prediction, test_gate)
    print(f'ZIP ringkas siap diunduh: {quick_path}')
    if 'google.colab' in sys.modules:
        from google.colab import files
        files.download(str(quick_path))
"""

FULL_EXPORT = """if model_stage_ready('Ekspor ZIP lengkap', ('test_gate', 'inspection_sheets')):
    report_path = output / 'Hasil_AES_Reset_E6_S3_MultiTrait.zip'
    workbooks = export_results(report_path, bundle, manifest, cfg, cv_summary, folds,
                               final_info, test_prediction, test_gate, inspection=inspection_sheets)
    display(pd.DataFrame(workbooks)[['file', 'bytes']].assign(MB=lambda files: files['bytes'] / 1_000_000))
    print(f'ZIP lengkap: {report_path}')
"""


def model_source(source):
    helper = (ROOT / 'scripts/reset_e6_trait_gate.py').read_text()
    return replace(source, {
        'class MultiTraitModel(nn.Module):': helper + '\n\nclass MultiTraitModel(nn.Module):',
        'self.head = IndependentTraitHeads(s3_mlp_dropout)': 'self.head = TraitGatedHeads(s3_mlp_dropout)',
        '        prediction = self.head(fusion)': """        if not return_features:
            return self.head(fusion)
        prediction, gate = self.head(fusion, return_features=True)""",
        '            "concat": fusion,': '            "concat": fusion,\n            "gate": gate,',
        'return (prediction, features) if return_features else prediction': 'return prediction, features',
        'shared four-output MLP': 'gated concat before each trait head',
    })


def adapt_cell(source, role):
    if role == 'storage':
        return replace(source, {"'outputs/aes_e43_multitrait'": "'outputs/aes_reset_e6_trait_gated_concat'"})
    if role == 'prepare':
        return replace(source, {
            "'runtime': runtime_information(),": """'runtime': runtime_information(),
    'fusion_gate': 'trait_softmax_2x_identity', 'gate_lr': 1e-4,""",
        })
    if role == 'definition:SingleScoreModel':
        return model_source(source)
    if role == 'architecture':
        return replace(source, {
            "    ['Fusion concat', '1024'],":
                "    ['Concat pembaca gate', '1024'],\n"
                "    ['Gate per trait', '1024 → 2'],\n"
                "    ['Concat berbobot per trait', '1024'],",
        })
    if role == 'definition:optimizer_groups':
        return replace(source, {
            "'pooling_norm': []}": "'pooling_norm': [], 'gate': []}",
            "        elif name.startswith('head.'):":
                "        elif name.startswith('head.gates.'):\n            group = 'gate'\n        elif name.startswith('head.'):",
            "'pooling_norm': cfg['s3_layers_lr']}": "'pooling_norm': cfg['s3_layers_lr'], 'gate': cfg['gate_lr']}",
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
            'Validasi silang satu resep E3': 'Validasi silang E6',
            "            prediction = predictions_array(engine.predict(validation), validation)":
                "            scores, gate = engine.predict(validation, return_gate=True)\n"
                "            prediction = predictions_array(scores, validation)",
            "'prediction': prediction.tolist(), 'training': info":
                "'prediction': prediction.tolist(), 'gate': gate.tolist(), 'training': info",
        })
    if role == 'test_definition':
        return replace(source, {
            "return predictions_array(result['prediction'], test)":
                "return predictions_array(result['prediction'], test), result['gate']",
            '    prediction = predictions_array(engine.predict(test), test)':
                '    scores, gate = engine.predict(test, return_gate=True)\n'
                '    prediction = predictions_array(scores, test)',
            "'prediction': prediction.tolist()}": "'prediction': prediction.tolist(), 'gate': gate.tolist()}",
            '\n    return prediction\n': '\n    return prediction, gate\n',
        })
    if role == 'test_execute':
        return replace(source, {
            '    test_prediction = evaluate_test': '    test_prediction, test_gate = evaluate_test',
        }) + '\n' + QUICK_EXPORT
    if role == 'definition:write_excel':
        return source + '\n\n' + (ROOT / 'scripts/reset_e5_excel_zip.py').read_text()
    if role == 'report':
        return REPORT_HELPERS + replace(source, REPORT_EDITS)
    if role == 'final_features':
        return replace(source, {
            '{"bagian": "MLP hidden", "dimensi": 256}':
                '{"bagian": "Gate per trait", "dimensi": "1024 → 2"},\n'
                '            {"bagian": "Hidden per trait", "dimensi": 64}',
        })
    if role == 'export_execute':
        return FULL_EXPORT
    if role == 'download':
        return source.replace('Unduh workbook', 'Unduh ZIP lengkap')
    return (source.replace('Susun Excel hasil', 'Inspeksi vektor dan susun ZIP lengkap')
            .replace('Unduh Excel', 'Unduh ZIP lengkap'))


def build():
    raw = BASE.read_bytes()
    notebook = json.loads(raw)
    base_hash = notebook['metadata']['aes_standalone']['code_sha256']
    cells = notebook['cells']
    for cell in cells:
        source = ''.join(cell['source']).replace(base_hash, PLACEHOLDER)
        source = (source.replace('Reset_E2', 'Reset_E6')
                  .replace('reset_e2_s3_multitrait', 'reset_e6_s3_multitrait')
                  .replace('E2 reset', 'E6 reset'))
        cell['source'] = adapt_cell(source, cell['metadata'].get('aes_multitrait_role'))

    # Move expensive inspection after the quick export while keeping all definitions in order.
    inspection_index = next(i for i, c in enumerate(cells)
                            if c['metadata'].get('aes_multitrait_role') == 'inspection')
    inspection = cells.pop(inspection_index)
    heading = cells.pop(inspection_index - 1)
    final_index = next(i for i, c in enumerate(cells)
                       if c['metadata'].get('aes_multitrait_role') == 'final_features')
    cells[final_index:final_index] = [heading, inspection]
    cells[0]['source'] = INTRO
    fingerprint = hashlib.sha256(
        '\n'.join(c['source'] for c in cells if c['cell_type'] == 'code').encode()
    ).hexdigest()
    for i, cell in enumerate(cells):
        cell['id'] = f'reset-e6-{i}'
        cell['source'] = cell['source'].replace(PLACEHOLDER, fingerprint)
        if cell['cell_type'] == 'code':
            ast.parse(cell['source'])
            cell['outputs'] = []
            cell['execution_count'] = None

    notebook['metadata']['colab']['name'] = NAME
    notebook['metadata']['aes_standalone'] = {
        'code_sha256': fingerprint, 'base': BASE.relative_to(ROOT).as_posix(),
        'base_sha256': hashlib.sha256(raw).hexdigest(), 'experiment': 'Reset_E6',
    }
    serialized = json.dumps(notebook, ensure_ascii=False, indent=1) + '\n'
    for folder in (BASE.parent, GROUP / 'notebooks'):
        (folder / NAME).write_text(serialized)
    print(f'{NAME} · {fingerprint}')


if __name__ == '__main__':
    build()
