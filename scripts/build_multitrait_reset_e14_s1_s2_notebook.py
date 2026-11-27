"""Build the E14 single-encoder comparison with one E13-style report ZIP."""

import hashlib
import json

from build_multitrait_reset_e6_notebook import BASE, GROUP, ROOT, replace
from build_multitrait_reset_e14_notebook import NAME as E14_NAME

NAME = 'AES_Reset_E14_S1_S2_Banding_MultiTrait.ipynb'
MARKER = '__RESET_E14_S1_S2_CODE_SHA256__'
INTRO = """# E14 · Bandingkan S1 dan S2 pada empat trait

S1 melatih RoBERTa Indonesia. S2 melatih head dari fitur BGE-M3 frozen.
Keduanya mengikuti resep S1/S2 E14: MLP bersama hidden 256, input soal,
kunci, dan jawaban, lima fold kelompok yang sama, maksimal 25 epoch, dan patience 5.

Pilih T4 GPU lalu jalankan dari atas. Setiap skenario menjalankan lima fold
validasi, satu refit, dan satu evaluasi test. Hasil fold yang selesai dan
checkpoint disimpan di Drive untuk melanjutkan runtime yang terputus.
Output setiap bagian tampil setelah tahap itu dijalankan.

Unduh `Hasil_AES_Reset_E14_S1_S2_MultiTrait.zip` setelah bagian 20.
Excel 01–03 berisi data dan inspeksi fitur, Excel 04 berisi prediksi dan
empat metrik per trait, dan Excel 05 berisi riwayat pelatihan. Kolom
`skenario` di Excel 04–05 membedakan S1 dan S2. Simpan hasil siswa secara privat.
"""

TEST_EXECUTE = """if model_stage_ready('Evaluasi test', ('bge', 'cv_results', 'refits')):
    test_predictions = {}
    for scenario in RUN_SCENARIOS:
        prediction = evaluate_test(refits[scenario], bundle, manifest, bge, cfg,
                                   output / scenario)
        test_predictions[scenario] = prediction
        result = metrics(bundle.y[manifest['test']], prediction)
        print(f'{scenario} · RMSE test gabungan {result["RMSE"]:.4f}')
        display(pd.DataFrame(trait_rows(result, skenario=scenario, n=len(prediction))).round(4))
"""

EXPORT_EXECUTE = """if model_stage_ready('Ekspor ZIP lengkap', ('test_predictions', 'inspection_sheets')):
    reports, metadata = [], None
    for scenario in RUN_SCENARIOS:
        summary, folds = cv_results[scenario]
        inspection = inspection_sheets if scenario == 'S1' else None
        tables, metadata = result_tables(
            bundle, manifest, cfg, summary, folds, refits[scenario]['training'],
            test_predictions[scenario], inspection=inspection,
        )
        reports.append((scenario, tables))

    groups = reports[0][1]
    for name in ('04_Prediksi_dan_Evaluasi', '05_Riwayat_Pelatihan'):
        for sheet in groups[name].copy():
            if sheet in ('Konfigurasi', 'Arsitektur'):
                continue
            groups[name][sheet] = pd.concat(
                [tables[name][sheet].assign(skenario=scenario) for scenario, tables in reports],
                ignore_index=True,
            )

    report_path = output / 'Hasil_AES_Reset_E14_S1_S2_MultiTrait.zip'
    workbooks = export_excel_zip(report_path, groups, {**metadata, 'skenario': list(RUN_SCENARIOS)})
    display(pd.DataFrame(workbooks)[['file', 'bytes']].assign(MB=lambda f: f['bytes'] / 1_000_000))
    print(f'ZIP lengkap S1 dan S2: {report_path}')
"""


def update_cell(role, source):
    if role == 'hyperparameters':
        return replace(source, {
            "# Untuk mengambil ZIP S1 dan S2 saja, ganti baris berikut dengan ('S1', 'S2').\n"
            "RUN_SCENARIOS = ('S1', 'S2', 'S3')":
                "# S1 dan S2 adalah kontrol encoder tunggal dari E14.\nRUN_SCENARIOS = ('S1', 'S2')",
        })
    if role == 'storage':
        return replace(source, {
            "OUTPUT_ROOT = Path('outputs/aes_reset_e14_s1_s2_s3')":
                """OUTPUT_NAME = 'aes_reset_e14_s1_s2_banding'
IN_COLAB = 'google.colab' in sys.modules or Path('/content').is_dir()
OUTPUT_ROOT = (DRIVE_MOUNT / 'MyDrive' / 'AES' / 'outputs' / OUTPUT_NAME
               if IN_COLAB else Path('outputs') / OUTPUT_NAME)""",
        })
    if role == 'prepare':
        return replace(source, {"'fusion_variant': 'concat'": "'fusion_variant': 'single_encoder'"})
    if role == 'architecture':
        return '\n'.join(line for line in source.splitlines() if "('S3'," not in line)
    if role == 'test_execute':
        return TEST_EXECUTE
    if role == 'definition:vector_lab':
        return replace(source, {"wrapper = make_model('S3',cfg)":
                                "wrapper = make_model(cfg['vector_source'],cfg)"})
    if role == 'inspection':
        return source.replace("if 'S3' in RUN_SCENARIOS and ", 'if ', 1)
    if role == 'final_features':
        return replace(source, {
            """if 'S3' in RUN_SCENARIOS and model_stage_ready('Inspeksi fitur final', ('bge', 'refit_result')):
    s3_folder = output / "S3"
    payload = torch.load(s3_folder / "final.pt", map_location="cpu", weights_only=True)
    model = make_model("S3", cfg)""":
                """if model_stage_ready('Inspeksi fitur final S1', ('bge', 'refits', 'inspection_sheets')):
    s1_folder = output / 'S1'
    checkpoint = s1_folder / 'final.pt'
    if digest(checkpoint) != refits['S1']['model_sha256']:
        raise ValueError('Checkpoint S1 berubah.')
    payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
    model = make_model('S1', cfg)""",
            '["roberta", "projection", "concat"]': '["roberta"]',
            '"Final_" + name': '"Final_S1_" + name',
            '"vector_source": "S3"': '"vector_source": "S1"',
            '        s3_folder,': '        s1_folder,',
            """            {"bagian": "RoBERTa CLS terakhir", "dimensi": 768},
            {"bagian": "BGE frozen", "dimensi": 1024},
            {"bagian": "Proyeksi BGE", "dimensi": 256},
            {"bagian": "Concat", "dimensi": 1024},""":
                """            {"bagian": "S1 RoBERTa CLS terakhir", "dimensi": 768},
            {"bagian": "S2 BGE frozen", "dimensi": 1024},""",
        })
    if role == 'report':
        return replace(source, {
            'def export_results(path, bundle': 'def result_tables(bundle',
            'return export_excel_zip(path, groups, {': 'return groups, {',
            "'kriteria_early_stopping': 'Akar rata-rata kuadrat error seluruh skor validasi dalam satuan asli.',\n    })":
                "'kriteria_early_stopping': 'Akar rata-rata kuadrat error seluruh skor validasi dalam satuan asli.',\n    }",
        })
    if role == 'export_execute':
        return EXPORT_EXECUTE
    if role == 'download':
        return source.replace("if 'S3' in RUN_SCENARIOS and ", 'if ', 1)
    return source


def build():
    base = BASE.parent / E14_NAME
    raw = base.read_bytes()
    notebook = json.loads(raw)
    old_hash = notebook['metadata']['aes_standalone']['code_sha256']
    markdown = {
        0: INTRO,
        60: '## 12 · Model multi-trait untuk S1 dan S2',
        61: 'S1 memakai RoBERTa fine-tuned. S2 memakai BGE frozen. Masing-masing memakai MLP 256 dengan empat keluaran.',
        116: '## 18 · Evaluasi test S1 dan S2\nHasil test ditampilkan setelah kedua model selesai refit.',
        127: '## 19 · Inspeksi dan susun ZIP lengkap\nExcel 02 memuat RoBERTa sebelum dan setelah fine-tuning S1, serta BGE frozen sebagai fitur S2. Excel 03 menampilkan contoh vektor kata dan kalimat.',
    }
    for index, cell in enumerate(notebook['cells']):
        source = ''.join(cell['source']).replace(old_hash, MARKER)
        role = cell['metadata'].get('aes_multitrait_role')
        cell['source'] = markdown[index] if index in markdown else update_cell(role, source)
        cell['id'] = f'reset-e14-s1s2-{index}'

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
