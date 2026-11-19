"""Build the grouped GitHub notebook catalog and public aggregate result page."""

import argparse
import hashlib
import html
import json
import re
from pathlib import Path
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / 'eksperimen'
BRANCH = 'codex/aes-learning'
REPOSITORY = 'MalikHariyanto/Automated_Essay_Scoring'
GROUPS = {
    '01_dua_input': ('Dua input dan eksperimen awal', 'Kelompok historis sebelum tiga input seragam. Varian jawaban saja dan context-pair diberi keterangan per notebook. Target satu skor rater 0–100.'),
    '02_tiga_input_single_trait': ('Tiga input · satu trait', 'Soal, kunci jawaban, dan jawaban siswa menjadi input. Target satu skor rater 0–100. Termasuk E21–E56, varian tiga input terdahulu, Ridge, dan diagnostik BGE.'),
    '03_tiga_input_multi_trait': ('Tiga input · empat trait', 'Pertanyaan, kunci jawaban, dan jawaban siswa menjadi input. Target isi, organisasi, kebahasaan, dan mekanik. Penomoran E1–E5 berlaku di kelompok ini.'),
}
ASSETS = {
    '01_dua_input': [
        ('docs/belajar_MLP_Spider.html', 'Ringkasan hasil dan panduan eksperimen awal'),
        ('downloads/Belajar_AES_MLP_Lengkap.html', 'Pengunduh buku lengkap eksperimen awal'),
    ],
    '02_tiga_input_single_trait': [
        ('docs/belajar_Tiga_Input_Spider.html', 'Ringkasan hasil tiga input'),
        ('downloads/Belajar_AES_Tiga_Input_Lengkap.html', 'Pengunduh buku lengkap tiga input'),
        ('downloads/Panduan_E30_MLP.html', 'Panduan E30 MLP'),
    ],
}
MULTI_NAMES = {
    'AES_S3_MultiTrait_3Sekolah.ipynb': ('E1', 'Baseline E43, head bersama'),
    'AES_E2_S3_MultiTrait_TraitHeads_3Sekolah.ipynb': ('E2', 'Empat head khusus trait'),
    'AES_E3_S3_MultiTrait_Dropout02_3Sekolah.ipynb': ('E3', 'E1 dengan dropout MLP 0,2'),
    'AES_E4_S3_MultiTrait_Dropout02_Sigmoid_3Sekolah.ipynb': ('E4', 'E3 dengan sigmoid pada keluaran'),
    'AES_E5_S3_MultiTrait_InterTraitAttention_3Sekolah.ipynb': ('E5', 'E3 dengan attention antartait'),
}


def classify(path):
    if path.name in MULTI_NAMES:
        return '03_tiga_input_multi_trait'
    notebook = json.loads(path.read_text())
    source = '\n'.join(''.join(cell['source']) for cell in notebook['cells'])
    if ('three_texts_all_scenarios_v1' in source or 'tiga_input' in path.name.lower()
        or 'Ridge' in path.name or 'BGE_Frozen_Readout' in path.name):
        return '02_tiga_input_single_trait'
    return '01_dua_input'


def organized_name(path):
    if path.name == 'AES_S3_MultiTrait_3Sekolah.ipynb':
        return 'AES_E1_S3_MultiTrait_3Sekolah.ipynb'
    return path.name


def colab(path):
    relative = path.relative_to(ROOT).as_posix()
    return f'https://colab.research.google.com/github/{REPOSITORY}/blob/{quote(BRANCH, safe="")}/{relative}'


def import_results(paths):
    """Export only aggregate statistics. Source workbooks stay outside the repository."""
    from openpyxl import load_workbook

    results = []
    for path in paths:
        book = load_workbook(path, read_only=True, data_only=True)
        try:
            def records(sheet):
                data = iter(book[sheet].values)
                headers = next(data)
                return [dict(zip(headers, row)) for row in data]

            info = {row['field']: row['value'] for row in records('Info')}
            experiment = 'E1' if info['eksperimen'] == 'E43' else info['eksperimen']
            cv = records('Ringkasan CV')[0]
            test = records('Test per Trait')
            oof = records('CV OOF per Trait')
            pooled_test = sum(row['RMSE'] ** 2 for row in test) / len(test)
            results.append({
                'experiment': experiment, 'dataset_sha256': info['source_sha256'],
                'split_sha256': info['split_sha256'],
                'workbook_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                'development_rows': info['development_rows'], 'test_rows': info['test_rows'],
                'cv_rmse': cv['mean_cv_RMSE'], 'test_rmse': pooled_test ** 0.5,
                'refit_epochs': cv['epoch_refit'],
                'test': [{key: row[key] for key in ('trait', 'RMSE', 'Pearson', 'QWK', 'Gwet AC2')}
                         for row in test],
                'oof': [{key: row[key] for key in ('trait', 'RMSE', 'Pearson', 'QWK', 'Gwet AC2')}
                        for row in oof],
            })
        finally:
            book.close()
    if len({r['experiment'] for r in results}) != len(results):
        raise ValueError('Workbook eksperimen duplikat.')
    if len({(r['dataset_sha256'], r['split_sha256']) for r in results}) != 1:
        raise ValueError('Dataset atau split workbook berbeda.')
    return sorted(results, key=lambda row: row['experiment'])


def results_page(results):
    def table(headers, rows):
        head = ''.join(f'<th scope="col">{html.escape(str(value))}</th>' for value in headers)
        body = ''.join('<tr>' + ''.join(f'<td>{html.escape(str(v))}</td>' for v in row) + '</tr>'
                       for row in rows)
        return f'<div class="table-wrap" tabindex="0"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'

    summary = table(
        ['Eksperimen', 'RMSE CV ↓', 'RMSE test ↓', 'Epoch refit', 'Perubahan'],
        [[r['experiment'], f"{r['cv_rmse']:.4f}", f"{r['test_rmse']:.4f}", r['refit_epochs'],
          next(v[1] for v in MULTI_NAMES.values() if v[0] == r['experiment'])] for r in results],
    )
    trait_tables = ''
    for subset, title in (('oof', 'Validasi out-of-fold'), ('test', 'Test')):
        rows = [[r['experiment'], t['trait'], *[f'{t[k]:.4f}' for k in ('RMSE', 'Pearson', 'QWK', 'Gwet AC2')]]
                for r in results for t in r[subset]]
        trait_tables += f'<section><h2>{title} per trait</h2>' + table(
            ['Eksperimen', 'Trait', 'RMSE ↓', 'Pearson ↑', 'QWK ↑', 'Gwet AC2 ↑'], rows,
        ) + '</section>'
    return f"""<!doctype html>
<html lang="id"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Hasil AES multi-trait · E1–E5</title>
<style>
:root{{font-family:system-ui,sans-serif;color:#20322b;background:#f4f5f0}}
*{{box-sizing:border-box}}body{{margin:0}}main{{max-width:1120px;margin:auto;padding:40px 22px}}
a{{color:#145b46}}header{{padding:28px 0}}h1{{font-size:clamp(2rem,5vw,3.5rem);margin:12px 0}}
p{{line-height:1.7;max-width:850px;color:#4b5e54}}section{{margin:30px 0}}
.table-wrap{{overflow:auto;border:1px solid #d5ded5;border-radius:12px;background:white}}
table{{width:100%;border-collapse:collapse;white-space:nowrap}}th,td{{padding:14px 16px;text-align:left;border-bottom:1px solid #e5ebe5;font-variant-numeric:tabular-nums}}
th{{background:#e4ece3;color:#244b3a}}tbody tr:hover{{background:#f1f7f0}}tr:last-child td{{border-bottom:0}}
footer{{margin-top:35px;padding-top:20px;border-top:1px solid #d5ded5}}
</style></head><body><main><header><a href="../README.md">Katalog multi-trait</a>
<h1>Hasil prediksi multi-trait</h1>
<p>Agregat dari workbook hasil E1–E3. Development 1.070 jawaban dan test 270 jawaban, dengan dataset serta split siswa yang sama. E1 adalah nama baru untuk baseline multi-trait E43.</p></header>
<section><h2>Perbandingan eksperimen</h2><p>Geser tabel ke samping untuk melihat seluruh kolom pada layar kecil.</p>{summary}
<p>E3 menjadi acuan aktif berdasarkan RMSE CV dan test terendah pada hasil yang tersedia. E4 dan E5 masih menunggu workbook hasil.</p>
<p>RMSE memakai poin rubrik asli. Isi berskala 1–4, tiga trait lain 1–2. RMSE test gabungan dihitung dari akar rata-rata kuadrat RMSE empat trait. RMSE CV merupakan rata-rata RMSE gabungan lima fold.</p></section>
{trait_tables}<footer><p>Sumber agregat, identitas dataset, dan split dicatat dalam <a href="hasil_agregat.json">hasil_agregat.json</a>. QWK dan AC2 memakai kategori skor. Hasil ini berasal dari satu run per eksperimen.</p></footer>
</main></body></html>
"""


def build(result_paths=()):
    CATALOG.mkdir(exist_ok=True)
    entries = {group: [] for group in GROUPS}
    manifest = []
    for source in sorted((ROOT / 'notebooks/experiments').glob('*.ipynb')):
        group = classify(source)
        target = CATALOG / group / 'notebooks' / organized_name(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
        entries[group].append((source, target))
        manifest.append({'source': source.relative_to(ROOT).as_posix(),
                         'organized': target.relative_to(ROOT).as_posix(),
                         'sha256': hashlib.sha256(source.read_bytes()).hexdigest()})

    for group, assets in ASSETS.items():
        for source_name, _ in assets:
            source = ROOT / source_name
            target = CATALOG / group / 'html' / source.name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source.read_bytes())
            manifest.append({'source': source_name, 'organized': target.relative_to(ROOT).as_posix(),
                             'sha256': hashlib.sha256(source.read_bytes()).hexdigest()})
    html_dir = CATALOG / '03_tiga_input_multi_trait/html'
    html_dir.mkdir(parents=True, exist_ok=True)
    aggregate = html_dir / 'hasil_agregat.json'
    if result_paths:
        aggregate.write_text(json.dumps(import_results(result_paths), ensure_ascii=False, indent=2) + '\n')
    results = json.loads(aggregate.read_text()) if aggregate.exists() else []
    if results:
        (html_dir / 'Hasil_AES_MultiTrait.html').write_text(results_page(results))

    for group, (title, description) in GROUPS.items():
        folder = CATALOG / group
        lines = [f'# {title}', '', description, '', '[Semua kelompok](../README.md)', '',
                 '## Notebook', '']
        if group == '03_tiga_input_multi_trait':
            lines += ['| Eksperimen | Konfigurasi | Notebook | Jalankan |',
                      '| --- | --- | --- | --- |']
        else:
            lines += ['| Notebook | Input / varian | Jalankan |', '| --- | --- | --- |']
        ordered = entries[group]
        if group == '03_tiga_input_multi_trait':
            ordered = sorted(ordered, key=lambda item: MULTI_NAMES[item[0].name][0])
        for source, target in ordered:
            link = f'[File](notebooks/{target.name})'
            run = f'[Colab]({colab(target)})'
            if group == '03_tiga_input_multi_trait':
                experiment, desc = MULTI_NAMES[source.name]
                lines.append(f'| {experiment} | {desc} | {link} | {run} |')
            else:
                variant = ('Context-pair pada S3' if 'ContextPair' in source.name else
                           'Jawaban saja / eksperimen awal') if group == '01_dua_input' else 'Tiga teks · satu skor'
                lines.append(f'| [{source.stem}](notebooks/{target.name}) | {variant} | {run} |')
        lines += ['', '## HTML hasil dan panduan', '']
        if group in ASSETS:
            lines += [f'- [{label}](html/{Path(source).name})' for source, label in ASSETS[group]]
            lines += ['', 'Pengunduh buku lengkap tetap mengambil bagian paket dari folder downloads. '
                      'Buka file HTML yang sudah diunduh untuk menjalankan pengunduh.']
        else:
            lines += ['- [Ringkasan hasil E1–E3](html/Hasil_AES_MultiTrait.html)',
                      '- [Agregat metrik](html/hasil_agregat.json)', '',
                      'E3 adalah acuan aktif. Workbook E4/E5 belum tersedia pada pembaruan ini. '
                      'E1 pada tabel hasil sama dengan E43 multi-trait pada workbook awal.']
        lines += ['', '## Lokasi file', '',
                  'Folder ini menjadi pintu masuk notebook dan HTML yang sudah dikelompokkan. '
                  'Salinan di notebooks/experiments dan aes-learning/notebooks/experiments menyediakan '
                  'jalur sumber untuk builder dan kompatibilitas tautan Colab yang sudah dibagikan.', '']
        (folder / 'README.md').write_text('\n'.join(lines))

    counts = {group: len(rows) for group, rows in entries.items()}
    (CATALOG / 'README.md').write_text(
        '# Katalog eksperimen AES\n\n'
        '| Kelompok | Target | Notebook |\n| --- | --- | --- |\n' +
        '\n'.join(f'| [{GROUPS[g][0]}]({g}/README.md) | '
                  f'{"Empat skor" if g.endswith("multi_trait") else "Satu skor"} | {counts[g]} |' for g in GROUPS) +
        '\n\nSetiap kelompok mempunyai folder notebooks dan html. Nomor E dibaca dalam kelompoknya. '
        'E1–E5 multi-trait merupakan rangkaian eksperimen terbaru.\n\n'
        '[Kembali ke halaman utama](../README.md)\n'
    )
    (CATALOG / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'grouped_notebooks': counts, 'copied_assets': len(manifest), 'result_workbooks': len(results)}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', nargs='*', type=Path, default=[])
    args = parser.parse_args()
    build(args.results)
