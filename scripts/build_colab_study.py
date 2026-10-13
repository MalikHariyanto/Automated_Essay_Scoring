"""Buku HTML Colab AI dari snapshot agregat aman + source publik.

--notebook menerima notebook terisi lokal, lalu mengambil field yang diizinkan.
Teks siswa, rubrik asli, ID, pasangan skor individual dan widget tidak diembed.
Tanpa argumen ini, build dapat diulang dari snapshot publik yang disanitasi.
"""
import argparse
import ast
import hashlib
import json
import re
from html.parser import HTMLParser
from pathlib import Path

from build_learning_html import MODULES, DESCRIPTIONS, functions

DESCRIPTIONS.update({
    "sheet_records": "Membaca header dan nilai eksplisit worksheet; menolak formula dan header duplikat.",
    "sheet_records.text": "Mengubah nilai Excel menjadi teks tanpa mengisi nilai yang kosong.",
    "import_workbook": "Mengimpor tab guru secara eksplisit ke CSV turunan, menjaga workbook dan memisahkan ringkasan AI.",
    "metric_trace": "Menghitung langkah empat metrik dari pasangan yang cocok; pemanggil memverifikasi acuan/split.",
})

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / 'docs/learning_snapshot_AI.json'
TARGET = ROOT / 'docs/belajar_colab_AI_interaktif.html'
SOURCE_PATHS = [
    'src/aes/io.py', 'src/aes/workbook.py', 'src/aes/audit.py',
    'src/aes/cleaning.py', 'src/aes/windows.py', 'src/aes/encoders.py',
    'src/aes/model.py', 'src/aes/trainer.py', 'src/aes/ai_learning.py',
    'src/aes/metrics.py', 'src/aes/metric_trace.py', 'src/aes/checkpoint.py',
    'src/aes/inference.py', 'scripts/notebook_step.py',
    'scripts/setup_notebook.py', 'scripts/ac2_reference.R', 'configs/default.json',
]


class TableReader(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tables = []
        self.table = self.row = self.cell = None

    def handle_starttag(self, tag, attrs):
        if tag == 'table': self.table = []
        elif tag == 'tr': self.row = []
        elif tag in ['td', 'th']: self.cell = ''

    def handle_data(self, data):
        if self.cell is not None: self.cell += data

    def handle_endtag(self, tag):
        if tag in ['td', 'th'] and self.cell is not None:
            self.row.append(self.cell)
            self.cell = None
        elif tag == 'tr' and self.row is not None:
            self.table.append(self.row)
            self.row = None
        elif tag == 'table':
            self.tables.append(self.table)
            self.table = None


def stream(cell):
    return '\n'.join(''.join(o.get('text', [])) for o in cell.get('outputs', [])
                     if o['output_type'] == 'stream')


def tables(cell):
    result = []
    for output in cell.get('outputs', []):
        html = output.get('data', {}).get('text/html')
        if html:
            parser = TableReader()
            parser.feed(''.join(html))
            result.extend(parser.tables)
    return result


def following_dict(text, marker):
    start = text.index('{', text.index(marker))
    return json.JSONDecoder().raw_decode(text[start:])[0]


def extract_snapshot(path):
    expected = 'f58c5d0e05d1d109cca69a114d73ffb08d2b427a73b52ed704c1889f323ad55f'
    if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
        raise ValueError('Buku ini mengikat snapshot Colab yang diaudit; notebook berbeda perlu audit baru.')
    nb = json.loads(Path(path).read_text())
    public = json.loads((ROOT / 'notebooks/02_latihan_acuan_AI.ipynb').read_text())
    if len(nb['cells']) != len(public['cells']) or any(
        c['cell_type'] != p['cell_type'] or ''.join(c['source']) != ''.join(p['source'])
        for c, p in zip(nb['cells'], public['cells'])
    ):
        raise ValueError('Source notebook berbeda; tinjau sumber sebelum membuat snapshot.')
    metric_table = next(t for t in tables(nb['cells'][22]) if t[0][:2] == ['Skenario', 'Trait'])
    rows = []
    for row in metric_table[1:]:
        if row[0] not in ['S1', 'S2', 'S3'] or row[1] not in ['isi', 'organisasi', 'kebahasaan', 'mekanik']:
            raise ValueError('Skenario/trait snapshot berbeda.')
        rows.append({'scenario': row[0], 'trait': row[1],
                     **{name: float(row[i + 2]) for i, name in enumerate(['RMSE', 'Pearson', 'QWK', 'AC2'])}})
    training_table = next(t for t in tables(nb['cells'][22]) if 'Step LR aktif' in t[0])
    training = {r[0]: {'initial_mse': float(r[1]), 'final_mse': float(r[2]),
                      'epochs': int(r[3]), 'active_steps': int(r[4]),
                      'parameter_samples_changed': ast.literal_eval(r[5])}
                for r in training_table[1:]}
    final_stream = stream(nb['cells'][22])
    rmse = following_dict(final_stream, '2. RMSE:')
    pearson = following_dict(final_stream, '3. Pearson:')
    # Sufficient statistics only; no centered/raw per-essay arrays are exported.
    pearson = {k: v for k, v in pearson.items()
               if not k.startswith('centered_')}
    matrix_tables = [t for t in tables(nb['cells'][22]) if t[0][0] == 'Kategori acuan AI']
    observed = [[int(float(v)) for v in row[1:]] for row in matrix_tables[0][1:]]
    inspected = stream(nb['cells'][14])
    roberta_stats = ast.literal_eval(re.search(r'RoBERTa window: (.*)', inspected).group(1))
    bge_stats = ast.literal_eval(re.search(r'BGE tokenisasi: (.*)', inspected).group(1))
    vector_lines = re.findall(r'Contoh lima elemen vektor: (.*)', final_stream)
    vectors = ast.literal_eval(vector_lines[-1])
    # A few coordinates of representations, not full embeddings or individual grades.
    vectors = {k: v[:5] for k, v in vectors.items()
               if k in ['roberta', 'bge', 'projection', 'concat']}
    out_of_range = {}
    for scenario, trait, value in re.findall(r'(S[123]) (\w+) \| Proporsi prediksi di luar 1–5: ([\d.]+)', final_stream):
        out_of_range[scenario + ':' + trait] = float(value)
    setup = stream(nb['cells'][4])
    versions = {name: re.search(re.escape(name) + r': (\S+)', setup).group(1)
                for name in ['numpy', 'scipy', 'scikit-learn', 'torch', 'transformers', 'FlagEmbedding']}
    snapshot = {
        'saved_commit': '0b52413771ed535b1767a23924f0b980e0159530',
        'notebook_sha256': hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        'source_notebook': 'notebooks/02_latihan_acuan_AI.ipynb',
        'scope': 'AI_PROVISIONAL_NOT_TEACHER', 'code_cells': 11,
        'saved_error_outputs': sum(o['output_type'] == 'error' for c in nb['cells'] for o in c.get('outputs', [])),
        'rows': rows, 'training': training, 'out_of_range': out_of_range,
        'versions': versions, 'models': ast.literal_eval(re.search(r'Revision/config: (.*)', inspected).group(1)), 'python': re.search(r'Python AES: (\S+)', setup).group(1),
        'rows_read': 10, 'eligible_drafts': 8, 'excluded_incomplete': 2,
        'train_count': 4, 'validation_count': 4, 'teacher_labels_complete': 0,
        'rubric_confirmed': False, 'groups_train': 2, 'groups_validation': 2,
        'token_summary': {'maximum_bge_tokens': max(s['tokens_with_special'] for s in bge_stats.values()),
                          'bge_truncated': sum(s['truncated'] for s in bge_stats.values()),
                          'bge_tokens_removed': sum(s['tokens_removed'] for s in bge_stats.values()),
                          'roberta_windows_total_inspection': sum(s['windows'] for s in roberta_stats.values()),
                          'example_content_tokens': next(iter(roberta_stats.values()))['content_tokens'],
                          'example_window_lengths': [512, 134]},
        'vector_first_five': vectors,
        'trace': {'scenario': 'S3', 'trait': 'isi', 'RMSE': rmse, 'Pearson': pearson,
                  'observed_counts': observed, 'Pa': 0.96875, 'Pe': 0.3515625,
                  'observed_disagreement': 0.03125, 'expected_disagreement': 0.03125},
        'privacy': 'No essay text/IDs, individual scores, original rubric descriptors, workbook, cache or weights.',
    }
    if len(rows) != 12 or len(training) != 3:
        raise ValueError('Snapshot bukan seluruh tiga skenario.')
    return snapshot


def build(notebook=None):
    if notebook:
        snapshot = extract_snapshot(notebook)
        SNAPSHOT.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2) + '\n')
    else:
        snapshot = json.loads(SNAPSHOT.read_text())
    sources = []
    for path in SOURCE_PATHS:
        source = (ROOT / path).read_text()
        title, purpose = MODULES[path] if path in MODULES else (
            'Langkah perhitungan', 'Menelusuri empat metrik dari pasangan label dan prediksi.')
        # metric_trace was introduced after the old explorer; describe it explicitly.
        fns = functions(ast.parse(source)) if path.endswith('.py') else []
        sources.append({'path': path, 'title': title, 'purpose': purpose, 'source': source,
                        'sha256': hashlib.sha256(source.encode()).hexdigest(), 'functions': fns})
    nb = json.loads((ROOT / 'notebooks/02_latihan_acuan_AI.ipynb').read_text())
    cells = []
    heading = ''
    for cell in nb['cells']:
        if cell['cell_type'] == 'markdown': heading = ''.join(cell['source']).splitlines()[0].lstrip('# ')
        else: cells.append({'number': len(cells) + 1, 'title': heading, 'source': ''.join(cell['source'])})
    payload = {'snapshot': snapshot, 'sources': sources, 'cells': cells}
    encoded = json.dumps(payload, ensure_ascii=False).replace('<', '\\u003c')
    template = (ROOT / 'scripts/templates/colab_study.html').read_text()
    TARGET.write_text(template.replace('__STUDY_DATA__', encoded))
    print('Built', TARGET.relative_to(ROOT), len(sources), 'source files;', len(cells), 'code cells; 12 actual metric rows.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--notebook', type=Path, help='Notebook terisi privat untuk memperbarui snapshot agregat.')
    build(parser.parse_args().notebook)
