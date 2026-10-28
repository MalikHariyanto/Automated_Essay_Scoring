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
SNAPSHOT = ROOT / 'docs/snapshots/learning_snapshot_AI.json'
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
    # Notebook terbaru yang diunggah: source 11 cell cocok dengan repo; output diaudit.
    expected = '473021666e2e38d21ac8911c816a30b23eb99dad81dc77896d891bee0d3da068'
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
    inspection = stream(nb['cells'][16])
    inspection_shapes = {scenario: ast.literal_eval(value)
                         for scenario, value in re.findall(r'(S[123]) \| SHAPES AKTUAL: (.*)', inspection)}
    trained_shapes = {scenario: ast.literal_eval(value)
                      for scenario, value in re.findall(r'(S[123]) FORWARD SETELAH LATIHAN AI: (.*)', final_stream)}
    # Shapes and aggregate matrices only: no per-essay predictions or IDs.
    for shapes in [inspection_shapes, trained_shapes]:
        if set(shapes) != {'S1', 'S2', 'S3'} or any(
            not isinstance(v, list) or not all(isinstance(n, int) for n in v)
            for values in shapes.values() for v in values.values()
        ):
            raise ValueError('Shape encoder harus tiga skenario dengan dimensi integer.')
    reference_table = next(t for t in tables(nb['cells'][18]) if t[0][0] == 'Fixture unit test')
    reference_rows = [{'fixture': int(r[0]), 'python': float(r[1]),
                       'R_irrCAC': float(r[2]), 'absolute_error': float(r[3])}
                      for r in reference_table[1:]]
    status = following_dict(stream(nb['cells'][20]), '')
    status = {key: status[key] for key in ['mode', 'status', 'ai_training_run',
                                         'teacher_training_run', 'reused_completed_run']}
    numeric_matrix = lambda table: [[float(v) for v in row[1:]] for row in table[1:]]
    pooled = ast.literal_eval(re.search(r'Proporsi kategori gabungan: (.*)', final_stream).group(1))
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
        'inspection_shapes': inspection_shapes, 'trained_shapes': trained_shapes,
        'ac2_fixture_rows': reference_rows, 'training_status': status,
        'cleaning_changed': re.search(r'Berubah: (True|False)', stream(nb['cells'][12])).group(1) == 'True',
        'public_cell_output_counts': [len(c.get('outputs', [])) for c in nb['cells'] if c['cell_type'] == 'code'],
        'trace': {'scenario': 'S3', 'trait': 'isi', 'RMSE': rmse, 'Pearson': pearson,
                  'observed_counts': observed, 'Pa': 0.96875, 'Pe': 0.3515625,
                  'observed_disagreement': 0.03125, 'expected_disagreement': 0.03125,
                  'expected_counts': numeric_matrix(matrix_tables[1]),
                  'disagreement_weights': numeric_matrix(matrix_tables[2]),
                  'agreement_weights': numeric_matrix(matrix_tables[3]),
                  'pooled_category_proportions': pooled},
        'privacy': 'No essay text/IDs, individual scores, original rubric descriptors, workbook, cache or weights.',
    }
    if len(rows) != 12 or len(training) != 3:
        raise ValueError('Snapshot bukan seluruh tiga skenario.')
    return snapshot


def build(notebook=None, private_output=None):
    if private_output and not notebook:
        raise ValueError('--private-output membutuhkan --notebook yang diaudit.')
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
    lessons = json.loads((ROOT / 'scripts/colab_code_lessons.json').read_text())
    by_path = {item['path']: item for item in sources}
    for lesson in lessons:
        lines = by_path[lesson['path']]['source'].splitlines()
        matches = [i for i, line in enumerate(lines) if lesson['anchor'] in line]
        if len(matches) != 1:
            raise ValueError('Potongan kode harus cocok satu baris source: ' + lesson['title'])
        start = matches[0]
        lesson['line'] = start + 1
        lesson['code'] = '\n'.join(lines[start:start + lesson['count']])
        context_start = max(0, start - 2)
        lesson['context_line'] = context_start + 1
        lesson['context_code'] = '\n'.join(lines[context_start:start + lesson['count'] + 3])
    cell_notes = json.loads((ROOT / 'scripts/colab_cell_notes.json').read_text())
    for note in cell_notes:
        lines = cells[note['cell'] - 1]['source'].splitlines()
        matches = [i for i, line in enumerate(lines) if note['anchor'] in line]
        if len(matches) != 1:
            raise ValueError('Catatan cell harus cocok satu baris: ' + note['title'])
        note['line'] = matches[0] + 1
    payload = {'snapshot': snapshot, 'sources': sources, 'cells': cells,
               'lessons': lessons, 'cell_notes': cell_notes}
    encoded = json.dumps(payload, ensure_ascii=False).replace('<', '\\u003c')
    template = (ROOT / 'scripts/templates/colab_study.html').read_text()
    template = template.replace('__OUTPUT_STUDY_JS__',
                                (ROOT / 'scripts/templates/colab_outputs.js').read_text())
    public_html = template.replace('__STUDY_DATA__', encoded).replace('__PRIVATE_NOTEBOOK__', 'null')
    TARGET.write_text(public_html)
    if private_output:
        destination = Path(private_output).resolve()
        private_root = (ROOT / 'outputs').resolve()
        if not destination.is_relative_to(private_root):
            raise ValueError('HTML pribadi harus di outputs/ yang diabaikan Git.')
        destination.parent.mkdir(parents=True, exist_ok=True)
        # Keep only source and passive output representations, never widget state
        # or executable notebook metadata. HTML is converted to safe text in JS.
        nb = json.loads(Path(notebook).read_text())
        private = {'cells': []}
        for cell in nb['cells']:
            if cell['cell_type'] != 'code':
                continue
            outputs = []
            for output in cell.get('outputs', []):
                safe = {k: output[k] for k in ['output_type', 'text', 'ename', 'evalue'] if k in output}
                if 'data' in output:
                    safe['data'] = {k: v for k, v in output['data'].items()
                                    if k in ['text/plain', 'text/html']}
                outputs.append(safe)
            private['cells'].append({'cell_type': 'code', 'source': cell['source'],
                                     'execution_count': cell.get('execution_count'), 'outputs': outputs})
        private_json = json.dumps(private, ensure_ascii=False).replace('<', '\\u003c')
        destination.write_text(template.replace('__STUDY_DATA__', encoded)
                               .replace('__PRIVATE_NOTEBOOK__', private_json))
        print('Built private HTML in ignored outputs/; notebook outputs are embedded locally.')
    print('Built', TARGET.relative_to(ROOT), len(sources), 'source files;', len(cells), 'code cells; 12 actual metric rows.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--notebook', type=Path, help='Notebook terisi privat untuk memperbarui snapshot agregat.')
    parser.add_argument('--private-output', type=Path, help='HTML pribadi di outputs/, berisi output notebook.')
    args = parser.parse_args()
    build(args.notebook, args.private_output)
