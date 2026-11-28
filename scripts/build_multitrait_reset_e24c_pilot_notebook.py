"""Build E24C, a development-only extractive QA evidence audit."""

import hashlib
import json

from build_multitrait_reset_e24b_pilot_notebook import BASE, GROUP, ROOT, NAME as E24B_NAME
from build_multitrait_reset_e23_notebook import replace

NAME = 'AES_Reset_E24C_Pilot_Ekstraksi_Bukti_QA.ipynb'
MARKER = '__E24C_PILOT_CODE_SHA256__'
EXPERIMENT = 'Reset_E24C_Pilot_Ekstraksi_Bukti_QA'

INTRO = """# E24C · Ambil bukti siapa, apa, kapan, di mana

Pilot satu kali pada 107 jawaban development soal 10, dengan split E21.
Model mDeBERTa question answering frozen memilih potongan literal dari jawaban
atau keluaran kosong jika model tidak menemukan jawaban. CPU bisa, T4 disarankan.

Notebook menampilkan delapan probe buatan, lalu ringkasan seluruh development.
Tab **Review Buta Privat** memuat 15 jawaban terpilih dengan seed tetap, tiga
per fold, beserta bukti kandidat. Cek apakah tiap potongan menjawab pertanyaan
dan merujuk kejadian utama. Label guru tersedia di tab evaluasi terpisah.

**Keputusan sebelum run.** Jangan gunakan fitur untuk CV jika probe pembeda
lokasi utama atau probe tanpa jawaban gagal. Bahkan bila probe lolos, baca
sampel buta untuk memastikan bukti benar. Tab ringkasan membandingkan pola
keluaran kosong dengan angka audit E24/E24B. Evaluasi ini belum membuktikan
pengaruh ekstraksi terhadap empat metrik prediksi S3.

Ide per kriteria berangkat dari [TRATES](https://aclanthology.org/2025.findings-acl.1054/)
dan metodologi eksplisit [García-Varela dkk.](https://doi.org/10.3390/educsci15080946).
Kuartil TRATES belum terverifikasi. Question answering ekstraktif ini adalah
adaptasi. ZIP berisi teks siswa dan tetap di penyimpanan privat.
"""

SETTINGS = """MODEL_ID = 'timpal0l/mdeberta-v3-base-squad2'
MODEL_REVISION = 'main'  # SHA efektif diambil sekali dan dicatat di artefak.
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH = 8 if DEVICE == 'cuda' else 4
MAX_TOKENS = 384
DOC_STRIDE = 128
MAX_ANSWER_TOKENS = 32
REVIEW_SEED = 2026
REVIEW_PER_FOLD = 3

QUESTIONS = {
    'pelaku': 'Siapa yang terlibat dalam kejadian utama?',
    'peristiwa': 'Apa kegiatan atau kejadian utama yang diceritakan?',
    'waktu': 'Kapan kejadian utama berlangsung?',
    'lokasi': 'Di mana kejadian utama berlangsung?',
}
"""

INFERENCE = '''class SpanReader:
    def __init__(self, qa):
        self.qa = qa

    def extract(self, text, keys=QUESTIONS):
        if not text.strip():
            raise ValueError('Jawaban kosong.')
        keys = tuple(keys)
        inputs = [{'question': QUESTIONS[key], 'context': text} for key in keys]
        answers = self.qa(
            inputs, top_k=1, handle_impossible_answer=True, max_seq_len=MAX_TOKENS,
            doc_stride=DOC_STRIDE, max_answer_len=MAX_ANSWER_TOKENS, batch_size=BATCH,
        )
        if isinstance(answers, dict):
            answers = [answers]
        if len(answers) != len(keys):
            raise ValueError('Jumlah respons QA tidak sesuai jumlah pertanyaan.')
        result = {}
        for key, answer in zip(keys, answers):
            span = answer['answer']
            start, end = int(answer['start']), int(answer['end'])
            if span and (not 0 <= start < end <= len(text) or text[start:end] != span):
                raise ValueError('Kutipan model tidak cocok dengan posisi pada teks asli.')
            result.update({f'{key}_bukti': span,
                           f'{key}_awal': start if span else None,
                           f'{key}_akhir_eksklusif': end if span else None,
                           f'{key}_score_qa': float(answer['score']),
                           f'{key}_kosong': not bool(span)})
        return result
'''

LOAD = """from huggingface_hub import HfApi
from transformers import AutoModelForQuestionAnswering, pipeline

model_info = HfApi().model_info(MODEL_ID, revision=MODEL_REVISION)
revision = model_info.sha
if not revision or not re.fullmatch(r'[0-9a-f]{40}', revision):
    raise ValueError('SHA revisi pretrained tidak ditemukan.')
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=revision, use_fast=True)
if not tokenizer.is_fast:
    raise RuntimeError('Pilot memerlukan tokenizer fast untuk offset kutipan.')
model_files = {entry.rfilename for entry in model_info.siblings}
use_safetensors = any(name.endswith('.safetensors') for name in model_files)
model = AutoModelForQuestionAnswering.from_pretrained(
    MODEL_ID, revision=revision, use_safetensors=use_safetensors,
).eval().requires_grad_(False)
qa = pipeline('question-answering', model=model, tokenizer=tokenizer,
              device=0 if DEVICE == 'cuda' else -1)
reader = SpanReader(qa)
print(f'Model QA frozen: {MODEL_ID} @ {revision} · {DEVICE}')
"""

PREFLIGHT = """# Jawaban yang tepat boleh memuat beberapa kata tambahan di sekitar bukti.
base = 'Jumat pagi, para siswa menanam pohon di halaman sekolah.'
probes = [
    ('Pelaku jelas', 'pelaku', base, 'siswa', False),
    ('Kejadian jelas', 'peristiwa', base, 'menanam pohon', False),
    ('Waktu jelas', 'waktu', base, 'jumat', False),
    ('Lokasi jelas', 'lokasi', base, 'sekolah', True),
    ('Lokasi pembanding', 'lokasi',
     'Kami biasanya kerja bakti di sekolah. Jumat ini kami membersihkan rumah bersama keluarga.',
     'rumah', True),
    ('Lokasi tidak disebut', 'lokasi', 'Kami menanam pohon bersama pada Jumat pagi.', None, True),
    ('Waktu tidak disebut', 'waktu', 'Para siswa menanam pohon di halaman sekolah.', None, True),
    ('Lokasi jauh di teks panjang', 'lokasi',
     ('Persiapan telah dibahas bersama. ' * 120) + base, 'sekolah', False),
]
probe_rows = []
for name, key, text, expected, critical in probes:
    answer = reader.extract(text, (key,))[f'{key}_bukti']
    passed = not answer if expected is None else expected in answer.casefold()
    if name == 'Lokasi pembanding':
        passed = passed and 'sekolah' not in answer.casefold()
    probe_rows.append({'kasus': name, 'unsur': key, 'kritis': critical, 'bukti_kandidat': answer,
                       'bukti_diharapkan': expected, 'lolos': passed})
critical_pass = all(row['lolos'] for row in probe_rows if row['kritis'])
display(pd.DataFrame(probe_rows))
print('Probe kritis:', 'arah lanjut audit' if critical_pass else 'tahan integrasi S3')
"""

RUN = """fold_of = {index: fold['fold'] for fold in manifest['folds'] for index in fold['validation']}
# Sampel dipilih sebelum label atau keluaran QA dibaca.
rng = np.random.default_rng(REVIEW_SEED)
review_indices = []
for fold in range(1, cfg['folds'] + 1):
    members = [i for i in indices if fold_of[i] == fold]
    review_indices.extend(rng.choice(members, size=min(REVIEW_PER_FOLD, len(members)),
                                     replace=False).tolist())
rng.shuffle(review_indices)
review_codes = {index: f'Audit_{position + 1:02d}' for position, index in enumerate(review_indices)}

evidence_rows, blind_map = [], {}
for step, index in enumerate(indices, 1):
    row = bundle.rows[index]
    outputs = reader.extract(row['text'])
    evidence_rows.append({'id_jawaban': row['row_id'], 'baris_excel': row['source_row'],
                          'fold': fold_of[index], 'review_id': review_codes.get(index),
                          'label_isi': int(bundle.y[index, 0]), **outputs})
    if index in review_codes:
        blind_map[index] = {'review_id': review_codes[index], 'jawaban_asli': row['text'],
                            **{f'{key}_bukti': outputs[f'{key}_bukti'] for key in QUESTIONS}}
    if step % 20 == 0 or step == len(indices):
        print(f'Inferensi development: {step}/{len(indices)}', flush=True)

blind_review = pd.DataFrame([blind_map[i] for i in review_indices])
for key in QUESTIONS:
    blind_review[f'{key}_audit'] = ''
blind_review['catatan'] = ''
evidence = pd.DataFrame(evidence_rows)
null_columns = [f'{key}_kosong' for key in QUESTIONS]
evidence['ada_unsur_kosong'] = evidence[null_columns].any(axis=1)
summary = evidence.groupby('label_isi').agg(
    n=('id_jawaban', 'size'), ada_unsur_kosong=('ada_unsur_kosong', 'sum'),
    **{col: (col, 'sum') for col in null_columns},
).reset_index()
# Hitungan pilot lama memakai 106 jawaban utuh. QA memakai semua 107.
controls = pd.DataFrame([
    {'pilot': 'E24 lokasi', 'n': 106, 'label_isi_1_ditandai': 2, 'label_isi_1_total': 8,
     'label_isi_2_4_ditandai': 2, 'label_isi_2_4_total': 98, 'aturan': 'Skor dukungan lokasi < 0,50'},
    {'pilot': 'E24B empat unsur', 'n': 106, 'label_isi_1_ditandai': 8, 'label_isi_1_total': 8,
     'label_isi_2_4_ditandai': 97, 'label_isi_2_4_total': 98, 'aturan': 'Ada skor dukungan unsur < 0,50'},
])
low = evidence['label_isi'].eq(1)
flag = evidence['ada_unsur_kosong']
controls.loc[len(controls)] = {
    'pilot': 'E24C ekstraksi QA', 'n': len(evidence), 'label_isi_1_ditandai': int((low & flag).sum()),
    'label_isi_1_total': int(low.sum()), 'label_isi_2_4_ditandai': int((~low & flag).sum()),
    'label_isi_2_4_total': int((~low).sum()), 'aturan': 'Model mengembalikan minimal satu bukti kosong',
}
display(summary)
display(controls)
print('Review buta: gunakan tab privat. Nilai bukti tiap unsur: tepat, salah konteks, atau semestinya kosong.')
"""

EXPORT = """from tempfile import TemporaryDirectory

path = OUTPUT_ROOT / 'Hasil_AES_Reset_E24C_Pilot_Ekstraksi_Bukti.zip'
path.parent.mkdir(parents=True, exist_ok=True)
with TemporaryDirectory() as temp:
    workbook = Path(temp) / '01_Audit_Ekstraksi_Bukti.xlsx'
    metadata = {
        'stage': 'pilot_development', 'model': MODEL_ID, 'model_revision': revision,
        'device': DEVICE, 'torch': version('torch'), 'transformers': version('transformers'),
        'scikit_learn': version('scikit-learn'), 'batch': BATCH, 'max_tokens': MAX_TOKENS,
        'max_answer_tokens': MAX_ANSWER_TOKENS, 'doc_stride': DOC_STRIDE,
        'questions': QUESTIONS, 'review_seed': REVIEW_SEED, 'review_rows': len(blind_review),
        'probe_kritis_lolos': critical_pass, 'notebook_code_sha256': NOTEBOOK_CODE_SHA256,
        'split_sha256': manifest['split_sha256'], 'source_sha256': bundle.source_sha256,
        'development_rows': len(evidence), 'test_inference': False, 's3_training': False,
    }
    with pd.ExcelWriter(workbook, engine='openpyxl') as writer:
        summary.to_excel(writer, sheet_name='Ringkasan', index=False)
        controls.to_excel(writer, sheet_name='Pembanding Historis', index=False)
        pd.DataFrame(probe_rows).to_excel(writer, sheet_name='Contoh Sintetis', index=False)
        blind_review.to_excel(writer, sheet_name='Review Buta Privat', index=False)
        evidence.to_excel(writer, sheet_name='Audit Development Privat', index=False)
        for tab in writer.book:
            tab.freeze_panes = 'B2'
            tab.auto_filter.ref = tab.dimensions
            for row in tab.iter_rows():
                for cell in row:
                    if cell.data_type == 'f':
                        cell.data_type = 's'
    metadata['workbook_sha256'] = digest(workbook)
    pending = path.with_suffix('.pending')
    with zipfile.ZipFile(pending, 'w', zipfile.ZIP_DEFLATED) as archive:
        archive.write(workbook, workbook.name)
        archive.writestr('manifest.json', json.dumps(metadata, ensure_ascii=False, indent=2))
    pending.replace(path)

print('ZIP audit privat:', path)
if IN_COLAB:
    from google.colab import files
    files.download(str(path))
"""

CELLS = {
    'pilot_settings': SETTINGS, 'pilot_inference': INFERENCE, 'pilot_load': LOAD,
    'pilot_synthetic': PREFLIGHT, 'pilot_run': RUN, 'pilot_export': EXPORT,
}


def build():
    base = BASE.parent / E24B_NAME
    raw = base.read_bytes()
    notebook = json.loads(raw)
    old_hash = notebook['metadata']['aes_standalone']['code_sha256']
    for index, cell in enumerate(notebook['cells']):
        source = ''.join(cell['source']).replace(old_hash, MARKER)
        role = cell['metadata'].get('aes_multitrait_role')
        if index == 0:
            source = INTRO
        elif role == 'storage':
            source = replace(source, {'aes_reset_e24b_pilot_nli': 'aes_reset_e24c_pilot_qa'})
        elif role in CELLS:
            source = CELLS[role]
        elif cell['cell_type'] == 'markdown':
            source = source.replace('Muat model NLI dan definisikan kriteria',
                                    'Muat model QA dan definisikan pertanyaan bukti')
        cell['source'] = source
        cell['id'] = f'reset-e24c-pilot-{index}'

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
