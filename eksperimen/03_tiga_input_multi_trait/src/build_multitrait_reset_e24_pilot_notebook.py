"""Build the standalone, development-only E24 contextual evidence pilot."""

import hashlib
import json

from build_multitrait_reset_e23_notebook import BASE, GROUP, ROOT, NAME as E23_NAME

NAME = 'AES_Reset_E24_Pilot_Bukti_Konteks_NLI.ipynb'
MARKER = '__E24_PILOT_CODE_SHA256__'

INTRO = """# E24 · Pilot bukti konteks soal 10

Uji apakah model NLI multibahasa bisa membedakan kejadian di sekolah, di rumah,
dan lokasi yang tidak jelas. Jalankan dari atas. T4 opsional, CPU juga bisa.

Model frozen membaca pasangan **teks → pernyataan tentang lokasi**. Ia memberi
tiga skor NLI untuk setiap pernyataan: implikasi, netral, dan kontradiksi.
Untuk tiap kriteria, notebook memperlihatkan kutipan dengan skor implikasi
tertinggi. Kutipan itu kandidat bukti yang perlu dibaca, bukan alasan otomatis.

Pilot memakai development E21 dan workbook semi-final yang sama. Skor guru
digunakan setelah inferensi untuk mengatur tabel audit, tidak masuk ke model.
ZIP audit mengandung kutipan jawaban. Simpan di Drive privat.

[Rujukan tugas per kriteria · TRATES](https://aclanthology.org/2025.findings-acl.1054/)
menjadi inspirasi eksploratif. Bukti kuartil venue belum terverifikasi.
[García-Varela dkk.](https://doi.org/10.3390/educsci15080946) memberi konteks
metodologi kriteria eksplisit. NLI E24 adalah adaptasi lokal. Pilot ini berhenti
setelah audit bukti, sebelum keputusan validasi dan training S3.
"""

SETUP_DATA = """cfg = {
    'dataset': str(DATASET_PATH), 'sheet': SHEET_NAME, 'expected_rows': 1340,
    'test_fraction': 0.20, 'folds': 5, 'split_seed': 2026, 'split_candidates': 128,
}
bundle = read_dataset(cfg)
manifest = make_split(bundle, cfg)
EXPECTED_SPLIT = 'f7fba02006ae61f96fa586fb6b87de678038b38679ac0629af101cb66e69c16c'
if manifest['split_sha256'] != EXPECTED_SPLIT:
    raise ValueError('Split pilot berbeda dari E21. Periksa dataset dan versi scikit-learn.')
indices = [i for i in manifest['development'] if bundle.rows[i]['response_number'].strip() == '10']
if len(indices) != 107:
    raise ValueError('Diharapkan 107 jawaban development untuk soal 10.')
print(f'{len(indices)} jawaban soal 10 dari {len(manifest["development"])} jawaban development.')
"""

SETTINGS = """MODEL_ID = 'MoritzLaurer/mDeBERTa-v3-base-mnli-xnli'
MODEL_REVISION = 'main'  # SHA efektif diputuskan sekali dan ditulis ke artefak.
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH = 8 if DEVICE == 'cuda' else 4
MAX_TOKENS = 384
CRITERIA = {
    'sekolah': 'Kejadian yang diceritakan berlangsung di sekolah.',
    'rumah': 'Kejadian yang diceritakan berlangsung di rumah.',
}
"""

INFERENCE = r'''def label_indices(config):
    names = {str(name).lower(): int(i) for i, name in config.id2label.items()}
    required = {'entailment', 'neutral', 'contradiction'}
    if set(names) != required or len(set(names.values())) != 3:
        raise ValueError(f'Label NLI tidak dikenali: {config.id2label}')
    return names


class EvidenceReader:
    def __init__(self, model, tokenizer, device, batch_size, max_tokens):
        self.model = model.eval().requires_grad_(False)
        self.tokenizer = tokenizer
        self.device = device
        self.batch_size = batch_size
        self.max_tokens = max_tokens
        self.labels = label_indices(model.config)
        # Sediakan ruang untuk hipotesis terpanjang dan tiga token pembatas.
        hypothesis_width = max(len(tokenizer(value, add_special_tokens=False)['input_ids'])
                               for value in CRITERIA.values())
        self.premise_cap = max_tokens - hypothesis_width - tokenizer.num_special_tokens_to_add(pair=True)
        if self.premise_cap <= 48:
            raise ValueError('Jendela premis terlalu sempit.')

    def passages(self, text):
        chunks = []
        for sentence in re.split(r'(?<=[.!?])\s+|\n+', text):
            sentence = sentence.strip()
            if not sentence:
                continue
            offsets = self.tokenizer(sentence, add_special_tokens=False, return_offsets_mapping=True)['offset_mapping']
            if len(offsets) <= self.premise_cap:
                chunks.append(sentence)
                continue
            for start in range(0, len(offsets), self.premise_cap - 32):
                window = offsets[start:start + self.premise_cap]
                chunks.append(sentence[window[0][0]:window[-1][1]])
                if start + self.premise_cap >= len(offsets):
                    break
        return chunks or [text]

    @torch.inference_mode()
    def probabilities(self, premises, hypotheses):
        scores = []
        for start in range(0, len(premises), self.batch_size):
            end = start + self.batch_size
            x = self.tokenizer(premises[start:end], hypotheses[start:end],
                               max_length=self.max_tokens, truncation='only_first',
                               padding=True, return_tensors='pt')
            logits = self.model(**x.to(self.device)).logits.float()
            scores.extend(logits.softmax(dim=-1).cpu().tolist())
        return scores

    def features(self, text):
        text = text.strip()
        if not text:
            raise ValueError('Jawaban kosong.')
        full_size = len(self.tokenizer(text, add_special_tokens=False)['input_ids'])
        full_available = full_size <= self.premise_cap
        windows = self.passages(text)
        views = [text, *windows] if full_available else windows
        pairs = [(view, hypothesis) for hypothesis in CRITERIA.values() for view in views]
        values = self.probabilities([pair[0] for pair in pairs], [pair[1] for pair in pairs])
        result = {'jawaban_utuh_dinilai': full_available, 'jumlah_kutipan': len(windows)}
        for i, key in enumerate(CRITERIA):
            block = values[i * len(views):(i + 1) * len(views)]
            for label, column in self.labels.items():
                result[f'{key}_utuh_{label}'] = block[0][column] if full_available else None
            snippet_values = block[int(full_available):]
            best = max(range(len(windows)), key=lambda index: snippet_values[index][self.labels['entailment']])
            result[f'{key}_kutipan_kandidat'] = windows[best]
            result[f'{key}_kutipan_entailment'] = snippet_values[best][self.labels['entailment']]
        return result
'''

LOAD = """from huggingface_hub import HfApi
from transformers import AutoModelForSequenceClassification

model_info = HfApi().model_info(MODEL_ID, revision=MODEL_REVISION)
revision = model_info.sha
if not revision or not re.fullmatch(r'[0-9a-f]{40}', revision):
    raise ValueError('SHA revisi pretrained tidak ditemukan.')
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=revision, use_fast=True)
model_files = {entry.rfilename for entry in model_info.siblings}
use_safetensors = any(name.endswith('.safetensors') for name in model_files)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_ID, revision=revision, use_safetensors=use_safetensors,
).to(DEVICE)
if not tokenizer.is_fast:
    raise RuntimeError('Pilot memerlukan tokenizer fast untuk memetakan kutipan ke teks asli.')
reader = EvidenceReader(model, tokenizer, DEVICE, BATCH, MAX_TOKENS)
print(f'Model frozen: {MODEL_ID} @ {revision} · {DEVICE}')
"""

PREFLIGHT = """# Contoh buatan untuk memeriksa arah fitur sebelum membuka kasus development.
probes = [
    ('Sekolah langsung', 'Para siswa kerja bakti membersihkan halaman sekolah.', 'sekolah'),
    ('Rumah langsung', 'Kami kerja bakti di rumah bersama keluarga.', 'rumah'),
    ('Sekolah tersirat', 'Seusai bel pelajaran, kami menyapu ruang kelas bersama wali kelas.', 'sekolah'),
    ('Rumah tersirat', 'Ayah dan ibu membersihkan ruang tamu bersama saya dan adik.', 'rumah'),
    ('Sekolah disebut sebagai pembanding', 'Di sekolah biasanya ada lomba. Kali ini kerja bakti kami berlangsung di rumah.', 'rumah'),
    ('Lokasi tak disebut', 'Kami menyelesaikan kegiatan pada pagi hari.', None),
]
probe_rows = []
for name, text, expected in probes:
    values = reader.features(text)
    school = values['sekolah_utuh_entailment']
    home = values['rumah_utuh_entailment']
    preferred = None if school == home else ('sekolah' if school > home else 'rumah')
    probe_rows.append({'kasus': name, 'teks_sintetis': text, 'arah_diharapkan': expected,
                       'arah_skor': preferred, 'arah_sesuai': None if expected is None else expected == preferred,
                       **values})
display(pd.DataFrame(probe_rows)[['kasus', 'arah_diharapkan', 'arah_skor', 'arah_sesuai',
                                 'sekolah_utuh_entailment', 'rumah_utuh_entailment']].round(4))
"""

RUN = """fold_of = {index: fold['fold'] for fold in manifest['folds'] for index in fold['validation']}
evidence_rows = []
for step, index in enumerate(indices, 1):
    row = bundle.rows[index]
    features = reader.features(row['text'])
    evidence_rows.append({'id_jawaban': row['row_id'], 'baris_excel': row['source_row'],
                          'fold': fold_of[index], 'label_isi': int(bundle.y[index, 0]), **features})
    if step % 20 == 0 or step == len(indices):
        print(f'Inferensi development: {step}/{len(indices)}', flush=True)
evidence = pd.DataFrame(evidence_rows)
score_columns = ['sekolah_utuh_entailment', 'rumah_utuh_entailment']
summary = evidence.groupby('label_isi').agg(n=('id_jawaban', 'size'), **{
    f'median_{column}': (column, 'median') for column in score_columns
}).reset_index()
display(summary.round(4))
print('Review prioritas · semua skor isi 1 dan kasus kerja bakti di rumah dari audit E21.')
display(evidence.loc[evidence['label_isi'].eq(1) | evidence['baris_excel'].eq(331),
                     ['baris_excel', 'label_isi', *score_columns,
                      'sekolah_kutipan_kandidat', 'rumah_kutipan_kandidat']].round(4))
"""

EXPORT = """from tempfile import TemporaryDirectory

path = OUTPUT_ROOT / 'Hasil_AES_Reset_E24_Pilot_Bukti_Konteks.zip'
path.parent.mkdir(parents=True, exist_ok=True)
with TemporaryDirectory() as temp:
    workbook = Path(temp) / '01_Audit_Bukti_Konteks.xlsx'
    metadata = {
        'stage': 'pilot_development', 'model': MODEL_ID, 'model_revision': revision,
        'device': DEVICE, 'torch': version('torch'), 'transformers': version('transformers'),
        'scikit_learn': version('scikit-learn'), 'batch': BATCH, 'max_tokens': MAX_TOKENS,
        'criteria': CRITERIA, 'notebook_code_sha256': NOTEBOOK_CODE_SHA256,
        'split_sha256': manifest['split_sha256'], 'source_sha256': bundle.source_sha256,
        'development_rows': len(evidence), 'test_inference': False, 's3_training': False,
    }
    review = evidence.copy()
    for field in ('lokasi_menurut_teks', 'kutipan_mendukung', 'catatan_review'):
        review[field] = ''
    with pd.ExcelWriter(workbook, engine='openpyxl') as writer:
        summary.to_excel(writer, sheet_name='Ringkasan', index=False)
        pd.DataFrame(probe_rows).to_excel(writer, sheet_name='Contoh Sintetis', index=False)
        review.to_excel(writer, sheet_name='Audit Development Privat', index=False)
        for tab in writer.book:
            tab.freeze_panes = 'B2'
            tab.auto_filter.ref = tab.dimensions
            for row in tab.iter_rows():
                for cell in row:
                    # Simpan kutipan sebagai teks walaupun diawali tanda sama dengan.
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


def build():
    base = BASE.parent / E23_NAME
    raw = base.read_bytes()
    previous = json.loads(raw)
    old_hash = previous['metadata']['aes_standalone']['code_sha256']
    inherited = {cell['metadata'].get('aes_multitrait_role'): ''.join(cell['source'])
                 for cell in previous['cells'] if cell['cell_type'] == 'code'}
    imports = inherited['imports'].replace(old_hash, MARKER)
    storage = inherited['storage'].replace('aes_reset_e23_s3_bobot_isi_2x', 'aes_reset_e24_pilot_nli')
    setup = inherited['dependency_setup'].replace("'openpyxl>=3.1,<4'",
                                                   "'openpyxl>=3.1,<4', 'sentencepiece>=0.2,<1', 'protobuf>=4,<7'")
    cells = []

    def markdown(source):
        cells.append({'cell_type': 'markdown', 'source': source, 'metadata': {}})

    def code(role, source):
        cells.append({'cell_type': 'code', 'source': source, 'outputs': [],
                      'execution_count': None, 'metadata': {'aes_multitrait_role': role}})

    markdown(INTRO)
    markdown('## 1 · Setup runtime')
    code('dependency_setup', setup)
    code('imports', imports)
    markdown('## 2 · Siapkan split development E21')
    for role in ('columns', 'dataset_definitions', 'definition:make_split'):
        code(role, inherited[role])
    code('storage', storage)
    code('prepare', SETUP_DATA)
    markdown('## 3 · Muat model NLI dan definisikan kriteria')
    code('pilot_settings', SETTINGS)
    code('pilot_inference', INFERENCE)
    code('pilot_load', LOAD)
    markdown('## 4 · Cek contoh buatan')
    code('pilot_synthetic', PREFLIGHT)
    markdown('## 5 · Baca bukti pada development soal 10')
    code('pilot_run', RUN)
    markdown('## 6 · Ekspor audit privat')
    code('pilot_export', EXPORT)

    fingerprint = hashlib.sha256('\n'.join(
        cell['source'] for cell in cells if cell['cell_type'] == 'code'
    ).encode()).hexdigest()
    metadata = previous['metadata']
    metadata.pop('accelerator', None)
    metadata['colab']['name'] = NAME
    metadata['aes_standalone'] = {
        'code_sha256': fingerprint, 'base': base.relative_to(ROOT).as_posix(),
        'base_sha256': hashlib.sha256(raw).hexdigest(), 'experiment': 'Reset_E24_Pilot_NLI',
    }
    for index, cell in enumerate(cells):
        cell['id'] = f'reset-e24-pilot-{index}'
        cell['source'] = cell['source'].replace(MARKER, fingerprint)
        if cell['cell_type'] == 'code':
            compile(cell['source'], cell['id'], 'exec')
    notebook = {**previous, 'cells': cells, 'metadata': metadata}
    content = json.dumps(notebook, ensure_ascii=False, indent=1) + '\n'
    for folder in (BASE.parent, GROUP / 'notebooks'):
        (folder / NAME).write_text(content, encoding='utf-8')


if __name__ == '__main__':
    build()
