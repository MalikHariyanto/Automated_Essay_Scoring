"""Derive the E24B multi-criterion pilot from the verified E24 NLI notebook."""

import hashlib
import json

from build_multitrait_reset_e24_pilot_notebook import BASE, GROUP, ROOT, NAME as E24_NAME
from build_multitrait_reset_e23_notebook import replace

NAME = 'AES_Reset_E24B_Pilot_Empat_Unsur_NLI.ipynb'
MARKER = '__E24B_PILOT_CODE_SHA256__'
EXPERIMENT = 'Reset_E24B_Pilot_Empat_Unsur_NLI'

INTRO = """# E24B · Pilot bukti untuk empat unsur soal 10

Bandingkan dukungan **lokasi kejadian utama, peristiwa, pelaku, dan waktu**
pada 107 jawaban development soal 10. Pakai workbook semi-final E21 dan
model NLI frozen dengan revisi yang sama seperti E24. CPU bisa dipakai.

**Sekolah E24** dijalankan kembali pada jawaban yang sama sebagai kontrol
lokasi saja. **Rumah utama** membantu membaca konteks yang meleset. Enam
hipotesis memberi skor implikasi, netral, kontradiksi, dan kutipan kandidat.
Satu kutipan terpisah bisa memberi sinyal yang berbeda dari jawaban utuh.

Ambang 0,50 hanya untuk menghitung pola bukti lemah di pilot. Label isi
dipakai sesudah inferensi untuk ringkasan. Tahap ini belum mengubah skor,
memilih checkpoint baru, atau training S3.

Gagasan cek per kriteria mengikuti [TRATES](https://aclanthology.org/2025.findings-acl.1054/)
dan metodologi eksplisit [García-Varela dkk.](https://doi.org/10.3390/educsci15080946).
Bukti kuartil TRATES belum terverifikasi. Hipotesis NLI adalah adaptasi lokal.
ZIP berisi kutipan jawaban siswa dan disimpan privat.
"""

SETTINGS = """MODEL_ID = 'MoritzLaurer/mDeBERTa-v3-base-mnli-xnli'
MODEL_REVISION = '8adb042d524ecd5c26d3e3ba0e3fbcf7e2d0864c'  # Revisi aktual E24.
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH = 8 if DEVICE == 'cuda' else 4
MAX_TOKENS = 384
AUDIT_THRESHOLD = 0.50
REQUIRED_CRITERIA = ('lokasi_utama', 'peristiwa', 'pelaku', 'waktu')
CRITERIA = {
    'sekolah_e24': 'Kejadian yang diceritakan berlangsung di sekolah.',
    'lokasi_utama': 'Peristiwa utama dalam jawaban ini berlangsung di sekolah.',
    'rumah_utama': 'Peristiwa utama dalam jawaban ini berlangsung di rumah.',
    'peristiwa': 'Jawaban ini menyebutkan tindakan atau peristiwa konkret yang terjadi.',
    'pelaku': 'Jawaban ini menyebutkan siapa yang terlibat dalam peristiwa utama.',
    'waktu': 'Jawaban ini menyebutkan kapan peristiwa utama berlangsung.',
}
"""

PREFLIGHT = """# Setiap pasangan mengubah satu petunjuk. Model yang memberi arah seri dihitung gagal.
pairs = [
    ('Lokasi sekolah versus rumah', 'lokasi_utama',
     'Pada Jumat pagi, siswa membersihkan halaman sekolah.',
     'Pada Jumat pagi, siswa membersihkan halaman rumah.'),
    ('Lokasi utama versus lokasi pembanding', 'lokasi_utama',
     'Di rumah biasanya kami membersihkan halaman. Jumat ini kami kerja bakti di sekolah.',
     'Di sekolah biasanya kami membersihkan halaman. Jumat ini kami kerja bakti di rumah.'),
    ('Peristiwa disebut versus samar', 'peristiwa',
     'Pada Jumat pagi, siswa menanam pohon di halaman sekolah.',
     'Pada Jumat pagi, siswa berada di halaman sekolah dan ada sesuatu.'),
    ('Pelaku disebut versus tidak disebut', 'pelaku',
     'Pada Jumat pagi, para siswa menanam pohon di halaman sekolah.',
     'Pada Jumat pagi, penanaman pohon berlangsung di halaman sekolah.'),
    ('Waktu disebut versus tidak disebut', 'waktu',
     'Pada Jumat pagi, para siswa menanam pohon di halaman sekolah.',
     'Para siswa menanam pohon di halaman sekolah.'),
]
probe_rows = []
for name, key, with_evidence, comparison in pairs:
    first = reader.features(with_evidence)[f'{key}_utuh_entailment']
    second = reader.features(comparison)[f'{key}_utuh_entailment']
    probe_rows.append({'kasus': name, 'kriteria': key, 'teks_dengan_bukti': with_evidence,
                       'teks_pembanding': comparison, 'skor_dengan_bukti': first,
                       'skor_pembanding': second, 'selisih': first - second,
                       'arah_sesuai': first > second})
display(pd.DataFrame(probe_rows)[['kasus', 'skor_dengan_bukti', 'skor_pembanding',
                                 'selisih', 'arah_sesuai']].round(4))
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
score_columns = [f'{key}_utuh_entailment' for key in REQUIRED_CRITERIA]
baseline_column = 'sekolah_e24_utuh_entailment'
whole = evidence['jawaban_utuh_dinilai']
full = evidence.loc[whole].copy()
if full.empty:
    raise ValueError('Tidak ada jawaban utuh untuk perbandingan. Cek batas token pilot.')
full['lokasi_e24_lemah'] = full[baseline_column].lt(AUDIT_THRESHOLD)
full['ada_unsur_lemah'] = full[score_columns].lt(AUDIT_THRESHOLD).any(axis=1)
full['tambahan_di_luar_lokasi_e24'] = full['ada_unsur_lemah'] & ~full['lokasi_e24_lemah']
full['dua_lokasi_didukung'] = (full['lokasi_utama_utuh_entailment'].ge(0.90)
                                & full['rumah_utama_utuh_entailment'].ge(0.90))
for key in REQUIRED_CRITERIA:
    full[f'{key}_lemah'] = full[f'{key}_utuh_entailment'].lt(AUDIT_THRESHOLD)
    full[f'{key}_kutipan_kuat_utuh_lemah'] = (
        full[f'{key}_kutipan_entailment'].ge(0.90) & full[f'{key}_lemah']
    )

counts = ['lokasi_e24_lemah', 'ada_unsur_lemah', 'tambahan_di_luar_lokasi_e24', 'dua_lokasi_didukung',
          *(f'{key}_lemah' for key in REQUIRED_CRITERIA),
          *(f'{key}_kutipan_kuat_utuh_lemah' for key in REQUIRED_CRITERIA)]
for column in counts:
    # Jawaban panjang tidak punya skor utuh; nilainya tetap kosong pada workbook.
    evidence[column] = full[column].reindex(evidence.index)
summary = full.groupby('label_isi').agg(
    n_utuh=('id_jawaban', 'size'),
    **{col: (col, 'sum') for col in counts},
    **{f'median_{col}': (col, 'median') for col in [baseline_column, *score_columns]},
).reset_index()
summary['n_semua'] = summary['label_isi'].map(evidence.groupby('label_isi').size())
print(f'{len(full)}/{len(evidence)} jawaban memiliki skor utuh. Hitungan memakai jawaban yang sama di kedua kondisi.')
display(summary[['label_isi', 'n_semua', 'n_utuh', 'lokasi_e24_lemah', 'ada_unsur_lemah',
                 'tambahan_di_luar_lokasi_e24', 'dua_lokasi_didukung']])
print('Audit prioritas · skor isi 1, kasus kerja bakti di rumah, dan konflik kutipan.')
conflict = evidence[[f'{key}_kutipan_kuat_utuh_lemah' for key in REQUIRED_CRITERIA]].eq(True).any(axis=1)
priority = evidence['label_isi'].eq(1) | evidence['baris_excel'].eq(331) | conflict
display(evidence.loc[priority, ['baris_excel', 'label_isi', *score_columns,
                               *(f'{key}_kutipan_kandidat' for key in REQUIRED_CRITERIA)]].round(4))
"""


def build():
    base = BASE.parent / E24_NAME
    raw = base.read_bytes()
    notebook = json.loads(raw)
    old_hash = notebook['metadata']['aes_standalone']['code_sha256']
    for index, cell in enumerate(notebook['cells']):
        source = ''.join(cell['source']).replace(old_hash, MARKER)
        role = cell['metadata'].get('aes_multitrait_role')
        if index == 0:
            source = INTRO
        elif role == 'storage':
            source = replace(source, {'aes_reset_e24_pilot_nli': 'aes_reset_e24b_pilot_nli'})
        elif role == 'pilot_settings':
            source = SETTINGS
        elif role == 'pilot_synthetic':
            source = PREFLIGHT
        elif role == 'pilot_run':
            source = RUN
        elif role == 'pilot_export':
            source = replace(source, {
                'Hasil_AES_Reset_E24_Pilot_Bukti_Konteks.zip': 'Hasil_AES_Reset_E24B_Pilot_Empat_Unsur.zip',
                '01_Audit_Bukti_Konteks.xlsx': '01_Audit_Empat_Unsur.xlsx',
                "'criteria': CRITERIA,": "'criteria': CRITERIA, 'audit_threshold': AUDIT_THRESHOLD,",
                "'lokasi_menurut_teks', 'kutipan_mendukung', 'catatan_review'":
                    "'review_lokasi', 'review_peristiwa', 'review_pelaku', 'review_waktu', 'catatan_review'",
            })
        elif cell['cell_type'] == 'markdown':
            source = source.replace('Cek contoh buatan', 'Cek pasangan contoh buatan')
        cell['source'] = source
        cell['id'] = f'reset-e24b-pilot-{index}'

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
