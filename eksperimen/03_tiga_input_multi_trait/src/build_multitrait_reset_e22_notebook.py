"""Build E22, the development-only S3 task-evidence pilot."""

import hashlib
import json

from build_multitrait_reset_e6_notebook import BASE, GROUP, ROOT, replace
from build_multitrait_reset_e20_seed43_notebook import NAME as E20_NAME
from build_multitrait_reset_e21_notebook import CAPTURE

NAME = 'AES_Reset_E22_S3_Bukti_Konteks_Isi.ipynb'
EXPERIMENT = 'Reset_E22_S3_Bukti_Konteks_Isi'
MARKER = '__RESET_E22_CODE_SHA256__'

INTRO = """# E22 · Bantuan bukti konteks untuk trait isi

Pilot ini memakai resep S3 E21: seed dasar 43, lima fold, bobot organisasi 1,2×,
serta checkpoint utama B dan margin validasi 0,01. Bobot loss isi tetap seperti E21.

Untuk soal 10, cek lima petunjuk dari teks jawaban: konteks sekolah, pelaku,
kejadian, waktu, dan konteks domestik tanpa petunjuk sekolah. Proyeksi linear
berbobot awal nol menambahkan kelima petunjuk hanya pada prediksi isi. Soal lain
menerima vektor nol. Petunjuk leksikal bisa meleset, misalnya jika konteks tersirat.
Pilot memakai pertanyaan yang sama dan tidak menguji transfer ke soal baru.

Gagasan kriteria eksplisit berangkat dari studi García-Varela dkk. (2025),
[Education Sciences](https://doi.org/10.3390/educsci15080946).
Fitur regex dan proyeksi ke S3 adalah adaptasi eksperimen ini.

Gunakan workbook semi-final yang sama dan pilih T4, lalu klik **Run all**.
Progres muncul per fold dan fold selesai tersimpan. Di akhir, unduh ZIP 04–05
dengan prediksi OOF, deteksi isi skor 1, dan riwayat fit. Tahap ini hanya development.
"""

CUES = r'''
EVIDENCE_NAMES = ('lokasi_sekolah', 'pelaku', 'kejadian', 'waktu', 'rumah_tanpa_sekolah')
SCHOOL = re.compile(r'\b(?:sekolah\w*|sman?|smpn?|smkn?|madrasah|kelas\w*|kampus)\b')
ACTORS = re.compile(r'\b(?:siswa\w*|siswi\w*|murid\w*|guru\w*|pelajar\w*|osis|peserta\w*|teman\w*|warga|saya|kami|mereka|keluarga\w*|ayah|ibu|adik)\b')
EVENT = re.compile(r'\b(?:lomba\w*|perlombaan\w*|kerja\s+bakti|gotong\s+royong|upacara|pertandingan\w*|kegiatan\w*|acara\w*|kecelakaan\w*|bertengkar|pentas|latihan)\b')
TIME = re.compile(r'\b(?:hari|senin|selasa|rabu|kamis|jum[’\']?at|sabtu|minggu|pukul|jam|kemarin|tadi|pagi|siang|sore|malam|istirahat|tanggal|bulan)\b')
HOME = re.compile(r'\b(?:di\s+rumah\w*|rumah\s+kami|keluarga\s+(?:saya|kami)|ayah|ibu|adik)\b')


def task_evidence(question_number, answer):
    if str(question_number).strip() != '10':
        return (0, 0, 0, 0, 0)
    text = answer.casefold()
    school = bool(SCHOOL.search(text))
    return (int(school), int(bool(ACTORS.search(text))), int(bool(EVENT.search(text))),
            int(bool(TIME.search(text))), int(not school and bool(HOME.search(text))))
'''

TRAIN_VALIDATION = CAPTURE.split('\ndef fit_and_predict_test', 1)[0] + '''
def fit_and_check_isi(training, validation, seed):
    with TemporaryDirectory(prefix='aes_e22_') as directory:
        weights = CandidateWeights(directory, cfg['checkpoint_rmse_margin'])
        model, engine, info, choices = fit(
            'S3', training, validation, bundle, bge, cfg, cfg['epochs'], seed, weights,
        )
        choice = choices['B']
        if choice['epoch'] != choices['A']['epoch']:
            model.load_state_dict(torch.load(
                weights.path(choice['epoch']), map_location='cpu', weights_only=True,
            ))
        fitted_valid = predictions_array(engine.predict(validation), validation)
        if not np.allclose(fitted_valid, choice['prediction'], rtol=0, atol=1e-6):
            raise ValueError('Bobot B tidak mereproduksi prediksi validasi.')
        info['isi_latih_B'] = isi_counts(training, engine.predict(training))
        info['bobot_fitur_isi_B'] = dict(zip(
            EVIDENCE_NAMES, model.head.evidence_residual.weight.detach().cpu().flatten().tolist(),
        ))
        del engine, model
    release()
    return info, choices, predictions_array(choices['A']['prediction'], validation)
'''

REPORT = '''
def isi_counts(indices, prediction):
    column = next(i for i, (name, *_) in enumerate(TRAITS) if name == 'isi')
    labels = bundle.y[indices, column]
    classes = ordinal_prediction(np.asarray(prediction)[:, column], RUBRIC['traits'][column])
    low = labels == 1
    return {
        'jumlah jawaban': len(indices),
        'label 1': int(low.sum()),
        'label 1 terdeteksi': int((low & (classes == 1)).sum()),
        'label 2–4 salah jadi 1': int((~low & (classes == 1)).sum()),
    }


def isi_audit_tables():
    comparisons, evidence_rows, learned = [], [], []
    for number, result in sorted(fold_results.items()):
        valid = result['identity']['validation']
        prediction = np.asarray(result['checkpoints']['B']['prediction'])
        comparisons.append({'fold': number, 'bagian': 'latih', **result['training']['isi_latih_B']})
        comparisons.append({'fold': number, 'bagian': 'validasi', **isi_counts(valid, prediction)})
        learned.append({'fold': number, **result['training']['bobot_fitur_isi_B']})
        for offset, index in enumerate(valid):
            source = bundle.rows[index]
            if str(source['response_number']).strip() == '10':
                evidence_rows.append({
                    'fold': number, 'id_jawaban': source['row_id'],
                    'label isi': int(bundle.y[index, 0]), 'prediksi isi B': float(prediction[offset, 0]),
                    **dict(zip(EVIDENCE_NAMES, task_evidence(source['response_number'], source['text']))),
                })
    return (pd.DataFrame(comparisons), pd.DataFrame(learned, columns=['fold', *EVIDENCE_NAMES]),
            pd.DataFrame(evidence_rows, columns=['fold', 'id_jawaban', 'label isi', 'prediksi isi B', *EVIDENCE_NAMES]))
'''


def update(role, s):
    changes = {
        'prepare': {
            "'runtime': runtime_information(),": "'runtime': runtime_information(), 'task_evidence': 'question10_five_indicators',",
        },
        'definition:SingleScoreModel': {
            'class IndependentTraitHeads': CUES + '\n\nclass IndependentTraitHeads',
            '            self.head = PromptAwareHeads(s3_mlp_dropout)': (
                '            self.head = PromptAwareHeads(s3_mlp_dropout)\n'
                '            self.head.evidence_residual = nn.Linear(len(EVIDENCE_NAMES), 1, bias=False)\n'
                '            nn.init.zeros_(self.head.evidence_residual.weight)'
            ),
            'question_mask, answer_mask, return_features=False):': 'question_mask, answer_mask, evidence, return_features=False):',
            '        features = {': (
                '        isi_delta = self.head.evidence_residual(evidence).to(prediction.dtype)\n'
                '        prediction = torch.cat([prediction[:, :1] + isi_delta, prediction[:, 1:]], dim=1)\n'
                '        features = {'
            ),
        },
        'definition:Engine': {
            'roberta_tokenizer=None, contexts=None):': 'roberta_tokenizer=None, contexts=None, evidence=None):',
            '        self.contexts = contexts': '        self.contexts = contexts\n        self.evidence = evidence',
            'bge=features, return_features=return_features)': (
                'bge=features, evidence=self.evidence[ids].to(self.device),\n'
                '                          return_features=return_features)'
            ),
        },
        'definition:fit': {
            'cfg, epochs, seed):': 'cfg, epochs, seed, candidate_weights):',
            "    engine = Engine(model, bundle.texts, bge, cfg, device, contexts=contexts_from_bundle(bundle))": (
                '    engine = Engine(model, bundle.texts, bge, cfg, device, contexts=contexts_from_bundle(bundle),\n'
                '                    evidence=task_evidence_matrix)'
            ),
            "            row.update(best_epoch_so_far": (
                '            candidate_weights.observe(model, row, best_rmse)\n'
                '            row.update(best_epoch_so_far'
            ),
        },
        'architecture': {
            "    ['Jumlah head', '4'],": (
                "    ['Jumlah head', '4'],\n"
                "    ['Proyeksi petunjuk tugas ke isi', '5 → 1 (awal nol)'],"
            ),
        },
        'fold_helpers': {
            'from time import perf_counter': (
                'from time import perf_counter\nfrom tempfile import TemporaryDirectory\n\n' + TRAIN_VALIDATION
            ),
            """        model, engine, info, checkpoints = fit('S3', training, validation, bundle, bge,
                                  cfg, cfg['epochs'], seed)
        prediction = predictions_array(engine.predict(validation), validation)""": (
                '        info, checkpoints, prediction = fit_and_check_isi(training, validation, seed)'
            ),
            '        del model, engine\n        release()\n': '',
            '    display(pd.DataFrame(options).round(4))': (
                '    display(pd.DataFrame(options).round(4))\n'
                "    display(pd.DataFrame([{'bagian': 'latih', **info['isi_latih_B']},\n"
                "                          {'bagian': 'validasi', **isi_counts(valid, result['checkpoints']['B']['prediction'])}]))"
            ),
        },
        'validation_report': {
            'def export_validation(path, final=False):': REPORT + '\n\ndef export_validation(path, final=False):',
            '    export_excel_zip(path, {': (
                "    isi_comparisons, isi_weights, evidence = isi_audit_tables()\n"
                "    worksheets['Isi 1 Latih dan Validasi B'] = isi_comparisons\n"
                "    worksheets['Bukti Isi Soal 10 OOF B'] = evidence\n"
                '    export_excel_zip(path, {'
            ),
            "            'Konfigurasi': pd.DataFrame": (
                "            'Bobot Fitur Isi B': isi_weights,\n"
                "            'Konfigurasi': pd.DataFrame"
            ),
        },
        'final_report': {
            "        print(f'ZIP validasi A/B · {final_path}')": (
                "        print('Deteksi skor isi 1 pada checkpoint B')\n"
                "        display(sheets['Isi 1 Latih dan Validasi B'])\n"
                "        print(f'ZIP validasi A/B · {final_path}')"
            ),
        },
    }
    return replace(s, changes[role]) if role in changes else s


def build():
    path = BASE.parent / E20_NAME
    raw = path.read_bytes()
    notebook = json.loads(raw)
    source_hash = notebook['metadata']['aes_standalone']['code_sha256']
    for index, cell in enumerate(notebook['cells']):
        s = ''.join(cell['source'])
        for old, new in {
            source_hash: MARKER,
            'Reset_E20_Konfirmasi_Seed43': EXPERIMENT,
            'aes_reset_e20_checkpoint_ab_seed43': 'aes_reset_e22_s3_bukti_konteks_isi',
            'Hasil_AES_Reset_E20_Seed43_Progres_04_05.zip': 'Hasil_AES_Reset_E22_Progres_04_05.zip',
            'Hasil_AES_Reset_E20_Seed43_Validasi_Checkpoint_AB_04_05.zip': 'Hasil_AES_Reset_E22_Validasi_04_05.zip',
        }.items():
            s = s.replace(old, new)
        role = cell['metadata'].get('aes_multitrait_role')
        cell['source'] = INTRO if index == 0 else update(role, s)
    # Sisipkan tahap persiapan fitur sesudah definisi model dan sebelum training.
    insert_after = next(i for i,c in enumerate(notebook['cells']) if c['metadata'].get('aes_multitrait_role') == 'architecture')
    notebook['cells'][insert_after + 1:insert_after + 1] = [
        {'cell_type': 'markdown', 'metadata': {}, 'source': '### Siapkan bukti konteks isi pada development'},
        {'cell_type': 'code', 'metadata': {'aes_multitrait_role': 'task_evidence'}, 'outputs': [],
         'execution_count': None, 'source': """task_evidence_matrix = torch.zeros((len(bundle.rows), len(EVIDENCE_NAMES)))
development = manifest['development']
task_evidence_matrix[development] = torch.tensor([
    task_evidence(bundle.rows[i]['response_number'], bundle.rows[i]['text']) for i in development
], dtype=torch.float32)
print('Kasus development pada soal 10:', sum(str(bundle.rows[i]['response_number']).strip() == '10' for i in development))
display(pd.DataFrame([task_evidence_matrix[development].sum(0).tolist()], columns=EVIDENCE_NAMES))
"""},
    ]
    code = '\n'.join(c['source'] for c in notebook['cells'] if c['cell_type'] == 'code')
    fingerprint = hashlib.sha256(code.encode()).hexdigest()
    for index, cell in enumerate(notebook['cells']):
        cell['id'] = f'reset-e22-{index}'
        cell['source'] = cell['source'].replace(MARKER, fingerprint)
        if cell['cell_type'] == 'code':
            compile(cell['source'], cell['id'], 'exec')
    notebook['metadata']['colab']['name'] = NAME
    notebook['metadata']['aes_standalone'] = {
        'code_sha256': fingerprint, 'base': path.relative_to(ROOT).as_posix(),
        'base_sha256': hashlib.sha256(raw).hexdigest(), 'experiment': EXPERIMENT,
    }
    content = json.dumps(notebook, ensure_ascii=False, indent=1) + '\n'
    for folder in (BASE.parent, GROUP / 'notebooks'):
        (folder / NAME).write_text(content, encoding='utf-8')


if __name__ == '__main__':
    build()
