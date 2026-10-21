"""Bangun notebook double frozen + MLP dengan checkpoint yang dibersihkan manual."""
import ast
import copy
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT/'notebooks/experiments/AES_300_S3_DoubleFrozen_MLP_Manual.ipynb'


def source(cell):
    return cell['source'] if isinstance(cell['source'], str) else ''.join(cell['source'])


def definitions(text):
    text = text.removeprefix('%%aes\n')
    return {node.name: ast.get_source_segment(text, node) for node in ast.parse(text).body
            if isinstance(node, (ast.FunctionDef, ast.ClassDef))}


def build():
    main_path = ROOT/'notebooks/03_AES_dataset_guru_300.ipynb'
    frozen_path = ROOT/'notebooks/experiments/AES_300_S3_DoubleFrozen_MLP.ipynb'
    originals = {path: path.read_bytes() for path in (main_path, frozen_path)}
    main, frozen = (json.loads(originals[path]) for path in (main_path, frozen_path))
    notebook = copy.deepcopy(main)
    old_hashes = [n['metadata']['aes_standalone']['code_sha256'] for n in (main, frozen)]
    replacements = ('class SingleScoreModel', 'specifications=pd.DataFrame', 'def vector_lab(',
                    'ROBERTA_CHECKPOINT=', 'def export_training_event(', 'def answer_vectors_report(')
    frozen_sections = {anchor: next(source(c) for c in frozen['cells'] if anchor in source(c))
                       for anchor in replacements}
    frozen_training = definitions(next(source(c) for c in frozen['cells']
                                       if c['cell_type'] == 'code' and 'def fit(' in source(c)))
    manual = definitions((ROOT/'src/aes/teacher_score/manual_checkpoints.py').read_text())

    for cell in notebook['cells']:
        text = source(cell)
        if cell['cell_type'] == 'code':
            for anchor in replacements:
                if anchor in text:
                    text = frozen_sections[anchor]
                    break
            if 'def save_training_progress(' in text:
                names = definitions(text)
                text = ('%%aes\n# Semua checkpoint epoch dan fold disimpan untuk dibersihkan manual.\n\n'
                        + '\n\n\n'.join(manual[name] if name in manual else frozen_training[name]
                                           for name in names if name != 'clear_training_progress') + '\n')
            if 'def train_all(' in text:
                text = '\n'.join(line for line in text.splitlines()
                                 if not line.lstrip().startswith('clear_training_progress(')) + '\n'
            text = text.replace("'AES300_REQUIRE_GPU':'1' if IN_COLAB else os.environ.get('AES300_REQUIRE_GPU','0')",
                                "'AES300_REQUIRE_GPU':'0'")
            text = text.replace("lesson_version='s3_double_frozen_mlp_autocleanup_v1'",
                                "lesson_version='s3_double_frozen_mlp_manual_v1'")
            text = text.replace("print('Progres disimpan otomatis setiap epoch. Saat sesi terputus, jalankan kembali notebook yang sama dari atas.')",
                                "print('Progres disimpan otomatis setiap epoch. Checkpoint lama tetap disimpan; penghapusan dilakukan manual.')")
            for old in old_hashes:
                text = text.replace(old, '__NOTEBOOK_CODE_SHA256__')
            cell['outputs'] = []
            cell['execution_count'] = None
        else:
            if text.startswith('# AES guru 300'):
                text = '''# AES 300 · Double frozen + MLP · Checkpoint manual

Versi eksperimen terpisah dari notebook MLP acuan/semi-final. **S1** tetap RoBERTa fine-tuned + head linear; **S2** tetap BGE-M3 frozen + MLP. **S3** memakai RoBERTa frozen dan BGE-M3 frozen; LayerNorm tambahan, proyeksi BGE dan MLP setelah concat tetap dilatih.

Alur mengikuti notebook MLP: setup → dataset → pembagian data → tokenisasi → embedding → model → training → prediksi/evaluasi → lima Excel. Dataset `fix_dataset_300.xlsx` menggunakan satu skor guru 0–100, split 80/20 dan 5-fold CV pada development. Maksimum 25 epoch per fold, patience 5, learning rate dan batch tetap seperti acuan.

**Penyimpanan sederhana:** progres tetap disimpan otomatis ke Drive setiap epoch. Tidak ada penghapusan checkpoint otomatis, pemeriksaan pembersihan melalui Drive API, antrean pembersihan atau otorisasi API tambahan. File checkpoint epoch/fold dan penunjuknya tetap disimpan setelah tahap selesai. Penunjuk memuat dua snapshot terbaru untuk resume; file epoch lebih lama tetap berada di folder untuk dihapus manual.

**Cara menjalankan:** jalankan dari atas, hubungkan Drive seperti biasa, lalu upload `fix_dataset_300.xlsx` jika diminta. CPU dan GPU didukung; T4 lebih cepat. Data dan hasil disimpan di `MyDrive/AES_300_DoubleFrozen_MLP_Manual_Privat`. Notebook utama tetap utuh. Ini run baru yang terpisah; checkpoint dari versi auto-delete tidak dipindahkan atau dipilih otomatis.

**Hasil:** prediksi skor, perbandingan nilai guru/prediksi/pembulatan, RMSE, Pearson, QWK, Gwet AC2, serta vektor jawaban/kata/kalimat tetap tersedia. Cell terakhir menyusun `Hasil_AES_Ringkas.zip` berisi lima Excel. S3 menggabungkan fitur pada tingkat jawaban; tidak dibuat vektor fusion per kata yang tidak dihasilkan arsitekturnya.

Notebook mandiri: kode runtime tersedia langsung di cell dengan komentar; tidak perlu clone/import proyek GitHub. Dataset, output pribadi dan bobot siswa tidak disertakan dalam file publik.
'''
            if '## 10 · Model' in text:
                text = next(source(c) for c in frozen['cells']
                            if c['cell_type'] == 'markdown' and '## 10 · Model' in source(c))
            if '## 12 · Pelatihan' in text:
                text += '\n\nCheckpoint setiap epoch dan fold tetap disimpan. Tidak ada penghapusan otomatis setelah epoch, fold, atau refit selesai. Riwayat dan penunjuk progres disimpan untuk pemulihan; penghapusan file dilakukan manual.\n'
            if '## 14 · Lihat vektor' in text:
                text += '\n\nEncoder RoBERTa S3 frozen: vektor mentah tetap pretrained; LayerNorm tambahan, proyeksi dan MLP setelah concat tetap dilatih.\n'
        for old in ('AES_300_DoubleFrozen_MLP_Privat', 'AES_300_MLP_Privat'):
            text = text.replace(old, 'AES_300_DoubleFrozen_MLP_Manual_Privat')
        for old in ('/content/AES_300_DoubleFrozen_MLP', '/content/AES_300_MLP', 'outputs/AES_300_MLP'):
            text = text.replace(old, '/content/AES_300_DoubleFrozen_MLP_Manual' if old.startswith('/content/')
                                else 'outputs/AES_300_DoubleFrozen_MLP_Manual')
        cell['source'] = text
        cell['metadata'] = {}
        cell.pop('attachments', None)

    normalized = [ast.dump(ast.parse(c['source'].removeprefix('%%aes\n')), include_attributes=False)
                  for c in notebook['cells'] if c['cell_type'] == 'code']
    fingerprint = hashlib.sha256('\n'.join(normalized).encode()).hexdigest()
    for cell in notebook['cells']:
        cell['source'] = cell['source'].replace('__NOTEBOOK_CODE_SHA256__', fingerprint)
    notebook['metadata']['aes_standalone']['code_sha256'] = fingerprint
    notebook['metadata']['colab']['name'] = DESTINATION.name
    notebook['metadata'].pop('accelerator', None)
    notebook['metadata']['aes_experiment'] = {
        'name': 'S3_DOUBLE_FROZEN_MLP_MANUAL', 's1_roberta': 'fine_tuned',
        's3_roberta': 'frozen', 'bge': 'frozen', 's3_head': '1024 -> 256 -> GELU -> dropout -> 1',
        'automatic_checkpoint_deletion': False, 'drive_api_cleanup': False,
        'drive_root': 'AES_300_DoubleFrozen_MLP_Manual_Privat'}
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    DESTINATION.write_text(json.dumps(notebook, ensure_ascii=False, indent=1)+'\n')
    for path, original in originals.items():
        assert path.read_bytes() == original
    print('Notebook:', DESTINATION.relative_to(ROOT))
    print('Fingerprint:', fingerprint)
    return DESTINATION


if __name__ == '__main__':
    build()
