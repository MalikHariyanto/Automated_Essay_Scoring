"""Varian CPU dari notebook tiga seed GPU; metode dan notebook GPU tetap utuh."""
import ast
import copy
import hashlib
import json
from pathlib import Path
from build_s3_lr_cpu_notebook import replace_once, source

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'notebooks/experiments/AES_300_E6_3Seed_GPU.ipynb'
DESTINATION = BASE.with_name('AES_300_E6_3Seed_CPU.ipynb')


def build():
    original = BASE.read_bytes()
    notebook = copy.deepcopy(json.loads(original))
    old_hash = notebook['metadata']['aes_standalone']['code_sha256']
    for cell in notebook['cells']:
        cell['source'] = source(cell).replace('AES_300_E6_3Seed_GPU', 'AES_300_E6_3Seed_CPU').replace(
            old_hash, '__NOTEBOOK_CODE_SHA256__')
        if cell['cell_type'] == 'code':
            cell['outputs'] = []
            cell['execution_count'] = None

    intro = source(notebook['cells'][0])
    intro = intro.replace('dilatih ulang · GPU', 'dilatih ulang · CPU')
    start = intro.index('**GPU wajib:**')
    end = intro.index('\n\n**Penyimpanan:**', start)
    intro = intro[:start] + '''**CPU dipilih eksplisit:** Runtime → Change runtime type → CPU, lalu jalankan dari atas. Tidak memerlukan kuota GPU. Perhitungan model memakai float32 tanpa AMP; dua thread digunakan untuk membatasi beban CPU. Waktu latihan jauh lebih lama daripada T4, terutama karena ada 45 pelatihan fold + 9 refit.

Folder baru **MyDrive/AES_300_E6_3Seed_CPU_Privat**; hasil lama tidak perlu dihapus. Notebook ini melatih ulang seluruh S1/S2/S3, tidak mengambil hasil versi GPU atau eksperimen lama. RUN_TAG=ulangan_01 menandai eksperimen ini; ganti tag untuk mengulang dari nol. Jalankan dari atas dengan tag, data, kode, dan lingkungan yang sama untuk melanjutkan progres CPU yang terputus. Berpindah dari GPU ke CPU bukan resume training versi GPU pada notebook ini.''' + intro[end:]
    intro = intro.replace('seed fold, GPU, versi pustaka, presisi', 'seed fold, perangkat, versi pustaka, presisi')
    intro = intro.replace('CUDA/AMP/versi pustaka dapat memberi variasi tambahan.',
                          'Perangkat, presisi dan versi pustaka dapat memberi variasi tambahan; angka CPU tidak dijanjikan identik dengan GPU.')
    intro = intro.replace('Deterministic algorithms diaktifkan dengan warn_only, TF32 dimatikan, workspace cuBLAS dikunci;',
                          'Deterministic algorithms diaktifkan dengan warn_only dan jumlah thread dibatasi;')
    notebook['cells'][0]['source'] = intro

    setup = source(notebook['cells'][7])
    setup = replace_once(setup, "('Perangkat','GPU: '+ready['device'] if ready['cuda_available'] else 'GPU belum terdeteksi')",
                         "('Perangkat','CPU — dipilih untuk seluruh eksperimen; tidak memakai kuota GPU')")
    setup = replace_once(setup, "'AES300_REQUIRE_GPU':'1', 'CUBLAS_WORKSPACE_CONFIG':':4096:8', 'PYTHONHASHSEED':'0'",
                         "'AES300_REQUIRE_GPU':'0', 'CUDA_VISIBLE_DEVICES':'', 'PYTHONHASHSEED':'0'")
    notebook['cells'][7]['source'] = setup
    notebook['cells'][10]['source'] = replace_once(source(notebook['cells'][10]), "'device': 'cuda'", "'device': 'cpu'")
    config = source(notebook['cells'][12])
    config = replace_once(config, "device='cuda'", "device='cpu'")
    config = replace_once(config, "lesson_version='e6_three_seed_gpu_v1'", "lesson_version='e6_three_seed_cpu_v1'")
    start = config.index('# GPU wajib agar eksperimen')
    end = config.index('show(pd.DataFrame([', start)
    config = config[:start] + '''# CPU dipilih melalui cfg; GPU tidak diwajibkan walau runtime memiliki CUDA.
REQUIRE_GPU=False
torch.set_num_threads(2)
gpu=check_gpu(require_gpu=False)  # Informasi perangkat saja, tidak menentukan perangkat latihan.
print('Perangkat training: CPU | presisi: float32 | thread:',torch.get_num_threads(),flush=True)
print('Tiga seed:',TRAINING_SEEDS,'| split tetap:',SPLIT_SEED,flush=True)
''' + config[end:]
    notebook['cells'][12]['source'] = config

    for cell in notebook['cells']:
        if cell['cell_type'] != 'markdown':
            continue
        text = source(cell)
        text = text.replace('GPU wajib; semua skenario dilatih untuk tiga seed. LR E6 tetap.',
                            'CPU float32 tanpa AMP; semua skenario dilatih untuk tiga seed. LR E6 tetap.')
        text = text.replace('Training GPU memakai AMP sesuai dukungan perangkat; presisi dicatat pada audit.',
                            'Training CPU memakai float32 tanpa AMP; presisi dicatat pada audit.')
        text = text.replace('PyTorch runtime dipertahankan agar akses GPU yang tersedia tidak diganti wheel CPU.',
                            'PyTorch yang cocok dipertahankan; model tetap dijalankan di CPU sesuai konfigurasi.')
        cell['source'] = text
    notebook['cells'][44]['source'] = '''## 15 · Unduh laporan tiga seed

Cell terakhir menyusun **Hasil_E6_3_Seed_CPU.zip** berisi 16 Excel: **Ringkasan_3_Seed.xlsx** dan lima laporan untuk masing-masing seed 42, 123, 2026. Ringkasan memuat rerata dan SD CV/test, 45 fold, perbandingan nilai guru dengan sembilan prediksi model final, serta audit model/seed/perangkat.

Lima laporan per seed berisi data/tokenisasi, vektor jawaban, vektor kata/kalimat, prediksi/evaluasi, dan riwayat pelatihan. Angka tidak diubah menjadi hasil penelitian baru sebelum latihan dengan data guru benar-benar selesai. Pengunduhan tidak melatih ulang model.
'''
    notebook['cells'][46]['source'] = replace_once(source(notebook['cells'][46]),
                                                   'Hasil_E6_3_Seed_GPU.zip', 'Hasil_E6_3_Seed_CPU.zip')
    codes = [source(c).removeprefix('%%aes\n') for c in notebook['cells'] if c['cell_type'] == 'code']
    for code in codes:
        ast.parse(code)
    fingerprint = hashlib.sha256('\n'.join(codes).encode()).hexdigest()
    for cell in notebook['cells']:
        cell['source'] = source(cell).replace('__NOTEBOOK_CODE_SHA256__', fingerprint)
    notebook['metadata']['aes_standalone']['code_sha256'] = fingerprint
    notebook['metadata'].pop('accelerator', None)
    notebook['metadata']['colab']['name'] = DESTINATION.name
    notebook['metadata']['aes_experiment'].update(
        name='E6_3SEED_CPU_ALL_SCENARIOS', device='cpu',
        drive_root='AES_300_E6_3Seed_CPU_Privat', precision='float32', cpu_threads=2,
        gpu_notebook_sha256=hashlib.sha256(original).hexdigest())
    DESTINATION.write_text(json.dumps(notebook, ensure_ascii=False, indent=1)+'\n')
    (ROOT/'aes-learning/notebooks/experiments'/DESTINATION.name).write_bytes(DESTINATION.read_bytes())
    assert BASE.read_bytes() == original
    print(json.dumps({'notebook':str(DESTINATION.relative_to(ROOT)), 'code_sha256':fingerprint, 'gpu_unchanged':True}))
    return DESTINATION


if __name__ == '__main__':
    build()
