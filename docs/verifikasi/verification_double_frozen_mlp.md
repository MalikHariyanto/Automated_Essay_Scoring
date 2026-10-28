# Verifikasi eksperimen double frozen + MLP

Notebook baru `notebooks/experiments/AES_300_S3_DoubleFrozen_MLP.ipynb` dibangun dari notebook acuan MLP pada commit `15bc7bbb2aee91ecaf44fb91a0139a8c21f64a20`. Perubahan metode hanya membekukan encoder RoBERTa pada S3; BGE tetap frozen, proyeksi/LayerNorm tambahan/MLP tetap dilatih. Mekanisme penyimpanan eksperimen ditambah penghapusan permanen checkpoint usang melalui Drive API.

## Identitas sumber

- Notebook utama dan mirror tetap SHA-256 `b06735cb45d632b92ad8ed30990e2e34d0fb66c46c612c528b812dd0d090440b`.
- Notebook eksperimen memiliki 47 cell, termasuk 30 cell kode, tanpa output eksekusi atau lampiran pribadi.
- Fingerprint runtime sesudah perbaikan penunjuk Drive: `c6d1ce2451195f3f2d410c8b6e5e55b0ef4c6fe40c7b4fbf3405c277433d96dd`.
- SHA-256 file eksperimen: `8ba884a4f71222e2f8dde94de6f7b9f8b045660b6abe67b0e88e23b485dae1c0`.
- Folder Drive dan identitas eksperimen baru terpisah dari acuan. Pembentukan ulang dengan `python scripts/build_double_frozen_notebook.py` pada Python 3.11 menghasilkan file identik.
- Dataset, hasil individual, notebook terisi, checkpoint, cache, log privat dan kredensial tidak disertakan dalam publikasi.

## Pengujian final

Dari root repository:

```bash
PYTHONPATH="$PWD/src:$PWD/tests" \
HF_HOME="$PWD/outputs/cache/huggingface" \
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 JUPYTER_PLATFORM_DIRS=1 \
outputs/dataset300_env_py313/bin/python -m pytest -q \
  tests/test_double_frozen_notebook.py \
  tests/test_drive_checkpoint_cleanup.py \
  tests/test_experimental_checkpoints.py \
  tests/test_double_frozen_drive_integration.py \
  --junitxml=outputs/double_frozen_mlp/drive_pointer_repair_tests.xml
```

Hasil revisi penunjuk Drive: **73 passed, 0 failed, 0 errors, 0 skipped**, exit code 0, durasi pytest 70,51 detik. Lingkungan: Python 3.13.5, PyTorch 2.6.0+cpu, Transformers 4.48.3, NumPy 2.2.6, pandas 2.2.3, openpyxl 3.1.5, pytest 8.3.5, nbclient 0.10.2. Angka pengujian mengukur perilaku kode, bukan akurasi AES.

Sesudah pengujian tersebut, cell perbaikan dari dokumentasi diuji tersendiri dengan command berikut dan variabel lingkungan yang sama:

```bash
outputs/dataset300_env_py313/bin/python -m pytest -q \
  tests/test_drive_checkpoint_cleanup.py::test_documented_kernel_repair_replaces_strict_cleaner_without_touching_worker \
  --junitxml=outputs/double_frozen_mlp/repair_cell_test.xml
```

Hasil tambahan: **1 passed, 0 failed/errors/skipped**, exit code 0. Total 74 kasus berhasil pada dua perintah. Implementasi tidak diubah di antara keduanya; notebook hanya menerima penjelasan Markdown tambahan.

## Perilaku yang diperiksa

Pengujian model memeriksa encoder S3 tetap frozen dan eval ketika `model.train()` dipanggil, forward tanpa gradien, serta bobot encoder tidak berubah. Proyeksi, LayerNorm tambahan dan MLP memperoleh gradien serta diperbarui; optimizer S3 hanya memuat kelompok proyeksi/head. Model kontrol S1–S2 tetap menghasilkan parameter/prediksi yang identik dengan acuan pada seed sama.

Revisi CPU menghapus kewajiban GPU pada setup eksperimen dan memberi sakelar terlihat `REQUIRE_GPU=False` pada cell konfigurasi. Cell konfigurasi asli dieksekusi dengan CUDA tidak tersedia, termasuk ketika flag lingkungan dari setup lama masih bernilai `1`: training tetap diizinkan dan perangkat dipilih CPU. Kasus CUDA/T4 tersedia juga diuji melalui simulasi untuk memastikan deteksi GPU otomatis tetap bekerja. Model, epoch, metrik dan kebijakan penyimpanan tidak diubah. Notebook utama/semi-final tidak diubah.

Training S3 yang diputus sesudah epoch 2 lalu dilanjutkan menghasilkan history, bobot dan prediksi identik dengan empat epoch tanpa putus pada fixture teknis. Eksekusi notebook penuh memakai encoder Transformer kecil berbobot acak dan data sintetis berlabel teknis: seluruh 15 fold, refit, evaluasi, lima Excel, ekspor layer MLP, dan pencocokan vektor mentah RoBERTa frozen dengan pretrained. Kernel baru dapat memakai hasil tersimpan tanpa training ulang. Import proyek diblokir selama eksekusi notebook mandiri.

Pengujian Drive API simulasi memeriksa cakupan folder/run/stage, pagination, dua generasi yang dilindungi, metadata/ukuran/checksum yang belum sinkron, metadata berubah saat penghapusan, CV/split tidak cocok, bukti model final rusak, izin tidak cukup, error server dan 404 setelah penghapusan. Nama file yang sama pada skenario/folder lain tetap terlindungi. Tidak ada pengosongan Sampah global.

Pengujian transport menjalankan worker subprocess dan fungsi kernel `aes_cell` asli dari notebook untuk memastikan permintaan cleanup menerima balasan pada keberhasilan maupun kegagalan, tanpa worker macet. Pengujian komposisi menyambungkan saver Torch asli, antrean retry persisten dan pembersih API: epoch lama benar-benar dihapus dari API simulasi, dua terbaru dipertahankan, hasil CV memicu pelepasan checkpoint fold, dan kegagalan checksum menahan penghapusan sampai bukti benar.

Revisi sinkronisasi memeriksa penunjuk server yang masih merujuk snapshot lama: snapshot terbaru dan rujukan server tetap bisa dimuat, training dapat menyimpan epoch berikutnya, dan rujukan lama dilepas setelah server tersinkron. Checksum salah, file rujukan hilang/di Sampah, scope atau tahap berbeda, snapshot server lebih baru, konflik isi nama yang sama dan perubahan penunjuk saat penghapusan tetap ditolak. Pencarian folder meminta metadata folder saja, tanpa checksum yang tidak diperlukan.

Pengujian cell perbaikan menjalankan kode Python persis dari [petunjuk pemulihan](../arsip/perbaikan_sinkronisasi_checkpoint_drive.md), dengan unduhan source diganti fixture lokal. Objek pembersih ketat diganti versi baru, sedangkan objek worker, konfigurasi dan fingerprint tetap identik. Ini memverifikasi jalur pemasangan pada runtime lama tanpa membuat run penelitian baru.

## Batas bukti

Tidak menjalankan ulang training penelitian siswa, GPU T4 fisik, autentikasi Colab fisik, atau akses langsung ke Drive pengguna. Sinkronisasi/API Drive nyata diperiksa saat notebook dijalankan pengguna; kegagalan verifikasi menghentikan training dengan progres disimpan. Pengujian tidak membuktikan bahwa double frozen menurunkan RMSE.

Log, XML, manifest hash sumber dan bukti privat disimpan di `outputs/double_frozen_mlp`; direktori tersebut tidak dipublikasikan.
