# Bukti verifikasi notebook dataset guru 300

**Perbaikan 17 Oktober 2026:** setup notebook03 kini mendukung Python3.13 selain3.12. [Bukti verifikasi perbaikan](verification_dataset300_python313.md). Laporan di bawah mencatat versi awal16 Oktober; hash notebook awal bukan hash versi perbaikan.

Verifikasi 16 Oktober 2026, checkout `codex/aes-learning`, basis perubahan `1874e60ae01dbebc93162139920cd59854fce6b3`. Pengujian dilakukan dari root repo `/workspace/Automated_Essay_Scoring`. Bukti rinci dan hasil berisi data pribadi tetap di `outputs/teacher_dataset300/`, di luar publikasi.

## Yang benar-benar dijalankan

| Pemeriksaan | Hasil | Batas makna |
| --- | --- | --- |
| Setup `python scripts/setup_dataset300.py` | Exit 0; venv baru berhasil memasang dependency dan mengimpor pustaka | CPU; tidak membuktikan GPU/AMP |
| Audit Excel unggahan asli | 300 respons, satu kolom nilai guru, 25 kelompok email, 12 pertanyaan; sumber tidak diubah | Tidak memverifikasi identitas/populasi/deskriptor rubrik |
| Split kelompok pada data asli | 240 development, 60 test; 17 dan 5 kelompok terhubung; duplikat/email tidak menyeberang | Split dibuat tanpa menggunakan hasil model atau label untuk memilih pembagian |
| Seluruh suite pytest | **75 passed, 0 failed, 0 skipped, 0 xfailed**; exit 0 | 14 kasus workflow baru, sisanya regresi kode lama; angka test bukan akurasi AES |
| Transformer fixture teknis | S1/S2/S3 forward, backward, gradien, freeze BGE, save/reload dan scoring | Arsitektur kecil berbobot acak; bukan checkpoint pretrained |
| CV/refit/test/resume fixture | 15 fold-fit dan 3 refit; test sesudah seluruh checkpoint final terkunci; pengulangan memakai cache | Memeriksa protokol program pada data sintetis, bukan hasil penelitian |
| Akumulasi gradien | Micro batch 1 cocok batch penuh pada effective batch 4, termasuk batch akhir pendek | CPU, dropout 0 pada fixture teknis |
| Vektor fixture | Lookup kata berulang sama untuk ID sama; hidden state berbeda; seluruh token RoBERTa tercakup; truncation BGE ditandai | Memeriksa mekanisme; bukan temuan linguistik checkpoint asli |
| API tokenizer fast XLM-R asli | Alignment/builder diuji memakai vocabulary fixture lokal | Bukan artifact tokenizer BGE asli yang harus diunduh |
| Referensi Gwet AC2 | Suite referensi lama R/irrCAC lulus; tambahan fixture **101 kategori** cocok R sampai toleransi pembulatan 5.01×10⁻⁶ | Semua rating fixture sintetis |
| Notebook dieksekusi | **16 code cell**, tanpa error, audit-only memakai Excel asli dan worker dari environment baru | Training, scoring model final dan encoder asli sengaja tidak diklaim berjalan |
| Format dan privasi notebook publik | 33 cell, seluruh code cell sintaks valid; output kosong; salinan mirror identik | Executed notebook privat tidak dipublikasikan |

Perintah suite:

```bash
PYTHONPATH="$PWD/src" HF_HOME="$PWD/outputs/cache/huggingface" \
AES_RSCRIPT=/workspace/aes-r-local/bin/Rscript OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
outputs/dataset300_env/bin/python -m pytest -q -rA
```

Suite awal menemukan R lokal belum memiliki `irrCAC` (73 passed, 1 failed). Instalasi package 1.4 dari mirror resmi CRAN di GitHub memperbaiki dependency, lalu suite akhir diulang setelah penambahan test batch. Mirror: `cran/irrCAC`, commit `720ab226e407e238bf85f86e9458b93e7793e8a3`. Satu warning tersisa adalah FutureWarning default `clean_up_tokenization_spaces` pada tokenizer fixture; bukan kegagalan.

Fixture tambahan AC2 menggunakan kategori 0…100: Python `0.9990618696937006`, R/irrCAC `0.99906`, selisih `0.0000018696937006`. Ini angka verifikasi rumus, **bukan hasil AES siswa**.

Notebook audit dijalankan melalui `nbclient.NotebookClient.execute`, dengan `AES300_AUDIT_ONLY=1`, `AES300_PYTHON` menunjuk environment baru, dan `AES300_DATASET` menunjuk unggahan privat. Controller notebook memakai environment Jupyter yang tersedia; worker numerik memakai `outputs/dataset300_env/bin/python`. Hasil executed disimpan privat. Versi publik tetap default training aktif.

## Lingkungan yang diuji

Python 3.12.14; torch 2.6.0+cpu; transformers 4.44.2; NumPy 2.2.6; Pandas 2.2.3; SciPy 1.15.3; scikit-learn 1.6.1; openpyxl 3.1.5; pytest 8.3.5; nbformat 5.10.4; nbclient 0.10.2. Tidak ada CUDA pada mesin verifikasi. Setup mempertahankan torch runtime bila sudah tersedia; cabang instalasi CPU tanpa torch telah dijalankan nyata.

Identitas notebook utama dan mirror saat verifikasi: SHA256 `b53aa1881b4759b3f9c48151580912f59eb33a6112374aaaef6aaa2c48801b90`. Manifest hash seluruh source yang diuji tercatat privat dalam `pytest_proof.json`.

## Yang belum dijalankan

Unduhan config revision RoBERTa dan BGE-M3 sama-sama ditolak proxy (`Tunnel connection failed: 403 Forbidden`). Mesin ini tidak memuat checkpoint asli, tidak menjalankan forward/backward checkpoint asli, dan **tidak menjalankan training/evaluasi penelitian dataset guru 300**. Karena itu tidak ada tabel hasil penelitian berisi angka yang diterbitkan. Runtime Colab dengan akses Hugging Face/GPU diperlukan untuk menjalankan tahap model yang tersedia dalam notebook.

Notebook publik tidak menyertakan file Excel, email, teks/label individual, output terisi, cache atau bobot. Prosedur publikasi memeriksa staged paths/isi dan memverifikasi SHA branch remote serta byte beberapa file setelah push. Bukti remote disimpan privat agar hash commit publikasi tidak menciptakan ketergantungan melingkar dalam commit itu sendiri.
