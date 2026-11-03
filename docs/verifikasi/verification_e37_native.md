# Verifikasi E37 native — 3 November 2026

Notebook E37 menguji LR encoder5e-6kontrol E30 vs7e-6kandidat pada tiga seed training dan fold tetap. Belum ada training penelitian E37, checkpoint pretrained asli yang diuji di GPU, atau angka akurasi penelitian baru. Tes berikut hanya membuktikan struktur, matematika dan alur teknis.

## Identitas source

- Cwd `/workspace/Automated_Essay_Scoring`, baseline kerja `31085137dab95259807f66458df4b94a7484cff8`.
- Format dipin ke E36 source pada `349f2e82e876333bf2270d47e7eb86b59e939a55`; SHA256 notebook induk `b2d5233fe18799e7a5df49dfad8e8f1fed3423fd9049a3b0e775651e195f1d89`.
- Builder `scripts/build_e37_notebook.py`; source helper build-time `scripts/e37_native_sections.py`. Fungsi helper diekstrak menjadi cell Python individual yang terlihat, tidak diimpor/dijalankan sebagai worker di Colab.
- Notebook utama/mirror byte-identik,190cell/94codecell. Semua output kosong dan execution_count null.
- Visible code fingerprint `158944e6e2c3a82419c3b8ddf071547d4b1b47452d44b95bb2093011d9180b28`; dihitung dengan placeholder hash agar identitas kode deterministik.
- Notebook E30/E36 tidak berubah. Arsitektur, Engine, optimizer grouping,fit,metrik,split,BGE,save/load model cocok AST E30. `train_all` menambahkan jalur CV-only, tanpa mengubah langkah pembaruan parameter.

## Perintah dan hasil

Cwd root repo; bukti privat berada di `outputs/e37_study/`.

```bash
PYTHONPATH=src:scripts /tmp/e36-verify/bin/python -m pytest \
 tests/test_e37_notebook.py -k 'not full_notebook' -q --tb=short
```

Exit0: **3passed,0failed,0skipped,0xfailed,1deselected**, satu warning deprecation plugin lingkungan. Struktur native/hash/builder, kelengkapan30hasil CV, larangan hasil parsial/NaN, aturan seri mempertahankan kontrol, pemilihan rerata seluruh seed, dan penghapusan slot hanya setelah JSON hasil tersimpan diperiksa. Uji satu langkah optimizer dengan encoder fixture membuktikan prediksi awal identik dan update kelompok selain RoBERTa identik; norma update RoBERTa kandidat/kontrol≈1,4. Split/kelompok/fold tidak berubah oleh seed training. S1/S2 ditolak oleh make_model.

```bash
PYTHONPATH=src:scripts /tmp/e36-verify/bin/python -m pytest \
 tests/test_e37_notebook.py -k full_notebook -q -s --tb=short \
 --basetemp=outputs/e37_study/kernel_fixture
```

Exit0: **1passed,0failed,0skipped,0xfailed,3deselected**, satu warning deprecation plugin lingkungan. Total empat test yang berbeda lolos; jumlah deselected tidak dihitung sebagai skipped. Eksekusi lengkap dan fresh-kernel resume selesai dalam sekitar110detik pada fixture ini, bukan estimasi waktu GPU penelitian.

Notebook benar-benar dijalankan dari awal di direktori terisolasi, tanpa import proyek atau file Python pendamping. Fixture24baris sintetis memakai tokenizer teknis dan random tiny RoBERTa/XLM-R satu layer berukuran keluaran768/1024. Default penelitian tetap300baris,GPU,25epoch; hanya salinan fixture memakai CPU,1epoch dan batas token kecil. Label fixture tidak menggantikan skor guru dan tidak dipublikasikan.

Terbukti30fit CV selesai sebelum3fit refit, tidak ada indeks holdout pada train/validation, kunci seleksi ada sebelum refit, dan refit hanya memakai LR yang dipilih CV. Enam laporan CV memuat30fold dengan nol evaluasi test; hanya tiga final_report/checkpoint dihasilkan. Semua tiga seed test dilaporkan, seed42selalu tabel/vektor utama. Lima workbook/ZIP standar mencakup seluruh tab LR/seed tambahan. Slot checkpoint fold selesai tidak tersisa. Kernel kedua memakai ulang hasil cocok hash dengan nolfit; ketiga model final byte-identik. Tidak ada S1/S2, file worker .py atau ensemble.

## Lingkungan dan batas

Python3.12.14,pytest9.2,nbclient0.11,nbformat5.11,torch2.6.0+cpu,transformers4.44.2,NumPy2.3.5,pandas2.2.3,scikit-learn1.8,openpyxl3.1.5. Kernel fixture menggunakan jaringan lokal Jupyter; pesan TCP tanpa enkripsi merupakan warning kernel lokal, bukan keluaran penelitian.

Koneksi Hugging Face tidak dibutuhkan/diklaim dalam test ini. Source Colab menggunakan checkpoint/revision asli E30 dan guard CUDA. Kompatibilitas setup native berasal dari E36, tetapi pengujian fixture CPU tidak membuktikan runtime Colab GPU terbaru atau kualitas model penelitian. Rujukan komponen serta status LR sebagai hipotesis terlihat di notebook.

## Publikasi

Commit hanya source tanpa output, test dan dokumentasi aman. Tidak ada dataset/ZIP unggahan, Excel, executed notebook, bobot atau kunci pada staged source. Buku hasil HTML dan ranking E21–E36 tidak berubah. Setelah push, SHA branch dan byte notebook/mirror/metode diverifikasi terpisah; rekaman privat `outputs/e37_study/remote_verification.json`.
