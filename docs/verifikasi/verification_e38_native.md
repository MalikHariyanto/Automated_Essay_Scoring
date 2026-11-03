# Verifikasi E38 native — 3 November 2026

E38 berangkat dari metode E30: hanya dropout hidden MLP S3 berubah0,1→0,2. Encoder5e-6 dan semua LR lain1e-4 tetap. Ini source kandidat; tidak ada hasil training penelitian E38 atau klaim perbaikan akurasi.

## Identitas dan cakupan

- Baseline kerja `c25a58026585b67715c6dc919bf9f81dca6db272`; cwd `/workspace/Automated_Essay_Scoring`.
- Builder `scripts/build_e38_notebook.py` membaca format native E36 dipin ke `349f2e82e876333bf2270d47e7eb86b59e939a55`.
- SHA256 notebook induk `b2d5233fe18799e7a5df49dfad8e8f1fed3423fd9049a3b0e775651e195f1d89`.
- Notebook utama/mirror byte-identik,184cell/89codecell, semua output kosong. Visible-code fingerprint `86370761bb10a7c299d5d33105b8708a273969db6fe7c2b1d91dcf941ea000c8`.
- Tidak ada perubahan source E30/E36/E37 atau HTML hasil. BGE frozen, S3 saja,5fold sekali,maks25epoch,patience5,1refit/1test, tanpa loop multi-seed.

## Pengujian

```bash
PYTHONPATH=src:scripts /tmp/e38-verify/bin/python -m pytest \
 tests/test_e38_notebook.py -q -s --tb=short \
 --basetemp=outputs/e38_study/fixtures
```

Exit0, **3passed,0failed,0skipped,0xfailed**, satu warning deprecation plugin pytest lingkungan;66,41detik pada fixture CPU, bukan estimasi durasi penelitian GPU. Log privat `outputs/e38_study/tests.log`.

1. **Struktur/source:** validasi nbformat,89cell Python biasa tanpa magic/exec/eval/worker/pemeriksaan cosine, hash kode dan builder deterministik, mirror cocok, notebook lama utuh. AST `AnswerBiGRUAttention`, `AnswerBiGRUCoAttentionPool`, `Engine`, `fit`, `train_all`, optimizer, split, BGE, metrik dan save_model identik dengan E30.
2. **Perubahan terbatas pada MLP:** konfigurasi dibanding seluruh DEFAULTS E30; hanya key tambahan s3_mlp_dropout0,2. Seluruh state_dict awal identik. Daftar semua modul Dropout diperiksa: satu-satunya probabilitas berbeda adalah `head.2`, dari0,1ke0,2; encoder dan parameter dropout lama tetap. Prediksi Engine mode eval identik bit-for-bit pada fixture. LR aktual roberta5e-6 dan head/projection/pooling_norm1e-4 diperiksa. Nilai p tidak sah ditolak; load_model mempertahankan p0,2 dan menolak config checkpoint dengan p berbeda. S1/S2 ditolak.
3. **Notebook lengkap dan resume:** semua section dijalankan dalam kernel terisolasi memakai24baris sintetis, tokenizer teknis dan random tiny RoBERTa/XLM-R satu layer di CPU. Hanya salinan fixture mengubah epoch menjadi1,token pendek dan guard CPU; source publik tetap25epoch/GPU/300baris. Lima fit CV, satu refit dan satu evaluasi test menghasilkan lima Excel/ZIP. X berisi tepat tiga teks, y terpisah dan sama dengan input training aktual. File checkpoint/config dan05→Pengaturan Eksperimen mencatat dropout MLP0,2,parameter lama0,1,seed42. Kernel baru melanjutkan hasil cocok hash tanpa mengubah checkpoint final. Tidak ada worker .py atau output S1/S2.

Torch/model numerik memang dijalankan pada fixture, tetapi bobot pretrained asli/Colab GPU belum diuji pada pekerjaan ini. Perbedaan probabilitas dropout saat training adalah perilaku yang diuji, bukan bukti kualitas penelitian. Model final menggunakan eval sehingga dropout tidak aktif saat prediksi.

## Lingkungan dan publikasi

Python3.12.14;torch2.6.0+cpu;transformers4.44.2;pytest9.2.0;nbformat5.11.1;nbclient0.11.0;NumPy2.3.5;pandas2.2.3;SciPy1.17.0;scikit-learn1.8.0;openpyxl3.1.5.

Lingkungan tes memakai venv terpisah `/tmp/e38-verify`. Tidak mengubah dependency proyek atau menjalankan training pada ZIP/data penelitian pengguna. Referensi dropout ditambahkan dengan batas bahwa p0,2 adalah pilihan ablasi, bukan angka optimal dari paper.

Hanya source tanpa output, builder, tes dan dokumentasi aman masuk commit. Dataset/log unggahan, executed notebook,Excel,bobot,kunci dan paket terenkripsi tetap lokal. SHA staged source dan hasil perintah tersimpan pada manifest privat `outputs/e38_study/verification.json`; SHA branch remote dan byte notebook/mirror/metode dicek setelah push di `remote_verification.json`.
