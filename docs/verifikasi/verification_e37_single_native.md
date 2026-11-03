# Verifikasi revisi E37 lima fold — 3 November2026

Revisi atas permintaan eksplisit pengguna: satu resep7e-6,S3,lima fold sekali,25epoch,patience5. Tidak ada loop tiga seed, pengulangan kontrol E30 atau seleksi LR otomatis. TRAIN_SEED42tetap satu konstanta pengacakan. Belum ada hasil penelitian kandidat E37.

Source dibangun dari notebook E36 yang dipin ke349f2e82e876333bf2270d47e7eb86b59e939a55, SHA notebook induk `b2d5233fe18799e7a5df49dfad8e8f1fed3423fd9049a3b0e775651e195f1d89`. Builder `scripts/build_e37_single_notebook.py`; notebook utama/mirror byte-identik,184cell/89codecell,semua output kosong. Visible-code SHA256 `3ef29e57fc50475eac091ea682884bca4735bfbe967585ef937cc84718c4f2b8`. Sourcefile lengkap dicatat SHA terpisah dalam manifest privat.

Semua definisi fungsi/kelas notebook cocok AST E36 setelah mengizinkan perubahan nama eksperimen pada pesan error dan nama arsip. Tidak ada perubahan arsitektur,fit,train_all,optimizer_groups,rumus metrik atau split. Hanya encoder3e-6E36diganti7e-6dan presentasi/identitas E37; parameter lain tetap E30.

Tes memakai cwd `/workspace/Automated_Essay_Scoring` dan `/tmp/e36-verify/bin/python`. Helper eksekusi fixture E36 diekstrak menjadi fungsi yang menerima path/role/LR, tanpa mengubah pemeriksaan sebelumnya; E36 ikut diuji ulang karena helper ini dipakai bersama.

```bash
PYTHONPATH=src:scripts /tmp/e36-verify/bin/python -m pytest \
 tests/test_e37_single_notebook.py tests/test_e36_notebook.py \
 -q -s --tb=short --basetemp=outputs/e37_single/fixtures
```

Tes struktur awal mendeteksi perbedaan literal pesan error `E36`→`E37`. Pemeriksa diperbaiki agar hanya mengizinkan penggantian nama eksperimen/arsip, tetap membandingkan AST seluruh definisi. Uji struktur diulang dan lolos. Run awal menghasilkan5passed/1failedpada pemeriksa nama tersebut, exit1. Dua test eksekusi penuh E37/E36 lolos. Setelah koreksi pemeriksa, perintah berikut selesai exit0:4passed,0failed,0skipped,0xfailed,2deselected. Bersama dua eksekusi penuh tadi, enam kasus berbeda telah lolos; notebook/runtime tidak diubah saat memperbaiki pemeriksa nama.

```bash
PYTHONPATH=src:scripts /tmp/e36-verify/bin/python -m pytest \
 tests/test_e37_single_notebook.py tests/test_e36_notebook.py -k 'not full' -q --tb=short
```

Log lengkap run awal disimpan untuk audit; tidak dibuat klaim berdasarkan pengecekan sintaks saja.

Fixture penuh menggunakan24baris sintetis, tokenizer teknis serta random tiny RoBERTa/XLM-R di CPU,1epoch,batas token kecil. Salinan source penelitian tetapGPU,300baris,25epoch. Kernel terisolasi menjalankan notebook awal hingga lima Excel/ZIP, lalu kernel baru mengulang tanpa mengubah checkpoint final. Pemeriksaan memastikan tepat5fold,1refit,1test,S3saja,LRencoder7e-6,MLP/proyeksi/pooling1e-4,X/y terpisah,dan tidak ada worker Python pendamping. Ini bukti teknis, bukan training GPU/pretrained asli atau peningkatan akurasi penelitian.

Tidak ada dataset,log unggahan,executed notebook,Excel,bobot,kunci atau paket siswa di staged source. Buku hasil E21–E36/ranking tidak diubah dari log parsial. Versi paired lama tetap arsip; katalog serta AGENTS menunjuk versi lima fold terbaru. Analisis log kontrol tersimpan di [review](../review/review_e37_partial_control_log.md).

Lingkungan tetap Python3.12.14,torch2.6.0+cpu,transformers4.44.2,pytest9.2,nbclient0.11,nbformat5.11,NumPy2.3.5,pandas2.2.3,sklearn1.8,openpyxl3.1.5. Log perintah/outcome dan SHA source/remote disimpan lokal di `outputs/e37_single/`.
