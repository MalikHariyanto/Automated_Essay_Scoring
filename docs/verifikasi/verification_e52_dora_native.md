# Verifikasi E52 · 6 November 2026

HEAD awal pekerjaan: `b64034f506df647119d6097b356e6b8d6a7711d0`. Arsitektur E43 dipin ke `7241a21dfec37c942948b754da3960fba4c7731c`; pelaporan lengkap dari E50 dipin ke `c36003670ad9476951b8b607a6018e5a24ceb7e2`, tanpa mengambil model multihead. Cwd: `/workspace/Automated_Essay_Scoring`.

Notebook utama dan mirror identik, 188 cell/91 cell kode, tanpa output. SHA256 notebook: `07974afd8dd33cfa75b0b4acd557a30c7ea1a631994c8811ad2103330377ad17`. SHA256 kode terlihat: `b04c0249f412f531a7284e901faaf7d44416868ebe02dbc3593798460c1bed4a`.

```bash
/tmp/e39-verify/bin/python -m pytest tests/test_e52_notebook.py \
  -q -s --tb=short --basetemp=outputs/e52_study/final_fixtures
```

Exit 0: **6 passed, 0 failed, 0 skipped, 0 xfailed**, 97,60 detik. Ada satu deprecation warning dari plugin lingkungan. Kernel Jupyter memerlukan socket lokal. Manifest, log dan artefak pengujian berada di `outputs/e52_study`, tidak dipublikasikan.

1. **Source/protokol.** Schema, hash, output kosong, mirror, rebuild deterministik, nama ZIP benar, kode native tanpa magic/worker/exec/eval. AST model fusion, ScalarMix, BiGRU, co-attention satu head, Engine, split, metrik, encoder/BGE, tokenisasi/window dan loop epoch tetap E43 setelah normalisasi nama eksperimen. Tidak ada LDS, ListNet, cosine, model multihead, fusion gate, ZIP hasil acuan, loop banyak seed atau scheduler refit E46/E47. Dua label gate historis yang tidak digunakan dibuang dari presentation helpers. Source E43 tidak berubah.
2. **DoRA resmi dan kondisi awal.** Memakai PEFT 0.14.0 dengan `use_dora=True`. Encoder fixture RoBERTa asli berparameter acak, empat layer/hidden768. Semua bobot dasar dan komponen fusion bersama identik E43; state RNG setelah pemasangan adapter juga identik. Keluaran awal S3 cocok E43 dengan toleransi 1e-6; split identik. Delapan target query/value memuat tepat 104.448 parameter A/B/magnitude dan B awal nol. Pada encoder publik 12 layer, rumus yang sama menghasilkan 313.344 parameter adapter.
3. **Pembelajaran dan precision.** Dua langkah AdamW pada target sintetis mengubah parameter arah A/B dan magnitude; gradien finite. Bobot dasar RoBERTa tetap bit-identik dan tidak menerima gradien. Proyeksi BGE dan matching menerima gradien; optimizer mencakup semua parameter trainable tepat sekali, kelompok DoRA LR1e-4. Forward/backward dalam autocast CPU BF16 juga finite. Ini bukan smoke CUDA FP16/BF16 atau bukti runtime GPU tertentu.
4. **Konfigurasi dan checkpoint.** LoRA biasa/DoRA nonaktif, rank/target/alpha/dropout tidak sesuai, LR nonfinite atau LR encoder tidak sama LR adapter ditolak. Full-state save/load mempertahankan parameter dan prediksi. Metadata DoRA berbeda atau magnitude checkpoint hilang ditolak; tidak ada fallback diam-diam ke base/LoRA.
5. **Resume.** Fit dua epoch tanpa interupsi dibanding interupsi setelah save epoch pertama dan resume. History, laporan adapter, jumlah parameter serta seluruh state model bit-identik. Model, optimizer, scheduler, scaler, loader dan RNG mengikuti alur resume E43.
6. **Notebook lengkap dan hasil terserialisasi.** Dua eksekusi di kernel Jupyter baru, tanpa impor source repo dari notebook. Fixture 24 baris sintetis, random Transformer empat layer, RoBERTa768/BGE1024, token pendek, CPU; CV dibatasi satu epoch untuk pengujian. Lima CV baru, satu refit/test menghasilkan lima Excel/ZIP, 18 OOF/6 test, 48 baris audit modul DoRA dari enam checkpoint. Run kedua tidak mengubah checkpoint final. Pemeriksaan ZIP memverifikasi metrik, ringkasan, OOF, rekap selisih, adapter aktif, target/rank/jumlah parameter dan bobot asli frozen. Mengubah status DoRA, status base trainable, rank, jumlah parameter, menghapus target, atau mengosongkan ringkasan test/CV membuat unduhan ditolak.

Source publik tetap memakai 300 baris, maksimal25epoch/patience5, satu 5-fold dan satu refit/test. Satu perubahan resep encoder adalah full fine-tuning E43 LR5e-6 → DoRA query/value rank8/alpha16 LR1e-4. Metode adaptasi dan LR aktif berubah bersama; bukan ablasi LR tunggal. Arsitektur fusion dan MSE tetap. Norma adapter bukan importance kausal. Parameter BGE pretrained tidak masuk model fusion karena fiturnya dihitung sekali dengan encoder frozen.

Lingkungan: Python3.12.14, torch2.6.0+cpu, transformers4.44.2, peft0.14.0, accelerate0.34.2, pytest9.2.0, nbformat5.11.1, nbclient0.11.0, NumPy2.3.5, pandas2.2.3, SciPy1.17.0, scikit-learn1.8.0, openpyxl3.1.5. Tidak ada pretrained GPU smoke atau training penelitian E52 dengan dataset asli. Angka fixture bukan hasil AES. Pengujian tidak menjamin bebas seluruh bug, kenaikan akurasi, atau target60/60.

Hanya source, notebook kosong, tests, builder/helper dan dokumentasi aman dipublikasikan. Data/unggahan, hasil, executed notebook, Excel, checkpoint dan log tetap privat. SHA branch GitHub dan pencocokan byte beberapa file setelah push dicatat di `outputs/e52_study/publication.json`.

[Metode dan referensi](../eksperimen/eksperimen_e52_dora_native_gpu.md).
