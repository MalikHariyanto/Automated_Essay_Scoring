# Verifikasi E53 · 6 November 2026

HEAD awal: `3e5938ce25a9fe87e471e2151b2805a6a9abd31b`. Arsitektur E43 dipin ke `7241a21dfec37c942948b754da3960fba4c7731c`; pelaporan dari E50 dipin ke `c36003670ad9476951b8b607a6018e5a24ceb7e2`, tanpa mengambil model multihead. Cwd: `/workspace/Automated_Essay_Scoring`.

Notebook utama dan mirror identik, 188 cell/91 cell kode, output kosong. SHA256 notebook: `f984d0394784e783e47459405bb6f910c3cecc4118c21fe9f9099281bc03b528`. SHA256 kode terlihat: `7a857bc7aa099822fdfc752f7ce487b3d7d5500c06381a33f67580daa456745b`.

```bash
/tmp/e39-verify/bin/python -m pytest tests/test_e53_notebook.py \
  -q -s --tb=short --basetemp=outputs/e53_study/fixtures
```

Exit0: **7 passed, 0 failed, 0 skipped, 0 xfailed**, 104,40 detik. Satu deprecation warning berasal dari plugin lingkungan. Jupyter memerlukan socket lokal. Bukti/manifest/fixture/log disimpan privat di `outputs/e53_study`.

1. **Source/protokol.** Schema notebook, hash, source native, output kosong, mirror/rebuild deterministik dan nama ZIP benar. AST arsitektur lengkap, ScalarMix, BiGRU, co-attention satu head, Engine, make_model, optimizer, train_all, metrik, split, encoder/BGE dan windowing tetap E43 setelah normalisasi nama eksperimen. Tidak ada DoRA, model multihead, fusion gate, LDS, ranking, cosine, magic/worker/exec/eval, ZIP acuan, sweep seed atau scheduler E46/E47. Dua label gate historis yang tidak dipakai dibuang dari presentation helpers. E43 tidak ditimpa.
2. **Persamaan EMA.** Lima update pada linear model dengan buffer integer dibandingkan perhitungan manual dan PyTorch `AveragedModel` dengan fungsi rata-rata yang sama. Update pertama copy; selanjutnya `min(.99,(1+t)/(10+t))`. Buffer disalin. Langkah LR0 atau AMP skipped tidak mengubah shadow/counter. Salinan tidak memerlukan gradien dan tidak mengonsumsi RNG. State/resep/counter dipulihkan persis; counter negatif/resep berbeda ditolak. Batas decay dan nilai update kedua diperiksa.
3. **Inisialisasi dan gradien.** Pada Transformer acak empat layer/hidden768, bobot dan prediksi awal EMA identik dengan model online. Pembuatan salinan tidak menggeser RNG. RoBERTa dan proyeksi BGE online menerima gradien; EMA tidak menerima gradien dan tidak berubah oleh backward online saja.
4. **Lintasan online dan refit.** Fit sintetis dua epoch tanpa validasi memakai batch/seed sama dibanding E43. Loss, LR dan seluruh bobot online yang tersimpan bit-identik E43. Model yang dikembalikan E53 memuat EMA akhir refit, bukan online akhir. Save/load mempertahankan seluruh parameter; penyimpanan model yang belum ditandai EMA, metadata resep berbeda atau final checkpoint berjenis online ditolak.
5. **Pemilihan epoch dan early stopping.** Diagnostik validasi dikendalikan sehingga minimum online dan EMA berbeda. Patience2 menghentikan run pada epoch4; best EMA epoch2 dipulihkan, sementara online masih membaik hingga epoch4. Seluruh parameter hasil cocok best EMA tersimpan. Counter/decay checkpoint epoch2 dibedakan dari akhir latihan epoch4. Test tidak menentukan pemilihan ini.
6. **Resume.** Interupsi setelah save epoch1 lalu resume dibanding run dua epoch tanpa interupsi. History, konfigurasi/counter EMA, ScalarMix/gamma dan seluruh bobot hasil bit-identik. Progres memulihkan bobot online, EMA, best EMA, optimizer, scheduler, scaler, loader dan RNG.
7. **Notebook lengkap.** Dua eksekusi dalam kernel Jupyter baru, tanpa impor source proyek dari notebook. Fixture24 baris sintetis, random Transformer4 layer, dimensi RoBERTa768/BGE1024, token pendek, CPU dan CV1epoch untuk waktu. Lima CV, satu refit/test menghasilkan lima Excel/ZIP,18OOF/6test, enam konfigurasi EMA, lima perbandingan validasi dan enam baris update epoch. Run kedua tidak mengubah checkpoint final. Pemeriksaan membaca ulang metrik, ringkasan, OOF, selisih, pilihan minimum EMA, counter dan decay. Mengubah jenis bobot, counter update, penanda best, selisih RMSE online/EMA atau mengosongkan ringkasan test/CV membuat unduhan ditolak.

Notebook publik tetap300baris, maksimal25epoch/patience5, satu5fold dan satu refit/test. EMA online/validation diagnostics bukan kontrol E43 independen, sebab stopping dipilih dari EMA. Counter mengikuti langkah optimizer berhasil dengan LR aktif; decay0,99 merupakan batas, bukan nilai tetap. Tidak ada Switch EMA atau penimpaan online per epoch. Parameter yang dioptimasi dan loss tetap E43, tetapi pilihan epoch/refit bisa berubah.

Python3.12.14, torch2.6.0+cpu, transformers4.44.2, pytest9.2.0, nbformat5.11.1, nbclient0.11.0, NumPy2.3.5, pandas2.2.3, SciPy1.17.0, scikit-learn1.8.0, openpyxl3.1.5. Tidak melakukan pretrained GPU smoke, simulasi overflow CUDA nyata, atau training penelitian E53 pada dataset asli. Pengujian skip AMP memakai kondisi terkontrol pada helper. Angka fixture bukan hasil AES; tidak ada jaminan peningkatan akurasi, seluruh runtime GPU, atau target60/60.

Hanya notebook kosong, source, builder/helper, tests dan dokumentasi aman dipublikasikan. Dataset, unggahan, hasil, executed notebook, bobot dan log tetap privat. SHA branch dan byte file GitHub setelah push dicatat di `outputs/e53_study/publication.json`.

[Metode dan rujukan](../eksperimen/eksperimen_e53_ema_native_gpu.md).
