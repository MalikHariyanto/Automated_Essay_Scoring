# Verifikasi E31/E32 · turunan E25

**10passed,0failed,0skipped,0xfail**,154,93detik. Satu warning deprecation Jupyter tentang platformdirs, bukan kegagalan. Python3.13.5,PyTorch2.6.0+cpu. Log/XML dan eksekusi fixture privat pada `outputs/e31_e32_verification/`.

```bash
outputs/dataset300_env_py313/bin/python -m pytest tests/test_e31_e32_notebooks.py -q --basetemp=outputs/e31_e32_verification/final_fixtures --junitxml=outputs/e31_e32_verification/final.xml
```

- Kedua notebook memiliki30cellkode, GPU wajib, tanpa mount Drive/clone/import file proyek; output publik kosong dan mirror identik. Builder deterministik dan fingerprint diverifikasi. Source E25 dipin `bc4713872c7a7ce23f7dad8ae3ea73a2b75d9b86`.
- E31 mempertahankan arsitektur/fitur/window/input/evaluator E25. Semua parameter trainable tercakup sekali; hanya MLP S2/S3 turun5e-6, gate/pooling/proyeksi/norm1e-4. S1 kontrol.
- S1/S2/S3 diuji forward/backward dan reload. Inisialisasi seluruh parameter E25 yang sudah ada tetap identik, termasuk E32; modul co-attention baru mengisolasi konsumsi RNG. Gate dimulai0,5, pengali1 pada kedua cabang. Tidak ada modul GRU pada kedua notebook.
- E32 diuji kedua jalur attention, jumlah bobot1, mask konteks/padding, sensitivitas referensi, konsistensi batch versus satu contoh, gradien respons/referensi/layer, AMP CPU dan penolakan mask tumpang tindih. Mengubah pooling mengubah hasil tetapi tidak mengubah matriks co-attention.
- Seluruh30cell notebook dijalankan pada fixture24baris/model Transformer acak satu layer dimensi768/1024:15foldCV+3refit,5Excel dan resume kernel baru tanpa mengulang training/test. E32 mengekspor tiga jenis bobot berjumlah1/window serta pasangan top1; gate dan kedua pengali tetap diekspor.

Tes CPU memakai guard fixture khusus; notebook publik tetap GPU wajib. Ini bukan training pretrained GPU atau hasil penelitian. Checkpoint sementara yang dibuat sendiri oleh tes sintetis dibersihkan setelah verifikasi untuk membatasi disk; paket HTML/ZIP penelitian pengguna tidak dihapus. Percobaan awal pada sandbox tidak menyelesaikan Jupyter dan diinterupsi; test-only perubahan parameter sebelum backward juga diperbaiki urutannya. Run final di atas semuanya lulus.

HTML hasil dan source eksperimen lama tidak diubah. Belum ada output penelitian E31/E32. Keberadaan file/mirror dan branch remote diperiksa setelah push; bukti pada `outputs/e31_e32_verification/remote.json`.
