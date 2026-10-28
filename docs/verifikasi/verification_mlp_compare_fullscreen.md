# Verifikasi layar penuh perbandingan AES

Perubahan tampilan pada empat ruang: perbandingan awal/MLP, meja metrik, double frozen, E5 LoRA. File model/notebook, workbook asli, snapshot metrik serta data hasil tidak diubah. Satu dialog menerima elemen ruang asli; ketika ditutup, elemen dikembalikan ke lokasi semula. Tidak menyalin tabel, sehingga nilai, tab, handler, input dan cache tetap sama.

Header/sidebar/footer besar disembunyikan saat fullscreen. Bar kontrol 42 px dan tab ringkas mempertahankan navigasi, skenario serta tombol keluar. Konten memakai sisa viewport; pembatas max-height tabel dilepas agar tidak membuat panel gulir vertikal tambahan. Tabel tetap dapat digeser mendatar. Body belakang dikunci dan modal mengisolasi fokus. Native fullscreen dipakai jika diizinkan; kegagalan API ditangani dengan modal ukuran viewport yang sama. Esc, tombol keluar, atau perpindahan ruang memulihkan elemen serta posisi halaman. Restorasi scroll memperhitungkan selesai keluarnya native fullscreen.

## Bukti browser

Cwd `/workspace/Automated_Essay_Scoring`, dengan server lokal root repo pada port 8765:

```bash
node scripts/check_mlp_compare_fullscreen.mjs
```

Exit 0, **4 kelompok lulus, 0 gagal**; Chromium, versi pada `outputs/mlp_study/comparison_fullscreen_browser.json`. Log: `outputs/mlp_study/comparison_fullscreen_browser.log`.

- Keempat ruang diuji dengan native fullscreen nyata browser, lalu dengan requestFullscreen ditolak melalui mock prototype: geometri tepat 0,0–lebar/tinggi viewport. Area konten lebih dari 88% tinggi pada viewport desktop pengujian. Header/footer/sidebar tersembunyi; HTML isi sebelum/sesudah masuk identik.
- Pilihan skenario bekerja. Wheel menggulir isi tanpa menggulir halaman belakang. Tombol keluar mengembalikan fokus ke tombol pembuka dan posisi halaman semula. Esc menutup dialog. Pengujian menemukan dan memperbaiki pengambilan posisi halaman setelah placeholder memicu scroll anchoring.
- Lebar 390, 800, 1440 px: tidak ada overflow halaman, tab training/review tetap bekerja, 15 fold tetap ada, dan geser mendatar mencapai kolom eksperimen terakhir. Perpindahan ke buku menutup fullscreen secara bersih.
- Pada HTML privat, tabel prediksi yang sudah dimuat tetap identik setelah masuk fullscreen; sheet Excel dapat dibaca. Kode fullscreen dan 30 blok tetap dapat dibuka setelah keluar dari perbandingan. Tidak ada pageerror browser pada hasil akhir.

Regresi:

```bash
outputs/dataset300_env_py313/bin/python -m pytest \
  tests/test_mlp_study.py tests/test_mlp_comparison.py tests/test_mlp_frozen.py \
  -q --disable-warnings
```

Exit 0, **17 passed, 0 failed, 0 skipped, 0 xfailed**, 0,24 detik. Log: `outputs/mlp_study/comparison_fullscreen_regression.log`.

HTML publik dibangun bersama HTML pribadi dari sumber yang sama; private bundle tetap berisi MLP, pembanding awal, frozen dan E5. Paket unduhan terenkripsi dibangun ulang dengan kunci yang sudah ada dan roundtrip AES-GCM diverifikasi oleh `scripts/package_private_study.py`. Jawaban/nilai individual dan kunci tidak diterbitkan dalam bentuk plaintext. Ini pemeriksaan UI/hasil tersimpan, bukan training atau evaluasi penelitian baru.
