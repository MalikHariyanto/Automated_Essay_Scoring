# Verifikasi buku hasil E29 dan tiga kode terpilih

Basis repository: `b783fe05896cd4a0efcc4a81508d70b0eedbd717`. Input E29: ZIP lima Excel unggahan pengguna; path asli dan checksum dicatat lokal pada `outputs/e29_study/all_input_paths.json` dan snapshot agregat. Source notebook rujukan E29 dipin ke `7a76191be20604482fa53f770699563b380120ba`; executed notebook tidak tersedia.

| Pemeriksaan | Hasil final | Makna |
| --- | --- | --- |
| Unit/integritas | **26passed,0failed/skipped/xfail** | Metrik, LR, kode/catatan, seleksi tiga buku, privasi dan preservasi delapan hasil lama |
| Browser hasil | **11 kelompok lulus,0gagal** | 1620prediksi,60teks/checksum,36duel,27model,kurva/fold/peta,scroll/fullscreen,Excel mandiri dan3kode |
| Browser identitas | **5 kelompok lulus,0gagal** | Sepuluh identitas;9hasil;3tombol kode;E30duaarea kosong;keyboard/mobile |
| Browser unduhan | **2 kelompok lulus,0gagal** | ZIP byte-identik; salah kunci lalu retry dengan kunci lama;9hasil/3kode offline |
| Arsip asli | **9ZIP/45Excel byte-identik** | File unduhan asli tetap utuh; publik tidak memuat arsip/teks/skor individual |

Perintah pada root repo:

```bash
outputs/dataset300_env_py313/bin/python scripts/build_three_input_study.py --reuse-audit --e21 PATH_E21 --e22 PATH_E22 --e23 PATH_E23 --e24 PATH_E24 --e25 PATH_E25 --e26 PATH_E26 --e27 PATH_E27 --e28 PATH_E28 --e29 PATH_E29
outputs/dataset300_env_py313/bin/python -m pytest tests/test_e29_study.py tests/test_three_input_study.py tests/test_three_input_identities.py tests/test_e27_e28_study.py tests/test_e21_study.py -q --junitxml=outputs/e29_study/unit.xml
node scripts/check_e29_study.mjs
node scripts/check_three_input_identities.mjs
AES_PACKAGE_CHECK_DIR=outputs/e29_study/ node scripts/check_e27_e28_package.mjs
```

Audit E29 menghitung ulang RMSE/Pearson/QWK/AC2 dari pasangan guru/prediksi, mencocokkan300baris/60test dan lima fold, epoch pilihan terhadap minima validasi, refit ceil median, bobot attention finite/berjumlah1, dimensi ringkasan128/proyeksi768 dan isi lima workbook. Audit keluarga mencocokkan tiga teks, label dan anggota split/fold di seluruh sembilan ZIP. Kontrol S1 E29 sama per jawaban dengan E26. Hasil ini bukan bukti identitas byte runtime atau replikasi pretrained GPU.

Kode E29 lengkap30cell/2815baris,143catatan dan10padanan screenshot cocok dengan notebook pin. E26/E24 tetap byte-identik dengan buku sebelumnya. Kode enam eksperimen lain sengaja tidak diembed atas instruksi terbaru; snapshot/hasil dan sumber notebook tetap ada. Snapshot/28duel lama tetap identik. Jumlah terdekat dihitung ulang untuk sembilan peserta; bukan perubahan prediksi lama.

Saat pemeriksaan, tes lama yang mengharuskan E21 ter-embed disesuaikan dengan instruksi tiga kode. Ekspektasi jumlah duel pada tes browser juga dikoreksi menjadi36×3=108. Ditemukan input komponen vektor memicu redraw saat blur sehingga klik sel pertama hilang; handler kini tidak merender ulang jika nilainya sama. Setelah perbaikan, klik dan Enter membaca nilai lengkap. Seluruh hasil final di tabel lulus.

Python3.13.5, NumPy2.2.6, Chromium/Playwright lokal. Log, versi browser, checksum, sumber manifest, proof privasi/ukuran dan perintah tersimpan lokal di `outputs/e29_study/`. Tidak ada training atau evaluasi penelitian baru dijalankan; yang diverifikasi ialah unggahan dan aplikasi pembacanya.

Paket menggunakan kunci lama dan empat bagian terenkripsi di bawah100MiB. Helper dipin ke commit data sebelum dipublikasikan. Setelah push, branch SHA, HTML/sumber dan seluruh bagian unduhan diperiksa byte/checksum serta CORS; bukti akhir pada `outputs/e29_study/remote.json`.
