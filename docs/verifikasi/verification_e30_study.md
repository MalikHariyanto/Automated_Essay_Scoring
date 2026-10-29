# Verifikasi E30 dalam buku hasil

Basis repository `f07d407288e99e913b01427a9170f7e9d800d92d`. ZIP E30 asli dari pengguna berisi lima Excel. Source rujukan dipin ke `7a76191be20604482fa53f770699563b380120ba`; executed notebook/checkpoint runtime tidak tersedia.

| Pemeriksaan | Hasil final | Cakupan |
| --- | --- | --- |
| Unit/integritas | **29passed,0failed/skipped/xfail** | Sembilan snapshot/prediksi/36duel lama utuh;10hasil;3kode;konfigurasi/metrik E30;146catatan tepat source |
| Browser hasil | **11kelompok lulus,0gagal** | 1800prediksi,60teks/checksum,45duel,30model,kurva/fold/peta,Excel,kode,scroll/fullscreen desktop/tablet/HP |
| Browser identitas | **5kelompok lulus,0gagal** | Seluruh10identitas aktual;tidak ada placeholder;3kode;filter/search/keyboard/mobile |
| Browser paket | **2kelompok lulus,0gagal** | Unduhan ZIP byte-identik;retry salah kunci lalu kunci lama;offline;10hasil/3kode |
| Arsip/privasi | **10ZIP/50Excel byte-identik** | Semua sumber asli utuh;preview publik tanpa data individual;aplikasi MLP lama tidak berubah |

Perintah utama pada root repository:

```bash
outputs/dataset300_env_py313/bin/python scripts/build_three_input_study.py --reuse-audit --e21 PATH_E21 --e22 PATH_E22 --e23 PATH_E23 --e24 PATH_E24 --e25 PATH_E25 --e26 PATH_E26 --e27 PATH_E27 --e28 PATH_E28 --e29 PATH_E29 --e30 PATH_E30
outputs/dataset300_env_py313/bin/python -m pytest tests/test_e30_study.py tests/test_e29_study.py tests/test_three_input_study.py tests/test_three_input_identities.py tests/test_e27_e28_study.py tests/test_e21_study.py -q --junitxml=outputs/e30_study/unit.xml
node scripts/check_e30_study.mjs
node scripts/check_three_input_identities.mjs
AES_PACKAGE_CHECK_DIR=outputs/e30_study/ node scripts/check_e27_e28_package.mjs
```

Audit Excel E30 menghitung ulang RMSE/Pearson/QWK/AC2; seluruh selisih terhadap laporan<1e-6. Data, label dan split/fold cocok dengan sembilan unggahan sebelumnya. Epoch minima/refit9/19/18, dimensi ringkasan128/proyeksi768 dan ketiga kelompok bobot diperiksa. Pemeriksaan tambahan membandingkan16946posisi/token pooling dengan co-attention respons; keduanya cocok dengan tokenisasi. Bobot referensi17600dan top116946 juga diverifikasi posisi/rentang/normalisasi. Ringkasan top1 tidak cukup untuk membuktikan seluruh matriks alignment atau maksimum secara independen.

Kontrol S1/S2 E30 sama per jawaban dengan E26. Semua36duel lama utuh, sementara jumlah terdekat dihitung ulang pada10peserta. E29 tetap terendah RMSE/utama kode; E30 unggul MAE/toleransi/jumlah terdekat. Kode E29/E26 byte-identik dengan buku sebelumnya; E24 sengaja tidak diembed lagi menurut aturan tiga RMSE S3 terendah. Kode E30 cocok seluruh30cell/2955baris/146catatan/10padanan screenshot dengan notebook pin.

Tes browser awal mengasumsikan angka terbaik berada pada kolom terakhir. Setelah E30 ditambahkan, kolom RMSE terbaik tetap E29, sehingga tes diperbaiki untuk memilih kolom sesuai data; angka dan sorotan aplikasi tidak diubah. Semua pemeriksaan final di atas lulus.

Python3.13.5,NumPy2.2.6,Chromium/Playwright lokal. Log/perintah,versi browser,source manifest,checksum dan proof privasi tersimpan privat di `outputs/e30_study/`. Pengujian tidak melatih model baru dan tidak membuktikan replikasi pretrained GPU. Paket memakai kunci lama dan empat bagian terenkripsi,masing-masing<100MiB; helper dipin ke commit data. SHA remote dan isi HTML/parts diperiksa sesudah push; proof akhir `outputs/e30_study/remote.json`.
