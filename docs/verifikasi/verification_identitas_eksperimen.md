# Verifikasi identitas eksperimen E21–E30

Pembaruan antarmuka pada29Oktober2026; basis `7a76191be20604482fa53f770699563b380120ba`. CWD root repository. Source manifest dan log lengkap tetap lokal pada `outputs/identity_study/`; data siswa tidak dipublikasikan.

| Pemeriksaan | Hasil | Cakupan |
| --- | --- | --- |
| Unit/integritas | **23 passed;0failed/skipped/xfail** | Seluruh perbandingan dan delapan buku anak identik; sepuluh identitas; E29/E30 tanpa hasil/kode embed; kontrak dan sumber |
| Browser katalog | **5 kelompok lulus;0gagal** | Pencarian/filter termasuk tanpa hasil, sitasi, empat ruang kosong, aksi kode, keyboard, fullscreen, desktop/tablet/HP |
| Regresi browser | **10 kelompok lulus;0gagal** | 1440prediksi,60teks/checksum,28duel,kurva/fold,peta,Excel,scroll,kode/catatan/fullscreen,preview publik |
| Paket browser | **2 kelompok lulus;0gagal** | ZIP unduhan byte-identik; salah kunci lalu retry kunci lama; offline;10identitas/4ruang kosong; kode lengkap/fullscreen |
| Arsip/privasi | **8ZIP/40Excel asli utuh** | ZIP di HTML privat byte-identik dengan unggahan; publik sama dengan parent privat tanpa arsip individual; aplikasi MLP lama utuh |
| Tautan Markdown | **110 dokumen diperiksa** | Pemeriksaan ulang dilakukan sebelum commit; jumlah tautan tercatat pada log lokal |

Perintah utama:

```bash
outputs/dataset300_env_py313/bin/python scripts/build_three_input_study.py --reuse-audit --e21 PATH_E21 --e22 PATH_E22 --e23 PATH_E23 --e24 PATH_E24 --e25 PATH_E25 --e26 PATH_E26 --e27 PATH_E27 --e28 PATH_E28
outputs/dataset300_env_py313/bin/python -m pytest tests/test_three_input_identities.py tests/test_three_input_study.py tests/test_e27_e28_study.py tests/test_e21_study.py -q --junitxml=outputs/identity_study/unit.xml
node scripts/check_three_input_identities.mjs
node scripts/check_e27_e28_study.mjs
AES_PACKAGE_CHECK_DIR=outputs/identity_study/ node scripts/check_e27_e28_package.mjs
```

Python3.13.5; Chromium/Playwright melalui browser lokal, alamat loopback8765. Tidak menjalankan training atau evaluasi penelitian baru. Kesalahan awal pada tes baru mengasumsikan semua pin sumber40karakter; source E23/E24 historis memakai pin pendek sah. Assertion dikoreksi untuk menerima pin7–40karakter; isi sumber/pin hasil tidak diubah. Final23tes semuanya lulus.

Unduhan menggunakan nama file dan kunci lama; bagian terenkripsi dipin ke commit publikasi data, kemudian helper dipublikasikan terpisah. SHA branch remote dan isi file/bagian unduhan diperiksa setelah push; bukti final pada `outputs/identity_study/remote.json`.
