# Perbandingan seluruh jawaban dan peringkat prediksi

> Dokumen tahap terdahulu. Angka dan keputusan dibaca dalam konteks versi ini; untuk kondisi aktif gunakan [status proyek](../../STATUS_PROYEK.md) dan [panduan AI](../../PANDUAN_AI_PROYEK.md).

Tab Guru & prediksi kini membuka seluruh 60 jawaban, tanpa memilih ID satu per satu. Skenario S1/S2/S3 dibandingkan terpisah agar jumlah kemenangan tidak mencampur model yang berbeda. Enam eksperimen tetap tersedia, termasuk E6 terbaru.

- **Semua jawaban:** satu baris per jawaban, enam prediksi di kiri, nilai guru di kanan. Warna konsisten per eksperimen. Sel terdekat diberi bingkai dan bintang, tidak hanya perubahan warna. Nilai guru dan ID tetap terlihat saat digeser mendatar. Nama kolom mengikuti saat digulir vertikal, tanpa menambah area gulir vertikal kedua.
- **Sorotan terdekat:** 60 baris yang sama, hanya eksperimen terdekat dan prediksinya. Semua pemenang seri ditampilkan. Tidak menyembunyikan jawaban yang sulit atau memilih contoh yang menguntungkan.
- **Peringkat keseluruhan:** eksperimen diurutkan dari jumlah jawaban terdekat terbanyak ke tersedikit. Jumlah terdekat sendiri dan seri dipisahkan; jumlah sama mendapat posisi peringkat sama. RMSE tetap tampil sebagai ukuran berbeda.

Pencarian ID dan pengurutan nilai guru/kesalahan tersedia. Filter hanya mengubah daftar yang terlihat; peringkat tetap menggunakan seluruh 60 jawaban. Klik ID untuk membuka teks jawaban dalam dialog, tanpa meninggalkan daftar. Mode fullscreen tetap tersedia. Data individual hanya ada di buku terenkripsi; versi publik memuat agregat peringkat.

## Definisi dan angka

Untuk satu jawaban dan satu skenario, hitung `abs(prediksi_mentah − nilai_guru)` pada enam eksperimen. Error terkecil menjadi terdekat. Beda error sampai 0,00001 poin dianggap seri numerik untuk menghindari kemenangan semu akibat perbedaan perhitungan sangat kecil. Ini bukan batas toleransi skor atau adjacent agreement. Tabel diringkas dua desimal; tooltip memuat nilai mentah. Pembulatan kategori tidak dipakai untuk memilih pemenang.

| Peringkat S3 | Eksperimen | Jawaban terdekat / 60 |
| --- | --- | ---: |
| 1 | E6 · MLP, LR S3 5e-6 | 18 |
| 2 | E4 · Konteks pair | 15 |
| 3 | E3 · Double frozen | 11 |
| 4 | E0 · Awal lima epoch | 8 |
| 5 bersama | E2 · MLP acuan | 4 |
| 5 bersama | E5 · LoRA + MLP | 4 |

Tidak ada seri per jawaban pada S3; seluruh hitungan berjumlah 60. Posisi sama pada E2/E5 berarti jumlah kemenangan sama, bukan prediksi kedua model identik. S1 memiliki 17 jawaban dengan kontrol yang seri; S2 memiliki 18. Seri dihitung pada setiap eksperimen terkait, sehingga total lintas eksperimen bisa melebihi 60. Kontrol yang dipakai ulang bukan pengulangan training independen.

Hitungan terdekat bukan persentase akurasi, kelayakan penggunaan atau pengganti RMSE. Contohnya E4 S3 menang pada 15 jawaban, lebih banyak daripada E5 (4), tetapi RMSE E4 6,7315 lebih besar daripada E5 6,5305. Besar kesalahan pada seluruh jawaban tetap penting. Tidak ada training atau pemilihan konfigurasi baru dalam pembaruan tampilan ini.

## Pemeriksaan

Dari root repo, interpreter `outputs/dataset300_env_py313/bin/python`:

```bash
python scripts/build_prediction_ranking.py --sources outputs/mlp_study/prediction_sources.json
python -m pytest tests/test_prediction_ranking.py tests/test_mlp_dashboard.py tests/test_mlp_study.py tests/test_mlp_comparison.py tests/test_mlp_frozen.py -q --disable-warnings
node scripts/check_mlp_dashboard.mjs
node scripts/check_mlp_lr_package.mjs
```

File sources adalah pemetaan lokal E0 ke workbook evaluasi dan E2–E6 ke ZIP asli; file serta jawaban tidak diterbitkan. Agregat diaudit terhadap ID unik dan nilai guru yang sama, serta penolakan data tidak lengkap, ID berbeda/duplikat, guru berbeda atau prediksi tidak valid.

Unit/regresi: 28 passed, 0 failed/skipped/xfail. Tujuh pengujian baru mencakup seri, konservasi satu poin bila seri dibagi, penggunaan prediksi mentah, kontrol nyaris identik, serta input tidak cocok. UI memakai jumlah terdekat termasuk seri sebagai dasar urutan, bukan skor seri terbagi; skor terbagi disimpan hanya untuk audit.

Browser dashboard mencocokkan 1.080 prediksi dan seluruh sorotan pada 60 jawaban × 6 eksperimen × 3 skenario, peringkat ketiga skenario, mode sorotan, pencarian/pengurutan, dialog, fullscreen, nilai guru di kanan, gulir tunggal, header mengikuti, dan ukuran tablet/HP. Regresi workbook serta 30 blok kode tetap diperiksa. Paket utuh terenkripsi dibuka dengan kunci sebelumnya dan unduhan diperiksa byte demi byte. Bukti lokal berada di `outputs/mlp_study/overall_*`.
