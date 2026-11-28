# E26 · Ensemble prediksi isi S1 dan S3

[Ringkasan metrik dan checksum](Ringkasan_Ensemble.json). Prediksi mentah Isi menggabungkan 75% S3 E21-B dan 25% S1 E14. Tiga trait lain berasal dari E21-B. Pemilihan bobot memakai development. Test 270 jawaban dari 27 kelompok sudah pernah ditinjau dalam rangkaian penelitian, sehingga hasil ensemble bersifat eksploratif.

ID jawaban, baris, nomor soal, label guru, dan kelompok diperiksa pada kedua split. Nomor fold juga cocok pada seluruh 1.070 prediksi development. Metrik memakai aturan E21. Hasil tidak melibatkan training ulang.

| Paket pada test | RMSE gabungan | Macro QWK |
|---|---:|---:|
| S1 E14 | 0,4639 | 0,3348 |
| S3 E21-B | 0,4466 | 0,4152 |
| E26 | 0,4451 | 0,4174 |

E26 membaik pada keempat metrik Isi dibanding E21-B, tetapi Pearson dan AC2 Isi masih di bawah S1. Isi skor 1 tetap 0/3 dan tidak ada label 2–4 yang salah turun ke 1. E21-B tetap hasil S3. E26 dicatat sebagai eksperimen tambahan ensemble di level prediksi.

Sumber: [E14 S1/S2](../E14-S1S2.json) dan [E21](../E21.json). Data dan prediksi individual tetap privat.
