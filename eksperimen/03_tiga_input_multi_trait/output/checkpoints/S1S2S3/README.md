# Audit test S1, S2, dan S3

Keempat paket prediksi E14-S1, E14-S2, E14-S3, dan E21-B cocok pada **270 ID jawaban**, nomor soal, baris sumber, kelompok, dan empat label guru. Ada 27 kelompok. Hash manifest E14 dan E21 berbeda, sehingga kecocokan test diverifikasi per jawaban pada workbook sumber. Audit ini tidak menyatakan bahwa seluruh manifest training identik.

| Resep | RMSE gabungan ↓ | Macro QWK ↑ |
|---|---:|---:|
| E14-S1 | 0,4639 | 0,3348 |
| E14-S2 | 0,4866 | 0,0236 |
| E14-S3 | 0,4684 | 0,3161 |
| E21-B S3 | 0,4466 | 0,4152 |

[Empat metrik per trait dan interval bootstrap berpasangan](Audit_Test_Sepadan.json). Metrik dihitung dari prediksi mentah dengan aturan E21. Macro QWK adalah rata-rata empat trait pada test gabungan. Bootstrap memakai 5.000 sampel ulang berpasangan pada tingkat kelompok, seed 2026. Interval bersifat deskriptif untuk prediksi tersimpan dan tidak mencakup variasi latihan atau seleksi resep.

E21-B mengungguli S1 pada keempat metrik organisasi, kebahasaan, dan mekanik. Pada isi, S1 lebih baik pada keempat metrik. E14-S3 berada di bawah S1 dalam RMSE gabungan dan macro QWK. E14 memakai refit tunggal untuk setiap skenario, sedangkan E21-B memakai ensemble lima fold, attention, dan resep organisasi yang berbeda. Hasil mendukung keunggulan **paket E21-B** pada test ini. Besaran manfaat yang khusus berasal dari penggabungan encoder memerlukan kontrol dengan protokol latihan dan inferensi sepadan.

Sumber agregat: [E14 S1/S2](../E14-S1S2.json), [E14 S3](../E14.json), dan [E21 A/B](../E21.json). Data dan prediksi per jawaban tetap privat.
