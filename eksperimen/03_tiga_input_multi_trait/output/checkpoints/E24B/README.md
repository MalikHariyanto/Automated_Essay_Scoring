# E24B · Hasil pilot empat unsur

[Ringkasan numerik dan checksum](Ringkasan_Pilot_Empat_Unsur.json). Model dan revisi sama dengan E24, inferensi GPU pada 107 jawaban development soal 10. Analisis berpasangan memakai 106 jawaban utuh. Satu jawaban skor 4 hanya memiliki skor kutipan.

| Aturan audit pada jawaban utuh | Isi 1 | Isi 2–4 |
|---|---:|---:|
| Dukungan lokasi sekolah E24 < 0,50 | 2/8 | 2/98 |
| Minimal satu dari empat dukungan E24B < 0,50 | 8/8 | 97/98 |
| Ditandai E24B di luar yang sudah ditandai E24 | 6 | 95 |

Tiga dari lima probe berpasangan menghasilkan arah benar. Probe pembeda lokasi utama dan pelaku gagal. Untuk probe sekolah langsung, arah lokasi utama benar tetapi skor dukungannya hanya sekitar 0,011. E24B memberi sinyal pelaku lemah pada 95 dari 98 jawaban isi 2–4. Hasil menunjukkan sensitivitas terhadap cara hipotesis dirumuskan. Label isi adalah acuan skor, sehingga hitungan audit tidak menentukan apakah lokasi, pelaku, dan unsur lain benar-benar ada.

**Keputusan.** Tahan integrasi empat fitur NLI dan pertahankan E21-B sebagai acuan S3. Gunakan E24 sebagai bukti terbatas bahwa lokasi dapat terbaca pada kasus tertentu. Bila pengujian diteruskan, mulai dengan ekstraksi potongan teks untuk pelaku, kejadian, waktu, dan tempat, lalu audit keberadaan bukti dan keterikatannya dengan kejadian utama. Hindari label isi untuk memilih contoh atau menyesuaikan ambang ekstraksi.

Kutipan dan hasil per jawaban tetap privat. Pilot tidak melatih S3, membuka test, atau menghasilkan metrik prediksi skor.
