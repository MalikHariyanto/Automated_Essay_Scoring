# Progres E1–E4 reset

Keempat eksperimen telah selesai training dan evaluasi pada Tesla T4; hasil diambil dari workbook pengguna. Dataset sama: 1.340 jawaban, 134 kelompok, 10 soal; development1.070/test270. Keanggotaan split dan lima fold dicocokkan langsung, tidak hanya hash metadata. RMSE test dihitung ulang dari prediksi dan cocok. E3/E4 diverifikasi terhadap checksum seluruh pecahan dan manifest.

E4 terbaik nominal: CV0,425284/test0,466443. E2 CV0,425449/test0,466544 tetap kontrol; perbaikan test E4 hanya0,000101poin (0,022%), belum bukti keunggulan yang meyakinkan dari satu run. E3 CV0,425778/test0,472422; membantu kebahasaan/mekanik tetapi mengorbankan isi/organisasi. E1 CV0,426831/test0,468598.

| Trait | RMSE E2 | RMSE E3 | RMSE E4 |
| --- | ---: | ---: | ---: |
| Isi | 0,667898 | 0,693697 | 0,667114 |
| Organisasi | 0,328005 | 0,335481 | 0,328252 |
| Kebahasaan | 0,417080 | 0,406660 | 0,417389 |
| Mekanik | 0,378184 | 0,365508 | 0,378514 |

E3 best epoch3/5/5/3/5; E4 2/10/5/3/5. Refit keduanya5epoch. Loss training turun sementara validasi memburuk setelah checkpoint terbaik, konsisten dengan overfitting. Menambah patience saja belum didukung hasil.

Attention E4 aktif: diagonal0, baris berjumlah1, norma regresor konteks dari0 menjadi0,0176–0,0189. Prediksi berubah rata-rata0,00372poin dari E2. Ini inspeksi checkpoint final, bukan penjelasan kausal atau OOF.

Skor rendah masih sulit dikenali. Organisasi skor1 dikenali E2 11/35, E3 9/35, E4 11/35; kebahasaan19/70,22/70,20/70; mekanik19/56,24/56,19/56. Kandidat berikutnya: uji penanganan distribusi skor timpang, bobot hanya dihitung dari training fold. Belum diimplementasikan.

Ini perbandingan varian S3; belum membuktikan S3 lebih baik daripada S1/S2. Rencana perbandingan tersebut perlu dataset/split yang sama. Empat metrik per trait tersedia di Excel dan HTML. Hasil satu run tidak membuktikan signifikansi statistik.
