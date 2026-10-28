# E6 2.0 — tiga seed dan jawaban lengkap

> Dokumen tahap terdahulu. Angka dan keputusan dibaca dalam konteks versi ini; untuk kondisi aktif gunakan [status proyek](../../STATUS_PROYEK.md) dan [panduan AI](../../PANDUAN_AI_PROYEK.md).

E6 2.0 merupakan pengulangan E6 dengan seed training 42, 123, dan 2026. Semua S1/S2/S3 dilatih ulang. Ini bukan eksperimen tiga input: model tetap membaca jawaban siswa. Soal dan jawaban referensi ditampilkan di buku untuk membantu membaca hasil, tanpa mengubah prediksi model.

Enam belas Excel sumber mencakup satu ringkasan tiga seed dan lima workbook untuk masing-masing seed. Audit mencocokkan 300 jawaban/label, pembagian development/test, anggota lima fold, serta 60 ID/skor guru test dengan E6 lama. Empat metrik test dihitung ulang dari 540 prediksi mentah. Lima belas hasil fold tiap seed dicocokkan dengan workbook ringkasan; epoch terpilih dicocokkan dengan minimum RMSE pada riwayat. Metrik CV tidak diklaim dihitung ulang dari pasangan prediksi tiap fold karena pasangan itu tidak tersedia pada unggahan.

| Skenario | CV RMSE mean ± SD | Test RMSE mean ± SD |
| --- | ---: | ---: |
| S1 | 7,9833 ± 0,3589 | 7,0959 ± 0,8666 |
| S2 | 6,4475 ± 0,0206 | 6,7987 ± 0,0396 |
| S3 | 6,1411 ± 0,2767 | 6,2904 ± 0,1192 |

SD adalah simpangan baku sampel dari tiga metrik per seed. Untuk CV, setiap seed lebih dahulu menghasilkan rerata lima fold; SD pada tabel dihitung atas tiga rerata itu. S3 memiliki rerata test terbaik dari tiga skenario. S1 seed 123 memiliki RMSE test individu terendah, 6,0959. Hasil tersebut tidak dipakai untuk memilih seed dari data uji atau membuktikan signifikansi statistik. Variasi S1 lebih besar daripada S2/S3 pada tiga pengulangan ini.

RMSE test S3 E6 lama adalah 6,3744 pada satu run; E6 2.0 menghasilkan rerata 6,2904. CV E6 lama 5,9915, sedangkan rerata CV E6 2.0 6,1411. Perbandingan ini tidak membuktikan bahwa pengulangan seed menyebabkan peningkatan model. Buku menampilkan hasil tiap seed dan ketidakpastiannya, tanpa mengganti hasil E6 lama.

## Tampilan dan sumber

Menu **E6 2.0 · Uji 3 seed** menyediakan ringkasan mean ± SD, sembilan hasil test, 45 hasil fold beserta riwayat epoch, 60 baris dengan sembilan prediksi, sembilan peta skor, semua sheet dari 15 workbook, serta unduhan ringkasan asli. Tabel lintas eksperimen memakai eksperimen sebagai kolom; kolom tiga seed diberi label mean ± SD. Tujuh run tunggal lama tetap utuh. Peringkat jumlah prediksi terdekat tujuh run tidak mencampurkan metrik rata-rata sebagai prediksi ensemble.

Menu **Guru & prediksi → Jawaban & skor E6** menampilkan 60 jawaban lengkap beserta soal, referensi, nilai guru dan tiga skor model. Semua jawaban dapat dibuka sekaligus atau dicari. Teks yang sama tersedia pada E6 2.0 setelah kecocokan ID/label diverifikasi. Header diringkas, tabel lebar dapat digeser mendatar, dan mode layar penuh tetap tersedia.

[Notebook sumber tiga seed](https://github.com/MalikHariyanto/Automated_Essay_Scoring/blob/d8c21accc6e0bbff7a9d95c02a553b8bfeed8340/notebooks/experiments/AES_300_E6_3Seed_CPU.ipynb) dipisahkan dari meja belajar E6 lama. Workbook saja tidak membuktikan bahwa kode runtime identik byte demi byte dengan notebook tersebut. [Mosbach dkk.](https://arxiv.org/abs/2006.04884) menjadi rujukan tentang stabilitas fine-tuning; penelitian itu bukan bukti bahwa tiga seed pasti cukup atau bahwa S3 pasti unggul pada dataset ini.

## Verifikasi dan privasi

44 unit/integritas test lulus (0 failed, 0 skipped, 0 xfailed), mencakup pengikatan konteks dengan ID/teks/label serta regresi buku lama. Delapan kelompok browser E6 2.0 lulus: 540 nilai prediksi cocok sumber, sembilan peta berisi masing-masing 60 pasangan, unduhan ringkasan dan tiga workbook evaluasi byte-identik, pagination baris/komponen, tiga ukuran layar, layar penuh, dan hasil tujuh run lama tetap tersedia tanpa error JavaScript. Tujuh kelompok browser jawaban E6 serta 11 kelompok regresi sebelumnya juga lulus. Dua pemeriksaan paket juga lulus: pengunduh merakit ZIP byte-identik dan buku terenkripsi terbuka dengan kunci sebelumnya, menampilkan hasil lama serta E6 2.0 tanpa jaringan berikutnya. Ini audit hasil tersimpan dan pengujian buku, bukan training baru.

Snapshot publik hanya berisi agregat, riwayat epoch, audit teknis, inventaris dan checksum. Teks, ID/skor individual, Excel asli serta notebook beroutput disimpan dalam buku terenkripsi dengan kunci lama. Kunci tidak diterbitkan. Bukti rinci lokal berada di `outputs/mlp_study/e62_*` dan `answers_*`.
