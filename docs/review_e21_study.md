# Buku E21 tiga input — terpisah dari buku eksperimen lama

`docs/belajar_E21_Spider.html` adalah salinan agregat aman. Buku lengkap berisi kode dan seluruh lima Excel E21 di dalam paket terenkripsi yang terpisah. Buku lama beserta hasil E0–E8/E6 2.0 tetap tersedia dan tidak ditimpa. E22 kemudian tersedia dalam [buku gabungan E21/E22](review_three_input_study.md); paket E21 terpisah tetap dapat dipakai.

Variabel input E21 adalah `soal`, `jawaban_referensi`, `response_text` pada seluruh S1/S2/S3. Label tetap `rater`, satu skor guru0–100. Perubahan jumlah input tidak menghasilkan tiga label. S1 memakai RoBERTa fine-tuned + linear; S2 BGE frozen + MLP; S3 fusion + MLP. LR encoder S1/S3 sama-sama5e-6; proyeksi/head1e-4. Input E21 adalah single sequence. Soal, referensi, penanda jawaban dan jawaban masuk urutan yang sama; E22 kelak menguji format pasangan.

## Audit hasil tersimpan

Lima workbook unggahan diperiksa:300 baris dengan ketiga teks/label lengkap,240development/60test, anggota lima fold tanpa irisan train/validation/test pada unit ID,15 hasilCV, seluruh riwayat dan refit sesuai ceil median epoch pilihanCV. Ini audit struktur hasil, bukan verifikasi independen penilaian guru atau bebas kebocoran semantik pada semua soal.

Empat metrik test dihitung ulang dari pasangan guru/prediksi. Deviasi terbesar sekitar 2,4e-7, dalam toleransi1e-6 untuk presisi nilai yang disimpan; angka evaluasi asli tidak diganti. Pembulatan untuk kategori memakai nearest0–100, tie ke kategori lebih rendah. CV dicocokkan dengan minimum epoch dalam riwayat; prediksi individual tiap fold tidak ikut ZIP, sehingga CV tidak diklaim dihitung ulang dari pasangan individual.

| E21 | S1 | S2 | S3 |
| --- | ---: | ---: | ---: |
| RMSE test | 8,3369 | 7,8672 | 6,7231 |
| Pearson | 0,4487 | 0,2690 | 0,5187 |
| QWK | 0,2515 | 0,1114 | 0,4340 |
| Gwet AC2 | 0,9681 | 0,9724 | 0,9789 |
| Mean RMSE CV | 7,3645 | 7,3506 | 6,6246 |
| SD antar-fold | 0,3966 | 0,7844 | 1,1658 |
| Epoch refit | 9 | 19 | 8 |
| Prediksi terdekat | 19/60 | 14/60 | 27/60 |

S3 terbaik pada empat metrik test dan meanCV run ini. SD antar-fold S3 juga paling besar; variasi fold tidak disamakan dengan variasi seed. Bias rata-rata S3 sekitar+0,0192 poin, S1−4,4888 poin, S2−2,1308 poin. Bias dekat nol tidak berarti tiap jawaban tepat. Persentase dalam±5/±10 poin hanya analisis toleransi eksploratif, bukan standar kelayakan universal. Satu run belum membuktikan keunggulan statistik atau stabilitas metode.

Kedua encoder pada 300 baris yang tersedia masing-masing memakai satu window per jawaban. Tidak ada token jawaban yang tidak tercakup atau pemotongan soal/referensi pada hasil ini. Mekanisme multi-window tetap dijelaskan karena tersedia pada kode, dengan anggaran memperhitungkan konteks/prefix dan overhead cadangan4; kapasitas E21 tidak dijelaskan sebagai510 token jawaban tanpa konteks.

## Isi dan tampilan

Tema, ilustrasi Spider-Man, peta belajar, kode berwarna, sticky note yang mengikuti baris, satu kanvas fullscreen, pilihan per blok/semua blok, grafik epoch, penjelajah Excel dan perpindahan scroll mempertahankan pola buku sebelumnya. Isi dirancang ulang khusus E21:16 bab,30 blok kode lengkap,2699 baris dan110 catatan. Tabel hasil membandingkan skenario sebagai kolom. Menu aktif meliputi ringkasan, input/label, training/fold, guru/prediksi, peta skor, arti metrik, semua Excel dan metode/sumber. Perbandingan dengan E22 kini berada pada buku gabungan.

Seluruh60 jawaban test dapat dilihat bersama sebagai tabel180 prediksi, kartu lengkap tiga teks, atau peringkat skenario. Nilai guru berada di kanan; warna model dan penanda terdekat membantu perbandingan. Teks dapat dibuka semua dan dicari. Dialog jawaban menyertakan soal/referensi/jawaban. Tiga peta skor memakai skala warna bersama. Seluruh43 sheet lima workbook tersedia pada buku lengkap; pagination dan rentang delapan komponen menjaga lebar tabel, tanpa membuang nilai. Klik sel untuk teks/nilai lengkap. Unduhan mengembalikan byte XLSX/ZIP asli.

Source kode berasal dari [E21 GPU pada commit ef712b4](https://github.com/MalikHariyanto/Automated_Essay_Scoring/blob/ef712b4c035d96178d9baa67caa7453ac9fd8738/notebooks/experiments/AES_300_E21_Tiga_Input_GPU.ipynb). Pengaturan cocok dengan workbook. Notebook eksekusi/log mentah tidak disertakan, sehingga buku tidak mengarang output cell atau mengklaim source runtime/GPU cocok byte demi byte. Panel output per tahap membaca tabel aktual dari Excel. Buku tidak menjalankan training atau model.

Rujukan primer RoBERTa, BGE-M3 dan stabilitas fine-tuning ditautkan dengan batas penerapannya. Sumber tersebut mendasari komponen, bukan bukti bahwa resep E21 pasti unggul pada dataset ini. Test lama telah dilihat dalam pengembangan; hasil lanjutan bersifat eksploratif.

## Privasi dan validasi

Snapshot/preview publik hanya berisi agregat, source dan inventaris. File berisi teks, ID/nilai individu, vektor dan ZIP asli berada di buku terenkripsi. Kunci sama dengan buku sebelumnya dan tidak diterbitkan. Paket baru memakai nama Belajar_AES_E21_Lengkap agar tidak menimpa buku lama.

Tujuh pengujian integritas/unit lulus (0 failed / 0 skipped / 0 xfailed), termasuk provenance source, pemetaan catatan ke baris aktual, epoch/CV, unit error, privasi agregat dan kompatibilitas nama paket lama/baru. Sepuluh kelompok browser memeriksa angka180 prediksi, semua teks, ranking,15 fold/grafik, peta, seluruh sheet/unduhan byte-identik, pagination/komponen,16 bab,30 blok/fullscreen dan layar390/768/1440. Pemeriksaan privasi mencocokkan ZIP asli dan lima workbook serta memeriksa714 prefix teks privat terhadap file publik tanpa menemukan kecocokan. Bukti rinci lokal ada di `outputs/e21_study/`.

Dua pemeriksaan paket memverifikasi bahwa HTML pengunduh menghasilkan ZIP byte-identik dan buku terenkripsi dapat dibuka dengan kunci lama, menampilkan60jawaban/180prediksi E21 setelah jaringan diblokir. Ini pengujian buku/audit hasil tersimpan, bukan training baru.
