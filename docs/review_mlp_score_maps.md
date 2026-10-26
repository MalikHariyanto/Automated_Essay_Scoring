# Akses peta guru × prediksi

Peta lama tetap tersedia di Meja belajar E6 → CV → refit → test → Alur & cara baca. Buku kumpulan eksperimen kini juga menampilkannya langsung pada Ringkasan semua eksperimen → Guru & prediksi → Peta guru × prediksi. Pembaca dapat memilih S1/S2/S3, satu eksperimen, atau membandingkan tujuh eksperimen sekaligus. Peta lama memiliki tombol langsung menuju tampilan baru.

Terdapat 21 matriks agregat (tujuh eksperimen × tiga skenario), masing-masing berisi 60 jawaban test. Baris adalah rentang nilai guru; kolom adalah rentang prediksi kategori. Angka berarti jumlah jawaban, titik berarti nol. Diagonal berarti rentang sama, bukan skor persis sama. Klik kotak berisi angka menjelaskan rentang dan arah perbedaannya. Ketika dibandingkan, seluruh peta pada skenario yang sama memakai satu skala warna bersama. Kelompok 10 poin hanya visualisasi, bukan perubahan kategori evaluator QWK/AC2.

Angka diturunkan dari Excel asli dengan pemeriksaan 60 ID/skor guru yang sama. E2/E6 cocok persis dengan matriks yang sudah ada. E2–E8 memakai kolom pembulatan Excel yang diverifikasi terhadap aturan evaluator. Workbook evaluasi E0 tidak mempunyai kolom pembulatan; kategorinya diturunkan dari prediksi mentah, clip 0–100, kategori terdekat, tie .5 ke bawah. RMSE/Pearson, peringkat dan prediksi sumber tidak diubah. Source publik hanya menyimpan hitungan agregat dan hash sumber.

Jumlah eksperimen pada header kini dibentuk dari katalog sumber sehingga tidak tertinggal pada angka enam setelah E8 ditambahkan. E6 tetap acuan meja belajar dan perbandingan 1vs1. HTML belajar kode E6 yang terpisah tidak diubah.

Verifikasi: 40 pengujian unit/integritas lulus, termasuk batas rentang, clipping, pembulatan tie, penolakan data tidak sah, serta pelestarian matriks lama. Enam kelompok browser khusus memeriksa akses menu, seluruh 21 matriks/121 kotak, klik penjelasan, skala warna bersama, layar 390/768/1440 dan fullscreen, serta tautan dari meja belajar. Sebelas kelompok regresi memeriksa navigasi, kode E6, tujuh eksperimen, 420 prediksi dan unduhan XLSX. Dua kelompok paket browser memeriksa ZIP byte-identik, pembukaan terenkripsi dengan kunci lama, prediksi dan peta bersama setelah jaringan diblokir.

Tidak ada training atau prediksi ulang model. Perubahan ini menyajikan hasil yang sudah tersimpan. Bukti lokal: `outputs/mlp_study/score_maps_*`.
