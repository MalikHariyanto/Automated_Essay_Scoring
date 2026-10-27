# Tampilan terbaru E24 dan pembersihan aplikasi belajar

Permintaan 28 Oktober 2026: sederhanakan duel, perjelas kurva, rapikan peringkat, jadikan kode E24 sebagai meja belajar utama dan sisakan dua aplikasi HTML terbaru. Ini perubahan antarmuka serta katalog; model, skor guru, prediksi dan hasil eksperimen tidak dilatih atau dihitung ulang untuk mengubah kesimpulan.

## Cara membaca yang baru

- **Duel langsung:** tiga kartu bernama E21 vs E22, E21 vs E24, E22 vs E24. Setiap kartu menyebut eksperimen, jumlah jawaban lebih dekat, jumlah seri, batang proporsi berwarna, dan kalimat kesimpulan. Default S3, pilihan S1/S2 tetap tersedia. Saat semua skenario dipilih, tiap skenario dipisah; total 180 perbandingan disimpan dalam rincian, bukan dianggap 180 siswa.
- **Kurva:** sumbu epoch dan unit RMSE/MSE eksplisit, angka sumbu mudah dibaca, skala sama untuk semua garis, warna dan pola garis berbeda. Legenda menyebut epoch pilihan beserta RMSE validasinya. Garis dapat ditampilkan/disembunyikan. Pemeriksa epoch dengan slider dan tombol panah menampilkan nilai tepat ketiga eksperimen, serta status saat suatu run sudah berhenti. Titik kuning tetap epoch pilihan berdasarkan validasi, bukan test. Grafik dapat digeser horizontal di HP tanpa memampatkan teks.
- **Hasil fold:** lima baris untuk skenario terpilih; eksperimen menjadi kolom. Angka RMSE disorot, epoch pilihan menjadi keterangan sekunder yang netral. Rerata dan SD tetap dipisahkan.
- **Peringkat:** kartu ringkas dengan posisi, nama eksperimen/metode, jumlah terdekat, batang proporsi, RMSE, MAE dan jumlah jawaban dengan selisih maksimal lima poin. Pengguna dapat mengurutkan menurut hitungan terdekat, RMSE atau MAE. Pembagian hasil seri tetap dijelaskan dan dihitung dengan benar; skor/prediksi tidak berubah ketika peserta perbandingan bertambah.
- **Meja belajar:** tombol utama dan pilihan awal sekarang E24. Kode tetap 30 blok/2.780 baris dari notebook E24 yang dipin pada `f3d25c4`, dengan catatan attention. E21/E22 tetap dapat dipilih dalam buku yang sama. Fullscreen menutup seluruh viewport tanpa heading induk; mode per blok dan seluruh kode tetap ada.
- Heading hasil diperpendek agar memberi lebih banyak ruang bagi tabel, grafik dan peringkat. Tema Spider-Man, 15 Excel, seluruh 60 jawaban dan output asli dipertahankan.

## Dua buku yang dipertahankan

| Buku | HTML publik | Pengunduh lengkap |
| --- | --- | --- |
| Eksperimen sebelum tiga variabel | [MLP / E6–E8](belajar_MLP_Spider.html) | [Unduh](../downloads/Belajar_AES_MLP_Lengkap.html) |
| Tiga variabel E21–E22–E24 | [E24 sebagai kode utama](belajar_Tiga_Input_Spider.html) | [Unduh](../downloads/Belajar_AES_Tiga_Input_Lengkap.html) |

Dihapus dari versi aktif GitHub: HTML E21/E22/E24 terpisah; tiga varian buku Colab lama; dua keluaran pustaka paper; panduan interaktif awal; ZIP HTML paper lama; pengunduh/paket E21 lama; HTML panduan E6 terpisah. Semuanya sudah digantikan dua buku di atas. Salinan notebook salah direktori `aes-learning/aes-learning/notebooks/03_AES_dataset_guru_300.ipynb` juga dihapus. File lama tetap dapat dipulihkan dari commit `b21e8b61f54cbe8f9f1a0deb4096a466b7b6aa61`; riwayat Git tidak ditulis ulang.

Notebook metode yang berbeda dan varian CPU/GPU tetap tersedia melalui [katalog eksperimen](../notebooks/experiments/README.md). Salinan kompatibilitas pada `aes-learning/notebooks` dipertahankan karena tautan Colab lama masih menggunakannya. README utama dan folder unduhan sekarang hanya menampilkan dua aplikasi terbaru. Source pembentuk buku historis tetap disimpan untuk penelusuran, bukan dijadikan aplikasi unduhan tambahan.

## Verifikasi

14 pengujian integritas/unit lulus. Pemeriksaan browser mencocokkan angka duel, kurva epoch, prediksi, peringkat dan Excel terhadap sumber, memeriksa default E24, desktop/HP, keyboard, scroll, fullscreen dan penggunaan tanpa jaringan. Pengujian paket memeriksa rekonstruksi ZIP dan pembukaan dengan kunci lama. Rincian command, versi, hasil dan hash disimpan lokal di `outputs/e24_ui_refresh/`. Tidak ada training/evaluasi model baru pada tugas antarmuka ini.

Data siswa, ZIP/XLSX asli dan kunci tidak diterbitkan dalam bentuk terbuka. Payload lengkap tetap terenkripsi; ringkasan publik hanya menyertakan agregat dan source. Audit isi penelitian tetap mengikuti [review hasil E24](review_e24_study.md).
