# E8 pada buku hasil AES

E8 menambahkan hasil konteks pair S3 + LR encoder 5e-6 ke buku Spider-Man. Meja belajar dan acuan kanan perbandingan 1vs1 tetap E6. E7 tiga seed belum memiliki hasil dan tidak diisi angka pengganti.

Lima workbook unggahan diperiksa terhadap 300 data, pembagian development/test, anggota lima fold, serta 60 ID/skor guru test yang sama dengan MLP awal. Empat metrik test dihitung ulang dari prediksi mentah untuk verifikasi; metrik CV dicocokkan dengan epoch terpilih pada riwayat. Pasangan prediksi tiap fold tidak ikut ZIP, sehingga metrik CV tidak diklaim dihitung ulang dari pasangan fold.

| S3 | Konteks pair E4 | LR 5e-6 E6 | Kombinasi E8 |
| --- | ---: | ---: | ---: |
| Rerata RMSE validasi | 6,0986 | 5,9915 | 5,8836 |
| SD antar-fold | 1,0025 | 0,8346 | 0,7559 |
| RMSE test | 6,7315 | 6,3744 | 6,3818 |

E8 memperkecil rerata RMSE CV pada run ini. RMSE test E8 lebih besar sekitar 0,0074 poin daripada E6; angka itu belum membuktikan perbedaan bermakna. Pearson/QWK E6 lebih tinggi, AC2 E8 lebih tinggi. Pada 60 jawaban S3, 28 prediksi lebih dekat pada E8 dan 32 lebih dekat pada E6. Jumlah terdekat berbeda dari besar error yang dihitung RMSE.

S1/S2 E8 mempunyai prediksi berbeda dari kontrol E6. Source E8 melatih ketiga skenario; E6 memakai ulang kontrol. Notebook eksekusi tidak disertakan: source rujukan GPU dipin pada commit `4f45615`, tetapi nama ZIP tidak membuktikan perangkat runtime atau kecocokan kode eksekusi byte demi byte. Tidak ada klaim bahwa LR/konteks adalah satu-satunya penyebab selisih.

Buku menampilkan tujuh kolom eksperimen, 105 hasil fold, 60 jawaban dengan 420 prediksi per skenario, sorotan/peringkat yang dihitung ulang atas tujuh run, sumber implementasi E8, dan seluruh sheet lima Excel E8. E8 juga tersedia pada perbandingan 1vs1, grafik riwayat, unduhan XLSX asli dan mode layar penuh. Sumber MLP awal dan seluruh arsip lama dipertahankan.

Pemeriksaan: 35 pengujian integritas/unit lulus; 11 kelompok regresi browser dan lima kelompok khusus E8 lulus. Nilai E8 pada ketiga skenario dicocokkan dengan sumber mentah, XLSX unduhan byte-identik, dan navigasi/fullscreen diperiksa pada layar 390/768/1440. Dua pemeriksaan paket browser memverifikasi penyusunan ZIP dan pembukaan buku terenkripsi dengan kunci lama. Pemeriksaan privat berikutnya berjalan setelah jaringan diblokir. Ini pemeriksaan buku dan hasil tersimpan, bukan training model baru.

Source publik hanya memuat kode, agregat dan pratinjau agregat. ZIP, teks, nilai individual dan HTML lengkap berada dalam payload terenkripsi; kunci tidak diterbitkan di repository. Bukti lokal: `outputs/mlp_study/e8_*`.
