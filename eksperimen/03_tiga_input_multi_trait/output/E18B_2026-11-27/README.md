# E18B · E13 dengan setelan Trial 60

Arsip ini berisi hasil agregat validasi E18B dari ZIP 04–05. Ukuran dan SHA-256 workbook diperiksa terhadap manifest. Perbandingan E13 dan E18B mencocokkan 1.070 ID jawaban unik, kelompok, label guru, dan fold. Workbook privat dan prediksi individual tetap di penyimpanan asli.

## Run

Lima fold S3 dengan arsitektur dan loss E13. Paket Trial 60 E17E menetapkan LR RoBERTa 1,6407e-5, LR proyeksi/head/layer lain 3,3453e-5, dropout head 0,30, weight decay 0,03, dan warmup 0,10. Seed dasar 42 dengan seed per fold 43–47. Batas 25 epoch dan patience 5. Epoch terpilih 3, 7, 5, 3, 5. Durasi lima fit sekitar 61 menit. Eksperimen berakhir setelah validasi OOF, sebelum refit dan test.

| RMSE | E13 | E18B |
|---|---:|---:|
| Rata-rata lima fold | 0,422854 | 0,419344 |
| Gabungan seluruh prediksi OOF | 0,423691 | 0,419987 |

| Trait | RMSE OOF E13 | RMSE OOF E18B | QWK OOF E13 | QWK OOF E18B | Deteksi skor 1 E13 | Deteksi skor 1 E18B |
|---|---:|---:|---:|---:|---:|---:|
| Isi | 0,5888 | 0,5865 | 0,5961 | 0,5711 | 0/17 | 0/17 |
| Organisasi | 0,3208 | 0,3225 | 0,3722 | 0,2837 | 57/158 | 39/158 |
| Kebahasaan | 0,3687 | 0,3640 | 0,4596 | 0,4115 | 108/226 | 97/226 |
| Mekanik | 0,3640 | 0,3537 | 0,5111 | 0,4935 | 137/235 | 126/235 |

Di fold 4, RMSE organisasi turun dari 0,3527 ke 0,3383. Semua 210 kategori prediksi organisasi tetap sama, termasuk 4 dari 30 skor 1 yang dikenali. Pada semua fold, E18B menambah 18 kesalahan organisasi skor 1 menjadi 2, sementara kesalahan skor 2 menjadi 1 berkurang 14. Penurunan RMSE gabungan tidak diikuti perbaikan QWK pada keempat trait.

Bootstrap berpasangan 5.000 kali pada 107 kelompok validation menghasilkan interval deskriptif persentil 95% sekitar −0,0107 sampai +0,0033 untuk selisih RMSE OOF gabungan E18B − E13. Perhitungan memakai prediksi yang sudah tersedia. Interval ini tidak mengukur variasi pelatihan lintas seed.

## Keputusan

Simpan E18B sebagai pembanding validasi. Pertahankan E13 sebagai kontrol untuk cabang arsitektur E13 dan tahan refit/test E18B. E17C tetap dicatat sebagai pembanding concat dalam jalur aktif repo.

- [Metrik OOF lengkap](CV_OOF_per_Trait.csv)
- [Perbandingan RMSE per fold](Perbandingan_RMSE_Fold.csv)
- [Perbandingan kesalahan skor rendah](Perbandingan_Skor_Rendah.csv)
- [Waktu dan epoch tiap fold](Progres_Fold.csv)
- [Notebook E18B](../../notebooks/AES_Reset_E18B_E13_Trial60_Validasi.ipynb)
