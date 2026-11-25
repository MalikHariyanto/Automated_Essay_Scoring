# Output E17D — tiga input, multi-trait

## Status arsip

Delapan workbook unggahan cocok ukuran dan SHA-256 manifest. Ini hasil E17D GPU (Tesla T4), code ID E17D_CONCAT_REFIT_HORIZON25_V1, bukan varian CPU dan bukan Bayesian optimization.

Folder ini menyimpan hasil evaluasi/pelatihan (Excel 04–05), tabel agregat CSV, manifest asli, dan audit. Workbook 01–03 berisi teks/token/vektor siswa sehingga salinan lengkap tetap privat. Status pemulihan lengkap dijelaskan di [BACKUP.md](BACKUP.md); tautan privat diberikan langsung dalam percakapan. Jangan menganggap seluruh workbook sudah tersimpan sebagai blob GitHub.

## Perbandingan terhadap E17C

ID, label, kelompok, soal, dan keanggotaan fold prediksi OOF/test cocok; dataset dan split hash sama. Mean CV adalah rata-rata RMSE lima fold; pooled OOF/test dihitung dari seluruh residual empat trait dalam satuan asli. Macro QWK adalah rata-rata empat QWK, bukan akurasi atau QWK skor total.

| Metrik | E17C | E17D |
|---|---:|---:|
| Mean CV RMSE | 0.4227455593 | 0.4227325631 |
| Pooled OOF RMSE | 0.4235832739 | 0.4235873203 |
| Pooled test RMSE | 0.4661693834 | 0.4999359828 |
| Macro QWK test | 0.3470285051 | 0.2938439852 |

CV praktis sama; penurunan mean CV sekitar 0.0031% tidak cukup menjadi bukti improvement. Test RMSE meningkat sekitar 7.24%, dan QWK turun pada semua trait. OOF berbeda sedikit walaupun resep CV sama; run ini bukan pasangan checkpoint CV yang identik. Tidak dapat mengatribusi seluruh selisih test secara kausal hanya ke scheduler dari satu run.

| Trait test | RMSE E17C | RMSE E17D | QWK E17C | QWK E17D | Di luar skala E17D |
|---|---:|---:|---:|---:|---:|
| Isi | 0.672247 | 0.749793 | 0.411765 | 0.348363 | 26.30% |
| Organisasi | 0.321273 | 0.329554 | 0.364706 | 0.297398 | 46.67% |
| Kebahasaan | 0.416175 | 0.421575 | 0.266494 | 0.233363 | 14.07% |
| Mekanik | 0.375395 | 0.388874 | 0.345149 | 0.296253 | 41.11% |

Out-of-range dihitung sebelum clipping kategori; bukan persentase prediksi salah. Recall skor terendah test: isi 0/3, organisasi 10/35, kebahasaan 17/70, mekanik 18/56. Seluruhnya tidak menunjukkan kemajuan terhadap E17C.

## Diagnostik scheduler pada development

Pada outer holdout development 220 jawaban, training 850 jawaban, inner selection memilih satu epoch. Duration-horizon RMSE 0.4633601201; fixed-horizon25 0.4751229945 (lebih buruk sekitar 2.54%). Warmup 10 vs 267 langkah. Hasil ini satu split/seed dan durasi satu epoch; jangan digeneralisasi sebagai bukti seluruh jadwal fixed selalu buruk. Namun hipotesis keunggulan horizon25 belum didukung bahkan sebelum melihat test.

Best epoch lima fold 3,5,4,3,4; refit empat epoch, horizon25, warmup335 langkah,536 langkah total terpakai. LR RoBERTa akhir4.6666667e-6, bukan nol. Semua fit utama mencatat nol AMP skipped steps. MSE training turun sementara validation memburuk setelah best epoch: indikasi overfitting, bukan bukti penyebab tunggal.

## Keputusan eksperimen

Jangan promosikan E17D sebagai baseline yang lebih baik. Pertahankan E17C sebagai kontrol perbandingan scheduler; jangan mengklaim E17C sudah terbaik dibanding seluruh eksperimen. Untuk pencarian hyperparameter selanjutnya, tetapkan kebijakan scheduler, objective, group split, dan budget sebelum tuning. Seleksi hanya development; test yang telah berulang kali dilihat bukan konfirmasi independen baru.

## Pemeriksaan

SHA-256 dan ukuran delapan workbook cocok manifest. RMSE, Pearson, QWK, dan proporsi keluar skala per trait dihitung ulang dari prediksi dan cocok toleransi1e-10; Gwet AC2 disalin dari workbook, belum dihitung ulang independen. Pooled metrics dihitung ulang. Tidak menjalankan training baru. CSV mempertahankan nilai sumber; workbook 04–05 asli tidak diubah. Manifest menjelaskan sumber lengkap, bukan klaim semua file ada sebagai blob publik.
