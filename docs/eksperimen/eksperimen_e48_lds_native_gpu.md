# E48 · E43 + LDS-weighted MSE

[Buka Colab E48](https://colab.research.google.com/github/MalikHariyanto/Automated_Essay_Scoring/blob/codex/aes-learning/notebooks/experiments/AES_300_E48_E43_S3_LDS_WeightedMSE_Native_Learnable_GPU.ipynb) · [source](../../notebooks/experiments/AES_300_E48_E43_S3_LDS_WeightedMSE_Native_Learnable_GPU.ipynb).

Pilih **GPU T4 → Run all → upload `fix_dataset_300.xlsx`**. Cukup satu Excel; tidak membutuhkan ZIP hasil eksperimen lama. Notebook mandiri, Python biasa per section tanpa magic atau worker. Belum ada hasil penelitian E48.

## Perubahan yang diuji

Pengguna mempertahankan arsitektur E43 dan menyetujui pengujian LDS-weighted MSE. Scalar Mix4+gamma, BiGRU64perarah, co-attention dua arah/matching, masked mean, RoBERTa fine-tuned, BGE frozen, concat dan MLP tetap. Tidak menggunakan blok Transformer pengganti, ListNet, cosine atau S1/S2.

| Bagian | Nilai |
|---|---|
| X | `soal`, `jawaban_referensi`, `response_text` |
| y | `rater`, integer0–100, label guru asli |
| LR | RoBERTa5e-6; MLP/proyeksi/pooling_norm/ScalarMix1e-4 |
| Training | S3; satu5foldCV baru; maksimal25epoch/patience5 |
| Refit | Ceil median epoch terbaik lima fold, satu refit seluruh240development |
| Jadwal | E43 asli: warmup10%, linear decay mengikuti durasi fit masing-masing |
| Seed | Satu konstanta42, resep fold seed+nomor fold; tidak ada sweep seed/kontrol ulang |
| Test | Satu evaluasi60held-out setelah pemilihan epoch/refit |
| LDS | Bin1poin; Gaussian5/sigma2poin; inverse setelah smoothing; mean bobot training=1 |

Ini kembali ke jadwal E43, bukan horizon refit tetap25 dari E46/E47. Mean RMSE CV acuan E43=5,8876 dan RMSEtest=5,1265 adalah hasil historis, bukan hasil E48. E46/E47 tidak memperbaiki hasil pada run pengguna. Test historis yang sudah berulang kali dilihat tetap eksploratif.

## Mekanisme LDS dan batas data

Untuk tiap fit, hitung histogram **hanya dari label bagian training** pada101bin0–100. Konvolusi Gaussian berukuran5/sigma2 dengan zero padding menghasilkan frekuensi efektif. Untuk jawaban i, `a_i=1/frekuensi_efektif(skor_i)`, lalu `w_i=a_i/mean(a pada seluruh training fit)`.

Objective adalah `mean(w_i * (prediksi_i − target_i)^2)`, dengan target0–1. Bin ditentukan dari skor asli0–100. Label tidak dimodifikasi/dihaluskan; yang dihaluskan adalah histogram frekuensinya. Bobot tidak menjadi input teks/encoder dan tidak ditambahkan ke hasil prediksi. Pada inferensi tidak diperlukan label/bobot LDS.

Bobot dihitung ulang pada masing-masing train fold (~192jawaban);48labelvalidasi tidak berkontribusi. Saat refit, bobot dihitung dari240development, tanpa60test. Tidak ada estimasi distribusi global sebelumCV, sampler baru, balancing label validasi/test atau modifikasi skor guru. Kelompok skor jarang bisa mengandung label sulit/noisy; pembobotan belum tentu memperbaiki hasil.

**Tidak menormalisasi ulang bobot per microbatch.** Dengan microbatch1, membagi loss dengan jumlah bobot microbatch akan membatalkan LDS. Implementasi memakai bobot mean1 dari keseluruhan training dan mengakumulasi gradien menurut ukuran batch efektif8, termasuk batch terakhir yang lebih kecil. Satu forward menghasilkan weighted loss untuk backward dan MSE biasa untuk log.

Checkpoint/resume merekam fingerprint indeks training, label/bobot/kernel LDS dan seluruhconfig; bobot dipulihkan bersama optimizer/scheduler/scaler/RNG/loader. Namespace, full-source hash dan lesson_version E48 terpisah dari E43, sehingga cache lama tidak dianggap hasil E48. Arsitektur/model tidak menambah parameter.

## Pemilihan dan laporan

Early stopping tetap menggunakan **RMSE validasi tanpa bobot LDS**. Evaluasi RMSE/Pearson mentah dan QWK/AC2 kategori juga tanpa bobot, sehingga aturan evaluasi tetap sebanding dengan E43. MAE, MSE, kesalahan terbesar dan cakupan±3/±5 adalah diagnostik tambahan; target60/60 belum dijamin.

Lima Excel tersimpan dalam satuZIP:

- **04→Nilai Guru dan Prediksi:** semua test dengan mentah0–100, normalisasi0–1, pembulatan dan selisih. **Selisih Tepat:** =0/1/2/3/4/5/10 serta lainnya. **Toleransi Selisih:** ≤0/1/2/3/4/5/10, jumlah/persentase mentah dan bulat. **Ringkasan Error Test:** MSE/MAE/error maksimum.
- **04→Prediksi Validasi OOF:**240development, tiap jawaban diprediksi model fold yang tidak melatihnya. OOF terpisah dari60test; tersedia soal, guru, prediksi, kategori, error.
- **05→Fusion(S3):** history lima fold dan refit, MSE biasa serta LDS-MSE terpisah. **MAE dan Toleransi CV:** per-fold/OOF, termasuk±3,±5,error maksimum.
- **05→Bobot LDS Training:** tahap/fold,ID,skor,bobot tiap jawaban training; total1.200baris untuk300data. **Distribusi LDS Training:**101bin×6fit=606baris. **Audit LDS:** jumlah training,mean/min/max,parameter,fingerprint.
- **05→Bobot Scalar Mix:**4bobot+gamma pada setiap checkpoint fold/refit;24baris. **01/02/03** mempertahankan data/tokenisasi, vektor jawaban dan kata/kalimat.

Generator laporan mengikat file sumber agar loopOOF tidak membuat ringkasan test/CV kosong seperti bug E44/E45. Sebelum ZIP dibuat, gate wajib membaca ulang Excel, menghitung ulang metrik test/OOF, ID/split/label, meanSDCV, rekap selisih, lossfinite dan setiap bobot LDS dari training fold yang sesuai. Kegagalan menghentikan pembuatan ZIP baru. Pemeriksaan ini tidak menjamin seluruh kemungkinan bug mustahil terjadi.

## Rujukan

Yang,Y.,Zha,K.,Chen,Y.-C.,Wang,H.,Katabi,D.(2021), **[Delving into Deep Imbalanced Regression](https://arxiv.org/abs/2102.09554)**, ICML2021. [Repositori penulis](https://github.com/YyzHarry/imbalanced-regression), [contoh LDS untuk dataset/model lain](https://github.com/YyzHarry/imbalanced-regression#lds), [STS-B-DIR](https://github.com/YyzHarry/imbalanced-regression/tree/main/sts-b-dir).

Implementasi E48 mengikuti resep umum histogram→Gaussian smoothing→inverse weighting di README penulis, dengan mean bobot1. Bukan reproduksi seluruh parameter/benchmark STS-B; sebagian pipeline benchmark lain memakai clipping frekuensi dan preprocessing berbeda. Bin1poin adalah adaptasi skala AES. Gaussian5/sigma2 adalah contoh resmi, bukan bukti optimum pada300data Indonesia. Rujukan komponen arsitektur tetap mengikuti [E43](eksperimen_e43_masked_mean_native_gpu.md).
