# Automated Essay Scoring · mulai di sini

Versi aktif memakai **tiga teks**—soal, kunci jawaban, respons siswa—dan **satu label `rater` (0–100)**. Notebook berjalan di Colab; HTML untuk belajar dan membaca hasil.

**Terbaru: pasangan E55/E56.** [Colab E55 · MSE](https://colab.research.google.com/github/MalikHariyanto/Automated_Essay_Scoring/blob/codex/aes-learning/notebooks/experiments/AES_300_E55_E43_S3_PairedBatch_MSE_Native_Learnable_GPU.ipynb) · [Colab E56 · ConR](https://colab.research.google.com/github/MalikHariyanto/Automated_Essay_Scoring/blob/codex/aes-learning/notebooks/experiments/AES_300_E56_E43_S3_PairedBatch_ConR_Native_Learnable_GPU.ipynb). Batch training sama, mendahulukan pasangan soal sama dengan nilai dekat/jauh; setiap jawaban sekali per epoch, split/label/model/LR tetap. E55 MSE saja (ConR diagnostik koefisien0), E56 MSE+0,001×ConR. Tidak menyimpan progres per epoch; model disimpan setelah fold/refit selesai. Jika putus di tengah fit, fit diulang. Cukup Excel; jalankan di dua runtime GPU terpisah. Belum ada hasil penelitian E55/E56.

**E54 dari E43:** [Colab ConR](https://colab.research.google.com/github/MalikHariyanto/Automated_Essay_Scoring/blob/codex/aes-learning/notebooks/experiments/AES_300_E54_E43_S3_ConR_Native_Learnable_GPU.ipynb) · [Source GitHub](notebooks/experiments/AES_300_E54_E43_S3_ConR_Native_Learnable_GPU.ipynb). Arsitektur/LR E43 tetap; loss `MSE + 0,001 × ConR` pada fitur fusion, pasangan soal sama/window5poin. Tanpa EMA, LDS, DoRA atau model baru. Cukup Excel, S3/5-fold/maks25epoch/patience5/satu refit-test. Rumus, referensi, pengaman anchor kosong, resume dan diagnostik cakupan ada di notebook. Belum ada hasil penelitian E54.

**E53 dari E43:** [Colab EMA](https://colab.research.google.com/github/MalikHariyanto/Automated_Essay_Scoring/blob/codex/aes-learning/notebooks/experiments/AES_300_E53_E43_S3_EMA099_Native_Learnable_GPU.ipynb). Full fine-tuning RoBERTa kembali LR5e-6; salinan rata-rata bobot EMA untuk validasi/inferensi, batas decay0,99 dengan warmup. Arsitektur, MSE dan LR lainnya tetap E43. S3, satu 5-fold, maksimal25epoch/patience5; cukup Excel. Laporan online/EMA per epoch dan resume lengkap. [Metode dan paper](docs/eksperimen/eksperimen_e53_ema_native_gpu.md). Hasil E53 yang diaudit: mean RMSE CV5,8627, RMSE test5,9063 dan MAE test4,6625; E43 tetap kontrol resep.

**E52 dari E43:** [Colab DoRA](https://colab.research.google.com/github/MalikHariyanto/Automated_Essay_Scoring/blob/codex/aes-learning/notebooks/experiments/AES_300_E52_E43_S3_DoRA8_Native_Learnable_GPU.ipynb). RoBERTa memakai adapter DoRA rank 8/alpha 16/query-value, LR 1e-4; bobot dasarnya frozen. BiGRU, co-attention satu head, ScalarMix/gamma, masked mean, BGE, concat dan MLP E43 tetap. S3, 5-fold, maksimal 25 epoch/patience 5; cukup Excel. [Metode dan paper](docs/eksperimen/eksperimen_e52_dora_native_gpu.md). Belum ada hasil penelitian E52.

**E51 dari E50:** [Colab gated concatenation](https://colab.research.google.com/github/MalikHariyanto/Automated_Essay_Scoring/blob/codex/aes-learning/notebooks/experiments/AES_300_E51_E50_S3_GatedConcat_Native_Learnable_GPU.ipynb). Gate per jawaban dengan dua alpha, kondisi awal identik E50, tambahan 2.050 parameter. MSE dan multihead 4 tetap; cukup Excel; S3, 5-fold, 25 epoch/patience 5. Bobot gate OOF/test tersimpan bersama prediksi. [Metode dan paper](docs/eksperimen/eksperimen_e51_gated_concat_native_gpu.md). Belum ada hasil penelitian E51.

**E50 dari E43:** [Colab multi-head co-attention4](https://colab.research.google.com/github/MalikHariyanto/Automated_Essay_Scoring/blob/codex/aes-learning/notebooks/experiments/AES_300_E50_E43_S3_MultiHeadCoAttention4_Native_Learnable_GPU.ipynb). BiGRU/ScalarMix/gamma/BGE/fusiontetap,MSEbiasa,tanpaLDS. CukupExcel;S3,5fold,25epoch/patience5. [Metode dan paper](docs/eksperimen/eksperimen_e50_multihead_coattention_native_gpu.md). BelumadahasilpenelitianE50.

**E49 dari E43/E48:** [Colab LDS inverse akar](https://colab.research.google.com/github/MalikHariyanto/Automated_Essay_Scoring/blob/codex/aes-learning/notebooks/experiments/AES_300_E49_E43_S3_LDS_InverseSqrt_Native_Learnable_GPU.ipynb). Pangkat inverse setelah smoothing1→0,5;model/protokol tetap. Cukup datasetExcel;S3,5fold,25epoch/patience5. [Metode dan paper](docs/eksperimen/eksperimen_e49_lds_inverse_sqrt_native_gpu.md). Belum ada hasilpenelitianE49;E43tetapacuan.

**E48 dari acuan E43:** [Colab LDS-weighted MSE](https://colab.research.google.com/github/MalikHariyanto/Automated_Essay_Scoring/blob/codex/aes-learning/notebooks/experiments/AES_300_E48_E43_S3_LDS_WeightedMSE_Native_Learnable_GPU.ipynb). Arsitektur tetap, hanya loss dibobot menurut distribusi skor training fold. S3,5fold,25epoch/patience5, cukup upload dataset Excel. Output OOF, rincian selisih test dan audit bobot LDS. [Metode dan paper](docs/eksperimen/eksperimen_e48_lds_native_gpu.md). Belum ada hasil penelitian E48.

**Prediksi lengkap E46/E47:** Excel04 kini memisahkan skor mentah, normalisasi, pembulatan, serta jumlah/persentase beda tepat 0/1/2/3/4/5/10 poin dan toleransi ≤n. Jika sudah punya ZIP hasil lengkap, gunakan [Colab pembaru ZIP tanpa training](https://colab.research.google.com/github/MalikHariyanto/Automated_Essay_Scoring/blob/codex/aes-learning/notebooks/05_Perbarui_Prediksi_Hasil_Tanpa_Training.ipynb). [Cara membaca](docs/eksperimen/eksperimen_e46_e47_refit_schedule_native_gpu.md#rincian-prediksi-tanpa-mengubah-training).

**Diagnostik terbaru E46/E47:** [E46 Colab](https://colab.research.google.com/github/MalikHariyanto/Automated_Essay_Scoring/blob/codex/aes-learning/notebooks/experiments/AES_300_E46_E43_S3_Refit18_Schedule25_Native_Learnable_GPU.ipynb) memakaiE43/refit18; [E47 Colab](https://colab.research.google.com/github/MalikHariyanto/Automated_Essay_Scoring/blob/codex/aes-learning/notebooks/experiments/AES_300_E47_E45_S3_Refit7_Schedule25_Native_Learnable_GPU.ipynb) memakaiE45/refit7. Keduanya horizonLR25,memakaiulang5CVterverifikasi,tanpaCVbaru. Uploaddataset+ZIPhasilacuan. [Metode/input/pengaman unduhan](docs/eksperimen/eksperimen_e46_e47_refit_schedule_native_gpu.md). Belum ada hasilpenelitian.

**Kandidat kedua E45:** [Colab Scalar Mix seluruh12layer dari E43](https://colab.research.google.com/github/MalikHariyanto/Automated_Essay_Scoring/blob/codex/aes-learning/notebooks/experiments/AES_300_E45_E43_S3_ScalarMix12_Native_Learnable_GPU.ipynb), loss tetapMSE,S3saja,5-fold/25epoch/patience5. [Metode dan paper](docs/eksperimen/eksperimen_e45_scalar_mix12_native_gpu.md). Belum ada hasil penelitianE45.

**Kandidat E44:** [Colab MSE + ListNet dari E43](https://colab.research.google.com/github/MalikHariyanto/Automated_Essay_Scoring/blob/codex/aes-learning/notebooks/experiments/AES_300_E44_E43_S3_MSE_ListNet_Native_Learnable_GPU.ipynb), ranking hanya jawaban untuk soal yang sama, S3 saja, 5-fold/25 epoch/patience5. [Metode dan paper](docs/eksperimen/eksperimen_e44_listnet_native_gpu.md). E43 tetap acuan; belum ada hasil penelitian E44.

**Acuan aktif E43:** [Colab masked mean pooling dari E42](https://colab.research.google.com/github/MalikHariyanto/Automated_Essay_Scoring/blob/codex/aes-learning/notebooks/experiments/AES_300_E43_E42_S3_MaskedMeanPooling_Native_Learnable_GPU.ipynb), S3 saja, 5-fold/25 epoch/patience 5, format Python biasa. [Metode dan paper](docs/eksperimen/eksperimen_e43_masked_mean_native_gpu.md). Hasil yang diaudit: mean RMSE CV5,8876, RMSE test5,1265, MAE test3,7817; tabel hasil historis di bawah tetap dipertahankan.

| Tujuan | Mulai dari |
| --- | --- |
| Melanjutkan dari akun/chat baru | **[Panduan untuk AI](PANDUAN_AI_PROYEK.md)**, [pindah akun](PINDAH_AKUN.md), [status proyek](STATUS_PROYEK.md) |
| Menjalankan notebook | **[Katalog eksperimen](notebooks/experiments/README.md)** |
| Membaca hasil lengkap | **[Dua HTML terbaru](downloads/README.md)** |
| Menentukan combo berikutnya | **[Rencana eksperimen](RENCANA_EKSPERIMEN.md)** |
| Mencari metode, review, atau bukti | **[Daftar dokumentasi](docs/README.md)** |

**Hasil E27/E28 sudah masuk buku tiga input.** [E27 · E23 + co-attention](notebooks/experiments/AES_300_E27_E23_BiGRU_CoAttention_GPU.ipynb), [E28 · E23 + LR MLP S3 5e-6](notebooks/experiments/AES_300_E28_E23_BiGRU_MLPLR5e6_GPU.ipynb). [Review hasil dan preservasi isi](docs/review/review_e27_e28_study.md).

Kode baru dari **E26**: **[E29 · LR MLP S2/S3 5e-6](notebooks/experiments/AES_300_E29_E26_BiGRU_Attention_MLPLR5e6_GPU.ipynb)** dan **[E30 · co-attention lalu attention pooling](notebooks/experiments/AES_300_E30_E26_BiGRU_CoAttention_Pooling_GPU.ipynb)**. GPU, tanpa Drive; hasil E29 dan E30 sudah tersedia. [Panduan](docs/eksperimen/eksperimen_e29_e30_gpu.md).

**Kode baru berbasis E25:** [E31/E32](docs/eksperimen/eksperimen_e31_e32_gpu.md), GPU tanpa Drive; belum ada hasil.

## Hasil terbaru: E21–E32

Dua belas eksperimen,36model,60jawaban test yang sama,60Excel/557sheet. Semua hasil lama dipertahankan; hanya tiga kode diembed agar buku lebih ringan.

| Ukuran S3 | Hasil unggahan saat ini |
| --- | --- |
| RMSE test terendah | **E29 · E26 + LR MLP S2/S3 5e-6: 5,4293 poin** |
| Rerata RMSE lima fold terendah | **E29: 5,7667 poin** |
| Paling sering menjadi prediksi terdekat | **E30:11/60**, pada dua belas peserta; bukan jumlah skor persis sama |
| Terbanyak selisih maksimal 5 poin | **E30:44/60** |

Kode utama HTML kini **E29**. Atas instruksi pengguna, meja belajar hanya memuat **E29, E30, E26**, tiga RMSE test S3 terendah. Semua hasil/Excel tetap lengkap di tab08. S2 E29 justru memburuk ke RMSE75,9702; hasil itu tidak disembunyikan. Perubahan peserta dapat mengubah jumlah terdekat tanpa mengubah prediksi lama. [Review terbaru](docs/review/review_e30_study.md).

## Dua buku yang dipakai

| Buku | Isi | Unduh lengkap | Preview publik |
| --- | --- | --- | --- |
| **Tiga input · E21–E32** | Perbandingan, jawaban/prediksi, kurva, Excel, kode dan sticky note | **[Pengunduh HTML](downloads/Belajar_AES_Tiga_Input_Lengkap.html)** | [Ringkasan](docs/belajar_Tiga_Input_Spider.html) |
| **Sebelum tiga input · sampai E8/E6 2.0** | Hasil dan kode eksperimen sebelumnya | **[Pengunduh HTML](downloads/Belajar_AES_MLP_Lengkap.html)** | [Ringkasan](docs/belajar_MLP_Spider.html) |

Di GitHub pilih **Download raw file**, buka pengunduh, unduh dan ekstrak ZIP, lalu gunakan **kunci lama**. Kunci dibawa terpisah. Buku lengkap bekerja offline; preview publik hanya memuat agregat dan kode.

## Peta folder

```text
notebooks/experiments/   Notebook eksperimen; mulai dari katalognya
notebooks/archive/      Varian lama yang dipertahankan untuk penelusuran
downloads/              Dua pengunduh dan bagian paket terenkripsi
docs/                  Metode, review, verifikasi, snapshot, arsip
scripts/                Pembuat notebook/HTML, helper, pemeriksa
src/                    Modul Python; sebagian workflow versi terdahulu
tests/                  Pengujian implementasi
aes-learning/           Salinan notebook untuk tautan Colab lama
configs/ dan schemas/   Workflow empat trait terdahulu
```

Dataset, bobot, checkpoint, kunci, dan `outputs/` bersifat lokal; tidak ikut clone. [Berkas yang perlu dibawa](PINDAH_AKUN.md#berkas-yang-perlu-dibawa).

Workflow empat trait dan latihan berlabel AI ada sebagai **arsip terpisah**. Jangan memakai konfigurasinya untuk mengubah eksperimen `rater` secara otomatis. [Kontrak aktif](STATUS_PROYEK.md#kontrak-aktif).

## Hasil E31/E32 ditambahkan

Buku tiga input kini mencakup E21–E32:12hasil,36model,60Excel/557sheet. E31 S3 RMSE6,5735; E32 S3 RMSE5,8059. Tiga kode tetap E29/E30/E26, utama E29. [Review E31/E32](docs/review/review_e31_e32_study.md).
