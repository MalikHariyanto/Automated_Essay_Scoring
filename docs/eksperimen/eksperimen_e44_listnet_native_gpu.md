# E44 · E43 + MSE dan ListNet · S3 native

[Buka Colab](https://colab.research.google.com/github/MalikHariyanto/Automated_Essay_Scoring/blob/codex/aes-learning/notebooks/experiments/AES_300_E44_E43_S3_MSE_ListNet_Native_Learnable_GPU.ipynb) · [Source](../../notebooks/experiments/AES_300_E44_E43_S3_MSE_ListNet_Native_Learnable_GPU.ipynb).

Pengguna menyetujui E44 setelah pembahasan fungsi ranking. Acuan pengembangan tetap E43: mean RMSE CV 5,8876, RMSE test 5,1265, MAE test 3,7817. E44 mengubah objective training menjadi **MSE + 0,1 × ListNet**. Belum ada hasil penelitian E44. E43/E30, dataset, hasil lama dan HTML tidak ditimpa.

## Rumus dan fungsi

Untuk setiap batch efektif 8 jawaban, MSE adalah rerata kuadrat selisih prediksi dan target pada skala 0–1. Seluruh jawaban, termasuk yang tidak mempunyai teman satu soal dalam batch, tetap mendapat MSE.

Untuk setiap kelompok minimal dua jawaban dengan teks soal yang persis sama, definisikan distribusi target `t=softmax(nilai_guru/100)` dan distribusi prediksi `p=softmax(prediksi_0_1)`. ListNet kelompok adalah `−sum(t × log(p))`, diimplementasikan dengan `log_softmax` yang stabil. Nilai guru di-detach pada distribusi target. ListNet batch adalah rerata loss kelompok yang memenuhi syarat, setiap kelompok berbobot sama. Tidak ada temperature tambahan. Jika tidak ada kelompok yang memenuhi syarat, ranking batch nol dengan gradien ranking nol; MSE tetap aktif.

Contoh: guru memberi A=90 dan B=75, tetapi model memprediksi A=80 dan B=85. Distribusi prediksi menempatkan B di atas A; gradien ranking mendorong A naik relatif terhadap B. MSE tetap memperbaiki angka absolut. Pergeseran kedua prediksi dengan konstanta yang sama tidak mengubah ranking, sehingga ranking sendiri tidak cukup untuk kalibrasi nilai.

Tidak ada `argsort` untuk mengubah dataset atau prediksi akhir. Pengelompokan hanya di dalam loss, mengikuti urutan sampler E43 yang sama. Dua soal berbeda tidak dibandingkan. Skor guru sama menghasilkan target probabilitas sama; tidak diberi urutan buatan. Karena target berupa distribusi, cross-entropy ListNet tidak harus nol saat prediksi tepat. Besar loss gabungan bukan MSE dan tidak digunakan memilih epoch.

Jika sepanjang suatu epoch tidak terdapat kelompok soal berukuran minimal dua, training berhenti dengan pesan bahwa ranking tidak aktif. Hal ini mencegah run MSE saja disebut sebagai hasil E44. Jumlah kelompok, jawaban dan batch ranking dicatat agar kontribusi aktual dapat diperiksa. Kelompok soal memakai teks bersih penuh dari kolom soal, bukan ID siswa atau target skor; bukan fitur baru untuk encoder.

## Batch, memori, dan gradien

Microbatch tetap 1 dan batch efektif tetap 8. Berbeda dari backward MSE per microbatch E43, E44 menggabungkan prediksi beberapa microbatch dalam satu objective batch efektif sebelum backward. Tidak ada detach pada prediksi; MSE dan ranking mengalir end-to-end ke RoBERTa, BiGRU, co-attention, Scalar Mix/gamma, proyeksi dan MLP. BGE encoder tetap frozen.

`RANKING_RECOMPUTE=True` memakai PyTorch activation checkpointing non-reentrant. Aktivasi dihitung ulang ketika backward agar tidak perlu menahan seluruh aktivasi delapan forward sekaligus. Ini bukan checkpoint file model atau model tambahan. Tensor kosong pada perangkat model diteruskan sebagai penanda agar PyTorch mendeteksi perangkat CUDA dan menyimpan RNG GPU, selain RNG CPU. `preserve_rng_state=True` menjaga mask dropout saat penghitungan ulang. Mode autocast juga dijaga oleh PyTorch. Tradeoff-nya tambahan forward saat backward; kebutuhan memori/waktu GPU aktual belum diukur dalam verifikasi CPU.

Saat validasi, test dan prediksi baru, jalur E43 tetap utuh: eval/no_grad, satu soal + referensi + jawaban → satu nilai. Tidak ada ranking, nilai guru atau kumpulan jawaban lain sebagai masukan prediksi.

## Pengaturan yang dipertahankan

| Komponen | E44 |
|---|---|
| X / y | `soal`, `jawaban_referensi`, `response_text` / `rater` 0–100 |
| RoBERTa | `flax-community/indonesian-roberta-base`, fine-tuned, LR 5e-6 |
| BGE | `BAAI/bge-m3`, frozen; proyeksi 1024→256 belajar |
| Scalar Mix | Empat layer terakhir, empat logit awal 0, gamma awal 1, LR 1e-4 |
| Representasi | BiGRU64/arah → co-attention/matching128 → masked mean → proyeksi768 |
| Fusion/head | Concat768+256 → MLP256/GELU/Dropout0,1 →1 |
| LR seluruh komponen di luar encoder | 1e-4 |
| Optimizer | AdamW, decay0,01, warmup10%, scheduler linear, clipping1 |
| Loss | MSE + bobot tetap0,1 × ListNet kelompok soal sama |
| Protokol | S3 saja, satu5-fold, maksimal25epoch, patience5 |
| Refit/test | Ceil median best epoch berdasarkan RMSE validasi; satu refit development, satu test |

Split/fold, satu TRAIN_SEED42, seed tiap fold, sampler dan jumlah optimizer step mengikuti E43. Tidak ada sweepseed, kontrol otomatis, S1/S2, cosine, atau perubahan dropout/LR. Semua parameter model dan inisialisasinya identik E43; objective tidak menambah parameter.

## Pemakaian dan output

Pilih GPU T4 → Run all → unggah `fix_dataset_300.xlsx`. Notebook memiliki 186 cell/90 cell kode Python biasa. Nama checkpoint, X/y dan hyperparameter ada pada section awal. `RANKING_WEIGHT` serta `RANKING_RECOMPUTE` ada di section5. Rumus `same_question_listnet`, wrapper perangkat dan `training_batch_loss` terlihat di cell training sebelum `fit`. Tanpa magic, worker tersembunyi atau clone repo.

Tetap lima Excel dalam `Hasil_AES_300_E44_E43_S3_MSE_ListNet_Native_Learnable_GPU.zip`:

- **05 → Fusion (S3)** memisahkan Loss Training (MSE), Loss Ranking (ListNet), Loss Gabungan, jumlah kelompok/jawaban/batch ranking, dan cakupan persentase per epoch. Loss epoch dirata-ratakan dengan bobot jumlah contoh pada batch aktual, sehingga gabungan epoch=MSE epoch+0,1×ListNet epoch.
- **05 → Pengaturan Eksperimen** mencatat resep loss, bobot ListNet dan recompute. Bobot Scalar Mix/gamma serta MAE dan Toleransi CV tetap tersedia.
- **04 → Prediksi Validasi OOF** memuat fold, ID jawaban, soal, nilai guru, prediksi desimal, pembulatan, selisih bertanda, absolut dan kuadrat. Setiap jawaban development muncul satu kali dari model yang tidak melatih jawaban itu. ID duplikat atau tumpang tindih dengan test ditolak.
- **04 → Nilai Guru dan Prediksi** tetap khusus test. OOF dan test dipisahkan, tidak digabung menjadi metrik penelitian baru. **02** tetap menyimpan bobot mean1/N, bobot co-attention, norma gamma dan vektor E43.

Hash/config/folder E44 terpisah; loader menolak resep atau bobot ranking berbeda. Resume menyimpan model, optimizer, scaler, scheduler, loader dan RNG, dengan history ranking. Unduh ZIP sebelum runtime dihapus. Pemilihan memakai mean RMSE CV; MAE/toleransi memberi konteks tradeoff, bukan alasan memilih dari test. Test historis sudah berulang kali dilihat, sehingga perbandingan tetap eksploratif.

## Rujukan dan batas adaptasi

Yang, Cao, Wen, Wu & He (2020), [Enhancing Automated Essay Scoring Performance via Fine-tuning Pre-trained Language Models with Combination of Regression and Ranking](https://aclanthology.org/2020.findings-emnlp.141/), menggabungkan MSE dan batch-wise ListNet berbobot dinamis pada AES/ASAP dan mengevaluasi QWK. Abstrak resmi sebelumnya diverifikasi dari [XML ACL Anthology](https://github.com/acl-org/acl-anthology/blob/master/data/xml/2020.findings.xml).

E44 memakai bobot tetap0,1 sebagai kandidat proyek, membatasi daftar pada soal sama, dan menerapkan objective pada fusion Indonesia E43. Ini adaptasi, bukan replikasi persis. Paper tidak menetapkan0,1 sebagai optimum dan tidak menjamin RMSE E44 membaik. Arsitektur mean pooling tetap berlandaskan Taghipour & Ng (2016), sementara Scalar Mix/gamma tetap memakai dasar ELMo dan komponen E43. Recompute mengikuti [dokumentasi PyTorch](https://docs.pytorch.org/docs/stable/checkpoint.html), sebagai teknik memori, bukan klaim peningkatan AES.

[Verifikasi teknis](../verifikasi/verification_e44_native.md).
