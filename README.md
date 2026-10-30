# Automated Essay Scoring · mulai di sini

Versi aktif memakai **tiga teks**—soal, kunci jawaban, respons siswa—dan **satu label `rater` (0–100)**. Notebook berjalan di Colab; HTML untuk belajar dan membaca hasil.

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
