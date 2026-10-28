# Automated Essay Scoring · mulai di sini

Versi aktif memakai **tiga teks**—soal, kunci jawaban, respons siswa—dan **satu label `rater` (0–100)**. Notebook berjalan di Colab; HTML untuk belajar dan membaca hasil.

| Tujuan | Mulai dari |
| --- | --- |
| Melanjutkan dari akun/chat baru | **[Panduan untuk AI](PANDUAN_AI_PROYEK.md)**, [pindah akun](PINDAH_AKUN.md), [status proyek](STATUS_PROYEK.md) |
| Menjalankan notebook | **[Katalog eksperimen](notebooks/experiments/README.md)** |
| Membaca hasil lengkap | **[Dua HTML terbaru](downloads/README.md)** |
| Menentukan combo berikutnya | **[Rencana eksperimen](RENCANA_EKSPERIMEN.md)** |
| Mencari metode, review, atau bukti | **[Daftar dokumentasi](docs/README.md)** |

Notebook baru yang diminta: **[E27 · E23 + co-attention](notebooks/experiments/AES_300_E27_E23_BiGRU_CoAttention_GPU.ipynb)** dan **[E28 · E23 + LR MLP S3 5e-6](notebooks/experiments/AES_300_E28_E23_BiGRU_MLPLR5e6_GPU.ipynb)**. GPU, tanpa Drive; [rincian metode](docs/eksperimen/eksperimen_e27_e28_gpu.md). Belum ada hasil penelitian baru; buku masih memakai hasil E21–E26.

## Hasil terbaru: E21–E26

Enam eksperimen, 18 model, 60 jawaban test yang sama, 30 Excel/270 sheet.

| Ukuran S3 | Hasil unggahan saat ini |
| --- | --- |
| RMSE test terendah | **E26 · BiGRU + attention: 5,7435 poin** |
| Rerata RMSE lima fold terendah | **E26: 5,9245 poin** |
| Paling sering menjadi prediksi terdekat | **E23: 13/60**, saat enam eksperimen dibandingkan |
| Terbanyak selisih maksimal 5 poin | **E23: 39/60** |

Kode utama HTML adalah **E23**, mengikuti aturan jumlah terdekat. **E26** tersedia sebagai pilihan RMSE terkecil dan acuan rencana berikutnya. Ukuran ini berbeda; satu run belum membuktikan kestabilan lintas seed. [Review terbaru](docs/review/review_e25_e26_study.md).

## Dua buku yang dipakai

| Buku | Isi | Unduh lengkap | Preview publik |
| --- | --- | --- | --- |
| **Tiga input · E21–E26** | Perbandingan, jawaban/prediksi, kurva, Excel, kode dan sticky note | **[Pengunduh HTML](downloads/Belajar_AES_Tiga_Input_Lengkap.html)** | [Ringkasan](docs/belajar_Tiga_Input_Spider.html) |
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
