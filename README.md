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

## Hasil terbaru: E21–E28

Delapan eksperimen,24model,60jawaban test yang sama,40Excel/363sheet. Enam buku lama beserta kode/catatan/snapshot hasil dipertahankan identik; E27/E28 ditambahkan.

| Ukuran S3 | Hasil unggahan saat ini |
| --- | --- |
| RMSE test terendah | **E26 · BiGRU + attention: 5,7435 poin** |
| Rerata RMSE lima fold terendah | **E26: 5,9245 poin** |
| Paling sering menjadi prediksi terdekat | **E23 dan E26:10/60**, pada delapan peserta; E23 sebelumnya13/60 pada enam peserta |
| Terbanyak selisih maksimal 5 poin | **E23: 39/60** |

Kode utama HTML tetap **E23** sesuai pilihan pengguna. **E26** tersedia sebagai RMSE terkecil. Penambahan peserta dapat mengubah jumlah terdekat tanpa mengubah satu prediksi lama. E27 RMSE S3 **6,3277**, E28 **7,0447**. Ukuran berbeda; satu run belum membuktikan kestabilan lintas seed. [Review terbaru](docs/review/review_e27_e28_study.md).

## Dua buku yang dipakai

| Buku | Isi | Unduh lengkap | Preview publik |
| --- | --- | --- | --- |
| **Tiga input · E21–E28** | Perbandingan, jawaban/prediksi, kurva, Excel, kode dan sticky note | **[Pengunduh HTML](downloads/Belajar_AES_Tiga_Input_Lengkap.html)** | [Ringkasan](docs/belajar_Tiga_Input_Spider.html) |
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
