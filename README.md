# Belajar dan eksperimen Automated Essay Scoring

Dua buku terbaru tersedia di bawah. Di GitHub, pilih **Download raw file**, buka HTML pengunduh, klik **Unduh buku**, ekstrak ZIP, lalu gunakan kunci sebelumnya. Kode dan hasil dibaca offline; HTML tidak menjalankan training.

| Buku | Isi utama | Unduh lengkap | Ringkasan publik |
| --- | --- | --- | --- |
| **Tiga variabel · E21–E22–E23–E24** | Kode utama **E24**, catatan sejajar seperti panduan E6, empat eksperimen, dua belas model, prediksi 60 jawaban, 20 Excel | **[Unduh](downloads/Belajar_AES_Tiga_Input_Lengkap.html)** | [Buka](docs/belajar_Tiga_Input_Spider.html) |
| **Eksperimen terdahulu · sampai E8 / E6 2.0** | Kumpulan hasil sebelum tiga variabel, kode belajar E6, seluruh Excel lama | **[Unduh](downloads/Belajar_AES_MLP_Lengkap.html)** | [Buka](docs/belajar_MLP_Spider.html) |

HTML terpisah, variasi tema lama, serta pengunduh E21 lama telah digantikan oleh dua buku ini. [Petunjuk unduhan](downloads/README.md) · [Audit E24](docs/review_e24_study.md) · [Perbaikan tampilan dan pembersihan](docs/review_e23_study.md).

**Eksperimen lanjutan GPU:** [E25 · attention + gated fusion](notebooks/experiments/AES_300_E25_Tiga_Input_Attention_GatedFusion_GPU.ipynb) dan [E26 · BiGRU + attention](notebooks/experiments/AES_300_E26_Tiga_Input_BiGRU_Attention_GPU.ipynb). Tiga input, rater sebagai label, tanpa Drive. [Cara menjalankan](docs/eksperimen_e25_e26_gpu.md). Hasil keduanya belum tersedia.

## Notebook eksperimen

**[Katalog seluruh eksperimen](notebooks/experiments/README.md)** memisahkan perubahan metode, pilihan CPU/GPU, dan versi yang memakai ulang hasil lama.

| Eksperimen tiga variabel | Metode S3 | Jalankan di Colab GPU |
| --- | --- | --- |
| E21 | RoBERTa fine-tuned + BGE frozen → fusion MLP | [Buka E21](https://colab.research.google.com/github/MalikHariyanto/Automated_Essay_Scoring/blob/codex/aes-learning/notebooks/experiments/AES_300_E21_Tiga_Input_GPU.ipynb) |
| E22 | E21 dengan format konteks pair | [Buka E22](https://colab.research.google.com/github/MalikHariyanto/Automated_Essay_Scoring/blob/codex/aes-learning/notebooks/experiments/AES_300_E22_Tiga_Input_ContextPair_GPU.ipynb) |
| E23 | E21 + BiGRU hanya pada S3 | [Buka E23](https://colab.research.google.com/github/MalikHariyanto/Automated_Essay_Scoring/blob/codex/aes-learning/notebooks/experiments/AES_300_E23_Tiga_Input_BiGRU_GPU.ipynb) |
| **E24** | **E21 + masked attention pooling token jawaban pada S3** | **[Buka E24](https://colab.research.google.com/github/MalikHariyanto/Automated_Essay_Scoring/blob/codex/aes-learning/notebooks/experiments/AES_300_E24_Tiga_Input_AttentionPooling_GPU.ipynb)** |

Input pada S1/S2/S3: `soal`, `jawaban_referensi`, `response_text`; label `rater` 0–100. S1 memakai head linear; S2/S3 MLP. LR RoBERTa 5e-6; proyeksi/head/layer baru 1e-4. Lima fold, maksimal 25 epoch, patience 5. E21–E24 menyimpan hasil lokal tanpa autosave Drive. [Penjelasan metode E23/E24](docs/eksperimen_e23_e24_sequence_pooling_gpu.md).

Eksperimen sebelum tiga variabel memakai [MLP awal](notebooks/03_AES_dataset_guru_300.ipynb). Versi metode lama tetap dapat ditelusuri melalui katalog dan riwayat Git; varian CPU/GPU yang berbeda dipertahankan. Jangan menafsirkan skor pada eksperimen single-output sebagai hasil empat trait.

## Source dan bukti

Repo publik berisi source, notebook **tanpa output**, pengujian, dokumentasi, ringkasan agregat dan paket buku terenkripsi. Dataset, skor individual, executed notebook, bobot dan kunci tetap privat. Pembaca lengkap menampilkan data siswa hanya setelah buku terenkripsi dibuka dengan kunci.

Perubahan HTML ini tidak mengubah model atau menjalankan training baru. [Audit tiga input](docs/review_three_input_study.md) membedakan hasil unggahan dari pemeriksaan teknis tampilan. Model terbaik pada satu run belum membuktikan kestabilan antar-seed.

Workflow empat trait terdahulu tetap terpisah: [kontrak implementasi](docs/implementation_contract.md), [alur model](docs/alur_model.md), [notebook belajar](notebooks/01_belajar_alur_AES.ipynb), dan [bukti verifikasi](docs/verification_report.md). Instalasi Python mengikuti [requirements.txt](requirements.txt); data/rubrik sumber diperlukan sesuai workflow yang dijalankan. Tidak ada data penelitian pengganti dalam repo.
