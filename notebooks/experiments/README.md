# Katalog eksperimen AES

**Kode belajar utama sekarang E24.** Seluruh notebook publik tidak berisi output siswa. Hasil penelitian dibaca dari buku, bukan dibuat ulang oleh HTML.

## Tiga variabel: soal, jawaban referensi, jawaban siswa

Label seluruh skenario adalah **rater**, skor 0–100. LR encoder S1/S3 5e-6; proyeksi/head 1e-4. S1 memakai head linear, S2/S3 MLP. E21–E24 memakai folder output lokal, tanpa autosave Drive.

| Eksperimen | Perubahan | GPU Colab | CPU |
| --- | --- | --- | --- |
| E21 | Tiga teks sebagai satu urutan, pooling CLS | [GPU](AES_300_E21_Tiga_Input_GPU.ipynb) | [CPU](AES_300_E21_Tiga_Input_CPU.ipynb) |
| E22 | Konteks soal + referensi dipasangkan dengan jawaban | [GPU](AES_300_E22_Tiga_Input_ContextPair_GPU.ipynb) | [CPU](AES_300_E22_Tiga_Input_ContextPair_CPU.ipynb) |
| E23 | E21 + BiGRU hanya pada S3 | [GPU](AES_300_E23_Tiga_Input_BiGRU_GPU.ipynb) | — |
| **E24** | **E21 + attention pooling token jawaban hanya pada S3** | **[GPU](AES_300_E24_Tiga_Input_AttentionPooling_GPU.ipynb)** | — |

**Notebook lanjutan, GPU:** [E25 · E24 + gated fusion](AES_300_E25_Tiga_Input_Attention_GatedFusion_GPU.ipynb) dan [E26 · BiGRU + attention](AES_300_E26_Tiga_Input_BiGRU_Attention_GPU.ipynb). Hasil belum tersedia; kode belajar utama hasil terverifikasi tetap E24. [Metode dan cara menjalankan](../../docs/eksperimen_e25_e26_gpu.md).

[Buku tiga variabel terbaru](../../downloads/Belajar_AES_Tiga_Input_Lengkap.html) · [Metode E23/E24](../../docs/eksperimen_e23_e24_sequence_pooling_gpu.md). Hasil E23 dan E24 sudah tersedia di buku; E24 tetap kode utama berdasarkan kriteria yang dijelaskan pada perbandingan. CPU/GPU adalah pilihan perangkat, bukan dua hasil eksperimen pengganti.

## Eksperimen sebelum tiga variabel

| Metode | Notebook yang digunakan |
| --- | --- |
| MLP awal | [Notebook utama](../03_AES_dataset_guru_300.ipynb) |
| Double frozen + MLP, hapus checkpoint manual | [Manual](AES_300_S3_DoubleFrozen_MLP_Manual.ipynb) |
| Konteks pair E4 | [Notebook](AES_300_S3_ContextPair_MLP.ipynb) |
| LoRA S3 + MLP E5 | [GPU](AES_300_S3_LoRA_MLP_GPU.ipynb) |
| E6 · S3 LR5e-6, ulang semua skenario | [CPU](AES_300_S3_MLP_LR5e6_CPU.ipynb) |
| E6 · memakai hasil S1/S2 lama | [GPU](AES_300_E6_S3_LR5e6_GPU_Reuse.ipynb) · [CPU](AES_300_E6_S3_LR5e6_CPU_Reuse.ipynb) |
| E6 2.0 / tiga seed | [GPU](AES_300_E6_3Seed_GPU.ipynb) · [CPU](AES_300_E6_3Seed_CPU.ipynb) |
| E8 · konteks pair + S3 LR5e-6 | [GPU](AES_300_E8_ContextPair_LR5e6_GPU.ipynb) · [CPU](AES_300_E8_ContextPair_LR5e6_CPU.ipynb) |

[Buku eksperimen terdahulu terbaru](../../downloads/Belajar_AES_MLP_Lengkap.html). Notebook pembentuk versi hasil lama tetap ada agar metode dapat diperiksa; katalog ini mengutamakan varian terbaru yang digunakan. Salinan pada `aes-learning/notebooks` dipertahankan untuk tautan Colab lama yang masih digunakan.
