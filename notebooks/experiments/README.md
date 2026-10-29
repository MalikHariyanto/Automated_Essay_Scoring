# Katalog eksperimen AES

**Mulai dari E26 untuk acuan RMSE; E23 untuk mempelajari prediksi yang paling sering terdekat.** [Status lengkap](../../STATUS_PROYEK.md) · [buku hasil](../../downloads/README.md) · [rencana berikutnya](../../RENCANA_EKSPERIMEN.md).

## E29/E30 · notebook baru dari E26, hasil belum tersedia

GPU Colab biasa, tiga input dan label `rater`, tanpa Drive; seluruh S1/S2/S3 dilatih baru. [Metode, dua attention dan kelompok LR](../../docs/eksperimen/eksperimen_e29_e30_gpu.md).

| Eksperimen | Perubahan dari E26 | Jalankan |
| --- | --- | --- |
| **E29** | LR **MLP S2 dan S3**5e-6; linear S1 dan BiGRU/attention/proyeksi/norm tetap1e-4 | **[Buka Colab](https://colab.research.google.com/github/MalikHariyanto/Automated_Essay_Scoring/blob/codex/aes-learning/notebooks/experiments/AES_300_E29_E26_BiGRU_Attention_MLPLR5e6_GPU.ipynb)** · [Source](AES_300_E29_E26_BiGRU_Attention_MLPLR5e6_GPU.ipynb) |
| **E30** | E26 + **co-attention respons–referensi → attention pooling**; MLP tetap1e-4 | **[Buka Colab](https://colab.research.google.com/github/MalikHariyanto/Automated_Essay_Scoring/blob/codex/aes-learning/notebooks/experiments/AES_300_E30_E26_BiGRU_CoAttention_Pooling_GPU.ipynb)** · [Source](AES_300_E30_E26_BiGRU_CoAttention_Pooling_GPU.ipynb) |

## Hasil baru E27/E28 · dasar E23

Keduanya turunan E23, GPU Colab biasa, tiga input dan satu label `rater`, tanpa Drive. Perubahan khusus S3; notebook meminta seluruh S1/S2/S3 dilatih baru. Hasil unggahan sudah masuk buku: RMSE S3 E27 **6,3277**, E28 **7,0447**. [Audit hasil](../../docs/review/review_e27_e28_study.md). [Metode dan LR](../../docs/eksperimen/eksperimen_e27_e28_gpu.md).

| Eksperimen | Perubahan dari E23 | Jalankan |
| --- | --- | --- |
| **E27** | E23 BiGRU + **co-attention respons–referensi**; MLP tetap1e-4 | **[Buka Colab](https://colab.research.google.com/github/MalikHariyanto/Automated_Essay_Scoring/blob/codex/aes-learning/notebooks/experiments/AES_300_E27_E23_BiGRU_CoAttention_GPU.ipynb)** · [Source](AES_300_E27_E23_BiGRU_CoAttention_GPU.ipynb) |
| **E28** | Arsitektur E23 sama; **LR MLP S3 saja5e-6**, pooling/norm/proyeksi tetap1e-4 | **[Buka Colab](https://colab.research.google.com/github/MalikHariyanto/Automated_Essay_Scoring/blob/codex/aes-learning/notebooks/experiments/AES_300_E28_E23_BiGRU_MLPLR5e6_GPU.ipynb)** · [Source](AES_300_E28_E23_BiGRU_MLPLR5e6_GPU.ipynb) |

## Tiga input · hasil tersedia

X pada seluruh skenario: `soal`, `jawaban_referensi`, `response_text`. y: satu `rater` 0–100. S1 linear; S2/S3 MLP. RoBERTa S1/S3 LR5e-6; proyeksi/head/layer tambahan1e-4. Maksimal25epoch, patience5, lima fold. Folder output lokal, tanpa Drive.

| Eksperimen | S3 | GPU | CPU | RMSE test S3 |
| --- | --- | --- | --- | ---: |
| E21 | CLS → concat → MLP | [Buka Colab](https://colab.research.google.com/github/MalikHariyanto/Automated_Essay_Scoring/blob/codex/aes-learning/notebooks/experiments/AES_300_E21_Tiga_Input_GPU.ipynb) · [Source](AES_300_E21_Tiga_Input_GPU.ipynb) | [Notebook](AES_300_E21_Tiga_Input_CPU.ipynb) | 6,7231 |
| E22 | E21 dengan format context pair | [Buka Colab](https://colab.research.google.com/github/MalikHariyanto/Automated_Essay_Scoring/blob/codex/aes-learning/notebooks/experiments/AES_300_E22_Tiga_Input_ContextPair_GPU.ipynb) · [Source](AES_300_E22_Tiga_Input_ContextPair_GPU.ipynb) | [Notebook](AES_300_E22_Tiga_Input_ContextPair_CPU.ipynb) | 7,1943 |
| **E23** | BiGRU → state akhir → proyeksi | [Buka Colab](https://colab.research.google.com/github/MalikHariyanto/Automated_Essay_Scoring/blob/codex/aes-learning/notebooks/experiments/AES_300_E23_Tiga_Input_BiGRU_GPU.ipynb) · [Source](AES_300_E23_Tiga_Input_BiGRU_GPU.ipynb) | — | 6,0439 |
| E24 | Attention pooling token respons | [Buka Colab](https://colab.research.google.com/github/MalikHariyanto/Automated_Essay_Scoring/blob/codex/aes-learning/notebooks/experiments/AES_300_E24_Tiga_Input_AttentionPooling_GPU.ipynb) · [Source](AES_300_E24_Tiga_Input_AttentionPooling_GPU.ipynb) | — | 5,9818 |
| E25 | E24 + gated fusion | [Buka Colab](https://colab.research.google.com/github/MalikHariyanto/Automated_Essay_Scoring/blob/codex/aes-learning/notebooks/experiments/AES_300_E25_Tiga_Input_Attention_GatedFusion_GPU.ipynb) · [Source](AES_300_E25_Tiga_Input_Attention_GatedFusion_GPU.ipynb) | — | 6,1053 |
| **E26** | BiGRU → attention → proyeksi | **[Buka Colab](https://colab.research.google.com/github/MalikHariyanto/Automated_Essay_Scoring/blob/codex/aes-learning/notebooks/experiments/AES_300_E26_Tiga_Input_BiGRU_Attention_GPU.ipynb)** · [Source](AES_300_E26_Tiga_Input_BiGRU_Attention_GPU.ipynb) | — | **5,7435** |

Untuk menjalankan: buka notebook di Colab melalui menu **File → Open notebook → GitHub**, atau ganti alamat GitHub dengan format `https://colab.research.google.com/github/OWNER/REPO/blob/REF/PATH.ipynb`. Pilih **T4 GPU → Run all → unggah dataset**. Seluruh setup ada dalam notebook. Varian CPU adalah pilihan perangkat; hasil tabel berasal dari ZIP unggahan, bukan bukti hasil identik pada setiap perangkat.

Metode: [E21/E22](../../docs/eksperimen/eksperimen_e21_e22_web_colab_gpu.md), [E23/E24](../../docs/eksperimen/eksperimen_e23_e24_sequence_pooling_gpu.md), [E25/E26](../../docs/eksperimen/eksperimen_e25_e26_gpu.md). [Audit hasil](../../docs/review/review_e25_e26_study.md).

## Sebelum keluarga tiga input

Nomor eksperimen lama mengikuti penamaan pada buku. Tabel ini tidak menyamakan input versi lama dengan E21–E26.

| Metode | Notebook | Catatan |
| --- | --- | --- |
| Linear awal | [Source](AES_300_S3_Linear.ipynb) | Acuan awal |
| MLP awal | [Utama](../03_AES_dataset_guru_300.ipynb), [varian eksperimen](AES_300_S3_MLP.ipynb) | Acuan sebelum tiga input |
| Double frozen + MLP | [Manual](AES_300_S3_DoubleFrozen_MLP_Manual.ipynb) | Pilihan penyimpanan manual |
| Double frozen dengan auto-cleanup lama | [Historis](AES_300_S3_DoubleFrozen_MLP.ipynb) | Tidak dipilih untuk workflow terbaru |
| E4 · context pair | [Notebook](AES_300_S3_ContextPair_MLP.ipynb) | Input sesuai versi E4 |
| E5 · LoRA S3 + MLP | [GPU](AES_300_S3_LoRA_MLP_GPU.ipynb) | Bukan full fine-tuning S3 acuan |
| E6 · S3 LR5e-6, ulang seluruh skenario | [CPU](AES_300_S3_MLP_LR5e6_CPU.ipynb) | S1 tetap resep E6 |
| E6 · reuse S1/S2 | [GPU](AES_300_E6_S3_LR5e6_GPU_Reuse.ipynb), [CPU](AES_300_E6_S3_LR5e6_CPU_Reuse.ipynb) | Kontrol bukan training independen |
| E6 2.0 / tiga seed | [GPU](AES_300_E6_3Seed_GPU.ipynb), [CPU](AES_300_E6_3Seed_CPU.ipynb) | Berbeda dari uji tiga seed E26 yang belum ada |
| E6 · tiga input CPU terdahulu | [Notebook](AES_300_E6_Tiga_Input_CPU.ipynb) | Varian sebelum E21; punya edit Colab |
| E4 · tiga input CPU terdahulu | [Notebook](AES_300_E4_Tiga_Input_CPU.ipynb) | Varian sebelum E22 |
| E8 · context pair + S3 LR5e-6 | [GPU](AES_300_E8_ContextPair_LR5e6_GPU.ipynb), [CPU](AES_300_E8_ContextPair_LR5e6_CPU.ipynb) | Bukan E22 |

[Buku hasil terdahulu](../../downloads/Belajar_AES_MLP_Lengkap.html). Varian mirror lama yang berbeda dipertahankan di [arsip](../archive/README.md). Source jalur utama tidak diganti dengan builder saat penataan, sehingga edit pengguna tidak tertimpa.
