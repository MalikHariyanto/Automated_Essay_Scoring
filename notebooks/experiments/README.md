# Katalog eksperimen AES

## E37 aktif · LR encoder7e-6 · S3 · 5-fold CV sekali

[**Buka Colab terbaru**](https://colab.research.google.com/github/MalikHariyanto/Automated_Essay_Scoring/blob/codex/aes-learning/notebooks/experiments/AES_300_E37_E30_S3_EncoderLR7e6_Native_Learnable_GPU.ipynb) · [Source](AES_300_E37_E30_S3_EncoderLR7e6_Native_Learnable_GPU.ipynb) · [Metode](../../docs/eksperimen/eksperimen_e37_single_lr_native_gpu.md).

Sesuai koreksi pengguna: langsung kandidat7e-6, tidak mengulang kontrol E30 atau tiga seed. S3 saja; MLP/layer/proyeksi1e-4;5fold,maks25epoch,patience5;1refit+1test. TRAIN_SEED42satu konstanta, tanpa sweep. Format native E35/E36 dan lima Excel standar. Belum ada hasil penelitian kandidat E37.

## Arsip E37 paired · digantikan versi lima fold di atas

[**Buka Colab**](https://colab.research.google.com/github/MalikHariyanto/Automated_Essay_Scoring/blob/codex/aes-learning/notebooks/experiments/AES_300_E37_E30_S3_EncoderLR5e6_vs7e6_3Seed_Native_Learnable_GPU.ipynb) · [Source](AES_300_E37_E30_S3_EncoderLR5e6_vs7e6_3Seed_Native_Learnable_GPU.ipynb) · [Metode](../../docs/eksperimen/eksperimen_e37_paired_lr_native_gpu.md).

Kontrol E30 encoder5e-6 dibanding kandidat7e-6, tiga seed42/142/242 dengan holdout/fold tetap. S3 saja, MLP/layer/proyeksi1e-4,5fold,maks25epoch,patience5.30latihan CV memilih LR; lalu3refit/test resep terpilih. Seed42selalu tabel/vektor utama, semua seed tetap dilaporkan. Lima Excel standar dengan tab tambahan per LR/seed; bukan ensemble. Format native E35/E36 tanpa magic/worker/pemeriksaan cosine. Belum ada hasil penelitian E37.

## E36 · S3 saja · E30 dengan LR RoBERTa 3e-6 · Native Learnable

[**Buka Colab**](https://colab.research.google.com/github/MalikHariyanto/Automated_Essay_Scoring/blob/codex/aes-learning/notebooks/experiments/AES_300_E36_E30_S3_EncoderLR3e6_Native_Learnable_GPU.ipynb) · [Source](AES_300_E36_E30_S3_EncoderLR3e6_Native_Learnable_GPU.ipynb) · [Metode](../../docs/eksperimen/eksperimen_e36_s3_native_gpu.md).

Arsitektur E30, format section native E35 tanpa magic/worker. Hanya LR encoder S3 berubah5e-6→3e-6; MLP, proyeksi BGE dan kelompok BiGRU/co-attention/pooling/norm tetap1e-4. BGE frozen. S1/S2 tidak dilatih atau dimuat dari hasil lama.5-fold CV,maksimal25epoch,patience5,lima Excel dengan kolom S3 saja. Hasil E36 telah diaudit:RMSEtest5,6738,meanCV5,9629;belum mengalahkan E30. [Analisis](../../docs/review/review_e36_study.md).

**E29 kini RMSE S3 terendah; S2 E29 memburuk tajam. Meja belajar HTML hanya memuat E29/E30/E36.** [Status lengkap](../../STATUS_PROYEK.md) · [buku hasil](../../downloads/README.md) · [rencana berikutnya](../../RENCANA_EKSPERIMEN.md).

## E34 · E33 + LR layer S3 5e-5 · kandidat berikutnya

[**Buka Colab**](https://colab.research.google.com/github/MalikHariyanto/Automated_Essay_Scoring/blob/codex/aes-learning/notebooks/experiments/AES_300_E34_E33_BiGRU_CoAttention_LayersLR5e5_GPU.ipynb) · [Source](AES_300_E34_E33_BiGRU_CoAttention_LayersLR5e5_GPU.ipynb) · [Metode](../../docs/eksperimen/eksperimen_e34_gpu.md).

Format E33/E30 asli:47cell/30codecell,S1/S2/S3,5fold CV,maksimal25epoch,patience5,lima Excel/ZIP. Hanya kelompok `pooling_norm` S3 turun1e-4→5e-5; MLP S3 tetap5e-5. Encoder5e-6; linear S1/MLP S2/proyeksi BGE1e-4. Prediktor akhir MLP, GPU Colab tanpa Drive. Semua model dilatih baru. Belum ada hasil penelitian E34.

## E33 · E30 + LR MLP S3 5e-5 · kandidat aktif

[**Buka Colab**](https://colab.research.google.com/github/MalikHariyanto/Automated_Essay_Scoring/blob/codex/aes-learning/notebooks/experiments/AES_300_E33_E30_BiGRU_CoAttention_MLPLR5e5_GPU.ipynb) · [Source](AES_300_E33_E30_BiGRU_CoAttention_MLPLR5e5_GPU.ipynb) · [Metode](../../docs/eksperimen/eksperimen_e33_gpu.md).

Format E30 asli:47cell/30codecell, S1/S2/S3,5fold CV, maksimal25epoch, patience5, lima Excel/ZIP. Hanya MLP S3 turun1e-4→5e-5; MLP S2 dan semua layer lain tetap resep E30. Prediktor akhir MLP. GPU Colab tanpa Drive, semua model dilatih baru. Belum ada hasil penelitian E33.

## E30 + Ridge · percobaan sebelumnya

[**Buka Colab**](https://colab.research.google.com/github/MalikHariyanto/Automated_Essay_Scoring/blob/codex/aes-learning/notebooks/experiments/AES_300_E30_BiGRU_CoAttention_Ridge_Colab.ipynb) · [Source](AES_300_E30_BiGRU_CoAttention_Ridge_Colab.ipynb) · [Metode](../../docs/eksperimen/eksperimen_e30_ridge.md).

Tiga input → RoBERTa fine-tuned + BiGRU + co-attention + attention pooling → concat BGE frozen → Ridge. **5-fold CV, maksimal25epoch, early stopping patience5**, lalu refit ceil median epoch terbaik. LR MLP1e-4 mengikuti E30; MLP disimpan sebagai pembanding. Alpha Ridge1,0 tetap sebelum CV. Progres Fold1/5 sampai5/5; tanpa nested CV/16fit. Lima Excel dan ZIP; belum ada hasil penelitian.

Notebook Ridge sebelumnya dipertahankan sebagai arsip; kandidat berikutnya E33 memakai E30 asli + MLP.

## Diagnosis BGE frozen · empat kondisi readout

[**Buka Colab**](https://colab.research.google.com/github/MalikHariyanto/Automated_Essay_Scoring/blob/codex/aes-learning/notebooks/experiments/AES_300_BGE_Frozen_Readout_Colab.ipynb) · [Source](AES_300_BGE_Frozen_Readout_Colab.ipynb) · [Metode dan referensi](../../docs/eksperimen/diagnostik_bge_frozen_readout.html).

BGE diekstrak sekali dan tetap frozen. Bandingkan MLP asli, MLP target-center, MLP StandardScaler, serta Ridge pada development/lima fold. Tiga seed MLP, head CPU float32, inner CV Ridge, lima Excel dan ZIP. Tidak menjalankan S1/S3 dan tidak mengevaluasi test.

## BGE-M3 frozen + Ridge · tiga input, CV dan test

[**Buka Colab**](https://colab.research.google.com/github/MalikHariyanto/Automated_Essay_Scoring/blob/codex/aes-learning/notebooks/experiments/AES_300_BGE_M3_Ridge_Tiga_Input_Colab.ipynb) · [Source](AES_300_BGE_M3_Ridge_Tiga_Input_Colab.ipynb) · [Metode dan referensi](../../docs/eksperimen/bge_ridge_tiga_input.html).

Notebook terpisah untuk soal + referensi + respons → BGE frozen → Ridge. Nested CV development, alpha final dari development, refit seluruh development, lalu evaluasi test historis. Lima Excel, prediksi per jawaban, model NPZ dan manifest untuk audit terhadap S2 E21–E32 pada partisi yang sama. Belum ada hasil penelitian dari notebook ini; skor OOF diagnosis sebelumnya bukan skor test.

## S3 · fusi RoBERTa/BGE → Ridge

[**Buka Colab**](https://colab.research.google.com/github/MalikHariyanto/Automated_Essay_Scoring/blob/codex/aes-learning/notebooks/experiments/AES_300_S3_Fusion_Ridge_Tiga_Input_Colab.ipynb) · [Source](AES_300_S3_Fusion_Ridge_Tiga_Input_Colab.ipynb).

Tiga teks → fusi E21 concat1024 → Ridge, dengan kontrol MLP pada fitur sama dan S2 Ridge tetap. RoBERTa/norm/proyeksi belajar dari train; encoder BGE frozen. Nested CV melatih ulang representasi di inner split, berbagi cache pasangan fold: 16 pelatihan pada konfigurasi5fold. Epoch tetap25 berbeda dari early stopping historis. Lima Excel+ZIP; checkpoint representasi final besar tetap lokal, tersedia opsi unduh. Referensi dan batas klaim ada di notebook; belum ada hasil penelitian dari kode baru ini.

## E31/E32 · E25 sebagai acuan, belum ada hasil

E31 mengulang perlakuan LR E29 pada E25; E32 menambah co-attention sebelum pooling/gate E25, tanpa BiGRU. GPU Colab biasa, tanpa Drive. [Metode](../../docs/eksperimen/eksperimen_e31_e32_gpu.md).

| Eksperimen | Perubahan | Jalankan |
| --- | --- | --- |
| E31 | MLP S2/S3 5e-6; gate/pooling/proyeksi/norm tetap1e-4 | [Buka Colab](https://colab.research.google.com/github/MalikHariyanto/Automated_Essay_Scoring/blob/codex/aes-learning/notebooks/experiments/AES_300_E31_E25_GatedFusion_MLPLR5e6_GPU.ipynb) |
| E32 | Co-attention128 → matching768 → pooling E25 → gate → MLP1e-4 | [Buka Colab](https://colab.research.google.com/github/MalikHariyanto/Automated_Essay_Scoring/blob/codex/aes-learning/notebooks/experiments/AES_300_E32_E25_CoAttention_GatedFusion_GPU.ipynb) |

## E29/E30 · dari E26, hasil keduanya sudah tersedia

GPU Colab biasa, tiga input dan label `rater`, tanpa Drive; seluruh S1/S2/S3 dilatih baru. [Metode, dua attention dan kelompok LR](../../docs/eksperimen/eksperimen_e29_e30_gpu.md).

| Eksperimen | Perubahan dari E26 | Jalankan |
| --- | --- | --- |
| **E29** | LR **MLP S2 dan S3**5e-6; linear S1 dan BiGRU/attention/proyeksi/norm tetap1e-4 | **[Buka Colab](https://colab.research.google.com/github/MalikHariyanto/Automated_Essay_Scoring/blob/codex/aes-learning/notebooks/experiments/AES_300_E29_E26_BiGRU_Attention_MLPLR5e6_GPU.ipynb)** · [Source](AES_300_E29_E26_BiGRU_Attention_MLPLR5e6_GPU.ipynb) |
| **E30** | E26 + **co-attention respons–referensi → attention pooling**; MLP tetap1e-4 | **[Buka Colab](https://colab.research.google.com/github/MalikHariyanto/Automated_Essay_Scoring/blob/codex/aes-learning/notebooks/experiments/AES_300_E30_E26_BiGRU_CoAttention_Pooling_GPU.ipynb)** · [Source](AES_300_E30_E26_BiGRU_CoAttention_Pooling_GPU.ipynb) |

[Review hasil E29](../../docs/review/review_e29_study.md). [Review hasil E30](../../docs/review/review_e30_study.md). E30 unggul MAE/toleransi, E29 tetap RMSE S3 terendah.

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

## S3 Ridge hemat · 5-fold CV, parameter tetap

[**Buka Colab**](https://colab.research.google.com/github/MalikHariyanto/Automated_Essay_Scoring/blob/codex/aes-learning/notebooks/experiments/AES_300_S3_Fusion_Ridge_Hemat_Colab.ipynb) · [Source](AES_300_S3_Fusion_Ridge_Hemat_Colab.ipynb).

Enam pelatihan × lima epoch (30 total), alpha S3 tetap1.0; ordinary grouped CV, tanpa nested tuning. Fusi dan tiga teks tetap, kontrol MLP/S2 Ridge. Unduhan backup kumulatif setelah setiap outer fold; unggah backup terakhir dan dataset asli sebelum Run all untuk restore fold selesai lintas runtime/perangkat. Tahap aktif serta encoder final tidak masuk backup kecil. Anggaran ini mengubah protokol, bukan jaminan akurasi atau kuota T4. Notebook nested lama dipertahankan; referensi dan batas klaim di notebook baru.

## E35 · Python biasa, section belajar jelas

[Buka Colab E35](https://colab.research.google.com/github/MalikHariyanto/Automated_Essay_Scoring/blob/codex/aes-learning/notebooks/experiments/AES_300_E35_E30_LayersLR5e5_Native_Learnable_GPU.ipynb) · [Metode dan peta cell](../../docs/eksperimen/eksperimen_e35_native_gpu.md). E30 dengan LR kelompok layer S3 5e-5; MLP S3 tetap1e-4. Tanpa cell magic/worker; nama model, X/y dan hyperparameter terlihat di bagian awal. Pemeriksaan cosine dihapus. 5fold,maks25epoch,patience5,T4 GPU. Belum ada hasil penelitian E35.
