# E39 · E30 MLP · BiGRU 128 per arah · S3 native

Pengguna memilih rencana E39 dengan **BiGRU128 per arah**, bukan GRU satu arah. Sumber konsep adalah E30 MLP; format section mengikuti E37/E38. E39 menguji kapasitas recurrent, bukan manfaat arah mundur. Belum ada hasil penelitian E39.

[Buka Colab](https://colab.research.google.com/github/MalikHariyanto/Automated_Essay_Scoring/blob/codex/aes-learning/notebooks/experiments/AES_300_E39_E30_S3_BiGRU128_Native_Learnable_GPU.ipynb) · [Source](../../notebooks/experiments/AES_300_E39_E30_S3_BiGRU128_Native_Learnable_GPU.ipynb).

| Bagian | E30 MLP | E39 |
|---|---|---|
| Hidden BiGRU per arah | 64 | **128** |
| Keluaran dua arah per token | 128 | **256** |
| Layer BiGRU | 1 | 1 |
| Proyeksi referensi | 768→128 | 768→256 |
| Matching co-attention | 640→128 | 1280→256 |
| Attention pooling | 128→64→1 | 256→64→1 |
| Proyeksi ringkasan RoBERTa | 128→768 | 256→768 |
| BGE frozen + proyeksi | 1024→256 | Sama |
| Fusion / prediktor | Concat1024 → MLP256 → GELU → Dropout0,1 → 1 | Sama |
| LR encoder / layer lain | 5e-6 / 1e-4 | Sama |

Satu faktor yang dipilih lebih besar adalah hidden BiGRU. Dimensi proyeksi referensi, matching dan masukan pooling mengikuti keluaran BiGRU agar tensor tetap cocok. **Bottleneck attention pooling tetap64**, tidak ikut dinaikkan. Jumlah parameter bertambah820.224, seluruhnya pada sequence_pool; ini konsekuensi kapasitas yang diuji. Fitur akhir RoBERTa768 dan concat1024 tidak berubah. Model dapat lebih mahal atau lebih mudah overfit; hasil lebih baik tidak dijamin.

## Struktur yang mudah dipelajari

Notebook memiliki184cell/89codecell. Python biasa tersedia langsung dalam kernel, dengan section nama model, X/y, hyperparameter, pembacaan/tokenisasi, arsitektur, optimizer, training, evaluasi dan unduhan. Tidak ada magic command, worker tersembunyi, clone repo, Drive atau pemeriksaan cosine. Komentar Python biasa menjelaskan langkah penting.

`BIGRU_HIDDEN_PER_DIRECTION = 128` dan `POOLING_ATTENTION_HIDDEN = 64` berada di section5. `S3_MLP_DROPOUT = 0.1` kembali ke E30; perlakuan dropout0,2 E38 tidak dibawa. Kelas SingleScoreModel dan Engine hanya berisi jalur S3; optimizer langsung memakai empat kelompok S3. Tiga input tetap `soal`, `response_text`, `jawaban_referensi`; `rater`0–100 menjadi y, tidak masuk encoder. RoBERTa fine-tuned; BGE frozen; MLP menjadi prediktor akhir. Tidak ada Ridge atau penyeimbangan bobot cabang baru.

Keluaran dua arah disejajarkan dengan posisi token asli; teks tidak diubah menjadi kalimat terbalik. Kunci fitur ringkasan menggunakan nama umum `bigru_summary`, sehingga jumlah komponen tidak salah diberi nama128. Pengaturan hidden disimpan di config/checkpoint; loader menolak hidden atau pooling yang berbeda.

## Pelatihan dan output

S3 saja,5fold CV sekali,maksimal25epoch/fold,early stopping patience5,1refit+1test. TRAIN_SEED42 adalah satu konstanta; tiap fold memakai seed dasar+nomor fold mengikuti E30 MLP asli. Tidak mengulang tiga seed atau melatih kontrol E30. Split_seed2026, kelompok, holdout20%, batch efektif8/micro1, optimizer, warmup10%, scheduler linear, weight decay0,01 dan clipping1 tetap E30.

Refit memakai ceil median best epoch kelima fold. Test tidak memilih kapasitas/LR/epoch. Test lama sudah dilihat berulang kali, sehingga hasil berikutnya bersifat eksploratif. Jangan menetapkan jumlah epoch refit dari hasil test.

T4 GPU → Run all → upload `fix_dataset_300.xlsx`. Output tetap lima Excel standar. Pada02, `Ringkasan CoAttn Pooling S3` berisi256komponen, `Pooling Jawaban S3`768,dan `Fusion S3`1024. Pada05→Pengaturan Eksperimen, `Hidden pooling S3` adalah hidden BiGRU **per arah**, bernilai128; `Hidden attention pooling`64;dropoutMLP0,1. ZIP akhir `Hasil_AES_300_E39_E30_S3_BiGRU128_Native_Learnable_GPU.zip`.

Cache/checkpoint E39 memiliki folder/code/config hash baru; tidak memuat hasil eksperimen lama. Resume berlaku dalam runtime lokal yang sama; file lokal hilang bila runtime dihapus. Unduh ZIP sebelum sesi berakhir.

## Rujukan dan batas adaptasi

- Tan,Lee&Lim(2023), [RoBERTa-GRU: A Hybrid Deep Learning Model for Enhanced Sentiment Analysis](https://doi.org/10.3390/app13063915). Rujukan pengguna ini menguji kapasitas GRU dan melaporkan pilihan256unit pada sentimen. BiGRU128perarah menghasilkan256fitur gabungan, tetapi tidak sama dengan GRU256searah dalam arsitektur paper. Angka128perarah adalah hipotesis AES, bukan optimum yang sudah dibuktikan paper.
- Schuster&Paliwal(1997), [Bidirectional Recurrent Neural Networks](https://doi.org/10.1109/78.650093), dasar pengolahan konteks dua arah.
- Cho dkk.(2014), [Learning Phrase Representations using RNN Encoder–Decoder](https://arxiv.org/abs/1406.1078), dasar GRU.
- Zhang&Litman(2018), [Co-Attention Based Neural Network for Source-Dependent Essay Scoring](https://aclanthology.org/W18-0549/), dasar pencocokan esai–sumber; proyek mengadaptasinya pada token respons–referensi.

Rujukan RoBERTa,BGE-M3,pooling,LayerNorm,GELU dan AdamW tetap berada pada section akhir notebook. [Verifikasi teknis](../verifikasi/verification_e39_native.md) membedakan fixture CPU dari training penelitian GPU. Jangan menambahkan fixture ke HTML atau ranking penelitian.
