# E36 · E30 S3 saja, LR encoder RoBERTa 3e-6

[Notebook](../../notebooks/experiments/AES_300_E36_E30_S3_EncoderLR3e6_Native_Learnable_GPU.ipynb) · [Buka Colab](https://colab.research.google.com/github/MalikHariyanto/Automated_Essay_Scoring/blob/codex/aes-learning/notebooks/experiments/AES_300_E36_E30_S3_EncoderLR3e6_Native_Learnable_GPU.ipynb).

Metode berasal dari E30; susunan Python native mengikuti E35. Ini satu kandidat LR, bukan sweep tiga nilai. Tidak ada hasil penelitian E36 sebelum notebook dilatih pada dataset guru dengan checkpoint pretrained.

| Kelompok S3 | E30 | E36 |
| --- | ---: | ---: |
| Encoder RoBERTa, full fine-tuning | 5e-6 | **3e-6** |
| MLP prediksi | 1e-4 | 1e-4 |
| BiGRU/co-attention/pooling/LayerNorm | 1e-4 | 1e-4 |
| Proyeksi BGE | 1e-4 | 1e-4 |
| Encoder BGE-M3 | Frozen | Frozen |

LR kelompok layer E35 5e-5 **tidak dipakai**. Prediktor akhir tetap MLP E30, bukan Ridge. RoBERTa `flax-community/indonesian-roberta-base` dan BGE `BAAI/bge-m3` memakai revision yang sama dengan E30. Inisialisasi, dropout0,1, loss MSE target0–1, AdamW decay0,01, warmup10%+linear scheduler, clipping1, microbatch1/batch efektif8, dan tokenizer/window mengikuti E30.

## S3 tanpa latihan S1/S2

`SCENARIOS = ('S3',)` mengendalikan CV, refit, test, prediksi baru dan ekspor fitur. `make_model()` menolak permintaan S1/S2. Tidak memakai checkpoint/hasil S1/S2; preprocessing BGE berada langsung di section17. Pembacaan RoBERTa sebelum training adalah inspeksi encoder, bukan pelatihan S1. Kelas umum E30 masih memuat cabang historis untuk keterlacakan source; cabang tersebut tidak dieksekusi.

Ketiga teks dalam satu baris adalah `soal`, `jawaban_referensi`, `response_text`; satu targetnya `rater`0–100. Ketiganya dipasangkan per esai. Target tidak masuk encoder. RoBERTa→BiGRU→co-attention respons/referensi→attention pooling→768→LayerNorm; BGE1024→256→LayerNorm; concat1024→MLP256→1.

Seed42 dan split_seed2026, 20% test, **5-fold development CV**, maksimal25epoch per fold, patience5 tanpa perbaikan RMSE validasi. Bobot epoch terbaik dipulihkan. Refit baru pada seluruh development memakai ceil median epoch terbaik kelima fold; tidak memakai early stopping pada test. Untuk dataset300 valid:240development,60test. Test historis sudah sering diperiksa sehingga hasil lanjutan bersifat eksploratif. Ini tidak menjamin mengalahkan E30.

## Menjalankan dan belajar

Pilih runtime Colab baru dengan T4 GPU, Run all, upload Excel. Tidak ada Drive, clone, file Python pendamping, cell magic, atau training worker. Semua nama model, input, target dan fungsi training dapat dicari langsung pada notebook.

| Bagian | Isi |
| --- | --- |
| 3 | Nama checkpoint dan revision |
| 4 | Kolom X/y, metadata kelompok, file dataset |
| 5 | LR, fold, epoch, patience dan hyperparameter lain |
| 11 | Variabel X/y aktual yang sama dengan input training |
| 17 | BGE frozen, tanpa menjalankan skenario S2 |
| 19 | BiGRU, co-attention, pooling, fusion dan MLP |
| 22 | Kelompok optimizer, MSE/backward, early stopping dan resume |
| 24 | Pemanggilan 5fold/refit/test S3 |
| 28 | Unduhan ZIP lima Excel |

Workbook04 hanya berisi prediksi dan evaluasi S3; tidak ada kolom prediksi S1/S2 kosong. Workbook05 mencatat LR encoder3e-6 dan LR layer/head1e-4. Attention, tokenisasi, vektor awal/akhir, prediksi, riwayat fold/epoch tetap tersedia. Tidak ada pemeriksaan cosine. Fitur seluruh300jawaban untuk inspeksi; evaluasi penelitian hanya dari60test. Folder/checkpoint dan identitas resume E36 terpisah dari eksperimen lama.

Format dipin ke notebook E35 pada commit `699aaa3be612400ccf264bd8a8b96984263c8c8c`; metode E30 mengikuti pin `7f3363643e6bf402a293de069bcfc23d85b19c89` yang tercatat pada metadata. Builder tidak mengubah source E30/E35. [Bukti pengujian](../verifikasi/verification_e36_native.md).

Rujukan komponen dan batas adaptasi tersedia di notebook serta [metode E30](eksperimen_e29_e30_gpu.md): Liu dkk.2019 (RoBERTa), Xiao dkk.2024 (BGE-M3), Cho dkk.2014 (GRU), Bahdanau dkk.2015 (attention), Zhang & Litman2018 (co-attention AES), Loshchilov & Hutter2019 (AdamW). LR3e-6 adalah hipotesis eksperimen proyek, bukan nilai optimal yang dibuktikan paper tersebut.
