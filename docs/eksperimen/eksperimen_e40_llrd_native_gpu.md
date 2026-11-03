# E40 · E30 MLP · LR RoBERTa per layer · S3 native

E40 menguji **layer-wise learning rate decay (LLRD), faktor0,95** pada arsitektur E30 MLP. Layer paling atas memakai LR puncak5e-6, lalu dikali0,95 untuk setiap layer di bawahnya. Embeddings satu tingkat di bawah layer0. Semua layer RoBERTa tetap dilatih. Jumlah layer diambil dari checkpoint aktual. E40 belum mempunyai hasil penelitian.

[Buka Colab](https://colab.research.google.com/github/MalikHariyanto/Automated_Essay_Scoring/blob/codex/aes-learning/notebooks/experiments/AES_300_E40_E30_S3_LLRD095_Native_Learnable_GPU.ipynb) · [Notebook](../../notebooks/experiments/AES_300_E40_E30_S3_LLRD095_Native_Learnable_GPU.ipynb).

## Resep yang diuji

Untuk indeks layer i dari L layer, LR puncak=5e-6×0,95^(L−1−i). Embeddings=5e-6×0,95^L. Pada encoder12layer:

| Kelompok | LR puncak |
|---|---:|
| Layer11, paling atas | 5e-6 |
| Layer8 | 4,286875e-6 |
| Layer5 | 3,675459e-6 |
| Layer2 | 3,151247e-6 |
| Layer0, paling bawah | 2,844000e-6 |
| Embeddings | 2,701800e-6 |
| BiGRU/co-attention/pooling/LayerNorm | 1e-4 |
| Proyeksi BGE dan MLP | 1e-4 |

LLRD membagi LR menurut kedalaman, sedangkan warmup10% dan scheduler linear membagi LR menurut waktu training. Scheduler mengalikan setiap kelompok dengan faktor waktu yang sama, sehingga rasio antarlayer dipertahankan pada langkah dengan LR aktif. Semua parameter trainable diperiksa masuk tepat satu kelompok. Weight decay0,01 tetap diterapkan seperti E30; tidak sekaligus memperkenalkan perlakuan lain terhadap bias/LayerNorm. Pooler opsional mendapat LR layer atas jika tersedia, tetapi loader eksperimen memuat encoder dengan add_pooling_layer=False.

Arsitektur kembali ke E30, bukan BiGRU128perarah E39: respons RoBERTa768→BiGRU64perarah→128; referensi768→128; co-attention respons–referensi; matching640→128; additive attention128→64→1; ringkasan128→768+LayerNorm. BGE frozen1024→256+LayerNorm; concat1024→MLP256→GELU→Dropout0,1→1. Jumlah parameter dan bentuk tensor sama dengan E30; MLP tetap prediktor akhir. LR layer baru tetap1e-4. Tidak ada Ridge atau pemeriksaan cosine.

Tiga input X adalah `soal`, `jawaban_referensi`, `response_text`. Satu target y=`rater`0–100 tidak masuk encoder. Teks, tokenisasi, mask, pengelompokan split dan panjang window mengikuti E30.

## Cara menjalankan dan belajar

Pilih runtime T4 GPU, jalankan Run all, lalu unggah `fix_dataset_300.xlsx`. Notebook native184cell/89codecell mempunyai section checkpoint, kolom X/y, hyperparameter, pembacaan data, tokenisasi, arsitektur, optimizer, training, evaluasi dan unduhan. Fungsi/variabel berada langsung di kernel Python. Tidak memakai cell magic, worker tersembunyi, clone repo atau Drive.

Section5 memuat `ENCODER_LR=5e-6`, `ENCODER_LAYER_DECAY=0.95`, `BIGRU_HIDDEN_PER_DIRECTION=64`, `POOLING_ATTENTION_HIDDEN=64` dan `S3_MLP_DROPOUT=0.1`. Section22 menjelaskan dan menampilkan fungsi `encoder_lr_schedule` serta `optimizer_groups`. Setiap fit mencetak kelompok LR aktual dan jumlah parameternya.

S3 saja,5fold CV sekali,maksimal25epoch/fold,patience5. TRAIN_SEED42adalah satu konstanta, dengan seed tiap fold=seed dasar+nomor fold mengikuti E30; tidak ada loop multi-seed atau pengulangan kontrol. Batch efektif8/micro1, AdamW, clipping1 dan MSE target0–1 tetap E30. Refit=ceil median best epoch lima fold, kemudian satu evaluasi test. Pada300baris:240development/60test; tiap fold192latih/48validasi. Jangan memilih LR/epoch dari test yang sudah berulang kali dilihat; hasil test selanjutnya bersifat eksploratif.

Output tetap lima Excel standar. Pada05→Pengaturan Eksperimen, `Learning rate encoder RoBERTa S3` berarti LR **puncak layer paling atas**, bersama `Faktor LR per layer RoBERTa`. Sheet tambahan **Learning Rate per Layer** mencatat tahap/fold, kelompok, LR puncak dan jumlah parameter aktual dari setiap fit. Pada02, Ringkasan CoAttn Pooling S3 kembali128komponen; Pooling Jawaban768dan Fusion1024. ZIP bernama `Hasil_AES_300_E40_E30_S3_LLRD095_Native_Learnable_GPU.zip`.

Cache/checkpoint memakai folder E40 serta hash kode/config baru. Loader menolak faktor decay atau arsitektur yang berbeda. Resume lokal menyimpan optimizer dan scheduler setiap epoch; jalankan ulang runtime yang sama untuk melanjutkan. Runtime Colab yang dihapus juga menghapus file lokal; unduh ZIP setelah selesai.

## Landasan dan batas kesimpulan

Sun,Qiu,Xu&Huang(2019), [How to Fine-Tune BERT for Text Classification?](https://arxiv.org/abs/1905.05583),Section5.3.4, membahas layer-wise decreasing learning rate. [README repositori penulis](https://github.com/xuyige/BERT4doc-Classification) menyebut parameter layer_learning_rate dan layer_learning_rate_decay. Dasar metode berasal dari fine-tuning BERT pada klasifikasi teks. Adaptasi ke RoBERTa fusion AES dan faktor0,95 adalah hipotesis eksperimen; paper tidak membuktikan faktor ini optimal untuk dataset penelitian ini.

Rujukan komponen E30 tetap tersedia pada section akhir notebook: RoBERTa,BGE-M3,GRU,co-attention untuk AES,pooling,LayerNorm,GELU dan AdamW. Bandingkan mean RMSE CV dengan E30 MLP5,9491beserta hasil per fold; evaluasi test hanya sesudah resep dikunci. E40 tidak dijamin menang. [Bukti verifikasi teknis](../verifikasi/verification_e40_native.md) harus dibedakan dari training penelitian GPU.
