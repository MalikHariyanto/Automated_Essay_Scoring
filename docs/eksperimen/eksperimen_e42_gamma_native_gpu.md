# E42 · E41 + gamma yang dipelajari · S3 native

Pengguna menyetujui uji gamma setelah batas rujukan dijelaskan. E42 mengubah satu faktor dari E41: satu parameter skala global yang dipelajari pada campuran empat layer RoBERTa terakhir. Belum ada hasil penelitian E42.

[Buka Colab](https://colab.research.google.com/github/MalikHariyanto/Automated_Essay_Scoring/blob/codex/aes-learning/notebooks/experiments/AES_300_E42_E41_S3_ScalarMix4_Gamma_Native_Learnable_GPU.ipynb) · [Source](../../notebooks/experiments/AES_300_E42_E41_S3_ScalarMix4_Gamma_Native_Learnable_GPU.ipynb).

## Resep

Representasi token=gamma×sum(softmax(logit_layer)×hidden_layer). Empat logit tetap mulai nol/bobot0,25. Gamma mulai1,0 sehingga representasi dan prediksi awal identik dengan E41 pada bobot yang sama. Gamma bebas dipelajari, tidak dibatasi positif secara eksplisit, mengikuti skalar pada implementasi ScalarMix. Tidak ada gamma tambahan pada BGE atau pada skor akhir,normalisasi baru,atau koreksi bias berdasarkan test.

| Bagian | E42 |
|---|---|
| Gamma | Satu parameter baru,awal1,0 |
| LR gamma dan empat logit | 1e-4,dalam kelompok scalar_mix yang sama |
| LR seluruh encoder RoBERTa | Seragam5e-6 |
| LR MLP/proyeksi/BiGRU/co-attention/pooling/norm | 1e-4 |
| Layer campuran | Empat Transformer terakhir;embeddings tidak termasuk |
| BiGRU / attention / dropoutMLP | 64perarah / 64 / 0,1 |
| Fusion dan prediktor | RoBERTa768+BGE256→concat1024→MLP256→1 |
| BGE-M3 | Frozen1024;proyeksi256 tetap belajar |

Gamma mengalikan campuran768dimensi sebelum respons masuk BiGRU dan sebelum referensi diproyeksikan ke128untuk co-attention. Jumlah parameter bertambah tepat1dari E41;modul ScalarMix memiliki4logit+1gamma. Semua input,mask,windowing,loss,optimizer AdamW,weight decay0,01,warmup10%,scheduler linear,clipping1dan batch efektif8/micro1tetap E41. Tidak menambah pustaka AllenNLP;implementasi memakai PyTorch yang sudah digunakan notebook.

S3 saja,5fold sekali,maksimal25epoch/fold,patience5,1refit+1test. TRAIN_SEED42satu konstanta,seed fold mengikuti resep E30/E41;tidak ada loop multi-seed/kontrol ulang. Refit=ceil median best epoch lima fold. X=`soal,jawaban_referensi,response_text`,y=`rater`0–100hanya target loss/evaluasi. Test tidak memilih parameter;test historis yang telah dilihat berulang kali tetap eksploratif.

## Pemakaian dan ekspor

T4 GPU → Run all → unggah fix_dataset_300.xlsx. Format native184cell/89codecell,seperti E41,tanpa magic,worker tersembunyi,clone repo,Drive atau pemeriksaan cosine. Section5menampilkan SCALAR_MIX_GAMMA_INITIAL=1,0dan SCALAR_MIX_LR=1e-4. Section19menampilkan ScalarMix,rumus gamma serta model;section22menunjukkan kelompok optimizer dan training.

Output lima Excel standar. Pada05→Bobot Scalar Mix,kolom Gamma ditambahkan pada empat baris bobot setiap fold/refit. Nilainya sama dalam satu checkpoint karena gamma global;empat bobot softmax tetap berjumlah1. Nilai diekspor setelah pemulihan epoch terbaik. Pengaturan Eksperimen mencatat Gamma awal Scalar Mix.

Pada02,ditambahkan Norma Campuran Sebelum,Norma Campuran Sesudah dan Norma Layer Terakhir. Setiap jawaban mempunyai satu rerata norm L2token valid pada tiap window,lalu rerata antar-window. Token khusus dan konteks yang valid ikut dihitung;padding tidak dihitung. Norma sesudah≈|gamma|×norma sebelum,hingga toleransi komputasi. Norma bukan skor siswa. Statistik diambil hanya saat inspeksi fitur setelah final test,dalam no_grad,tanpa sinkronisasi CPU tambahan pada setiap langkah training. Ini membantu memeriksa hipotesis skala,tanpa menjadikannya loss tambahan.

Cache/checkpoint/folder memakai identitas E42baru;loader menolak metadata gamma awal berbeda. ZIP `Hasil_AES_300_E42_E41_S3_ScalarMix4_Gamma_Native_Learnable_GPU.zip`. Resume lokal hanya selama runtime masih ada;unduh ZIP sebelum runtime dihapus. E41 dan E30 acuan lengkap tidak ditimpa.

## Dasar penelitian dan batas adaptasi

Peters dkk.(2018), [Deep Contextualized Word Representations / ELMo](https://aclanthology.org/N18-1202/), memakai campuran layer berbobot dan skala gamma untuk tugas tertentu. [AllenNLP ScalarMix](https://github.com/allenai/allennlp/blob/main/allennlp/modules/scalar_mix.py) menginisialisasi gamma1,0dan mengembalikan gamma×campuran. [PretrainedTransformerEmbedder](https://github.com/allenai/allennlp/blob/main/allennlp/modules/token_embedders/pretrained_transformer_embedder.py) juga memakai ScalarMix pada layer Transformer.

Ini dasar metode/implementasi, bukan bukti langsung bahwa gamma meningkatkan AES dengan RoBERTa+BiGRU+co-attention+BGE. Paper AES dengan susunan persis itu yang mengisolasi gamma belum diverifikasi. LR1e-4dipertahankan dari kelompok campuran E41,bukan optimum yang ditetapkan paper. Bobot E41 yang hampir merata tidak membuktikan perlunya gamma atau LR lebih besar. Hipotesis skala akan diperiksa lewat CV,metrik,error/bias serta norm yang diekspor. Tidak ada jaminan mengalahkan E41/E30. [Verifikasi teknis](../verifikasi/verification_e42_native.md).
