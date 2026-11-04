# E41 Scalar Mix dan E30 MLP acuan native

**Pembaruan pengguna:** E30 acuan terbaru harus lengkap S1/S2/S3. Gunakan [E30 acuan lengkap native](e30_acuan_lengkap_native_gpu.md). Uraian E30 S3 saja di bawah merekam varian yang dikirim sebelumnya; E41 tetap S3 saja.

Pengguna meminta dua notebook: E41 dengan campuran empat layer RoBERTa terakhir dan E30 MLP acuan dengan format kode terbaru, terstruktur serta tanpa cell magic. Keduanya merupakan notebook mandiri S3 saja. Source E30 historis dan seluruh hasil/HTML lama tetap utuh.

| Notebook | Colab | Source |
|---|---|---|
| E41 · Scalar Mix4 | [Buka](https://colab.research.google.com/github/MalikHariyanto/Automated_Essay_Scoring/blob/codex/aes-learning/notebooks/experiments/AES_300_E41_E30_S3_ScalarMix4_Native_Learnable_GPU.ipynb) | [Notebook](../../notebooks/experiments/AES_300_E41_E30_S3_ScalarMix4_Native_Learnable_GPU.ipynb) |
| E30 · MLP acuan native | [Buka](https://colab.research.google.com/github/MalikHariyanto/Automated_Essay_Scoring/blob/codex/aes-learning/notebooks/experiments/AES_300_E30_S3_MLP_Native_Learnable_GPU.ipynb) | [Notebook](../../notebooks/experiments/AES_300_E30_S3_MLP_Native_Learnable_GPU.ipynb) |

## Apa yang dibandingkan

E30 native memakai `last_hidden_state` seperti resep S3 historis. Perubahannya adalah penyajian Python biasa dalam kernel, dengan section nama checkpoint, kolom X/y, hyperparameter, pembacaan data, tokenisasi, arsitektur, optimizer, training, evaluasi dan unduhan. E30 lama tidak ditimpa. Ini menjalankan ulang resep, bukan memuat prediksi/checkpoint lama; RMSE historis5,5247tidak dijamin terulang persis pada GPU/precision/dependency berbeda.

E41 memakai satu RoBERTa yang sama dan meminta `output_hidden_states=True`. `hidden_states[0]` adalah embeddings, kemudian indeks1–L adalah keluaran Transformer. Hanya empat keluaran Transformer terakhir dipilih. Pada12layer, indeks encoder yang dipilih8–11atau layer ke9–12; embeddings tidak ikut dicampur. Jumlah layer aktual harus mencukupi.

Empat logit global diinisialisasi nol; softmax membuat bobot awal0,25masing-masing. Setiap token pada posisi sama digabung dengan jumlah berbobot, sehingga keluaran tetap768dimensi. Empat logit ini merupakan seluruh tambahan parameter; tidak ada skala gamma, normalisasi tambahan atau encoder baru. Bobot dipelajari lewat loss skor guru, dengan LR1e-4. Bobot global sama untuk semua jawaban pada checkpoint tertentu, bukan attention khusus per siswa. Respons dan referensi memakai hasil campuran yang sama sebelum BiGRU/proyeksi/co-attention; CLS referensi untuk inspeksi tetap berasal dari layer terakhir.

| Bagian | E30 acuan | E41 |
|---|---|---|
| Representasi RoBERTa untuk token | Layer terakhir | Campuran berbobot empat layer terakhir |
| Parameter campuran | Tidak ada | Empat logit, softmax, awal0,25 |
| Dimensi token ke BiGRU | 768 | 768 |
| BiGRU | 64perarah,keluaran128 | Sama |
| Co-attention/matching/pooling | Respons–referensi,640→128,attention128→64→1 | Sama |
| Fusion dan head | RoBERTa768+BGE256→concat1024→MLP256→GELU→Dropout0,1→1 | Sama |
| LR seluruh encoder | Seragam5e-6 | Sama |
| LR komponen baru/pooling/head/proyeksi | 1e-4 | Sama,ditambah scalar_mix1e-4 |

BiGRU128perarah E39 dan LLRD E40 tidak dibawa. RoBERTa fine-tuned; BGE-M3 frozen1024→proyeksi256+LayerNorm. Referensi768→128dan ringkasan128→768+LayerNorm mengikuti E30. Tidak ada Ridge atau pemeriksaan cosine.

## Menjalankan dan membaca output

Gunakan runtime baru T4 GPU untuk setiap notebook, Run all, lalu unggah `fix_dataset_300.xlsx`. Masing-masing notebook184cell/89codecell; seluruh kode yang menentukan model/training berada langsung dalam kernel, tanpa cell magic,worker tersembunyi,clone repo atau Drive. Nama checkpoint dan revision terkunci seperti E30.

Section5memuat seluruh hyperparameter. E41 menambahkan `SCALAR_MIX_LAST_LAYERS=4` dan `SCALAR_MIX_LR=1e-4`. Section19memuat kelas `ScalarMix` dan `SingleScoreModel`; section22menunjukkan optimizer dan loss. Empat parameter campuran mempunyai kelompok LR tersendiri; semua parameter trainable diperiksa masuk tepat satu kelompok.

X terdiri dari `soal`, `jawaban_referensi`, `response_text`; y adalah `rater`integer0–100dan tidak masuk encoder. Split_seed2026,TRAIN_SEED42dengan seed per fold=seed+nomor fold,holdout20%,grouping,panjang token dan windowing tetap E30. S3 saja,5fold sekali,maksimal25epoch/fold,patience5,1refit+1test. Tidak ada loop tiga seed,ensemble atau kontrol otomatis. AdamW,weight decay0,01,warmup10%,scheduler linear,clipping1,batch efektif8/micro1dan MSE target0–1tetap E30. Refit=ceil median best epoch lima fold. Jangan memilih konfigurasi atau epoch dari test lama yang telah berulang kali dilihat; hasil test selanjutnya bersifat eksploratif.

Setiap notebook menghasilkan lima Excel standar dan ZIP yang berbeda. Pada02,ringkasan co-attention128komponen,pooling768dan fusion1024. Pada05→Pengaturan Eksperimen tersimpan resep aktual. E41 menambahkan sheet **Bobot Scalar Mix**, berisi4baris×(5fold+1refit)=24baris, dengan indeks layer,logit,bobot softmax dan bobot awal. Bobot fold diambil setelah epoch terbaik dipulihkan, sesuai checkpoint validasi yang dinilai; refit memakai checkpoint akhir pilihan CV. Nilai bobot bukan skor siswa atau penjelasan kausal.

Folder/hash config/kode berbeda, sehingga cache dan checkpoint tidak tercampur. Resume memakai progres lokal yang cocok; runtime yang dihapus juga menghapus file lokal. Unduh ZIP sebelum sesi berakhir. Dua notebook dijalankan terpisah; pengguna boleh menjalankan E41 saja atau menjalankan ulang E30 acuan untuk pembandingan pada lingkungan yang sama.

## Rujukan dan batas klaim

- Peters dkk.(2018), [Deep Contextualized Word Representations / ELMo](https://aclanthology.org/N18-1202/), menjadi dasar campuran layer berbobot yang dipelajari. ELMo memakai arsitektur berbeda dan skala gamma. E41 mengadaptasi bagian campuran softmax saja pada empat layer RoBERTa terakhir tanpa gamma/normalisasi tambahan.
- Tenney dkk.(2019), [BERT Rediscovers the Classical NLP Pipeline](https://aclanthology.org/P19-1452/), mendukung bahwa informasi bahasa tersebar pada kedalaman berbeda; bukan bukti empat layer terakhir optimal untuk AES ini.
- Rujukan RoBERTa,BGE-M3,GRU,BiGRU,co-attention AES,pooling,LayerNorm,GELU dan AdamW tetap tersedia pada bagian akhir kedua notebook.

E41 belum memiliki hasil penelitian dan tidak dijamin mengalahkan E30. Patokan historis mean RMSE CV E30=5,9491; nilai tiap fold juga diperiksa. [Verifikasi teknis](../verifikasi/verification_e41_dan_e30_native.md) membedakan fixture CPU dari training penelitian GPU.
