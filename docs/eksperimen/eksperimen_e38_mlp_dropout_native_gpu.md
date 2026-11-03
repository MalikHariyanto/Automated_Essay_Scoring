# E38 · E30 S3 dengan dropout MLP 0,2

E38 menguji apakah regularisasi MLP tambahan membantu generalisasi E30. E36 (`3e-6`) dan E37 (`7e-6`) belum mengalahkan encoder E30 (`5e-6`), sehingga LR kembali ke E30. Satu perlakuan baru adalah probabilitas dropout hidden MLP S3, dari 0,1 menjadi 0,2. Ini hipotesis, belum ada hasil penelitian E38.

[Buka Colab](https://colab.research.google.com/github/MalikHariyanto/Automated_Essay_Scoring/blob/codex/aes-learning/notebooks/experiments/AES_300_E38_E30_S3_MLPDropout02_Native_Learnable_GPU.ipynb) · [Source notebook](../../notebooks/experiments/AES_300_E38_E30_S3_MLPDropout02_Native_Learnable_GPU.ipynb).

| Pengaturan | Acuan E30 | E38 |
|---|---|---|
| Dropout MLP S3 setelah GELU | 0,1 | **0,2** |
| LR encoder RoBERTa S3 | 5e-6 | 5e-6 |
| LR MLP/proyeksi BGE/BiGRU/pooling/norm | 1e-4 | 1e-4 |
| Input | soal, response_text, jawaban_referensi | Sama |
| Target | rater 0–100, tidak masuk encoder | Sama |
| Arsitektur | BiGRU + co-attention + attention pooling + concat + MLP | Sama |
| BGE-M3 | Frozen | Frozen |
| Fold/epoch/patience | 5 / maksimal25 / 5 | Sama |

## Letak perubahan yang terlihat

Section5 menampilkan `S3_MLP_DROPOUT = 0.2`. Nilai ini diteruskan melalui `cfg['s3_mlp_dropout']` ke constructor `SingleScoreModel`, khusus `self.head[2]` pada S3. Urutan MLP adalah Linear1024→256, GELU, Dropout0.2, Linear256→1. Dropout menonaktifkan aktivasi secara acak saat training; bobot tidak dihapus. Saat `model.eval()`, dropout dinonaktifkan.

Parameter lama `DROPOUT=0.1` dipertahankan untuk constructor historis; bukan pengaturan dropout internal pretrained RoBERTa. `self.dropout` historis tidak dipakai dalam forward S3. Dropout RoBERTa berasal dari konfigurasi checkpoint yang sama dengan E30; BiGRU/co-attention tidak ditambah dropout baru. State awal dan jumlah parameter tetap sama. Uji membandingkan semua modul Dropout: hanya `head.2.p` yang berubah.

Probability dropout tidak berada pada state_dict bobot. Karena itu konfigurasi checkpoint juga menyimpan p dan loader menolak `s3_mlp_dropout` yang berbeda. Config/cache/code hash dan workspace E38 terpisah dari eksperimen lama; tidak memakai hasil lama sebagai hasil E38.

## Protokol dan cara menjalankan

S3 saja, lima fold development sekali, maksimal25epoch/fold, early stopping patience5. Refit sekali pada seluruh development memakai ceil median best epoch lima fold, lalu test sekali. TRAIN_SEED42 hanya satu konstanta pengacakan; tidak ada loop tiga seed, pencarian beberapa dropout, atau pengulangan kontrol E30. Split_seed2026, pengelompokan, ukuran holdout, token/window, batch, optimizer dan scheduler mengikuti E30. Total5fit CV+1refit.

Runtime Colab baru → T4 GPU → Run all → upload `fix_dataset_300.xlsx`. Format native E35/E36 berisi184cell/89codecell; semua fungsi dan pemanggilan berupa Python biasa dalam kernel. Tidak ada magic/worker tersembunyi, pemeriksaan cosine, Ridge, clone repo atau Drive. S1/S2 tidak dilatih/dimuat; kedua cabang encoder tetap ada di dalam S3.

Output tetap lima Excel standar. `05_Riwayat_Pelatihan.xlsx` → `Pengaturan Eksperimen` mencatat `Dropout MLP S3`, parameter dropout lama dan seed training. `Model yang Dibandingkan` juga mencantumkan probabilitas MLP aktual. ZIP akhir `Hasil_AES_300_E38_E30_S3_MLPDropout02_Native_Learnable_GPU.zip`.

File dan checkpoint berada pada disk runtime lokal. Run ulang dalam runtime yang sama melanjutkan hasil/checkpoint cocok hash; penghapusan runtime menghilangkan file lokal. Unduh ZIP sebelum sesi berakhir.

## Interpretasi dan rujukan

Nilai utama tetap mean RMSE CV dan perubahan setiap fold; test untuk laporan akhir. Loss training E38 bisa lebih tinggi akibat regularisasi, sehingga loss training paling kecil bukan target pemilihan model. Peningkatan CV/test tidak dijamin. Data test lama telah dilihat berulang kali; hasil berikutnya tetap eksploratif, bukan konfirmasi pada test baru. Tidak menambah epoch refit atau memilih resep baru berdasarkan hasil test.

Referensi komponen E30 tetap ada pada notebook. Tambahan: Srivastava dkk.(2014), [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](https://jmlr.org/papers/v15/srivastava14a.html), JMLR15:1929–1958. Paper mendukung metode dropout; p=0,2 adalah pilihan ablasi proyek, bukan nilai optimal yang diwajibkan paper.

[Verifikasi teknis](../verifikasi/verification_e38_native.md). Pengujian fixture tidak boleh dimasukkan sebagai hasil penelitian atau ranking HTML.
