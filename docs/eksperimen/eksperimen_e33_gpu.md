# E33 · E30 asli dengan LR MLP S3 5e-5

Pengguna memilih rekomendasi pertama: turunkan hanya LR MLP S3 dari1e-4 menjadi5e-5. Source acuan ialah notebook E30 asli pada commit `6b3c52569851b06720cb80a0e9534e4f46b70eaf`, bukan notebook E30 Ridge. E33 meneruskan penomoran setelah E31/E32 yang sudah tersedia.

| Kelompok | E30 | E33 |
|---|---:|---:|
| RoBERTa S1/S3 | 5e-6 | 5e-6 |
| Linear S1 | 1e-4 | 1e-4 |
| MLP S2 | 1e-4 | 1e-4 |
| **MLP S3 `head.*`** | **1e-4** | **5e-5** |
| BiGRU/co-attention/pooling/norm S3 | 1e-4 | 1e-4 |
| Proyeksi BGE | 1e-4 | 1e-4 |

Arsitektur S3: tiga input → RoBERTa fine-tuned → BiGRU → co-attention respons–referensi → additive attention pooling → concat proyeksi BGE frozen → MLP. Label tunggal `rater`0–100. BGE tetap frozen. Tidak ada Ridge.

Notebook47cell/30codecell mengikuti format E30, termasuk setup/upload, inspeksi token/vektor awal, training S1/S2/S3, prediksi, fitur final dan unduhan lima Excel/ZIP. GPU Colab biasa, tanpa clone/Drive. Semua skenario dilatih baru; S1/S2 menjadi kontrol dengan resep E30, bukan hasil lama yang disalin.

5fold CV, maksimal25epoch per fold, patience5 berdasarkan RMSE validasi; checkpoint terbaik dipulihkan. Refit model baru pada240development memakai ceil median epoch terbaik lima fold; evaluasi60test sesudah refit. Seed42/split2026, microbatch1/efektif8, MSE target0–1, dropout0,1, warmup10%+linear scheduler, AdamW decay0,01 dan clipping1 mengikuti acuan.

Kode runtime sama dengan E30 setelah normalisasi nama folder dan hash, kecuali dua cell konfigurasi: default `s3_mlp_lr` dan konfigurasi pengguna/log identitas. Arsitektur, optimizer, loss, CV/refit, tokenisasi dan exporter tidak diubah. Nama folder/ZIP, fingerprint, versi resep dan dokumentasi E33 terpisah agar checkpoint/cache tidak tertukar. Metadata mencatat sumber E30 dan perubahan satu kelompok LR secara eksplisit.

Belum ada hasil penelitian E33. Kandidat5e-5 berasal dari hipotesis ablation, bukan nilai optimal yang sudah terbukti. Bandingkan kontrol E30 dan E33 dengan fold/seed yang sama melalui CV; test historis sudah sering dilihat sehingga perbandingan berikutnya eksploratif.
