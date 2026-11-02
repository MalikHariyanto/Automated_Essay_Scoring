# E35 · E30 dengan LR kelompok layer S3 5e-5, native Colab

Permintaan pengguna terbaru: notebook mudah dipelajari dengan section eksplisit, tanpa cell magic, namespace worker atau kode yang dieksekusi melalui string. Kode berjalan langsung di kernel Colab. Nama model, kolom X/y, hyperparameter, kelas arsitektur dan pemanggilan training terpisah. Ini perubahan format eksekusi yang diotorisasi pengguna, menggantikan format worker sebelumnya untuk E35; source dan hasil lama dipertahankan.

## Resep eksperimen

| Kelompok | E30 | E35 |
|---|---:|---:|
| Encoder RoBERTa S1/S3 | 5e-6 | 5e-6 |
| Linear S1 | 1e-4 | 1e-4 |
| MLP S2 | 1e-4 | 1e-4 |
| MLP S3 | 1e-4 | 1e-4 |
| sequence_pool.* dan roberta_norm.* S3 | 1e-4 | **5e-5** |
| Proyeksi BGE projection.* | 1e-4 | 1e-4 |

Kelompok LR yang diturunkan mencakup BiGRU, proyeksi referensi, co-attention/matching, attention pooling, proyeksi ringkasan128→768, serta LayerNorm tambahan RoBERTa. Ini ablasi satu kelompok optimizer, bukan satu layer individual. MLP tetap prediktor akhir, BGE frozen dan RoBERTa full fine-tuning. Tidak ada Ridge. Fungsi/kolom inspeksi cosine dihapus; vektor serta perbandingan komponen tetap diekspor.

Arsitektur S3: tiga input → RoBERTa → token respons BiGRU64perarah/128; referensi proyeksi768→128; scaled dot product/√128 → co-attention dua arah → matching640→128 → additive attention128→64→1 → weighted sum128 → proyeksi768 → LayerNorm. Cabang BGE1024→Linear256+LayerNorm. Concat1024 → Linear256 → GELU → Dropout0,1 → Linear1. RoBERTa membaca konteks soal/referensi; mask memastikan posisi respons/referensi yang sesuai.

5fold CV, maksimal25epoch, patience5 berdasarkan minimum RMSE validasi; bobot terbaik dipulihkan. Refit model baru menggunakan ceil median best epoch5fold pada seluruh development. Test dipisahkan. Seed42/split2026, microbatch1/efektif8, MSE target0–1, AdamW decay0,01, warmup10%+linear scheduler dan clipping1 sama dengan kontrol E30. S1/S2 dilatih baru; bukan reuse hasil lama.

## Peta belajar

| Dicari pengguna | Bagian notebook | Cell kode ke- |
|---|---|---:|
| Nama pretrained RoBERTa/BGE | 3 · Nama model pretrained | 7 |
| Kolom input/target dan file Excel | 4 · Kolom input X, target y | 9 |
| Learning rate/epoch/fold/patience | 5 · Hyperparameter | 11 |
| Menyalakan training/inspeksi | 6 · Kontrol eksekusi | 13 |
| X dan y aktual | 11 · Variabel X dan y | 33 |
| Kelas model dan head MLP | 19 · Arsitektur | 111 |
| Co-attention dan attention pooling | 19 · Arsitektur | 109 |
| Pembagian LR parameter | 22 · Optimizer | 127 |
| Loss/backward/early stopping | 22 · Training | 133 |
| Pemanggilan CV/refit/test | 24 · Menjalankan training | 152 |
| ZIP lima Excel | 28 · Unduh hasil | 183 |

`X` adalah DataFrame tiga kolom setelah cleaning; `y` adalah Series skor guru. Fungsi training membawa nilai yang sama dalam Bundle, bersama ID/kelompok yang diperlukan split: bundle.texts adalah respons, contexts_from_bundle(bundle) adalah soal/referensi, bundle.y adalah target. Pemilihan kolom di bagian4 benar-benar diterapkan pada pembacaan Excel; nilai guru tidak masuk encoder.

## Runtime dan hasil

Pilih runtime Colab baru dengan T4 GPU → Run all → unggah fix_dataset_300.xlsx. Setup memakai subprocess pip biasa untuk pustaka yang diperlukan, tanpa shell/cell magic atau Python worker. Torch/CUDA dan pustaka numerik yang sudah cocok dipertahankan agar kernel tidak memakai ABI campuran. Jika pustaka dasar tidak sesuai atau Transformers versi berbeda telah diimpor, setup memberi petunjuk restart/runtime baru sebelum training; tidak berpindah diam-diam ke CPU. Versi aktual dicatat dalam hasil sehingga pengaruh lingkungan dapat dinilai. Pengujian teknis tidak membuktikan bahwa semua versi Colab akan menghasilkan trajectory training identik.

Notebook mandiri; tidak membutuhkan clone/Drive/file Python pendamping. Checkpoint dua slot lokal tetap tersedia; runtime dihapus berarti progres hilang. ZIP akhir berisi lima Excel standar, termasuk tokenisasi, seluruh vektor, prediksi/evaluasi dan riwayat fold/epoch. Bobot final besar tidak otomatis dimasukkan ke ZIP.

Source E30 dipin ke7f3363643e6bf402a293de069bcfc23d85b19c89. Metode inti tokenizer/model/optimizer/fit/CV diuji sama secara AST dan perilaku forward/update awal; pengaturan E35 mengubah LR pooling_norm saja. Namun belum ada hasil penelitian E35 pada data guru/pretrained GPU. Nilai LR5e-5 adalah hipotesis proyek. Gunakan development untuk memilih kandidat; test historis sudah sering dilihat dan hasil lanjutannya eksploratif.
