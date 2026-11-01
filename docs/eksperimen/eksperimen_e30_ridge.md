# E30 + Ridge · versi aktif

Tiga input `soal`, `jawaban_referensi`, `response_text`; label guru `rater`0–100.

RoBERTa fine-tuned → BiGRU token respons64/arah → co-attention respons–referensi → attention pooling128 → proyeksi768 → LayerNorm. BGE-M3 frozen1024 → proyeksi256 → LayerNorm. Concat1024 → Ridge untuk prediksi akhir. MLP bantu1024→256→1 tetap dilatih dan disimpan sebagai pembanding pada checkpoint sama.

Source E30 asli pada commit `47ddb8280592231fbb682f2e6a8b3c5735d185fb`, path `notebooks/experiments/AES_300_E30_E26_BiGRU_CoAttention_Pooling_GPU.ipynb`. Arsitektur, window, optimizer dan fungsi training dipertahankan. LR encoder5e-6; MLP, BiGRU, co-attention, attention pooling, proyeksi/norm1e-4. Ini resep E30, bukan LR MLP5e-6 dari E29.

## Training dan evaluasi

1. Split kelompok development/test dan lima fold sesuai keluarga E21–E32.
2. Fold1/5 sampai5/5: model baru, maksimal25epoch; RMSE validasi MLP memilih checkpoint. Early stopping setelah5epoch berturut-turut tanpa rekor RMSE baru. Pulihkan bobot epoch terbaik.
3. Bekukan representasi terlatih untuk ekstraksi fitur. Fit Ridge hanya pada train fold, evaluasi validation; MLP dari checkpoint sama menjadi kontrol.
4. Alpha Ridge1,0 ditetapkan sebelum CV, tidak dipilih melalui test atau pencarian alpha. Ini pilihan implementasi, bukan alpha optimal yang telah dibuktikan.
5. Refit model baru pada seluruh development selama ceil median epoch terbaik lima fold. Final refit tanpa early stopping test. Fit Ridge pada fitur development, lalu evaluasi test.

Total lima training CV dan satu refit; tidak ada16fit atau nested CV. Ridge tidak mempunyai epoch, LR atau patience; aturan25/5 berlaku pada training neural E30 dengan MLP bantu. Validation turut memilih epoch sehingga CV bukan evaluasi yang independen dari seleksi checkpoint. Test historis telah sering dilihat dan tetap eksploratif.

Notebook mandiri: T4 GPU → Run all → unggah `fix_dataset_300.xlsx`. Tanpa clone/Drive. Checkpoint/cache lokal memiliki kontrak kode/data/config/label train dan validation untuk resume. Cache E29 atau resep lama tidak dipakai ulang. Lima Excel dan ZIP di akhir; checkpoint final besar tetap lokal, dapat diunduh dengan `DOWNLOAD_FULL_ENCODER=True`. NPZ Ridge saja tidak cukup untuk prediksi teks baru.

Belum ada hasil penelitian E30 Ridge. Tes dengan fixture dan random Transformer hanya bukti alur teknis, bukan hasil akurasi penelitian atau bukti eksekusi GPU Colab. E29 Ridge lama dipertahankan sebagai arsip, bukan versi yang dipilih pengguna sekarang.
