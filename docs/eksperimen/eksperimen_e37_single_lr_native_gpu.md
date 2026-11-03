# E37 aktif · satu resep LR encoder7e-6 · lima fold CV

Pengguna membatalkan desain paired tiga seed dan meminta lima fold saja. Notebook aktif langsung menguji kandidat7e-6; tidak melatih ulang kontrol E30 dan tidak mencari LR/seed terbaik. Rancangan paired sebelumnya tetap arsip, bukan default.

[Buka Colab](https://colab.research.google.com/github/MalikHariyanto/Automated_Essay_Scoring/blob/codex/aes-learning/notebooks/experiments/AES_300_E37_E30_S3_EncoderLR7e6_Native_Learnable_GPU.ipynb) · [Source](../../notebooks/experiments/AES_300_E37_E30_S3_EncoderLR7e6_Native_Learnable_GPU.ipynb).

Satu perlakuan dari E30: encoder RoBERTa S3 berubah5e-6→7e-6. MLP, proyeksi BGE, BiGRU/co-attention/pooling/norm tetap1e-4. Tiga input `soal`, `response_text`, `jawaban_referensi` → RoBERTa fine-tuned → BiGRU → co-attention respons–referensi → attention pooling → concat proyeksi BGE frozen → MLP → satu skor0–100. Label `rater` membimbing MSE, tidak masuk encoder. Tidak ada S1/S2 terpisah, Ridge atau pemeriksaan cosine.

**5fold CV sekali, maksimal25epoch per fold, patience5.** Setelah semua fold selesai,1refit pada development memakai ceil median best epoch5fold, lalu1test. TRAIN_SEED42hanya konstanta pengacakan agar perbandingan dapat diulang, bukan loop tiga seed. Split_seed2026,kelompok/holdout/fold dan pengaturan lain mengikuti E30. Uji LR7e-6merupakan hipotesis, bukan jaminan peningkatan. Test lama sudah berulang kali diamati sehingga perbandingan tetap eksploratif.

Format native E35/E36:184cell/89codecell, fungsi Python terpisah serta section X/y, nama model, hyperparameter, training dan evaluasi yang jelas. Tanpa magic/worker tersembunyi, clone repo atau Drive. Output tetap lima workbook standar dan ZIP `Hasil_AES_300_E37_E30_S3_EncoderLR7e6_Native_Learnable_GPU.zip`.

Hentikan training notebook paired lama dan buka link terbaru ini. Runtime T4GPU → Run all → upload `fix_dataset_300.xlsx`. Workspace, code hash dan cache baru berbeda dari versi paired sehingga hasil kontrol/seed lama tidak dipakai sebagai hasil kandidat. Tidak mengubah/menghapus data sumber atau artefak eksperimen lama. Checkpoint dua slot dan resume runtime yang sama mengikuti E36; file lokal hilang bila runtime dihapus.

Rujukan komponen E30 tetap terlihat di notebook. Nilai LR7e-6adalah pilihan eksperimen, bukan angka yang diwajibkan paper. [Verifikasi teknis](../verifikasi/verification_e37_single_native.md).
