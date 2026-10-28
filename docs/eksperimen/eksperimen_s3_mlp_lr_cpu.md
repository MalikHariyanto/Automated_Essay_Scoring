# S3 MLP kandidat B · LR RoBERTa 5e-6 · CPU

[Buka di Colab](https://colab.research.google.com/github/MalikHariyanto/Automated_Essay_Scoring/blob/codex/aes-learning/notebooks/experiments/AES_300_S3_MLP_LR5e6_CPU.ipynb) · [Notebook](../../notebooks/experiments/AES_300_S3_MLP_LR5e6_CPU.ipynb)

Eksperimen terpisah dari `AES_300_S3_MLP.ipynb`, menggunakan **CPU** dan jawaban siswa saja. Tidak memakai LoRA, konteks pair, Ridge, atau fitur linguistik tambahan. Satu perubahan hyperparameter model: **encoder RoBERTa S3 memakai LR 5e-6**. S1 tetap memakai LR encoder 1e-5; proyeksi BGE dan head tetap 1e-4. Nilai tersebut adalah LR dasar sebelum warmup dan scheduler linear, bukan angka konstan sepanjang training.

| Skenario | Encoder | Layer prediksi | LR encoder | LR proyeksi | LR head |
| --- | --- | --- | --- | --- | --- |
| S1 | RoBERTa fine-tuned | Linear 768→1 | 1e-5 | — | 1e-4 |
| S2 | BGE-M3 frozen | MLP 1024→256→GELU→dropout→1 | Frozen | — | 1e-4 |
| S3 | RoBERTa fine-tuned + BGE-M3 frozen | LayerNorm, proyeksi BGE 1024→256 + LayerNorm, concat 1024→MLP256→GELU→dropout→1 | **5e-6 untuk RoBERTa** | 1e-4 | 1e-4 |

`s3_encoder_lr` dipilih hanya ketika `model.scenario=='S3'` pada `optimizer_groups`. BGE-M3 frozen tidak menerima pembaruan optimizer. Dropout 0,1, weight decay 0,01, warmup 10%, clipping 1, maksimum 25 epoch/fold, patience 5, microbatch 1/batch efektif 8 tetap seperti acuan. Nama model dan revision tokenizer/bobot juga tetap. Lima fold berada di development 80%; test 20% dipakai setelah pilihan epoch dikunci. Semua tiga skenario tetap dijalankan untuk menjaga laporan lengkap seperti acuan.

## Menjalankan

1. Buka notebook di atas, pilih **Runtime → Change runtime type → Hardware accelerator: None**.
2. Jalankan dari atas. Sambungkan Drive, lalu pilih `fix_dataset_300.xlsx` bila diminta. Dataset memiliki satu nilai guru 0–100. Header tetap `Email address`, `response_number`, `response_text`, `soal`, `jawaban_referensi`, `rater`; hanya `response_text` masuk encoder.
3. Pengaturan layar memperlihatkan CPU aktif dan LR S1/S2/S3. `device='cpu'` berlaku meskipun CUDA tersedia; tidak ada guard wajib GPU. Float32 digunakan saat training; AMP tidak aktif.
4. Data/hasil/checkpoint berada di **MyDrive/AES_300_S3_MLP_LR5e6_CPU_Privat**. Notebook acuan dan folder eksperimen lama tidak ditimpa. Jika runtime terputus, jalankan notebook yang sama dari atas dengan konfigurasi/versi numerik yang sama untuk resume.
5. Setelah prediksi dan ekspor selesai, cell terakhir menyusun dan mengunduh `Hasil_AES_Ringkas.zip` berisi lima Excel. Tidak perlu file source proyek dari GitHub untuk menjalankan notebook.

Full fine-tuning RoBERTa di CPU dapat jauh lebih lama daripada GPU; batas epoch dan batch tidak dikurangi diam-diam. Setup menggunakan overlay teks baru, mempertahankan torch runtime, memasang dependensi tanpa menarik ulang torch, serta menghindari import torchvision/TF/Flax untuk tugas teks. PEFT tidak dipasang dan tidak diimpor. Bila PyTorch belum tersedia pada runtime, setup memberi alasan dan tidak mengklaim siap.

## Penyimpanan dan hasil

Progres disimpan setiap epoch, termasuk state model, bobot terbaik, optimizer, scheduler, RNG dan riwayat. **Tidak ada auto-delete atau Drive API cleanup.** Semua file epoch/fold tetap ada untuk pembersihan manual. Pertahankan `final.pt` dan JSON hasil; untuk tahap aktif, pertahankan JSON progres dan dua snapshot `.pt` yang dirujuknya. JSON bukan pengganti bobot. Model S3 tetap dilatih penuh sehingga checkpoint lebih besar daripada adapter-only LoRA.

Lima Excel seperti acuan: data/tokenisasi, vektor jawaban, vektor kata/kalimat, prediksi/evaluasi, riwayat pelatihan. Hasil tetap menyertakan nilai guru, prediksi mentah S1/S2/S3 dan pembulatan kategori; RMSE/Pearson memakai nilai mentah, QWK/Gwet AC2 memakai salinan kategori. LR encoder S1 dan S3 ditampilkan terpisah di pengaturan laporan pelatihan. Vektor aktual semua komponen disimpan; tidak dibuat vektor fusion per kata yang tidak dihasilkan arsitektur.

## Batas perbandingan

Angka acuan CV S3 6,0516 poin berasal dari run pengguna sebelumnya. Tuning tidak menjamin RMSE lebih kecil. Selain LR, perpindahan dari GPU AMP ke CPU float32 mengubah lingkungan numerik; perbedaan hasil tidak dapat seluruhnya dianggap efek LR. Untuk pembandingan yang lebih terkendali, ulangi acuan 1e-5 dan kandidat 5e-6 pada perangkat, precision, seed dan split yang sama. Gunakan CV untuk memilih kandidat, tanpa memilih LR berdasarkan test lama yang sudah berulang kali dilihat. Belum ada hasil penelitian CPU baru sebelum notebook dijalankan pada checkpoint asli dan dataset pengguna.

[Bukti pengujian teknis](../verifikasi/verification_s3_mlp_lr_cpu.md).
