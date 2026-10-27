# E21 dan E22 — web Colab biasa, GPU, tanpa Google Drive

Buka `AES_300_E21_Tiga_Input_GPU.ipynb` atau `AES_300_E22_Tiga_Input_ContextPair_GPU.ipynb` langsung pada website Colab. Pilih **Runtime → Change runtime type → T4 GPU**, lalu **Runtime → Run all**. Cell setup menampilkan **Choose Files**; pilih `fix_dataset_300.xlsx` dari komputer dan tunggu upload selesai. Bila dataset sudah ada di filesystem, tidak diminta lagi. Cell terakhir mengunduh ZIP lima Excel melalui browser. Tidak perlu ekstensi/editor lain, mount Drive atau perubahan kode tambahan.

| Pengaturan | E21 | E22 |
| --- | --- | --- |
| Input seluruh S1/S2/S3 | `soal`, `jawaban_referensi`, `response_text` | Sama |
| Label | `rater`, satu skor guru0–100 | Sama |
| Format tokenizer | Satu urutan tiga teks | Pasangan (soal + referensi, jawaban siswa) |
| S1 | RoBERTa fine-tuned + head linear | Sama |
| S2 | BGE-M3 frozen + MLP | Sama |
| S3 | Fusion RoBERTa fine-tuned + BGE frozen + MLP | Sama |
| LR encoder RoBERTa S1/S3 | 5e-6 | 5e-6 |
| LR head / proyeksi | 1e-4 / 1e-4 | Sama |
| Training | GPU CUDA, AMP sesuai perangkat | Sama |
| Drive | Tidak mount/autosave | Sama |

S1 tetap linear sesuai pilihan pengguna. RoBERTa full fine-tuning; tidak memakai LoRA. BGE frozen hanya dihitung sebagai fitur, sedangkan MLP/proyeksi tetap dilatih. Seluruh skenario membaca ketiga teks; `rater` tidak masuk encoder. Pengelompokan email/duplikat, normalisasi label0–1 untuk MSE, dan inverse transform prediksi ke poin0–100 mengikuti E6.

Perbedaan metode antara E21/E22 hanya format single sequence versus context pair. Konteks192token, soal64token, prefix, isi dan starts window jawaban disamakan dengan mencadangkan overhead pasangan pada kedua format. RoBERTa512/BGE1024, overlap128. Pasangan RoBERTa/XLM-R memakai separator khusus, tanpa token-type embedding A/B baru. Ini bukan jaminan RMSE turun. Rincian metode bersama ada di [dokumen tiga input](eksperimen_e21_e22_tiga_input_cpu.md).

Seed42, split_seed2026, development/test80/20, lima fold development, maksimal25epoch/fold, patience5, microbatch1/batch efektif8. AdamW, warmup10%, scheduler linear dan pemilihan refit ceil median epoch CV sama. GPU wajib; guard CUDA berjalan sebelum encoder dimuat. T4 memakai AMP fp16; GPU yang mendukung bf16 memakai bf16. Torch CUDA bawaan Colab dipertahankan oleh setup; tidak diganti wheel CPU. Perbandingan E21/E22 sebaiknya memakai perangkat/versi/seed sama. CPU dan GPU tidak dijanjikan menghasilkan prediksi identik.

Default folder lokal: `/content/AES_300_E21_Tiga_Input_GPU` dan `/content/AES_300_E22_Tiga_Input_ContextPair_GPU`. Dataset, cache, checkpoint dua slot, metadata dan Excel hanya berada di disk runtime. File lokal dapat hilang bila runtime dihapus/diganti; unduh sebelum sesi berakhir. ZIP lima Excel tidak berisi bobot model; `S1/final.pt`, `S2/final.pt`, `S3/final.pt` dan metadata dapat diunduh manual dari panel Files bila diperlukan lagi. Tidak ada backup Drive tersembunyi.

Upload memakai `google.colab.files.upload()` pada proses notebook utama, bukan worker perhitungan. Jika tombol abu-abu berasal dari output sesi lama, jalankan ulang cell setup di sesi browser saat ini. Jika unduhan otomatis diblokir browser, path ZIP tercetak pada cell terakhir dan file dapat diunduh dari panel Files/Berkas. Jangan membagikan notebook yang berisi teks/skor individu melalui repository publik.

Builder GPU diturunkan dari source bersih CPU yang dibuat builder E21/E22. Sumber dasar E6 dipin pada8380355; notebook E6 yang disimpan ulang pengguna tidak diubah. Semua fungsi model, tokenizer tiga teks, split, loss, CV/test dan laporan dipertahankan; varian GPU mengubah pemilihan perangkat, guard, AMP yang sudah ada, identitas dan petunjuk runtime. Metadata mencatat hash sumber CPU.

## Verifikasi

25 pemeriksaan versi GPU lulus (0 failed, 0 skipped, 0 xfailed; satu warning pustaka). Guard perangkat diperiksa agar versi publik meminta CUDA dan menolak kondisi CUDA tidak tersedia. Model, input tiga teks, target, gradien, LR, cache, cakupan token, upload browser termasuk pembatalan, identitas source dan kesetaraan E21/E22 diperiksa. Kedua notebook menjalankan seluruh30codecell,15fitCV+3refit, scoring, fitur dan lima Excel, lalu kernel baru membaca hasil lokal yang sama. Eksekusi penuh tersebut menggunakan Transformer kecil berbobot acak pada CPU dengan guard/perangkat diganti khusus fixture; bukan eksekusi GPU asli atau hasil penelitian300jawaban. Upload browser diuji dengan modul Colab tiruan; sesi browser Colab pengguna tidak dioperasikan dari mesin ini.

Varian CPU web melewati23pemeriksaan penuh dan dua pemeriksaan guard tambahan. GPU hardware/CUDA kernels/AMP asli belum dijalankan di lingkungan pemeriksaan ini. Pengujian tokenizer pretrained tambahan sebelumnya terhalang proxy403; tidak diklaim telah menjalankan bobot pretrained penuh. Bukti command, source hash, versi, hasil XML dan bukti notebook disimpan privat pada `outputs/e21_e22_gpu_web_*` dan `outputs/e21_e22_web_*`.
