# E8 · Konteks pair + MLP + LR S3 5e-6 · GPU

[Buka Colab](https://colab.research.google.com/github/MalikHariyanto/Automated_Essay_Scoring/blob/codex/aes-learning/notebooks/experiments/AES_300_E8_ContextPair_LR5e6_GPU.ipynb) · [Notebook](../notebooks/experiments/AES_300_E8_ContextPair_LR5e6_GPU.ipynb)

Pilih Runtime → Change runtime type → T4 GPU, lalu jalankan seluruh notebook dari atas. Unggah `fix_dataset_300.xlsx` jika diminta. Kode berdiri sendiri; tidak mengambil source proyek atau data siswa dari GitHub. Setup mengisolasi pustaka teks dan mempertahankan PyTorch runtime; tidak memerlukan PEFT.

E8 menurunkan LR encoder RoBERTa S3 dari 1e-5 pada konteks pair E4 menjadi 5e-6. Pembanding faktorialnya adalah E4 (konteks pair/LR1e-5), E6 (jawaban saja/LR5e-6), dan MLP awal (jawaban saja/LR1e-5). E7 merupakan tiga seed; E8 ini satu seed dan belum mempunyai hasil penelitian.

| Skenario | Input dan model | LR encoder | LR head/proyeksi |
|---|---|---|---|
| S1 | Jawaban → RoBERTa full fine-tuning → dropout → linear | 1e-5 | 1e-4 |
| S2 | Jawaban → BGE frozen → MLP | Encoder frozen | 1e-4 |
| S3 | Konteks pair → RoBERTa full fine-tuning; jawaban → BGE frozen; proyeksi + LayerNorm + concat → MLP | RoBERTa5e-6 | 1e-4 |

Konteks maksimal192token termasuk prefix, soal maksimal64token, window RoBERTa512 termasuk4special pasangan, overlap128. Jawaban panjang dicakup lewat window; truncation konteks dilaporkan. BGE tetap menerima jawaban saja. Tidak menambahkan LoRA, fitur linguistik atau membekukan encoder RoBERTa.

Dataset satu skor guru0–100, seed42, split_seed2026, development/test80/20, lima fold pada development, maksimal25epoch/fold, patience5, microbatch1/batch efektif8. Warmup10%, scheduler linear, AdamW dan loss tetap mengikuti konteks pair. S1/S2/S3 dilatih baru; tidak memakai ulang kontrol lama. Total15pelatihan fold +3refit. Setelah E8 selesai, hasil cocok akan dipakai ulang ketika notebook yang sama dijalankan lagi.

Folder baru: `MyDrive/AES_300_E8_ContextPair_LR5e6_GPU_Privat`. Checkpoint otomatis memakai dua nama tetap A/B per fold/refit, dengan manifest SHA dan fallback slot sebelumnya. Tidak ada Drive API atau auto-delete. Slot seluruh fold, model final, versi file dan Sampah Drive tetap dapat menggunakan ruang; notebook ini tidak membersihkan eksperimen lama. Untuk resume, jalankan dari atas dengan sumber/data/konfigurasi/perangkat yang cocok.

Cell terakhir mengunduh `Hasil_AES_E8_ContextPair_LR5e6_GPU.zip`: lima Excel data/tokenisasi (termasuk segmen konteks/jawaban), vektor jawaban, vektor kata/kalimat, prediksi/evaluasi, dan riwayat training. Nilai guru, prediksi mentah dan pembulatan tetap dibedakan. Metrik RMSE/Pearson memakai prediksi mentah; QWK/GwetAC2 memakai kategori salinan. Jawaban baru memerlukan soal dan referensi yang tersedia ketika diprediksi.

Source tokenisasi pasangan, model, split, scoring dan ekspor fitur sama dengan konteks pair sebelumnya; perubahan metode terfokus pada LR S3. Installer teks dan retensi checkpoint diperbaiki secara terpisah. Tidak menjamin RMSE turun. Untuk mengisolasi LR, bandingkan E4/E8 dengan lingkungan/perangkat/seed sama; E6 CPU vs E8 GPU mempunyai perbedaan presisi. Pemilihan kandidat melalui CV; data uji yang sudah dilihat selama pengembangan tetap eksploratif.

## Verifikasi teknis

```bash
AES_CONTEXT_NOTEBOOK=AES_300_E8_ContextPair_LR5e6_GPU.ipynb outputs/dataset300_env_py313/bin/python -m pytest tests/test_context_pair_notebook.py tests/test_e8_context_lr_notebook.py -q --disable-warnings
```

Pengujian memakai 24baris sintetis, tokenizer teknis dan Transformer kecil dengan bobot acak, bukan hasil AES pada siswa. Pemeriksaan mencakup cakupan jawaban panjang/pasangan, independensi konteks dari skor guru, bentuk fitur/gradien, LR tiap skenario, resume identik setelah interupsi, fallback checkpoint, seluruh cell notebook hingga lima Excel, dan runtime baru tanpa melatih ulang hasil selesai. GPU T4 dan training pretrained dengan dataset300 pengguna belum dijalankan di host ini.

Hasil pemeriksaan: **19 passed,0 failed,0 skipped,0 xfailed**, satu warning. Seluruh30cell notebook dieksekusi dengan fixture teknis sampai lima Excel, lalu dijalankan dari runtime baru untuk memeriksa reuse hasil lengkap. Pemeriksaan tambahan dibatasi pada fungsi konteks pair lama, tanpa membangun ulang notebook lama atau mengulang eksekusi lamanya. Bukti command/versi/hash dan log tersimpan privat pada `outputs/e8_context_lr_*`.
