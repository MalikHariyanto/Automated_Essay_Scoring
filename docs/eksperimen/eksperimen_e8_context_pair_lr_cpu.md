# E8 CPU · konteks pair + MLP + LR S3 5e-6

[Buka Colab](https://colab.research.google.com/github/MalikHariyanto/Automated_Essay_Scoring/blob/codex/aes-learning/notebooks/experiments/AES_300_E8_ContextPair_LR5e6_CPU.ipynb) · [Unduh notebook](../../notebooks/experiments/AES_300_E8_ContextPair_LR5e6_CPU.ipynb)

Pilih Runtime → Change runtime type → CPU, lalu jalankan dari atas. Dataset `fix_dataset_300.xlsx` diunggah jika diminta; source proyek dan data siswa tidak diunduh dari GitHub. Seluruh model dipaksa memakai CPU float32 tanpa AMP, dua thread. Tidak memerlukan kuota GPU. CPU jauh lebih lama daripada T4; anggaran epoch dan metode tetap sama.

Metode mengikuti [E8 GPU](eksperimen_e8_context_pair_lr_gpu.md). S3 memakai RoBERTa full fine-tuning dengan pasangan soal/referensi–jawaban, BGE frozen membaca jawaban saja, LayerNorm/proyeksi, concat, dan MLP. LR encoder S3 5e-6, S1 1e-5, proyeksi/head 1e-4. Batas konteks192, soal64, window512, overlap128, seed42, split_seed2026, lima fold, maksimal25epoch/fold, patience5, microbatch1/batch efektif8 tetap. S1–S3 dilatih baru, satu seed; E7 tiga seed tetap terpisah.

Folder baru `MyDrive/AES_300_E8_ContextPair_LR5e6_CPU_Privat`. Versi CPU tidak mengambil progres GPU atau eksperimen lain. Tidak perlu menghapus hasil lama. Resume berlaku untuk run CPU dengan kode/data/konfigurasi/perangkat yang cocok. Dua slot A/B per fold/refit dipakai untuk autosave/fallback; tanpa Drive API dan tanpa auto-delete. Model final, JSON, slot fold selesai dan versi file Drive tetap membutuhkan ruang.

Cell terakhir mengunduh `Hasil_AES_E8_ContextPair_LR5e6_CPU.zip` berisi lima Excel: data/tokenisasi, vektor jawaban, vektor kata/kalimat, nilai guru/prediksi/pembulatan serta empat metrik, dan riwayat training. Tidak menjamin RMSE lebih kecil. Perbandingan CPU/GPU tidak mengisolasi efek LR karena presisi/perangkat berbeda; pemilihan metode melalui CV, dengan test pengembangan yang sudah dilihat tetap eksploratif.

Pemeriksaan teknis:

```bash
AES_CONTEXT_NOTEBOOK=AES_300_E8_ContextPair_LR5e6_CPU.ipynb outputs/dataset300_env_py313/bin/python -m pytest tests/test_context_pair_notebook.py tests/test_e8_context_lr_notebook.py -q --disable-warnings
```

Fixture 24baris sintetis dengan tokenizer teknis dan Transformer kecil berbobot acak memeriksa seluruh cell notebook hingga prediksi/lima Excel dan resume pada runtime baru. Pengujian ini bukan hasil penelitian dengan pretrained/dataset300 siswa; training penuh Colab belum dijalankan di sini. Notebook GPU dan sumber eksperimen lama dipertahankan.

Hasil: **20 passed,0 failed,0 skipped,0 xfailed**, satu warning; seluruh30cell dijalankan dengan fixture teknis lalu diulang dalam runtime baru untuk resume. Tiga pemeriksaan tambahan konfigurasi/builder versi GPU lulus (17kasus lain tidak dipilih pada pemeriksaan tambahan GPU). Log/command/versi/hash disimpan privat pada `outputs/e8_context_lr_cpu_*`.
