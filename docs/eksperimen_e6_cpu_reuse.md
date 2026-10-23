# Eksperimen 6 · CPU · memakai ulang S1/S2

[Buka di Colab](https://colab.research.google.com/github/MalikHariyanto/Automated_Essay_Scoring/blob/codex/aes-learning/notebooks/experiments/AES_300_E6_S3_LR5e6_CPU_Reuse.ipynb) · [Notebook](../notebooks/experiments/AES_300_E6_S3_LR5e6_CPU_Reuse.ipynb)

Versi CPU dari E6 reuse. Model final S1/S2 yang sudah selesai dapat dibaca pada CPU meskipun sumbernya dilatih di GPU. Tidak perlu menghapus file S3 atau mengubah hasil lama. Hanya S3 E6 dilatih baru; S1/S2 memakai hasil CV, model final dan prediksi test dari sumber yang lolos verifikasi.

| Bagian | Pengaturan |
| --- | --- |
| Perangkat baru | CPU eksplisit, float32 tanpa AMP, tidak wajib GPU |
| S1/S2 | Hasil selesai dipakai ulang dari MLP acuan atau LoRA; tidak dilatih/diuji ulang |
| S3 | RoBERTa fine-tuned LR 5e-6 + BGE frozen; proyeksi/MLP LR 1e-4 |
| Protokol | Development/test 80/20; 5-fold CV development; maksimal 25 epoch, patience 5 |
| Arsitektur | MLP acuan, tanpa LoRA maupun konteks pair |

## Menjalankan

1. Pilih **Runtime → Change runtime type → Hardware accelerator: None**.
2. Jalankan notebook dari atas dan hubungkan Drive yang menyimpan S1/S2 lama. Pilih dataset Excel yang sama persis jika diminta.
3. Biarkan `REUSE_RUN=''` untuk pencarian otomatis pada folder MLP acuan, lalu LoRA. Sumber dipilih berdasarkan urutan folder/tanggal, bukan skor.
4. Jika perlu, isi `REUSE_RUN` dengan folder hash run lama yang langsung berisi `effective_config.json`, `final_report.json`, `S1/` dan `S2/`; bukan path file `.pt` saja.
5. Cell training memverifikasi sumber sebelum memakai S1/S2. Jika sumber tidak cocok, notebook berhenti dengan alasan; tidak melatih kontrol ulang secara diam-diam.
6. Jalankan hingga cell akhir untuk lima Excel dan ZIP. Sheet **Asal Model** menandai S1/S2 dari run lama dan S3 yang dilatih baru.

Hasil baru berada di **MyDrive/AES_300_E6_S3_LR5e6_CPU_Reuse_Privat**. Autosave Drive setiap epoch tetap aktif; tidak ada auto-delete atau pembersihan Sampah. CPU dapat jauh lebih lambat daripada T4.

## GPU ke CPU dan pemulihan

Sumber kontrol dimuat dengan `map_location='cpu'`. Dataset, split, konfigurasi S1/S2, versi numerik Torch/Transformers/NumPy, pilihan epoch, checksum dan arsitektur tetap diperiksa. Perangkat sumber boleh berbeda, tetapi ini tidak mengizinkan versi numerik yang berbeda. Jika versi pustaka tidak cocok, sumber ditolak sebelum training.

Ini memakai **model final yang selesai**, bukan memaksa progres optimizer GPU berlanjut pada CPU. S3 GPU yang masih berjalan tetap di folder GPU; S3 CPU dimulai baru. Setelah versi CPU terputus, jalankan notebook CPU yang sama dari atas untuk melanjutkan progres CPU yang cocok. Pertahankan JSON progres serta snapshot yang dirujuknya dan model final; JSON bukan pengganti bobot.

Metode training, reuse, evaluasi dan ekspor identik versi GPU; perangkat/precision dan folder run berbeda. Catat CPU float32 versus GPU AMP saat membandingkan LR. Hasil S1/S2 adalah reuse, bukan dua training baru. Pilih LR melalui CV; data uji lama yang sudah berulang kali dilihat tidak menjadi konfirmasi independen.

[Rincian verifikasi dan penyalinan sumber](eksperimen_e6_gpu_reuse.md#yang-diperiksa-dan-disalin) · [Bukti pengujian CPU](verification_e6_cpu_reuse.md).
