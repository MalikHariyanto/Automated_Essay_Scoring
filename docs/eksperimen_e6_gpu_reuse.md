# Eksperimen 6 · S3 LR 5e-6 · GPU · memakai ulang S1/S2

[Buka di Colab](https://colab.research.google.com/github/MalikHariyanto/Automated_Essay_Scoring/blob/codex/aes-learning/notebooks/experiments/AES_300_E6_S3_LR5e6_GPU_Reuse.ipynb) · [Notebook](../notebooks/experiments/AES_300_E6_S3_LR5e6_GPU_Reuse.ipynb)

**Tidak ada file lama yang perlu dihapus untuk memulai E6.** Notebook ini membuat folder baru, memverifikasi hasil selesai S1/S2 dari MLP acuan atau LoRA, lalu hanya melatih S3 kandidat B di GPU. S3 lama tetap tersimpan sebagai pembanding. Versi E6 CPU, MLP acuan dan LoRA tidak diubah.

| Bagian | Pelaksanaan |
| --- | --- |
| S1 | Lima hasil CV, model final dan hasil test dipakai ulang; tidak dilatih/diuji ulang |
| S2 | Lima hasil CV, model final dan hasil test dipakai ulang; tidak dilatih/diuji ulang |
| S3 | Mulai baru dari pretrained; RoBERTa fine-tuned LR **5e-6**, BGE frozen, proyeksi/head MLP **1e-4** |

S3 baru mengikuti arsitektur MLP acuan, tanpa LoRA, konteks pair, atau fitur linguistik tambahan. Lima fold dalam development 80%, test 20%, maksimum 25 epoch, patience 5, microbatch 1/batch efektif 8, dropout, warmup dan scheduler tetap. S3 baru tidak melanjutkan optimizer atau bobot S3 eksperimen lama yang metodenya berbeda.

## Menjalankan

1. Pilih **Runtime → Change runtime type → T4 GPU**, lalu jalankan notebook dari atas. GPU NVIDIA lain diterima. `device='cuda'` dan guard GPU mencegah CPU dipakai tanpa sengaja.
2. Hubungkan **Drive akun yang menyimpan hasil S1/S2 lama**, lalu pilih dataset Excel yang sama persis jika diminta.
3. Biarkan `REUSE_RUN=''` untuk pencarian otomatis pada run lengkap di `MyDrive/AES_300_MLP_Privat/outputs/dataset300_inline/`, lalu `MyDrive/AES_300_S3_LoRA_MLP_Privat/outputs/dataset300_inline/`. Di masing-masing folder, run terbaru dicoba dahulu. Sumber dipilih berdasarkan urutan folder/tanggal, tidak berdasarkan nilai metrik.
4. Jika hasil berada di folder lain, isi `REUSE_RUN` dengan **folder hash run lama**, misalnya `/content/drive/MyDrive/AES_300_S3_LoRA_MLP_Privat/outputs/dataset300_inline/<hash>`. Folder tersebut harus langsung berisi `effective_config.json`, `final_report.json`, `S1/`, dan `S2/`. Jangan isi path ke `S1` atau file `.pt` saja.
5. Saat cell training dimulai, notebook menampilkan sumber S1/S2. Jika sumber tidak cocok, proses berhenti sebelum training dan menjelaskan alasannya; S1/S2 tidak dilatih ulang diam-diam.
6. Jalankan hingga cell unduhan akhir untuk lima Excel. Tabel **Asal Model** di layar/laporan pelatihan membedakan S1/S2 dari run lama dan S3 yang dilatih baru.

Folder tujuan baru: **MyDrive/AES_300_E6_S3_LR5e6_GPU_Reuse_Privat**. Tidak perlu memindahkan, menghapus, atau mengubah folder S3 lama. Menghapus S3 di run lama saja tidak cukup untuk membuat E6 memakai ulang kontrol: laporan/penguncian lama masih merujuk ketiga model, sedangkan kode/config baru memiliki identitas run berbeda.

## Yang diperiksa dan disalin

Reuse memerlukan sumber lengkap dengan identitas MLP acuan/LoRA yang dikenal. Dataset hash, split hash/indeks validasi, konfigurasi yang memengaruhi S1/S2, versi Transformers/NumPy, kompatibilitas PyTorch, pilihan epoch dari lima fold, LR aktual kontrol, checksum model/prediksi test, ID jawaban dan nilai guru harus cocok. Model final dimuat di CPU dengan weights_only dan layer strict sebelum sumber diterima. PyTorch stabil 2.x mulai 2.6 boleh berbeda build/patch/minor untuk reuse final state_dict, bukan resume optimizer. Versi sumber dan aktif dicatat dalam rencana reuse, laporan akhir dan sheet Asal Model; metrik S1/S2 tetap hasil runtime sumber. Inferensi baru tidak dijamin identik bit demi bit. Konfigurasi khusus S3/LoRA boleh berbeda karena S3 sumber tidak dipakai.

Hanya `S1/S2/cv_fold_N.json`, `S1/S2/final.pt`, `S1/S2/test_result.json` dan `S1/S2/test_predictions.csv` yang disalin. Tidak menyalin checkpoint epoch/fold lama, bobot S3 lama, root `refit_locked.json`, atau root `final_report.json`. Model final yang disalin memakai ruang Drive tambahan; seluruh sumber tetap utuh. Penyalinan menggunakan file sementara dan checksum. Rencana sumber/checksum dikunci sebelum copy supaya penyalinan yang terputus dapat dilanjutkan tanpa berganti sumber. File tujuan yang tidak cocok ditolak, bukan ditimpa.

Setelah pilihan CV S3 terkunci, metadata refit S1/S2 pada tujuan mencatat bahwa model berasal dari run lama. Evaluasi test S1/S2 tidak dihitung lagi; report menyatakan `test_evaluations_this_run = {S1: 0, S2: 0, S3: 1}`. Inferensi untuk inspeksi vektor dan contoh prediksi tetap dapat membaca ketiga model final. Setelah runtime terputus, jalankan notebook yang sama dari atas untuk memakai kontrol dan progres S3 yang cocok. GPU/versi numerik harus cocok dengan kontrak checkpoint epoch aktif.

## File yang dipertahankan

- Pertahankan semua hasil lama untuk perbandingan. Tidak perlu menghapus S1, S2 maupun S3 sumber.
- Pertahankan `final.pt`, JSON hasil, Excel dan ZIP. JSON bukan pengganti bobot.
- Untuk training S3 E6 yang masih aktif, pertahankan JSON progres dan dua snapshot yang dirujuknya.
- Tidak ada auto-delete, Drive API cleanup atau pengosongan Sampah. Pembersihan checkpoint lama yang sudah selesai tetap manual dan terpisah dari reuse ini.

Hasil S1/S2 adalah hasil yang dipakai ulang, sehingga jangan dilaporkan sebagai dua run training baru. Test lama sudah berulang kali dilihat selama pengembangan; pilih LR melalui CV dan gunakan data baru yang belum disentuh untuk konfirmasi independen bila tersedia. Belum ada skor penelitian S3 E6 baru sebelum pengguna menjalankannya pada model pretrained asli di GPU.

[Bukti pengujian teknis](verification_e6_gpu_reuse.md).
