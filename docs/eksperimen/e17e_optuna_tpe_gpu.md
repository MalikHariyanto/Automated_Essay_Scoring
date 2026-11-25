# E17E — Optuna TPE untuk S3, GPU Colab + Drive

Notebook: `eksperimen/03_tiga_input_multi_trait/notebooks/AES_Reset_E17E_S3_Optuna_TPE_Native_Learnable_GPU.ipynb`.

## Menjalankan

1. Buka notebook di Colab; pilih Runtime → Change runtime type → T4 GPU.
2. Jalankan Run all dalam mode bawaan `STAGE='search'`. Hubungkan Drive. Dataset tetap `MyDrive/AES/datasets/semi_final/data_cleaned_3sekolah.xlsx` dengan checksum semi-final yang dikunci. Upload pertama ke Drive tersedia seperti E17C.
3. Setelah 60 trial COMPLETE/PRUNED selesai, ubah STAGE menjadi `confirm`, lalu Run all. Lima konfigurasi COMPLETE terbaik yang unik, di luar baseline, dan baseline diuji pada lima fold yang sama. Kandidat dengan mean CV RMSE terendah pada seed42 dipilih; baseline tetap eligible. Kemudian pemenang dan baseline diperiksa dengan seed42/43/44. Seed42 memakai hasil CV yang sudah selesai; stabilitas tidak memilih parameter dari test.
4. Setelah `selection_locked.json` tersimpan, ubah STAGE menjadi `final`, lalu Run all. Refit memakai ceil median best epoch CV seed42; horizon scheduler mengikuti durasi refit seperti E17C. Evaluasi test hanya melalui mode final. Prediksi test yang selesai dipakai ulang bila checksum checkpoint cocok.

Hasil privat: `MyDrive/AES/outputs/aes_reset_e17e_tpe_gpu/studies/<study_id>/`. Semua fungsi terlihat langsung di sel notebook; tidak ada hidden worker atau magic training. Hanya kode kosong output yang dipublikasikan.

## Desain pencarian

Arsitektur RoBERTa full fine-tuning + BGE frozen/projection256/concat/shared MLP256/empat output tetap E17C. Tidak ada cosine similarity. AdamW tetap memperbarui bobot, Optuna4.6.0 memilih pengaturan. Loss MSE pada label ternormalisasi rubrik0–1. Objective seleksi: pooled validation RMSE pada satuan asli; Isi memiliki rentang1–4 dan trait lain1–2. Objective ini sengaja dipertahankan untuk keterbandingan; bukan macro normalized RMSE. QWK/AC2/Pearson dan out-of-range tetap dilaporkan.

| Parameter | Ruang pencarian |
|---|---|
| LR RoBERTa | 1e-6 sampai2e-5, log |
| LR head/projection/LayerNorm | 3e-5 sampai1e-3, log; diikat bersama |
| Dropout MLP | 0,05 sampai0,40, langkah0,05 |
| Weight decay | 0;0,001;0,005;0,01;0,03;0,05 |
| Warmup ratio | 0,03;0,05;0,10;0,15 |

Total awal60 trial mencakup16 kombinasi awal yang ditentukan sebelum hasil dan tidak dipruning. Sisanya disarankan TPE dari riwayat study. Setiap attempt memakai seed sampler `2026+nomor_trial` agar pemulihan tidak memerlukan pickle state RNG. Fit setiap trial memakai seed42 yang sama. Pemilihan split pencarian menggunakan GroupShuffleSplit pada development saja, seed2029; tanpa label atau test. Seluruh trial memakai split yang sama. Tidak ada perubahan batch efektif8/micro1, maksimum25epoch/patience5, clip1, atau arsitektur.

MedianPruner:16 trial selesai sebelum pruning aktif, paling awal epoch5 (setelah warmup maksimum3,75epoch), minimal5 pembanding. Ini rancangan awal, bukan pruner yang terbukti optimal untuk dataset ini. Epoch terpilih dan metrik dihitung ulang dari model dengan bobot best epoch dipulihkan.

## Anggaran dan pemulihan

Maksimal sekitar111 fit:60 search,30 CV (top5+baseline),20 tambahan stabilitas (dua kandidat×dua seed tambahan×lima fold),1refit. Lebih sedikit jika baseline menang atau pruning/early stopping menghemat epoch. FAIL tidak dihitung ke anggaran60; exception menghentikan sesi agar penyebab diperiksa. Jumlah jam belum diukur; tidak ada janji selesai dalam satu sesi T4.

SQLite bekerja di disk lokal, bukan transaksi langsung di Drive. Sesudah setiap epoch/trial, SQLite backup API membuat snapshot lengkap, memeriksa integrity, menghitung SHA-256, lalu mengganti pointer Drive secara atomik. Snapshot lama dihapus sesudah pointer baru tersimpan. Runtime baru memverifikasi snapshot dan memulihkannya. Trial RUNNING/FAIL yang belum dijadwalkan ulang diberi retry parameter lengkap. Fit terputus diulang dari awal; tidak ada resume bobot/optimizer per epoch. Trial selesai dan fold selesai tidak diulang. Cache BGE dipakai bersama seluruh trial selama kontrak preprocessing cocok.

Jalankan satu runtime saja untuk folder study ini. Tidak ada distributed lock atau dukungan dua runtime menulis snapshot bersama. Jika koneksi Drive putus di tengah penulisan, snapshot terakhir yang berhasil adalah titik pemulihan; pekerjaan setelahnya dapat diulang. Perubahan versi runtime/model/config/ruang pencarian/kode membuat study_id baru. Saat mengubah kode training manual, regenerasi hash kode/ubah code ID; jangan mengklaim source yang diedit cocok hash yang tertanam. Menaikkan HPO_TRIALS melanjutkan study yang sama, tetapi seleksi perlu diulang sebelum final.

## File hasil

- `06_Pencarian_Hyperparameter.xlsx`: seluruh trial COMPLETE/PRUNED/FAIL/WAITING, parameter, waktu, best epoch, dan riwayat training.
- `07_Konfirmasi_CV.xlsx`: perbandingan kandidat dan fold pada seed42.
- `08_Stabilitas_Seed.xlsx`: hasil tiap seed pemenang dan baseline.
- `Hasil_AES_Reset_E17E_Ringkas_04_05.zip`: format evaluasi/OOF/test serta riwayat/config E17C, ditambah tabel HPO. Export ringkas ini tidak mengekstrak workbook token/vektor01–03 untuk setiap trial.
- Kontrak, snapshot DB, trial history, hasil fold, selection seal, checkpoint final dan prediksi test berhash disimpan di Drive.

Seleksi dikunci terhadap fingerprint study dan hash file fold. Study/CV berubah atau konfirmasi belum lengkap memblokir mode final. Gate ZIP memeriksa hash workbook, ID/urutan/jumlah prediksi test, nilai prediksi, RMSE per trait, dan keberadaan tabel HPO.

## Dasar dan batas penelitian

Kontrol kode: E17C di branch `codex/e17c-concat-dropout01`, code ID `E17C_CONCAT_SHARED256_DROPOUT01_FULL_FINETUNING_V1`. Hasil E17D tidak mendukung keunggulan horizonrefit25; E17E mempertahankan kebijakan E17C sebagai kontrol tanpa menyimpulkan kausalitas dari satu run.

Rentang pencarian adalah adaptasi proyek di sekitar baseline; bukan angka optimum yang disalin dari paper. Workbook `docs/skripsi/Brainstorming_Referensi_AES_MultiTrait.xlsx` membedakan angka paper, konfigurasi kode penulis, serta adaptasi lokal; tidak menyediakan bukti bahwa seluruh batas HPO di atas optimal atau berasal dari jurnal berkuartil. Jangan menulis seolah seluruh parameter bersumber langsung dari paper indeks.

Sumber metode/perangkat:

- Akiba dkk.2019, *Optuna: A Next-generation Hyperparameter Optimization Framework*: https://arxiv.org/abs/1907.10902
- Bergstra dkk.2011, *Algorithms for Hyper-Parameter Optimization*: https://papers.nips.cc/paper/2011/hash/86e8f7ab32cfd12577bc2619bc635690-Abstract.html
- API TPE: https://optuna.readthedocs.io/en/v4.6.0/reference/samplers/generated/optuna.samplers.TPESampler.html
- MedianPruner: https://optuna.readthedocs.io/en/v4.6.0/reference/generated/optuna.pruners.MedianPruner.html

Tuning/shortlist/CV memakai development yang saling tumpang tindih; skor CV pemenang bersifat post-selection dan optimistis, bukan estimasi nested-CV independen. Stabilitas seed adalah diagnostik. Test penelitian telah beberapa kali dilihat dalam eksperimen terdahulu; bukan holdout konfirmatori baru. Tidak menjamin konfigurasi optimal global atau improvement terhadap E17C.

## Verifikasi kode sebelum publikasi

Seluruh49 cell Python diparse/compile; notebook103cell tanpa output atau data siswa. Blok dataset, split, tokenisasi, encoder/BGE, arsitektur, optimizer groups dan metrik dipertahankan byte-identik dari E17C. Fit hanya ditambah callback pruning dan log scheduler, tanpa mengubah loss/arsitektur.

`python tests/test_reset_e17e_hpo.py` menggunakan Optuna4.6.0/SQLite sungguhan dan model fixture: pencarian18trial, pemulihan snapshot, retry orphan, tidak mengulang trial selesai, isolasi kelompok/test, CV reuse, selection lock, deteksi korupsi, cache refit/test, gate ZIP/XLSX dan masa tunggu pruning. Tes ini bukan training pretrained. Forward/backward RoBERTa/BGE, konsumsi memori T4 dan training60trial sebenarnya belum dijalankan di sesi penyusunan. Belum ada hasil penelitian E17E.
