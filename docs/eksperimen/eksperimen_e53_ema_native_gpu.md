# E53 · E43 + Exponential Moving Average (EMA) · S3 Native Learnable

[Buka E53 di Colab](https://colab.research.google.com/github/MalikHariyanto/Automated_Essay_Scoring/blob/codex/aes-learning/notebooks/experiments/AES_300_E53_E43_S3_EMA099_Native_Learnable_GPU.ipynb) · [Source notebook](../../notebooks/experiments/AES_300_E53_E43_S3_EMA099_Native_Learnable_GPU.ipynb).

**GPU T4 → Run all → upload `fix_dataset_300.xlsx`.** Cukup dataset Excel, tanpa ZIP hasil lama. Kode Python biasa terlihat per section: input, model, hyperparameter, EMA, training, evaluasi dan unduhan. Tanpa magic, worker tersembunyi, clone repo atau Drive.

**E53 kembali ke E43 utuh**, dengan satu perubahan metode: salinan bobot EMA digunakan untuk validasi, pemilihan epoch dan prediksi akhir. RoBERTa dilatih penuh lagi dengan LR `5e-6`; tidak memakai DoRA E52. BGE encoder tetap frozen. ScalarMix empat layer + gamma, BiGRU64 per arah, co-attention satu head, masked mean, concat dan MLP tetap E43.

## Input dan alur model

`soal`, `jawaban_referensi`, `response_text` adalah tiga kolom X yang saling terhubung melalui konteks encoder dan pencocokan respons–referensi. `rater` adalah y, skor guru 0–100, hanya untuk loss/seleksi/evaluasi. Label tidak masuk encoder atau rumus EMA.

Tiga teks → RoBERTa full fine-tuning → ScalarMix4 + gamma → BiGRU64/arah → co-attention dua arah + roundtrip → matching → masked mean → fitur768; BGE-M3 frozen1024 → proyeksi256; concat1024 → MLP256/GELU/dropout0,1 → satu skor. Nama checkpoint dan revision sama E43. Tidak ada multihead, fusion gate, atau encoder tambahan.

## Apa yang berubah saat training?

Bobot **online** adalah bobot model yang menerima gradien MSE dan diperbarui AdamW. Bobot **EMA** adalah salinan tanpa gradien yang mengikuti rata-rata bergerak:

```text
t = jumlah update optimizer yang berhasil dan LR-nya aktif
update pertama: bobot_EMA = bobot_online
update berikutnya:
    decay_t = min(0,99, (1+t)/(10+t))
    bobot_EMA = decay_t × bobot_EMA_lama + (1−decay_t) × bobot_online
```

**0,99 adalah batas maksimum, bukan decay tetap sejak awal.** Warmup offset 10 memperkecil pengaruh bobot MLP acak di awal. Pada run pendek, decay aktual bisa belum mencapai 0,99; semua nilai aktual dicatat. Batas/rumus warmup merupakan resep awal proyek, bukan angka optimal AES yang dibuktikan paper.

Salinan EMA meliputi seluruh parameter model S3: RoBERTa, BiGRU/matching, normalisasi, proyeksi BGE, ScalarMix/gamma dan MLP. Encoder BGE tidak dirata-ratakan karena frozen dan fiturnya dihitung sekali. Buffer posisi/token-type disalin langsung, tidak dirata-ratakan sebagai integer. EMA tidak menambah parameter yang dioptimasi. Tambahan memori diperlukan untuk satu salinan model; evaluasi online+EMA menambah forward validasi.

| Pengaturan | E53 |
|---|---|
| LR RoBERTa / komponen lain | `5e-6` / `1e-4`, kembali E43 |
| Loss | MSE biasa, target dinormalisasi0–1; loss training berasal dari model online |
| EMA | Batas decay 0,99, warmup offset 10; update per langkah optimizer berhasil dengan LR aktif |
| Seleksi epoch | **RMSE validasi EMA**, bukan hasil online atau test |
| Protokol | S3 saja, satu 5-fold CV baru, maksimal 25 epoch, patience 5 |
| Refit / test | Ceil median best epoch EMA lima fold; satu refit development dan satu evaluasi test memakai EMA akhir refit |
| Lainnya | AdamW/decay 0,01, warmup LR 10%, linear scheduler sesuai durasi fit E43; microbatch 1/batch efektif 8, clip 1 |

Satu seed dasar 42, fold mengikuti acuan; tanpa loop banyak seed atau kontrol ulang E43. Tidak memakai LDS, ranking, cosine, S1/S2, scheduler refit E46/E47 atau Switch EMA. Bobot online tidak diganti EMA selama pembelajaran; model yang dikembalikan untuk prediksi memuat EMA terpilih.

## Output yang perlu dibaca

- **Excel 05 → Perbandingan EMA:** RMSE validasi online dan EMA pada epoch yang sama, selisih EMA−online, jumlah update, decay aktual dan penanda checkpoint terpilih. Ini diagnostik dari satu lintasan training; bukan run kontrol E43 independen, sebab early stopping dipilih memakai EMA.
- **Excel 05 → Konfigurasi EMA:** enam checkpoint fold/refit, jenis bobot prediksi, resep, epoch terpilih/dijalankan, update dan decay pada checkpoint terpilih serta pada akhir training. Keduanya dibedakan karena training bisa berlanjut melewati epoch terbaik.
- **Excel 05 → Update EMA per Epoch:** langkah optimizer, LR aktif, AMP yang dilewati, update EMA dan decay. History utama melaporkan MSE online serta metrik validasi EMA; Bobot ScalarMix/gamma diambil setelah EMA terpilih dipulihkan.
- **Excel 04:**60 test, 240 OOF terpisah, skor mentah/normalisasi/pembulatan, selisih tepat 0/1/2/3/4/5/10 dan lainnya, toleransi ≤n, MSE/MAE/error maksimum. Seluruh prediksi S3 tersebut menggunakan bobot EMA.
- **Excel 01/02/03:**input/tokenisasi, fitur model EMA final, BGE, kata/kalimat tetap. Inspeksi model refit terhadap 300 jawaban bukan evaluasi 300 data.

Checkpoint progres menyimpan bobot online, EMA, counter, best EMA, optimizer/scheduler/scaler, loader dan RNG. Resume tidak menghitung ulang EMA dari nol. Checkpoint final diberi jenis bobot `ema`; loader menolak jenis/resep yang berbeda. Report ulang membaca hasil tersimpan tanpa mengevaluasi test lagi. Penyimpanan progres lebih besar karena memuat dua set bobot; ZIP unduhan tetap hanya lima Excel.

Pemeriksaan sebelum ZIP membaca ulang metrik test/OOF, mean/SD CV, rekap selisih, dua RMSE per epoch, minimum RMSE EMA yang dipilih, counter/update aktif dan rumus decay. Tabel wajib kosong atau metadata tidak cocok menghentikan pembuatan ZIP baru. Source/hasil/HTML lama tetap tersedia.

## Referensi dan batas bukti

Morales-Brotons, Vogels & Hendrikx, **[Exponential Moving Average of Weights in Deep Learning: Dynamics and Benefits, TMLR 2024](https://arxiv.org/abs/2411.18704)**. [PyTorch AveragedModel/EMA](https://docs.pytorch.org/docs/stable/generated/torch.optim.swa_utils.AveragedModel.html) menyediakan rujukan implementasi recurrence EMA. E53 menampilkan rumus/update/resume sebagai kode Python langsung, bukan menyalin seluruh rancangan paper. Warmup dan batas decay adalah adaptasi proyek. Ini EMA biasa; tidak menyalin bobot EMA ke model online pada setiap epoch seperti Switch EMA.

Acuan E43: meanRMSE CV 5,8876, MAE OOF 4,6122, selisih mentah ≤5 sebanyak 151/240; RMSE test historis 5,1265. **Belum ada hasil penelitian E53.** Pilih melalui validasi; test historis bersifat eksploratif. Paper bukan bukti langsung bahwa EMA memperbaiki AES Indonesia atau menjamin 60/60 dalam ±3/±5.

## Sumber dan reproduksi

`scripts/build_e53_notebook.py` membaca arsitektur E43 pada commit `7241a21dfec37c942948b754da3960fba4c7731c`, lalu memakai pelaporan terverifikasi dari E50 pada commit `c36003670ad9476951b8b607a6018e5a24ceb7e2` tanpa mengambil model multihead. Helper `scripts/ema_native.py` tampil langsung di notebook, tanpa impor source proyek pada Colab. Notebook utama dan mirror identik; data, bobot dan hasil tidak ditanam dalam source publik.

[Bukti pengujian teknis](../verifikasi/verification_e53_ema_native.md). Pengujian tidak membuktikan kenaikan akurasi atau kompatibilitas seluruh runtime GPU.
