# E51 · E50 + gated concatenation

[Buka E51 di Colab](https://colab.research.google.com/github/MalikHariyanto/Automated_Essay_Scoring/blob/codex/aes-learning/notebooks/experiments/AES_300_E51_E50_S3_GatedConcat_Native_Learnable_GPU.ipynb) · [Source notebook](../../notebooks/experiments/AES_300_E51_E50_S3_GatedConcat_Native_Learnable_GPU.ipynb).

**GPU T4 → Run all → upload `fix_dataset_300.xlsx`.** Cukup dataset Excel, tanpa ZIP hasil lama. Kode Python biasa disusun per section, tanpa magic atau worker tersembunyi.

E51 menguji **satu perubahan dari E50: gate per jawaban sebelum concatenation**. Gate membaca fitur RoBERTa 768 dimensi dan proyeksi BGE 256 dimensi. `Linear(1024, 2)` diikuti softmax menghasilkan `alpha_R` dan `alpha_B`, lalu:

```text
fusion = concat(2 × alpha_R × roberta, 2 × alpha_B × bge)
```

Fusion tetap 1.024 dimensi dan masuk ke MLP yang sama. Tambahan **2.050 parameter** dilatih bersama head dengan LR `1e-4`. Bobot dan bias gate dimulai dari nol: alpha awal `0,5 / 0,5`, sehingga pengali fitur `1 / 1`. Dengan demikian fitur dan prediksi awal identik dengan E50. Inisialisasi gate menjaga urutan pengacakan dan bobot komponen bersama tetap sama.

**Alur model:** tiga teks → RoBERTa fine-tuned → ScalarMix 4 layer + gamma → BiGRU 64 per arah → multihead co-attention 4 × 32 dengan dua arah dan roundtrip → matching + masked mean → proyeksi 768; BGE frozen → proyeksi 256; kedua cabang → gate → concat 1.024 → MLP 256/GELU/dropout 0,1 → satu skor. Encoder BGE tetap frozen, sedangkan proyeksinya ikut belajar.

| Pengaturan | Nilai |
|---|---|
| Input X | `soal`, `jawaban_referensi`, `response_text` |
| Target y | `rater`, skor guru 0–100; target loss dinormalisasi ke 0–1 |
| Gate | `FUSION_GATE_MODE='softmax_2x_identity'`; alpha 0–1, jumlah 1, pengali fitur 2 × alpha |
| Learning rate | Encoder RoBERTa `5e-6`; komponen lain dan gate `1e-4` |
| Loss / pemilihan epoch | MSE biasa / RMSE validasi tanpa bobot |
| Optimizer / scheduler | AdamW, decay 0,01; warmup 10%, linear decay sesuai durasi fit seperti E50 |
| Training | S3 saja; satu 5-fold CV baru; maksimal 25 epoch, patience 5; satu refit dan satu evaluasi test |
| Pengacakan | Seed dasar 42; seed fold mengikuti acuan; tanpa loop banyak seed atau menjalankan kontrol ulang |

Tiga teks saling terhubung melalui konteks encoder dan co-attention. **Label tidak masuk ke encoder atau gate**; label digunakan untuk loss, pemilihan epoch, dan evaluasi. Tidak ada LDS, ranking, cosine, S1/S2, atau perubahan jadwal refit seperti E46/E47. Gate tidak mengubah dimensi fusion, MLP, dropout, atau attention E50. Softmax gate dihitung dalam FP32 saat AMP; pengali mengikuti dtype fitur.

## Membaca hasil gate

- **Excel 04 → Gate Validasi OOF:** 240 jawaban development, masing-masing dari checkpoint fold terpilih yang tidak melatih jawaban tersebut. **Gate Test:** 60 jawaban test dari checkpoint refit. Keduanya menyimpan ID, nilai guru, prediksi, alpha, dan pengali fitur.
- **Excel 04 → Ringkasan Gate:** jumlah jawaban, mean, SD populasi, minimum, dan maksimum alpha; terpisah per fold, OOF gabungan, dan test.
- **Excel 05 → Konfigurasi Gate:** resep dan norma parameter dari lima checkpoint fold serta satu checkpoint refit. History, bobot ScalarMix, dan 24 baris proyeksi multihead tetap tersedia.
- **Excel 02 → Fusion S3 Sebelum Gate / Fusion S3:** vektor concat langsung dan vektor sesudah gate yang masuk MLP. **Gate Final Semua Jawaban** berisi inspeksi 300 jawaban memakai model refit; ini bukan prediksi OOF atau evaluasi 300 data.

Laporan prediksi lengkap tetap memuat skor mentah, normalisasi, pembulatan, selisih tepat 0/1/2/3/4/5/10 dan lainnya, toleransi ≤n, MSE, MAE, serta error maksimum. Normalisasi berbeda dari pembulatan.

Alpha disimpan **dari forward yang sama dengan prediksi**, dalam cache fold dan CSV test. Resume dapat membangun laporan dari hasil tersimpan tanpa mengevaluasi test lagi. Pemeriksaan sebelum ZIP memverifikasi metrik, OOF, rekap selisih, ID, nilai, alpha, pengali, statistik ringkasan, dan resep gate. Checkpoint E51 memakai identitas kode dan konfigurasi tersendiri; source E50 dan HTML lama tetap tersedia.

Alpha gate mengatur fitur, **bukan persentase kontribusi terhadap skor atau probabilitas jawaban benar**. Gate bisa mempelajari bobot hampir konstan atau saturasi. Variasi antarjawaban perlu diamati dari hasil; tidak dipaksakan. Bobot attention Excel 02 tetap rata-rata empat head.

## Landasan penelitian

Arevalo dkk. (2017), [Gated Multimodal Units for Information Fusion](https://openreview.net/forum?id=Hy-2G6ile), [kode penulis](https://github.com/johnarevalo/gmu-mmimdb). GMU asli memadukan representasi berdimensi sama dengan sigmoid gate dan penjumlahan tertimbang. **E51 merupakan adaptasi gated concatenation dua encoder teks**, memakai dua scalar softmax dan faktor 2 untuk menjaga kondisi awal identik. Ini bukan reproduksi persis GMU atau bukti bahwa resep ini optimal untuk AES. Rujukan multihead dan co-attention tetap mengikuti E50.

Acuan E50: mean RMSE CV 5,7954; MAE validasi 4,4512; selisih mentah ≤5 poin sebanyak 157/240. Test E50: RMSE 5,3059 dan MAE 3,8856. E43 dengan RMSE test historis 5,1265 tetap pembanding. **Belum ada hasil penelitian E51**. Pilih resep melalui validasi; perbandingan test historis bersifat eksploratif. Gate belum menjamin peningkatan RMSE atau target 60/60 dalam ±3/±5 poin.

## Reproduksi dan pemeriksaan

Source E50 dipin ke commit `c36003670ad9476951b8b607a6018e5a24ceb7e2`. Builder `scripts/build_e51_notebook.py` menghasilkan notebook utama dan mirror yang identik. Namespace checkpoint memuat hash kode dan lesson version E51. Mode fusion/gate diperiksa saat memuat checkpoint. Source E43/E50 dan HTML historis tetap tersedia.

Cache JSON fold memuat indeks validasi, prediksi, alpha, dan metadata training. Indeks dipetakan ke ID baris aktual sehingga baris Excel kosong tidak menggeser pasangan gate/prediksi. CSV test menyimpan `gate_roberta` dan `gate_bge` bersama prediksi dan dilindungi hash hasil test. Resume memulihkan model, optimizer, scaler, scheduler, loader dan RNG.

Pemeriksaan unduhan membaca ulang hasil Excel: metrik test, mean/SD CV, prediksi OOF, rekap selisih, konfigurasi multihead, serta ID/jumlah/label/prediksi gate. Alpha harus finite, dalam 0–1 dan berjumlah 1; pengali harus 2×alpha. Statistik ringkasan dihitung ulang. Enam konfigurasi checkpoint harus sesuai resep dimensi 1.024, dua cabang, faktor 2 dan 2.050 parameter. Tabel wajib kosong atau nilai yang tidak sesuai menghentikan pembuatan ZIP baru.

[Bukti pengujian teknis](../verifikasi/verification_e51_gated_concat_native.md). Pengujian ini bukan jaminan bebas seluruh bug atau bukti peningkatan akurasi.
