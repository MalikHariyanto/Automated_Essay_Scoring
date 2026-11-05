# E49 · E43 + LDS inverse akar setelah smoothing

[Buka Colab E49](https://colab.research.google.com/github/MalikHariyanto/Automated_Essay_Scoring/blob/codex/aes-learning/notebooks/experiments/AES_300_E49_E43_S3_LDS_InverseSqrt_Native_Learnable_GPU.ipynb) · [Source](../../notebooks/experiments/AES_300_E49_E43_S3_LDS_InverseSqrt_Native_Learnable_GPU.ipynb).

**GPU T4 → Run all → upload `fix_dataset_300.xlsx`.** Cukup satu Excel. Tidak membutuhkan ZIP hasil lama. Native Python persection, seluruh fungsi terlihat, tanpa magic/worker. Belum ada hasil penelitian E49.

## Satu perubahan

E49 diturunkan dari source E48 pada `437b6ff15808c2122a92cfc32874c815d33af633`. Perlakuan hanya kekuatan pembobotan setelah smoothing:

- E48: `a_i = effective_frequency(y_i)**(-1)`.
- E49: `a_i = effective_frequency(y_i)**(-0.5)`.
- Keduanya: `w_i = a_i / mean(a seluruh training fit)`; loss `mean(w_i * (prediction-target)**2)`.

Tidak menambah clipping/cap, tidak mengubah Gaussian5/sigma2/bin1poin, sampler, LR, dropout atau arsitektur. `LDS_POWER=0.5` terlihat pada hyperparameter; config E49 mengunci0,5. Rater tidak diganti; targetloss0–1, histogram pada poin0–100. Bobot tidak dinormalisasi ulang per microbatch1, agar pembobotan tidak hilang.

**Arsitektur E43 utuh:** RoBERTa fine-tuned → ScalarMix4+gamma → BiGRU64/arah + co-attention dua arah/matching + maskedmean → proyeksi768; gabung proyeksiBGE256 → concat1024 → MLP. BGE encoder frozen. Tiga X=`soal,jawaban_referensi,response_text`;satu y=`rater`. Tidak menggunakan blok Transformer pengganti, cosine, ListNet atau S1/S2.

## Protokol dan batas data

Satu5foldCV baru, maksimal25epoch/patience5, seed42konstanta/foldseed+nomorfold, tanpa sweepseed/kontrolulang. LRencoder5e-6/komponenlain1e-4. AdamW, decay0,01, warmup10%,linear scheduler,clipgrad1,microbatch1/batchefektif8 tetapE43. Refit satu kali dari ceil median best epochCV. Jadwal mengikuti durasi fit masing-masing sepertiE43, bukan horizonrefit25 E46/E47.

LDS hanya membaca label trainfold (~192jawaban). Validasi tidak menentukan bobot. Refit menghitung ulang240development;60test tidak menentukan distribusi, bobot atau epoch. Semua metrik dan earlystop tetap tanpa bobot LDS. Saat inferensi tidak memerlukan rater/bobot LDS. Namespace,lesson_version,fullcodehash dan fingerprintpower/bobot E49 terpisah sehingga checkpoint E48 tidak dipakai sebagaiE49. Resume memulihkan optimizer/scheduler/scaler/loader/RNG.

E48 memburuk terhadapE43:meanRMSECV6,3908vs5,8876,RMSEtest6,4777vs5,1265,MAE4,7292vs3,7817. Satu skor65langka mendapatbobot27,7485saatrefit. Perhitungan E49 pada distribusi training yang sama menghasilkan5,7106 (~2,38%totalbobot vs11,56%). Ini perhitungan distribusi training, **bukan hasil prediksi/perkiraan RMSE E49**, dan bukan bukti sebab tunggal penurunan E48. Acuan tetapE43. Testhistoris yangsudahdilihatberulangbersifateksploratif;nilai kandidatmelaluivalidasi dahulu. Sasaran60/60dalam±3atau±5 belumdijamin.

## Output dan pengaman

Lima Excel dalam satu ZIP, mengikuti E48:

- **04→Nilai Guru dan Prediksi:** mentah0–100,normalisasi0–1,pembulatan,selisih. **Selisih Tepat:** =0/1/2/3/4/5/10danlainnya. **Toleransi Selisih:** ≤n,jumlah/persentase mentah/bulat. **Ringkasan Error Test:** MSE/MAE/maxerror. **Prediksi Validasi OOF:**240developmentterpisahdari60test.
- **05→Fusion(S3):** MSEbiasa dan LDS-MSEterpisah. **Bobot LDS Training:**1.200baris trainfold/refit. **Distribusi LDS Training:**606baris. **Audit LDS:**`weight_power=0.5`,N,mean/min/max,kernel,fingerprint. **Bobot ScalarMix:**24baris. **MAE dan Toleransi CV:**perfold/OOFtermasuk±3/±5/maxerror.
- **01/02/03:**data/tokenisasi,vektorjawaban,kata/kalimat tetap. Gate wajib sebelum ZIP memeriksa lima file,ringkasan nonempty,metriktest/OOFtanpabobot,meanSDCV,ID/split/label,rekapselisih,historyfinite,serta setiapsampleweightdanpower/fingerprint audit dari trainfold yang sesuai. Bobot/audit E48 tidak boleh diterima sebagaiE49.

## Landasan dan adaptasi

Yangdkk.(ICML2021), [Delving into Deep Imbalanced Regression](https://arxiv.org/abs/2102.09554), [repositori penulis](https://github.com/YyzHarry/imbalanced-regression). Inverse dan inverse-square-root tersedia dalam pilihan pembobotan repositori;paperjuga mencakupSTS-B-DIRuntukregresiskorteks.

E49 mengambil akar **sesudah smoothing**, sehingga hanya eksponen E48 yang berubah. Sebagian pipeline resmi mengambil akar frekuensi histogram **sebelum smoothing**;keduaoperasitidakkamutatif dan tidak diklaimidentik. E49 merupakan adaptasi berbasis keluarga metode tersebut, bukan reproduksi seluruhbenchmarkpaper atau resepoptimalAESyangsudahterbukti. Bin1poin mengikuti skaladataAES. [RancanganE48](eksperimen_e48_lds_native_gpu.md) dan [komponenE43](eksperimen_e43_masked_mean_native_gpu.md) tetaprujukan.
