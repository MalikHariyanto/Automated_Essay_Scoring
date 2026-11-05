# E46/E47 · CV dipakai ulang, satu refit, horizon LR25

Pengguna menyetujui dua diagnostik untuk menyelaraskan jadwal learning rate refit dengan CV. Model acuan tetap E43; E45 menjadi pembanding karena CV membaik tetapi test memburuk. Belum ada hasil penelitian E46/E47.

| Notebook | Model acuan | Training baru | Horizon jadwal LR | ZIP acuan yang diperlukan |
|---|---|---:|---:|---|
| [E46 Colab](https://colab.research.google.com/github/MalikHariyanto/Automated_Essay_Scoring/blob/codex/aes-learning/notebooks/experiments/AES_300_E46_E43_S3_Refit18_Schedule25_Native_Learnable_GPU.ipynb) | E43, ScalarMix4+gamma | Refit18epoch | 25epoch | `Hasil_AES_300_E43_E42_S3_MaskedMeanPooling_Native_Learnable_GPU.zip` |
| [E47 Colab](https://colab.research.google.com/github/MalikHariyanto/Automated_Essay_Scoring/blob/codex/aes-learning/notebooks/experiments/AES_300_E47_E45_S3_Refit7_Schedule25_Native_Learnable_GPU.ipynb) | E45, ScalarMix12+gamma | Refit7epoch | 25epoch | `Hasil_AES_300_E45_E43_S3_ScalarMix12_Native_Learnable_GPU.zip` |

[Source E46](../../notebooks/experiments/AES_300_E46_E43_S3_Refit18_Schedule25_Native_Learnable_GPU.ipynb) · [Source E47](../../notebooks/experiments/AES_300_E47_E45_S3_Refit7_Schedule25_Native_Learnable_GPU.ipynb).

## Hipotesis dan satu perubahan numerik

Kode lama menghitung `total_steps=len(loader)*epochs`. PadaCV,epochs=25. Pada refit,epochs=median best epochCV, yaitu18untukE43dan7untukE45. Karena warmup10%dan decaylinear mengikuti totalsteps, jalurLRikut berubah. DiCVakhir epoch7,LRsekitar80%puncak;diakhirrefit7,LRmenjadi0. Warmup juga berubah dari2,5epochCVmenjadi0,7epochrefit7atau1,8epochrefit18.

E46/E47 memisahkan **jumlah epoch training** dari **horizon scheduler**. Jumlahtraining tetap18/7dariCVlama;horizon tetap25. Untuk240development/batchefektif8,ada30step/epoch,totaljadwal750,warmup75step. Jika tidakada AMPskip,LRakhirE46=14/45≈31,11%puncak,LRakhirE47=80%puncak. Riwayat mencatatoptimizerstepsaktual,sehinggavalidasiLRtetapbenar bilaGradScalermelompatiupdate.

Ini perubahan protokoloptimisasi yang diuji, bukan kesalahanmatematislama ataupenyebabburuknyatestyangtelahterbukti. Tidak memaksa kedua model dilatih25epoch. TidakmengubahLRpuncak,arsitektur,lossMSE,dropout,seed,split,atauinput. CVtidakdiulangdanangkaCVtidakbaru. Satu refit dimulai dari pretrained, bukan melanjutkan checkpoint final/CVlama.

## Input dan verifikasi sebelum training

PilihT4GPU→Run all→unggah **dua file**: `fix_dataset_300.xlsx` danZIPacuan sesuai tabel. Ada dua pemilihfile;pathmanual tersedia melalui `DATASET_FILE`/`CV_SOURCE_FILE`. GunakanZIPasli yang sudahdiunggah pada chat, tanpa mengubah isi atau mengemasulang. HashZIP dikunci:

- E43: `bc7b5bd88f574a478b40dc6cc5514c52e02655e19b4c33568524812511c5dd0a`.
- E45: `6fea631fe1f86c3e37072506b8cbc4248ef41bf0b8d22290464a9c04d069ef64`.

Sebelum encoder dilatih, kode memeriksa seluruhteks/label/ID/urutan,development/test,anggotalimafold,26pengaturan yangtercatat,historyepoch/metricterbaik,bobotcampuran sertaagregatMAE/toleransi. Medianepoch harus18/7. UntukE45,240pasanganOOFacuan juga diperiksa terhadapfold/label/metric. E43tidakmenyertakanOOFperjawaban,sehinggatidakdibuatprediksipengganti. Tabelhasiltestlama tidakdibacauntukmemilihepochatauLR.

Verifikasiini terbataspadafieldyangtersimpan. ZIPtidakmencatatsemuarevisimodel,versiruntime,weightdecay/warmupdanparameterlain. Pengaturanyangtidaktercatat mengikuti sourceacuan yangdipin;kode tidakmengklaimseluruhlingkunganlamaterbuktiidentik. FileXLSXyangberbedabiner tetapi semantikdata/foldtetapsama dapatlolos;ZIPacuan haruscocokhashasli.

FiturBGEfrozen300×1024dibacadarifile02ZIPterverifikasi,ID/dimensi/finite diperiksa,dikembalikankeFP32asli. Ini menjaga inputcabangBGEidentikdanmenghindariekstraksiulang300jawaban. Inspeksivektorkatamasihmenggunakanmodelpretrainedsepertinotebooklama. RoBERTa/BiGRU/co-attention/proyeksi/MLPtetapdilatihbaru;BGEencoderfrozen. X=`soal,jawaban_referensi,response_text`;y=`rater`0–100untukloss/evaluasi,tidakmasukencoder.

## Pengaturan yang tetap

S3saja;ScalarMix4atau12+gamma;BiGRU64perarah;co-attention/matching128;maskedmean1/Nvalid,padding0;RoBERTa768+BGEproyeksi256→concat1024→MLP256/GELU/Dropout0,1→1. LRencoder5e-6,semuakomponenlain1e-4. MSEtarget0–1,AdamW,decay0,01,warmup10%,clipping1,microbatch1/batchefektif8. TRAIN_SEED42satu konstanta,tanpasweepseed/ListNet/cosine/S1/S2. Patience5diverifikasipadaCVlama;earlystoppingtidakditerapkanpadarefitdurasiterkunci.

Notebook189cell/92cellkodePythonbiasa,semuafungsiterlihatpersection,tanpamagic/worker/importrepo. `EXPECTED_REFIT_EPOCHS`dan`REFIT_SCHEDULER_EPOCHS`terlihatpadahyperparameter. PetaX/y/model/concat/optimizer/fitmengikutiE43/E45. Cache,folder,checkpointdanZIPterpisah. Resumememulihkanoptimizer/scaler/scheduler/loader/RNGdanmenolaksumber/configberbeda.

## Lima Excel dan pengaman unduhan

OutputtetaplimaExcelstandar,satupaket`Hasil_AES_300_E46_E43_S3_Refit18_Schedule25_Native_Learnable_GPU.zip`atau`Hasil_AES_300_E47_E45_S3_Refit7_Schedule25_Native_Learnable_GPU.zip`.

- **04→Perbandingan Model** berisihasiltestmodelrefitbaru. **Hasil Tiap Fold/Ringkasan Validasi** adalahCVacuan,dengan **Provenansi CV** yangmenyatakan0CVbaru/5folddipakaiulang. **Nilai Guru dan Prediksi** khusustestbaru.
- **05→Fusion (S3)** hanyahistoryrefitbaru18/7epoch. **CV Acuan (dipakai ulang)** dan **Ringkasan Fold CV Acuan** memisahkanhistorylama. **Bobot Scalar Mix** hanyacheckpointrefitbaru4/12baris;20/60barisbobotCVlamaadadi **Bobot Mix CV Acuan**.
- **05→Jadwal LR Refit** memuatepoch,kelompok,LRpuncak,LRakhir,optimizersteps,horizon,totalstepsdanwarmup. **MAE Toleransi CV Acuan** hanyaringkasanvalidasilama.
- **04→OOF Acuan (dipakai ulang)** hanyaadajikaZIPacuanmenyertakannya(E47). TidakmembuatOOFbarudarimodelyangsudahmelatihseluruhdevelopment. TidakmencampurOOFacuandengantest.
- **01/02/03** tetapberisidata/tokenisasi,vektorjawaban,dankata/kalimat. Vektorfusionbaru,dengandimensiyangsama;fiturBGEtrainingdipakaiulang.

BugE44/E45diperbaikipadaexporterbaru:dalamgenerator`final_rows`dan`summaries`,filediikatmelaluidefaultargument,sehinggatidakberubahketikaloopOOFberjalan. SumberhistorisE43/E44/E45tidakditimpa.

**SebelumZIPdibuat**,gatewajibmemeriksalimaExcel,datasetdanvektor1024/128finite/lengkap,tabeltest1baris,CV5baris,ringkasanCV1baris,jumlah/ID/labelprediksitest,keempatmetriktestdariprediksitersimpan,mean/SDCV,historirefit,bobotcampuran,danLRaktualsesuaischeduler. Kegagalanmenimbulkanpesan**Unduhan dihentikan**;celltidakmenawarkanZIPbaru. Bukti`verifikasi_unduhan.json`disimpanlokal,paketZIPtetaptepat5Excel. Pemeriksaaninimencakupbugyangdiketahui,bukanjaminanbebasseluruhkemungkinanbug.

## Rincian prediksi tanpa mengubah training

Pembaruan pelaporan 5 November 2026 menambah tiga tabel pada **04_Prediksi_dan_Evaluasi.xlsx** dan tampilan Colab:

| Sheet | Isi |
|---|---|
| Nilai Guru dan Prediksi | Seluruh jawaban **test**, skor guru, prediksi mentah 0–100, pembulatan, normalisasi keduanya, selisih bertanda/absolut, kategori selisih |
| Selisih Tepat | Sama persis, beda **tepat** 1/2/3/4/5/10 poin, serta selisih lain; jumlah dan persentase sebelum/sesudah pembulatan |
| Toleransi Selisih | Jumlah/persentase dengan selisih **maksimal** 0/1/2/3/4/5/10 poin (≤n), sebelum/sesudah pembulatan |

Normalisasi = `(nilai−score_min)/(score_max−score_min)`. Pada rentang 0–100, 1 poin setara 0,01, dan 10 poin setara 0,10. Normalisasi hanya mengganti satuan: tidak membuat prediksi lebih tepat. Jumlah kategori yang setara tetap sama; rekap dihitung dalam poin agar artefak floating point saat membandingkan 0,01 tidak mengubah batas. Prediksi mentah yang di luar rentang dapat memiliki salinan normalisasi di luar 0–1; disimpan apa adanya. Kolom prediksi dibatasi/pembulatan terpisah dari mentah.

Contoh ilustrasi: guru80, prediksi84,6 → selisih mentah4,6; normalisasi0,846 versus0,8, selisih0,046; dibulatkan85 → **beda tepat5poin**. Sebelum pembulatan termasuk ≤5, tetapi bukan beda tepat5. Sama persis pada nilai mentah berarti benar-benar sama, tanpa toleransi tersembunyi. Kategori tepat yang tidak dipilih, termasuk beda6/7/8/9 atau desimal, masuk `Selisih lain`; total kategori tepat tetap seluruh N test. Rekap tidak menggabungkan data training/OOF dengan test.

Metrik penelitian tetap memakai aturan sebelumnya: RMSE/Pearson dari mentah, QWK/AC2 dari salinan kategori terdekat (clip, tepat .5 ke bawah). Gate unduhan membaca kembali Excel tersimpan dan menghitung ulang semua rincian/rekap sebelum ZIP dibuat. Tidak ada perubahan training, arsitektur, LR, label, split atau prediksi.

`TRAINING_CODE_SHA256` mempertahankan identitas training versi awal E46/E47 agar folder/checkpoint yang sama masih dapat dilanjutkan pada workspace yang sama. `NOTEBOOK_CODE_SHA256` tetap berubah untuk mencatat source laporan terbaru. Pengujian membandingkan definisi training dan identitas cache dengan commit `2181cfc`. Jika kelak training berubah, identitas training harus diperbarui.

Jika training sudah berjalan, biarkan selesai dan simpan ZIP. Buka [Colab pembaru ZIP tanpa training](https://colab.research.google.com/github/MalikHariyanto/Automated_Essay_Scoring/blob/codex/aes-learning/notebooks/05_Perbarui_Prediksi_Hasil_Tanpa_Training.ipynb), upload ZIP E46/E47 lengkap, lalu unduh `_Prediksi_Lengkap.zip`. Tidak memerlukan GPU/dataset/model. ZIP sumber utuh, hanya Excel04 diubah; empat Excel lain disalin byte-identik. Metrik/CV yang sudah ada dipertahankan. Pembaru menolak hasil dengan ringkasan kosong; perbaikan E44/E45 lama tidak termasuk cakupannya.

[Verifikasi pembaruan pelaporan](../verifikasi/verification_prediction_breakdown.md).

## Rujukan dan interpretasi

Mosbach,Andriushchenko&Klakow(2021), [On the Stability of Fine-tuning BERT: Misconceptions, Explanations, and Strong Baselines](https://arxiv.org/abs/2006.04884), [repositori penulis](https://github.com/uds-lsv/bert-stable-fine-tuning), membahasoptimisasi danvariasigeneralisasiBERT/RoBERTa/ALBERT. Papertidakmenetapkanhorizonrefit25sebagaioptimumAES;penyelarasanadalahhipotesisberdasarkanauditkodeproyek. KomponenmodeltetapmemakairujukanE43/E45.

KarenaCVdipakaiulang,E46/E47tidakdapatdiklaimmeningkatkanCV. Testhistorissudahdilihatberulangkalidanhasilnyaeksploratif. PerubahanjadwalbelumtentumengalahkanE43. [Verifikasi teknis](../verifikasi/verification_e46_e47_refit_native.md).
