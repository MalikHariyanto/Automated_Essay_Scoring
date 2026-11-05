# Eksperimen lanjutan · acuan aktif E43

## Keputusan terbaru: E49 LDS dengan inverse akar

Pengguna menyetujui satu uji lanjutan setelah E48:kurangi eksponen inverseSETELAHsmoothing1→0,5. Gaussian5/sigma2/bin1,model/LR/dropout/scheduler/input/split E43 tetap. Tidak menambahcap/sampleratauTransformerpengganti. S3,5CVbaru,25epoch/patience5,saturefit+test,tanpaZIPacuan. [Notebook/metodeE49](docs/eksperimen/eksperimen_e49_lds_inverse_sqrt_native_gpu.md).

E48sudahselesai:meanCV6,3908/test6,4777/MAE4,7292;empatfoldmemburuk,satuFold4membaik. Targettepatsetelahbulat6/60,raw≤3=29/60,≤5=39/60,>10=9/60. Bobotrare65 mencapai27,7485saatrefit;versiakar menghasilkan5,7106padadistribusitrainingyangsama. Ini temuanpembobotan,belumbuktisebabtunggalatauhasilE49. E43tetapacuan. Bandingkan melaluiCVterlebihdahulu;testhistoristetapeksploratif. Jika versi lebihlembut tetapkalah,pertahankanMSEE43sebelummengubahkomponenlain.

## Keputusan terbaru: E48 LDS-weighted MSE dari E43

Pengguna mempertahankan seluruh arsitektur E43 dan menyetujui pengujian LDS-weighted MSE. [Notebook dan rancangan E48](docs/eksperimen/eksperimen_e48_lds_native_gpu.md). Distribusi label hanya train fold;Gaussian5/sigma2,bin1poin,inverse setelah smoothing,mean bobot training1. Label asli tidak diubah. Refit menghitung ulang240development;test tidak menentukan bobot. Metrik/earlystop tetap tanpa bobot. S3,5CVbaru,25epoch/patience5,saturefit+test;upload cukupExcel tanpaZIPacuan. Jadwal kembali E43 asli. Landasan Yangdkk.ICML2021DIR,termasukSTS-B;adaptasiAESbelumterbuktiunggul. Belum ada hasilpenelitianE48.

HasilE46/E47 sudah dianalisis:RMSEtest5,3863/8,9816 danMAE4,1258/7,6945,lebihburukE43. CV keduanya adalahhasil lama yangdipakaiulang. Targetpengguna60/60dalam±3,alternatif±5;E43mentah28/60dalam±3dan45/60dalam±5. Hasiltesthistorisbersifateksploratif;pilihpengaturanmelaluivalidasi. Bagian di bawah menyimpan urutan keputusan historis.

## Setelah hasil E44/E45: audit refit E46/E47 disetujui

E44 telah selesai:meanRMSECV6,2100/test6,0337/MAEtest4,5917. E45selesai:meanRMSECV5,7645/test6,3814/MAEtest4,8481. E45bukan kalahpadasemuaukuran;CV-nya lebihbaikE43,tetapitestlebihburuk. E43tetapacuanpengguna. Keduanyarefit7epoch,sedangkanE43refit18. Exporterrusakpadaduasheet ringkasan akibatvariabel`file`late-boundtertimpaloopOOF;60test/240OOF tersimpan,makaanalisisdapatdihitungulangtanparetraining. Sumber/HTMLlama belumdiubah.

Auditkode memastikan `total_steps=len(loader)*epochs` danwarmup10%dari total. CVmenggunakanhorizon25epoch,refitmenggunakanselectedepochs7/18. Dengan192training/batch8,CVepoch7mempunyaiLRakhir80%puncak;refit240data7epochmencapaiLR0. WarmupCV2,5epoch menjadi0,7epoch saatrefit7,atau1,8epoch saatrefit18. Ini perbedaanprotokoloptimisasi yangdapatdiuji,bukankesalahanmatematis ataupenyebabturunnyatestyangsudahterbukti.

**E46 disetujui:** resepmodelE43/ScalarMix4+gamma/MSE,trainrefit18epochpilihanCV,horizonscheduler25epoch. **E47 disetujui:** resepE45/ScalarMix12+gamma/MSE,trainrefit7epochpilihanCV,horizonscheduler25epoch. Tidakmenambahepochtrainmenjadi25,tidakubahLRpuncak/arsitektur/dropout/split/seed,tidakmemakaiListNet. [Dua notebook dan metode](docs/eksperimen/eksperimen_e46_e47_refit_schedule_native_gpu.md). BelumadahasilpenelitianE46/E47. LimaCVlama dipakaisebagaiinputpilihanepochsetelahhashZIP/dataset/fold/parametertersimpandiverifikasi;berartiinisecarajelasdiagnostikrefit,bukan5CVbaruatauangkaCVbaru. BGEfrozenvectorsdipakaiulangdarifile02,hasiltestlama tidakmemilihepoch. Exporterbaru memperbaikigeneratorfilelate-bound dangatewajibmemblokirdownloadbilatable/metric/count/vektor/LRtidakvalid. Testhistoristetapeksploratif;perubahanini tidakdapatdiklaimmeningkatkanCVkarenaCVsama. Dua refitlebihhematdaripada12fitduaresepCVbaru.

Rujukan auditoptimisasi: Mosbach,Andriushchenko&Klakow(2021), [On the Stability of Fine-tuning BERT: Misconceptions, Explanations, and Strong Baselines](https://arxiv.org/abs/2006.04884), [repositori penulis](https://github.com/uds-lsv/bert-stable-fine-tuning). README/abstrakresmidiverifikasi:membahasBERT/RoBERTa/ALBERT,kesulitanoptimisasi danvariasigeneralisasi. Paper tidakmenetapkanhorizonrefit25sebagaioptimumAES;penyelarasan25adalahhipotesismetodologisberdasarkankodeproyek.

## Dua kandidat paralel E44 dan E45

Pengguna menyetujui E45 sebagai kode kedua,langsung dari E43. E44 mengujiMSE+0,1×ListNet;E45 tetapMSE dan memperluas ScalarMix4layerterakhir menjadi seluruh12layerTransformerRoBERTa. Gamma,LR5e-6/1e-4,BiGRU64perarah,co-attention,maskedmean,BGEfrozen,MLP dan dropout tetapE43. Setiap kandidat S3saja,satu5fold,maks25epoch,patience5,satu refit+test,tanpasweepseed/kontrolulang. E43tetapacuan;belum ada hasil penelitianE45. [Notebook/metode E45](docs/eksperimen/eksperimen_e45_scalar_mix12_native_gpu.md).

Dasar E45:ELMo/Petersdkk.(2018)dan implementasi resmiAllenNLP yang mencampur seluruhlayerTransformer dengan `ScalarMix(config.num_hidden_layers)`/`hidden_states[1:]`. Embeddingawal tidak dicampur langsung;parameterembeddingtetapdilatih. E45menambah8logit tanpaRNGtambahan,12bobotawal1/12,gammaawal1;guard jumlahlayer haruscocok modelaktual. Tidak menambahforwardencoderbaru;overheadmixing/gradien tetapperlu diukur. Excel05menyimpan72barisbobot/gamma,Excel04menambahOOFperjawaban,terpisahdaritest. Pilih dariCV,bukan melihat test terlebih dahulu;gabungan ranking+12layerbaru dipertimbangkan setelahhasilmasing-masingmendukung.

## Keputusan 5 November 2026 dan implementasi E44

Pengguna menetapkan **E43 sebagai acuan pengembangan berikutnya**. Hasil yang diaudit: mean RMSE CV5,8876; RMSE test5,1265; MAE test3,7817. E43 mempertahankan Scalar Mix empat layer+gamma, BiGRU64 per arah, co-attention respons–referensi, masked mean token valid, RoBERTa768+BGEproyeksi256→concat1024→MLP. Encoder5e-6,komponen lain1e-4,BGEfrozen,dropoutMLP0,1,S3saja,5fold sekali,maks25epoch,patience5. E30 tetap kontrol historis. E43 unggul pada3/5fold atas E42;Fold4masih7,5279. Kenaikan kinerja test belum menjadi bukti keunggulan yang pasti pada data baru. Hasil/HTML/source lama dipertahankan.

**Pengguna menyetujui pembuatan E44:** pertahankan arsitektur E43, uji loss gabungan MSE + bobot kecil ListNet untuk belajar nilai sekaligus urutan kualitas. [Notebook dan metode E44](docs/eksperimen/eksperimen_e44_listnet_native_gpu.md). Belum ada training/hasil penelitian E44. Sumber langsung AES: Yang,Cao,Wen,Wu&He(2020), [Enhancing Automated Essay Scoring Performance via Fine-tuning Pre-trained Language Models with Combination of Regression and Ranking](https://aclanthology.org/2020.findings-emnlp.141/). Abstrak resmi diverifikasi melalui XML koleksi2020.findings pada repositori acl-org/acl-anthology: paper menggabungkan MSE dan batch-wise ListNet dengan bobot dinamis,dievaluasi QWK pada ASAP. Ini tidak menjamin RMSE membaik pada fusion Indonesia.

Rancangan kandidat proyek memakai bobot ranking tetap0,1sebagai ablasi yang ditulis eksplisit,sehingga bukan replikasi persis bobot dinamis paper dan bukan nilai optimum yang ditetapkan paper. Nilai rater0–100tetap target asli;normalisasi0–1untuk kedua loss tetap. Ranking hanya antarsiswa dengan soal yang sama dari12soal aktual,tidak memperlakukan beda soal sebagai daftar yang setara. Gunakan kelompok soal yang muncul alami dalam batch efektif8;satu jawaban dalam kelompok hanya memakai MSE. Sampler/split/fold tetap. Akumulasi gradien microbatch1padaE43tidak otomatis menghasilkan ListNet atas beberapa prediksi;E44menggabungkan graphprediksi sebelum backward dan memakai activation recompute non-reentrant/RNGpreserve dengan tensor penanda perangkat untuk menekan memori aktivasi. Penggunaan memori GPU aktual belum diukur pada verifikasiCPU. Jangan mengklaim lossrankingaktif bila setiap softmax hanya berisi satu jawaban;logmengukurcakupan dan run berhenti jika seluruh epoch tidak memiliki kelompokranking.

Bandingkan mean RMSE CV terhadap5,8876sebagai ukuran utama;lihat pula MAE CV4,6122dan toleransi≤5poin151/240untuk tradeoff. Seluruh fold dilaporkan;tidak mengoptimalkan Fold4secara terpisah atau menyeleksi berdasarkan test. Ekspor prediksi validasi OOF perjawaban pada eksperimen berikutnya:ZIP E43memuat ringkasan MAE/toleransi,tetapi tidak menyertakan seluruh pasangan prediksi OOF untuk audit residual individual. Tidak ada sweepseed,S1/S2,perubahanLR,atau resepRidgebaru dalam usulan ini.

## Riwayat keputusan E26 dan eksperimen lama

Pengguna sudah menjalankan E27/E28 berbasis E23, lalu menetapkan E26 sebagai acuan berikutnya. **E29 = E26 + LR MLP S2 dan S3 5e-6**; **E30 = E26 + co-attention respons–referensi lalu attention pooling**. MLP E30 tetap1e-4; kedua resep encoder5e-6 dan BiGRU/pooling/proyeksi/norm1e-4, linear S1 tetap1e-4. Kode GPU dan kedua hasil kini tersedia. E29 RMSE S3 terendah5,4293; E30 MAE4,0215 dan≤5poin44/60, tetapi RMSE5,5247. [Review E30](docs/review/review_e30_study.md). [Notebook dan metode](docs/eksperimen/eksperimen_e29_e30_gpu.md).

Usulan berikutnya adalah menguji E26/kandidat dengan tiga seed dan, bila sesuai protokol CV, merata-ratakan prediksi setiap jawaban. **Mean metrik tiga seed bukan ensemble prediksi.** Ensemble belum dibuat. Nomor E31 kini dipakai untuk ablasi LR dari E25, dan E32 untuk co-attention dari E25; jangan menyebut keduanya ensemble.

## Riwayat dua jalur E27/E28 dari E23

**E27/E28 dibuat dari E23 atas pilihan eksplisit pengguna; hasilnya kini tersedia.** RMSE S3 E27=6,3277, E28=7,0447; [review](docs/review/review_e27_e28_study.md). Angka frekuensi enam peserta di bawah menjadi riwayat; delapan peserta kini membuat E23/E26 seri10/60. Prioritas pengguna ialah banyaknya prediksi mendekati guru: E23 paling sering terdekat13/60 dan selisih≤5poin39/60 pada enam eksperimen. RMSE test E23 **6,0439**, MAE **4,8576**, mean CV **6,4143**. E26 tetap lebih rendah RMSE/MAE **test** (5,7435/4,4797), bukan hanya bagus saat training. Perbedaan ukuran menjelaskan pilihan dasar; satu run belum membuktikan konsistensi. Tetap tiga input, label `rater`, RoBERTa fine-tuned, BGE frozen, kontrol S1/S2. [Notebook dan metode](docs/eksperimen/eksperimen_e27_e28_gpu.md).

## Jalur yang dapat dibandingkan

| Jalur | Combo | Yang diuji | Yang dipertahankan |
| --- | --- | --- | --- |
| A · parameter | **E23 + LR MLP saja `5e-6`** | Apakah langkah MLP lebih kecil memperbaiki validasi dan kedekatan prediksi? | Encoder `5e-6`; BiGRU/proyeksi/norm `1e-4`, MSE |
| B · layer | **E23 + co-attention respons–referensi** | Apakah pencocokan eksplisit dengan kunci membantu? | MLP/proyeksi/layer baru `1e-4`, encoder `5e-6`, MSE |
| C · pengukuhan | **Acuan dan kandidat × 3 seed** | Apakah peningkatan konsisten pada 42, 123, 2026? | Split/fold tetap; resep kandidat dikunci |

A dan B diuji terpisah dahulu: **A menjadi E28**, **B menjadi E27**. Gabungan keduanya baru layak setelah CV memberi dukungan masing-masing. Jalur C masih rencana; tidak ada nomor baru atau hasil tambahan yang dibuat otomatis.

## A: LR MLP yang dimaksud pengguna

Head MLP sampai E26 tetap `1e-4`. **`5e-6` turun 20 kali**, bukan kenaikan. Head baru diinisialisasi sehingga LR terlalu kecil dapat membuatnya kurang belajar dalam anggaran 25 epoch. Kandidat lebih moderat `5e-5` dapat direncanakan melalui CV sebelum test, bukan dipilih setelah melihat test.

Pada `optimizer_groups()` E23 asal, seluruh parameter yang bukan `roberta.*` atau `projection.*` masuk kelompok `head`, termasuk:

- `head.*`: MLP regresi;
- `sequence_pool.*`: BiGRU dan proyeksi ringkasan128→768;
- `roberta_norm.*`: LayerNorm tambahan.

Jangan sekadar mengubah `HEAD_LR` untuk uji **MLP saja**. E28 sudah memisahkan `model.head.*` ke kelompok LR sendiri; pooling/norm tetap `1e-4`. Setiap parameter trainable masuk tepat satu kelompok. Perlakuan khusus S3; kontrol S1/S2 tetap memakai resep asal.

## B: combo layer yang paling relevan

**E23 + co-attention respons–referensi** menguji hubungan jawaban dengan kunci secara eksplisit. RoBERTa saat ini sudah membaca ketiga teks melalui self-attention; tambahan ini bukan memperkenalkan referensi yang sebelumnya tidak dibaca. Draf dasar E26 diganti sebelum publikasi; E27 tidak menambahkan additive attention pooling E26.

Rancangan yang diimplementasikan pada E27 dan diperiksa melalui fixture teknis:

```text
RoBERTa membaca soal + referensi + respons
                 │
       ┌─────────┴──────────┐
 token respons768     token referensi768
       │                    │
 BiGRU → urutan128     proyeksi → urutan128
       └─────────┬──────────┘
       co-attention dua arah
                 │
 fitur pencocokan → proyeksi128
                 │
  ujung maju64 + ujung mundur64
                 │
 ringkasan128 →768 → LayerNorm → concat BGE256 → MLP
```

Ini adaptasi, bukan reproduksi paper. Mask referensi/respons harus memakai posisi token sebenarnya, bukan kesamaan ID token. Padding, prefix, special dan referensi terpotong ditangani. Attention membandingkan urutan token; satu key berupa vektor BGE global membuat softmax selalu satu.

Rujukan: **Zhang & Litman (2018), [Co-Attention Based Neural Network for Source-Dependent Essay Scoring](https://aclanthology.org/W18-0549/)**, DOI `10.18653/v1/W18-0549`. Full text diperiksa melalui [salinan pada repo situs penulis](https://raw.githubusercontent.com/Rokeer/academic/master/files/publications/W18-0549.pdf). Tabel4: QWK SELF-ATTN→CO-ATTN **0,677→0,697** pada ASAP3, **0,681→0,697** pada RTA MVP, **0,669→0,684** pada RTA Space. Ketiganya signifikan terhadap SELF-ATTN; peningkatan ASAP4–6 lebih kecil dan tidak signifikan terhadap SELF-ATTN.

Paper memakai **MSE**, CNN/LSTM dan co-attention tingkat kalimat dengan teks sumber. Model, unit, bahasa, label/skala dan dataset berbeda. Bukti ini mendukung alasan menguji pencocokan sumber, bukan jaminan RMSE proyek turun atau ablation terisolasi satu layer.

## Yang belum diprioritaskan

E26 + gate E25 mungkin dicoba kemudian. Namun gate E25 final hanya **0,495721–0,496997**; kedua pengali masih dekat1, dan RMSE E25 lebih tinggi dari E24 pada run itu. Bukti belum mendukung memprioritaskan penumpukan gate.

Huber menjadi uji loss terpisah bila residual development menunjukkan beberapa error besar mendominasi. Belum ada perbandingan AES MSE–Huber terkontrol yang berhasil diverifikasi pada penelusuran proyek. Tidak ada klaim Huber pasti lebih akurat/cepat. Pada target0–1, ambang5poin berarti delta0,05; default delta1 dapat hanya menjadi setengah MSE selama error≤1.

## Cara memutuskan

Gunakan split/fold sama. Catat konfigurasi sebelum membuka test, bandingkan RMSE lima fold, biaya/epoch dan residual. Kunci kandidat/refit melalui development; test yang telah berulang kali dilihat bersifat eksploratif. Tiga seed melatih acuan dan kandidat pada seed yang sama, bukan memilih seed paling bagus. Bedakan mean/SD metrik per seed dari metrik prediksi yang dirata-ratakan.

Acuan implementasi: [E23 GPU](notebooks/experiments/AES_300_E23_Tiga_Input_BiGRU_GPU.ipynb), [metode E27/E28](docs/eksperimen/eksperimen_e27_e28_gpu.md). Pembanding hasil: [review E25/E26](docs/review/review_e25_e26_study.md).
