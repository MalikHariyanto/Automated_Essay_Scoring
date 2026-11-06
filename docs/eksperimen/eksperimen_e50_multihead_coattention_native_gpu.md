# E50 · E43 + multi-head co-attention4

[Buka Colab E50](https://colab.research.google.com/github/MalikHariyanto/Automated_Essay_Scoring/blob/codex/aes-learning/notebooks/experiments/AES_300_E50_E43_S3_MultiHeadCoAttention4_Native_Learnable_GPU.ipynb) · [Source](../../notebooks/experiments/AES_300_E50_E43_S3_MultiHeadCoAttention4_Native_Learnable_GPU.ipynb).

**GPU T4 → Run all → upload `fix_dataset_300.xlsx`.** Cukup satu Excel;tidak perlu ZIP eksperimen lama. Native Python persection,tanpa magic/worker. Belum ada hasil penelitian E50.

## Perlakuan dan bagian yang tetap

E50 langsung memakai model E43 pada `7241a21dfec37c942948b754da3960fba4c7731c`. Pelaporan memakai perbaikan export E48 pada `437b6ff15808c2122a92cfc32874c815d33af633`;training/model tidak mengambil LDS dari E48/E49.

Perubahan modul adalah co-attention tunggal E43 → **multi-head co-attention4** dengan proyeksi Q/K/V/output. Dimensi total128tetap,4head×32. Tambahan **65.536parameter**,empatLinear128×128 tanpa bias. Tidak menambah attentiondropout,LayerNorm atau residualbaru. Ini penggantian mekanisme attention termasuk proyeksi yang dipelajari,bukan sekadar mengubah angka jumlahhead pada dotproduct lama.

Tiga X=`soal,jawaban_referensi,response_text`;satu y=`rater`0–100. RoBERTa fine-tuned→ScalarMix4+gamma→BiGRU64/arah→multiheadco-attention+matching→maskedmean→proyeksi768;concat dengan BGEfrozenproyeksi256→MLP256/GELU/dropout0,1→satuskor. Soal tetap masuk konteksencoder,sedangkanattention eksplisit memakai token jawaban/referensi. Tidak menghapus BiGRU,ScalarMix,BGE ataufusion.

LRencoder5e-6;proyeksiBGE,MLP,pooling_norm/seluruhQKVO danScalarMix1e-4. Loss kembali **MSE biasa**,target0–1. AdamW/decay0,01,warmup10%,linear scheduler,clipgrad1,microbatch1/batchefektif8 tetapE43. S3saja,satu5CVbaru,maksimal25epoch/patience5,saturefit dari ceil median bestepochCV,lalu satutest. Jadwal mengikuti durasi fit masing-masing sepertiE43,bukan horizonrefit25 E46/E47. Seed42konstanta/foldseed+nomorfold,tanpa sweepseed/kontrolulang. Tidak memakai LDS,ListNet,cosine,S1/S2 atauTransformerblock pengganti BiGRU.

## Mekanisme empathead dan dua arah

BiGRU menghasilkanrespons`[B,T,128]`;proyeksireferensi memberi`[B,R,128]`. ProyeksiQ/K/Vmembagikeduanya menjadi`[B,4,token,32]`. Untuk tiaphead:

```
A = softmax(Q(jawaban) K(referensi)^T / sqrt(32))
B = softmax(Q(referensi) K(jawaban)^T / sqrt(32))
aligned_h    = A V(referensi)
round_trip_h = A (B V(jawaban))
```

ParameterQ/K/V/output yang sama digunakan di kedua arah,sementara kedua matriks softmax dihitung independen. `B` tidak diwajibkan menjadi softmax transpose skor`A`,berbeda dari rumusdotproductE43. Setiaphasilhead digabung ke128 lalu diproyeksikan oleh`W_O`. Matching mempertahankan5fitur:respons,aligned,respons−aligned,respons×aligned,roundtrip→Linear640→128+tanh. Maskedmean merangkumtokenresponsvaliddengan1/N,kemudianLinear128→768danLayerNorm/fusion/MLPsepertisebelumnya.

Padding key diberi−inf sebelumsoftmax;querypadding dinolkan;setiapurutan harusmemilikitokenvalid. Softmax/produkattention dihitungFP32saatencoderAMP. Tidak ada causal mask;ini pencocokan dua teks untuk penilaian. Informasi urutan tetap berasal dariRoBERTa/BiGRU. Head tidak diberi tugas semantik tertentu dan tidak dijamin mempelajari pola yang berbeda.

Seluruh parameter/layer bersama diinisialisasi dalam urutanE43. Proyeksi baru dibentuk setelahlayerbersama di dalam`torch.random.fork_rng(devices=[])`padaCPU,supayastateRNGumumtetap. Semua proyeksi baru dilatih pada1e-4. Identitasconfig/hash/lesson_version/checkpointE50baru;loader menolakmetadatajumlahheadberbeda. Resume memulihkan model,optimizer,scheduler,scaler,loader/RNG.

## Laporan dan pemeriksaan unduhan

Lima Excel dalam satu ZIP:

- **04→Nilai Guru dan Prediksi:**mentah0–100,normalisasi0–1,pembulatan,error. **Selisih Tepat:**=0/1/2/3/4/5/10danlainnya. **Toleransi Selisih:**≤n,jumlah/persentase mentah/bulat. **Ringkasan Error Test:**MSE/MAE/maxerror. **Prediksi Validasi OOF:**240developmentterpisahdari60test.
- **05→Fusion(S3):**MSEbiasa danriwayatCV/refit. **MAE dan Toleransi CV:**perfold/OOFtermasuk±3/±5/maxerror. **Bobot ScalarMix:**24baris checkpointterpilih. **Proyeksi Multihead:**4baris×6checkpoint=24baris,jumlahhead/dimensidanparameter,normaQ/K/Vperhead/normaoutputkolomperhead. Norma parameter bukanbobotattention atauukuranpentingnyasemantik.
- **02/trace:**bobotalignmentyangditampilkan mempertahankan bentukmatriks3DE43 melalui **rata-rata probabilitas empathead**;bukanheadtertentu. Matriksperheadtersedia pada`sequence_pool.last_cross_attention_heads`dan`last_reverse_attention_heads`hanyasaat`capture_alignment=True`;tidakmenyimpansemuamatriksseluruhdataset. Maskedmean tetapbobot1/N. **01/03**tetapdata/tokenisasidankata/kalimat.

Metrik/earlystop tetapRMSEvalidasitanpabobot;testtidakmemilihepoch. ExportbindingfilemencegahbugringkasankosongE44/E45. Gate wajibsebelumZIPmembacaulangExcel,memeriksalima file,ringkasannonempty,test/OOFmetric,meanSDCV,ID/split/label,rekapselisih,historyfinite,serta24barisproyeksidengandimensi128/head4/32perhead/param65.536/normfinite. KegagalanmemblokirZIPbaru. Source/hasil/HTMLhistoristidakditimpa.

## Rujukan dan interpretasi

- Vaswani dkk.(2017), **[Attention Is All You Need](https://arxiv.org/abs/1706.03762)**;[AnnotatedTransformer](https://github.com/harvardnlp/annotated-transformer) untuk proyeksiQ/K/V,multiheaddanpenggabunganoutput.
- Zhang & Litman(2018), **[Co-Attention Based Neural Network for Source-Dependent Essay Scoring](https://aclanthology.org/W18-0549/)**,sebagai landasan pencocokan sumber–esai;implementasiproyekberbeda daripaper.

E50 mengadaptasimultiheadkejalurduaarah/roundtripE43. Empathead,total128,biasFalse danpilihanberbagiproyeksiadalahrancanganproyek,bukanoptimumAESyangditetapkanpaper. AcuanE43meanCV5,8876/test5,1265;pilihperubahanmelaluiCVsebelummenggunakanhasiltest. Hasiltesthistoristetapeksploratif. Belumadajaminanmeningkatkanmetrikataumencapai60/60dalam±3atau±5.
