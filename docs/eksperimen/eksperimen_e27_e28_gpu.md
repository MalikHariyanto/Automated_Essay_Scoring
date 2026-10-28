# E27 dan E28 · dua ablation dari E23, Colab GPU

Pengguna meminta dua notebook terpisah, kemudian memilih E23 sebagai dasar karena memprioritaskan banyaknya prediksi mendekati guru. Pilihan ini mengganti draf E26 yang belum dipublikasikan; tidak mengubah fakta bahwa E26 memiliki RMSE/MAE test lebih rendah. **E27 menambah co-attention; E28 hanya menurunkan LR MLP S3.** Keduanya berasal dari [E23](../../notebooks/experiments/AES_300_E23_Tiga_Input_BiGRU_GPU.ipynb) pada commit `f3d25c4`. Notebook E21–E26 dan buku hasil tidak dibangun ulang atau ditimpa. Belum ada hasil penelitian E27/E28.

| Eksperimen | Source | Perubahan S3 | LR MLP S3 |
| --- | --- | --- | --- |
| E27 | [Notebook](../../notebooks/experiments/AES_300_E27_E23_BiGRU_CoAttention_GPU.ipynb) | E23 + pencocokan token respons–referensi dua arah | `1e-4` |
| E28 | [Notebook](../../notebooks/experiments/AES_300_E28_E23_BiGRU_MLPLR5e6_GPU.ipynb) | Arsitektur E23 persis; LR `model.head.*` saja berubah | `5e-6` |

## Konfigurasi yang dipertahankan

- X seluruh skenario: `soal`, `jawaban_referensi`, `response_text`; y: satu label guru `rater`0–100. ID/email dipakai untuk pemetaan/kelompok, bukan fitur encoder.
- S1 RoBERTa full fine-tuning + linear; S2 BGE frozen + MLP; S3 RoBERTa full fine-tuning + BGE frozen + proyeksi/fusion MLP.
- RoBERTa Indonesia dan BGE-M3 beserta revision mengikuti E23. Input **single_sequence**, bukan format context pair E22.
- LR encoder S1/S3 `5e-6`; head S1/S2, proyeksi BGE dan layer tambahan S3 `1e-4`. Nilai LR adalah awal/puncak; scheduler warmup+linear tetap berjalan.
- MSE target0–1; maksimal25epoch, patience5, lima fold, microbatch1/batch efektif8, trainingseed42/splitseed2026. Development240/test60 untuk dataset300baris; refit memakai ceil median epoch pilihan CV.
- Seluruh S1/S2/S3 dilatih baru; tidak otomatis mengambil hasil E23. Seed/perangkat/versi yang sama tetap diperlukan untuk membandingkan kontrol.
- GPU wajib; AMP bf16/fp16 otomatis. Tanpa LoRA, tanpa membekukan RoBERTa, tanpa mount/autosave Drive.

## E27: letak dan bentuk co-attention

```text
RoBERTa: soal + referensi + respons
          │                 │
    respons768         referensi768
          │                 │
   BiGRU64/arah          Linear128
          │                 │
    urutan128 ← co-attention dua arah → urutan128
          │
 matching640 → Linear128 + tanh
          │
 ujung maju64 + mundur64 → Linear768 → LayerNorm
          │
 concat dengan proyeksi BGE256 → MLP1024→256→1
```

Untuk respons `R` dan referensi `K`, keduanya128dimensi: skor `S=R Kᵀ/√128`. `A=softmax(S)` sepanjang token referensi dan `B=softmax(Sᵀ)` sepanjang token respons. Referensi selaras `A K`; konteks jalur balik `A(B R)`. Fitur pencocokan `[R, AK, R−AK, R⊙AK, A(BR)]`640dimensi diproyeksikan ke128; ringkasan mengambil64komponen pertama dari token akhir arah maju dan64komponen terakhir dari token awal arah mundur, lalu Linear768. Ini mempertahankan pola ujung dua arah E23 tanpa additive attention pooling E26. Matriks/probabilitas dihitung FP32 saat AMP agar masking stabil. Gradien kedua arah tetap mencapai RoBERTa, BiGRU, proyeksi referensi dan proyeksi pencocokan.

Model membaca tiga teks bersama seperti E23; pencocokan ini menambah hubungan eksplisit, bukan pertama kalinya referensi masuk model. Biaya pencocokan bertambah sebanding jumlah token respons×referensi per window; ini bukan attention atas satu vektor BGE.

Mask berasal dari layout posisi token dan offset karakter. Slow tokenizer tetap dipakai untuk input RoBERTa; fast tokenizer hanya dibutuhkan untuk offset referensi dan ID-nya wajib identik. Token isi dipilih berdasarkan rentang karakter, bukan menebak ID token yang sama. Prefix/soal/special/padding dikecualikan dari pooling/pencocokan langsung. Informasi konteks tetap tersedia melalui self-attention encoder. Referensi tetap mengikuti anggaran konteks E23; jika seluruh isi referensi terpotong, notebook berhenti dengan pesan jelas.

Inisialisasi semua parameter E23 yang sudah ada dipertahankan; dua proyeksi co-attention dibuat sesudah layer acuan. Keluaran model E27 tetap berubah karena arsitektur bertambah.

**Rujukan:** Zhang & Litman2018, [Co-Attention Based Neural Network for Source-Dependent Essay Scoring](https://aclanthology.org/W18-0549/), DOI `10.18653/v1/W18-0549`; [kode penulis](https://github.com/Rokeer/co-attention). Paper memakai CNN/LSTM/co-attention tingkat kalimat dengan teks sumber. Notebook ini adaptasi token RoBERTa/BiGRU/BGE dan kunci jawaban; bukan reproduksi atau klaim peningkatan yang sudah terbukti.

## E28: LR yang benar-benar berubah

| Parameter S3 | Kelompok optimizer | LR |
| --- | --- | ---: |
| `roberta.*` | `roberta` | `5e-6` |
| `projection.*` | `projection` | `1e-4` |
| **`head.*`** · dua Linear MLP | **`head`** | **`5e-6`** |
| `sequence_pool.*`, `roberta_norm.*` | `pooling_norm` | `1e-4` |

S1/S2 head tetap `1e-4`. GELU/dropout tidak mempunyai parameter trainable. Fungsi optimizer memeriksa setiap parameter trainable tercakup tepat sekali. E23 menyatukan pooling/norm ke `head`; pemisahan ini diperlukan agar penurunan20× terbatas pada MLP. Tidak ada co-attention pada E28. Loss, input, arsitektur, inisialisasi dan evaluator E28 tetap E23. LR lebih kecil bisa kurang belajar dalam25epoch; jangan menjanjikan hasil lebih bagus.

## Cara menjalankan dan keluaran

Buka source di Colab, pilih **Runtime → Change runtime type → T4 GPU → Run all**, unggah `fix_dataset_300.xlsx`. Keduanya notebook mandiri dengan47cell/30cellkode; tidak perlu clone repo atau file Python pendamping. Dataset mengikuti schema E23; kolom metadata yang diperlukan harus tetap ada.

Workspace lokal berbeda:

- `AES_300_E27_E23_BiGRU_CoAttention_GPU`
- `AES_300_E28_E23_BiGRU_MLPLR5e6_GPU`

Masing-masing menyimpan dua slot progres per tahap dan checkpoint final secara lokal. Ini membatasi file per epoch, tetapi seluruh fold/model masih membutuhkan ruang runtime. Tidak menghapus Drive atau checkpoint eksperimen lain. Runtime hilang berarti progres lokal hilang. ZIP ringkas tidak menyertakan bobot model; simpan berkas model terpisah jika diperlukan. Jangan menyalin checkpoint antar-resep untuk melanjutkan training.

Keluaran akhir tetap lima Excel: data/tokenisasi, vektor jawaban, vektor kata/kalimat, prediksi/evaluasi, dan riwayat pelatihan. LR MLP S3 dan LR pooling/norm dicatat terpisah pada `Pengaturan Eksperimen` dan checkpoint training.

E27 menambah sheet pada `02_Vektor_Jawaban.xlsx`:

- **Ringkasan CoAttention S3**: vektor128 ringkasan ujung dua arah setelah pencocokan, sebelum proyeksi768.
- **Bobot Referensi S3**: rata-rata bobot respons→referensi per window, jumlah1 pada token isi referensi.
- **Pasangan Token Top1 S3**: satu token referensi dengan bobot terbesar untuk setiap token respons. Ini ringkasan, bukan seluruh matriks, skor kebenaran, atau bukti sebab-akibat. Posisi pada Excel mulai1, data mentah internal mulai0.

Sheet **Bobot CoAttention Respons S3** memuat rata-rata bobot referensi→respons per window (jumlah1); ini bobot pencocokan, bukan bobot pooling. E28 tetap tidak memiliki attention pooling seperti E23. Prediksi mentah test tetap sumber metrik; tabel fitur seluruh300jawaban hanya untuk inspeksi, bukan evaluasi generalisasi.

## Validasi dan keputusan eksperimen

Tes implementasi mencakup shape/mask/padding/urutan batch, gradien respons dan referensi, softmax finite, cakupan optimizer, perubahan LR MLP saja, kesamaan kontrol S1/S2 dan arsitektur E28 dengan E23, reload state, seluruh cell dengan Transformer acak kecil, lima laporan, dan resume kernel baru pada folder lokal yang sama. Fixture teknis bukan training pretrained GPU pada data penelitian. Rincian hasil tes final dicatat pada [bukti verifikasi](../verifikasi/verification_e27_e28.md).

Bandingkan E27 dan E28 **secara terpisah terhadap E23**. Gunakan split/fold sama, pilih melalui development/CV, catat biaya dan seluruh hasil. Test lama sudah berulang kali dilihat sehingga hasil berikutnya bersifat eksploratif. Setelah kandidat dikunci, uji baseline dan kandidat pada seed42/123/2026; jangan memilih seed terbaik berdasarkan test.
