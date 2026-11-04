# E30 acuan lengkap · S1/S2/S3 · Python biasa

Koreksi permintaan pengguna: **E30 sebagai kode acuan harus lengkap S1,S2 dan S3**, bukan hanya S3. Notebook ini menjalankan ketiga skenario dengan resep historis E30 dan struktur native terbaru. E41 tetap S3 saja. Source E30 lama dan varian native S3 saja tidak ditimpa.

[Buka Colab](https://colab.research.google.com/github/MalikHariyanto/Automated_Essay_Scoring/blob/codex/aes-learning/notebooks/experiments/AES_300_E30_Lengkap_S1_S2_S3_Native_Learnable_GPU.ipynb) · [Source lengkap](../../notebooks/experiments/AES_300_E30_Lengkap_S1_S2_S3_Native_Learnable_GPU.ipynb).

| Skenario | Model | Learning rate puncak |
|---|---|---|
| S1 | RoBERTa fine-tuned,CLS/mean window768,dropout0,1,linear768→1 | Encoder5e-6,head1e-4 |
| S2 | BGE-M3 frozen1024,MLP1024→256→GELU→Dropout0,1→1 | Head1e-4 |
| S3 | RoBERTa→BiGRU64perarah→co-attention respons–referensi→attention pooling→768+LayerNorm;BGE1024→256+LayerNorm;concat1024→MLP256→GELU→Dropout0,1→1 | Encoder5e-6,head/proyeksi/pooling_norm1e-4 |

S1 dan S3 memakai instance RoBERTa yang dilatih terpisah; S2 dan S3 dapat memakai fitur BGE frozen yang sama. X selalu tiga teks `soal`,`jawaban_referensi`,`response_text`; y=`rater`integer0–100tidak dimasukkan ke encoder. S3 memakai keluaran RoBERTa terakhir,matching640→128,attention128→64→1,ringkasan128→768dan concat1024. MLP tetap prediktor akhir S2/S3. Tidak ada Scalar Mix atau LLRD; fungsi/kolom inspeksi cosine tetap dihapus sesuai permintaan sebelumnya.

## Menjalankan dan belajar

Pilih runtime baru T4 GPU → Run all → unggah `fix_dataset_300.xlsx`. Notebook184cell/89codecellmemakai Python biasa langsung di kernel. Nama checkpoint,kolom X/y,hyperparameter,arsitektur,optimizer,loss/training,evaluasi dan unduhan dipisahkan per section. Tanpa cell magic,worker tersembunyi,clone repo atau Drive. Section3memuat nama/revision model,section4kolom X/y,section5hyperparameter,section19arsitektur,section22optimizer/loss dan section24pemanggilan training.

Setiap skenario memakai **5fold CV sekali,maksimal25epoch/fold,patience5,1refit+1test** pada split dan fold yang sama. Total15fit CV dan3refit; masing-masing model final dievaluasi sekali. Ini tiga skenario penelitian, bukan pengulangan tiga seed. TRAIN_SEED42satu konstanta; seed fold=seed+nomor fold. Split_seed2026,holdout20%,grouping,batch efektif8/micro1,AdamW,weight decay0,01,warmup10%,scheduler linear,gradient clipping1dan MSE target0–1mengikuti E30.

Refit setiap skenario memakai ceil median best epoch lima fold. Seluruh pemilihan dikunci sebelum test. Jangan memilih LR/epoch dari test historis yang telah berulang kali dilihat; hasil berikutnya bersifat eksploratif.

## Output dan pemulihan

ZIP `Hasil_AES_300_E30_Lengkap_S1_S2_S3_Native_Learnable_GPU.zip` berisi lima workbook standar: data/tokenisasi,vektor jawaban,vektor kata/kalimat,prediksi/evaluasi dan riwayat training. Workbook04memuat prediksi dan metrik ketiga skenario; workbook05memuat sheet RoBERTa(S1),BGE-M3(S2),Fusion(S3),Ringkasan Lima Fold15baris dan Pengaturan Eksperimen. Masing-masing model dilatih baru, bukan mengambil angka hasil lama.

Cache/checkpoint memakai folder/hash kode/config versi lengkap yang berbeda dari versi S3 saja. Progres lokal dapat dilanjutkan dengan menjalankan ulang runtime yang sama; file hilang bila runtime dihapus. Unduh ZIP sebelum sesi berakhir. Bobot model besar tidak otomatis dimasukkan ke ZIP.

Format diambil dari notebook native E35,kemudian LR pooling_norm S3 dikembalikan ke1e-4agar sama dengan E30. Model,tokenisasi,optimizer,fit,CV/refit/test dan metrik inti identik secara AST dengan source E30 historis. Pengujian per skenario memeriksa kesetaraan state awal,prediksi dan update AdamW. Ini menjalankan ulang resep E30; tidak menjamin RMSE historis5,5247terulang persis pada lingkungan GPU/precision/dependency berbeda. Belum ada hasil penelitian baru pada notebook kosong ini.

Rujukan RoBERTa,BGE-M3,GRU,BiGRU/co-attention AES,pooling,LayerNorm,GELU dan AdamW tetap tersedia di bagian akhir notebook. [Bukti verifikasi teknis](../verifikasi/verification_e30_full_native.md) membedakan fixture CPU dari training penelitian GPU.
