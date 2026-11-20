# Rangkaian reset · S3 multi-trait

| Eksperimen | Head penilaian | Notebook | Jalankan |
| --- | --- | --- | --- |
| E1 reset | MLP bersama 1024→256→4 | [File](notebooks/AES_Reset_E1_S3_MultiTrait.ipynb) | [Colab](https://colab.research.google.com/github/MalikHariyanto/Automated_Essay_Scoring/blob/codex%2Faes-learning/eksperimen/03_tiga_input_multi_trait/reset/notebooks/AES_Reset_E1_S3_MultiTrait.ipynb) |
| E2 reset | Empat MLP 1024→64→1 | [File](notebooks/AES_Reset_E2_S3_MultiTrait_TraitHeads.ipynb) | [Colab](https://colab.research.google.com/github/MalikHariyanto/Automated_Essay_Scoring/blob/codex%2Faes-learning/eksperimen/03_tiga_input_multi_trait/reset/notebooks/AES_Reset_E2_S3_MultiTrait_TraitHeads.ipynb) |
| E3 reset | Attention jawaban per trait + empat MLP E2 | [File](notebooks/AES_Reset_E3_S3_MultiTrait_TraitAttention.ipynb) | [Colab](https://colab.research.google.com/github/MalikHariyanto/Automated_Essay_Scoring/blob/codex%2Faes-learning/eksperimen/03_tiga_input_multi_trait/reset/notebooks/AES_Reset_E3_S3_MultiTrait_TraitAttention.ipynb) |
| E4 reset | Hidden64 per trait + attention antartrait + regresor128→1 | [File](notebooks/AES_Reset_E4_S3_MultiTrait_InterTraitAttention.ipynb) | [Colab](https://colab.research.google.com/github/MalikHariyanto/Automated_Essay_Scoring/blob/codex%2Faes-learning/eksperimen/03_tiga_input_multi_trait/reset/notebooks/AES_Reset_E4_S3_MultiTrait_InterTraitAttention.ipynb) |

Seluruhnya memakai fusion sederhana, dataset semi-final, split siswa yang sama, dan dropout0,2. E2 adalah adaptasi ide representasi khusus trait dari [CTS](https://github.com/robert1ridley/cross-prompt-trait-scoring/blob/HEAD/models/CTS.py). Dimensi64 dipilih untuk menjaga anggaran parameter head mendekati E1.

E1 dan E2 dapat dijalankan di runtime GPU Colab terpisah. Hasil E2 diunduh sebagai Hasil_AES_Reset_E2_S3_MultiTrait.xlsx, dengan cakupan sheet yang sama seperti E1. Setiap konfigurasi mempunyai folder run dan checkpoint sendiri. Hasil penelitian dibaca dari workbook run masing-masing; notebook publik tidak berisi hasil atau data siswa.

## E4 · Attention antartrait dari dasar E2

E4 dapat dijalankan paralel dengan E3 pada runtime GPU terpisah. E4 memakai CLS RoBERTa, mean antarwindow, proyeksi BGE frozen, dan concat1024 seperti E2. Empat jalur Linear1024→64→GELU→Dropout0,2 membuat hidden per trait. Dot-product/√64 dan softmax menghubungkan satu hidden dengan tiga trait lainnya pada jawaban yang sama. Diagonal attention0; tiap baris berjumlah1. Hidden sendiri64 dan konteks64 digabung menjadi128, lalu Linear128→1 memberi skor.

Attention tidak mempunyai proyeksi Q/K/V atau dropout tambahan. Head bertambah256 parameter menjadi262.916. Bobot hidden sendiri dan bias mengikuti inisialisasi E2; bobot regresor untuk konteks dimulai0 dan ikut dilatih dengan LR1e-4. Jalur ini mempertahankan prediksi awal E2, kemudian dapat mempelajari kontribusi antartrait. Parameter bersama dan keadaan RNG E2 juga dipertahankan. Label guru hanya target loss, bukan input attention. Tidak ada pencampuran jawaban antarsiswa.

Resep training E2 dipertahankan: RoBERTa LR5e-6, komponen lain1e-4, AdamW/decay0,01, warmup10%/linear, clip1, micro1/efektif8, 5CV, maksimal25epoch/patience5, satu refit dan test. Loss tetap MSE0–1 dan seleksi epoch memakai RMSE skala asli. [CTS/Ridley dkk.2021](https://github.com/robert1ridley/cross-prompt-trait-scoring/blob/HEAD/models/CTS.py) menjadi dasar ide attention dari satu trait ke trait lainnya dan concat sebelum prediksi. Hidden64 RoBERTa+BGE, pembagi√64, serta inisialisasi bobot konteks0 merupakan adaptasi proyek; efektivitas E4 belum diketahui.

Satu Hasil_AES_Reset_E4_S3_MultiTrait.xlsx mencakup keluaran E2 ditambah hidden/konteks/concat per trait, matriks attention final seluruh jawaban, dan norma bobot regresor konteks dibanding awal0. Fitur dan attention adalah inspeksi checkpoint refit final, bukan OOF atau bukti sebab-akibat. Metrik penelitian tetap dari prediksi OOF/test tersendiri. Ukuran workbook dapat lebih besar daripada E2.

Uji teknis semua cell telah menyelesaikan 5CV, refit, test, satu workbook71sheet dan pemanggilan unduh memakai Transformer RoBERTa kecil berbobot acak di CPU. Pemakaian ulang hasil selesai, mask diagonal, normalisasi attention, isolasi antarjawaban, gradien antartrait, mixed precision, anggaran parameter, dan kesamaan inisialisasi E2 juga diperiksa. Dataset asli berhasil dibaca1.340×4label dan dibagi1.070/270 pada runtime tanpa CUDA; tahap model memberi status menunggu GPU. Training pretrained E4 di T4 belum dijalankan.

## E3 · Attention pooling khusus trait

E3 berangkat dari E2 reset. CLS bersama diganti empat ringkasan berbobot dari token jawaban pada layer terakhir RoBERTa. Soal dan referensi tetap menjadi konteks encoder. Prefix, special token, konteks, dan padding dikecualikan dari pooling. Empat query768 menambah3.072 parameter, dengan skor dot-product/√768 dan softmax per trait/window. Hasilnya dirata-ratakan antarwindow, lalu masuk norm, concat BGE256, dan MLP trait1024→64→1.

LR, dropout0,2, optimizer, scheduler, split, MSE target0–1, RMSE skala asli, 5CV, maksimal25epoch/patience5, dan aturan refit mengikuti E2. Inisialisasi parameter bersama dan RNG juga dipertahankan. Query awal normal std0,02; LR1e-4. Pilihan rumus dan angka tersebut adalah rancangan eksperimen, belum terbukti paling baik untuk dataset ini.

Dasar ide: representasi dan attention khusus trait pada [CTS/Ridley dkk.2021](https://github.com/robert1ridley/cross-prompt-trait-scoring/blob/HEAD/models/CTS.py), serta attention pada AES hierarkis [Dong, Zhang, dan Yang2017](https://aclanthology.org/K17-1017/). CTS memakai LSTM dan attention tingkat kalimat/antartait; E3 mengadaptasi token attention pada RoBERTa+BGE.

Satu Hasil_AES_Reset_E3_S3_MultiTrait.xlsx memuat cakupan keluaran E2, dengan fitur RoBERTa/concat terpisah per trait, query final, dan bobot token empat trait pada satu contoh development. Bobot berjumlah1 untuk setiap trait di tiap window. Angka ini adalah inspeksi checkpoint refit, bukan evaluasi OOF atau penjelasan sebab-akibat. Empat set fitur final membuat ukuran Excel lebih besar daripada E2.

Uji teknis semua cell telah melewati 5CV, refit, test, satu workbook65sheet, unduh, dan pemakaian ulang fold/refit selesai memakai Transformer RoBERTa kecil berbobot acak di CPU. Mask/gradien/query/head/optimizer serta kesamaan inisialisasi E2 diperiksa tersendiri. Workbook dataset asli melewati pembacaan1.340×4label dan split1.070/270 pada runtime tanpa CUDA; tahap model memberi status menunggu GPU. Training pretrained E3 pada T4 belum dijalankan.

## E1 · Baseline multi-trait reset

Rangkaian ini dimulai kembali dari E1 dengan fusion sederhana. RoBERTa memakai CLS layer terakhir, BGE-M3 frozen diproyeksikan ke 256 dimensi, concat1024 masuk MLP256 dan menghasilkan empat skor.

| Bagian | Rancangan |
| --- | --- |
| Input | Pertanyaan, kunci jawaban, jawaban siswa |
| Target | Isi 1–4, organisasi 1–2, kebahasaan 1–2, mekanik 1–2 |
| RoBERTa | Fine-tuned, LR 5e-6, mean CLS antarwindow |
| BGE | Frozen, proyeksi1024→256 + LayerNorm |
| Head | Concat1024→256→GELU→Dropout0,2→4 |
| Optimizer | AdamW, weight decay0,01, warmup10%, linear scheduler |
| Batch dan training | Micro1/efektif8, maksimal25epoch, patience5 |
| Evaluasi | Lima fold development, refit median epoch, test sekali |
| Seleksi epoch | RMSE skala asli |
| Hasil penelitian | Lihat workbook privat dari run GPU masing-masing |

[Notebook](notebooks/AES_Reset_E1_S3_MultiTrait.ipynb) · [Colab](https://colab.research.google.com/github/MalikHariyanto/Automated_Essay_Scoring/blob/codex%2Faes-learning/eksperimen/03_tiga_input_multi_trait/reset/notebooks/AES_Reset_E1_S3_MultiTrait.ipynb)

## Satu file Excel

Hasil diunduh sebagai Hasil_AES_Reset_E1_S3_MultiTrait.xlsx. Workbook memuat 57 sheet pada uji teknis, dengan penambahan bagian sheet otomatis jika melewati kapasitas Excel.

- Data asli dan hasil cleaning, empat target, kelompok development/test, serta anggota lima fold.
- Token, ID, mask, window, dan cakupan tiga input untuk kedua encoder.
- Vektor RoBERTa sebelum training dan BGE frozen untuk seluruh jawaban.
- Vektor token awal dan kontekstual, kata, kalimat, serta kata berulang untuk satu contoh jawaban development.
- Fitur final RoBERTa, proyeksi BGE, dan concat untuk seluruh jawaban.
- Inspeksi token/kata/kalimat pada checkpoint RoBERTa final untuk contoh development yang sama.
- Riwayat training, konfigurasi, arsitektur, RMSE CV, OOF, dan hasil test per trait.

Fitur seluruh jawaban adalah inspeksi checkpoint refit. Hasil penelitian memakai prediksi OOF dan test tersendiri. Workbook memuat teks siswa dan disimpan secara privat.

Dataset dibaca dari Drive semi-final yang sama. Pilih GPU, jalankan dari atas, lalu hubungkan Drive. Rangkaian E1–E5 sebelumnya dapat dibuka melalui [katalog historis](../README.md).

## Pemeriksaan runtime E2

E2 menampilkan perangkat yang tersedia sebelum pembacaan dataset. Cell **Siapkan dataset dan split** menyiapkan cfg, bundle, manifest, dan output sekaligus. Pada runtime CPU dengan konfigurasi GPU, tahap model menampilkan status belum dijalankan dan menunggu GPU. Pilih T4, kemudian Run all dari awal pada sesi GPU.

Revisi ini diuji dari setup sampai satu workbook 57 sheet menggunakan Transformer RoBERTa kecil berbobot acak di CPU. Workbook asli diuji untuk pembacaan label dan split tanpa CUDA. Pemakaian ulang fold/refit selesai juga diperiksa. Pengujian tersebut adalah uji teknis, sementara training pretrained pada T4 bergantung runtime Colab.

## Membagikan workbook besar

[Pecah Excel di Colab](https://colab.research.google.com/github/MalikHariyanto/Automated_Essay_Scoring/blob/codex%2Faes-learning/notebooks/06_Pecah_File_Excel.ipynb) menerima upload dari komputer, membagi file menjadi bagian maksimal30MiB, dan menyediakan tombol unduh beserta manifest checksum. Gunakan runtime CPU pada sesi baru.
