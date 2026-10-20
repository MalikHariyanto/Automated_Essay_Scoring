# Colab guru300: kode inline dan Excel setiap tahap

Buka [notebook03 di Colab](https://colab.research.google.com/github/MalikHariyanto/Automated_Essay_Scoring/blob/codex/aes-learning/notebooks/03_AES_dataset_guru_300.ipynb), pilih GPU dan jalankan dari atas. Versi20 Oktober2026 mengikuti urutan file BGE-M3 unggahan dan mengganti cetakan `show_code(...)` dengan **kode yang benar-benar dieksekusi di cell**, disertai komentar `#` pada baris penting. Output berisi tabel/angka/status file, bukan source yang dicetak ulang.

## Cara menjalankan

1. Simpan output penting dari tab lama secara privat; buka ulang link notebook terbaru. GitHub tidak otomatis mengubah cell tab Colab yang sudah terbuka.
2. Cell awal mengambil source, menyiapkan dependency Python3.12/3.13 dan memilih `fix_dataset_300.xlsx` tab `Sheet1`. File yang sudah ada pada runtime yang sama dipakai kembali. Runtime baru memerlukan upload ulang.
3. Jalankan cell `%%aes` berurutan. Awalan ini menjalankan kode cell pada **worker Python persisten yang sama**, bukan mencetak kode. Variabel/parameter tetap tersedia di cell berikutnya. Kernel hanya menangani pemilih file dan tampilan HTML, sehingga NumPy kernel lama tidak merusak import numerik worker.
4. Baca preview setiap tahap. File Excel mencakup seluruh baris/koordinat, walaupun preview hanya menampilkan beberapa kolom. Cell yang hanya mendefinisikan fungsi belum menghitung skor; fungsi tersebut dipanggil pada tahap berikutnya.
5. Setelah selesai, unduh `AES_300_semua_tahapan_Excel_PRIVAT.zip` dari cell terakhir. Indeks file ada di `00_indeks_output.xlsx`. Bobot/cache tidak dimasukkan ke ZIP Excel.

Jangan mengunggah Excel, arsip hasil, bobot atau notebook terisi ke repo publik. `/content` Colab hilang jika runtime dihapus; download hasil atau gunakan folder Drive privat untuk penyimpanan antarsesi. Setup yang dijalankan ulang akan menghentikan worker lama; setelah itu jalankan cell berikutnya dari awal.

## Satu nilai guru, bukan empat rubrik

Audit unggahan:300respons,25kelompok email,12pertanyaan, satu nilai `rater` teramati65–99. Empat label dimensi/deskriptor rubrik tidak tersedia. Head empat output mudah diprogram, tetapi satu angka guru tidak menentukan empat nilai terpisah. Workflow ini menghasilkan **satu skor** pada S1/S2/S3; tidak menyalin/membagi satu label menjadi empat.

Untuk penelitian empat dimensi, perlu indikator/deskriptor dan empat label manusia per jawaban. File ini membahas metodologi penelitian dan tidak membuktikan responden SMAkelasXI. Nama kolom `rater` diperlakukan sebagai skor sesuai notebook unggahan, bukan ID penilai. Rentang0–100 ditetapkan mengikuti notebook unggahan, tidak diambil dari min/max test.

Hanya `response_text` masuk model. `soal`, `jawaban_referensi`, email, nomorrespons dan skor tidak disisipkan sebagai fitur. Email digunakan sebagai kelompok pencegah kebocoran. Tiga set teks duplikat lintas email menghubungkan25email menjadi22kelompok terhubung. Kandidat split deterministik dipilih berdasarkan kedekatan jumlah baris ke80/20 saja, tanpa memakai nilai/hasil model. Pada file ini pembagian aktual240development dan60test, dengan17/5kelompok terhubung.

## Alur dan output Excel

| Tahap | Output yang dihitung/disimpan | Workbook |
| --- | --- | --- |
| Konfigurasi/rujukan | Parameter, versi dependency, judul/tahun/halaman dan batas penerapan paper | `01_konfigurasi`, `01_rujukan_paper` |
| Dataset/cleaning | Sumber, audit, teks asli/bersih, skor guru dan target loss | `02_dataset_dan_cleaning` |
| Split/CV | RowID, kelompok, peran development/test dan train/validation tiapfold | `03_split_dan_cv` |
| Tokenisasi | Token/ID, window RoBERTa, padding, attentionmask dan truncation BGE | `04_tokenisasi` |
| RoBERTa pretrained | Seluruh jawaban×768 dan CLS setiapwindow, sebelum fine-tuning | `05_roberta_pretrained` |
| BGE frozen | Seluruh jawaban×1024, CLS+L2, sumber/revision dan truncation | `06_bge_frozen` |
| Kata/kalimat contoh | Koordinat lengkap lookup/kontekstual setiaptoken/kata, CLS kalimat, span/coverage, pasangan kata berulang | `07_vektor_kata_kalimat_pretrained` |
| Arsitektur | Spesifikasi S1/S2/S3, diberi statusDESIGN bukan angka prediksi | `08_arsitektur` |
| CV | Loss setiapepoch, empat metrik, epochpilihan, prediksi/label validation | `09_S1_fold_1` … `09_S3_fold_5` |
| Refit | Riwayat latihan development, epoch tetap dan bukti sampel parameter berubah | `10_S1_refit` … `10_S3_refit` |
| Finaltest | Prediksi mentah/kategori, label, selisih dan empat metrik | `11_S1_test` … `11_S3_test` |
| Ringkasan | Semua fold, mean/SD/jumlahvalid tiapmetrik dan tabel test3skenario | `12_evaluasi_tiga_skenario` |
| Scoring | Satu teks contoh/baru dan tiga prediksi model final | `13_prediksi_satu_jawaban` |
| Fitur setelahtraining | RoBERTa, BGE, hiddenMLP/GELU, proyeksi sebelum/sesudahLayerNorm, concat, outputhead, hanya komponen yang ada pada skenario | `14_S1_fitur_setelah_training` … S3 |
| Kata/kalimat setelahtraining | Contoh yang sama dengan RoBERTa finalS1 danS3; BGE tetap frozen | `15_S1_vektor_kata_setelah_training`, S3 |

Semua nama di atas berakhiran `.xlsx`. Run lengkap default menghasilkan38workbook termasuk indeks. Tahap yang belum dieksekusi tidak diberi angka pengganti. `Info` mencatat dataset hash, revision/checksum sumber fitur, metode pooling, status dan provenance; indeks mendaftar shape setiap sheet. Vektor memakai satu koordinat perkolom, bukan satu string daftar angka. Formula teks seperti `=...` disimpan sebagai teks literal, bukan dijalankan Excel. Sel kosong pada vektor tak tercakup/metricundefined bukan nol; lihat coverage/alasan.

## Metode S1/S2/S3

| Skenario | Alur | Yang belajar |
| --- | --- | --- |
| S1 | RoBERTa→mean CLS window768→dropout→linear1 | RoBERTa dan head |
| S2 | BGE dense CLS1024→L2→MLP1024→256→1 | MLP; BGE frozen |
| S3 | RoBERTa768+BGE1024→proyeksi256+LayerNorm→concat1024→dropout→linear1 | RoBERTa, proyeksi, head; BGE frozen |

BGE pada notebook unggahan fullyfine-tuned. Workflow ini mempertahankan S2/S3frozen yang sudah dibahas; perubahan tersebut dinyatakan terbuka. RoBERTa menggunakan `flax-community/indonesian-roberta-base` revision`cc5ecee7de8532880ad08d802292a6f42865bc83`, dimensi768. BGE `BAAI/bge-m3` revision`5617a9f61b028005a4858fdac845db406aefb181`, dimensi1024. Tidak memakai sparse/ColBERT BGE.

RoBERTa memakai tokenizer slow karena postprocessor fast artifact tersebut tidak menambahkan special dengan benar. Window512 termasuk2special, kapasitas510, overlap128. BGE batas eksperimen1024 termasukspecial; bukan batas maksimum8192 checkpoint. Fitur frozen BGE boleh dihitung untuk semua teks karena tidak ada transformasi yang di-fit pada label/gabungan data; training/pemilihan tetap hanya development.

Default5epoch/fold,5fold, microbatch1/effectivebatch8, AdamW, LRencoder2e−5/head1e−3, warmup10%, weightdecay0.01, clipgrad1. Batchakhir dibobot sesuai ukuran aktual. MSE memakai label `(rater−0)/100` seperti fileunggahan; output dikembalikan ke poin asli sebelum evaluasi. Maksimum epoch dapat diubah sebelum eksperimen, bukan berdasarkan test. Batas5mengikuti rancangan sebelumnya; fileunggahan menggunakan20.

Lima fold adalah validasi di development. Epoch tiapfold dipilih minimum RMSEvalidation; refit menggunakan ceil median epochpilihan. Konfigurasi ketiga model dikunci dan seluruhrefit selesai sebelumfinaltest. Test hanya untuk pelaporan. Fold yang berbagi data tidak independen; SD bukan intervalkepercayaan. Resume memeriksa hash checkpoint/prediksi dan menggunakan testtersimpan. Excel setiapfold tersimpan saat fold selesai dan dapat dibangunulang dari artefak run yang cocok.

## Empat metrik dan arti vektor

RMSE dan Pearson memakai skor mentah0–100: RMSE dalam poin, tidak dibatasi1; Pearson mengukur hubungan linear, bukan ketepatan nilai. QWK dan AC2kuadratik memakai salinan clip0–100 lalu nearestinteger, tie.5 ke bawah, seluruh101kategori dipertahankan. Undefined kosong+alasan, bukan0. AC2 dapat tinggi walaupun QWK buruk karena koreksi peluang/kategori berbeda; baca bersama empatmetrik.

Kata ejaan sama belum tentu IDsubword sama (spasi/case). ID sama pada checkpoint sama menghasilkan lookup yang sama, sedangkan hiddenstate akhir bisa berubah karena konteks dan posisi. Vektor kata=mean subword beririsan span; overlap window dirata-rata pertoken dahulu. Vektor kalimat dibuat dengan membaca kalimat sendiri dan mengambil CLS(meanwindow untukRoBERTa, CLS+L2BGE). Ini bukan mean vektor kata dari jawaban utuh. Kalimat dipisah aturan tandabaca, singkatan bisa keliru. Vektor BGE di luartruncation ditandai tidaktercakup. Jangan menghitung cosine langsung antara ruang768RoBERTa dan1024BGE.

Untuk membandingkan sebelum/sesudah, cocokkan `row_id` pada05dengan14S1/S3. Untuk kata/kalimat cocokkan07dengan15S1/S3. Ekstraksi fitur final dilakukan **setelah test selesai**, modeeval tanpa dropout, untuk inspeksi; tabel semua jawaban bukan hasil evaluasi test baru.

## Rujukan dan batas bukti

Rujukan terpilih ada di [daftar sumber](aes_dataset300_references.json), berasal dari paper library pengguna dan bagian PDF gabungan yang dibaca kembali. Contoh: RoBERTa(m04), M3-Embedding(p14,§3.2/hlm.gabungan3135), hybridAES(p10), AESIndonesia(p08), QWKisNotEnough(m06), dependensiCV(m09). Paperhybrid memakai fiturmanual+RoBERTa, bukan arsitektur proyek yang identik. PaperAESIndonesia memakai pasanganjawaban-acuan, berbeda dari input hanyaresponse_text. Cuplikan bukum08tidak memuat bab rumusAC2; kode AC2 dicocokkan terpisah dengan implementasiirrCAC.

[Bukti eksekusi](verification_dataset300_inline_excel.md) membedakan auditdataasli, fixturetransformeracak, eksekusinotebook, trainingcheckpointasli dan publikasi. Akses HuggingFace di mesin verifikasi ditolak; hasil penelitian harus dihasilkan oleh run Colab yang mengunduh checkpoint asli. Tidak ada angka fixture yang diterbitkan sebagai akurasi siswa. Notebook lama01/02 dan HTML lama tetap mempunyai scope empattrait yang terpisah.
