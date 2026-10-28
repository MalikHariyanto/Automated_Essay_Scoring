# Pemeriksaan notebook tersimpan — 14 Oktober 2026

> Dokumen tahap terdahulu. Angka dan keputusan dibaca dalam konteks versi ini; untuk kondisi aktif gunakan [status proyek](../../STATUS_PROYEK.md) dan [panduan AI](../../PANDUAN_AI_PROYEK.md).

Notebook yang diperiksa adalah `notebooks/02_latihan_acuan_AI.ipynb` pada commit `0b52413771ed535b1767a23924f0b980e0159530` (Created using Colab). File benar-benar tersedia di branch `codex/aes-learning`, bukan hanya terlihat pada dialog simpan. SHA-256 notebook terisi: `f58c5d0e05d1d109cca69a114d73ffb08d2b427a73b52ed704c1889f323ad55f`.

Seluruh 25 cell memiliki source yang sama dengan versi awal `1936057`; 11 cell kode mempunyai nomor eksekusi 1–11, tanpa output error yang tersimpan. Source modul lain tidak berubah pada commit Colab tersebut. Metadata/output notebook berubah karena eksekusi dan penyimpanan.

## Pemeriksaan setiap tahap

| Tahap | Hasil dan batas bukti |
|---|---|
| Setup | Python 3.12.13, PyTorch 2.6.0+cpu, NumPy 2.2.6, SciPy 1.15.3, scikit-learn 1.6.1, Transformers 4.44.2, FlagEmbedding 1.3.5; import berhasil. Ini belum loading model. |
| Input/audit | Workbook lengkap dibaca sebagai xlsx, tab Untuk_CSV_Guru; tabel AI terpisah. Rubrik SHA cocok sumber lokal. Audit 10 draf, 0 label guru lengkap, rubrik belum disahkan. Workbook SHA berbeda dari salinan lokal; angka acuan yang ditampilkan cocok, tetapi seluruh byte workbook Colab tidak tersedia untuk pemeriksaan langsung. |
| Teks/rubrik/cleaning | Contoh teks draf, deskriptor JSON dan aturan ditampilkan. Contoh cleaning tidak berubah; tidak case folding atau koreksi ejaan. Teks siswa dalam output tidak layak dipublikasikan. |
| Token/window | BGE tidak memotong teks pada data ini (0 esai/0 token terpotong). RoBERTa capacity 510, stride 382. Esai contoh memakai dua window, padded input [2,512], panjang window [512,134], mapping [0,0]. Mean CLS yang ditampilkan cocok hitung ulang dari kedua window. |
| Forward awal | S1/S2/S3 memiliki empat output; S3 memperlihatkan RoBERTa [1,768], BGE [1,1024], projection [1,256], concat [1,1024], output [1,4]. Output awal berlabel UNTRAINED; tidak digunakan sebagai label guru. |
| AC2 dan unit test | Lima point estimate fixture AC2 Python/R ditampilkan dan cocok pada presisi backend. Notebook tidak menyimpan stdout runner pytest, sehingga jumlah test Colab tidak dapat dibuktikan dari output tersimpan. Run ulang pada source yang sama di cloud: 61 passed, 0 failed, 0 skipped, 0 xfailed. |
| Training AI | Metadata menyatakan completed_ai_reference_learning, teacher_training_run=false, dan reuse=false pada cell training. Dua epoch per skenario, hanya satu step dengan LR aktif. Tabel menunjukkan sampel parameter yang sesuai berubah serta MSE train turun. Checkpoint/bobot/log lengkap dari runtime Colab tidak diunggah; tidak dapat diperiksa langsung. |
| Evaluasi | 12 baris numerik tersedia, bersumber acuan AI. Walkthrough S3–isi dihitung ulang dari angka presisi penuh yang tersimpan: empat metrik cocok. Semua 12 baris mendekati run checkpoint asli terpisah sebelumnya, selisih maksimum sekitar 0.00001136; kecocokan ini bukan pengganti akses semua prediksi/checkpoint Colab. |
| Reuse | Cell training memakai run baru; cell laporan memakai hasil run yang baru selesai. Dua nilai reuse yang berbeda ini konsisten dengan kode, bukan training diam-diam dua kali. |

## Perhitungan benar tidak berarti kualitas sudah baik

Untuk S3–isi, prediksi kategorikal seluruhnya 3; acuan AI terdiri dari dua 3 dan dua 4. RMSE kontinu sekitar 0.990726; Pearson sekitar −0.995694; QWK 0; AC2 sekitar 0.951807. Semua ini cocok dengan rumus. Point estimate AC2 walkthrough juga dicocokkan ulang dengan R/irrCAC (toleransi 0.00000501).

Bobot agreement kuadratik memberi 1 untuk pasangan sama, dan 15/16 untuk selisih satu kategori. Maka Pa = 0.96875, Pe = 0.3515625, AC2 = 0.951807. Nilai tinggi ini dapat terjadi walaupun semua prediksi berada di satu kategori. QWK nol menunjukkan observed disagreement sama dengan expected disagreement pada marginal tersebut. Pearson negatif menunjukkan urutan prediksi kontinu berlawanan dengan acuan. AC2 tinggi sendirian tidak membuktikan kemampuan membedakan kualitas esai.

Sebagai pemeriksaan kewajaran, ambil rata-rata **acuan AI pada train saja**, lalu gunakan satu angka tetap untuk semua esai validation. Ini hitungan pembanding sederhana, bukan encoder/model keempat, bukan eksperimen tambahan, dan tidak masuk tabel S1/S2/S3 atau pemilihan model.

| Trait | RMSE S3 | RMSE angka tetap dari rata-rata train | S3 lebih kecil? |
|---|---:|---:|---|
| isi | 0.990726 | 0.559017 | Tidak |
| organisasi | 0.609764 | 0.707107 | Ya |
| kebahasaan | 1.042307 | 0.559017 | Tidak |
| mekanik | 1.505894 | 0.750000 | Tidak |

S3 tampak lebih baik daripada S1/S2 pada RMSE run singkat ini, tetapi pada tiga trait bahkan kalah dari angka tetap tersebut. Tidak ada dasar menyatakan fusion unggul secara penelitian.

## Risiko bias dan ketidakpastian

**Acuan:** skor diberikan AI pada transkripsi draf, bukan standar guru terverifikasi. Model dilatih meniru AI dan diuji terhadap AI; jika AI keliru, metrik tidak dapat mendeteksi kekeliruan itu. Mekanik khususnya berisiko mengukur artefak transkripsi. Isi dan kebahasaan kebetulan identik pada seluruh delapan acuan yang masuk latihan; hal ini mempersempit bukti bahwa dua trait dipelajari berbeda, tetapi bukan bukti broadcasting loss.

**Sampel/distribusi:** hanya empat teks validation dan dua kelompok kemiripan usulan. Train memakai kentang/sagu, validation singkong/jagung. Perbedaan topik dapat bercampur dengan pola skor. Organisasi train semuanya bernilai 4, sedangkan validation memiliki 3 dan 4. Korelasi pada empat teks yang berpasangan mirip sangat tidak stabil. Empat baris bukan otomatis empat siswa independen.

**Split/leakage:** kelompok usulan tidak terpecah pada train/validation; pemeriksaan kode menolak kandidat duplikasi lintas kelompok pada threshold yang digunakan. Tidak ditemukan indikasi validation dipakai dalam loss, tuning, early stopping atau memilih checkpoint latihan. Namun identitas/relasi siswa belum terverifikasi dan detektor kemiripan tidak membuktikan semua hubungan ditemukan. Tidak dapat menjamin tidak ada leakage identitas yang belum diketahui.

**Latihan terlalu singkat:** warmup satu dari dua optimizer step membuat step pertama LR nol, lalu satu step aktif. Bobot memang berubah, tetapi head belum belajar cukup; S2 menghasilkan seluruh prediksi kontinu di luar rentang 1–5. Perbedaan hasil antararsitektur dapat sangat dipengaruhi random initialization dan kecepatan belajar. Bukan perbandingan konvergensi yang adil.

**Fairness:** tidak ada metadata demografis terverifikasi atau sampel cukup untuk menguji apakah kelompok siswa tertentu dirugikan. Jangan menyatakan model bebas bias, atau mengklaim bias demografis telah terbukti, dari notebook ini.

## Tindakan dan langkah berikutnya

Tidak mengubah arsitektur, seed, split, jumlah epoch, nilai AI, status sumber atau metrik demi memperbagus hasil. Gunakan angka yang ada untuk belajar membedakan korelasi, agreement, error dan kemampuan membedakan esai. Penelitian tetap memerlukan teks/identitas/group mapping yang ditinjau, skor guru empat trait, rubrik disahkan, serta data cukup untuk protokol proposal yang sah.

Notebook terisi disalin privat sebelum pembersihan. Output/nomor eksekusi pada versi terbaru di repo dibersihkan; kode cell pengguna dipertahankan. Teks/deskriptor/output privat tidak disalin ke laporan publik ini. **Commit lama masih memuat output dalam riwayat GitHub**; pembersihan terbaru tidak menghapusnya. Tidak melakukan force push atau penulisan ulang riwayat. Jika data tersebut perlu ditarik dari akses publik, pemilik repo perlu menangani akses repository/riwayat lama secara terpisah.

Bukti mesin yang aman tersedia di `docs/verifikasi/saved_colab_review_20261014.json`. Bukti lengkap dengan output/perhitungan per esai tetap di direktori outputs lokal dan tidak dipush.
