# Menghubungkan model, test, dan Git

Lo ingin memahami alur satu esai, cara membaca test, dan kenapa kode tidak terlihat di GitHub. Ketiganya saling terhubung: kita membangun **alur**, menguji **aturan alur**, lalu menyimpan **versi alur yang diperiksa**.

## Ikuti satu esai dulu

Esai dibaca dari CSV. `text_raw` tetap disimpan; cleaning hanya merapikan Unicode, akhir baris, dan artefak kontrol yang jelas. Kesalahan ejaan atau kapital tidak diperbaiki otomatis karena justru bisa memengaruhi trait mekanik. Dengan cara ini, model menerima ciri tulisan siswa yang hendak dinilai.

**Tokenisasi** mengubah teks menjadi nomor kamus model (**token ID**). Nomor itu bukan skor tulisan. Encoder kemudian mengubah token menjadi **embedding**, yaitu vektor yang menyimpan representasi bahasa. BGE menghasilkan 1024 fitur, sedangkan RoBERTa menghasilkan 768 fitur per window. Satu esai panjang bisa memiliki beberapa window; `chunk_to_essay` memasangkan setiap window kembali ke esai asal.

Pada S3, 1024 fitur BGE diproyeksikan menjadi 256 fitur dan digabung dengan 768 fitur RoBERTa. Hasilnya kembali 1024 fitur per esai. Head linear memetakan representasi ini menjadi empat angka dengan urutan isi, organisasi, kebahasaan, mekanik. `[1,4]` berarti **satu esai dan empat output**, bukan empat esai atau empat epoch.

BGE **frozen**, artinya bobotnya tidak diperbarui. RoBERTa **fine-tuned**, artinya bobotnya akan menyesuaikan melalui training. Mean window RoBERTa tetap terhubung ke **gradien**, yaitu arah perubahan bobot untuk mengurangi loss. **Loss** adalah ukuran selisih prediksi terhadap skor guru saat belajar.

Dengan label kosong, kita belum memiliki jawaban guru yang harus didekati model. Forward tetap bisa menghasilkan empat angka, sama seperti kalkulator yang bisa mengeluarkan angka setelah diberi input; itu belum membuktikan angka tersebut sesuai kualitas tulisan. Target 3 yang dipakai smoke backward hanya fixture untuk melihat apakah bobot bisa berubah. Target itu tidak disimpan sebagai skor penelitian.

## Baca satu test sebagai janji kode

`test_loss_rejects_broadcast` memberikan output `[2,4]` dan label `[2,1]`. Kalau kode membiarkan broadcasting, satu skor guru bisa diam-diam disalin ke semua trait. Test harus menolaknya; bentuk empat kolom mencegah isi, organisasi, kebahasaan dan mekanik menerima satu target yang sama tanpa sengaja.

`test_metrics` menggunakan angka kecil yang jawabannya bisa dihitung. Contoh midpoint 2.5 dipetakan ke kategori 2. RMSE tetap dihitung dari prediksi kontinu sebelum clipping; QWK/AC2 dari salinan kategori. Ini menguji definisi perhitungan, bukan akurasi model pada siswa. Fixture tidak menggantikan dataset penelitian.

`test_forward_backward_reload` memakai encoder tiruan: embedding kecil yang mengikuti antarmuka encoder nyata. Test ini berguna untuk memastikan output `[B,4]`, agregasi window, gradien dan save/reload. Ia tidak membuktikan checkpoint BGE/RoBERTa asli berhasil dimuat. Bukti itu berasal dari `scripts/smoke_real.py`, yang dijalankan terpisah menggunakan checkpoint asli dan teks CSV lokal.

Collection pytest menghitung **kasus test**. Satu fungsi parametrized bisa dijalankan enam kali, misalnya tiga skenario dikalikan batch1/batch2. Jadi "26 passed" adalah 26 kasus teknis yang selesai, bukan 26 esai dinilai, 26 epoch, atau 26 model yang tervalidasi. **Skipped** berarti tidak dijalankan, misalnya backend R belum terpasang; itu harus terlihat dalam laporan.

Ada pula test jadwal eksperimen dengan fungsi training tiruan. Ia mengecek test baru dipanggil setelah konfigurasi/refit semua skenario terkunci. Fungsi tiruan itu tidak menjalankan 25 training penelitian. Kita perlu test aturan jadwal sekaligus evaluasi penelitian yang sebenarnya nanti; keduanya memiliki tujuan berbeda.

## Pahami kenapa test lulus tetapi GitHub kosong

Kode di folder kerja bisa dijalankan walaupun Git belum mencatatnya. **Untracked** berarti file itu belum masuk rekaman Git. **Staging** memilih file untuk commit; kita memilih source yang aman dan meninggalkan data siswa lokal. **Commit** menyimpan pilihan itu sebagai versi dengan identitas SHA. **Push** baru mengirim commit ke GitHub.

Pada audit awal, checkout benar-benar mengarah ke repository tujuan, tetapi branch `work` masih **unborn**: nama jalur sudah ada, rekaman pertama belum ada. `git log` karena itu mengembalikan "does not have any commits yet". Ini keadaan Git yang sah dan menjelaskan mengapa tidak ada riwayat untuk dilihat pengguna. `git ls-remote` juga berhasil tetapi mengembalikan nol refs, sehingga snapshot remote kosong diverifikasi ulang.

Sesudah publikasi, kita cocokkan SHA commit lokal dengan SHA branch remote dan beberapa blob file. **Blob** adalah isi file yang disimpan Git. Kalau SHA branch sama tetapi file yang diharapkan belum diperiksa, masih ada celah salah lokasi/salah pilihan file; verifikasi isi membantu menutupnya. Git lokal saja tidak cukup sebagai bukti GitHub berubah.

Repo kosong tidak punya `main` sebagai base PR biasa. Branch hasil dapat menjadi commit awal tanpa merge ke main. Menyiapkan branch hasil bukan alasan membuat base palsu atau mengklaim PR yang belum ada.

## Urutan belajar yang bisa lo ulang

Buka notebook dan ikuti satu esai asli **secara lokal**: teks → token IDs → window → embedding → empat angka UNTRAINED. Lalu baca test mapping/loss untuk melihat janji bentuk dan urutan angka tersebut. Terakhir lihat commit/branch GitHub yang menyimpan kode serta laporan test. Ketika skor guru dan verifikasi tersedia, baru tambahkan tahap belajar parameter dan mengukur kualitas prediksi pada data yang dipisahkan dengan benar.

Pertanyaan yang bisa kita bedah bersama: mengapa label `[1,1]` harus ditolak ketika prediksi `[1,4]`, dan bagian test mana yang membuktikannya?
