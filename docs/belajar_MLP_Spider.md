# Buku belajar hasil MLP

Unduh [belajar_MLP_Spider.html](belajar_MLP_Spider.html) dengan **Download raw file**, lalu buka di Chrome, Edge, atau Firefox. Halaman berjalan offline tanpa instalasi, CDN, atau training. Tampilan memakai tema Spider-Man full colour dari buku paper: splash warna, kota komik, pose di tepi halaman dan sticky note yang menimpa border.

Source mengikuti [commit 15bc7bb](https://github.com/MalikHariyanto/Automated_Essay_Scoring/commit/15bc7bbb2aee91ecaf44fb91a0139a8c21f64a20). Seluruh 30 cell kode unggahan `AES_300_S3_MLP (1).ipynb` sama dengan source pada commit itu. Buku tidak mengikuti eksperimen double-frozen yang dibuat kemudian, dan tidak mengubah notebook utama.

## Menu perbandingan eksperimen

Klik **Bandingkan eksperimen** pada beranda atau pemilih ruang di atas buku. Menu ini memiliki lima pembahasan: **Peta perubahan**, **Layer & kode**, **Training & CV**, **Prediksi & metrik**, serta **Kenapa lanjut?**. Pilih S1/S2/S3 dan versi awal pembanding; run MLP terakhir tetap di sebelah kanan. Area pembahasan mempunyai scroll sendiri.

Jejak eksperimen dipisahkan berdasarkan bukti:

| Versi | Source | Perubahan penting | Bukti hasil |
| --- | --- | --- | --- |
| E0 · awal | `3afad79` | Lima fold, batas 5 epoch, tanpa early stopping; head S3 linear | Notebook tersimpan: 15 hasil fold, riwayat CV/refit dan metrik test |
| E1 · revisi training | `312f394` | Batas 25 epoch, patience 5, LR lebih kecil, LayerNorm tambahan RoBERTa S3; head S3 masih linear | Source terverifikasi; hasil akhir lengkap belum terpasangkan secara pasti |
| E2 · MLP | `15bc7bb` | Head S3 MLP 1024→256→1, konfigurasi training revisi dipertahankan | Notebook dan lima Excel tersimpan |

Kedua versi hasil memakai **lima fold**, bukan 5 versus 25 fold. Yang berubah adalah batas epoch per fold. S2 sudah memakai MLP sejak awal; S1 tetap head linear. Versi awal juga sudah memilih bobot CV menurut RMSE validation terbaik meskipun belum memakai early stopping.

Klik layer untuk sticky note fungsi lapisan, lalu lihat source asli berdampingan dengan sorotan baris berubah. Grafik training dapat dipilih per skenario, fold, dan satuan RMSE/MSE. Semua epoch yang ditampilkan berasal dari riwayat tersimpan, bukan angka simulasi. Tabel test menampilkan empat metrik semua skenario; tabel CV memperlihatkan keempat metrik tiap fold.

Contoh hasil S3: RMSE test **11,7224 → 6,7085** dan rata-rata RMSE CV **10,9542 → 6,0516**. Perubahan awal → MLP mencakup beberapa pengaturan sekaligus, sehingga hasil tersebut tidak membuktikan MLP sebagai penyebab tunggal. Dataset, 240 development/60 test, anggota kelima fold, serta nilai guru sekarang sudah dicocokkan persis dari kedua workbook. Test lama juga sudah dilihat sebelum pengembangan selanjutnya, sehingga perbandingan lanjutan dijelaskan sebagai eksplorasi.

Prediksi individual tetap lokal. HTML pribadi terbaru menyertakan notebook awal juga: tiga prediksi contoh lama tampil terpisah dari nilai guru/prediksi/pembulatan contoh MLP. Teks contoh juga sudah dicocokkan. Tabel contoh lama tetap ditampilkan sesuai sumbernya; tabel lengkap 60 pasangan awal/MLP dengan nilai guru tersedia pada menu Tabel hasil & arti angka setelah workbook pribadi dimuat.

Salinan publik sudah memuat angka agregat perbandingan. Bila memiliki notebook awal yang persis cocok, tombol di tab Prediksi & metrik dapat membukanya lokal. **Simpan HTML lengkap** ikut mempertahankan notebook awal yang telah dimuat.

## Tabel hasil & arti angka

Tombol **Tabel hasil & arti angka** membuka meja hasil tersendiri dengan scroll. Enam baris metrik test S1/S2/S3 untuk kedua eksperimen tampil langsung, diikuti pilihan penjelasan RMSE, MSE training, Pearson, QWK, dan Gwet AC2. Ada tabel enam loss refit terakhir, seluruh 15 fold, serta rata-rata RMSE CV dan simpangan baku.

Penjelasan mengikuti skenario yang dipilih. RMSE/MSE tidak diberi batas bagus/jelek universal. RMSE dibandingkan dengan prediktor sederhana yang selalu memakai rata-rata nilai guru development: RMSE test **8,1428 poin**. Rata-rata itu dihitung tanpa memakai label test untuk fitting. Contoh pilihan batas 5/10 poin diberi label ilustrasi, bukan standar AES. MSE memakai target latihan 0–1 tetapi tidak dibatasi maksimum 1; loss training dibedakan dari error test. Pearson/QWK/AC2 dijelaskan melalui titik acuan −1/0/1 yang relevan, serta contoh aktual. AC2 tinggi tidak disamakan dengan persen prediksi tepat.

ZIP `Hasil_AES_Ringkas (1).zip` yang dikirim kemudian terverifikasi sebagai hasil **awal lima epoch**, bukan double-frozen: pengaturannya lima epoch, head S3 linear, serta metriknya cocok sumber awal. File ini melengkapi pembuktian bahwa data, train/test dan anggota semua fold sama dengan run MLP. Metrik awal dan riwayat sekarang memakai nilai numerik penuh dari Excel.

HTML pribadi memuat workbook evaluasi awal asli. Tabel **60 pasangan prediksi awal/MLP** dicocokkan melalui ID dan nilai guru, bukan posisi baris. Ada prediksi mentah kedua versi, pembulatan MLP, serta error absolut per jawaban. Salinan GitHub hanya menyimpan agregat; pilih ZIP awal pada meja hasil dan ZIP MLP pada tombol di atas untuk membuka nilai individual lokal. Tombol Simpan HTML lengkap ikut menyimpan workbook awal, sehingga tidak perlu memilihnya lagi ketika dibuka ulang. Hasil double-frozen belum ditambahkan.

## Cara membaca

Ada 16 bab. Mulai dari hasil, lalu ikuti setup → dataset → split → token → vektor → arsitektur → training → CV/refit/test → prediksi → lima laporan Excel.

Setiap bab mempunyai empat tab:

- **Alur & cara baca:** proses input sampai output, istilah kontekstual, diagram lapisan yang dapat diklik, serta catatan asli notebook.
- **Kode + sticky note:** seluruh kode, nomor baris, syntax highlight, pencarian dan salin kode. Klik baris bertanda atau daftar catatan untuk membaca fungsi baris tersebut. Ada 120 catatan, termasuk 52 titik penting seperti nama checkpoint, pemuat model, freeze/fine-tuning, LayerNorm, concat dan head MLP.
- **Output tersimpan:** output asli notebook, tanpa menjalankan ulang kode. Log dan tabel mempunyai scroll sendiri; catatan menjelaskan satuan angka serta kaitannya dengan proses. Fungsi yang baru didefinisikan dibedakan dari tahap yang benar-benar sudah dieksekusi.
- **Jelajah Excel:** pilih salah satu dari lima workbook dan 41 sheet. Tabel menampilkan 25 baris serta delapan kolom komponen per halaman. Semua baris/komponen tetap tersedia. Klik sel untuk nilai lengkap; XLSX asli dapat diunduh tanpa perubahan.

Daftar bab, kode, catatan, output, dan tabel mempunyai area scroll agar halaman utama tidak memanjang mengikuti seluruh hasil. Ada navigasi keyboard, dialog istilah, tampilan HP, dan tombol jeda animasi.

### Kode dan catatan dalam satu layar

Di tab **Kode + sticky note**, klik **⛶ Layar penuh**. Kode dan catatan memenuhi layar browser, tetap berdampingan di desktop maupun HP. Pilihan blok kode, pencarian, salin kode dan kamus istilah tetap bisa dipakai. Keluar melalui **Kembali ke buku** atau **Esc**; blok dan catatan yang sedang dibaca tetap terbuka.

Klik nomor baris bertanda atau judul catatan: sticky note disejajarkan dengan baris yang disorot. Saat kode digulir, catatan mengikuti posisi baris itu. Jika baris keluar dari area baca, catatan beralih ke penanda penting yang terlihat di dekat bagian atas kode. Gerak kertas dibatasi tepi panel agar penjelasannya tetap terbaca. Daftar catatan di kanan juga dapat digulir untuk memilih baris lain. Perilaku ini berlaku pada tampilan biasa saat kedua panel berdampingan; di HP, gunakan layar penuh untuk membacanya berdampingan.

[Preview layar penuh](previews/mlp_study/10_kode_layar_penuh.png) · [Preview HP](previews/mlp_study/11_kode_layar_penuh_mobile.png).

## Memuat seluruh output pribadi

Salinan GitHub memuat kode lengkap, hasil agregat aktual, riwayat epoch, grafik validasi, metadata seluruh sheet, dan matriks guru/prediksi yang dikelompokkan per rentang sepuluh poin. Matriks itu membantu membaca hasil; metrik QWK/AC2 aslinya tetap memakai kategori integer 0–100.

Teks siswa, ID, skor individual dan seluruh vektor tidak dipublikasikan. Untuk menampilkan semuanya:

1. Buka HTML yang sudah diunduh.
2. Pilih **Notebook .ipynb** dan buka `AES_300_S3_MLP (1).ipynb` yang diunggah dalam percakapan ini.
3. Pilih **Hasil_AES_Ringkas.zip** dan buka `Hasil_AES_Ringkas (2).zip` dari run yang sama.
4. Seluruh 34 objek output tersimpan dan 41 sheet akan tersedia. Pembaca memeriksa kesesuaian source notebook dan SHA-256 ZIP; hasil run lain tidak dicampurkan.
5. Klik **Simpan HTML lengkap** untuk membuat satu HTML pribadi dengan kedua file sudah tertanam. Buka kembali HTML itu tanpa memilih file lagi.

File diproses di browser, tidak dikirim ke server atau disimpan ke localStorage. Script/widget dari output notebook tidak dijalankan. Simpan HTML lengkap sebagai berkas pribadi karena di dalamnya ada data siswa.

Versi lengkap siap pakai juga dibuat secara lokal di `outputs/mlp_study/belajar_MLP_Spider_Privat.html`; berkas ini tidak masuk GitHub.

## Hasil dan batas penjelasan

Run ini memakai satu skor guru 0–100 untuk 300 jawaban, bukan empat target rubrik. Development berjumlah 240 dan test 60. Validasi dilakukan dengan lima fold di development.

| Skenario | Arsitektur pada run ini | RMSE test, poin |
| --- | --- | ---: |
| S1 | RoBERTa fine-tuned → dropout → linear 768→1 | 6.3805 |
| S2 | BGE-M3 frozen → MLP 1024→256→1 | 6.9481 |
| S3 | RoBERTa fine-tuned → LayerNorm 768; BGE frozen → proyeksi 256; concat 1024 → MLP 256→1 | 6.7085 |

S3 memiliki rata-rata RMSE CV terkecil pada run tersimpan, sedangkan S1 terbaik di test. Hasil ini tidak membuktikan satu arsitektur selalu unggul. AC2 mendekati satu bukan persentase prediksi tepat. Loss MSE memakai target 0–1; RMSE memakai poin skor asli. Pembahasan dan grafik memakai hasil tersimpan, bukan angka hasil training baru.

Vektor kata RoBERTa S3 berasal dari cabang RoBERTa sesudah latihan. Fusion berlangsung pada tingkat jawaban; buku tidak mengarang vektor fusion per kata. Simulator window diberi label ilustrasi dan tidak dianggap tokenisasi aktual.

## Membangun ulang

Perintah berikut hanya membaca source Git dan snapshot agregat aman:

```bash
python scripts/build_mlp_study.py
```

Untuk membuat ulang snapshot dan HTML pribadi dengan file asli, gunakan Python yang mempunyai `openpyxl`:

```bash
python scripts/build_mlp_study.py \
  --notebook '/path/AES_300_S3_MLP (1).ipynb' \
  --results '/path/Hasil_AES_Ringkas (2).zip' \
  --baseline '/path/notebook_awal_5_epoch.ipynb'
```

Argumen `--baseline` bersifat opsional untuk bundel pribadi. Tambahkan `--baseline-evaluation outputs/mlp_study/baseline_eval.xlsx` untuk menanamkan workbook evaluasi awal yang sudah diverifikasi. Ekstraksi ulang snapshot pembanding dilakukan dengan `python scripts/build_mlp_comparison.py --baseline '/path/notebook_awal_5_epoch.ipynb' --baseline-results '/path/hasil_awal.zip' --mlp-results '/path/hasil_mlp.zip'`; source awal wajib cocok commit `3afad79`. Sertakan `--baseline-results` dan `--mlp-results` dengan kedua ZIP asli untuk memverifikasi seluruh manifest dan menyimpan evaluasi awal privat. Snapshot publik hanya menyimpan source, riwayat agregat, metrik, checksum dan bukti pencocokan berupa hitungan/status; bukan jawaban atau nilai individual.

Builder membaca source dari commit tetap, bukan dari notebook eksperimen terbaru. Template, isi catatan, dan JSZip berlisensi MIT/GPL disertakan di repository. Tidak ada bobot encoder yang diunduh atau dilatih oleh builder. Bukti pemeriksaan ada di [laporan verifikasi](verification_mlp_study.md).
