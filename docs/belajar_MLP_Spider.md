# Buku belajar hasil MLP

Unduh [belajar_MLP_Spider.html](belajar_MLP_Spider.html) dengan **Download raw file**, lalu buka di Chrome, Edge, atau Firefox. Halaman berjalan offline tanpa instalasi, CDN, atau training. Tampilan memakai tema Spider-Man full colour dari buku paper: splash warna, kota komik, pose di tepi halaman dan sticky note yang menimpa border.

Source mengikuti [commit 15bc7bb](https://github.com/MalikHariyanto/Automated_Essay_Scoring/commit/15bc7bbb2aee91ecaf44fb91a0139a8c21f64a20). Seluruh 30 cell kode unggahan `AES_300_S3_MLP (1).ipynb` sama dengan source pada commit itu. Buku MLP tetap mengikuti run acuan tersebut. Hasil double frozen kini tersedia sebagai pembanding E3 tersendiri; notebook utama tidak diubah.

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

Tombol **Tabel hasil & arti angka** membuka meja hasil tersendiri dengan scroll. Versi awal dan MLP menjadi kolom berdampingan; S1/S2/S3 dan keempat metrik menjadi 12 baris, diikuti pilihan penjelasan RMSE, MSE training, Pearson, QWK, dan Gwet AC2. Tabel loss membandingkan epoch dan MSE refit per skenario secara mendatar. Seluruh 15 fold serta rata-rata RMSE CV dan simpangan baku tetap tersedia.

Penjelasan mengikuti skenario yang dipilih. RMSE/MSE tidak diberi batas bagus/jelek universal. RMSE dibandingkan dengan prediktor sederhana yang selalu memakai rata-rata nilai guru development: RMSE test **8,1428 poin**. Rata-rata itu dihitung tanpa memakai label test untuk fitting. Contoh pilihan batas 5/10 poin diberi label ilustrasi, bukan standar AES. MSE memakai target latihan 0–1 tetapi tidak dibatasi maksimum 1; loss training dibedakan dari error test. Pearson/QWK/AC2 dijelaskan melalui titik acuan −1/0/1 yang relevan, serta contoh aktual. AC2 tinggi tidak disamakan dengan persen prediksi tepat.

ZIP `Hasil_AES_Ringkas (1).zip` yang dikirim kemudian terverifikasi sebagai hasil **awal lima epoch**, bukan double-frozen: pengaturannya lima epoch, head S3 linear, serta metriknya cocok sumber awal. File ini melengkapi pembuktian bahwa data, train/test dan anggota semua fold sama dengan run MLP. Metrik awal dan riwayat sekarang memakai nilai numerik penuh dari Excel.

HTML pribadi memuat workbook evaluasi awal asli. Tabel **60 pasangan prediksi awal/MLP** dicocokkan melalui ID dan nilai guru, bukan posisi baris. Ada prediksi mentah kedua versi, pembulatan MLP, serta error absolut per jawaban. Salinan GitHub hanya menyimpan agregat; pilih ZIP awal pada meja hasil dan ZIP MLP pada tombol di atas untuk membuka nilai individual lokal. Tombol Simpan HTML lengkap ikut menyimpan workbook awal, sehingga tidak perlu memilihnya lagi ketika dibuka ulang. Hasil double frozen yang diunggah kemudian tersedia pada menu **MLP vs double frozen**.

## Pembanding baru: double frozen + MLP

Tombol **MLP vs double frozen** di pemilih ruang atau **E3** di daftar eksperimen membuka empat tab: **Hasil & kesimpulan**, **Proses, kode & latihan**, **Guru & prediksi**, serta **Excel run baru**. Tabel test menempatkan tiga eksperimen sebagai kolom: **Awal—5 epoch**, **MLP acuan—maksimal 25 epoch**, dan **Double frozen + MLP**. Tiap kelompok S1/S2/S3 memiliki empat baris RMSE, Pearson, QWK dan Gwet AC2; baca satu baris mendatar untuk membandingkan hasil. Ada perbandingan kelima fold, mean/SD, loss refit dan grafik seluruh epoch MLP versus double frozen. Tabel loss juga memakai eksperimen sebagai kolom. S1 tetap fine-tuned; istilah double frozen berlaku pada kedua encoder S3.

Header menjelaskan head S3 dan aturan berhenti: lima epoch tanpa perbaikan RMSE validasi, bukan lima perbaikan. S1 tetap memakai head linear dan S2 memakai MLP pada ketiga eksperimen. Meja **Tabel hasil & arti angka** memakai susunan kolom yang sama untuk dua eksperimen awal/MLP. Kartu E2 yang sebelumnya hanya berupa penanda tetap dan tidak dapat diklik telah dihapus dari sidebar; hasil MLP tetap menjadi pembanding.

[Lihat preview tabel dengan eksperimen sebagai kolom](previews/mlp_study/16_eksperimen_sebagai_kolom.png). Di HP, geser tabel ke samping; label skenario/metrik tetap terlihat.

Hasil ZIP baru menunjukkan RMSE test S1 **6,8829**, S2 **6,8435**, dan S3 **7,8611**. S3 MLP acuan **6,7085**, sehingga pembekuan belum memperbaiki S3 pada run ini. Dataset, pembagian 240/60, anggota semua fold serta nilai guru test telah dicocokkan. S1/S2 juga bergeser meskipun arsitektur tetap; perbandingan belum mengisolasi satu penyebab. [Review dan bukti lengkap](review_double_frozen_results.md).

Semua 300 × 768 komponen RoBERTa mentah S3 baru sama persis dengan pretrained. LayerNorm tambahan, proyeksi dan MLP tetap dilatih. Kode pembanding di tab proses adalah rujukan notebook manual dari repo; notebook eksekusi double frozen belum diunggah, sehingga kecocokan source runtime byte demi byte tidak diklaim.

Berkas `Hasil_AES_Ringkas (3).zip` dibuka lokal melalui pemilih ZIP double frozen. Seluruh 41 sheet tersedia dengan paging baris dan delapan komponen; komponen terakhir tetap dapat dijangkau. Tabel 60 pasangan guru/prediksi/pembulatan/error untuk MLP dan frozen dicocokkan melalui ID. HTML pribadi terbaru memuat ZIP ini otomatis; tombol **Simpan HTML lengkap** mempertahankannya saat diekspor dan dibuka lagi. Teks, ID, skor individual dan vektor tetap berada pada file pribadi.

## Pembanding E5 · LoRA + MLP

Buka **E5 · LoRA + MLP** pada pemilih ruang atau tombol E5 di daftar eksperimen. Tabel sekarang menempatkan awal, MLP acuan, double frozen dan E5 sebagai kolom. Empat tab E5 menampilkan metode/layer, empat metrik test, kelima fold dan kurva training, pasangan nilai guru/prediksi/pembulatan, serta seluruh sheet dan unduhan Excel asli. Buku pribadi memuat ZIP E5 otomatis; Simpan HTML lengkap juga mempertahankannya.

S3 E5 mempunyai RMSE test **6,5305**, lebih kecil daripada acuan **6,7085**, tetapi rata-rata RMSE CV **6,2509**, lebih besar daripada acuan **6,0516**. S1 tetap RMSE test terkecil. Sheet ringkasan metrik tidak ada dalam ZIP E5; HTML menjelaskan bahwa metrik test dihitung ulang dari 60 pasangan prediksi, sedangkan CV diambil dari epoch terpilih pada log. [Audit dan penjelasan lengkap](review_eksperimen5_lora.md).

## Layar penuh untuk perbandingan

Klik **⛶ Layar penuh** di ruang Bandingkan eksperimen, Tabel hasil & arti angka, MLP vs double frozen, atau E5 · LoRA + MLP. Header dekoratif, sidebar dan footer besar disembunyikan. Kontrol kecil di atas mempertahankan pilihan S1/S2/S3, tab pembahasan serta tombol **Kembali**; isi memakai sisa layar dengan satu area gulir vertikal. Tabel lebar tetap bisa digeser mendatar.

Tekan **Esc** atau **Kembali** untuk keluar. Posisi halaman dan fokus dikembalikan; data, pilihan tab dan tabel tidak dibuat ulang. Browser yang menolak native fullscreen tetap mendapat modal seluas viewport. Fullscreen kode tetap terpisah, termasuk mode semua blok. [Preview](previews/mlp_study/19_perbandingan_layar_penuh.png) · [Bukti pengujian](verification_mlp_compare_fullscreen.md).

## Cara membaca

Ada 16 bab. Mulai dari hasil, lalu ikuti setup → dataset → split → token → vektor → arsitektur → training → CV/refit/test → prediksi → lima laporan Excel.

Setiap bab mempunyai empat tab:

- **Alur & cara baca:** proses input sampai output, istilah kontekstual, diagram lapisan yang dapat diklik, serta catatan asli notebook.
- **Kode + sticky note:** seluruh kode, nomor baris, syntax highlight, pencarian dan salin kode. Klik baris bertanda atau daftar catatan untuk membaca fungsi baris tersebut. Ada 120 catatan, termasuk 52 titik penting seperti nama checkpoint, pemuat model, freeze/fine-tuning, LayerNorm, concat dan head MLP.
- **Output tersimpan:** seluruh output asli notebook, tanpa menjalankan ulang kode. Setiap kartu menyebut nama blok dan memiliki tombol menuju kode terkait. Log panjang dibuka sesuai kebutuhan. Di bawahnya ada pratinjau hasil Excel tahap tersebut, dengan label sumber yang berbeda dari output Colab. Fungsi yang baru didefinisikan dibedakan dari tahap yang benar-benar sudah dieksekusi.
- **Hasil Excel:** lima pintasan membuka prediksi/evaluasi, data/tokenisasi, vektor jawaban, vektor kata/kalimat dan riwayat training. Semua 41 sheet tersedia pada HTML pribadi. Pilih 25/50/100 baris; tabel prediksi menampilkan seluruh 11 kolom. Pengatur delapan komponen hanya muncul pada tabel vektor. Pencarian memeriksa seluruh kolom tabel biasa atau kolom identitas/kata pada vektor. Angka layar memakai maksimal empat desimal; klik atau Enter pada sel untuk nilai asli. Excel dan ZIP asli dapat diunduh tanpa perubahan.

Penjelasan **Cara membaca tabel ini** dapat dibuka di bawah pilihan laporan/tabel. Tombol **Lihat kode tahap** dan **Output Colab** menghubungkan hasil ke proses penghasilnya. Tombol CSV telah dihapus. Navigasi baris disembunyikan bila tabel cukup satu halaman; unduhan file serta pengatur komponen tidak ditampilkan ketika datanya tidak tersedia atau kontrolnya tidak relevan. Tab yang aktif tetap terlihat saat strip tab digeser pada HP.

HTML pribadi siap pakai memuat notebook MLP asli, ZIP MLP, notebook/workbook awal dan ZIP double frozen yang sudah dikirim. Tidak perlu memilih ulang file ketika membukanya. Versi publik hanya menawarkan tabel agregat dan penjelasan yang benar-benar tersedia; sembilan tabel panduan/pengaturan/arti metrik ditambahkan dari Excel asli, tanpa tabel siswa. Hasil konteks pair yang belum lengkap tidak dicampurkan ke run MLP acuan ini.

Daftar bab, kode, catatan, output, dan tabel mempunyai area scroll agar halaman utama tidak memanjang mengikuti seluruh hasil. Ada navigasi keyboard, dialog istilah, tampilan HP, dan tombol jeda animasi.

Scroll mengikuti area di bawah kursor. Saat tabel mencapai ujung, guliran diteruskan ke panel buku, lalu ke halaman; pola yang sama berlaku untuk kode biasa, daftar bab, hasil dan perbandingan. Swipe HP juga dapat melewati batas panel. Pada kode layar penuh, guliran tetap berada di kanvas bersama dan halaman belakang tetap diam. Keluar dari layar penuh mengembalikan scroll halaman. Perbaikan ini mempertahankan tampilan, kode, catatan dan hasil; [bukti pemeriksaan scroll](verification_mlp_scroll.md).

### Kode dan catatan dalam satu layar

Di tab **Kode + sticky note**, klik **⛶ Layar penuh**. Browser masuk ke fullscreen bila mendukungnya; bila tidak, kanvas tetap memenuhi area halaman. Tidak ada header besar atau panel catatan terpisah. Kode dan sticky note menempati satu latar dan **satu area scroll**. Baris panjang dibungkus secara visual agar tidak memerlukan scrollbar mendatar; kode yang disalin tetap persis seperti sumber.

Catatan menempel di samping baris yang disorot dan ikut bergerak bersama kode ketika digulir, termasuk saat roda mouse berada di atas kertas catatan. Penjelasan panjang dapat dibaca sampai akhir dengan scroll yang sama. Setelah catatan keluar dari area baca, penanda penting yang sedang terlihat menjadi catatan aktif berikutnya.

Kontrol kecil mengambang di pojok: tombol **Semua blok / Per blok** mengganti cara membaca, **☰ Kode** membuka pilihan blok, catatan baris, pencarian dan salin kode; **×** atau **Esc** mengembalikan buku. Pilihan kode disembunyikan saat tidak diperlukan. Blok dan posisi baca dipertahankan dengan mengikuti nomor baris ketika ukuran baris berubah. Pada tampilan buku biasa, susunan kode dan catatan sebelumnya tetap tersedia.

**Semua blok:** seluruh 30 blok dan 2.488 baris disusun sesuai urutan notebook pada satu scroll. Setiap blok mempunyai pembatas bernama, seperti Environment & folder kerja, Tokenisasi aktual, Arsitektur S1–S3, atau Unduh hasil. Nomor baris tetap lokal per blok. Sticky note mengenali blok dan baris sekaligus; pencarian menjangkau seluruh kode. Menu **Kode** dapat melompat ke blok/catatan mana pun. **Salin blok aktif** tetap menyalin source blok asli, bukan menggabungkan cell Python dan `%%aes` menjadi program baru. Tombol **Per blok** mengembalikan tampilan satu blok.

[Preview layar penuh](previews/mlp_study/10_kode_layar_penuh.png) · [Preview HP](previews/mlp_study/11_kode_layar_penuh_mobile.png).

[Preview semua blok](previews/mlp_study/13_semua_blok_layar_penuh.png) · [Semua blok di HP](previews/mlp_study/15_semua_blok_mobile.png) · [Pembanding double frozen](previews/mlp_study/12_double_frozen_review.png).

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
  --baseline '/path/notebook_awal_5_epoch.ipynb' \
  --frozen-results '/path/Hasil_AES_Ringkas (3).zip'
```

Argumen `--baseline` bersifat opsional untuk bundel pribadi. Tambahkan `--baseline-evaluation outputs/mlp_study/baseline_eval.xlsx` untuk menanamkan workbook evaluasi awal yang sudah diverifikasi. Ekstraksi ulang snapshot pembanding dilakukan dengan `python scripts/build_mlp_comparison.py --baseline '/path/notebook_awal_5_epoch.ipynb' --baseline-results '/path/hasil_awal.zip' --mlp-results '/path/hasil_mlp.zip'`; source awal wajib cocok commit `3afad79`. Sertakan `--baseline-results` dan `--mlp-results` dengan kedua ZIP asli untuk memverifikasi seluruh manifest dan menyimpan evaluasi awal privat. Snapshot publik hanya menyimpan source, riwayat agregat, metrik, checksum dan bukti pencocokan berupa hitungan/status; bukan jawaban atau nilai individual.

Builder membaca source dari commit tetap, bukan dari notebook eksperimen terbaru. Template, isi catatan, dan JSZip berlisensi MIT/GPL disertakan di repository. Tidak ada bobot encoder yang diunduh atau dilatih oleh builder. Bukti pemeriksaan ada di [laporan verifikasi](verification_mlp_study.md).

Snapshot double frozen dibangun dengan `python scripts/build_mlp_frozen.py --frozen '/path/hasil_frozen.zip' --mlp '/path/hasil_MLP_acuan.zip'`. Builder memeriksa identitas dataset/split, empat metrik terhadap 60 pasangan (toleransi absolut 1e-6), epoch CV terpilih, serta kesamaan vektor S3/pretrained. Snapshot publik hanya menyimpan agregat, metadata dan kode rujukan.
