# Verifikasi buku hasil MLP

## Kanvas tanpa header dengan satu scroll — 22 Oktober 2026

Revisi dari HEAD `8324b1f140ee0ca18d363973e1e1c2f62f2ae95e`, sesuai koreksi pengguna terhadap versi layar penuh pertama. Header, toolbar permanen, pembatas kolom dan scroll terpisah dihapus dari mode layar penuh. `codeLines` dan `notePane` sekarang anak dari satu `codeCanvas` di dalam `codeScroll`. Latar catatan transparan tanpa border; kertas kuning ditempel pada koordinat baris sumber di kanvas. Kode dan catatan bergerak bersama, termasuk scroll di atas catatan. Baris panjang dibungkus secara visual; tombol salin tetap mengambil source asli.

Kontrol hanya tombol × dan menu kecil **Kode** yang mengambang. Menu berisi pemilih blok, pemilih catatan, pencarian dan salin kode. Fullscreen memakai Fullscreen API browser dari klik pengguna; bila API tidak tersedia atau ditolak, kanvas tetap memenuhi viewport. Keluar dari fullscreen browser juga menutup kanvas dan kamus yang masih terbuka. Nomor baris menjadi acuan pemulihan posisi karena tinggi baris berbeda antara kode terbungkus dan tampilan biasa.

`node scripts/check_mlp_fullscreen.mjs http://127.0.0.1:8765/docs/belajar_MLP_Spider.html http://127.0.0.1:8765/outputs/mlp_study/belajar_MLP_Spider_Privat.html` dari cwd `/workspace/Automated_Essay_Scoring`: **11 kelompok lolos, 0 gagal, 0 dilewati**, exit 0. Versi Chromium `154.0.8037.57`, Playwright `1.62.1`, Node `24.19.0`. Pemeriksaan menguji satu-satunya elemen scroll vertikal adalah `codeScroll`, tidak ada overflow mendatar, tidak ada header, latar catatan transparan dan border nol, serta kedua elemen berada dalam kanvas yang sama. Enam ukuran viewport mencakup HP 320×568 dan landscape. Scroll melalui kertas menggeser kode dan catatan dengan jarak sama; penjelasan panjang tidak berganti ketika masih dibaca.

Seluruh 16 bab desktop/HP diperiksa, termasuk menu, keyboard, kamus, pencarian, salin source persis, fallback saat clipboard ditolak, pemulihan struktur buku dan blok terpilih. Fullscreen browser asli diuji pada context terpisah; keluar melalui Fullscreen API memulihkan buku dan menutup kamus. Fallback diuji ketika API tidak tersedia maupun mengembalikan penolakan. Bundel pribadi serta HTML hasil ekspor browser dibuka ulang: satu kanvas, seluruh 34 output dan workbook awal tetap tersedia. Tidak ada error JS atau jaringan eksternal.

Unit test `python -m unittest discover -s tests -p 'test_mlp_*.py' -v`: **14 passed, 0 failed, 0 skipped, 0 xfailed**, exit 0, Python `3.12.14`. Regresi pembaca lengkap **12 kelompok**, pembanding **8 kelompok**, dan interpretasi angka **6 kelompok**, semuanya exit 0/0 gagal. Template, kode sumber yang ditampilkan dan hasil penelitian diperiksa terpisah: JSON kode/catatan/output yang diembed identik dengan HEAD sebelumnya, notebook training tidak diubah, dan tidak ada eksekusi model/training baru.

Preview desktop/HP diperbarui dari HTML publik tanpa memuat file pribadi. HTML pribadi tetap di `outputs/` dan tidak diterbitkan. Bukti log, hasil browser, manifest SHA-256 dan audit privasi disimpan lokal pada `outputs/mlp_study/canvas_*`.

## Layar penuh dan catatan mengikuti kode, versi pertama — 22 Oktober 2026

Dilanjutkan dari HEAD `52199cfb46fca416ca40b9b5467533d42134850c`, branch `codex/aes-learning`. Tab **Kode + sticky note** mempunyai tombol **Layar penuh**: satu dialog memenuhi viewport dan memindahkan panel yang sama, sehingga kode dan catatan tetap berdampingan tanpa menggandakan source. Tombol kembali/Esc mengembalikan panel, fokus, blok, catatan dan posisi gulir. Pengambilan posisi dilakukan sebelum dialog ditutup karena elemen tersembunyi melaporkan posisi gulir nol.

Sticky note disejajarkan secara vertikal dengan baris terpilih. Scroll kode menggeser catatan; ketika baris terpilih keluar dari area baca, penanda penting yang terlihat dekat bagian atas menjadi catatan aktif. Geraknya dibatasi tinggi panel, dan catatan panjang tetap dapat digulir. Perilaku juga berlaku pada mode biasa dengan panel berdampingan. Di HP, mode layar penuh mempertahankan kedua kolom, sedangkan mode biasa tetap bertumpuk. Kamus istilah memakai dialog di atasnya; Esc pertama menutup kamus, berikutnya menutup layar penuh.

Perintah dari cwd `/workspace/Automated_Essay_Scoring`:

```bash
python -m unittest discover -s tests -p 'test_mlp_*.py' -v
node scripts/check_mlp_fullscreen.mjs \
  http://127.0.0.1:8765/docs/belajar_MLP_Spider.html \
  http://127.0.0.1:8765/outputs/mlp_study/belajar_MLP_Spider_Privat.html
node scripts/check_mlp_comparison.mjs
node scripts/check_mlp_metrics.mjs
```

Unit test: **14 passed, 0 failed, 0 skipped, 0 xfailed**, exit 0. Browser Chromium `154.0.8037.57`, Playwright `1.62.1`, Node `24.19.0`; Python unit test `3.12.14`. Uji layar penuh: **11 kelompok lolos, 0 gagal, 0 dilewati**, exit 0. Cakupan enam ukuran 320×568 sampai 1440×1000 termasuk landscape, seluruh 16 bab desktop/HP, dua scroll mandiri, posisi kertas terhadap sorotan, pergantian catatan ketika scroll, batas panel, fokus keyboard/background inert, kamus bertingkat, pencarian, salin kode asli dan fallback clipboard, pemulihan posisi, serta ekspor/reopen HTML pribadi dengan 34 output dan workbook awal tetap utuh. Tidak ada error JS atau permintaan jaringan eksternal.

Regresi pembaca lengkap `check_mlp_study.mjs` dijalankan dengan notebook/ZIP asli dan `outputs/mlp_study/expected_workbooks.json` seperti perintah di bawah: **12 kelompok lolos**, exit 0. Regresi perbandingan **8 kelompok**, interpretasi metrik **6 kelompok**, semuanya 0 gagal/exit 0. Kode, catatan dan JSON hasil yang diembed identik dengan versi sebelumnya; source notebook training tidak berubah dan tidak ada training/encoder yang dijalankan. Hasil double-frozen tetap belum dimasukkan. Build publik deterministik; bundel pribadi dibangun ulang dari berkas asli yang sama dan tetap berada di `outputs/`.

[Preview desktop](previews/mlp_study/10_kode_layar_penuh.png) dan [HP](previews/mlp_study/11_kode_layar_penuh_mobile.png) diambil dari HTML publik tanpa memuat berkas pribadi. Daftar source, SHA-256, log dan hasil pemeriksaan privasi disimpan lokal di `outputs/mlp_study/fullscreen_*`; tidak ikut diterbitkan.

## Meja tabel dan interpretasi — 22 Oktober 2026

Dilanjutkan dari HEAD `8c226b3d4f276f54cf8909382da9687f32f0ae28` setelah lingkungan cloud pulih. ZIP `Hasil_AES_Ringkas (1).zip` berhasil dibaca: pengaturan dan seluruh hasilnya cocok eksperimen awal **5 epoch**, bukan double-frozen. Source metadata workbook menyatakan RoBERTa fine-tuned, BGE frozen, head S3 linear. Hasil/log double-frozen tidak ditambahkan sesuai arahan pengguna.

Ekstraksi memakai `scripts/build_mlp_comparison.py --baseline ... --baseline-results ... --mlp-results ...`. Workbook awal dibandingkan dengan notebook awal pada seluruh 15 fold dan metrik test (toleransi `1e-5` terhadap tampilan notebook yang sebelumnya dibulatkan). Metrik/riwayat awal kini memakai angka penuh dari Excel. RMSE tiap skenario juga dihitung ulang dari 60 pasangan mentah dan cocok metrik sumber dengan toleransi `1e-5`.

Dataset 300 baris, 240/60 pembagian development/test, 1.200 baris keanggotaan training/validation kelima fold, nilai guru dan teks contoh **cocok persis** antara kedua workbook. Kedua tabel test dicocokkan melalui ID unik dan nilai guru. Bukti ini menggantikan batas keanggotaan split yang belum terverifikasi pada versi buku sebelumnya. Data individual tidak ditulis ke snapshot publik.

Model konstan yang hanya memakai rata-rata nilai guru **development** menghasilkan RMSE test `8.14280634295627` poin. Tidak memakai label test untuk fitting rata-rata. Nilai ini menjadi acuan relatif, bukan ambang kelulusan universal. Pilihan batas RMSE 5/10 poin pada UI diberi label ilustrasi dan bukan standar AES.

Menu **Tabel hasil & arti angka** menampilkan enam baris metrik test, enam MSE refit terakhir, 15 baris perbandingan fold untuk metrik yang dipilih, serta mean/SD RMSE CV. Penjelasan mengikuti skenario dan metrik: RMSE/MSE mendekati nol lebih kecil; MSE target 0–1 tidak dibatasi maksimum satu; Pearson mengukur hubungan, bukan kedekatan; QWK/AC2 dibaca terhadap acuan peluang masing-masing; AC2 tinggi bukan persentase prediksi tepat. MSE log selama pembaruan/dropout tidak disamakan dengan MSE test.

Unit test gabungan: **14 passed, 0 failed, 0 skipped, 0 xfailed**, exit 0. Dua pemeriksaan baru memastikan source ZIP memang awal lima epoch dan bukti pencocokan/benchmark hanya berupa agregat. Build publik deterministik dan sintaks ketiga modul JS diperiksa. Tidak ada perubahan notebook training atau eksekusi encoder/model baru.

Pengujian baru `node scripts/check_mlp_metrics.mjs`: **6 kelompok lolos, 0 gagal**, exit 0. Cakupan 60 kombinasi tiga skenario × lima metrik/loss × empat viewport; jumlah baris tabel; makna ambang ilustrasi; skala MSE/AC2; penolakan workbook berbeda; ekspor/reopen; serta tidak adanya error browser/jaringan eksternal. Enam puluh prediksi privat tetap cocok ketika urutan baris awal dibalik, membuktikan join berdasarkan ID, bukan posisi. Regresi menu pembanding dan pembaca lama tetap dijalankan; prasyarat uji klik layer diperbaiki agar target sudah terlihat sebelum mengukur posisi halaman.

Workbook evaluasi awal asli disimpan pada `outputs/mlp_study/baseline_eval.xlsx` dan tertanam hanya pada HTML pribadi. Pembaca publik menerima ZIP awal lewat pemilih lokal; SHA-256 ZIP/workbook diverifikasi. Ekspor HTML pribadi ikut menyimpan workbook itu. Semua nilai siswa, notebook terisi, ZIP unggahan dan hasil unduhan tetap tidak diterbitkan. [Preview meja hasil](previews/mlp_study/08_tabel_arti_angka.png) dan [HP](previews/mlp_study/09_tabel_mobile.png) diambil dari versi publik.

## Penambahan menu pembanding — 22 Oktober 2026

Pekerjaan lanjutan dimulai dari HEAD `2a67b89641ae50e5fad8a19a6341acdacdea4c98`, branch `codex/aes-learning`. Menu **Bandingkan eksperimen** memakai snapshot agregat terpisah `docs/mlp_comparison_snapshot.json`; buku MLP 16 bab dan notebook training tetap dipertahankan.

Notebook awal privat yang ditemukan di arsip kerja mempunyai SHA-256 `d4f00d143e9389841d8741586417c787dc5e03b77b254124ce15e9df2e9e171e`. Seluruh code cell cocok commit `3afad79`; output tersimpan memuat 15 fold, 75 epoch CV, 14 epoch refit, dan tabel test tiga skenario dengan keempat metrik RMSE/Pearson/QWK/AC2. Riwayat refit aktual berjumlah 5/4/5 epoch untuk S1/S2/S3. Metrik awal berasal dari tampilan sekitar delapan digit; log loss enam desimal dan RMSE empat desimal, sehingga tidak diklaim mempunyai presisi file Excel aslinya.

Perkembangan source antara diambil dari `312f394`: LR lebih kecil, maksimum 25 epoch, patience 5, LayerNorm tambahan RoBERTa S3, tetapi head S3 masih linear. Hasil akhir yang dapat dipasangkan secara pasti dengan versi antara itu belum tersedia; snapshot menyimpan `null`, bukan nol atau angka dari log run lain. Run MLP tetap mengikuti `15bc7bb` dan lima Excel pengguna.

Perintah pengujian terakhir dari cwd `/workspace/Automated_Essay_Scoring`:

```bash
python scripts/build_mlp_comparison.py --baseline outputs/revision25_before/user_saved_original.ipynb
python -m unittest discover -s tests -p 'test_mlp_*.py' -v
node --check scripts/templates/mlp_study.js
node --check scripts/templates/mlp_comparison.js
node scripts/check_mlp_comparison.mjs
```

Unit test gabungan: **12 passed, 0 failed, 0 skipped, 0 xfailed**, exit 0. Tambahan pemeriksaan menguji source tiap versi terhadap commit-nya sendiri, kelengkapan 15 fold awal dan seluruh epoch, kecocokan minimum riwayat dengan tabel, parser metrik negatif, pemisahan CV/refit tanpa newline, penolakan source berbeda, dan ketiadaan hasil antara maupun prediksi individual pada snapshot publik.

Pemeriksaan browser pembanding: **8 kelompok lolos, 0 gagal**, exit 0. Cakupannya 120 kombinasi dua versi pembanding × tiga skenario × lima tab × empat lebar layar (1440/900/390/320), serta 60 konfigurasi grafik aktual (dua versi × tiga skenario × lima fold × dua jenis kurva). Tidak ada page error, angka NaN/undefined, overflow halaman utama, atau permintaan jaringan eksternal. Navigasi keyboard, sticky note layer, source berdampingan dengan baris berubah, dan pergantian kembali ke buku MLP bekerja. Pengecekan regresi buku lama tetap mencakup 34 output dan seluruh 41 sheet asli.

HTML pribadi memuat notebook awal selain notebook/ZIP MLP. Tiga prediksi contoh awal dan tiga prediksi contoh MLP dirender terpisah. Skor guru contoh awal tidak tersedia dan tidak diisi dari contoh MLP. Identitas contoh maupun manifest anggota test/fold awal belum dapat dipasangkan lintas run. Ekspor/reopen browser mempertahankan ketiga file pribadi. Tabel lengkap 60 pasangan awal tidak tersedia pada sumber yang dapat dibuka; ZIP lama melebihi batas transfer 32 MiB alat unggahan. Buku menandai pasangan awal belum tersedia, tanpa membuat prediksi pengganti.

Angka RMSE S3 test berubah 11,722398 → 6,708539824280181; rata-rata CV 10,95416436 → 6,051639733368358. Perubahan bukan ablation satu komponen: head, LR, batas epoch, aturan berhenti dan LayerNorm juga berubah. Test awal sudah pernah dibaca. Buku menjelaskan batas kausalitas, ketidakpastian keanggotaan split, dan status eksploratif pemakaian ulang test. Double-frozen/MLP, Ridge dan pemeriksaan kontribusi head dibahas sebagai pertanyaan lanjutan dengan status hasil yang jelas.

Preview aman tambahan: [peta perbandingan](previews/mlp_study/04_perbandingan.png), [susunan layer](previews/mlp_study/05_layer.png), [training/CV](previews/mlp_study/06_training.png), dan [HP](previews/mlp_study/07_perbandingan_mobile.png). Diambil sebelum file pribadi dimuat. Output privat, contoh prediksi individual, notebook terisi dan bundel HTML lengkap tetap tidak dipublikasikan.

Pemeriksaan ini tidak mengeksekusi model, training atau evaluasi penelitian baru. Publikasi hanya memperbarui buku pembelajaran dan alat pembangunnya.

## Verifikasi awal — 21 Oktober 2026

Dikerjakan pada 21 Oktober 2026, di `/workspace/Automated_Essay_Scoring`, branch `codex/aes-learning`, dari HEAD awal `0c79d655898f967e855922be3c5fad1865efb1aa`. Buku mengikuti source MLP **15bc7bbb2aee91ecaf44fb91a0139a8c21f64a20**; notebook utama maupun eksperimen double-frozen tidak diubah.

## Identitas sumber

| Sumber | SHA-256 |
| --- | --- |
| Notebook source bersih dari commit 15bc7bb | `b06735cb45d632b92ad8ed30990e2e34d0fb66c46c612c528b812dd0d090440b` |
| Notebook unggahan dengan output | `76ab7f0fb9fb6d1b8b67da45bbf78e495b30b496c640e28ae9d3b026a0cc6026` |

Hash ZIP dan setiap workbook dicatat pada `docs/mlp_study_snapshot.json`. Seluruh 30 code cell unggahan cocok source commit tersebut. Ada 17 markdown cell, 2.488 baris kode, 120 catatan, dan 52/52 anchor fundamental ditemukan pada baris aslinya. Sumber data adalah lima workbook asli dari ZIP pengguna, dengan 41 sheet dan 34 objek output notebook tersimpan.

## Pemeriksaan yang dijalankan

Semua perintah menggunakan cwd di atas. Exit code pemeriksaan terakhir adalah **0**.

```bash
python -m unittest discover -s tests -p test_mlp_study.py -v
node --check scripts/templates/mlp_study.js
python scripts/build_mlp_study.py
```

Unit test: **6 passed, 0 failed, 0 skipped, 0 xfailed**. Pemeriksaan mencakup source lengkap, reversibilitas syntax highlight dan escaping markup, ketepatan baris sticky note, whitelist output publik, inventaris workbook, jumlah observasi matriks agregat, dan epoch CV terpilih terhadap riwayat aktual.

Build ulang publik dari snapshot aman menghasilkan byte HTML yang identik. Build privat dilakukan dengan argumen `--notebook` dan `--results` menunjuk kedua unggahan lokal. Python pengujian unit: 3.12.14; ekstraksi workbook: Python 3.13.5, openpyxl 3.1.5. Versi ini adalah lingkungan pembaca buku, bukan klaim versi training pengguna.

Untuk browser, server HTTP lokal hanya dipakai sebagai sarana pengujian karena lingkungan pengujian membatasi navigasi `file://`:

```bash
python -m http.server 8765 --bind 127.0.0.1 --directory /workspace/Automated_Essay_Scoring
node scripts/check_mlp_study.mjs \
  http://127.0.0.1:8765/docs/belajar_MLP_Spider.html \
  '/path/AES_300_S3_MLP (1).ipynb' \
  '/path/Hasil_AES_Ringkas (2).zip' \
  outputs/mlp_study/expected_workbooks.json
```

JSON pembanding privat disusun melalui openpyxl: seluruh header, jumlah baris, serta sampel baris pertama/tengah/akhir dan kolom pertama/kedua/tengah/terakhir di setiap sheet. JSON dan hasil unduhan browser tidak dipublikasikan.

Playwright dengan Chromium **154.0.8037.57**: **12 kelompok pemeriksaan lolos, 0 gagal**, tanpa page error atau permintaan jaringan eksternal:

- 16 bab × 4 tab × 2 ukuran layar = 128 kombinasi, tanpa overflow mendatar halaman utama.
- Seluruh 30 blok kode tampil dengan jumlah baris asli; sticky note menggulir kode tanpa memindahkan halaman utama.
- Navigasi tab menggunakan keyboard dan dialog istilah bekerja.
- Kedua unggahan dapat dibuka lokal; seluruh 34 output asli dirender pada bab terkait.
- Seluruh 41 sheet cocok pembanding openpyxl pada header, jumlah baris dan sampel nilai asli, termasuk nilai numerik dengan toleransi relatif `1e-12`.
- Komponen fusion ke-1024, pergantian halaman baris, tampilan angka lengkap dan inspeksi sel bekerja.
- Unduhan XLSX dari browser byte-identik dengan workbook asli.
- Ekspor satu HTML pribadi dari browser berhasil dibuka kembali dengan notebook dan lima workbook otomatis terisi.
- Output notebook yang sengaja disisipi script, event handler dan iframe tidak mengeksekusi kode atau memuat sumber eksternal; ZIP yang hash-nya berbeda ditolak.

[Preview desktop](previews/mlp_study/01_beranda.png), [kode dan catatan](previews/mlp_study/02_kode_catatan.png), serta [tampilan HP](previews/mlp_study/03_mobile.png) diambil dari versi publik sebelum file pribadi dimuat. Tampilan mobile juga diperiksa langsung pada viewport ruang baca; screenshot full-page pada perpindahan viewport dapat melewatkan lapisan scroll Chromium sehingga tidak dipakai sebagai preview HP.

## Privasi dan batas bukti

Audit membandingkan 300 jawaban asli terhadap HTML/snapshot dan source baru: **0 kecocokan teks jawaban lengkap**. Payload privat pada HTML publik kosong. Native output publik hanya memakai whitelist cell 12, 18, 32 dan bagian agregat cell 39 (indeks cell berbasis nol). Tabel individual tidak dimasukkan ke snapshot. Matriks publik hanya berisi jumlah agregat per rentang sepuluh poin, bukan pasangan nilai per siswa.

Seluruh file pribadi, executed notebook, ZIP asli, sampel pembanding dan HTML lengkap disimpan di `outputs/` atau direktori unggahan, tidak di-stage. HTML pribadi menyertakan file asli lengkap sesuai permintaan pengguna dan harus disimpan secara pribadi.

Pekerjaan ini **tidak menjalankan encoder, forward/backward, training, atau evaluasi AES baru**. Angka model berasal dari run tersimpan. Pemeriksaan browser dan unit test mengukur integritas buku serta pembacaan file, bukan peningkatan akurasi model. Perubahan hanya menambahkan buku dan alat pembangunnya; publikasi branch tidak melakukan merge ke main.
