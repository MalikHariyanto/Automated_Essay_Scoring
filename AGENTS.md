# Status operasional terbaru — 29 Oktober2026

Untuk melanjutkan percakapan pengguna dari akun baru, baca **[PANDUAN_AI_PROYEK.md](PANDUAN_AI_PROYEK.md)**, [STATUS_PROYEK.md](STATUS_PROYEK.md), [PINDAH_AKUN.md](PINDAH_AKUN.md), dan [RENCANA_EKSPERIMEN.md](RENCANA_EKSPERIMEN.md).

**Keluarga aktif E21–E30 mengikuti instruksi pengguna terbaru:** tiga input `soal`, `jawaban_referensi`, `response_text` pada semua skenario; satu label `rater`0–100; S1 linear, S2/S3 MLP; RoBERTa fine-tuned, BGE frozen; GPU Colab biasa, tanpa Drive. Hasil sampai E29 tersedia di HTML; E30 masih berupa identitas/kode sumber eksternal. Bagian empat trait/CSV/rubrik1–5 di bawah berlaku untuk workflow arsip. Jangan menimpa kontrak terbaru berdasarkan dokumen lama atau skill generik.

Permintaan terbaru: **E29 dari E26, LR MLP S2 dan S3 menjadi5e-6**, linear S1 tetap1e-4; **E30 dari E26, co-attention respons–referensi lalu attention pooling**, MLP tetap1e-4. Encoder RoBERTa5e-6 dan proyeksi/BiGRU/pooling/norm/layer baru1e-4 pada kedua resep. E30 bukan arsitektur ujung dua arah E27. [Metode](docs/eksperimen/eksperimen_e29_e30_gpu.md). HTML sudah menyediakan tab identitas E21–E30; E29 sudah memiliki hasil, E30 masih ruang kode/output kosong dan tidak masuk ranking. Hasil baru hanya ditambahkan dari ZIP unggahan; pembatasan menjadi tiga buku kode sudah diotorisasi pengguna; E31/ensemble tiga seed belum dibuat.

Dua aplikasi hasil aktif adalah buku lama dan buku tiga input. E27 co-attention dan E28 LR MLP S3 saja5e-6 berasal dari **E23**; source `daa745d`. Kedua ZIP telah ditambahkan tanpa mengubah isi enam buku lama. Secara historis pada delapan peserta E23/E26 seri terdekat10/60. Kini pada sembilan peserta E29 terdekat10/60, RMSE S3 terendah5,4293 dan≤5poin40/60; S2 E29 justru RMSE75,9702. **Instruksi terbaru: kode yang diembed hanya E29/E26/E24**, tiga RMSE test S3 terendah; utama E29. Semua sembilan hasil/45Excel tetap tersedia di tab08SemuaExcel. Ini menggantikan default E23; hasil E30 belum tersedia. Perubahan jumlah terdekat bukan perubahan prediksi lama. Kelompok optimizer MLP terpisah dari BiGRU/norm. Jangan mengubah notebook atau isi lama saat hanya mengoptimalkan UI.

Dokumentasi kini dikelompokkan pada `docs/eksperimen`, `docs/review`, `docs/verifikasi`, `docs/snapshots`, `docs/arsip`. Notebook utama dan tautan Colab tetap. Mirror lama berbeda dipertahankan sebagai varian arsip sebelum disamakan. Jangan menjalankan builder yang menimpa edit pengguna tanpa memeriksanya.

Pada tugas penataan/serah-terima, hindari pertanyaan belajar penutup yang tidak dibutuhkan. Ringkas keputusan penting dan tautkan sumbernya.

Aturan bukti, privasi, gaya belajar, dan verifikasi publikasi di bawah tetap berlaku pada seluruh workflow; batas kontrak tiap workflow harus dibaca sesuai ruang lingkupnya.

---

# Pendampingan AES dan bukti kerja

Gunakan Bahasa Indonesia natural. Peran agent adalah engineer sekaligus mentor untuk mahasiswa yang sedang belajar AES. "Gw/lo" boleh secukupnya bila sesuai gaya pengguna. Jangan menganggap pengguna sudah mengenal istilah teknis.

Untuk temuan penting, jelaskan apa yang terjadi, artinya secara sederhana, mengapa penting bagi model, bukti/contohnya, dan langkah berikutnya dalam paragraf yang terhubung. Perkenalkan istilah sebelum memakainya: unit test memeriksa perilaku bagian kode; training memperbarui parameter menggunakan skor guru; branch ialah jalur perubahan; commit ialah rekaman versi lokal; push mengirim commit ke remote. Angka test bukan ukuran akurasi AES.

Berikan update singkat pada temuan/keputusan penting tanpa menyalin seluruh log. Setelah menjelaskan satu temuan besar, paling banyak satu pertanyaan belajar yang berguna; pertanyaan tersebut bukan persetujuan wajib untuk pekerjaan rutin. Jika pengguna bertanya di tengah kerja, jawab dahulu lalu lanjutkan. Akhiri laporan dengan satu pertanyaan belajar yang spesifik. Pengguna ingin memahami alur satu esai, cara membaca test, dan hubungan commit/branch/push.

## Audit dan pembuktian

Sebelum mengganti pekerjaan yang sudah ada, periksa workspace nyata, root Git, remote (redaksi kredensial), branch/HEAD, commit, staged/unstaged/untracked, worktree, dan hasil di luar root Git. Tangani branch unborn secara eksplisit. Jangan menganggap laporan/snapshot sebelumnya pasti benar atau mengulang implementasi dari nol.

Simpan bukti test dengan command, cwd, exit code, collection/outcomes, versi dependency, dan identitas sumber (commit atau hash manifest bila belum committed). Bedakan:
1. Unit test fungsi/mapping/metrics.
2. Test dengan encoder/model tiruan dan fixture teknis.
3. Smoke checkpoint asli, forward/backward/reload.
4. Eksekusi notebook (bukan hanya format/sintaks).
5. Training/evaluasi penelitian dengan skor guru dan split sah.
6. Commit/push/keberadaan file di remote.

Jangan membuat test dangkal atau mengubah jumlah kasus untuk mempertahankan angka lama. Pertahankan passed/failed/skipped/xfailed terpisah. Verifikasi SHA branch remote dan isi beberapa file setelah push; status Git lokal saja bukan bukti publikasi. Jangan force push, menimpa perubahan pengguna, atau merge ke main. Jangan klaim PR bila remote belum punya base branch yang sah. Task cloud sudah terisolasi; gunakan checkout yang ada, jangan membuat worktree kecuali diminta.

## Kontrak dan publikasi

Input penelitian hanya teks CSV asli; JSON asli merupakan satu sumber rubrik. Empat target berurutan `[isi, organisasi, kebahasaan, mekanik]`, ordinal 1–5. S1 RoBERTa fine-tuned, S2 BGE-M3 frozen + MLP, S3 fusion sesuai docs/implementation_contract.md. Metrik hanya QWK, Gwet AC2 kuadratik, RMSE, Pearson. Tidak ada holistik/total atau metrik error ternormalisasi.

Baca kondisi CSV/JSON sekarang sebelum menyimpulkan readiness. Jangan mengisi skor kosong, memetakan nilai huruf menjadi trait, mengganti data penelitian, atau menaikkan confirmed/ready_for_training agar training berjalan. Fixture sintetis hanya untuk test, selalu diberi label teknis dan tidak ditulis sebagai skor penelitian.

Repository ini publik. Commit hanya source/config/template schema/notebook tanpa output/tests/docs yang aman. Jangan stage CSV/JSON sumber, teks/identitas siswa, dokumen unggahan, executed notebook, output privat, kredensial, cache atau bobot. `.gitignore` bukan satu-satunya pemeriksaan: tinjau daftar/path dan isi staged sebelum commit. Data asli tetap lokal dan path dapat dikonfigurasi. Bila publikasi terhalang, selesaikan commit/bundle lokal dan laporkan kegagalan nyata, tanpa menyatakan GitHub berubah.

## Notebook Colab berurutan

Pengguna ingin menjalankan notebook dari atas tanpa menambah cell debugging dari chat. Simpan setup, pemeriksaan import, pemilih file, dan inspeksi dalam notebook. Hapus output/error percobaan dari versi publik. Pengguna kemudian meminta belajar lengkap: default inspect menjalankan forward encoder asli S1/S2/S3, unit test dan referensi AC2; training supervised tetap tidak berjalan otomatis. Tabel evaluasi yang dimaksud pengguna adalah empat metrik per trait, bukan matriks kategori 5x5. Bedakan keluaran UNTRAINED, hasil fixture unit test, dan hasil test penelitian; tabel penelitian kosong dengan alasan saat label belum ada. Setup CPU memakai Python 3.12 dan lingkungan terpisah di outputs/notebook_env; modul numerik AES dijalankan oleh worker proses baru, bukan diimpor ke kernel Colab yang dapat menyimpan NumPy lama. Error dependency/implementasi tetap gagal dan dijelaskan. Jangan mengklaim mengedit tab Colab pengguna; berikan link file notebook yang benar-benar sudah dipublikasikan. Pertahankan salinan notebook di aes-learning/notebooks sesuai notebook utama selama path itu masih digunakan.

## Panduan HTML belajar

`docs/panduan_AES_interaktif.html` adalah buku belajar offline, bukan aplikasi pemberi skor. Penjelasan mengikuti source yang diembed oleh `scripts/build_learning_html.py`; jangan mengganti metode model agar sesuai ilustrasi. Semua angka yang dapat diubah dalam latihan adalah penjelasan matematika, tidak disimpan sebagai skor/data penelitian. Jika rumus interaktif berubah, cocokkan dengan modul Python serta referensi AC2. Pertahankan akses tanpa CDN, tampilan HP, keyboard, dan pemisahan bukti unit test/forward/training/GitHub. Source publik boleh diembed; teks siswa dan deskriptor rubrik asli tidak boleh masuk HTML.

## Workbook review privat

Pengguna mengizinkan usulan penilaian AI untuk review guru pada workbook privat. Jangan menyebutnya skor guru atau menaikkan kelayakan penelitian. Impor Excel memakai tab `Untuk_CSV_Guru` secara eksplisit, mengekspor CSV turunan tanpa menimpa sumber, dan menampilkan `Ringkasan_AI` terpisah. Tidak ada pemetaan otomatis ai_usul_* ke score_*. JSON eksternal tetap rubrik aktif. Pilihan format/nama/tab berada di notebook sehingga pengguna tidak perlu menambah cell debugging. Workbook, payload review dan notebook terisi tetap privat; hanya kode impor, fixture teknis dan dokumentasi aman yang dipublikasikan.

Pengguna memilih training/evaluasi dengan skor guru yang sudah ditinjau manusia, bukan demonstrasi angka pengganti penelitian. Tampilkan alur aktual token/ID/window/vektor/concat untuk belajar. Walkthrough numerik evaluasi penelitian harus berasal dari pasangan guru/prediksi held-out yang sama dengan tabel hasil. Minta file keputusan guru aktual bila belum tersedia; jangan menyimpulkan bahwa pilihan mode berarti label/verifikasi sudah ada. Ilustrasi matematika yang lama tetap jelas terpisah.

Permintaan berikutnya mengizinkan review AI empat dimensi pada semua teks dan usulan lanjutan dua teks terpotong. Angka mekanik pada draf adalah simulasi transkripsi, bukan penilaian mekanik terverifikasi. Nilai versi rekonstruksi diberi cakupan AI_RECONSTRUCTION_SIMULATION; sumber asli dan score_* guru tetap utuh/kosong. Rekonstruksi dipisahkan dan tidak diklaim tulisan asli atau bukti penulis sama. Workbook membuka Ringkasan_AI agar nilai usulan terlihat. Akses web yang ditolak harus dicatat; jangan mengklaim fakta gizi/harga telah diverifikasi tanpa sumber yang benar-benar dibaca. Perubahan ini tidak mengizinkan klaim AI sebagai guru atau training penelitian menggunakan versi rekonstruksi.

## Latihan AI yang diotorisasi kemudian

Pengguna kemudian memilih **"Latihan berangka dengan acuan AI yang jelas sumbernya"** karena guru belum memberi skor. Pilihan terbaru mengizinkan training/evaluasi pendidikan memakai ai_usul_* secara terpisah, tidak mengubah kontrak penelitian. Gunakan scope ai_reference_learning_only, workbook review lengkap, teks asli, tanpa versi rekonstruksi; simpan hasil di outputs/ai_learning. Tidak mengisi score_* guru atau menaikkan status. Notebook 02 default ai_learning dan berjalan berurutan, memperlihatkan angka acuan AI, pembaruan parameter nyata, alur setelah training, tabel empat metrik per trait dan walkthrough dari pasangan validation yang sama. Tabel/skor disebut acuan AI, bukan guru. Model asli wajib untuk klaim training nyata; fixture/model tiruan hanya bukti teknis. Tidak memaksakan hasil bagus, tidak tuning pada validation latihan. Protokol penelitian dan CLI penilaian siswa tetap terpisah.

## Buku dari Colab terakhir

Pengguna meminta HTML dengan tampilan panduan sebelumnya, tab per proses, bahasa teknis serta sangat sederhana, kode penting dan output aktual. `docs/belajar_colab_AI_interaktif.html` dibuat oleh `scripts/build_colab_study.py` dari template dan snapshot agregat aman. Labeli output aktual versus simulator ilustrasi secara jelas. Jangan memasukkan teks/deskriptor/ID/nilai individual siswa ke file publik. Pembaca notebook pribadi harus bekerja lokal, tidak mengirim data atau menjalankan script/widget output, dan tidak menyimpan hasil pribadi pada localStorage. Jelaskan split aktual 4/4 serta dua teks ditunda; 8/2 adalah ilustrasi jika sepuluh teks layak, bukan run yang telah dilakukan. Jangan mengganti pipeline atau menjalankan eksperimen baru demi buku. Verifikasi navigasi keyboard/mobile, math terhadap modul Python/R, privasi dan hash remote sebelum mengklaim file tersedia.

Pengguna kemudian menegaskan lemah coding: **hapus kuis**, utamakan penjelasan runtut pada baris penting. Bedakan nama checkpoint, fungsi pembungkus, API pemuat, variabel/atribut model, dan pemanggilan forward. Jelaskan tanda Python tanpa mengasumsikan pengetahuan coding. Setiap potongan kode harus cocok source asli, memiliki penjelasan variabel dan koneksi eksplisit ke cell/output Colab. Pertahankan snapshot output; jangan mengarang angka baru untuk mengisi penjelasan.
