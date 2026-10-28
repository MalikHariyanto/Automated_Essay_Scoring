# Buku belajar Colab terakhir

> Dokumen tahap terdahulu. Angka dan keputusan dibaca dalam konteks versi ini; untuk kondisi aktif gunakan [status proyek](../../STATUS_PROYEK.md) dan [panduan AI](../../PANDUAN_AI_PROYEK.md).

> Catatan historis: HTML terpisah pada dokumen ini telah digantikan oleh [dua buku terbaru](../../downloads/README.md). Kode utama buku tiga variabel sekarang E24; file lama dapat ditelusuri pada commit versi asal.

File **belajar_colab_AI_interaktif.html** dapat dibuka sendiri di Chrome/Edge/Firefox, tanpa Colab, Python, instalasi atau CDN. Tampilan memakai warna kertas pastel, header gelap, dan sticky note di pinggir baris kode penting. Setiap tab proses memiliki **Meja belajar** dengan scroll sendiri dan pilihan **Kode + catatan**, **Output aktual**, **Colab lo**, serta **Kode lengkap**. Pilihan Santai / Dua bahasa / Teknis tetap tersedia. Fitur cetak PDF dihapus.

## Membuka

1. Buka file `docs/belajar_colab_AI_interaktif.html` di GitHub pada branch `codex/aes-learning`.
2. Tekan **Download raw file** (ikon unduh pada file), lalu simpan dengan ekstensi `.html`.
3. Buka file yang tersimpan di browser. GitHub menampilkan source HTML; interaksi berjalan setelah file dibuka di browser sebagai HTML.
4. Mulai dari tab **Mulai & setup**. Tab bisa dipilih langsung, lewat Sebelumnya/Berikutnya, atau tombol panah keyboard. Pada HP tab dapat digeser horizontal.

Pilihan **Santai** menjelaskan istilah dengan bahasa sederhana; **Teknis** memperlihatkan definisi dan detail implementasi; **Dua bahasa** menampilkan keduanya. Kode tetap tersedia pada mode mana pun.

## Isi

Sebelas bab membahas setup, input/audit, cleaning, token/window, vektor/CLS, S1/S2/S3, loss/training, empat metrik, error/bias, pertanyaan 8/2, dan peta kode. Buku memuat 27 langkah penjelasan kode dalam sticky note dan 23 catatan pada baris cell Colab, 11 source cell Colab, 17 file source/config, 12 baris hasil evaluasi aktual, shapes dan contoh lima koordinat vektor, MSE train sebelum/sesudah, serta walkthrough agregat S3–isi.

Snapshot diperbarui dari notebook terbaru yang diunggah pengguna (SHA-256 `473021666e2e38d21ac8911c816a30b23eb99dad81dc77896d891bee0d3da068`). Source sebelas cell cocok dengan notebook publik; seluruh angka agregat sama dengan audit sebelumnya pada source `0b52413`. Ini pembacaan hasil tersimpan, bukan eksperimen baru. Training menggunakan usulan AI pada delapan teks draf; empat train dan empat validation, dua teks terpotong ditunda. Tidak ada skor guru/penelitian baru. Ilustrasi 8/2 berlaku bila sepuluh teks layak; sekitar 6/2 jika delapan layak, dengan kelompok tetap utuh. Simulator tidak mengubah split, model, atau hasil notebook.

## Melihat output pribadi lengkap

Tab **Colab lo** langsung menampilkan source sebelas cell serta ringkasan output terbaru tanpa unggahan. Pilih cell dan gunakan tombol **Ke output tersimpan**. Ini ringkasan yang disaring dari notebook pengguna, bukan salinan seluruh stdout.

Teks siswa, rubric descriptors dan nilai per esai tidak dipublikasikan di buku. Untuk bagian pribadi itu, di Colab pilih **File → Download → Download .ipynb**. Pada tab pertama HTML pilih **Pilih notebook pribadi**; source dan output tersimpan muncul pada pilihan **Colab lo** di Meja belajar tiap tab. Pilih cell yang ingin dibaca; tombol **Ke output tersimpan** membawa langsung ke hasil di dalam box. Tabel ditampilkan sebagai tabel teks aman, disertai catatan cara membacanya.

Pembaca memakai file lokal dalam memori browser. Tidak mengunggahnya ke internet atau menyimpan data tersebut dalam localStorage. Script/widget dari output tidak dijalankan. Gunakan **Lepaskan notebook pribadi** atau reload untuk melepaskannya. Tabel ringkas publik tetap snapshot audit; tidak dicampur dengan hasil file yang mungkin berbeda. Jangan membagikan notebook terisi/hasil privat ke repo publik.

## Simulator dan bukti

Window simulator menunjukkan coverage isi dengan capacity 510/stride 382. Diagram S1/S2/S3 dapat mengganti B untuk mempelajari shapes. Simulator LR memperlihatkan mengapa 4 dan 8 esai train, pada effective batch 8 dan dua epoch, sama-sama hanya menghasilkan satu update aktif. Lab metrik memakai angka ilustrasi yang bisa diubah; RMSE/Pearson raw, QWK/AC2 kategori, midpoint ke bawah, undefined tetap dijelaskan. Semua latihan ini matematika, tidak disimpan sebagai dataset/skor siswa.

Navigasi, mode baca, sticky note, highlight baris, scroll di dalam box, tema, penyimpanan progres, input kosong/extreme, pembaca file, dan tampilan HP diperiksa di Chromium. Pada versi baru, 55 fixture metrik JS dicocokkan ke Python. Referensi R tidak dijalankan ulang karena paket irrCAC belum tersedia; bukti pemeriksaan R versi sebelumnya tetap disimpan. Bukti versi awal tersimpan di `docs/verifikasi/colab_study_html.json`; versi tanpa kuis dengan penjelasan kode terdapat di `docs/verifikasi/colab_code_explanations.json`. Bukti tampilan sticky note terbaru tersimpan di `docs/verifikasi/colab_sticky_notes.json`: 11 tab, 44 output pribadi, scroll terbatas, keyboard dan HP lulus. Tidak menjalankan training ML baru pada tugas HTML ini.

## Memperbarui buku sebagai developer

`python scripts/build_colab_study.py` membangun ulang HTML dari snapshot agregat aman yang sudah ada dan whitelist source publik. Tidak memerlukan dataset siswa. Template berada di `scripts/templates/colab_study.html`; modul penjelasan output di `scripts/templates/colab_outputs.js` dimasukkan saat build agar HTML tetap satu file offline.

`--notebook` dibatasi checksum notebook Colab yang diaudit; bukan importer umum bagi hasil baru tanpa audit. Untuk snapshot baru, audit dulu sumber/hasil dan aturan privasi sebelum memperbarui extractor. Notebook terbaru di atas sudah diaudit; source cocok dan hasil agregat tidak berubah. Snapshot tidak berisi teks, ID, skor individual, rubric descriptors, workbook, cache atau bobot.

Browser verification menggunakan Playwright + Chromium yang tersedia di lingkungan developer dan dependency proyek, bukan dependency untuk pembaca HTML. Contoh perintah cloud: `AES_RSCRIPT=/workspace/aes-r-local/bin/Rscript /workspace/aes-venv/bin/python scripts/check_colab_study.py`. `--private-notebook` opsional menguji pembaca dengan notebook lokal; hasil privat tidak dicetak ke bukti publik.

## Membaca kode sebagai pemula

Kuis dihapus sesuai preferensi pengguna. Pada setiap proses, baca **Baca kode pelan-pelan** sebelum kotak output: nama yang dipanggil → tugas baris → arti variabel/tanda → hubungan ke cell/output Colab. Potongan diambil dari source asli dan diperiksa kesesuaiannya saat build. Baris yang disorot dan sticky note memakai warna yang sama. Beberapa baris konteks di sekitar baris penting tetap ditampilkan agar alurnya tidak terputus. Pada HP, catatan pindah ke bawah kode. Scroll box dapat difokuskan dengan Tab; tab bacaan di dalamnya memakai tombol panah kiri/kanan, Home, dan End. Fungsi lengkap masih tersedia sebagai bacaan lanjutan.

Contoh RoBERTa: checkpoint `flax-community/indonesian-roberta-base` → fungsi proyek `load_roberta(...)` → API Transformers `AutoModel.from_pretrained(...)` → pemakaian model `self.roberta(**windows)` → ambil hidden token awal → mean window → head empat output. Tab **Jelajah kode** menyediakan peta nama model dan lima tanda Python yang cukup dikenali dulu.

Output aktual terikat ke notebook unggahan terbaru; perubahan penjelasan tidak melatih ulang atau mengganti angka. Contoh CLS sebelum latihan (dua window) dan forward setelah latihan (contoh satu window) diberi konteks agar shape berbeda tidak dianggap error. Data pribadi lengkap tetap dibuka lokal dari notebook sendiri.

## Edisi pribadi dengan output langsung terpasang

Builder dapat membuat satu HTML pribadi yang langsung memuat sebelas cell dan 44 output dari notebook terbaru, tanpa memilih file lagi:

```bash
python scripts/build_colab_study.py --notebook /path/notebook-terbaru.ipynb \
  --private-output outputs/colab_study/Belajar_Colab_Terbaru.html
```

File pribadi harus berada dalam `outputs/`, yang diabaikan Git. File ini berisi data pribadi dari output notebook dan tidak untuk publikasi. HTML publik di `docs/` hanya menyertakan agregat aman. Melepas notebook menghapus hasil dari tampilan; file HTML pribadi tetap menyimpan snapshot aslinya dan dapat dimuat kembali dengan membuka ulang file. Tidak ada training baru atau perubahan notebook sumber.

## Tiga preview warna classy

Tema utama terbaru yang dipilih pengguna adalah **Navy & Cognac** (atas kiri): navy, putih, dan cokelat. Preferensi warna percobaan sebelumnya tidak mengganti pilihan awal ini.

Di bagian atas halaman pertama, klik **Navy & Cognac** (atas kiri), **Olive & Cream** (bawah kiri), atau **Rose & Navy** (bawah kanan). Warna diadaptasi dari referensi Old Money pengguna. Pilihan berlaku pada seluruh buku dan sticky note selama halaman dibuka. Saat membuka ulang buku, tema pilihan pengguna **Navy & Cognac** selalu diterapkan; pilihan preview yang tersimpan sebelumnya tidak menimpanya. Tema terang/gelap tetap tersedia.

Gambar halaman pertama: [navy](../previews/colab_classy_navy.png), [olive](../previews/colab_classy_olive.png), [rose](../previews/colab_classy_rose.png). Gambar hanya berasal dari buku publik yang menampilkan hasil agregat.

Untuk membuat ulang ketiga preview dan memeriksa tombol, keyboard, penyimpanan pilihan, mode gelap, serta layout 390/768/1024/1440 px, jalankan `python scripts/preview_colab_palettes.py`. Build HTML tetap `python scripts/build_colab_study.py`. Bukti ada di `docs/verifikasi/colab_classy_palettes.json`.

Versi bernama **belajar_colab_Rose_Navy.html** memuat buku publik yang sama dengan tema bawah kanan saat dibuka. Untuk membangunnya ulang: `python scripts/build_colab_study.py` lalu `cp docs/belajar_colab_AI_interaktif.html docs/belajar_colab_Rose_Navy.html`. Mode perbandingan developer harus diaktifkan eksplisit dengan `?preview=1&palette=navy` (atau `olive`/`rose`); URL `?palette=navy` saja tidak mengganti tema buku terbitan.

## Output yang dibedah seperti kode

**Output aktual** kini memiliki catatan untuk field setup, audit, cleaning, token/window, matriks representasi/shape, koordinat vektor, concat, status training, MSE, langkah optimizer, empat metrik, dan pembanding sederhana. Catatan menjelaskan apa angkanya, arti sumbu/field, dan batas kesimpulannya.

Pada tab **Empat metrik → Output aktual**, pilih salah satu dari empat matriks S3–isi: hitungan pasangan, hitungan harapan QWK, bobot disagreement, atau bobot agreement AC2. Klik sel; tombol panah memindahkan pilihan. Baris selalu acuan AI, kolom selalu prediksi. Sel baris 4/kolom 3 pada hitungan aktual berisi 2: dua pasangan memiliki acuan 4 dan prediksi kategori 3. Angka 0 di matriks hitungan berarti tidak ada pasangan; diagonal bobot disagreement 0 berarti kategori cocok; diagonal agreement 1 berarti bobot cocok penuh. Ini tiga arti berbeda.

Rumus harapan, bobot, Pa, Pe, dan QWK diverifikasi ulang terhadap matriks dari notebook dan modul Python. Hasil angka tersimpan tidak diganti; tidak melatih model atau menjalankan ulang notebook. Bukti pemeriksaan versi ini ada di `docs/verifikasi/colab_output_explanations.json`.

File baru **belajar_colab_Navy_Cognac.html** adalah salinan buku utama terbaru. Buat ulang dengan `python scripts/build_colab_study.py` lalu `cp docs/belajar_colab_AI_interaktif.html docs/belajar_colab_Navy_Cognac.html`. Gunakan file ini untuk versi navy terbaru; file/tautan Rose_Navy sebelumnya merupakan versi historis.

Preview versi ini: [Colab lo sudah terisi](../previews/colab_public_cells.png) dan [penjelasan sel matriks](../previews/colab_output_matrix.png).
