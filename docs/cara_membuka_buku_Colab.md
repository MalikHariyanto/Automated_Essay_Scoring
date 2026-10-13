# Buku belajar Colab terakhir

File **belajar_colab_AI_interaktif.html** dapat dibuka sendiri di Chrome/Edge/Firefox, tanpa Colab, Python, instalasi atau CDN. Tema dan struktur mengikuti panduan HTML sebelumnya, dengan tab halaman per proses dan penjelasan Santai / Dua bahasa / Teknis.

## Membuka

1. Buka file `docs/belajar_colab_AI_interaktif.html` di GitHub pada branch `codex/aes-learning`.
2. Tekan **Download raw file** (ikon unduh pada file), lalu simpan dengan ekstensi `.html`.
3. Buka file yang tersimpan di browser. GitHub menampilkan source HTML; interaksi berjalan setelah file dibuka di browser sebagai HTML.
4. Mulai dari tab **Mulai & setup**. Tab bisa dipilih langsung, lewat Sebelumnya/Berikutnya, atau tombol panah keyboard. Pada HP tab dapat digeser horizontal.

Pilihan **Santai** menjelaskan istilah dengan bahasa sederhana; **Teknis** memperlihatkan definisi dan detail implementasi; **Dua bahasa** menampilkan keduanya. Kode tetap tersedia pada mode mana pun.

## Isi

Sebelas bab membahas setup, input/audit, cleaning, token/window, vektor/CLS, S1/S2/S3, loss/training, empat metrik, error/bias, pertanyaan 8/2, dan peta kode. Buku memuat 11 source cell Colab, 17 file source/config, 12 baris hasil evaluasi aktual, shapes dan contoh lima koordinat vektor, MSE train sebelum/sesudah, serta walkthrough agregat S3–isi.

Angka aktual berasal dari notebook pada commit `0b52413` yang sudah diaudit, bukan eksperimen baru. Training menggunakan usulan AI pada delapan teks draf; empat train dan empat validation, dua teks terpotong ditunda. Tidak ada skor guru/penelitian baru. Ilustrasi 8/2 berlaku bila sepuluh teks layak; sekitar 6/2 jika delapan layak, dengan kelompok tetap utuh. Simulator tidak mengubah split, model, atau hasil notebook.

## Melihat output pribadi lengkap

Teks siswa, rubric descriptors dan nilai per esai tidak dipublikasikan di buku. Di Colab pilih **File → Download → Download .ipynb**. Pada tab pertama HTML pilih **Pilih notebook pribadi**; source dan output tersimpan muncul pada bagian **Output pribadi untuk proses ini** di tiap tab.

Pembaca memakai file lokal dalam memori browser. Tidak mengunggahnya ke internet atau menyimpan data tersebut dalam localStorage. Script/widget dari output tidak dijalankan. Gunakan **Lepaskan notebook pribadi** atau reload untuk melepaskannya. Tabel ringkas publik tetap snapshot audit; tidak dicampur dengan hasil file yang mungkin berbeda. Jangan membagikan notebook terisi/hasil privat ke repo publik.

## Simulator dan bukti

Window simulator menunjukkan coverage isi dengan capacity 510/stride 382. Diagram S1/S2/S3 dapat mengganti B untuk mempelajari shapes. Simulator LR memperlihatkan mengapa 4 dan 8 esai train, pada effective batch 8 dan dua epoch, sama-sama hanya menghasilkan satu update aktif. Lab metrik memakai angka ilustrasi yang bisa diubah; RMSE/Pearson raw, QWK/AC2 kategori, midpoint ke bawah, undefined tetap dijelaskan. Semua latihan ini matematika, tidak disimpan sebagai dataset/skor siswa.

Navigasi, mode baca, quiz, theme, penyimpanan progres, input kosong/extreme, pembaca file, mobile dan print diperiksa di Chromium. Perhitungan JS dicocokkan ke Python dan backend R. Bukti tersimpan di `docs/verification/colab_study_html.json`. Tidak menjalankan training ML baru pada tugas HTML ini.

## Memperbarui buku sebagai developer

`python scripts/build_colab_study.py` membangun ulang HTML dari snapshot agregat aman yang sudah ada dan whitelist source publik. Tidak memerlukan dataset siswa. Template berada di `scripts/templates/colab_study.html`.

`--notebook` dibatasi checksum notebook Colab yang diaudit; bukan importer umum bagi hasil baru tanpa audit. Untuk snapshot baru, audit dulu sumber/hasil dan aturan privasi sebelum memperbarui extractor. Snapshot tidak berisi teks, ID, skor individual, rubric descriptors, workbook, cache atau bobot.

Browser verification menggunakan Playwright + Chromium yang tersedia di lingkungan developer dan dependency proyek, bukan dependency untuk pembaca HTML. Contoh perintah cloud: `AES_RSCRIPT=/workspace/aes-r-local/bin/Rscript /workspace/aes-venv/bin/python scripts/check_colab_study.py`. `--private-notebook` opsional menguji pembaca dengan notebook lokal; hasil privat tidak dicetak ke bukti publik.
