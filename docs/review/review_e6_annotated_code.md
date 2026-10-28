# Panduan kode E6 dengan catatan di samping

> Dokumen tahap terdahulu. Angka dan keputusan dibaca dalam konteks versi ini; untuk kondisi aktif gunakan [status proyek](../../STATUS_PROYEK.md) dan [panduan AI](../../PANDUAN_AI_PROYEK.md).

> Catatan historis: HTML terpisah pada dokumen ini telah digantikan oleh [dua buku terbaru](../../downloads/README.md). Kode utama buku tiga variabel sekarang E24; file lama dapat ditelusuri pada commit versi asal.

`downloads/Belajar_Kode_E6.html` adalah satu HTML offline ringan, terpisah dari buku hasil eksperimen. Memuat seluruh 30 code cell E6 (2.695 baris), 153 catatan, sepuluh padanan screenshot mDeBERTa ke E6, dan pencarian 99 bagian dari panduan Excel. Tidak memuat output notebook, dataset, nilai individual, atau arsip hasil.

Source dipin pada `2257cb0f601148764d7844fc6ffca925a9b2ab95`, path `notebooks/experiments/AES_300_E6_S3_LR5e6_CPU_Reuse.ipynb`, SHA256 `ddaf09d78d962ea3a15dabb5d43fcaead5ca3c96e9ec4bc01c0b241d37dc8232`. Source sama dengan meja belajar E6 dan Excel panduan sebelumnya. Notebook dan metode tidak diubah.

Baris penting disorot. Tiap catatan mulai sejajar dengan baris sumbernya pada desktop; di HP, catatan berada setelah ruas kode terkait. Tinggi tiap ruas mengikuti kode atau catatan yang lebih panjang. Seluruh kode dan catatan memakai satu scroll vertikal bersama, tanpa panel notes yang bergulir sendiri. Layar penuh menyembunyikan halaman di belakang dan mendukung Esc, termasuk fallback ketika Fullscreen API tidak tersedia.

Catatan membedakan nama checkpoint, objek encoder, definisi fungsi/kelas dan pemanggilannya; fitur dan target; CLS tiap window serta pooling antar-window; head linear S1 dan MLP S2/S3; LayerNorm/proyeksi/concat; normalisasi target, loss, backward, optimizer, validasi, refit, serta pemakaian ulang S1/S2. Peta screenshot menjelaskan bahwa E6 tidak memakai konteks pair ataupun ContextPooler mDeBERTa. MLP merupakan head regresi, bukan encoder.

Verifikasi browser: sepuluh kelompok pemeriksaan lulus, tanpa error JavaScript. Seluruh source yang ditampilkan direkonstruksi dari DOM dan cocok persis dengan source 30 blok. Diperiksa pula kesejajaran seluruh catatan, sepuluh tombol pemetaan screenshot, pencarian 99 bagian, pencarian dalam source, clipboard source utuh, scroll mouse, layar penuh native/fallback, serta viewport 390/768/1440. Pemeriksaan halaman utama sesudah pemuatan dilakukan setelah jaringan diblokir. Ini pemeriksaan panduan, bukan training atau evaluasi model baru.

Build: `outputs/dataset300_env_py313/bin/python scripts/build_e6_annotated_code.py`.
Pemeriksaan browser: `node scripts/check_e6_annotated_code.mjs`.
Bukti dan pratinjau privat: `outputs/e6_annotated/`.
