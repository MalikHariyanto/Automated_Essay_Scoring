# Penataan repo dan serah-terima akun · 28 Oktober 2026

Penataan ini menambahkan jalur masuk dan konteks lintas akun, tanpa menjalankan training atau mengganti model/hasil eksperimen.

## Jalur masuk baru

- [README utama](../../README.md): pilihan notebook, HTML, status, dan serah-terima.
- [Panduan AI](../../PANDUAN_AI_PROYEK.md): sintesis keputusan percakapan dari MLP awal sampai E26, dengan hubungan eksperimen, batas bukti, kebutuhan belajar/HTML, serta hal yang belum diimplementasikan.
- [Status proyek](../../STATUS_PROYEK.md): kontrak aktif dan angka hasil.
- [Pindah akun](../../PINDAH_AKUN.md): chat/Codex, GitHub, Colab, berkas privat, dan prompt singkat.
- [Rencana berikutnya](../../RENCANA_EKSPERIMEN.md): LR MLP dan co-attention sebagai jalur terpisah; belum merupakan notebook/hasil baru.

## Susunan dan kompatibilitas

138 dokumen/snapshot dikelompokkan ke `docs/eksperimen`, `docs/review`, `docs/verifikasi`, `docs/snapshots`, dan `docs/arsip`. Index README menjelaskan peran setiap folder; laporan lama diberi konteks versi agar tidak ditafsirkan sebagai kondisi terbaru.

Referensi filesystem pada pembuat buku dan pemeriksa diperbarui. Referensi **commit:path** pada `git show` dipertahankan karena menunjuk lokasi historis yang benar. Lima alamat panduan yang ditautkan notebook lama tetap tersedia sebagai penunjuk ke dokumen yang dipindahkan.

Jalur notebook/Colab, dua aplikasi HTML, pengunduh dan seluruh bagian paket tetap. Tidak ditambahkan aplikasi HTML ketiga. Buku tiga input berhasil dibangun ulang dari snapshot yang dipindahkan, dengan byte HTML dan unduhan tetap sama.

## Notebook dan salinan lama

27 source notebook utama dipertahankan. Lima notebook yang masih mempunyai output/count eksekusi dibersihkan pada versi terbaru. Empat mirror lama mempunyai source berbeda; source mirror tersebut dipertahankan pada `notebooks/archive/mirror_variants/` sebelum23mirror disamakan dengan jalur utama. Seluruh54notebook pada tree terbaru tidak mempunyai output eksekusi.

Salinan privat sebelum pembersihan tersimpan hanya lokal pada lingkungan kerja saat penataan. Riwayat commit lama tidak ditulis ulang. Penataan tidak mengklaim menghapus data dari seluruh riwayat atau memindahkan akun secara otomatis.

## Bukti yang dipertahankan

- Kedua HTML, kedua pengunduh dan enam bagian paket: **10 berkas byte-identik** dengan versi sebelum penataan.
- Seluruh55JSON snapshot/bukti yang dipindahkan: byte-identik.
- Source27notebook utama: identik pada cell source; source varian mirror yang berbeda tetap diarsipkan.
- Seluruh23mirror kini identik dengan jalur utama.
- Tidak ada perubahan LR, layer, label, prediksi atau nilai guru.

Pemeriksaan subset integritas buku/snapshot lulus **30 tes**. Syntax Python diperiksa, tautan lokal dan anchor Markdown diperiksa, serta rebuild buku tiga input diuji. Bukti command, versi, mapping pemindahan dan hasil disimpan lokal pada `outputs/repo_handoff/`. Tidak diklaim seluruh checker historis atau training pretrained GPU diulang.
