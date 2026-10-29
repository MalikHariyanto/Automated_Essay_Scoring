# Riwayat percakapan dan progres proyek AES

Diperbarui29Oktober2026. Ini ringkasan kronologis keputusan dari percakapan yang tersedia, bukan salinan verbatim seluruh chat. Tidak mengarang pesan/tanggal yang tidak tersedia. Pertanyaan berulang dan salah ketik diringkas; keputusan yang digantikan tetap dijelaskan agar AI berikutnya tidak mengaktifkan rencana lama.

## Perjalanan eksperimen

1. Pengguna membandingkan RoBERTa, BGE-M3 dan fusion melalui S1/S2/S3. Eksperimen awal memakai5epoch dan head linear. Lima fold berbeda dari jumlah epoch.
2. MLP awal menjadi acuan presentasi: maksimal25epoch,patience5. S1 tetap linear; S2/S3 memakai MLP. Pengguna meminta pembacaan skor guru/prediksi, tokenisasi, vektor dan penjelasan kode.
3. Double frozen+MLP dibuat terpisah sebagai eksperimen. Pembersihan checkpoint otomatis menimbulkan masalah sinkronisasi Drive; pengguna meminta menghapus fitur auto-delete dan melakukan pembersihan manual. Jangan menyimpulkan semua `.pt` aman dihapus saat training masih aktif.
4. Konteks pair, LoRA dan perubahan LR diuji terpisah. LoRA adalah salah satu cara fine-tuning; bukan mengganti encoder menjadi MLP. Konflik pustaka dan perpindahan GPU/CPU ditangani sebagai masalah teknis, bukan hasil ilmiah.
5. E6 menguji encoder S3 5e-6, S1 tetap1e-5 pada keluarga lama; proyeksi/head1e-4. E8 menggabungkan konteks pair dan LR tersebut. Hasil tiga seed menunjukkan satu run bagus belum membuktikan konsistensi. Jangan mencampur keluarga lama dengan keluarga tiga input.
6. Pengguna meminta tiga input pada semua skenario: `soal`, `jawaban_referensi`, `response_text`; `rater` sebagai satu label0–100. E21 memakai single sequence; E22 memakai context pair. Kontrak ini menggantikan asumsi lama untuk keluarga aktif; bukan empat label trait.
7. E23 menambah BiGRU pada S3; E24 attention pooling; E25 attention+gated fusion; E26 BiGRU lalu attention pooling. BGE tetap frozen, RoBERTa fine-tuned. Head S1 linear; S2/S3 MLP.
8. Ketika pengguna memprioritaskan jumlah prediksi mendekati guru, E23 dipilih sebagai dasar E27/E28: E27 co-attention respons–referensi, E28 hanya LR MLP S3 5e-6. E27 memakai ringkasan ujung, bukan additive pooling E26.
9. Pengguna kemudian kembali memilih E26 sebagai dasar. E29 menurunkan LR MLP S2/S3 ke5e-6; linear S1 dan layer lainnya tetap1e-4. E30 menambah co-attention sebelum attention pooling, dengan MLP1e-4. Encoder RoBERTa S1/S3 5e-6. Kedua hasil sudah diunggah dan diaudit.

Nomor E9–E20 tidak direkonstruksi dari dugaan. Detail historis yang tersedia tetap mengikuti [panduan AI](../../PANDUAN_AI_PROYEK.md) dan review tiap eksperimen.

## Perjalanan aplikasi belajar

Pengguna meminta HTML bertema Spider-Man, source lengkap, output aktual dan sticky note sejajar dengan baris penting. Fullscreen harus menutupi viewport, menyatukan kode/catatan dan menghindari dua scroll vertikal. Mode semua blok maupun per blok tetap tersedia.

Tabel perbandingan diubah agar eksperimen menjadi kolom. Ditambahkan metrik, toleransi, seluruh jawaban, peringkat, duel pasangan, kurva/fold dan peta skor. Jumlah terdekat harus dihitung ulang ketika peserta bertambah; prediksi lama tidak boleh berubah. Terdekat bukan skor persis sama, dan toleransi5poin bukan exact accuracy.

Dua aplikasi aktif tetap dipertahankan: keluarga lama dan keluarga tiga input. File hasil individual hanya berada dalam paket privat terenkripsi; preview publik berisi agregat/source aman. Pengunduh dipublikasikan di GitHub karena tautan unduhan chat bermasalah; kunci lama tetap digunakan.

Sebelum hasil E29/E30 tersedia, pengguna meminta identitas/sitasi dengan ruang kode/output kosong. Setelah ZIP masuk, placeholder diisi hasil aktual. Seluruh sepuluh identitas sekarang memiliki hasil.

Pengguna menanyakan Laravel karena HTML berat. Dijelaskan bahwa ukuran terutama berasal dari embedding ZIP/Excel dalam satu berkas; framework saja tidak otomatis mengurangi payload. Pengguna memilih tetap HTML dan membatasi meja belajar menjadi **tiga kode dengan RMSE test S3 terendah**. Kini kode yang diembed adalah **E29,E30,E26**; utamaE29. Kode eksperimen lain tersedia lewat tautan notebook; seluruh50Excel/458sheet tetap dapat dibaca pada tab08SemuaExcel.

## Hasil terbaru dan batas kesimpulan

| Ukuran S3 | E29 | E30 |
| --- | ---: | ---: |
| RMSE test · poin | **5,4293** | 5,5247 |
| MAE test · poin | 4,1713 | **4,0215** |
| Mean RMSE CV | **5,7667** | 5,9491 |
| Selisih≤5poin | 40/60 | **44/60** |
| Selisih≤10poin | 55/60 | **56/60** |
| Terdekat di antara10peserta | 10/60 | **13/60** |

E29 tetap terendah RMSE; E30 unggul MAE/toleransi/jumlah terdekat. Keduanya0/60 persis pada skor mentah. E29 S2 justru RMSE75,9702; hasil buruk ini tetap ditampilkan. E30 S1/S2 sama per jawaban dengan E26. Tidak menyatakan satu eksperimen terbaik untuk semua ukuran.

Sembilan snapshot/prediksi dan36duel lama tetap utuh saat E30 ditambahkan. Kini10eksperimen/30model/45duel pada60jawaban test yang sama. Test sudah sering dilihat: hasil lanjutan bersifat eksploratif, belum konfirmasi seed/test baru. ZIP tidak menyertakan executed notebook/checkpoint untuk membuktikan identitas kode runtime/perangkat. Audit unggahan dan tes aplikasi bukan training ulang.

## Titik lanjut bagi akun/AI berikutnya

- Baca [status aktif](../../STATUS_PROYEK.md), [panduan AI](../../PANDUAN_AI_PROYEK.md), [AGENTS](../../AGENTS.md), dan [pindah akun](../../PINDAH_AKUN.md).
- Versi aplikasi hasil terbaru dipublikasikan pada commit `6132421663a992c9df1b33d9da533da399a74661`; [review E30](../review/review_e30_study.md) dan [verifikasi](../verifikasi/verification_e30_study.md).
- [Pengunduh tiga input](../../downloads/Belajar_AES_Tiga_Input_Lengkap.html) memakai kunci lama, dibawa secara privat. Jangan mempublikasikan teks/skor individual, ZIP asli, checkpoint atau kunci.
- Notebook sumber E29/E30 dipin ke `7a76191be20604482fa53f770699563b380120ba`. Source E26 dipin ke `bc4713872c7a7ce23f7dad8ae3ea73a2b75d9b86`.
- Ensemble belum dibuat; E31/E32 kini resep baru berbasis E25 sebagaimana pembaruan terakhir di bawah. Tiga seed dan rata-rata prediksi masih usulan; rata-rata metrik bukan ensemble. Jangan memulai eksperimen atau mengganti metode tanpa instruksi baru.

## Keputusan lanjutan setelah E30: kembali ke E25

Pengguna meminta perlakuan E29/E30 diulang dari source E25 karena unggul pada pembulatan dan toleransi1poin. Dibuat kode baru E31 (E25 + LR MLP S2/S3 5e-6) dan E32 (E25 + co-attention sebelum pooling/gate), bukan menimpa E29/E30. E32 tidak menambah BiGRU; matching640→768 menjaga dimensi pooling E25. Kedua resep GPU, tiga input, label rater, tanpa Drive; belum memiliki hasil penelitian. Ini menggantikan pernyataan historis di atas bahwa nomor E31 belum dibuat; E31 sekarang ablasi LR dari E25, bukan ensemble. [Metode](../eksperimen/eksperimen_e31_e32_gpu.md).
