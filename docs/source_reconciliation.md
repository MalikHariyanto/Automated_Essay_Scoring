# Pencocokan proposal terbaru dengan kontrak aktif

Proposal dibaca dari salinan asli `project_sources/02-PROPOSAL-MALIK-HARIYANTO-REVISI-AES.docx`. Nomor di bawah adalah subbab/tabel/gambar dokumen, bukan nomor halaman Word yang bergantung rendering. SHA256 ada di `outputs/source_review/review.json`. Teks, sembilan tabel dan gambar metode 3.1–3.3 diperiksa. Paper merged sekarang tersedia melalui ZIP; 4.051 halaman diindeks dan bagian relevan diperiksa. Lihat `docs/paper_review.md` untuk lokasi halaman, temuan dan batas klaim.

Instruksi pengguna pada tugas implementasi tetap mengungguli dokumen yang baru dilampirkan. Lampiran ini tidak otomatis meminta migrasi protokol.

## Bagian yang cocok

- §1.4, §3.2.5, Tabel 3.2: empat target berurutan isi/organisasi/kebahasaan/mekanik, skor guru lengkap, nilai huruf hanya metadata; tidak mengisi target kosong dengan rata-rata/prediksi. Skala mengikuti rubrik guru. **DOCX terbaru tidak mewajibkan lima output atau skala 1–4**; pernyataan tentang versi lama pada prompt sebelumnya tidak diatribusikan ke file terbaru.
- §3.2.3: foto/tugas awal dengan nilai final belum menjadi dataset training multi-trait; rencana Google Form 300–400 belum membuktikan pengumpulan selesai. CSV sepuluh transkripsi draf tetap dilaporkan sesuai provenance aktual.
- §3.4.1: NFC/newline/artefak kontrol, preservasi ejaan/kapital/tanda baca/paragraf, tanpa stemming/stopword removal/case folding. Contoh teori §2.1.7 merapatkan spasi, tetapi metode §3.4.1 mempertahankan spasi yang relevan mekanik; kode mengikuti metode dan instruksi pengguna. Tidak ada bukti untuk penggabungan baris transkripsi.
- §3.4.2, §3.5.2–3.5.3: encoder BGE frozen/eval, dense 1024 L2, RoBERTa 768 fine-tuned, mean token awal setiap window RoBERTa tetap differentiable. Cache hanya encoder BGE, bukan projection/LayerNorm/head.
- §3.4.3, Gambar 3.2: 80% development/20% test, split esai/group, lima fold, seed split 2026, tanpa strata holistik. Seluruh skenario memakai indeks sama.
- §3.5.4: projection BGE Linear(1024,256)+LayerNorm, gabungan berdimensi 1024.
- Tabel 3.3–3.4: S2 MLP256/GELU/dropout .1. Tabel 3.4 juga merinci kandidat LR A/B, dropout .1, decay .01, clipping 1.0, AdamW, batch efektif delapan, warmup 10%, patience dua. Nilai ini kini **terverifikasi**, tidak lagi disebut detail yang hilang dari proposal.
- §3.8.1: RMSE/Pearson dari kontinu skala asli, ordinal nearest allowed category/tie rendah, semua kategori rubrik tetap, QWK dan AC2 kuadratik, undefined diberi alasan, proporsi di luar rentang dicatat. Empat metrik yang diizinkan tetap dipertahankan.

## Perbedaan yang diselesaikan oleh instruksi pengguna

1. **BGE panjang.** §3.4.2/Gambar 3.2 merancang window kedua encoder dengan overlap 128. Kontrak pengguna secara eksplisit memilih **BGE truncation 1024 termasuk special, tanpa window**; RoBERTa tetap window 512/overlap isi128. Model capacity BGE 8192 bukan batas yang dipakai tugas.
2. **Head.** §3.6/Tabel 3.3 memakai MLP256/GELU/dropout pada semua skenario; §3.5.4/Gambar 3.3 juga memperlihatkan hidden MLP S3. Kontrak pengguna memilih **S1 dropout+Linear(768,4)** dan **S3 dropout+Linear(1024,4)** setelah projection/concat. Tidak menambahkan hidden S3. S2 tetap MLP256. Head S1 kini dijelaskan sebagai pilihan eksplisit pengguna yang berbeda dari proposal terbaru, bukan asumsi akibat head tidak dirinci.
3. **Loss.** §3.7.1 menormalisasi target ke 0–1. Kontrak meminta **MSE mentah pada empat label 1–5** dengan bobot sama. Kode tidak menormalisasi target dan tidak clip output di loss. Asumsi jarak ordinal yang setara untuk MSE/RMSE tetap perlu dibahas.
4. **Monitor dan hasil.** Monitor galat ternormalisasi pada §3.7.2/§3.8 diganti oleh **mean QWK empat trait**, maximize, fallback validation MSE yang ditetapkan sebelum run. Metrik yang dilarang pengguna tidak dihitung, tidak masuk tabel/grafik hasil, dan tidak ditambahkan sebagai backend. Source asli tidak diedit.
5. **Epoch.** §3.7.2/Tabel 3.4 merancang maksimal **10 epoch**; kontrak tetap **5**, patience dua. Dokumentasi tidak lagi menyebut lima sebagai angka yang terbaca di proposal terbaru.
6. **Tuning dan CV.** §3.7.2 menguji dua kandidat pada kelima fold dengan seed42, lalu memakai fold tetap dengan seed inisialisasi2026/3407. Kontrak tetap tuning **fold pertama repeat pertama**, lalu **lima fold × lima repeat** dengan seeds [42,123,2026,3407,9999]. Jangan mengklaim ini identik dengan rancangan tiga seed proposal. Tuning fold masuk ringkasan CV; ketergantungan fold tidak diperlakukan sebagai observasi independen.
7. **Final.** §3.7.3 memakai median best epoch berbatas1–10, tiga final seed dan ensemble kontinu. Kontrak tetap ceil(median best_epoch CV), batas1–5, **satu final refit seed42**, tanpa ensemble otomatis. Test baru dievaluasi setelah konfigurasi dikunci.
8. **Hipotesis dan analisis tambahan.** §3.8.2 membahas perbandingan pertrait, bootstrap test2000 dan ensemble. Tidak ada ambang holistik khusus yang ditemukan pada versi ini. Tidak mengadaptasi ambang versi lama menjadi klaim pertrait, menambahkan bootstrap/eksperimen lain, atau mengklaim fusion unggul.

## Koreksi implementasi yang tetap kompatibel

§3.3 menetapkan distribusi skor test dilaporkan setelah konfigurasi selesai ditentukan. Manifest sekarang menampilkan distribusi development/train/validation saja; distribusi test disimpan saat evaluasi final. Seluruh konfigurasi dan final refit skenario yang diminta diselesaikan sebelum evaluasi test pertama, sehingga laporan S1 tidak terbuka saat S2/S3 masih dituning. Pemeriksaan eligibility seluruh label sumber tetap berlangsung untuk memastikan target sah; label test tidak menjadi dasar tuning.

## Paper yang sudah tersedia

`PAPER_RUJUKAN_MERGED_compressed.zip` menyelesaikan hambatan transfer. PDF hasil ekstraksi disimpan di path yang sudah configurable; isi member ZIP tidak diubah. Pembacaan bagian teknis relevan, indeks halaman, hash, versi arXiv M3, serta pemeriksaan tabel/rumus dicatat di `docs/paper_review.md` dan `outputs/source_review/review.json`. Tidak ada migrasi kontrak atau penggantian dataset karena paper dilampirkan.
