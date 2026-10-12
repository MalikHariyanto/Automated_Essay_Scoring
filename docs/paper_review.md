# Pemeriksaan paper rujukan dari ZIP

Arsip `PAPER_RUJUKAN_MERGED_compressed.zip` berhasil diunduh (30.249.465 byte), CRC ZIP diperiksa, dan satu PDF di dalamnya diekstrak tanpa mengubah byte member. Nama member asli `PAPER_RUJUKAN_MERGED_compressed.pdf`; path proyek dibuat sesuai config: `project_sources/01-PAPER_RUJUKAN_MERGED.pdf`. PDF hasil ekstraksi berukuran 34.521.843 byte dan berisi **4.051 halaman**. Batas 32 MiB sebelumnya berlaku pada transfer file, sehingga ZIP ini menyelesaikan hambatan tersebut. Hash arsip/PDF tercatat di `outputs/source_review/review.json`.

Seluruh halaman diindeks dan teks diekstrak; **pembacaan detail difokuskan pada metode dan temuan paper yang relevan untuk implementasi**. Ini bukan klaim membaca setiap paragraf dari seluruh buku/artikel dalam 4.051 halaman. Ada 19 halaman tanpa teks yang dapat diekstrak; indeks tidak menyebut halaman tersebut sudah dibaca melalui OCR. Dua halaman teknis juga diperiksa secara visual: rumus dense BGE dan Tabel 7 Faseeh. Nomor di bawah selalu nomor halaman **PDF gabungan, berbasis 1**, bukan otomatis halaman cetak paper.

Indeks seluruh halaman: `outputs/source_review/paper_page_index.json`. Teks: `outputs/source_review/papers_extracted.txt`. Kutipan per artikel berada di `outputs/source_review/paper_excerpts/`. Batas halaman memakai objek PDF, bukan formfeed yang kebetulan muncul dalam isi teks; jumlah halaman diverifikasi terhadap `pdfinfo` dan parser PDF.

## Rujukan yang dicocokkan

| Rujukan | Lokasi PDF gabungan | Bagian yang digunakan |
| --- | --- | --- |
| Schröer et al., CRISP-DM (2021) | 1–9 | Alur data understanding/preparation/modeling/evaluation yang dapat berulang; kesiapan data mendahului eksperimen. |
| Liu et al., RoBERTa (2019) | 42–54 | Strategi pretraining, tokenisasi byte BPE, konteks 512, fine-tuning; identitas/dimensi checkpoint Indonesia tetap diperiksa dari config dan model card. |
| Albatarni & Elsayed, *QWK is Not Enough* (2026) | 85–94 | §3.2 metrik, §4 ukuran/distribusi/kategori, §5–6 rekomendasi pelaporan beberapa metrik. |
| Nadeau & Bengio (2003) | 388–430 | Pendahuluan dan penjelasan variansi evaluasi silang yang bergantung pada data latih serta contoh uji. |
| Faseeh et al., hybrid AES (2024) | 1108–1135 | Arsitektur penggabungan fitur/RoBERTa, §5 dan Tabel 7; bukan arsitektur atau angka hasil proyek ini. |
| Pack et al., validitas/reliabilitas LLM (2024) | 2853–2861 | Tujuan serta hasil studi reliabilitas; tidak mengotorisasi LLM sebagai pengisi skor guru. |
| Eltanbouly et al., TRATES (2025) | 3067–3082 | Abstrak dan §1–4: trait/rubrik, fitur LLM dan regresor; bukan algoritme otomatis untuk memberi label dataset kita. |
| Chen et al., M3-Embedding | 3132–3149; salinan lain mulai 2898 | §3.2 dense CLS yang dinormalisasi, retrieval multilingual/long context, batas transfer klaim. |
| García-Varela et al. (2025) | 3171–3196 | Abstrak dan bagian reliabilitas: rubrik dan kontrol penilaian penting, hasil LLM bukan validasi rubrik draf kita. |
| Chamidah et al., AES Indonesia (2023) | 4044–4051 | Abstrak dan §3–4: pasangan jawaban siswa–acuan, 2.157 jawaban, sentence tokenization, RMSE/Pearson. |

Daftar ini merupakan lokasi rujukan yang relevan, bukan inventaris seluruh buku dalam bundel. Versi M3 yang terlihat di PDF ialah **arXiv:2402.03216v5, 12 Desember 2025**; jangan mengklaim salinan ini identik dengan versi penerbit ACL 2024 hanya karena proposal mengutip Chen et al. (2024). Dua salinan judul tersebut ada dalam bundel; catatan implementasi memakai salinan mulai halaman 3132 dan tetap mengunci revision checkpoint secara terpisah.

## Keputusan implementasi setelah pembacaan

**BGE dense tetap CLS + L2.** §3.2 pada halaman 3135 menulis `e_q = norm(H_q[0])`, sesuai ekstraksi `dense_vecs` resmi yang sudah diuji. Paper membahas dense/sparse/multi-vector untuk retrieval, bukan empat skor rubrik. Dense retrieval dan kemampuan 8192 token tidak membuktikan validitas AES Indonesia dan tidak mengubah pilihan truncation 1024. Tidak menambahkan sparse/ColBERT, window BGE, atau tahap pretraining/contrastive loss paper ke proyek.

**RoBERTa tetap checkpoint Indonesia yang ditetapkan.** Paper umum RoBERTa mendukung dasar encoder dan fine-tuning, tetapi tidak menetapkan rubrik, empat target, model BGE fusion atau kualitas skor siswa lokal. Hidden 768 diperiksa pada checkpoint `flax-community/indonesian-roberta-base`, bukan diasumsikan dari semua varian RoBERTa. Koreksi tokenizer slow dari checkpoint sama tetap diperlukan berdasarkan temuan teknis runtime sebelumnya.

**Empat metrik tetap QWK, AC2 kuadratik, RMSE, Pearson.** Artikel Albatarni–Elsayed menunjukkan dampak ukuran data, distribusi dan kategori terhadap kestabilan ranking. Artikel juga membandingkan varian AC2 linear dan metrik lain; pembandingan itu tidak meminta penambahan metrik di tugas ini atau perubahan bobot AC2 yang sudah ditetapkan pengguna. Temuan jumlah contoh tertentu pada ASAP/ELLIPSE bukan batas minimum universal atau bukti kesiapan sepuluh esai draf. Per-trait, kategori tetap 1–5, prediksi kontinu untuk RMSE/Pearson, proporsi di luar rentang, dan undefined dengan alasan tetap berlaku. Backend AC2 sudah diverifikasi tersendiri terhadap R irrCAC pada fixture teknis; kode yang benar bukan bukti agreement penelitian tanpa skor guru.

**Ringkasan CV tetap bukan 25 observasi independen.** Nadeau–Bengio membahas variansi akibat pilihan train/test. Ini mendukung catatan bahwa fold yang berbagi data tidak bisa langsung dimasukkan ke uji statistik biasa seolah independen. Tidak menambahkan uji/eksperimen lain. Test tetap disimpan sampai konfigurasi terkunci.

**Rujukan hybrid tidak menjamin S3 unggul.** Faseeh memakai RoBERTa, fitur manual, LwXGBoost dan ASAP. Itu berbeda dari BGE frozen + RoBERTa fine-tuned serta head linear empat trait pada data kita. Angka paper tidak menjadi target performa atau hasil eksperimen proyek. Tabel 7 pada halaman 1127 memang memasangkan MSE dan RMSE yang tidak memenuhi hubungan `RMSE = sqrt(MSE)` jika dihitung dari residual dan agregasi yang sama; misalnya pasangan yang diterbitkan untuk LwXGBoost ialah MSE 0.071 dan RMSE 0.100, sedangkan akar 0.071 sekitar 0.2665. Pemeriksaan visual mengonfirmasi bahwa ini bukan semata kesalahan ekstraksi teks. Konteks agregasi paper tidak cukup untuk menyelesaikan ketidaksesuaian tersebut; tidak mengoreksi angka sumber secara spekulatif atau menyalinnya sebagai hasil kita.

**AES Indonesia rujukan bukan dataset pengganti.** Chamidah menggunakan jawaban siswa dan jawaban acuan, tiga penilai, skala berbeda, serta Siamese MaLSTM/SBERT. Hasil sentence tokenization yang mereka laporkan khusus untuk tugas tersebut; tidak membatalkan window token RoBERTa pada kontrak kita, tidak menambahkan reference answer sebagai fitur, dan tidak menyediakan label untuk CSV sepuluh esai kita.

**Rubrik/trait tetap berasal dari JSON.** TRATES memakai LLM untuk fitur berbasis rubrik dan regresi; Pack serta García-Varela meneliti validitas/reliabilitas LLM. Hasil-hasil itu adalah konteks literatur, bukan otorisasi menyimpulkan skor guru, membuat deskriptor baru, mengubah `confirmed`/`ready_for_training`, atau mengisi label dengan LLM. Perbedaan trait dan rubrik tiap paper dicatat tanpa mengubah empat trait 1–5 di proyek.

## Dampak pada proyek

Pembacaan ini memperkuat dan membatasi atribusi literatur; **tidak ada alasan yang kompatibel dengan kontrak untuk mengganti kode arsitektur, loss, data, split, atau metrik**. README, notebook, kontrak, status sumber, dan startup diperbarui untuk mencabut status paper terhalang. Semua empat sumber yang diminta kini tersedia dan digunakan sesuai prioritasnya; cakupan pembacaan PDF dicatat secara eksplisit.

Label guru tetap kosong dan rubrik belum disahkan. Pembacaan paper tidak menghasilkan model terlatih, metrik penelitian, atau eksperimen penuh. Bukti teknis sebelumnya tetap terpisah dari hasil penelitian.
