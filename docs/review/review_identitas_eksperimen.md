# Identitas eksperimen E21–E30

Buku tiga input memiliki tab **07 · Identitas & sitasi**. Sepuluh kartu memperlihatkan tujuan, eksperimen acuan, alur S3, input/label, format, head, kelompok LR, tautan notebook yang dipin, dan sitasi per eksperimen. Pencarian dan filter membedakan delapan hasil tersedia dari dua resep yang menunggu hasil.

E29/E30 ditambahkan atas permintaan pengguna sebagai **identitas dengan ruang kode/output kosong**. Notebook sumber sudah dipublikasikan secara terpisah, sehingga tautannya tetap dapat dibuka. Kode keduanya belum diembed ke meja belajar dan ZIP hasil belum diterima. Ruang kosong tidak diisi nol, angka ilustrasi, atau output sintetis.

| Identitas baru | Acuan | Perubahan | Hasil dalam buku |
| --- | --- | --- | --- |
| E29 | E26 | LR MLP S2/S3 menjadi5e-6; linear S1 dan layer lain tetap | Belum tersedia |
| E30 | E26 | Co-attention respons–referensi lalu attention pooling | Belum tersedia |

Katalog `identities` terpisah dari `comparison.experiments` dan arsip privat. Karena itu E29/E30 belum masuk metrik, ketepatan, duel, kurva, peta skor, ataupun peringkat. Seluruh objek perbandingan dan delapan dokumen anak dibandingkan dengan commit `7a76191be20604482fa53f770699563b380120ba`: tetap identik. Kode utama meja belajar tetap E23. Aplikasi MLP lama tidak diubah.

## Sitasi dan batas dukungan

RoBERTa dan BGE-M3 ditautkan pada seluruh kartu. Komponen khusus ditautkan pada eksperimen yang memakai komponen tersebut: GELU/LayerNorm, BiGRU dan attention AES, gated fusion, atau co-attention. Deskripsi membedakan adaptasi tingkat token proyek dari arsitektur paper yang berbeda. Tidak mengklaim paper membuktikan kombinasi ini lebih unggul atau menetapkan LR5e-6 sebagai nilai terbaik. Format pair dan pengubahan LR diberi label ablasi proyek.

Sumber utama: [RoBERTa](https://arxiv.org/abs/1907.11692), [BGE-M3](https://arxiv.org/abs/2402.03216), [GELU](https://arxiv.org/abs/1606.08415), [LayerNorm](https://arxiv.org/abs/1607.06450), [Dong et al.2017](https://aclanthology.org/K17-1017/), [PaSTO-GNN2026](https://doi.org/10.3389/frai.2026.1842542), [GMU2017](https://openreview.net/forum?id=Hy-2G6ile), [Zhang & Litman2018](https://aclanthology.org/W18-0549/). Pembaruan ini menata sumber yang telah digunakan; tidak mengklaim pencarian/full-text review baru.

## Antarmuka dan privasi

Kartu dua kolom pada desktop menjadi satu kolom pada tablet/HP. Status dan asal eksperimen ditulis langsung, alur memakai panel kontras, rincian LR memakai pasangan label–nilai, sitasi dapat dibuka per kartu. Ruang kosong E29/E30 memiliki judul di luar area kosong; tidak memakai tombol mati. Tab mendukung keyboard dan layar penuh perbandingan yang sudah ada.

HTML publik hanya berisi metadata, agregat, serta source edukasi yang sudah aman. Data individual tetap di buku privat terenkripsi dengan kunci lama; delapan ZIP/40 Excel dipertahankan byte demi byte. Tidak ada training penelitian baru atau perubahan metode dalam tugas antarmuka ini.
