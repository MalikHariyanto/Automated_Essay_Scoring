# E24C · Hasil pilot ekstraksi bukti

[Ringkasan numerik dan checksum](Ringkasan_Pilot_Ekstraksi_Bukti.json). Inferensi mDeBERTa QA frozen pada 107 jawaban development soal 10 dengan split E21.

Enam dari delapan probe lolos. Tiga dari empat probe kritis lolos. Probe lokasi pembanding gagal karena model memilih lokasi yang disebut sebagai kebiasaan dan melewatkan lokasi kejadian utama. Probe lokasi di ujung teks panjang juga gagal. Kolom audit buta untuk 15 jawaban masih kosong.

| Pilot | Label isi 1 ditandai | Label isi 2–4 ikut ditandai |
|---|---:|---:|
| E24 · dukungan lokasi rendah | 2/8 | 2/98 |
| E24B · minimal satu unsur dengan dukungan rendah | 8/8 | 97/98 |
| E24C · model mengembalikan minimal satu bukti kosong | 1/8 | 9/99 |

E24 dan E24B memakai 106 jawaban utuh, sedangkan E24C memakai 107 jawaban. Keluaran kosong dan skor dukungan rendah merupakan dua aturan audit yang berbeda. Potongan terisi masih dapat salah konteks. Hitungan ini belum mengukur ketepatan ekstraksi pada development atau performa scoring S3.

**Keputusan.** Hentikan jalur ekstraksi ini sesuai kriteria pilot dan pertahankan E21-B sebagai acuan test S3. E23 bobot isi 2× tetap kandidat development dengan deteksi 2/17 pada dua seed dan penurunan QWK kebahasaan. Pilot E24C tidak membuka test atau melatih S3. Jawaban dan kutipan siswa tetap privat.
