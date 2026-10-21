# Rencana eksperimen setelah S3 MLP

**Perkembangan 21 Oktober 2026:** prioritas 1 dan pembersihan checkpoint otomatis kini tersedia di [eksperimen double frozen + MLP terpisah](eksperimen_double_frozen_mlp.md). Colab utama/semi-final tetap. Rencana Ridge dan MLP pada S1 di bawah belum diterapkan.

Dokumen ini adalah rencana, bukan perubahan metode yang sudah diterapkan. Colab utama tetap memakai kode notebook MLP unggahan pengguna: S1 RoBERTa fine-tuned + head linear; S2 BGE-M3 frozen + MLP; S3 RoBERTa fine-tuned + BGE frozen + proyeksi, LayerNorm, concat dan MLP. Tidak ada metode yang diketahui pasti optimal sebelum diuji pada data sasaran.

## Sebelum pelatihan panjang: perbaiki penyimpanan

Penghapusan file melalui filesystem Drive dapat meninggalkan checkpoint di Sampah. Perbaikan penyimpanan perlu mempertahankan checkpoint terbaru dan satu cadangan yang sudah diverifikasi, serta menghapus permanen hanya checkpoint usang yang tercatat untuk run dan tahap yang sama. Setelah hasil fold atau model final tersimpan dan diperiksa, checkpoint sementara tahap tersebut dapat dibersihkan. File dataset, hasil CV, model final, Excel, run lain, dan checkpoint yang masih dirujuk harus tetap dilindungi. Permintaan API perlu menangani rate limit, error server sementara, dan file yang sudah terhapus. Perbaikan operasional ini telah diterapkan hanya pada notebook eksperimen baru; bukan metode penurun RMSE.

## Prioritas 1: double frozen + MLP pada S3

Pertanyaan: apakah pembaruan encoder RoBERTa membantu, atau justru membuat hasil lebih bergantung pada sedikit contoh latihan?

Pertahankan arsitektur S3 MLP saat ini, termasuk RoBERTa 768, proyeksi BGE 1024 ke 256, LayerNorm, concat 1024 dan MLP 1024 ke 256 ke 1. Ubah hanya kebijakan encoder RoBERTa menjadi frozen. Encoder BGE tetap frozen. Proyeksi, LayerNorm tambahan, dan MLP tetap dilatih. Ekstraksi fitur encoder frozen memakai mode eval agar dropout encoder tidak mengubah fitur; membekukan gradien saja tidak otomatis menonaktifkan dropout.

Frozen dibanding fine-tuning adalah jalur penggunaan representasi yang dibahas dalam BERT, tetapi bukan jaminan hasil pada AES ini. Rujukan: [Devlin et al. (2019), §5.3](https://aclanthology.org/N19-1423/). Ini hipotesis yang dipindahkan ke RoBERTa proyek, bukan replikasi identik eksperimen NER dalam paper.

## Prioritas 2: bandingkan prediktor pada fitur frozen yang sama

Bangun fitur tetap dari RoBERTa 768 dan BGE-M3 1024, concatenate menjadi 1792 komponen. Bandingkan:

- StandardScaler lalu MLP 1792 ke 256, GELU, dropout dan linear satu skor.
- StandardScaler lalu Ridge regression satu skor.

Kedua jalur menerima embedding yang sama dan pembagian fold yang sama. Scaler di-fit hanya pada training fold, lalu dipakai untuk validation fold. Nilai alpha Ridge dipilih dari daftar kandidat yang dinyatakan sebelum evaluasi, menggunakan data development. MLP dan Ridge adalah alternatif head; tidak ditumpuk secara otomatis. Jangan mengambil fitur proyeksi yang telah dilatih memakai validation fold lalu memasukkannya ke Ridge.

Perbandingan ini mengisolasi jenis head antara dua jalur baru. Perbandingan dengan S3 sebelumnya juga mengubah jalur fitur, karena BGE tidak lagi dipadatkan ke 256. Kenaikan hasil terhadap S3 lama tidak boleh diatribusikan hanya ke Ridge. Dense CLS ternormalisasi mengikuti [M3-Embedding, §3.2](https://arxiv.org/abs/2402.03216); paper tersebut tidak membuktikan bahwa Ridge atau arsitektur concat 1792 optimal untuk AES.

## Eksperimen pendamping: MLP pada S1

Bandingkan S1 saat ini dengan RoBERTa fine-tuned, mean CLS 768, linear 256, GELU, dropout, dan linear satu skor. Pertahankan konfigurasi lain untuk memeriksa kontribusi perubahan head. MLP menambah fleksibilitas dan parameter; ia tidak otomatis menghilangkan fluktuasi validation RMSE. Head linear tetap pembanding sah.

## Cara memilih hasil

Gunakan data development dan manifest lima fold yang sama; pisahkan pilihan arsitektur dari evaluasi akhir. Tetapkan kandidat, learning rate, aturan pemilihan epoch, dan metrik sebelum membandingkan hasil. Setelah kandidat disaring, pengulangan dengan beberapa seed inisialisasi dapat mengukur kepekaan hasil, dengan pembagian fold tetap. Tidak perlu langsung memperpanjang epoch, menambah GRU, mengganti pooling, dan mengganti head bersamaan.

Test pada run sebelumnya sudah dilihat. Perubahan selanjutnya tidak membuat test yang sama kembali menjadi holdout yang belum pernah memengaruhi pengembangan. Laporkan penggunaan ulang itu sebagai eksplorasi; untuk konfirmasi independen gunakan holdout baru atau rancangan nested cross-validation bila layak. Tujuan perbandingan adalah menemukan konfigurasi yang didukung data, bukan memastikan fusion selalu menang.
