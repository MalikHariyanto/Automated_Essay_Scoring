# Memperbaiki pembersih Drive tanpa mengganti run training

> Dokumen tahap terdahulu. Angka dan keputusan dibaca dalam konteks versi ini; untuk kondisi aktif gunakan [status proyek](../../STATUS_PROYEK.md) dan [panduan AI](../../PANDUAN_AI_PROYEK.md).

Log `Metadata Drive belum cocok dengan hasil lokal: progress_fold_1.json` berarti penunjuk progres di server belum sama dengan versi lokal. Pada laporan pengguna, training berhasil memulihkan epoch 10 dan menghitung epoch 11; proses kemudian berhenti saat verifikasi pembersihan. HTTP 500/503 juga menunjukkan kegagalan layanan Drive yang terpisah dari perhitungan model.

Pembersih sebelumnya menuntut kedua penunjuk identik sebelum melanjutkan. Versi perbaikan dapat menerima penunjuk server yang tertinggal jika seluruh rujukannya sah. Ia melindungi dua snapshot terbaru dan paling banyak dua snapshot yang masih dirujuk server. Ketika server tersinkron, versi usang dilepas. Verifikasi scope, tahap, ukuran, checksum, izin dan lokasi tetap berlaku. Bukti hasil CV/model final tetap wajib cocok dengan server; file rusak atau respons API gagal tidak dianggap berhasil.

## Untuk runtime yang sekarang masih tersambung

Pastikan cell training telah berhenti merah, lalu tambahkan **satu cell Python biasa, tanpa `%%aes`**, dan jalankan:

```python
from urllib.request import urlopen

# Ambil hanya kode pembersih publik. Data/model tidak dikirim ke GitHub.
url = "https://raw.githubusercontent.com/MalikHariyanto/Automated_Essay_Scoring/refs/heads/codex/aes-learning/src/aes/teacher_score/drive_checkpoint_cleanup.py"
source = urlopen(url, timeout=60).read().decode("utf-8")
exec(compile(source, "<perbaikan-pembersih-drive>", "exec"), globals())

# Ganti objek pembersih di kernel; worker training dan konfigurasi tetap.
_aes_cleaner = DriveCheckpointCleaner(_aes_cleaner.api, _aes_cleaner.root)
print("Pembersih diperbarui. Jalankan ulang cell training yang merah.")
```

Setelah pesan muncul, jalankan ulang **cell training yang sama**. Tidak perlu menjalankan setup atau konfigurasi ulang. Antrean cleanup dicoba kembali, lalu checkpoint terbaru yang valid dipakai untuk resume. Epoch 12 hanya dapat dimulai jika checkpoint epoch 11 berhasil dipulihkan; cadangan dipakai jika checkpoint terbaru tidak utuh.

Cell di atas mengganti pembersih pada kernel notebook. Ia tidak mengubah kode worker, fingerprint, folder hasil, model, learning rate, epoch, bobot atau pengacak. Membuka notebook versi baru untuk melanjutkan run lama justru menghasilkan fingerprint baru; gunakan perbaikan kernel ini selama runtime lama masih tersedia. Versi notebook baru sudah membawa pembersih perbaikan untuk run baru.

Jika runtime sudah hilang, gunakan salinan notebook/config/perangkat yang sama dengan run semula untuk memulihkan identitas run, lalu pasang perbaikan pada kernel sebelum cell training. Perbaikan ini tidak merekonstruksi checkpoint yang sudah hilang dan tidak menghilangkan gangguan server Google Drive. HTTP 500/503 yang terus muncul tetap dapat menghentikan proses setelah retry; tidak ada file yang dihapus tanpa bukti.

## Bukti teknis

Pengujian simulasi mencakup penunjuk server lama yang tetap bisa dipakai memuat checkpoint, training yang terus menyimpan epoch baru sambil mempertahankan rujukan server, pembersihan setelah sinkron, serta penolakan checksum salah, rujukan hilang/di Sampah, tahap salah, snapshot server lebih baru dan penunjuk berubah saat penghapusan. Pengujian ini tidak mengakses akun Drive pengguna. [Laporan verifikasi](../verifikasi/verification_double_frozen_mlp.md).
