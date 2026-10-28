# Audit kerja AES, pengujian, dan publikasi

Audit ini memeriksa ulang mesin tugas; laporan sebelumnya tidak dijadikan bukti tunggal. Tanggal audit 12 Oktober 2026, waktu pengguna Asia/Bangkok. Detail command/versi/outcome publik ada di `docs/verifikasi/audit_summary.json`. Log privat dan notebook yang memuat teks tetap di `outputs/publication_audit/`, tidak masuk commit publik.

## Mengapa GitHub belum berubah pada audit awal

Direktori kerja utama `/workspace/Automated_Essay_Scoring` adalah root Git yang benar; remote `origin` tersanitasi ialah `https://github.com/MalikHariyanto/Automated_Essay_Scoring.git`. Branch `work` **unborn** (belum ada commit pertama). `git status --short --branch --untracked-files=all` menunjukkan seluruh file proyek masih untracked; tidak ada staged/modified tracked files. `git log -5 --oneline --decorate` exit128 dengan pesan belum punya commit; `git diff --stat` dan `git diff --cached --stat` kosong bukan berarti source belum ada, tetapi karena belum ada file tracked/staged. Hanya checkout ini muncul pada `git worktree list`; tidak ada worktree lain untuk kode proyek.

`git ls-remote origin` exit0 dan nol refs. Jadi keadaan remote kosong diperiksa ulang, bukan disimpulkan dari snapshot lama. File kode dan hasil lokal sudah ada, tetapi tidak pernah menjadi commit/push. Menyimpan draft lingkungan instalasi/startup tidak sama dengan menyimpan source di GitHub.

Di luar root Git terdapat `/workspace/aes-venv`, `/workspace/aes-r-local`, `/workspace/aes-apt`, `/workspace/hf-cache`, `/workspace/jupyter` serta lampiran `/workspace/attachments`. Itu dependency/cache/kernel/input yang dipakai tugas; bukan branch/worktree kode tersembunyi. File laporan lama berada di `outputs/` dalam root Git tetapi untracked. Seluruh input/hasil privat tetap lokal.

## Apa yang sebenarnya diuji dalam klaim 26

Runner ditemukan sebagai **pytest**. Log `outputs/unit_tests.log` dan `outputs/source_review/tests_after_review.log` menyatakan `26 passed in 9.45s`. Riwayat percakapan memuat command pytest dari root proyek, dan file test terkait ditemukan. Akan tetapi, source saat itu belum memiliki commit; log lama tidak mengikat hasil ke SHA Git dan tidak membuktikan remote berubah.

Sebelum mengedit, pengujian diulang pada source untracked yang ada: collection **26 kasus**, run **26 passed, 0 failed, 0 skipped, 0 xfailed, 0 xpassed**, exit0. Versi dan hash file source sebelum edit direkam. Bukti collection dan node IDs/outcomes ada di `docs/verifikasi/baseline_collection.txt` serta `baseline_tests.txt`; identitas source di `original_untracked_source_hashes.json`. Hash adalah sidik isi file, bukan commit historis yang tidak pernah ada.

Kasus tersebut berasal dari fungsi test yang sebagian **parametrized**: satu fungsi dijalankan dengan beberapa ukuran batch/skenario. Angka 26 bukan jumlah esai, epoch, fold penelitian, atau model tervalidasi.

| Kelompok | Input/bukti | Makna dan batas |
| --- | --- | --- |
| Loader/audit, cleaning, windows, mapping, loss, metrics | Baseline memakai input privat untuk audit draft; aritmetika dan token/model memakai fixture teknis | Memeriksa missing/version/range, cakupan token, `[B,4]`, larangan broadcasting, QWK scikit-learn dan kebijakan ordinal; bukan akurasi AES |
| Model/gradien/reload | Encoder kecil tiruan, vektor BGE buatan, target teknis | Memeriksa wiring S1/S2/S3 dan perubahan parameter; tidak memuat checkpoint asli |
| Trainer dan jadwal eksperimen | Tokenizer/encoder/training tiruan, label fixture | Effective batch/optimizer steps, fallback, dan test setelah refit semua skenario; tidak menjalankan 25 training penelitian |
| AC2 | Lima pasangan kategori fixture, R 4.5.0 / irrCAC1.4 | Point estimate cocok dengan backend AC2 hingga pembulatan R; bukan agreement siswa |
| Smoke terpisah | Checkpoint BGE/RoBERTa asli, teks CSV lokal, target3 teknis saja | Model asli dimuat; forward batch1/batch2, backward, dan save/reload dijalankan di CPU |
| Notebook terpisah | nbclient, kernel baru, CSV/JSON asli lokal | Delapan cell kode dieksekusi tanpa error, mode inspect; bukan pemeriksaan sintaks saja |

Smoke asli mengunci BGE `5617a9f61b028005a4858fdac845db406aefb181` dan RoBERTa `cc5ecee7de8532880ad08d802292a6f42865bc83`. S1/S2/S3 menghasilkan `[1,4]` dan `[2,4]`. Parameter trainable yang diperiksa berubah pada kelompok yang sesuai; BGE seluruh `requires_grad=False`, dan satu parameter perwakilan encoder dibandingkan tetap. Ini tidak mengklaim setiap parameter dibandingkan byte demi byte. Target 3 hanya memancing gradien; tidak disimpan sebagai skor guru. Checkpoint teknis sementara dihapus sesudah test; tidak tersedia model penelitian terlatih.

## Perbaikan agar clone publik dapat diuji

Test lama yang bergantung pada CSV/JSON penelitian dialihkan ke fixture teknis yang dibuat di direktori sementara, dengan deskriptor berlabel fixture, bukan salinan rubrik guru. Fungsi/aturan yang diperiksa tetap sama, sedangkan pemeriksaan input asli tetap dilakukan terpisah. Dokumen proposal bukan prasyarat unit test path configurable. Tidak membuat data penelitian alternatif.

Dua test bermakna ditambahkan: pesan yang menjelaskan cara menyediakan input privat ketika clone belum memiliki CSV/JSON, serta routing inference CLI yang memakai rubrik tertanam dalam checkpoint tanpa memerlukan CSV/JSON lokal. Test routing menggunakan fungsi predict tiruan; ia tidak mengklaim memuat checkpoint atau menghasilkan skor sah.

Suite setelah perubahan: **28 kasus, 28 passed, 0 failed, 0 skipped, 0 xfailed/xpassed** pada lingkungan dengan R tersedia, exit0. Log publik di `docs/verifikasi/public_tests.txt`. Salinan terisolasi dari staged files, tanpa data/ dan project_sources/, juga menghasilkan 28 passed dengan R tersedia. Tanpa binding dan executable R, hasilnya 27 passed + 1 skipped, exit0; alasan skip terlihat. Probe terpisah dengan binding R yang sengaja salah menghasilkan 1 failed + 27 passed: itu kesalahan konfigurasi yang benar-benar dilaporkan, bukan diubah menjadi skip/passed. Probe negatif tersebut bukan hasil suite final yang dikonfigurasi benar. Versi sumber yang diuji dicatat sebagai hash source sebelum commit; pemeriksaan setelah commit akan memakai SHA Git yang sebenarnya.

## Data dan tahap yang belum tersedia

CSV saat audit berisi 10 baris; semua empat kolom skor kosong. Rubrik `confirmed=false`, seluruh `ready_for_training=false`, transkripsi draf, identitas belum diverifikasi, empat group masih usulan, dan dua teks continuation belum dipasangkan. Hash input sebelum/sesudah setup/publikasi diperiksa agar sumber tidak berubah.

Contohnya, empat angka dari head baru tidak bisa dibandingkan dengan nilai guru jika kolom guru kosong. Loss fixture yang turun hanya memperlihatkan mekanisme dapat belajar angka buatan; tidak membuktikan kualitas isi/organisasi/kebahasaan/mekanik siswa. Training supervised, evaluasi pada data berlabel dengan split sah, dan eksperimen proposal penuh **belum dijalankan**. GPU/AMP dan penanganan OOM nyata juga belum diuji di mesin CPU ini.

## Penyimpanan ke repository

Branch hasil `codex/aes-learning` disiapkan sebagai rekaman awal. Tidak merge atau push ke main, tidak force push, tidak membuat PR terhadap base yang belum ada. Staging menggunakan daftar source/config/schema/notebook tanpa output/tests/docs; semua data dan hasil privat dikecualikan. `.gitignore` ditambah dan isi staged diperiksa, bukan hanya mengandalkan ignore.

Source commit `3f4aa3af102bed8d067ec0f9b02069438cd864b5` sudah dibuat dan push berhasil. `git ls-remote` dari jaringan mengembalikan SHA sama; independent bare fetch membaca README, AGENTS, model, notebook dan test dengan bytes yang sama. URL publik raw GitHub untuk README/AGENTS/model juga memberikan HTTP200 dan cocok. Receipt ada di `docs/verifikasi/remote_verification.json`. Commit dokumentasi sesudahnya menyimpan receipt dan hasil test source commit ini; source/tests/config/notebook tidak berubah dalam commit dokumentasi tersebut. SHA head terakhir diperiksa kembali setelah push terakhir dan disampaikan pada jawaban.

Source commit juga diuji dari `git archive` yang tidak memiliki data privat: 28 passed, exit0, sumber terikat ke SHA di `docs/verifikasi/post_commit_checks.json`, bukan hanya folder lokal tanpa versi. Branch: https://github.com/MalikHariyanto/Automated_Essay_Scoring/tree/codex/aes-learning . Source commit: https://github.com/MalikHariyanto/Automated_Essay_Scoring/commit/3f4aa3af102bed8d067ec0f9b02069438cd864b5 . Tidak ada main yang dibuat/di-merge dan tidak ada PR yang diklaim. Receipt publik mengacu ke source commit yang sudah diperiksa, bukan klaim SHA self-referential dari file receipt sendiri.

## Cara mengulang

Baca README untuk instalasi. `python -m pytest --collect-only -q` menampilkan kasus, `python -m pytest -q -rA` menampilkan outcomes. Isi CSV/JSON asli secara lokal lalu jalankan inspect/notebook; `scripts/smoke_real.py` memakai checkpoint asli, terpisah dari unit test. Gunakan `git log` untuk rekaman lokal dan `git ls-remote origin refs/heads/codex/aes-learning` untuk SHA remote. Cocokkan SHA/file sebelum mengatakan pekerjaan ada di GitHub.
