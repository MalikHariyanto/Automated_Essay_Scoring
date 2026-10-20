# Verifikasi kode inline dan Excel tiap tahap

**Pembaruan penyajian:** paket akhir kini lima laporan utama, bukan38workbook terpisah. Penghitungan model dipertahankan; lihat [verifikasi kesamaan angka dan konversi ZIP lama](verification_dataset300_readable.md). Laporan di bawah adalah bukti versi awal kode inline.

20 Oktober 2026. Branch `codex/aes-learning`; basis perubahan `cb6e599ff6d6a03ace1ab547b17294b80b5664c8`. CWD verifikasi `/workspace/Automated_Essay_Scoring`. Notebook publik39cell,23codecell; source SHA256 `50431d9db5b65108a819ea45056da862b313a1dabac31e7f4c29a319c98061a7`, mirror byte-identik. Seluruh output publik dikosongkan.

## Pemeriksaan yang benar-benar selesai

| Jenis bukti | Hasil | Batas makna |
| --- | --- | --- |
| Audit dua unggahan terbaru | Excel300respons dengan satu nilai `rater`; hash cocok versi sebelumnya. Notebook BGE-M3 asli8codecell diperiksa kembali. | Tidak menambah label rubrik atau menganggap populasi SMA terverifikasi. |
| Pembacaan paper | Arsip ZIP30.249.465byte lolos CRC; PDF gabungan diekstrak privat. Bagian RoBERTa43–47, BGE3134–3136, fusion1113–1117, evaluasi87–92, AESIndonesia4047–4049 dibaca kembali. | Nomor halaman PDF gabungan; bukan klaim membaca ulang seluruh4051halaman/semua sumber. |
| Setup worker Python3.13 | Instalasi/impor nyata selesai; torch2.6.0+cpu, Transformers4.48.3, NumPy2.2.6, Pandas2.2.3, SciPy1.15.3, sklearn1.6.1, openpyxl3.1.5. | CPU; GPU/AMP Colab belum diuji pada hardware. |
| Test terkait dan regresi pipeline | **52passed,0failed,0skipped,0xfailed**, exit0. | Test teknis, bukan akurasi AES. |
| Excel round-trip | Koordinat terakhir1024dimensi tidak hilang; teks `=...` tetap literal; NaN/coverage tidak diganti0; sheet/header/rowID dan metrik terbaca kembali. | Nilai fixture sintetis. |
| Fusion dan provenance | Vektor concat pada workbook sama dengan penyambungan768RoBERTa+256proyeksi; riwayat per-epoch, CV/test dan indeks workbook cocok. | Tidak mengklaim fusion unggul pada data siswa. |
| Worker persisten | Nilai bertahan antarcells; tabel numerik dikirim ke kernel; exception dihentikan/dilaporkan. | Bukan sekadar mencetak source sebagai output. |
| Notebook data asli | Seluruh23codecell dieksekusi dalam mode audit, tanpa error;240development/60test, **6Excel aktual**, arsip privat berhasil. | Encoder/training sengaja tidak dijalankan pada audit. |
| Notebook alur lengkap | Seluruh23codecell dieksekusi sampai prediksi/evaluasi/ekspor fitur final dan kata setelah latihan; **38Excel**, tanpa error. | Menggunakan24baris sintetis,1epoch, transformer kecil berbobot acak dan tokenizer fixture, diberi scopeTECHNICAL; bukan hasil penelitian/checkpoint pretrained. |
| Format publik | Sintaks valid termasuk body `%%aes`; tidak ada `show_code`/`display(Code)`; definisi actual model/optimizer/training ada di cell; output kosong, mirror identik. | Tidak ada executed notebook atau Excel siswa yang dipublikasikan. |

Command suite:

```bash
PYTHONPATH="$PWD/src" HF_HOME="$PWD/outputs/cache/huggingface" \
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
outputs/dataset300_env_py313/bin/python -m pytest -q -rA \
tests/test_teacher_excel.py tests/test_teacher_score.py tests/test_dataset300_setup.py \
tests/test_experiment_protocol.py tests/test_trainer.py tests/test_pipeline.py tests/test_reporting.py
```

Eksekusi notebook menggunakan `nbclient.NotebookClient` dengan kernel Python3.13 dan worker numerik3.13. Notebook audit menggunakan Excel unggahan asli dengan `AES300_AUDIT_ONLY=1`. Run teknis memakai salinan notebook privat dengan override eksplisit dataset24baris,1epoch dan encoder/tokenizer fixture; metadata workbook diberi `TECHNICAL RANDOM TRANSFORMER FIXTURE ONLY; NOT STUDENT RESULTS`. Source notebook publik tetap menggunakan checkpoint asli dan dataset300baris, maksimum5epoch/fold.

Run test awal menemukan satu assertion pembacaan blank terakhir: reader openpyxl streaming menghilangkan sel kosong trailing dari iterasi tanpa lebar eksplisit. Assertion diperbaiki untuk membaca seluruh1026kolom (ID+teks+1024koordinat); bukan mengganti data kosong dengan nol. Eksekusi notebook teknis awal menemukan import `gc` belum tersedia dalam namespace inline; import ditambahkan dan seluruh notebook dieksekusi ulang sampai akhir. Suite final52kasus lulus.

Log, manifest hash source, executed notebook, ZIP dan workbook tetap privat di `outputs/lesson`/`outputs/dataset300_inline`. Excel berisi informasi input/fitur nyata pada tiap run; tahap yang belum berjalan tidak mendapat angka dummy. Ukuran file bergantung panjang jawaban; token/kata/kalimat lengkap dibatasi satu jawaban contoh, embedding per-jawaban mencakup semua300respons.

## Batas verifikasi penelitian

Akses config checkpoint BGE di Hugging Face ditolak proxy dengan `Tunnel connection failed:403Forbidden`. Tidak ada klaim forward/backward pretrained atau training/evaluasi penelitian dataset300 pada mesin ini. Angka fixture tidak dipindahkan menjadi skor siswa. Program untuk menjalankan penelitian lengkap tersedia di notebook; perlu Colab dengan akses model/GPU. Semua model final dikunci sebelum finaltest, dan ekstraksi fitur seluruhjawaban dilakukan sesudah evaluasi final untuk inspeksi.

Dasar metode tidak diatribusikan berlebihan: RoBERTa umum berbeda dari checkpoint Indonesia; M3dense mendukung CLS+L2, bukan skor AES langsung; paperhybrid memakai fiturmanual, bukan fusion identik proyek; p08 memakai jawaban-acuan, berbeda dari input hanyaresponse_text. Detail rujukan dan batasnya ada di [daftar referensi](aes_dataset300_references.json).

Publikasi diperiksa melalui SHA branch remote dan byte notebook/source setelah push. Bukti remote disimpan privat agar laporan dalam commit tidak merujuk hash commit dirinya sendiri. Tidak ada source Excel, teks/email/nilai individual, bobot, cache atau output terisi yang masuk commit.
