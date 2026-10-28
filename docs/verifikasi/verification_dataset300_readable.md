# Verifikasi lima laporan AES

20 Oktober2026, branch `codex/aes-learning`, basis perubahan `3d05165a93f0c120db52418c5cc6b7e8f70dc15d`. CWD `/workspace/Automated_Essay_Scoring`. Perubahan hanya pada ekspor/tampilan dan penyaringan satu pemberitahuan checkpoint yang diketahui tidak relevan.

## Bukti yang dijalankan

| Pemeriksaan | Hasil | Batas makna |
| --- | --- | --- |
| Suite terkait | **55passed,0failed,0skipped,0xfailed**, exit0 | Fixture teknis, bukan ukuran akurasi siswa. |
| Angka model sebelum/sesudah | Seluruh prediksi test S1/S2/S3, seluruh metrik CV dan metrik finaltest **identik persis** pada fixture notebook yang sama | Model transformer kecil berbobot acak; tidak diklaim sebagai hasil checkpoint asli. |
| Angka workbook | Prediksi, label, selisih, metrik dan seluruh koordinat vektor cocok sumber setelah penamaan ulang | Tidak ada penghitungan metrik ulang pada converter. |
| Arsip utama | Lima XLSX dengan sheet Panduan, nama kolom terbaca, tanpa sheet dropout/GELU/outputnormalized | Rincian sumber tetap privat; nilai vektor utama tidak dipangkas. |
| Riwayat | Lima fold dan pelatihan final digabung per model; tidak hilang epoch/fold | Loss berskala target tetap diberi nama MSE, bukan RMSEpoin. |
| Sumber tetap utuh | Hash workbook sumber sama sebelum/sesudah konversi | Konversi hanya membaca hasil lama. |
| Notebook03 |23codecell dieksekusi sampai akhir; paket utama5Excel berhasil |24baris sintetis,1epoch, tokenizer/encoder fixture. |
| Notebook04 |3codecell dieksekusi tanpa error; ZIP lama38Excel menjadi5laporan, sekitar9detik pada fixture | Tidak membutuhkan checkpoint/model; bukan bukti membuka ZIP pengguna yang melebihi batas32MB transfer tool. |
| Hasil parsial | Sumber yang baru mempunyai data tidak memperoleh hasil test/vektor fiktif | Paket hanya mencakup laporan yang datanya ada. |
| Warning pooler | Hanya pemberitahuan `pooler.dense.bias`+`pooler.dense.weight` yang tidak digunakan disaring | Warning bobot encoder lain, bobot missing dan error tetap diteruskan. |

Command suite:

```bash
PYTHONPATH="$PWD/src" HF_HOME="$PWD/outputs/cache/huggingface" \
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
outputs/dataset300_env_py313/bin/python -m pytest -q -rA \
tests/test_teacher_presentation.py tests/test_teacher_excel.py tests/test_teacher_score.py \
tests/test_dataset300_setup.py tests/test_experiment_protocol.py tests/test_trainer.py \
tests/test_pipeline.py tests/test_reporting.py
```

Lingkungan CPU: Python3.13.5, torch2.6.0+cpu, Transformers4.48.3, NumPy2.2.6, Pandas2.2.3, openpyxl3.1.5, pytest8.3.5. Kernel/worker notebook menggunakan lingkungan yang sama seperti verifikasi sebelumnya.

Implementasi `experiment.py`, `data.py`, `windows.py` dan `metrics.py` tidak diubah. `SingleScoreModel.forward` dan `Engine` tidak diubah. Perubahan loader encoder hanya memasang/melepas filter logger di sekitar pemanggilan `AutoModel.from_pretrained` dengan argumen yang sama. Perubahan BGE/vector lab lain hanya menyederhanakan pesan progress.

Laporan ringkas membaca XLSX lama melalui openpyxl. Tidak mengimpor modul training/encoder atau memanggil forward model. ID/tanda posisi ditampilkan mulai1, `v_0000` dinamai `Komponen 1`, dan persentase di luar rentang ditampilkan sebagai persen; angka mentah model tidak ditimpa. Selisih pada tabel gabungan disalin dari hasil sumber. CV mean/SD disalin dari ringkasan yang sudah dihitung.

Artefak privat: `outputs/presentation_process_parity.json`, `outputs/repack_notebook_proof.json`, `outputs/presentation_tests_final.log`, manifest hash source dan notebook terisi. Semua fixture jelas dilabeli TECHNICAL dan label itu diteruskan ke Panduan laporan. Tidak ada angka fixture atau dokumen siswa yang dipublikasikan sebagai hasil penelitian.

Pengujian checkpoint asli/training penelitian bukan bagian perubahan presentasi ini dan tidak diklaim dijalankan. ZIP pengguna tidak diproses oleh tool chat karena batas32MB; notebook04 menyediakan pemrosesan ZIP itu langsung di Colab, tanpa training ulang. Bukti publikasi remote dicatat privat setelah SHA branch dan byte file diverifikasi.
