# Setup ringkas dan catatan kode Colab utama

20 Oktober 2026. CWD `/workspace/Automated_Essay_Scoring`, branch `codex/aes-learning`, basis `a7f345229880ca7e2327eb60b2b9c1c0ef3fb97d`.

Setup lama meneruskan seluruh log pip ke output utama dan selalu menjalankan instalasi. Dengan `system_site_packages=True`, pip melihat versi pustaka Colab yang berada di luar worker; upaya penggantian versi menghasilkan pesan tidak dapat uninstall. Perbaikan memeriksa kebutuhan AES terlebih dahulu, melewati instalasi jika sesuai, dan memakai `--ignore-installed` untuk memasang overlay lokal jika diperlukan. Tidak ada pemutakhiran pip wajib. Flag tersebut mengatur tempat/cara instalasi, bukan mengabaikan kegagalan verifikasi.

Versi pin untuk Python 3.12/3.13 tetap sama. Pemeriksaan menelusuri dependensi aktif dari requirements AES dan torch, kemudian memeriksa import dalam proses worker baru. Torch runtime yang sudah ada dipertahankan; wheel CPU hanya dipasang bila torch memang tidak tersedia. Aplikasi bawaan lain yang tidak dipakai AES tidak termasuk klaim konsistensi dependensi. Log installer lengkap tetap disimpan; error dan ketidakcocokan kebutuhan AES tetap menghentikan proses sebelum metadata siap diterbitkan.

## Bukti

| Pemeriksaan | Hasil dan batas bukti |
| --- | --- |
| Suite terkait | **33 passed, 0 failed, 0 skipped, 0 xfailed**, exit 0. |
| Pip nyata dengan wheel fixture lokal | Runtime fixture menyediakan paket versi 2.0, worker memerlukan 1.0. Versi 1.0 terpasang lokal, metadata/file versi 2.0 tetap utuh, tidak ada `Attempting uninstall` atau `Can't uninstall`. Tidak mengakses indeks jaringan dan tidak mengganti torch. |
| Setup nyata pada lingkungan yang sesuai | Python 3.13.5, torch 2.6.0+cpu; 53 pustaka pada graph dependensi AES sesuai; import lolos. Instalasi dilewati. Output utama hanya empat baris status. |
| Salah versi atau import gagal | Status siap lama dihapus; tidak diterbitkan status siap baru. Mode pemeriksaan worker eksternal tidak memasang paket. Torch terpasang yang gagal import tidak diganti diam-diam dengan CPU. |
| Preservasi GPU | Routing torch bawaan dan metadata GPU dicakup fixture untuk Python 3.12/3.13; bukan uji hardware GPU Colab nyata. |
| Notebook utama | 22 cell kode selesai pada fixture teknis. Tabel status dan rincian log terlipat muncul sebagai HTML; ZIP akhir tetap lima Excel. |
| Model dan angka | Definisi numerik encoder, tokenizer, model, Engine, vektor, fit, training dan metrik identik secara AST dengan versi sebelumnya. Semua metrik CV/test serta seluruh CSV prediksi test fixture identik. |
| Catatan kode | Nama checkpoint terlihat di cell konfigurasi; komentar pada inisialisasi, tokenizer, gradien, window, fusion, loss dan evaluasi. Fungsi-fungsi penting diberi catatan singkat. Tidak mengganti checkpoint/revision atau rumus. |

Command suite:

```bash
PYTHONPATH="$PWD/src" HF_HOME="$PWD/outputs/cache/huggingface" \
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
outputs/dataset300_env_py313/bin/python -m pytest -q -rA \
tests/test_dataset300_setup.py tests/test_teacher_presentation.py \
tests/test_teacher_excel.py tests/test_teacher_score.py
```

Setup nyata dijalankan dengan `outputs/dataset300_env_py313/bin/python scripts/setup_dataset300.py`. Lingkungan: Python 3.13.5, torch 2.6.0+cpu, Transformers 4.48.3, NumPy 2.2.6, Pandas 2.2.3, openpyxl 3.1.5, pytest 8.3.5. Bukti privat mencatat command, cwd, exit code, versi dan hash source pada `outputs/setup_readable_tests_proof.json`, XML/log pengujian, serta `outputs/setup_readable_real_proof.json`.

Eksekusi notebook memakai 24 jawaban sintetis, satu epoch, tokenizer fixture dan transformer kecil berbobot acak, berlabel TECHNICAL. Tidak diklaim sebagai training penelitian, forward checkpoint pretrained asli, atau pengujian hardware pengguna dengan Python 3.13.16/torch 2.11.0. Notebook terisi, log, data dan hasil tetap privat. Notebook publik dikosongkan dan mirror dipertahankan; SHA serta byte file GitHub diperiksa sesudah push.
