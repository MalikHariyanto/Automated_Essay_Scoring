# Verifikasi penambahan hasil E35 — 2 November 2026

Cwd `/workspace/Automated_Essay_Scoring`;basis699aaa3be612400ccf264bd8a8b96984263c8c8c. Ini audit ZIP unggahan dan HTML hasil,tidak menjalankan training penelitian baru.

- ZIP E35 SHA256`1e23d65c95a5b258a7d2ec263046cfd6674582afe59d26d92f16bb7f99d4bdb9`;5workbook lengkap.300baris,240development,60test,15fold S1/S2/S3. Teks/label/split/fold cocok keluarga lama. Prediksi S1/S2 identik per jawaban E30/E33/E34. Metrik test dihitung ulang,toleransi1e-6;deviasi maksimum2,400250771472656e-7. Epoch terpilih minimum RMSE riwayat;refit9/19/11sesuai median.
- Audit attention/co-attention:16.946token respons,17.600referensi,bobotfinite/nonnegatif/jumlah1perwindow;posisi token cocok sumber. Ringkasan128danpooling768lengkap300jawaban. Matriks alignment penuh tidak ada,sehingga top1 tidak dihitung ulang dari matriks. Workbook Kata Berulang8kolom;source E35 native tanpa fungsi cosine.
- `PYTHONPATH=src:scripts /tmp/e29-verify/bin/python -m pytest tests/test_e35_study.py tests/test_e34_study.py tests/test_e33_study.py tests/test_e31_e32_study.py tests/test_three_input_study.py tests/test_three_input_identities.py -q --tb=short`:exit0;27passed,0failed/skipped/xfailed;3,34detik. Memeriksa provenance,LR aktual,hasil campuran,CV/duel,privasi dan pelestarian semua hasil/codebooks lama.
- `NODE_PATH=/workspace/Sistem-BTD/node_modules node scripts/check_e35_study.mjs`:exit0;2kelompokpassed,0failed.15identitas,metode/LR E35,3buku kode,layar390/1440tanpa overflow;Tampilkan kunci berfungsi,kunci salah ditolak,kunci lama membuka dashboard. Setelah jaringan diblokir2700prediksi tersedia,Excel E35 byte-identik,8kolom Kata Berulang tanpa cosine,serta kode/output E30 tetap dapat dibuka. Chromium154.0.8037.57. Ini browser desktop/emulasi layout,bukan pengujian HP fisik.
- Integritas15ZIP/75XLSX asli dibuktikan SHA256;704sheet tetap tersedia.14snapshot/audit lama,91duel lama,dan3dokumen kode lama dipertahankan. Total105duel baru. Buku MLP lama tidak berubah.
- Paket standar menggunakan nama pengunduh/ZIP/HTML lama dan kunci lama. Enkripsi+gzip roundtrip,HTML dalamZIP dan gabungan6bagian terverifikasi. Paket lokal439,98MiB;SHA256`d15bc39e92e7213e7f15289e14071c15de80e8b3a7a97767aef78c1b26e41506`. Kunci tidak dicetak,ditanam dalam source,atau diunggah.
- Hasil E35 meanCV5,8550,RMSEtest5,9452,MAE4,4253. CV turun dibandingE30,tetapi test E30 tetap lebih baik. Tidak ada klaim satu seed membuktikan keunggulan stabil. Notebook eksekusi/checkpoint/perangkat tidak disertakan dalamZIP sehingga byte runtime/GPU tidak dibuktikan.
- Publikasi paket terenkripsi sebelumnya ditolak review persetujuan otomatis. Pembaruan pengunduh GitHub sampaiE35 tidak diklaim online sebelum bagian paket benar-benar diunggah. Paket siap lokal pada`outputs/e21_study/Belajar_AES_Tiga_Input_Lengkap.zip`. Manifest/log dan status publikasi disimpan di`outputs/e35_study/`.

Manifest sumber aktual:

- `scripts/build_e21_study.py`: `9486c2f9c1661d1e4786de25d8d695188e3390cad97b72d9659f2811caf987f5`
- `scripts/build_three_input_study.py`: `b67b22a9b2141df0a43405d28a1f5caa055229db718e4b618cc4c42b15e74350`
- `scripts/three_input_identities.py`: `6f768edc4fef6c109d4ecd881b2b877e3a2f544a619570a72dc9f081300f4313`
- `scripts/templates/three_input_study.js`: `2822f71f16f4c4ee9479b6aee36af409bfb0f20ca034ce426b4d5bbbbf087bd5`
- `scripts/templates/three_input_study.css`: `5564ca1f0508ce7c183db47ca5e8fe5987579d9ef3470dcf274ed8ed10ae41e2`
- `scripts/templates/three_input_download.html`: `956bcd806e2c23247e3c9ac3cb69b6b4a029da358d1a6dfe4705e586aed46511`
- `scripts/check_e35_study.mjs`: `18540fcd619a3304e4069513036d7da8d7dc449e7be79f26bd1f1b0ad5027f38`
- `tests/test_e35_study.py`: `15fb485f3e2fe6f110785b203a8169c8ad9c89286b12278e658afd487c0cfd86`
- `docs/snapshots/e35_study_snapshot.json`: `d40e31e9bc7caa9e5583546bd39f2178234263cf31e24543d2b994232f4a7b0a`
- `docs/belajar_Tiga_Input_Spider.html`: `668ea3eb66a2c0b1d6624cf1ab8f6e40ea7513b9d4330cb9163e57c4dc73fa0d`
- `docs/review/review_e35_study.md`: `e4b90cb127d255961e20ea1bd340ea5f5b3a9f3622f65627897558bf73ab22c9`
