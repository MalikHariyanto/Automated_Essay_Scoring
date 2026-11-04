# Verifikasi E30 acuan lengkap native — 4 November 2026

Koreksi ruang lingkup pengguna: E30 acuan harus menjalankan S1/S2/S3 lengkap. Notebook baru mempertahankan resep asli ketiga skenario dan format Python biasa. E41 tidak berubah. Tidak ada hasil penelitian baru atau klaim reproduksi RMSE historis E30 pada verifikasi ini.

## Identitas

Cwd `/workspace/Automated_Essay_Scoring`,baseline kerja `02d4bf384421811fce77335e103cbb065a5bdfe2`. Builder `scripts/build_e30_full_native.py` mengambil format native E35 pada commit tersebut,kemudian mengembalikan LR pooling_norm S3 ke1e-4. Notebook utama/mirror byte-identik,184cell/89codecell,output kosong. Raw SHA256 `3d4f035d5e89989c5c7bd506e10bc06cc7f38d2726137568e204ca6163bfce4a`;visible-code SHA256 `f579187f59892435a7ec223a326857e1f86363bc8909db86ce62c33c0154605b`.

## Tes

```bash
PYTHONPATH=src:scripts /tmp/e39-verify/bin/python -m pytest \
 tests/test_e30_full_native.py -q -s --tb=short \
 --basetemp=outputs/e30_full_study/verified_fixtures
```

Exit0, **5passed,0failed,0skipped,0xfailed**,89,75detik. Satu warning deprecation plugin pytest lingkungan. Log privat `outputs/e30_full_study/tests.log`. Pengujian awal empat kasus non-notebook juga lolos4,16detik/1deselected; empat kasus tersebut diulang dalam lima kasus akhir,bukan sembilan kasus berbeda.

1. **Struktur dan metode:** schema/hash,mirror,rebuild deterministik dan source historis,E41 serta E30S3native utuh. Tanpa magic,exec/eval,worker tersembunyi,cosine,Scalar Mix,LLRD atau loop multi-seed. Tiga skenario,15CV,3refit,encoder5e-6dan seluruh head/proyeksi/pooling_norm1e-4tercatat benar. AST model,Engine,optimizer,fit,CV/refit/test,metrik,tokenisasi,mask dan fitur identik dengan E30 historis.
2–4. **Kesetaraan per skenario:** S1,S2 dan S3 diperiksa terpisah menggunakan fixture24baris,tokenizer teknis dan encoder teknis. Seluruh nilai DEFAULTS yang digunakan sama dengan E30,split identik,state awal,prediksi eval dan seluruh state setelah backward/satu update AdamW bit-identik. Semua parameter trainable tercakup tepat sekali; LR kelompok benar untuk tiap model.
5. **Eksekusi notebook lengkap dan pemulihan:** seluruh89codecelldijalankan dalam kernel terisolasi dengan24baris sintetis,random RoBERTa/XLM-R1layer berdimensi768/1024,CPU,1epoch dan token pendek. Source publik tetap pretrained/GPU/300baris/25epoch. Lima fold setiap skenario,total15CV,3refit,3model final dan5Excel/ZIP selesai. Workbook05memuat tiga riwayat dan ringkasan15fold;workbook04memuat prediksi ketiga skenario. Config/checkpoint memastikan pooling_norm S3=1e-4. Kernel baru memakai cache sesuai hash tanpa mengubah ketiga checkpoint final. X/y terlihat sama dengan masukan training; import proyek dilarang dan tidak ada worker .py.

## Batas bukti dan publikasi

Venv `/tmp/e39-verify`,Python3.12.14,torch2.6.0+cpu,transformers4.44.2,pytest9.2.0,nbformat5.11.1,nbclient0.11.0,NumPy2.3.5,pandas2.2.3,SciPy1.17.0,scikit-learn1.8.0,openpyxl3.1.5. Kernel Jupyter memakai izin socket localhost. Durasi tes bukan estimasi training penelitian di GPU. Fixture tidak membuktikan akurasi atau kompatibilitas semua runtime GPU Colab.

Hanya source notebook kosong,builder,tes dan dokumentasi aman dipublikasikan. Dataset/unggahan siswa,executed notebook,Excel,log,cache,bobot dan kunci tetap lokal. Manifest source/perintah/hasil tersimpan privat di `outputs/e30_full_study/verification.json`;SHA branch dan byte notebook/mirror/metode diperiksa setelah push. Publikasi tidak membuktikan training penelitian.
