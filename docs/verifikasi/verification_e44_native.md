# Verifikasi E44 MSE + ListNet native — 5 November 2026

E44 mempertahankan model E43 dan mengubah objective training menjadi MSE+0,1×ListNet kelompok soal sama. Verifikasi memakai data sintetis dan Transformer kecil berbobot acak di CPU. Belum ada training penelitian atau pengukuran memori/waktu GPU; tidak ada hasil AES E44 yang diklaim.

## Identitas sumber

Cwd `/workspace/Automated_Essay_Scoring`, baseline/format pin `7241a21dfec37c942948b754da3960fba4c7731c`. Builder membaca notebook E43 pada commit tersebut. Notebook utama/mirror byte-identik, 186 cell/90 cell kode, output kosong. Raw SHA256 `078d4908b1795d899a3ea70835c94e90852c92311f76acca0c859e9c87ea6a1f`; visible-code SHA256 `25738722fdd326f1a0fcda604f4ae0e1f2b397ccfd75cc021a5af60df986b6d2`.

Fixed bobot ranking0,1, target/skor0–1, mean ListNet antarkelompok soal berukuran≥2, tanpa temperature. Microbatch1, batch efektif8, sampler E43 tetap. Activation recompute non-reentrant mempertahankan RNG dropout, dengan tensor kosong pada perangkat model untuk deteksi perangkat/RNGCUDA. Inferensi, arsitektur, optimizer, LR, split/fold, pemilihan epoch dan protokol5CV+1refit+1test tetap. Source E43/E30 dan HTML lama tidak berubah.

## Perintah dan hasil

```bash
/tmp/e39-verify/bin/python -m pytest tests/test_e44_notebook.py \
 -q -s --tb=short --basetemp=outputs/e44_study/verified_fixtures
```

Exit0, **6passed,0failed,0skipped,0xfailed**,93,75detik menurut pytest;95,83detik dengan startup proses. Satu warning deprecation plugin pytest lingkungan. Bukti privat `outputs/e44_study/final_tests.log`, `test_command.json`, `verification.json`, dan `verified_fixtures`. Eksekusi CPU awal juga lolos, kemudian sumber ditambah penanda perangkat karena argumen Python tanpa tensor CUDA tidak cukup untuk deteksi RNGGPU. Pengujian akhir di atas memakai sumber setelah perbaikan tersebut; actual CUDA RNG/AMP tidak dijalankan pada CPU ini.

1. Schema/hash, output kosong, mirror/rebuild deterministik, E43 utuh, tanpa magic/exec/eval/worker/cosine/LLRD/sweepseed. Model, ScalarMix, co-attention/mean, Engine inferensi, optimizer, train_all, split, metrik, encoder/BGE, mask/window dan event ekspor sama secara AST dengan E43; nama versi dinormalisasi pada pesan error.
2. ListNet cocok dengan rumus cross-entropy softmax target dan log_softmax prediksi. Gradien analitik=(probabilitas prediksi−probabilitas target)/jumlah kelompok; arah gradien memperbaiki urutan terbalik. Beda soal tidak dibandingkan, singleton memberi gradien nol, target ranking tidak mendapat gradien. Permutasi daftar atau pergeseran konstanta dalam kelompok mempertahankan ranking loss. Target/prediksi ties sama memberi log2 dan gradien nol, menunjukkan loss tidak harus nol. Nonfinite ditolak.
3. Batch campuran8jawaban/6soal menghasilkan tepat2kelompok/4jawaban ranking. Forward dan gradien recompute cocok dengan versi tanpa recompute, termasuk dropout head yang aktif dan keadaan RNG setelah backward. Penanda perangkat, non-reentrant dan RNGpreserve diperiksa pada8microbatch. Gradien encoder, BiGRU, proyeksi referensi, matching, ScalarMix/gamma, proyeksiBGE dan MLP nonzero. Ini bukti CPU dan argumen perangkat, bukan pengujianCUDA.
4. Semua bobot awal E44 identik E43, tanpa parameter baru dan default lama berubah. Bobot ranking0 menghasilkan MSE/gradien yang cocok dengan akumulasi E43 dalam toleransi floating-point. Save/load menjaga model dan menolak resep/bobot berbeda. Bobot negatif,nonfinite atau boolean ditolak.
5. Simulasi training dua epoch dihentikan setelah save epoch1. Resume memulihkan model, optimizer/scaler/scheduler, generator loader dan RNG. Riwayat serta seluruh bobot checkpoint terpilih bit-identik dengan training dua epoch tanpa interupsi; ranking benar-benar aktif.
6. Seluruh90cell kode dijalankan dalam kernel terisolasi dengan24baris sintetis/3soal, random RoBERTa/XLM-R4layer berdimensi768/1024, CPU,satu epoch/token pendek. Source publik tetap pretrained/GPU/300baris/25epoch. LimaCV,satu refit,satu test,limaExcel dan ZIP selesai. Kernel baru memakai cache tanpa perubahan checkpoint. X/y cocok dengan masukan training dan tidak ada import proyek/worker tersembunyi. History enam tahap memisahkan MSE/ranking/gabungan dengan identitas MSE+0,1×ListNet dan cakupan ranking valid. OOF unik/disjointtest,jumlah development+test24,soal terisi,kelima fold ada. Prediksi OOF cocok dengan sumber validasi; error absolut/kuadrat dan ringkasan toleransi cocok. Bobot mean tetap1/N dan nama ZIP benar.

## Lingkungan dan publikasi

Python3.12.14,torch2.6.0+cpu,transformers4.44.2,pytest9.2.0,nbformat5.11.1,nbclient0.11.0,NumPy2.3.5,pandas2.2.3,SciPy1.17.0,scikit-learn1.8.0,openpyxl3.1.5. Jupyter memerlukan socket localhost. Durasi fixture bukan estimasi trainingT4. Fixture tidak membuktikan kualitas AES,performa pretrained asli,atau kecukupan memori setiap runtimeGPU.

Hanya notebook kosong,builder,tes dan dokumentasi aman dipublikasikan. Dataset,unggahan siswa,executed notebook,Excel,bobot/cache/log tetap privat. Sesudah push,SHAbranch dan byte notebook utama/mirror/metode diperiksa;catatan privat di `outputs/e44_study/publication.json`. Paper Yangdkk.(2020) mendukung kombinasi regression/ranking pada AES dengan bobot dinamis;fixed0,1dan pengelompokan soal pada fusion ini adalah adaptasi yang belum terbukti meningkatkan RMSE.
