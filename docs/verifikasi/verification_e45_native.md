# Verifikasi E45 Scalar Mix seluruh12layer — 5 November 2026

E45 memperluas Scalar Mix empat layer terakhir E43 menjadi seluruh12layer Transformer, dengan MSE tetap. Verifikasi memakai data sintetis dan Transformer berbobot acak di CPU. Tidak ada training penelitian, checkpoint pretrained asli, atau pengukuran performa GPU E45.

## Identitas sumber

Cwd `/workspace/Automated_Essay_Scoring`, HEAD awal `1c18a80644e9026a6867665f17b467723f8b4db8`. Builder E45 dipin langsung ke E43 pada `7241a21dfec37c942948b754da3960fba4c7731c`, bukan diturunkan dari E44. Notebook utama/mirror byte-identik,184cell/89cellkode,output kosong. Raw SHA256 `597c679d77189cbec9e9be7921795b3554845406d238c3f85d072e0949f74ab5`; visible-code SHA256 `f04274b9ef7d7a7895e713764026670ab9880e0e73840d943ce6b556bbb0777a`.

Parameter campuran12logit+satu gamma=13, bertambah8dari E43. Bobotawal1/12,gammaawal1,LRcampuran1e-4;encoder5e-6,komponenlain1e-4tetap. LossMSE,5CVsekali,maks25epoch,patience5,satu refit+test;tanpa ranking/recomputeE44/sweepseed/kontrolulang. E43/E44/E30 dan HTML lama tidak diubah.

## Perintah dan hasil

```bash
/tmp/e39-verify/bin/python -m pytest tests/test_e45_notebook.py \
 -q -s --tb=short --basetemp=outputs/e45_study/verified_fixtures
```

Exit0, **5passed,0failed,0skipped,0xfailed**,104,03detik menurut pytest;106,07detik termasuk startup proses. Satu warning deprecation plugin pytest lingkungan. Bukti privat di `outputs/e45_study/final_tests.log`, `test_command.json`, `verification.json` dan `verified_fixtures`.

1. Schema,hash,outputkosong,mirror dan rebuild deterministik; E43/E44utuh. Tidak ada magic/exec/eval/worker/cosine/LLRD/sweepseed/ListNet/activationrecompute. Optimizer,fitMSE,train_all,split,metrik,Engineinferensi,encoder/BGE,co-attention/mean,mask/windowing,eventekspor danloader sama secara AST dengan E43;nama versi dinormalisasi pada pesan error.
2. Dua belas tensor Transformer bernilai1–12 menghasilkan rerata6,5; embedding awal bernilai10000 tidak masuk campuran dan tidak mendapat gradien langsung dari ScalarMix terisolasi. Semua12tensor terpilih mendapat gradien1/12,logit dan gamma memperoleh gradien. Hanya11layer Transformer ditolak. Modulcampuran tepat13parameter.
3. Model fixture dengan RoBERTa12layer mempunyai seluruh bobot bersama dan RNGawal identik dengan E43,selisih tepat8parameter. Saatdelapan bobotawal dinolkan danempat terakhir0,25,prediksi,ringkasan128,pooling768,normcampuran danconcat1024 bit-identik E43. Mengembalikan bobot seragam12layer mengubah prediksi dan menghasilkan gradien pada semua12blok encoder,logit/gamma. Kelompok scalar_mix berisi13parameter pada1e-4;dimensi128/768/1024tetap.
4. Model12layer dengan campuran4ditolak oleh guard seluruhlayer. Save/load mempertahankan12layer dan loader menolak metadata campuran4. Checkpoint tetap memiliki format aman weights_only.
5. Seluruh89cell kode dijalankan dalam kernel terisolasi dengan24baris sintetis,randomRoBERTa/XLM-R **12layer** berdimensi768/1024,CPU,satu epoch/tokenpendek. Sourcepublik tetappretrained/GPU/300baris/25epoch. LimaCV,satu refit,satu test,limaExcel danZIP selesai. Kernelbaru memakai cache tanpa perubahan hashcheckpoint. X/y cocok dengan inputtraining,tanpa importproyek/worker tersembunyi. Excel05berisi72baris bobot/gamma dari6checkpoint,indeks0–11/layer1–12,12bobot percheckpoint berjumlah1 danbobotawal1/12. Bobotfinal cocok dengan softmax checkpoint. History tetapMSE,tanpaListNet;normsesudah≈|gamma|×normsebelum danmeanpooling1/N. OOFunik,soalterisi,terpisahdaritest,jumlahOOF+test24;selisih,prediksi mentah per-fold danringkasanMAE/toleransi cocok. NamaZIPbenar.

## Lingkungan dan batas bukti

Python3.12.14,torch2.6.0+cpu,transformers4.44.2,pytest9.2.0,nbformat5.11.1,nbclient0.11.0,NumPy2.3.5,pandas2.2.3,SciPy1.17.0,scikit-learn1.8.0,openpyxl3.1.5. Jupyter memerlukan socketlocalhost. Durasi fixture tidak menjadi estimasi trainingT4 atau bukti kualitas AES. Dukungan operasi12layer diperiksa pada model acakCPU,bukan GPU/pretrainedpenelitian.

Landasan ELMo dan implementasi AllenNLPseluruhlayer mendukung mekanisme,bukan membuktikan12layer optimal. Hanya notebookkosong,builder,tes dan dokumentasi aman dipublikasikan. Data,unggahan,executednotebook,Excel,bobot,cache danlog tetapprivat. Setelahpush,SHAbranch serta byte notebookutama/mirror/metode danE44diperiksa;catatan di `outputs/e45_study/publication.json`.
