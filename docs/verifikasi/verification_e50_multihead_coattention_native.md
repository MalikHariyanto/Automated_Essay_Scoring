# Verifikasi E50 — 6 November2026

Source model E43 dipin ke `7241a21dfec37c942948b754da3960fba4c7731c`;pelaporan ke E48 pada `437b6ff15808c2122a92cfc32874c815d33af633`. HEAD kerja awal `f2bbf14a6ce1560c80efcdcba2fead0fa84f5cbc`,cwd `/workspace/Automated_Essay_Scoring`. E50 mengganti modul co-attention menjadi4head denganQKVO,bukan hasil penelitian baru.

Notebook188cell/91cellkode,outputkosong,mirroridentik. RawSHA256 `fe9c433d4445fbd90cc3f98b20c5982e635147186b9a63990442ff86c579fed0`;visible-codeSHA256 `e2421fb98676f528cc079086603f0e853c8f579379db6a9c85509fc52886553c`.

```bash
/tmp/e39-verify/bin/python -m pytest tests/test_e50_notebook.py \
 -q -s --tb=short --basetemp=outputs/e50_study/final_fixtures
```

Exit0; **5passed,0failed,0skipped,0xfailed**,82,31detik. Satuwarningdeprecationpluginlingkungan. Manifestprivat `outputs/e50_study/verification.json` dan log `final_tests.log`. Pengujian mencakup lima kelompok berikut:

1. **Source/protokol:**schema/hash/native/nooutputs/mirror/rebuilddeterministik. AST ScalarMix,BiGRUdasar,optimizer,train_all,metrics,AC2,split,Engine,encoder/BGE,windowing samaE43 setelahteksversipadaerror dinormalisasi. LooptrainingepochdanforwardSingleScoreModelsamaE43. Tidak ada LDS/ListNet,magic/worker,cosine,ZIPacuan,loopseed atauhorizonrefit25. SumberE43takditimpa.
2. **Persamaanmultihead:**untuk setiapempathead,softmaxjawaban→referensi danreferensi→jawaban cocokperhitunganmanualQK/sqrt32;aligneddanroundtripcocokA VreferensidanA(B Vjawaban)sesudahconcat/outputproyeksi. Rowvalidsum1,paddingquery0. Perturbasi paddingtidakmengubahkeluarantokenvalid. Gradienfinite/nonzerokejawaban,referensi,Q/K/V/O. MengubahproyeksiQhead1mengubahattentionhead1tanpamengubahhead2–4. Masksemuakosong/headtidaksahditolak.
3. **Inisialisasi/model/learning:**seluruhbobotbersamadanstateRNGbit-identikE43padaseedsama,dengantambahan65.536parameter. Splitidentik. Poolmempertahankanmaskedmean1/N,ekspormeanempathead,tidakdipengaruhitokenluarmask. Optimizer mencakupQKVOdikelompokpooling_normLR1e-4;keempatmatriksberubahsetelahbackward/step. Confighead2ditolak.
4. **Resume:**fit2epochdenganinterupsisetelahsaveepoch1+resume menghasilkanhistory,laporannormaproyeksidanmodelstatebit-identikdenganrun2epochtanpainterupsi. Loss tetapMSEbiasa,target0–1.
5. **Notebooklengkap:**kernelJupyterterisolasi,24datasintetis/randomTransformer4layer,RoBERTa768+BGE1024,tokenpendek,CPU,CV1epochuntukwaktu. LimaCV+saturefit+satutest menghasilkan5Excel/ZIP,18OOF/6test,24barisProyeksiMultihead. Kernelbaru menjalankannotebookulangtanpamengubahcheckpointfinal. Sourcepubliktetap300baris/maks25epoch/patience5. Gate membacaulangmetriktest/OOF,meanSDCV,ID/split/label,rekapselisih,historydankonfigurasihead. Mengosongkanringkasantest/CV,mengubahcount,IDOOFatauJumlahHeadpadaExceldapatmemblokirunduhan.

Matriksattention02adalahmeanprobabilitasempathead,tidakdiklaimheadspesifik. NormaQKVO05menggambarkanparametercheckpoint,bukanbobotattentionataukepentingankausal.

Python3.12.14,torch2.6.0+cpu,transformers4.44.2,pytest9.2.0,nbformat5.11.1,nbclient0.11.0,NumPy2.3.5,pandas2.2.3,SciPy1.17.0,scikit-learn1.8.0,openpyxl3.1.5. Jupytermembutuhkansocketlokal. TidakmelakukanpretrainedGPUforward/trainingpenelitian300dataataumengklaimperbaikanRMSE/target60/60. PengujianteknisbukanjaminanbebasseluruhbugataukompatibilitassemuaruntimeGPU.

Hanyasourcekosong,builder/helper,tests,dokumentasiaman dipublikasikan. Dataset,ZIPhasil,executednotebook,Excel,bobot/logtetapprivat. SHAbranchdanbytefileGitHubsetelahpushdicatatdi `outputs/e50_study/publication.json`. [Metode/paper](../eksperimen/eksperimen_e50_multihead_coattention_native_gpu.md).
