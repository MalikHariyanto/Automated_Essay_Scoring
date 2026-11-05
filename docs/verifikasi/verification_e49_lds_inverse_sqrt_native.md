# Verifikasi E49 — 6 November2026

Source awal E48 pada `437b6ff15808c2122a92cfc32874c815d33af633`. Cwd `/workspace/Automated_Essay_Scoring`. E49 mengubah eksponen inverse setelah LDS1→0,5, mempertahankan arsitektur dan protokolE43. Pengujian berikut bersifat teknis;belum ada hasil penelitianE49.

Notebook `AES_300_E49_E43_S3_LDS_InverseSqrt_Native_Learnable_GPU.ipynb`:186cell/90cellkode,outputkosong,mirroridentik. RawSHA256 `6c6ed63995123ff1acc0ccb4e411d0b8feb125a0a6ae4c4776716bf1ae901053`;visible-codeSHA256 `aded0768fa088b6346426a45fed152cc4f679f375899d4ea07c8881e05b2b757`.

```bash
/tmp/e39-verify/bin/python -m pytest tests/test_e49_notebook.py \
 -q -s --tb=short --basetemp=outputs/e49_study/final_fixtures
```

Exit0; **5passed,0failed,0skipped,0xfailed**,89,24detik. Satuwarningdeprecationpluginlingkungan. Logprivat `outputs/e49_study/final_tests.log`;manifest `verification.json`.

1. Notebook schema/hash/native/nooutputs/mirror/rebuild deterministik lolos. SourceE43/E48tidakditimpa. AST model,optimizer,train_all,split,metrics,Engine,tokenwindowing/encoder/BGE samaE43 setelah teksversipadaerror dinormalisasi. Fit,weightedloss,jurnaltrain danexportcompleted samaE48;perubahanmetodeberadapadabobotLDS/config/fingerprint. Tidakadamagic,worker,exec/evaldalamnotebook,cosine,ListNet,ZIPacuan,loopseed atauhorizonrefit25.
2. Bobot cocok histogram→Gaussian `scipy.ndimage.convolve1d`→inverseakar→mean1, termasuk batas100. Cocok `sqrt(bobotE48)/mean(sqrt(bobotE48))`;max lebihkecil padafixturetakmerata. Auditpower0,5,fingerprintberbeda. Perubahanlabel/teks di luartrain tidakmengubahbobot;indekskosong/duplikat/luarbatas,kernelgenapdanpower0/1/NaNditolak.
3. WeightedMSE/gradiencocokrumusanalitis. Akumulasimicrobatch1/2/3/4termasukbatchtakmerata cocokbatchpenuh;tidakdinormalisasiulangpermicrobatch. UnitweightmenghasilkanMSEbiasa;shape broadcastditolak.
4. Modelawalbit-identikE43dan splitidentik. Fit2epochdenganinterupsisetelahsaveepoch1+resume menghasilkanhistory,auditLDSdanmodelstatebit-identikdenganrun2epochtanpainterupsi. MSE/LDS-MSEterpisah. Indekstrain/validasitumpangtindihditolak.
5. SeluruhnotebookdijalankandarikernelJupyterterisolasi:24datasintetis,randomTransformer4layer/RoBERTa768+BGE1024,tokenpendek,CPU,CV1epochuntukwaktu. LimaCV+saturefit+satutest menghasilkan5Excel/ZIP,18OOF/6testterpisah,90barisbobottrain. Kernelbarumenjalankannotebooklagi tanpa mengubahcheckpointfinal. Sourcepubliktetap300baris/25epoch/patience5. Gate statusPASSED memverifikasimetrik tanpa bobot,OOF/fold,rekapselisih danbobot/auditpower0,5daritrainfold. Ringkasankosong,countdiubah,IDOOFsalah,bobotLDSdiubahmasing-masingmemblokirunduhan.

Pemeriksaantambahan padaoutputfixturesama,tanparetrain:mengganti`weight_power`padaExcelAuditLDSmenjadi1ditolakgateeksponen/fingerprint. Setelahdikembalikan,gatePASSED. Buktiprivat `extra_gate_check.json`.

Terpisahdaritrainingfixture,helperE49menghitungbobotdarilabeltrainyangtersimpandiZIPE48asli;keenamfitcocoktransformasiakarbobotE48. Maxrefit5,7106353563;mean1. Ini validasipembobotan saja,bukantraining/pretrainedforwardatauestimasiRMSE. Buktiprivat `actual_distribution_check.json`.

Python3.12.14,torch2.6.0+cpu,transformers4.44.2,pytest9.2.0,nbformat5.11.1,nbclient0.11.0,NumPy2.3.5,pandas2.2.3,SciPy1.17.0,scikit-learn1.8.0,openpyxl3.1.5. Jupytermembutuhkansocketlokal. Tidakmenguji trainingpenelitian300data/pretrainedGPU/downloadmodelasli,pencapaian60/60,ataukompatibilitasseluruhruntime. Hasilteknistidakmenjaminsemuabugmustahilterjadi.

Hanyasourcekosong,builder,tests,dokumentasiaman dipublikasikan. Dataset,ZIPhasil,executednotebook,Excel,bobot/logtetapprivat. SHAbranchdanbytefileGitHubsetelahpushdicatatdi `outputs/e49_study/publication.json`. [Metode/paper](../eksperimen/eksperimen_e49_lds_inverse_sqrt_native_gpu.md).
