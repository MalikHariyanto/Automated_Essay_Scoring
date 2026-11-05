# Verifikasi E46/E47 refit-only dan gate unduhan — 5 November 2026

Kedua notebook memakai ulang CV acuan dan melakukan satu refit baru dengan horizon scheduler25. Pengujian berikut adalah fixture sintetis/randomTransformerCPU, bukan training/pretrainedGPU atau hasil penelitian E46/E47.

## Identitas source

Cwd `/workspace/Automated_Essay_Scoring`, HEAD awal `6875780f709d1436f25a4706292d71c137f2fc55`. Builder meminjam E43 pada`7241a21dfec37c942948b754da3960fba4c7731c`untukE46,dan E45padaHEADawaluntukE47. Masing-masing187cell/91cellkode,outputkosong,mirroridentik.

| Notebook | Raw SHA256 | Visible-code SHA256 |
|---|---|---|
| E46 | `a5726e52ac13b12f98baa90ba6e143e47fd2cb41bce6ce09434251728b9f20d7` | `162193a0b2ab2bbb10fcbff69ee9400515b726d4f33cd8eef410194e69f0de21` |
| E47 | `415aacac32ea4f8be562d4d09336f576b735cab7fab4eb86908f97bb0c5b59cb` | `e303f5375a5d11aeed78d315f8c53a34c2d81a988e149e6d6e9bd330111ca429` |

SourceE43/E44/E45danHTMLlama tidakditimpa. Model/input/optimizer/lossMSE mengikutiacuan,sedangkanfitrefit memiliki guards,durasiterkunci18/7,horizonjadwal25danLRtrace. CVbaru0,foldlama5,saturefit+testbaru. SHAZIPacuanasli dikunci,fiturBGEfrozen dipakaiulang.

## Perintah dan hasil

```bash
/tmp/e39-verify/bin/python -m pytest tests/test_e46_e47_notebooks.py \
 -q -s --tb=short --basetemp=outputs/e46_e47_study/verified_fixtures
```

Exit0, **8passed,0failed,0skipped,0xfailed**,376,77detik menurutpytest;379,17detik termasukstartup. Satuwarningdeprecationpluginpytestlingkungan. Buktiprivat`outputs/e46_e47_study/final_tests.log`,`test_command.json`,`verification.json`dan`verified_fixtures`.

Pemeriksaanmeliputi:

1. Schema/hash,mirror,rebuilddeterministik,outputkosong,sourceacuanutuh. Tanpamagic/exec/eval/worker/cosine/sweepseed/ListNet. ScalarMix,SingleScoreModel,co-attention/mean,optimizer,Engine,metrik,split,windowingdanencoderidentikacuansecaraAST,setelahnama versipadaerror dinormalisasi.
2. JadwalPyTorchaktualhorizon25menghasilkanLRakhir80%puncakpadaepoch7baik24step/epochCVmaupun30step/epochrefit. Pada30step/epochdan18epoch,hasil14/45puncak. Ini pengujian25epochhorizon,terpisahdariend-to-endyangdipendekkanuntukCPU.
3. Untukmasing-masingmodel4/12layer,`fit`menolakvalidationbaru,training2epochmemakaihorizon25. Simulasiinterupsisetelahsaveepoch1danresumemenghasilkanhistory/bobotbit-identikdenganrun2epochtanpainterupsi. Horizon,totalsteps,warmupdanLRnonzerodiakhirmengikutijadwal,tidakmenjadihorizon2.
4. FixtureCVacuan dibuatdenganbenar-benar menjalankannotebookE43/E45lama pada24datasintetis/modelacak4/12layer,satuepoch,limaCV,saturefit/test,termasukresumekernel. Inisetupfixturelama,bukanCVbaruyangdijalankanE46/E47. Verifier baru menerimaZIP tersebutdanmenolakhashsalah,teksberubah,anggotaFoldberubah,LRberbeda,medianepochberbedadanhistoryhilang. Pembaca dicatatuntukmemastikanmetrik/prediksitestlama tidakdibacauntukmemilihresift.
5. Masing-masingnotebookE46/E47dijalankanseutuhnyadalamkernelterisolasi. Dataset24/tokenpendek/CPU,sourceCVfixture1epoch,horizonfixture1danrefit1dipakaiuntukend-to-end. Sourcepubliktetap300baris/CVlama25/horizon25/refit18atau7. Spy memastikan setiapfitbaru mempunyai`validation=None`,hanyasaturefitbaru,0CVbaru,satutest. LimaExcel/ZIPSHAbenar,checkpointfinaltidakberubahpadakernelbaru. X/yterlihatcocokinputtraining,tanpaimportproyek/workertersembunyi.
6. Keduatabelyangdulukosongkini1barismetriktestdan1barisringkasanCV. Currenthistory/bobotcampuranhanyarefitbaru,historyCVlama terpisah. VectorBGEbaruidentikdenganacuan;OOFacuandisalinhanyapadaE47. PaketZIPtepat5Excel;gateunduhanstatusPASSED.
7. Setelahoutputvaliddibuat,tesmengosongkanPerbandinganModel,laluRingkasanValidasi,mengubahprediksitest,menghapusbarisprediksi,danmerusakangkaLR. Masing-masingditolakdenganRuntimeErrorUnduhanDihentikan. Gatebukansekadarcekfileada:metricdihitungulang,ID/countdanjadwalLRdiperiksa. Build_reportsmemanggilgate sebelumZIPdibuatdanmengembalikanbuktiverifikasi;notebookbaru hanyamengunduhsetelahitulolos.

Terpisahdarifixture,schemaZIPasliE43/E45yangdiunggahpenggunadiperiksamelaluiverifierbaru:300baris,hashsesuaipin,median18/7lolos. Data/partisidirekonstruksidarifinalExceluntukpemeriksaanskemaini;**bukanvalidasiXLSXasliatautraining300baris**. Buktidi`actual_archive_schema_check.json`.

## Batas bukti dan publikasi

Python3.12.14,torch2.6.0+cpu,transformers4.44.2,pytest9.2.0,nbformat5.11.1,nbclient0.11.0,NumPy2.3.5,pandas2.2.3,SciPy1.17.0,scikit-learn1.8.0,openpyxl3.1.5. Jupyter memerlukansocketlocalhost. Tidakmengklaimsemuabugmustahilterjadi,semuaruntimeGPUkompatibel,ataudurasifixturemewakilitrainingT4. Gate mencakupbugringkasanterdahuludanbeberapakerusakanhasilnyata.

ZIPacuan tidakmerekamsemuarevisimodel/versiruntime/parameter;verifikasihanyapadafieldyangtersimpan,ditambahidentitashashasli. TidakadaklaimbaruatasCVyangdipakaiulang,danhasiltesthistoristetapeksploratif.

Hanyanotebookkosong,builder/helperyangdiembedsebagaiPythonterlihat,tesdandokumentasiaman dipublikasikan. Dataset,ZIPsiswa,executednotebook,Excel,bobot/cache/log tetapprivat. Setelahpush,SHAbranchdanbytekeduanotebookutama/mirror/metodediperiksa;catatandi`outputs/e46_e47_study/publication.json`.
