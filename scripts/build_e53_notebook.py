"""Native E53: pinned E43 full fine-tuning, with ramped EMA for selection/inference."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
PIN = '7241a21dfec37c942948b754da3960fba4c7731c'
BASE = 'notebooks/experiments/AES_300_E43_E42_S3_MaskedMeanPooling_Native_Learnable_GPU.ipynb'
REPORT_PIN = 'c36003670ad9476951b8b607a6018e5a24ceb7e2'
REPORT_BASE = 'notebooks/experiments/AES_300_E50_E43_S3_MultiHeadCoAttention4_Native_Learnable_GPU.ipynb'
NAME = 'AES_300_E53_E43_S3_EMA099_Native_Learnable_GPU'
PLACEHOLDER = '__NOTEBOOK_CODE_SHA256__'

INTRO = '''# E53 · E43 + Exponential Moving Average (EMA) · S3 Native Learnable

**GPU T4 → Run all → upload `fix_dataset_300.xlsx`.** Cukup dataset Excel, tanpa ZIP hasil lama. Kode Python biasa terlihat per section: input, model, hyperparameter, EMA, training, evaluasi dan unduhan. Tanpa magic, worker tersembunyi, clone repo atau Drive.

**E53 kembali ke E43 utuh**, dengan satu perubahan metode: salinan bobot EMA digunakan untuk validasi, pemilihan epoch dan prediksi akhir. RoBERTa dilatih penuh lagi dengan LR `5e-6`; tidak memakai DoRA E52. BGE encoder tetap frozen. ScalarMix empat layer + gamma, BiGRU64 per arah, co-attention satu head, masked mean, concat dan MLP tetap E43.

## Input dan alur model

`soal`, `jawaban_referensi`, `response_text` adalah tiga kolom X yang saling terhubung melalui konteks encoder dan pencocokan respons–referensi. `rater` adalah y, skor guru 0–100, hanya untuk loss/seleksi/evaluasi. Label tidak masuk encoder atau rumus EMA.

Tiga teks → RoBERTa full fine-tuning → ScalarMix4 + gamma → BiGRU64/arah → co-attention dua arah + roundtrip → matching → masked mean → fitur768; BGE-M3 frozen1024 → proyeksi256; concat1024 → MLP256/GELU/dropout0,1 → satu skor. Nama checkpoint dan revision sama E43. Tidak ada multihead, fusion gate, atau encoder tambahan.

## Apa yang berubah saat training?

Bobot **online** adalah bobot model yang menerima gradien MSE dan diperbarui AdamW. Bobot **EMA** adalah salinan tanpa gradien yang mengikuti rata-rata bergerak:

```text
t = jumlah update optimizer yang berhasil dan LR-nya aktif
update pertama: bobot_EMA = bobot_online
update berikutnya:
    decay_t = min(0,99, (1+t)/(10+t))
    bobot_EMA = decay_t × bobot_EMA_lama + (1−decay_t) × bobot_online
```

**0,99 adalah batas maksimum, bukan decay tetap sejak awal.** Warmup offset10 memperkecil pengaruh bobot MLP acak di awal. Pada run pendek, decay aktual bisa belum mencapai0,99; semua nilai aktual dicatat. Batas/rumus warmup merupakan resep awal proyek, bukan angka optimal AES yang dibuktikan paper.

Salinan EMA meliputi seluruh parameter model S3: RoBERTa, BiGRU/matching, normalisasi, proyeksi BGE, ScalarMix/gamma dan MLP. Encoder BGE tidak dirata-ratakan karena frozen dan fiturnya dihitung sekali. Buffer posisi/token-type disalin langsung, tidak dirata-ratakan sebagai integer. EMA tidak menambah parameter yang dioptimasi. Tambahan memori diperlukan untuk satu salinan model; evaluasi online+EMA menambah forward validasi.

| Pengaturan | E53 |
|---|---|
| LR RoBERTa / komponen lain | `5e-6` / `1e-4`, kembali E43 |
| Loss | MSE biasa, target dinormalisasi0–1; loss training berasal dari model online |
| EMA | Batas decay0,99, warmup offset10; update per langkah optimizer berhasil dengan LR aktif |
| Seleksi epoch | **RMSE validasi EMA**, bukan hasil online atau test |
| Protokol | S3 saja, satu5fold CV baru, maksimal25epoch, patience5 |
| Refit / test | Ceil median best epoch EMA limafold; satu refit development dan satu evaluasi test memakai EMA akhir refit |
| Lainnya | AdamW/decay0,01, warmup LR10%, linear scheduler sesuai durasi fit E43; microbatch1/batch efektif8, clip1 |

Satu seed dasar42, fold mengikuti acuan; tanpa loop banyak seed atau kontrol ulang E43. Tidak memakai LDS, ranking, cosine, S1/S2, scheduler refit E46/E47 atau Switch EMA. Bobot online tidak diganti EMA selama pembelajaran; model yang dikembalikan untuk prediksi memuat EMA terpilih.

## Output yang perlu dibaca

- **Excel05 → Perbandingan EMA:** RMSE validasi online dan EMA pada epoch yang sama, selisih EMA−online, jumlah update, decay aktual dan penanda checkpoint terpilih. Ini diagnostik dari satu lintasan training; bukan run kontrol E43 independen, sebab early stopping dipilih memakai EMA.
- **Excel05 → Konfigurasi EMA:** enam checkpoint fold/refit, jenis bobot prediksi, resep, epoch terpilih/dijalankan, update dan decay pada checkpoint terpilih serta pada akhir training. Keduanya dibedakan karena training bisa berlanjut melewati epoch terbaik.
- **Excel05 → Update EMA per Epoch:** langkah optimizer, LR aktif, AMP yang dilewati, update EMA dan decay. History utama melaporkan MSE online serta metrik validasi EMA; Bobot ScalarMix/gamma diambil setelah EMA terpilih dipulihkan.
- **Excel04:**60test,240OOF terpisah, skor mentah/normalisasi/pembulatan, selisih tepat0/1/2/3/4/5/10 dan lainnya, toleransi≤n, MSE/MAE/error maksimum. Seluruh prediksi S3 tersebut menggunakan bobot EMA.
- **Excel01/02/03:**input/tokenisasi, fitur model EMA final, BGE, kata/kalimat tetap. Inspeksi model refit terhadap300jawaban bukan evaluasi300data.

Checkpoint progres menyimpan bobot online, EMA, counter, best EMA, optimizer/scheduler/scaler, loader dan RNG. Resume tidak menghitung ulang EMA dari nol. Checkpoint final diberi jenis bobot `ema`; loader menolak jenis/resep yang berbeda. Report ulang membaca hasil tersimpan tanpa mengevaluasi test lagi. Penyimpanan progres lebih besar karena memuat dua set bobot; ZIP unduhan tetap hanya lima Excel.

Pemeriksaan sebelum ZIP membaca ulang metrik test/OOF, mean/SD CV, rekap selisih, dua RMSE per epoch, minimum RMSE EMA yang dipilih, counter/update aktif dan rumus decay. Tabel wajib kosong atau metadata tidak cocok menghentikan pembuatan ZIP baru. Source/hasil/HTML lama tetap tersedia.

## Referensi dan batas bukti

Morales-Brotons, Vogels & Hendrikx, **[Exponential Moving Average of Weights in Deep Learning: Dynamics and Benefits, TMLR2024](https://arxiv.org/abs/2411.18704)**. [PyTorch AveragedModel/EMA](https://docs.pytorch.org/docs/stable/generated/torch.optim.swa_utils.AveragedModel.html) menyediakan rujukan implementasi recurrence EMA. E53 menampilkan rumus/update/resume sebagai kode Python langsung, bukan menyalin seluruh rancangan paper. Warmup dan batas decay adalah adaptasi proyek. Ini EMA biasa; tidak menyalin bobot EMA ke model online pada setiap epoch seperti Switch EMA.

Acuan E43: meanRMSE CV5,8876, MAE OOF4,6122, selisih mentah≤5 sebanyak151/240; RMSE test historis5,1265. **Belum ada hasil penelitian E53.** Pilih melalui validasi; test historis bersifat eksploratif. Paper bukan bukti langsung bahwa EMA memperbaiki AES Indonesia atau menjamin60/60dalam±3/±5.
'''


def once(source, old, new):
    assert source.count(old) == 1, old[:150]
    return source.replace(old, new, 1)


def ema_fit(source):
    source = once(source, "    groups = optimizer_groups(model, cfg)", '''    ema = ExponentialWeightAverage(model, cfg)
    ema_engine = Engine(ema.model, bundle.texts, bge, cfg, device,
                        roberta_tokenizer=engine.tokenizer, contexts=engine.contexts)
    groups = optimizer_groups(model, cfg)''')
    source = once(source, "        model.load_state_dict(saved['model'])", '''        model.load_state_dict(saved['model'])
        if 'ema_state' not in saved:
            raise ValueError('Checkpoint progres E53 tidak memuat bobot/counter EMA.')
        ema.load_state_dict(saved['ema_state'])
        if ema.updates != saved['active_steps']:
            raise ValueError('Counter EMA berbeda dari langkah optimizer LR aktif.')''')
    source = once(source, "                active_steps += int(lr_active)", '''                active_steps += int(lr_active)
                ema.update_after_optimizer(model, lr_active=lr_active, skipped=False)''')
    source = once(source, "        row = {'epoch': epoch,", '''        if ema.updates == 0 or ema.updates != active_steps:
            raise RuntimeError('Belum ada update EMA yang sah atau counter berbeda dari LR aktif.')
        row = {'epoch': epoch, 'ema_updates':ema.updates,
               'ema_decay':ema_decay_at(ema.updates,ema.maximum,ema.offset),''')
    source = once(source, "            row['validation'] = metrics(bundle.y[validation], predicted, cfg)", '''            row['validation_online'] = metrics(bundle.y[validation], predicted, cfg)
            averaged_prediction = ema_engine.predict(validation)
            row['validation'] = metrics(bundle.y[validation], averaged_prediction, cfg)''')
    source = once(source, "for k, v in model.state_dict().items()}", "for k, v in ema.model.state_dict().items()}")
    source = source.replace(" | RMSE validasi: {row['validation']['RMSE']:.4f} poin", " | RMSE EMA: {row['validation']['RMSE']:.4f} poin | online: {row['validation_online']['RMSE']:.4f} poin")
    source = once(source, "'contract': contract, 'epoch': epoch, 'model': model.state_dict(),", "'contract': contract, 'epoch': epoch, 'model': model.state_dict(), 'ema_state':ema.state_dict(),")
    source = source.replace("f'Bobot epoch {best_epoch} dipulihkan.'", "f'Bobot EMA epoch {best_epoch} dipulihkan.'")
    source = once(source, "        model.load_state_dict(best_state)\n", "        model.load_state_dict(best_state)\n    else:\n        model.load_state_dict(ema.model.state_dict())\n    model.inference_weights_kind = 'ema'\n")
    source = source.replace("'minimum validation RMSE in original score units'", "'minimum EMA validation RMSE in original score units'")
    source = once(source, "    info['scalar_mix_layers'] = cfg['scalar_mix_layers']", "    info['scalar_mix_layers'] = cfg['scalar_mix_layers']\n    info['inference_weights'] = 'ema'\n    info['ema'] = ema.report(history,best_epoch)")
    return source


def build():
    raw = subprocess.check_output(['git','show',PIN+':'+BASE],cwd=ROOT)
    report_raw = subprocess.check_output(['git','show',REPORT_PIN+':'+REPORT_BASE],cwd=ROOT)
    n = copy.deepcopy(json.loads(raw)); report = json.loads(report_raw)
    reports = {c['metadata'].get('aes50_role'):c for c in report['cells']}
    report_roles = {'definition:evaluation_report','definition:training_report','definition:build_reports',
                    'download_execute','training_execute','definition:export_training_event'}
    for cell in n['cells']:
        role = cell['metadata'].pop('aes43_role','')
        if role: cell['metadata']['aes53_role'] = role
        origin = report if role in report_roles else json.loads(raw)
        source = reports[role]['source'] if role in report_roles else cell['source']
        source = source if isinstance(source,str) else ''.join(source)
        source = source.replace(origin['metadata']['aes_standalone']['code_sha256'],PLACEHOLDER)
        for name in [Path(BASE).stem,Path(REPORT_BASE).stem,'AES_300_E50_E50_S3_MultiHeadCoAttention4_Native_Learnable_GPU']:
            source = source.replace(name,'__E53_NAME__')
        source = source.replace('E43','E53').replace('E50','E53').replace('__E53_NAME__',NAME)
        if role == 'imports': source = 'import copy\n'+source
        if role == 'dependency_setup': source = source.replace('/tmp/aes35_native_constraints.txt','/tmp/aes53_native_constraints.txt')
        if role == 'presentation_helpers':
            source = '\n'.join(line for line in source.splitlines() if "LABELS.update({'fusion_gate':" not in line)+'\n'
        if role == 'hyperparameters':
            source += "\n# EMA adalah salinan bobot evaluasi; model online tetap dilatih penuh seperti E43.\nEMA_DECAY_MAX = 0.99\nEMA_WARMUP_OFFSET = 10\nINFERENCE_WEIGHTS = 'ema'\n"
        if role == 'configuration_defaults':
            source = once(source,"    's3_fusion': 'concat',", "    'ema_decay_max':EMA_DECAY_MAX, 'ema_warmup_offset':EMA_WARMUP_OFFSET,\n    'ema_update_policy':'successful_active_optimizer_step', 'inference_weights':INFERENCE_WEIGHTS,\n    's3_fusion': 'concat',")
            source = once(source,'    return result\n','    validate_ema_config(result)\n    return result\n')
        if role == 'configuration_apply':
            source = source.replace('e43_native_masked_mean_v1','e53_native_ema099_v1')
            source += "\nprint('Bobot prediksi:',cfg['inference_weights'],'| batas decay EMA:',cfg['ema_decay_max'],'| offset warmup:',cfg['ema_warmup_offset'])\n"
        if role == 'definition:fit': source = ema_fit(source)
        if role == 'definition:save_model':
            source = once(source,"    payload = {'scope': SCOPE,", "    if getattr(model,'inference_weights_kind',None)!='ema' or info.get('inference_weights')!='ema':\n        raise ValueError('Checkpoint final E53 harus memuat bobot EMA terpilih.')\n    payload = {'inference_weights':'ema', 'scope': SCOPE,")
        if role == 'definition:load_model':
            source = once(source,"    for key in ('roberta',", "    if payload.get('inference_weights')!='ema' or payload.get('training',{}).get('inference_weights')!='ema':\n        raise ValueError('Checkpoint final E53 bukan bobot EMA.')\n    for key in ('inference_weights','ema_decay_max','ema_warmup_offset','ema_update_policy','lesson_version','roberta',")
            source = once(source,'    model.eval()\n','    model.eval()\n    model.inference_weights_kind = \'ema\'\n')
        if role == 'architecture_export':
            source = source.replace("'output':1}","'output':1, 'inference_weights':'ema'}")
        if role == 'definition:export_training_event':
            source = once(source,"'Proyeksi_multihead':pd.DataFrame(training['multihead_projections']),", "'Konfigurasi_EMA':pd.DataFrame([training['ema']]),\n                                                           'Perbandingan_EMA':pd.DataFrame(ema_comparison_rows(training)),\n                                                           'Update_EMA':pd.DataFrame(ema_history_rows(training)),")
            source = once(source,"'Proyeksi_multihead':pd.DataFrame(result['multihead_projections'])", "'Konfigurasi_EMA':pd.DataFrame([result['ema']]),\n                                            'Update_EMA':pd.DataFrame(ema_history_rows(result))")
            source = source.replace("if k != 'validation'", "if k not in ('validation','validation_online')")
        if role == 'definition:training_report':
            source = once(source,"settings={'coattention_heads':'Jumlah Head Co-Attention',", "settings={'inference_weights':'Jenis Bobot Prediksi','ema_decay_max':'Batas Decay EMA','ema_warmup_offset':'Offset Warmup EMA','ema_update_policy':'Aturan Update EMA',")
            source = source.replace('sheets.update(multihead_report_tables(source))','sheets.update(ema_report_tables(source))')
            source = source.replace("'RMSE Validasi (poin)','Pearson Validasi','QWK Validasi','Gwet AC2 Validasi'", "'RMSE Validasi EMA (poin)','Pearson Validasi EMA','QWK Validasi EMA','Gwet AC2 Validasi EMA'")
        if role == 'definition:evaluation_report':
            source = source.replace(' + Multihead Co-Attention4',' + Co-Attention').replace(' + Masked Mean Pooling',' + Masked Mean Pooling (bobot EMA)')
        if role == 'definition:build_reports': source = source.replace('validate_e50_download','validate_e53_download')
        cell['source'] = source
    helper = (ROOT/'scripts/ema_native.py').read_text()
    helper = '\n\n'.join(ast.get_source_segment(helper,node) for node in ast.parse(helper).body if isinstance(node,(ast.FunctionDef,ast.ClassDef)))+'\n'
    index = next(i for i,c in enumerate(n['cells']) if c['metadata'].get('aes53_role')=='configuration_defaults')
    n['cells'][index:index] = [
        {'cell_type':'markdown','metadata':{},'source':'## EMA: salinan bobot tanpa gradien dan laporan perbandingan\n\nUpdate pertama menyalin bobot online; selanjutnya decay naik bertahap sampai batas0,99. Counter hanya bertambah setelah optimizer berhasil dengan LR aktif. State EMA disimpan saat resume.\n'},
        {'cell_type':'code','metadata':{'aes53_role':'definition:ema_helpers'},'source':helper,'outputs':[],'execution_count':None}]
    source = reports['definition:download_helpers']['source'].replace('validate_e50_download','validate_e53_download').replace('OOF/multihead','OOF/EMA')
    start = source.index("    projections=read(names[4],'Proyeksi Multihead')"); end = source.index("    history=read(names[4],'Fusion (S3)')",start)
    source = source[:start]+'''    try:
        ema_proof=verify_ema_exports(read(names[4],'Konfigurasi EMA'),read(names[4],'Perbandingan EMA'),
            read(names[4],'Update EMA per Epoch'),folds,cfg)
    except ValueError as error:
        raise RuntimeError('Unduhan dihentikan: laporan EMA tidak cocok: '+str(error)) from error
'''+source[end:]
    source = source.replace("'multihead_projections_verified':True,","**ema_proof,")
    source = source.replace("'coattention_heads':4,'multihead_projection_rows':len(projections),",'')
    index = next(i for i,c in enumerate(n['cells']) if c['metadata'].get('aes53_role')=='definition:fit')
    n['cells'][index:index] = [
        {'cell_type':'markdown','metadata':{},'source':'## Rincian prediksi dan pemeriksaan sebelum ZIP\n\nMetrik, OOF, kategori selisih, checkpoint EMA terpilih, jumlah update dan decay diverifikasi dari Excel tersimpan.\n'},
        {'cell_type':'code','metadata':{'aes53_role':'definition:download_helpers'},'source':source,'outputs':[],'execution_count':None}]
    n['cells'][0]['source'] = INTRO
    fingerprint = hashlib.sha256('\n'.join(c['source'] for c in n['cells'] if c['cell_type']=='code').encode()).hexdigest()
    for i,cell in enumerate(n['cells']):
        cell['source'] = cell['source'].replace(PLACEHOLDER,fingerprint)
        cell['id'] = f'e53-{i:03d}-'+hashlib.sha256(cell['source'].encode()).hexdigest()[:8]
        if cell['cell_type']=='code': ast.parse(cell['source']); cell['outputs']=[]; cell['execution_count']=None
    n['metadata']['colab']['name'] = NAME+'.ipynb'
    n['metadata']['aes_standalone']['code_sha256'] = fingerprint
    n['metadata']['aes_experiment'].update(name='E53',format_commit=PIN,format_path=BASE,format_sha256=hashlib.sha256(raw).hexdigest(),
        reporting_commit=REPORT_PIN,loss='MSE',ranking=False,inference_weights='ema',ema_decay_max=.99,ema_warmup_offset=10,
        full_encoder_fine_tuning=True,s3_encoder_lr=5e-6,prediction_breakdown=True,oof_predictions=True,
        requires_previous_zip=False,export_summary_closure_fixed=True,download_validation_required=True,
        treatment='E43 architecture/training MSE and full encoder LR5e-6; ramped EMA for validation checkpoint selection and final inference; online same-epoch validation diagnostics')
    text = json.dumps(n,ensure_ascii=False,indent=1)+'\n'
    for folder in ['notebooks/experiments','aes-learning/notebooks/experiments']:
        (ROOT/folder/(NAME+'.ipynb')).write_text(text)
    print(json.dumps({'name':NAME,'cells':len(n['cells']),'code_sha256':fingerprint}))
    return n


if __name__ == '__main__':
    build()
