"""Matched batch-control (E55) and ConR (E56), with completed-fit saves only."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
PIN = '9e09b5c898d4e165e43516cd3bdf606220501dc8'
BASE = 'notebooks/experiments/AES_300_E54_E43_S3_ConR_Native_Learnable_GPU.ipynb'
PLACEHOLDER = '__NOTEBOOK_CODE_SHA256__'


def once(s,old,new):
    assert s.count(old)==1,old[:120]
    return s.replace(old,new,1)


def fit_source(s):
    s = once(s,', checkpoint=None):','):')
    start=s.index('    generator = torch.Generator()');end=s.index('    warmup = ',start)
    s=s[:start]+"    total_steps = math.ceil(len(training)/cfg['effective_batch'])*epochs\n"+s[end:]
    start=s.index('    start_epoch = 1');end=s.index('        model.train()',start)
    s=s[:start]+'''    for epoch in range(1, epochs+1):
        plans, plan_info = paired_batch_plan(training,bundle,cfg,seed,epoch)
'''+s[end:]
    s=once(s,'        for effective in loader:\n            batch = effective.tolist()', '        for batch in plans:')
    s=once(s,"        row = {'epoch': epoch,", "        row = {'epoch': epoch, **plan_info, 'epoch_checkpoints':False,")
    s=once(s,"int(lr_active and counts['conr_active_anchors'] > 0)","int(cfg['conr_weight'] > 0 and lr_active and counts['conr_active_anchors'] > 0)")
    start=s.index('        if checkpoint is not None:');end=s.index('        if early_stopped:',start)
    s=s[:start]+s[end:]
    s=once(s,"    info['loss_recipe'] = cfg['loss_recipe']", "    info['loss_recipe'] = cfg['loss_recipe']\n    info['batch_recipe'] = cfg['batch_recipe']\n    info['epoch_checkpoints'] = False")
    return s


def train_all_source(s):
    s=once(s,",\n                                          checkpoint=destination/f'progress_fold_{fold[\"fold\"]}.json')",')')
    s=once(s,", checkpoint=output/scenario/'progress_refit.json')",')')
    s=once(s,'                dump(file, result)', '''                # Save the selected model ONCE, only after this fold completes.
                model_path = destination/f'cv_fold_{fold["fold"]}.pt'
                save_model(model_path,model,cfg,bundle,manifest,info)
                result['model_sha256'] = digest(model_path)
                dump(file, result)''')
    s=once(s,"                print(f'{scenario}: memakai hasil validasi silang", """                model_path = destination/f'cv_fold_{fold["fold"]}.pt'
                if not model_path.is_file() or digest(model_path) != result.get('model_sha256'):
                    raise ValueError('Model fold selesai hilang/berubah; cache tidak dipakai.')
                print(f'{scenario}: memakai hasil validasi silang""")
    return s


def introduction(number,name,weight):
    other=56 if number==55 else 55
    loss='MSE saja (ConR berbobot0, hanya diagnostik)' if number==55 else 'MSE +0,001×ConR'
    return f'''# E{number} · batch pasangan soal sama · {loss}

**GPU T4 → Run all → upload `fix_dataset_300.xlsx`.** Jalankan E55 dan E56 pada dua runtime GPU terpisah; folder/ZIP masing-masing berbeda. Cukup Excel asli, tanpa ZIP hasil lama atau Drive. Python native per section.

**Pasangan eksperimen:** E55 adalah kontrol MSE; E56 memakai batch yang sama dengan tambahan ConR. Arsitektur, inisialisasi, LR, dropout dan split tetap E43/E54: RoBERTa full fine-tuning5e-6, ScalarMix4/gamma, BiGRU64/arah, co-attention satu head, masked mean, BGE frozen/proyeksi, concat1024→MLP→satu skor. Tidak menambah parameter model, EMA, DoRA, LDS, ListNet atau multihead. E54 lama tetap tersedia.

## Apa yang diubah?

1. **Batch TRAINING disusun dengan label guru training saja.** Pilih acak anchor yang mempunyai teman dekat(≤5poin) dan jauh(>5poin) dalam soal bersih persis sama. Ambil anchor+teman dekat+jauh dan bila tersedia satu teman soal sama lagi. Dua kelompok hingga4 jawaban membentuk batch8. Sisa jawaban diacak untuk mengisi batch. Tidak ada pengulangan, pembuangan, oversampling atau perubahan label; tiap jawaban training dipakai tepat sekali per epoch.
2. **E{number}: {loss}.** E55 mencatat ConR sebagai diagnostik dengan koefisien0; gradien yang dioptimasi hanya MSE. E56 koefisien0,001. Rumus ConR E54 dipertahankan: fitur concat sebelum MLP, pasangan hanya soal sama, window5poin/temperature0,07/eta0,01perpoin, uniform weights. Cosine hanya similarity loss training, bukan skor inferensi.
3. **Tidak ada checkpoint progres tiap epoch.** Bobot validasi terbaik tetap di RAM CPU. Model dipulihkan ke epoch terbaik dan baru ditulis setelah fold selesai; model refit ditulis setelah refit selesai. Jika Colab putus di tengah fit, fit itu diulang dari awal. Fold/refit yang selesai dapat dipakai ulang dalam workspace yang masih tersedia; filesystem Colab yang dihapus tetap menghilangkan file lokal. Recompute aktivasi untuk menghemat GPU tetap aktif; ini berbeda dari checkpoint FILE dan tetap diperlukan untuk batch graph.

Sampler deterministik memakai RNG lokal(seed fit+1000003×epoch), tidak tergantung model, prediksi, koefisien loss atau metrik validasi. **Untuk input/split/fold/seed dan nomor epoch yang sama, rencana E55/E56 identik**, termasuk urutan microbatch. SHA256 rencana dicatat untuk perbandingan. Early stopping dapat memilih jumlah epoch berbeda; jumlah epoch refit dan jadwal LR refit mengikuti aturan median CV masing-masing, sehingga rencana hanya dibandingkan pada epoch yang sama.

S3, satu5fold, maksimal25epoch/patience5, seed42, tanpa sweep/kontrol tambahan. MSE target0–1. ConR menghitung jarak label/prediksi dalam poin0–100. Hard negative juga memerlukan beda prediksi≤5; pasangan potensial lebih banyak **tidak menjamin** anchor aktif atau akurasi meningkat. Tidak memakai prediksi atau label validation/test untuk memilih batch. Semua metrik validasi/test tanpa pembobotan.

**Output lima Excel:**04 mempertahankan60test/240OOF terpisah, skor mentah/normalisasi/bulat, selisih tepat/toleransi;05 menambah **Rencana Batch** (hash/seed/jumlah/unik/cakupan potensial/checkpoint epoch=False) dan **Diagnostik ConR** (MSE, ConR diagnostik, koefisien, gabungan, cakupan dan langkah optimizer ConR). E55 Langkah ConR Berhasil=0 karena koefisien0, walaupun anchor diagnostik bisa aktif. Gate memeriksa laporan sebelum ZIP. Model akhir fold disimpan sebagai cv_fold_N.pt, bukan snapshot optimizer tiap epoch.

Perbandingan E55 terhadap E43/E54 membantu mengamati perubahan batch; **E56 terhadap E55** mengisolasi penambahan objective ConR pada sampler baru dalam resep ini. Batch baru dapat mengubah hasil MSE ke arah baik atau buruk. Jangan menaikkan bobot ConR/LR/epoch berdasarkan test historis. Koefisien/window/sampler adalah hipotesis awal proyek, belum optimum AES. Belum ada hasil penelitian E55/E56 atau janji60/60 dalam±3/±5; GPU T4 aktual belum diverifikasi pada pengujian CPU teknis.

[Buka kode pasangan E{other}](https://github.com/MalikHariyanto/Automated_Essay_Scoring/blob/codex/aes-learning/notebooks/experiments/AES_300_E{other}_E43_S3_PairedBatch_{'MSE' if other==55 else 'ConR'}_Native_Learnable_GPU.ipynb).

Dasar ConR: [Keramati, Meng & Evans, ICLR2024](https://openreview.net/forum?id=RIuevDSK5V), [kode](https://github.com/BorealisAI/ConR). Kode helper ConR yang diadaptasi beratribusi Royal Bank of Canada/BorealisAI, [CC BY-NC-SA4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). Modifikasi E54: same-question, point units, log-space, empty-anchor masks, uniform weights. Sampler baru adalah adaptasi proyek, bukan sampler resmi paper atau klaim optimalitas. Kode AES historis mempertahankan hak aslinya.
'''


def build(number):
    assert number in (55,56)
    weight=0. if number==55 else .001
    name=f'AES_300_E{number}_E43_S3_PairedBatch_{"MSE" if number==55 else "ConR"}_Native_Learnable_GPU'
    raw=subprocess.check_output(['git','show',PIN+':'+BASE],cwd=ROOT)
    n=copy.deepcopy(json.loads(raw));old_hash=n['metadata']['aes_standalone']['code_sha256']
    n['cells']=[c for c in n['cells'] if c['metadata'].get('aes54_role') not in ['definition:save_training_progress','definition:load_training_progress']]
    for c in n['cells']:
        role=c['metadata'].pop('aes54_role','')
        if role:c['metadata'][f'aes{number}_role']=role
        s=c['source'] if isinstance(c['source'],str) else ''.join(c['source'])
        s=s.replace(old_hash,PLACEHOLDER).replace(Path(BASE).stem,'__PAIR_NAME__')
        s=s.replace('E54',f'E{number}').replace('e54',f'e{number}').replace('__PAIR_NAME__',name)
        if role=='hyperparameters':s=s.replace('CONR_WEIGHT = 0.001',f'CONR_WEIGHT = {weight}')
        if role=='configuration_defaults':
            s=once(s,"    's3_fusion': 'concat',", "    'batch_recipe':'same_question_near_far_once_v1', 'epoch_checkpoints':False,\n    's3_fusion': 'concat',")
            s=once(s,'    validate_conr_config(result)', "    if result['batch_recipe']!='same_question_near_far_once_v1' or result['epoch_checkpoints'] is not False or result['effective_batch']<3:\n        raise ValueError('Resep batch/checkpoint tidak cocok E55/E56.')\n    validate_conr_config(result)")
        if role=='definition:conr_helpers':
            s=s.replace("if cfg['conr_weight'] > 0 and eligible == 0:","if eligible == 0:")
            s=s.replace("conr, counts = same_question_conr(representation, targets, prediction,", "conr, counts = same_question_conr(representation if cfg['conr_weight'] else representation.detach(), targets, prediction,")
        if role=='definition:fit':s=fit_source(s)
        if role=='definition:fit' and number==55:
            s=s.replace(' | ConR: ', ' | ConR diagnostik (bobot0): ')
        if role=='definition:train_all':s=train_all_source(s)
        if role=='definition:load_model':
            s=once(s,"'conr_recompute'):","'conr_recompute', 'batch_recipe', 'epoch_checkpoints'):")
        if role=='definition:training_report':
            s=once(s,"settings={", "settings={'batch_recipe':'Resep Batch','epoch_checkpoints':'Checkpoint Tiap Epoch',")
            s=once(s,'    return sheets','    sheets.update(paired_batch_report(source))\n    return sheets')
        if role=='definition:download_helpers':
            s=once(s,"    history=read(names[4],'Fusion (S3)')", "    try:\n        batch_proof=verify_paired_batch_exports(read(names[4],'Rencana Batch'),read(names[4],'Diagnostik ConR'),cfg,bundle,manifest)\n    except ValueError as error:\n        raise RuntimeError('Unduhan dihentikan: rencana batch tidak cocok: '+str(error)) from error\n    history=read(names[4],'Fusion (S3)')")
            s=s.replace('**conr_proof,','**conr_proof, **batch_proof,')
        if c['cell_type']=='markdown':
            s=s.replace('MSE + CONR_WEIGHT(0,001)',f'MSE + CONR_WEIGHT({weight})')
            if '## 22 ·' in s:s=s.split('\n\n')[0]+'\n\nModel terbaik di RAM; file model hanya setelah fold/refit selesai. Recompute aktivasi tetap dipakai untuk menghemat GPU.\n'
            if '### `fit`' in s:s='### `fit`\n\nSatu fit dengan batch deterministik training-only; tidak ada save/load progres per epoch. Jika terputus, fit diulang.\n'
        c['source']=s
    at=next(i for i,c in enumerate(n['cells']) if c['metadata'].get(f'aes{number}_role')=='configuration_apply')
    n['cells'].insert(at,{'cell_type':'code','metadata':{f'aes{number}_role':'definition:paired_batch_helpers'},
        'source':(ROOT/'scripts/paired_batches_native.py').read_text(),'outputs':[],'execution_count':None})
    n['cells'][0]['source']=introduction(number,name,weight)
    fp=hashlib.sha256('\n'.join(c['source'] for c in n['cells'] if c['cell_type']=='code').encode()).hexdigest()
    for i,c in enumerate(n['cells']):
        c['source']=c['source'].replace(PLACEHOLDER,fp)
        c['id']=f'e{number}-{i:03d}-'+hashlib.sha256(c['source'].encode()).hexdigest()[:8]
        if c['cell_type']=='code':ast.parse(c['source']);c['outputs']=[];c['execution_count']=None
    n['metadata']['colab']['name']=name+'.ipynb';n['metadata']['aes_standalone']['code_sha256']=fp
    n['metadata']['aes_experiment'].update(name=f'E{number}',format_commit=PIN,format_path=BASE,format_sha256=hashlib.sha256(raw).hexdigest(),
        loss='MSE' if number==55 else 'MSE + same-question ConR',conr_weight=weight,
        batch_recipe='same_question_near_far_once_v1',epoch_checkpoints=False,
        treatment='Matched training-only near/far batch plan; each answer once; completed-fit saves only; '+('MSE control' if number==55 else 'ConR treatment'))
    text=json.dumps(n,ensure_ascii=False,indent=1)+'\n'
    for folder in ['notebooks/experiments','aes-learning/notebooks/experiments']:(ROOT/folder/(name+'.ipynb')).write_text(text)
    print(json.dumps({'name':name,'code_sha256':fp}))
    return n


if __name__=='__main__':
    build(55);build(56)
