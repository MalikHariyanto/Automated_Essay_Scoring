"""E54: pinned E43 architecture + same-question ConR; standalone native Colab."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
PIN = '7241a21dfec37c942948b754da3960fba4c7731c'
BASE = 'notebooks/experiments/AES_300_E43_E42_S3_MaskedMeanPooling_Native_Learnable_GPU.ipynb'
REPORT_PIN = '83c2d213c6c1b8531d741f0225cc6c237c36946d'
REPORT_BASE = 'notebooks/experiments/AES_300_E53_E43_S3_EMA099_Native_Learnable_GPU.ipynb'
NAME = 'AES_300_E54_E43_S3_ConR_Native_Learnable_GPU'
PLACEHOLDER = '__NOTEBOOK_CODE_SHA256__'
CONR_KEYS = ('loss_recipe','conr_weight','conr_window_points','conr_temperature','conr_eta_per_point','conr_recompute')

INTRO = '''# E54 · E43 + ConR · S3 Native Learnable

**GPU T4 → Run all → upload `fix_dataset_300.xlsx`.** Cukup Excel asli, tanpa ZIP hasil lama/Drive/clone repo. Python biasa per section; tidak ada worker tersembunyi.

**Satu perlakuan terhadap E43: `loss = MSE + 0,001 × ConR`.** Semua arsitektur, LR, dropout, tiga input dan label guru tetap. RoBERTa full fine-tuning5e-6 → ScalarMix4/gamma → BiGRU64/arah → co-attention satu head → masked mean; BGE frozen1024/proyeksi256; concat1024 → MLP → satu skor. Tidak memakai EMA E53, DoRA E52, gate, multihead, LDS atau ListNet. Tidak ada parameter model baru.

ConR mengatur **fitur concat1024 sebelum MLP**. Pasangan positif: jawaban berbeda, soal bersih persis sama, beda nilai guru ≤5poin. Hard negative: soal sama, beda nilai >5poin, tetapi beda prediksi ≤5poin. Anchor harus mempunyai positif dan hard negative; bila tidak, kontribusi ConR-nya nol, MSE tetap berjalan. Kedekatan fitur memakai cosine similarity hanya dalam loss training, **bukan rumus nilai prediksi**. Inferensi tetap satu jawaban→MLP→satu angka tanpa nilai guru/pasangan.

| Pengaturan | E54 |
|---|---|
| X / y | soal, jawaban_referensi, response_text / satu rater0–100 |
| Loss | MSE target0–1 +0,001×ConR; ConR mengonversi jarak label/prediksi ke poin0–100 |
| ConR | window5poin, temperature0,07, eta0,01 per poin; uniform weights, tanpa inverse frequency |
| LR | encoder5e-6; proyeksi/BiGRU/matching/norm/ScalarMix/MLP1e-4 |
| Batch | efektif8/micro1; sampler E43 tetap; tanpa memory bank atau pasangan validation/test |
| CV | S3 saja, satu5fold baru, maksimal25epoch/patience5; seed42 tanpa sweep/kontrolulang |
| Pemilihan/refit | RMSE validasi biasa; ceil median bestepoch CV; satu refit dan satu test |

Bobot0,001/window5 adalah pilihan awal proyek, bukan optimum paper. Baca nilai ini sebagai tambahan loss, bukan perubahan skor guru. ConR diadaptasi dari **Keramati, Meng & Evans, ICLR2024**, [paper](https://openreview.net/forum?id=RIuevDSK5V), [kode sumber](https://github.com/BorealisAI/ConR). Bukti asli terutama regresi visual; bukan jaminan peningkatan AES Indonesia. Adaptasi ini membatasi pasangan ke soal sama, memakai satu view, uniform weights, aritmetika log-space dan mask anchor kosong. Denominator sumber yang memasukkan exp(0)=1 pada posisi masked dipertahankan dalam tiap kelompok soal. Temperatur dihormati, bukan ditimpa konstanta di dalam fungsi.

**Colab dan stabilitas:** ConR membutuhkan beberapa graph prediksi bersama, bukan accumulation microbatch terpisah. `CONR_RECOMPUTE=True` menghitung ulang aktivasi saat backward, dengan RNG dropout dan perangkat CUDA dipertahankan. Ini menambah kerja komputasi; puncak memori/waktu T4 aktual belum diverifikasi. Log menampilkan loss dan cakupan tiap epoch. Batch tanpa pasangan aktif sah dan mendapat ConR0; tidak ada pembagian0/NaN. Preflight memakai label TRAINING masing-masing fold/refit sebelum unduh encoder. Sampler tidak diubah diam-diam. Run ditolak bila sebuah fit tidak pernah mengaktifkan ConR; zero-hard-negative pada satu epoch saja bisa sah jika pasangan sudah terpisah.

**Lima Excel:**04 memuat60test/240OOF terpisah, metrik mentah, normalisasi/bulat, selisih tepat dan toleransi. 05 memuat history MSE, **Diagnostik ConR** per epoch (loss terpisah/gabungan, anchor potensial/aktif, pasangan, cakupan), konfigurasi, CV dan ScalarMix. ZIP baru hanya dibuat bila metrik, OOF, loss dan cakupan terserialisasi lolos gate. Resume lengkap memulihkan optimizer/scheduler/scaler/loader/RNG dan menolak resep berbeda. Tidak mengganti label, membetulkan bias manual atau tuning pada test historis.

E53 terbaru memberi meanCV5,8627/test5,9063/MAEtest4,6625; perbaikan CV kecil belum terbawa ke test. **E54 memakai E43 sebagai kontrol resep** (meanCV5,8876/test historis5,1265), bukan menumpuk EMA. Belum ada hasil penelitian E54; target60/60 dalam±3/±5 belum dijamin.

Kode helper loss ConR yang diadaptasi di notebook ini beratribusi Royal Bank of Canada/BorealisAI dan dilisensikan [CC BY-NC-SA4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/); modifikasi dijelaskan di atas. Kode AES historis lainnya mempertahankan hak aslinya.
'''


def once(source, old, new):
    assert source.count(old) == 1, old[:150]
    return source.replace(old, new, 1)


def fit_source(s):
    begin = s.index("            for start in range(0, len(batch), cfg['micro_batch']):")
    end = s.index('            scaler.unscale_(optimizer)', begin)
    s = s[:begin]+'''            loss, mse, conr, counts = training_batch_loss(engine, batch, bundle, cfg, device, dtype, use_cuda)
            if not torch.isfinite(loss):
                raise RuntimeError('Loss E54 nonfinite; run dihentikan.')
            scaler.scale(loss).backward()
            loss_total += mse.detach().item()*len(batch)
            conr_total += conr.detach().item()*len(batch)
            combined_total += loss.detach().item()*len(batch)
            for key, value in counts.items():
                conr_counts[key] += value
            conr_active_batches += int(counts['conr_active_anchors'] > 0)
            del loss, mse, conr
'''+s[end:]
    s = once(s,'        loss_total = 0.\n', '''        loss_total, conr_total, combined_total = 0., 0., 0.
        conr_counts = dict(conr_answers=0, conr_positive_anchors=0, conr_eligible_anchors=0,
                           conr_active_anchors=0, conr_positive_pairs=0, conr_hard_negative_pairs=0)
        conr_active_batches, conr_successful_batches = 0, 0
''')
    s = once(s,"        row = {'epoch': epoch, 'train_mse_scaled': loss_total/len(training),", """        row = {'epoch': epoch, 'train_mse_scaled': loss_total/len(training),
               'train_conr':conr_total/len(training), 'train_combined':combined_total/len(training),
               'conr_weight':cfg['conr_weight'], **conr_counts, 'conr_active_batches':conr_active_batches,
               'conr_successful_batches':conr_successful_batches,""")
    s = once(s, '                active_steps += int(lr_active)', '''                active_steps += int(lr_active)
                conr_successful_batches += int(lr_active and counts['conr_active_anchors'] > 0)''')
    s = once(s,'        history.append(row)', '''        message += (f" | ConR: {row['train_conr']:.6f} | loss gabungan: {row['train_combined']:.6f}"
                    f" | anchor aktif: {row['conr_active_anchors']}/{row['conr_answers']}")
        history.append(row)''')
    s = once(s,'    if active_steps == 0:', '''    if cfg['conr_weight'] > 0 and sum(r['conr_successful_batches'] for r in history) == 0:
        raise RuntimeError('ConR tidak pernah aktif pada fit ini; tidak mengekspor hasil seolah perlakuan berhasil.')
    if active_steps == 0:''')
    s = once(s,"    info['scalar_mix_layers'] = cfg['scalar_mix_layers']", "    info['scalar_mix_layers'] = cfg['scalar_mix_layers']\n    info['loss_recipe'] = cfg['loss_recipe']\n    info['conr_config'] = {k:cfg[k] for k in "+repr(CONR_KEYS)+"}")
    return s


def build():
    raw = subprocess.check_output(['git','show',PIN+':'+BASE],cwd=ROOT)
    report_raw = subprocess.check_output(['git','show',REPORT_PIN+':'+REPORT_BASE],cwd=ROOT)
    n = copy.deepcopy(json.loads(raw)); report = json.loads(report_raw)
    reports = {c['metadata'].get('aes53_role'):c for c in report['cells']}
    # Reporting comes from the verified E53 gate, without EMA history/metadata.
    report_roles = {'definition:evaluation_report','definition:build_reports','training_execute','download_execute'}
    for c in n['cells']:
        role = c['metadata'].pop('aes43_role','')
        if role:c['metadata']['aes54_role'] = role
        source = reports[role]['source'] if role in report_roles else c['source']
        s = source if isinstance(source,str) else ''.join(source)
        original = report if role in report_roles else json.loads(raw)
        s = s.replace(original['metadata']['aes_standalone']['code_sha256'],PLACEHOLDER)
        s = s.replace(Path(BASE).stem,'__NEW_NAME__').replace(Path(REPORT_BASE).stem,'__NEW_NAME__')
        s = s.replace('E43','E54').replace('E53','E54').replace('__NEW_NAME__',NAME)
        s = s.replace('validate_e53_download','validate_e54_download')
        if role == 'imports':
            s = s.replace('from torch.utils.data import DataLoader','from torch.utils.data import DataLoader\nfrom torch.utils.checkpoint import checkpoint as activation_checkpoint')
        if role == 'dependency_setup':s = s.replace('/tmp/aes35_native_constraints.txt','/tmp/aes54_native_constraints.txt')
        if role == 'hyperparameters':
            s += '\n# SATU PERLAKUAN E54: ConR sebagai pendamping MSE, tanpa EMA.\nCONR_WEIGHT = 0.001\nCONR_WINDOW_POINTS = 5.0\nCONR_TEMPERATURE = 0.07\nCONR_ETA_PER_POINT = 0.01\nCONR_RECOMPUTE = True\n'
        if role == 'configuration_defaults':
            s = once(s,"    's3_fusion': 'concat',", "    'loss_recipe':'mse_same_question_conr_v1', 'conr_weight':CONR_WEIGHT,\n    'conr_window_points':CONR_WINDOW_POINTS, 'conr_temperature':CONR_TEMPERATURE,\n    'conr_eta_per_point':CONR_ETA_PER_POINT, 'conr_recompute':CONR_RECOMPUTE,\n    's3_fusion': 'concat',")
            s = once(s,'    return result\n','    validate_conr_config(result)\n    return result\n')
        if role == 'configuration_apply':s = s.replace('e43_native_masked_mean_v1','e54_native_same_question_conr_v1')
        if role == 'presentation_helpers':
            s = '\n'.join(line for line in s.splitlines() if "LABELS.update({'fusion_gate':" not in line)+'\n'
        if role == 'split_execute':s += "\nif RUN_TRAINING:\n    preflight = conr_preflight(bundle,manifest,cfg)\n    show(pd.DataFrame(preflight),'Kelayakan ConR — hanya label training')\n    dump(output/'conr_preflight.json',preflight)\n"
        if role == 'definition:SingleScoreModel':
            s = once(s,'return_features=False, answer_mask=None, reference_mask=None):','return_features=False, answer_mask=None, reference_mask=None, return_conr_features=False):')
            s = once(s,'        return (prediction, features) if return_features else prediction', "        if return_conr_features:\n            return prediction, features['concat']\n        return (prediction, features) if return_features else prediction")
        if role == 'definition:Engine':
            s = once(s,'def forward(self, indices, return_features=False):','def forward(self, indices, return_features=False, return_conr_features=False):')
            s = once(s,'return_features=return_features, answer_mask=mask, reference_mask=ref_mask)', 'return_features=return_features, answer_mask=mask, reference_mask=ref_mask,\n                          return_conr_features=return_conr_features)')
        if role == 'definition:fit':s = fit_source(s)
        if role == 'definition:load_model':
            s = once(s,"'s3_pooling', 's3_fusion'):","'s3_pooling', 's3_fusion', "+', '.join(repr(k) for k in CONR_KEYS)+'):' )
        if role == 'definition:training_report':
            # E50 reporting improvements, but no multihead architecture/report.
            r50 = json.loads(subprocess.check_output(['git','show','c36003670ad9476951b8b607a6018e5a24ceb7e2:notebooks/experiments/AES_300_E50_E43_S3_MultiHeadCoAttention4_Native_Learnable_GPU.ipynb'],cwd=ROOT))
            s = next(''.join(x['source']) for x in r50['cells'] if x['metadata'].get('aes50_role')==role)
            s = s.replace("'coattention_heads':'Jumlah Head Co-Attention',",'').replace('sheets.update(multihead_report_tables(source))','sheets.update(conr_export_tables(source))')
            s = once(s,"settings={", "settings={'loss_recipe':'Resep Loss','conr_weight':'Bobot ConR','conr_window_points':'Window ConR (poin)',\n              'conr_temperature':'Temperature ConR','conr_eta_per_point':'Eta ConR per poin','conr_recompute':'Recompute ConR',")
        if c['cell_type'] == 'markdown':
            if '## 5 · Hyperparameter' in s:
                s = '''<a id="hyperparameter"></a>
## 5 · Hyperparameter E54

Seluruh LR/arsitektur tetap E43. Perlakuan baru: MSE + CONR_WEIGHT(0,001) × ConR; window5poin, temperature0,07, eta0,01 per poin. Recompute mempertahankan graph batch efektif8 melalui microbatch1. Angka awal ini belum terbukti optimal; label guru tidak diubah.
'''
            elif '## 22 ·' in s:
                s = s.replace('loss MSE dan checkpoint','loss MSE + ConR dan checkpoint')
            elif '## Rujukan dan batas kesimpulan' in s:
                s = s.replace('E54','kontrol E43')
                s += '\n**Perlakuan E54:** [Keramati, Meng & Evans, ConR, ICLR2024](https://openreview.net/forum?id=RIuevDSK5V). Adaptasi same-question/uniform weights/point units/log-space; cosine hanya loss fitur training. Resep dan lisensi dijelaskan di pembuka serta cell helper. Tidak ada bukti bahwa resep ini optimal AES Indonesia.\n'
        c['source'] = s
    helper = (ROOT/'scripts/conr_native.py').read_text()
    at = next(i for i,c in enumerate(n['cells']) if c['metadata'].get('aes54_role')=='configuration_apply')
    n['cells'].insert(at,{'cell_type':'code','metadata':{'aes54_role':'definition:conr_helpers'},'source':helper,'outputs':[],'execution_count':None})
    gate = reports['definition:download_helpers']['source']; gate = gate if isinstance(gate,str) else ''.join(gate)
    gate = gate.replace('validate_e53_download','validate_e54_download')
    start = gate.index('        ema_proof=verify_ema_exports(');end = gate.index('\n    history=read',start)
    gate = gate[:start]+'''        conr_proof=verify_conr_exports(read(names[4],'Diagnostik ConR'),read(names[4],'Fusion (S3)'),
            read(names[4],'Ringkasan Lima Fold'),cfg)
    except ValueError as error:
        raise RuntimeError('Unduhan dihentikan: laporan ConR tidak cocok: '+str(error)) from error'''+gate[end:]
    gate = gate.replace('**ema_proof','**conr_proof').replace('OOF/EMA','OOF/ConR')
    at = next(i for i,c in enumerate(n['cells']) if c['metadata'].get('aes54_role')=='definition:fit')
    n['cells'].insert(at,{'cell_type':'code','metadata':{'aes54_role':'definition:download_helpers'},'source':gate,'outputs':[],'execution_count':None})
    n['cells'][0]['source'] = INTRO
    fingerprint = hashlib.sha256('\n'.join(c['source'] for c in n['cells'] if c['cell_type']=='code').encode()).hexdigest()
    for i,c in enumerate(n['cells']):
        c['source'] = c['source'].replace(PLACEHOLDER,fingerprint)
        c['id'] = f'e54-{i:03d}-'+hashlib.sha256(c['source'].encode()).hexdigest()[:8]
        if c['cell_type']=='code':ast.parse(c['source']);c['outputs']=[];c['execution_count']=None
    n['metadata']['colab']['name'] = NAME+'.ipynb'
    n['metadata']['aes_standalone']['code_sha256'] = fingerprint
    n['metadata']['aes_experiment'].update(name='E54',format_commit=PIN,format_path=BASE,format_sha256=hashlib.sha256(raw).hexdigest(),
        reporting_commit=REPORT_PIN,loss='MSE + same-question ConR',conr_weight=.001,conr_window_points=5.,conr_temperature=.07,
        conr_eta_per_point=.01,conr_recompute=True,added_parameters_vs_e43=0,model_parameters_unchanged_vs_e43=True,
        prediction_breakdown=True,oof_predictions=True,requires_previous_zip=False,download_validation_required=True,
        treatment='E43 model and original sampler; same-question ConR on fusion features, uniform weights, no EMA')
    text = json.dumps(n,ensure_ascii=False,indent=1)+'\n'
    for folder in ['notebooks/experiments','aes-learning/notebooks/experiments']:
        (ROOT/folder/(NAME+'.ipynb')).write_text(text)
    print(json.dumps({'name':NAME,'cells':len(n['cells']),'code_sha256':fingerprint}))
    return n


if __name__ == '__main__':
    build()
