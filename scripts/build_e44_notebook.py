"""Build native E44 from pinned E43: MSE + same-question ListNet, no model change."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
PIN = '7241a21dfec37c942948b754da3960fba4c7731c'
BASE = 'notebooks/experiments/AES_300_E43_E42_S3_MaskedMeanPooling_Native_Learnable_GPU.ipynb'
NAME = 'AES_300_E44_E43_S3_MSE_ListNet_Native_Learnable_GPU'
PLACEHOLDER = '__NOTEBOOK_CODE_SHA256__'

LOSSES = '''def same_question_listnet(prediction, targets, questions):
    """ListNet top-one: bandingkan hanya jawaban dalam soal yang persis sama."""
    if prediction.ndim != 2 or prediction.shape[1] != 1 or targets.shape != prediction.shape:
        raise ValueError('ListNet memerlukan prediction dan targets berbentuk [B,1].')
    if len(questions) != len(prediction) or not len(questions):
        raise ValueError('Satu soal diperlukan untuk setiap prediksi dalam batch.')
    if not torch.isfinite(prediction).all() or not torch.isfinite(targets).all():
        raise ValueError('Prediksi/target ListNet harus finite.')
    groups = {}
    for position, question in enumerate(questions):
        if not isinstance(question, str) or not question.strip():
            raise ValueError('Soal untuk ranking harus teks yang tidak kosong.')
        groups.setdefault(question, []).append(position)
    terms = []
    ranked_answers = 0
    for positions in groups.values():
        if len(positions) < 2:
            continue  # Singleton tetap dilatih lewat MSE; tidak dibuat pasangan beda soal.
        indices = torch.tensor(positions, dtype=torch.long, device=prediction.device)
        scores = prediction.index_select(0, indices).reshape(-1).float()
        teacher = targets.index_select(0, indices).reshape(-1).detach().float()
        # Skala 0–1 yang sama dengan MSE; tidak ada temperature atau label tambahan.
        target_probability = torch.softmax(teacher, dim=0)
        log_probability = torch.log_softmax(scores, dim=0)
        terms.append(-(target_probability * log_probability).sum())
        ranked_answers += len(positions)
    # Nol terhubung ke graph, sehingga singleton memberi gradien ranking tepat nol.
    loss = torch.stack(terms).mean() if terms else prediction.float().sum() * 0.
    return loss, {'ranking_groups': len(terms), 'ranking_answers': ranked_answers}


def forward_microbatch_for_recompute(device_marker, indices, engine):
    """Marker perangkat memastikan PyTorch menyimpan RNG dropout pada GPU juga."""
    return engine.forward(indices)


def training_batch_loss(engine, batch, bundle, cfg, device, dtype, use_cuda):
    """Satu objective batch efektif, dengan forward microbatch dan gradien end-to-end."""
    predictions = []
    # Tensor kosong ini bukan fitur model; checkpoint memakainya untuk mengenali GPU.
    device_marker = torch.empty(0, device=device)
    for start in range(0, len(batch), cfg['micro_batch']):
        ids = batch[start:start+cfg['micro_batch']]
        with torch.autocast(device_type=device, dtype=dtype, enabled=use_cuda):
            if cfg['ranking_recompute']:
                # Menghitung ulang aktivasi saat backward untuk menghemat memori.
                # Ini bukan penyimpanan file bobot. RNG dropout dipertahankan.
                value = activation_checkpoint(forward_microbatch_for_recompute, device_marker, ids, engine,
                    use_reentrant=False, preserve_rng_state=True)
            else:
                value = engine.forward(ids)
        predictions.append(value.float())  # Tidak detach: encoder sampai MLP menerima gradien.
    prediction = torch.cat(predictions, dim=0)
    targets = torch.as_tensor((bundle.y[batch]-cfg['score_min'])/(cfg['score_max']-cfg['score_min']),
                              dtype=torch.float32, device=device).reshape(-1, 1)
    if prediction.shape != targets.shape:
        raise ValueError('Shape prediksi/target harus [B,1], tanpa broadcasting.')
    mse = torch.mean((prediction-targets)**2)
    questions = [bundle.rows[index]['question'] for index in batch]
    ranking, counts = same_question_listnet(prediction, targets, questions)
    total = mse + cfg['ranking_weight'] * ranking
    return total, mse, ranking, counts
'''

NEW_BATCH = '''            # Kumpulkan prediksi microbatch dalam satu graph loss batch efektif.
            # Urutan batch tetap E43; kelompok soal hanya dibentuk di dalam loss.
            loss, mse, ranking, counts = training_batch_loss(
                engine, batch, bundle, cfg, device, dtype, use_cuda)
            if not torch.isfinite(loss):
                raise RuntimeError('Loss nonfinite; run dihentikan, tidak diklaim berhasil.')
            scaler.scale(loss).backward()
            loss_total += mse.detach().item()*len(batch)
            ranking_total += ranking.detach().item()*len(batch)
            combined_total += loss.detach().item()*len(batch)
            ranking_groups += counts['ranking_groups']
            ranking_answers += counts['ranking_answers']
            ranking_batches += int(counts['ranking_groups'] > 0)
            del loss, mse, ranking
'''

OOF_EXPORT = r'''    # Pasangan OOF untuk audit residual; seluruhnya dari VALIDASI development.
    questions = {r['row_id']:r.get('soal') for r in source.records('02_dataset_dan_cleaning.xlsx','Sebelum_sesudah')}
    oof_rows, seen = [], set()
    for file in sorted(source.files):
        match = re.fullmatch(r'09_S3_fold_(\d+)\.xlsx', file)
        if not match:
            continue
        for row in source.records(file, 'Prediksi_validation'):
            identifier = row['row_id']
            if identifier in seen or identifier in predicted:
                raise ValueError('ID OOF duplikat atau bercampur dengan data test.')
            seen.add(identifier)
            error = row['prediksi_mentah']-row['skor_guru']
            oof_rows.append([int(match[1]), friendly_value('row_id',identifier), questions.get(identifier),
                row['skor_guru'], row['prediksi_mentah'], row['kategori_QWK_AC2'], error, abs(error), error**2])
    if oof_rows:
        sheets['Prediksi Validasi OOF'] = Table(
            ['Fold','ID Jawaban','Soal','Nilai Guru','Prediksi S3','Pembulatan S3',
             'Selisih S3','Selisih Absolut','Selisih Kuadrat'], oof_rows)
'''

INTRO = '''# E44 · E43 + MSE dan ListNet · S3 Native Learnable

**Satu perlakuan: loss training menjadi `MSE + 0,1 × ListNet`.** Arsitektur dan seluruh LR tetap E43. MSE mengarahkan angka prediksi ke nilai guru; ListNet menambah pembelajaran urutan kualitas jawaban untuk soal yang sama. Ini bukan mengurutkan hasil setelah prediksi selesai.

RoBERTa → Scalar Mix empat layer + gamma → BiGRU 64 per arah + co-attention → masked mean → proyeksi 768; gabung proyeksi BGE 256 → concat 1024 → MLP → satu nilai. RoBERTa dan komponen tambahan tetap dilatih; BGE-M3 tetap frozen. Tidak ada parameter model baru.

| Pengaturan | Nilai |
|---|---|
| X / y | `soal`, `jawaban_referensi`, `response_text` / `rater` 0–100 |
| LR encoder / seluruh komponen lain | 5e-6 / 1e-4 |
| Loss | MSE + 0,1 × rerata ListNet kelompok soal yang memenuhi syarat |
| Skala target dan skor untuk kedua loss | 0–1; softmax ListNet tanpa temperature tambahan |
| Batch | Efektif 8, forward microbatch 1; sampler tetap E43 |
| Protokol | S3 saja, satu 5-fold, maksimal 25 epoch, patience 5 |
| Pemilihan | RMSE validasi skala 0–100; refit ceil(median best epoch), satu test |

**Ranking hanya menggunakan kelompok dengan ≥2 jawaban untuk teks soal yang persis sama di dalam batch efektif.** Singleton hanya menerima MSE. Tidak menyusun ulang dataset, mengganti split, atau menambah pasangan dari validation/test. Ties nilai guru menghasilkan distribusi target yang sama, bukan urutan buatan. Riwayat mencatat berapa kelompok/jawaban benar-benar menerima ranking; run dihentikan jika tidak ada kelompok ranking sepanjang suatu epoch.

Microbatch 1 tidak diperlakukan sebagai daftar ranking sendiri. Prediksi beberapa microbatch digabung dengan graph gradien yang utuh. `RANKING_RECOMPUTE=True` menghitung ulang aktivasi saat backward dengan RNG dropout yang sama untuk mengurangi memori; ini tidak menyimpan file model dan tidak mengubah batch efektif. Kebutuhan memori/waktu GPU aktual tetap bergantung runtime.

Saat evaluasi dan prediksi baru, model tetap membaca satu soal + referensi + jawaban dan menghasilkan satu nilai. Tidak membutuhkan nilai guru atau jawaban lain untuk diurutkan. Tidak ada cosine, S1/S2, loop multi-seed, kontrol ulang otomatis, atau worker tersembunyi. Seluruh Python ditulis per section di notebook.

**Dasar AES:** [Yang dkk. (2020)](https://aclanthology.org/2020.findings-emnlp.141/) menggabungkan MSE dan batch-wise ListNet dengan bobot dinamis. E44 mengadaptasinya ke fusion E43, memakai bobot tetap 0,1 dan kelompok dalam soal yang sama. Bobot 0,1 adalah pilihan eksperimen proyek, bukan optimum paper atau replikasi persis. Paper mengevaluasi QWK pada ASAP; RMSE E44 belum tentu membaik.

**Output:** lima Excel dalam satu ZIP. Excel 05 memisahkan MSE, ListNet, loss gabungan, dan cakupan ranking. Cross-entropy ListNet tidak harus mendekati nol meski prediksi tepat karena target berupa distribusi softmax. Excel 04 menambah **Prediksi Validasi OOF** untuk setiap jawaban development; Excel 05 tetap memiliki **MAE dan Toleransi CV**. Pasangan OOF terpisah dari 60 test.

Pilih **GPU T4 → Run all → unggah `fix_dataset_300.xlsx`**. Parameter loss ada di section 5, rumus dan pembentukan batch di section training. Folder/hash/checkpoint E44 terpisah. Belum ada hasil penelitian E44; acuan tetap E43 (mean CV 5,8876). Test historis yang telah dilihat berulang kali tetap bersifat eksploratif.
'''


def replace_once(source, old, new):
    assert source.count(old) == 1, old[:100]
    return source.replace(old, new, 1)


def build():
    raw = subprocess.check_output(['git','show',PIN+':'+BASE], cwd=ROOT)
    n = copy.deepcopy(json.loads(raw))
    old_hash = n['metadata']['aes_standalone']['code_sha256']
    for cell in n['cells']:
        role = cell['metadata'].pop('aes43_role', '')
        if role:
            cell['metadata']['aes44_role'] = role
        s = cell['source'] if isinstance(cell['source'], str) else ''.join(cell['source'])
        s = s.replace(old_hash, PLACEHOLDER).replace(Path(BASE).stem,'__E44_NAME__')
        s = s.replace('E43','E44').replace('__E44_NAME__',NAME)
        if role == 'imports':
            s = s.replace('from torch.utils.data import DataLoader',
                          'from torch.utils.data import DataLoader\nfrom torch.utils.checkpoint import checkpoint as activation_checkpoint')
        if role == 'hyperparameters':
            s += '\n# SATU PERLAKUAN E44: fungsi loss, bukan LR atau arsitektur baru.\nRANKING_WEIGHT = 0.1\nRANKING_RECOMPUTE = True  # Hitung ulang aktivasi saat backward agar microbatch tetap kecil.\n'
        if role == 'configuration_defaults':
            s = s.replace("    's3_fusion': 'concat',", "    'loss_recipe': 'mse_same_question_listnet_v1',\n    'ranking_weight': RANKING_WEIGHT, 'ranking_recompute': RANKING_RECOMPUTE,\n    's3_fusion': 'concat',")
            s = replace_once(s, '    return result\n', '''    if result['loss_recipe'] != 'mse_same_question_listnet_v1':
        raise ValueError('loss_recipe tidak cocok notebook E44.')
    if isinstance(result['ranking_weight'], bool) or not math.isfinite(result['ranking_weight']) or result['ranking_weight'] < 0:
        raise ValueError('ranking_weight harus finite dan tidak negatif.')
    if type(result['ranking_recompute']) is not bool:
        raise ValueError('ranking_recompute harus boolean.')
    return result
''')
        if role == 'configuration_apply':
            s = s.replace("    input_format='single_sequence', device=DEVICE,", "    loss_recipe='mse_same_question_listnet_v1', ranking_weight=RANKING_WEIGHT,\n    ranking_recompute=RANKING_RECOMPUTE,\n    input_format='single_sequence', device=DEVICE,")
            s = s.replace('e43_native_masked_mean_v1','e44_native_mse_listnet_v1')
            s += "\nprint('Loss: MSE +', cfg['ranking_weight'], '× ListNet | soal sama | recompute:', cfg['ranking_recompute'])\n"
        if role == 'definition:fit':
            begin = s.index("            for start in range(0, len(batch), cfg['micro_batch']):")
            end = s.index('            scaler.unscale_(optimizer)', begin)
            s = s[:begin] + NEW_BATCH + s[end:]
            s = replace_once(s, '        loss_total = 0.\n', '''        loss_total, ranking_total, combined_total = 0., 0., 0.
        ranking_groups, ranking_answers, ranking_batches = 0, 0, 0
''')
            s = replace_once(s, "        row = {'epoch': epoch, 'train_mse_scaled': loss_total/len(training),", '''        if cfg['ranking_weight'] > 0 and ranking_groups == 0:
            raise RuntimeError('Tidak ada kelompok soal dengan >=2 jawaban dalam batch epoch ini; ranking tidak aktif. Periksa kolom soal dan komposisi data.')
        row = {'epoch': epoch, 'train_mse_scaled': loss_total/len(training),
               'train_listnet_loss': ranking_total/len(training), 'train_combined_loss': combined_total/len(training),
               'ranking_groups': ranking_groups, 'ranking_answers': ranking_answers,
               'ranking_batches': ranking_batches, 'ranking_coverage_percent': 100*ranking_answers/len(training),''')
            s = s.replace('        print(f\'{scenario} · epoch {epoch}/{epochs} · loss latih (MSE, target 0–1): {row["train_mse_scaled"]:.6f}\'+message, flush=True)',
'''        print(f'{scenario} · epoch {epoch}/{epochs} | MSE: {row["train_mse_scaled"]:.6f}'
              f' | ListNet: {row["train_listnet_loss"]:.6f} | gabungan: {row["train_combined_loss"]:.6f}'
              f' | ranking: {ranking_answers}/{len(training)} jawaban, {ranking_groups} kelompok'+message, flush=True)''')
            s = s.replace("    info['scalar_mix_layers'] = cfg['scalar_mix_layers']", "    info['scalar_mix_layers'] = cfg['scalar_mix_layers']\n    info['loss_recipe'] = cfg['loss_recipe']\n    info['ranking_weight'] = cfg['ranking_weight']\n    info['ranking_recompute'] = cfg['ranking_recompute']")
        if role == 'definition:load_model':
            s = s.replace("for key in ('roberta',", "for key in ('loss_recipe', 'ranking_weight', 'ranking_recompute', 'roberta',")
        if role == 'definition:evaluation_report':
            s = s.replace('    return sheets\n', OOF_EXPORT+'    return sheets\n')
            s = s.replace("['Ringkasan Validasi','", "['Prediksi Validasi OOF','Setiap jawaban development diprediksi oleh model fold yang tidak melatih jawaban tersebut; bukan hasil test.'],\n        ['Ringkasan Validasi','")
        if role == 'definition:training_report':
            s = s.replace("row.get('RMSE'),row.get('Pearson'),row.get('QWK'),row.get('AC2')]", "row.get('RMSE'),row.get('Pearson'),row.get('QWK'),row.get('AC2'),\n                           row.get('train_listnet_loss'),row.get('train_combined_loss'),row.get('ranking_groups'),\n                           row.get('ranking_answers'),row.get('ranking_batches'),row.get('ranking_coverage_percent')]")
            s = s.replace("'Gwet AC2 Validasi'],history())", "'Gwet AC2 Validasi',\n            'Loss Ranking (ListNet)','Loss Gabungan','Kelompok Soal untuk Ranking',\n            'Jawaban untuk Ranking','Batch dengan Ranking','Cakupan Ranking (%)'],history())")
            s = s.replace("settings={'scalar_mix_gamma_init'", "settings={'loss_recipe':'Resep Loss','ranking_weight':'Bobot ListNet','ranking_recompute':'Hitung ulang aktivasi ranking','scalar_mix_gamma_init'")
            s = s.replace("['Epoch','Satu putaran", "['Loss Ranking dan Gabungan','ListNet cross-entropy kelompok soal sama; gabungan = MSE + bobot × ListNet. Dirata-ratakan per epoch dengan bobot ukuran batch aktual. ListNet tidak harus nol karena target berupa distribusi.'],\n        ['Cakupan Ranking','Jumlah jawaban yang berada dalam kelompok soal berukuran minimal dua dalam batch. Singleton hanya memakai MSE.'],\n        ['Epoch','Satu putaran")
        if cell['cell_type'] == 'markdown' and '## Rujukan dan batas kesimpulan' in s:
            s = s.replace('**Perlakuan E44:** Taghipour', '**Pooling acuan E43:** Taghipour')
            s += '\n**Perlakuan E44:** [Yang, Cao, Wen, Wu & He (2020), Enhancing Automated Essay Scoring Performance via Fine-tuning Pre-trained Language Models with Combination of Regression and Ranking](https://aclanthology.org/2020.findings-emnlp.141/). Paper menggabungkan MSE dan batch-wise ListNet berbobot dinamis pada AES/ASAP, dievaluasi QWK. E44 mengadaptasinya dengan bobot tetap 0,1, skala target 0–1, dan kelompok soal sama pada fusion E43; bukan replikasi persis atau jaminan RMSE lebih baik. [PyTorch activation checkpointing](https://docs.pytorch.org/docs/stable/checkpoint.html) dipakai untuk recomputation dengan RNG dropout dipertahankan; tidak menambah model prediksi.\n'
        cell['source'] = s
    # Rumus terlihat sebagai cell Python tersendiri tepat sebelum fit.
    index = next(i for i,c in enumerate(n['cells']) if c['metadata'].get('aes44_role')=='definition:fit')
    n['cells'][index:index] = [
        {'cell_type':'markdown','metadata':{},'source':'### Loss MSE + ListNet dan batch efektif\n\nFungsi di bawah membentuk distribusi nilai guru dan prediksi untuk setiap soal yang sama. Tidak ada pengurutan dataset atau nilai guru pada masukan encoder. Recompute hanya menghemat aktivasi; semua prediksi tetap terhubung ke graph gradien.\n'},
        {'cell_type':'code','metadata':{'aes44_role':'definition:ranking_losses'},'source':LOSSES,'outputs':[],'execution_count':None}]
    n['cells'][0]['source'] = INTRO
    fp = hashlib.sha256('\n'.join(c['source'] for c in n['cells'] if c['cell_type']=='code').encode()).hexdigest()
    for i,c in enumerate(n['cells']):
        c['source'] = c['source'].replace(PLACEHOLDER, fp)
        c['id'] = f'e44-{i:03d}-'+hashlib.sha256(c['source'].encode()).hexdigest()[:8]
        if c['cell_type']=='code':
            ast.parse(c['source']); c['outputs']=[]; c['execution_count']=None
    n['metadata']['colab']['name']=NAME+'.ipynb'
    n['metadata']['aes_standalone']['code_sha256']=fp
    n['metadata']['aes_experiment'].update(name='E44',s3='E43 unchanged model; MSE + same-question ListNet',
        format_commit=PIN,format_path=BASE,format_sha256=hashlib.sha256(raw).hexdigest(),
        loss_recipe='mse_same_question_listnet_v1',ranking_weight=0.1,ranking_recompute=True,
        ranking_target_scale='0–1, same as MSE; temperature1',ranking_grouping='exact cleaned question inside original effective batch',
        added_parameters_vs_e43=0,model_unchanged_vs_e43=True,
        treatment='MSE + fixed0.1 ListNet, mean over same-question groups of size>=2; original sampler preserved')
    output=json.dumps(n,ensure_ascii=False,indent=1)+'\n'
    for folder in ['notebooks/experiments','aes-learning/notebooks/experiments']:
        (ROOT/folder/(NAME+'.ipynb')).write_text(output)
    print(json.dumps({'name':NAME,'cells':len(n['cells']),'code_sha256':fp}))
    return n


if __name__ == '__main__':
    build()
