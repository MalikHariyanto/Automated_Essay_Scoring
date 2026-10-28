"""Dua ablation mandiri dari E23 yang dipin; tidak membangun ulang E21–E26."""
import ast
import copy
import hashlib
import json
import subprocess
from pathlib import Path
from build_context_pair_notebook import source, replace_once, definitions

ROOT = Path(__file__).resolve().parents[1]
PIN = 'f3d25c4'
BASE = 'notebooks/experiments/AES_300_E23_Tiga_Input_BiGRU_GPU.ipynb'
VARIANTS = {
    'E27': ('AES_300_E27_E23_BiGRU_CoAttention_GPU', 'bigru_coattention', 'E23 BiGRU + Co-Attention', 1e-4),
    'E28': ('AES_300_E28_E23_BiGRU_MLPLR5e6_GPU', 'bigru', 'E23 BiGRU; LR MLP S3 5e-6', 5e-6),
}


def add_reference_layout(s):
    s = replace_once(s, '    ids = tokenizer.encode(text, add_special_tokens=False)\n    answer_prefix',
        "    reference_positions = reference_context_positions(context, tokenizer, cfg, context_info)\n"
        "    context_layout = [-1]*len(context_ids)\n"
        "    for position in reference_positions:context_layout[position] = -4\n"
        '    ids = tokenizer.encode(text, add_special_tokens=False)\n    answer_prefix')
    # Hanya penanda posisi berubah; ID input encoder tetap identik dengan E23.
    s = s.replace('tokenizer.build_inputs_with_special_tokens([-1]*len(context_ids),',
                  'tokenizer.build_inputs_with_special_tokens(context_layout,')
    s = replace_once(s, '[-1]*len(context_ids)+[-3]*len(answer_prefix)+[-2]*len(answer))',
                     'context_layout+[-3]*len(answer_prefix)+[-2]*len(answer))')
    s = replace_once(s, 'chunks, answer_positions, segments = [], [], []',
                     'chunks, answer_positions, reference_window_positions, segments = [], [], [], []')
    s = replace_once(s, '        answer_positions.append(positions)',
                     '        answer_positions.append(positions)\n        reference_window_positions.append([i for i, value in enumerate(layout) if value == -4])')
    s = replace_once(s, "else 'Penanda jawaban' if v == -3 else 'Token khusus' for v in layout]",
                     "else 'Penanda jawaban' if v == -3 else 'Isi referensi' if v == -4 else 'Token khusus' for v in layout]")
    s = replace_once(s, "'answer_positions':answer_positions, 'segments':segments, 'answer_tokens_uncovered':0}",
                     "'answer_positions':answer_positions, 'reference_positions':reference_window_positions,\n"
                     "             'reference_content_kept':len(reference_positions), 'segments':segments, 'answer_tokens_uncovered':0}")
    return s.replace("('answer_positions','segments')", "('answer_positions','reference_positions','segments')")


def add_coattention_export(s):
    s = replace_once(s, '        pooling_rows = []', '        pooling_rows, reference_rows, alignment_rows = [], [], []')
    anchor = '                    for name, value in features.items():'
    s = replace_once(s, anchor, '''                    if scenario == 'S3':
                        windows, answer_mask, reference_mask = engine.coattention_batch
                        pool = model.sequence_pool
                        for window in range(len(answer_mask)):
                            answer_pos = answer_mask[window].nonzero(as_tuple=False).flatten().tolist()
                            ref_pos = reference_mask[window].nonzero(as_tuple=False).flatten().tolist()
                            ref_ids = [int(windows['input_ids'][window, p]) for p in ref_pos]
                            ref_tokens = engine.tokenizer.convert_ids_to_tokens(ref_ids)
                            for position, token_id, token in zip(ref_pos, ref_ids, ref_tokens):
                                reference_rows.append({'row_id':row['row_id'], 'window':window, 'position':position,
                                    'token_id':token_id, 'token':token,
                                    'attention_weight':float(pool.last_reference_weights[window, position])})
                            # Hanya kecocokan top-1 setiap token respons; bukan seluruh matriks atau skor kebenaran.
                            matrix = pool.last_cross_attention[window, :len(answer_pos), :len(ref_pos)].cpu()
                            for answer_index, position in enumerate(answer_pos):
                                best = int(matrix[answer_index].argmax())
                                token_id = int(windows['input_ids'][window, position])
                                alignment_rows.append({'row_id':row['row_id'], 'window':window,
                                    'answer_position':position, 'answer_token':engine.tokenizer.convert_ids_to_tokens([token_id])[0],
                                    'reference_position':ref_pos[best], 'reference_token':ref_tokens[best],
                                    'alignment_weight':float(matrix[answer_index, best])})
''' + anchor)
    s = replace_once(s, '        if window_values:', "        if reference_rows:\n            sheets['Bobot_token_referensi'] = pd.DataFrame(reference_rows)\n"
                     "        if alignment_rows:\n            sheets['Pencocokan_token_top1'] = pd.DataFrame(alignment_rows)\n        if window_values:")
    return s


def build(experiment):
    name, mode, label, mlp_lr = VARIANTS[experiment]
    raw = subprocess.check_output(['git', 'show', PIN + ':' + BASE], cwd=ROOT)
    n = copy.deepcopy(json.loads(raw))
    old = n['metadata']['aes_standalone']['code_sha256']
    helper = definitions((ROOT/'scripts/s3_coattention_helpers.py').read_text())
    for index, cell in enumerate(n['cells']):
        s = source(cell).replace(old, '__NOTEBOOK_CODE_SHA256__').replace(Path(BASE).stem, name)
        if cell['cell_type'] == 'code':
            if index == 10:
                s = s.replace("'s3_pooling': 'bigru'", "'s3_pooling': '"+mode+"'").replace("result['s3_pooling'] != 'bigru'", "result['s3_pooling'] != '"+mode+"'")
                s = replace_once(s, "    'vector_source': 'pretrained'", f"    's3_mlp_lr': {mlp_lr!r}, 's3_layers_lr': 1e-4,\n    'vector_source': 'pretrained'")
                s = replace_once(s, "'head_lr', 'gradient_clip'", "'head_lr', 's3_mlp_lr', 's3_layers_lr', 'gradient_clip'")
            if index == 12:
                s = replace_once(s, 'PROJECTION_LR=HEAD_LR=1e-4', f'''PROJECTION_LR=HEAD_LR=1e-4  # HEAD_LR tetap untuk kontrol S1/S2.
S3_MLP_LR={mlp_lr!r}  # HANYA model.head.* pada S3.
S3_LAYERS_LR=1e-4  # BiGRU, proyeksi ringkasan dan LayerNorm baru.''')
                s = replace_once(s, 'head_lr=HEAD_LR,', 'head_lr=HEAD_LR,\n                s3_mlp_lr=S3_MLP_LR,s3_layers_lr=S3_LAYERS_LR,')
                s = s.replace("s3_pooling='bigru'", "s3_pooling='"+mode+"'")
                s = s.replace("lesson_version='e23_s3_bigru_e21_gpu_v1'", "lesson_version='"+experiment.lower()+"_s3_gpu_v1'")
                s = replace_once(s, "'LR encoder':S3_ENCODER_LR,'LR head':HEAD_LR", "'LR encoder':S3_ENCODER_LR,'LR head':S3_MLP_LR")
                s = s.replace("'Tiga input seluruh skenario; LR encoder 5e-6; proyeksi/head 1e-4'", "'LR awal sebelum scheduler; LR S3 MLP terpisah dari pooling/norm'")
                s = '\n'.join(line for line in s.splitlines() if not line.startswith("print('E23 |"))+'\n'
                s += f"print('{experiment} | S3: {label} | LR MLP:',cfg['s3_mlp_lr'],'| LR pooling/norm:',cfg['s3_layers_lr'],flush=True)\n"
            if experiment == 'E27':
                if index == 11:
                    s += "\nLABELS.update({'reference_content_kept':'Token isi referensi dibaca','answer_position':'Posisi Token Jawaban','answer_token':'Token Jawaban','reference_position':'Posisi Token Referensi','reference_token':'Token Referensi','alignment_weight':'Bobot Kecocokan'})\n"
                    s = replace_once(s, "'token_index','position','window')", "'token_index','position','window','answer_position','reference_position')")
                if index == 20:
                    s = replace_once(s, 'def pair_content_windows(', helper['reference_context_positions']+'\n\ndef pair_content_windows(')
                    s = add_reference_layout(s)
                if index == 30:
                    s = replace_once(s, '# Satu head satu nilai;', helper['reference_token_mask']+'\n\n'+helper['AnswerReferenceCoAttention']+'\n\n# Satu head satu nilai;')
                    s = replace_once(s, 'self.sequence_pool = AnswerBiGRU(sequence_hidden)', 'self.sequence_pool = AnswerReferenceCoAttention(sequence_hidden)')
                    s = replace_once(s, 'return_features=False, answer_mask=None):', 'return_features=False, answer_mask=None, reference_mask=None):')
                    s = replace_once(s, 'pooled = self.sequence_pool(hidden, answer_mask)', 'pooled = self.sequence_pool(hidden, answer_mask, reference_mask)')
                    s = replace_once(s, "        if return_features and self.model.scenario == 'S3':", "        ref_mask = reference_token_mask(windows, mapping, details, self.tokenizer.padding_side) if self.model.scenario == 'S3' else None\n"
                        "        if self.model.scenario == 'S3':\n            self.model.sequence_pool.capture_alignment = return_features\n"
                        "        if return_features and self.model.scenario == 'S3':")
                    s = replace_once(s, '            self.pooling_batch = (windows, mask)', '            self.pooling_batch = (windows, mask)\n            self.coattention_batch = (windows, mask, ref_mask)')
                    s = replace_once(s, 'return_features=return_features, answer_mask=mask)', 'return_features=return_features, answer_mask=mask, reference_mask=ref_mask)')
                if index == 32:s = s.replace('BiGRU128 → proyeksi768', 'BiGRU128 → co-attention respons-referensi → ujung dua arah128 → proyeksi768')
                if index == 43:s = add_coattention_export(s)
            if index == 36:
                original = definitions(s)['optimizer_groups']
                s = replace_once(s, original, helper['optimizer_groups_isolated_mlp'].replace('def optimizer_groups_isolated_mlp(', 'def optimizer_groups('))
            if index == 45:
                s = replace_once(s, "settings={'s3_pooling'", "settings={'s3_mlp_lr':'LR MLP S3 saja','s3_layers_lr':'LR pooling dan norm S3','s3_pooling'")
                s = s.replace("'head_lr':'Learning rate head'", "'head_lr':'Learning rate head kontrol S1 S2'")
                if experiment == 'E27':
                    s = replace_once(s, "('14_S3_fitur_setelah_training.xlsx','bigru_hidden128','BiGRU S3 State Akhir')]",
                        "('14_S3_fitur_setelah_training.xlsx','bigru_hidden128','Ringkasan CoAttention S3'),\n"
                        "              ('14_S3_fitur_setelah_training.xlsx','Bobot_token_jawaban','Bobot CoAttention Respons S3'),\n"
                        "              ('14_S3_fitur_setelah_training.xlsx','Bobot_token_referensi','Bobot Referensi S3'),\n"
                        "              ('14_S3_fitur_setelah_training.xlsx','Pencocokan_token_top1','Pasangan Token Top1 S3')]")
                    s = s.replace('BiGRU satu layer, hidden64 per arah; state akhir128 → proyeksi768. Tidak memakai attention pooling.',
                        'BiGRU128 mencocokkan respons dengan isi referensi128; ujung dua arah setelah pencocokan diringkas128 lalu768, tanpa additive attention pooling E26.')
                    s = s.replace("' + BiGRU'", "' + BiGRU + Co-Attention'")
                    s = replace_once(s, "['Batas perbandingan','Jangan", "['Co-attention','Bobot Referensi adalah rata-rata respons ke referensi; Bobot CoAttention Respons adalah rata-rata referensi ke respons, masing-masing jumlah1 per window, bukan bobot pooling; Pasangan Token Top1 hanya satu kecocokan tertinggi per token respons, bukan seluruh matriks atau bukti kebenaran jawaban.'],\n        ['Batas perbandingan','Jangan")
                else:
                    s = s.replace("' + BiGRU'", "' + E23 BiGRU; LR MLP S3 5e-6'")
            cell['outputs'] = []
            cell['execution_count'] = None
        cell['source'] = s
    change = ('**S3 saja:** E23 + co-attention dua arah respons–referensi pada urutan output BiGRU. '
              'LR MLP/layer baru tetap1e-4.' if experiment == 'E27' else
              '**S3 saja:** LR seluruh `model.head.*` (dua Linear MLP) menjadi5e-6. '
              'Arsitektur E23 identik; BiGRU/proyeksi/norm tetap1e-4. S1/S2 tidak ikut diturunkan.')
    n['cells'][0]['source'] = f'''# {experiment} · {label}

{change}

Dasar: E23 sesuai pilihan pengguna untuk memprioritaskan jumlah prediksi mendekati guru, dipin ke `{PIN}`. Tiga input **soal, jawaban_referensi, response_text** pada semua skenario; label **rater 0–100**. S1 RoBERTa full fine-tuning + linear, S2 BGE frozen + MLP, S3 RoBERTa fine-tuned + BGE frozen + fusion MLP.

**Colab biasa → Runtime → Change runtime type → T4 GPU → Run all → unggah fix_dataset_300.xlsx.** Tidak perlu clone repo, ekstensi, file pendukung, atau Drive. GPU wajib; AMP otomatis. Seluruh S1/S2/S3 dilatih baru. Lima Excel dan ZIP diunduh pada cell terakhir.

Encoder RoBERTa S1/S3 **5e-6**; head kontrol S1/S2 **1e-4**; proyeksi BGE dan layer baru S3 **1e-4**; MLP S3 **{mlp_lr:g}**. Nilai ini LR awal/puncak, scheduler warmup+linear tetap E23. MSE target0–1; maksimal25epoch, patience5, lima fold, batch efektif8; seed42/split2026. Pemilihan epoch melalui CV, lalu refit development dan satu evaluasi test.

Folder lokal `{name}` terpisah dari E23 dan eksperimen satunya. Checkpoint dua slot per tahap; tanpa autosave Drive. Runtime dihapus berarti progres lokal hilang. Jangan memindahkan checkpoint E23/E27/E28 untuk resume silang. Hasil belum tersedia dan tidak dijamin membaik; test lama yang sudah dilihat bersifat eksploratif.
'''
    n['cells'][8]['source'] = f'## 3 Konfigurasi {experiment}: tiga input dan LR terpisah\n\nUbah parameter sebelum training; skor guru tidak masuk fitur.\n'
    n['cells'][29]['source'] = ('''## 10 E27: pencocokan respons–referensi setelah BiGRU

Hidden respons768 → BiGRU64/arah → urutan128. Hidden **isi referensi**768 → Linear128. Skor dot product/√128 menghasilkan attention dua arah. Fitur `[respons, referensi selaras, selisih, produk, respons hasil jalur balik]`640 → Linear128 + tanh → gabung64komponen ujung maju dan64komponen ujung mundur → ringkasan128 → Linear768 → LayerNorm → concat BGE256 → MLP.

Mask memakai posisi layout dan offset karakter tokenizer yang ID-nya diverifikasi sama; prefix, soal, special, dan padding tidak menjadi key referensi atau query respons. Referensi mengikuti truncation konteks E23; jika tidak ada token isi referensi tersisa, proses berhenti dengan penjelasan. Jawaban tetap seluruh window. RoBERTa tetap membaca tiga teks bersama; ini menambah pencocokan eksplisit.

Rujukan: [Zhang & Litman2018](https://aclanthology.org/W18-0549/), DOI10.18653/v1/W18-0549, dan [kode penulis](https://github.com/Rokeer/co-attention). Paper co-attention tingkat kalimat dengan CNN/LSTM dan teks sumber; implementasi token RoBERTa/BiGRU/BGE ini **adaptasi**, bukan reproduksi atau jaminan RMSE.
''' if experiment == 'E27' else '''## 10 E28: arsitektur E23 sama, LR MLP S3 berubah

RoBERTa token respons768 → BiGRU64/arah → state akhir maju+mundur128 → Linear768 → LayerNorm → concat BGE256 → MLP1024→256→1. Tidak ada co-attention atau attention pooling tambahan.

`optimizer_groups()` memisahkan `head.*` dari `sequence_pool.*` dan `roberta_norm.*`. Hanya **MLP S3** memakai5e-6; BiGRU, proyeksi ringkasan dan norm tetap1e-4. S1 linear dan S2 MLP tetap1e-4. Penurunan20× ini eksperimen parameter, bukan hasil yang dipastikan lebih baik.
''')
    if experiment == 'E27':
        n['cells'][1]['source'] += '\nRujukan tambahan E27: Zhang & Litman2018, DOI10.18653/v1/W18-0549. Co-attention token di notebook ini adalah adaptasi; hasil paper tidak dipindahkan menjadi hasil eksperimen ini.\n'
    codes = [source(c).removeprefix('%%aes\n') for c in n['cells'] if c['cell_type'] == 'code']
    for code in codes:ast.parse(code)
    fingerprint = hashlib.sha256('\n'.join(codes).encode()).hexdigest()
    for cell in n['cells']:cell['source'] = source(cell).replace('__NOTEBOOK_CODE_SHA256__', fingerprint)
    n['metadata']['colab']['name'] = name+'.ipynb'
    n['metadata']['aes_standalone']['code_sha256'] = fingerprint
    n['metadata']['aes_experiment'].update(name=experiment, base_commit=PIN, base_path=BASE,
        base_sha256=hashlib.sha256(raw).hexdigest(), s3=label, s3_pooling=mode,
        s3_mlp_lr=mlp_lr, s3_layers_lr=1e-4, mlp_lr_scope='S3 model.head.* only',
        head_lr=1e-4, new_layer_lr=1e-4, coattention=experiment=='E27')
    path = ROOT/'notebooks/experiments'/(name+'.ipynb')
    path.write_text(json.dumps(n, ensure_ascii=False, indent=1)+'\n')
    (ROOT/'aes-learning/notebooks/experiments'/path.name).write_bytes(path.read_bytes())
    print(experiment, path.relative_to(ROOT), fingerprint)
    return path


if __name__ == '__main__':
    for experiment in VARIANTS:build(experiment)
