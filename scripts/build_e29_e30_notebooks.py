"""E29 LR kedua MLP dan E30 co-attention+pooling; dua turunan E26 terpisah."""
import ast
import copy
import hashlib
import json
import subprocess
from pathlib import Path
from build_context_pair_notebook import source, replace_once, definitions
from build_e27_e28_notebooks import add_reference_layout, add_coattention_export

ROOT=Path(__file__).resolve().parents[1]
PIN='bc4713872c7a7ce23f7dad8ae3ea73a2b75d9b86'
BASE='notebooks/experiments/AES_300_E26_Tiga_Input_BiGRU_Attention_GPU.ipynb'
VARIANTS={
    'E29':('AES_300_E29_E26_BiGRU_Attention_MLPLR5e6_GPU','bigru_attention','E26; LR MLP S2 dan S3 5e-6',5e-6),
    'E30':('AES_300_E30_E26_BiGRU_CoAttention_Pooling_GPU','bigru_coattention_pooling','E26 + Co-Attention + Attention Pooling',1e-4),
}


def export_both_attention(s):
    s=add_coattention_export(s)
    s=replace_once(s,'pooling_rows, reference_rows, alignment_rows = [], [], []',
                   'pooling_rows, reference_rows, alignment_rows, coresponse_rows = [], [], [], []')
    s=replace_once(s,'                            ref_ids =', '''                            for position in answer_pos:
                                token_id = int(windows['input_ids'][window, position])
                                coresponse_rows.append({'row_id':row['row_id'], 'window':window, 'position':position,
                                    'token_id':token_id, 'token':engine.tokenizer.convert_ids_to_tokens([token_id])[0],
                                    'attention_weight':float(pool.last_response_coattention_weights[window, position])})
                            ref_ids =''')
    s=replace_once(s,'        if reference_rows:', "        if coresponse_rows:\n            sheets['Bobot_coattention_respons'] = pd.DataFrame(coresponse_rows)\n        if reference_rows:")
    return s


def build(experiment):
    name,mode,label,mlp_lr=VARIANTS[experiment]
    raw=subprocess.check_output(['git','show',PIN+':'+BASE],cwd=ROOT)
    n=copy.deepcopy(json.loads(raw));old=n['metadata']['aes_standalone']['code_sha256']
    shared=definitions((ROOT/'scripts/s3_coattention_helpers.py').read_text())
    extra=definitions((ROOT/'scripts/s3_coattention_pooling_helpers.py').read_text())
    optimizer=shared['optimizer_groups_isolated_mlp'].replace('def optimizer_groups_isolated_mlp(', 'def optimizer_groups(')
    optimizer=optimizer.replace('MLP S3 saja dapat diberi LR berbeda;', 'MLP S2/S3 mempunyai LR terpisah dari linear S1 dan pooling;')
    optimizer=replace_once(optimizer,"'head': cfg['s3_mlp_lr'] if model.scenario == 'S3' else cfg['head_lr'],",
        "'head': cfg['head_lr'] if model.scenario == 'S1' else cfg['s2_mlp_lr'] if model.scenario == 'S2' else cfg['s3_mlp_lr'],")
    for index,cell in enumerate(n['cells']):
        s=source(cell).replace(old,'__NOTEBOOK_CODE_SHA256__').replace(Path(BASE).stem,name)
        if cell['cell_type']=='code':
            if index==10:
                s=s.replace("'s3_pooling': 'bigru_attention'", "'s3_pooling': '"+mode+"'").replace("result['s3_pooling'] != 'bigru_attention'", "result['s3_pooling'] != '"+mode+"'")
                s=replace_once(s,"    's3_fusion': 'concat',",f"    's2_mlp_lr': {mlp_lr!r}, 's3_mlp_lr': {mlp_lr!r}, 's3_layers_lr': 1e-4,\n    's3_fusion': 'concat',")
                s=replace_once(s,"'head_lr', 'gradient_clip'", "'head_lr', 's2_mlp_lr', 's3_mlp_lr', 's3_layers_lr', 'gradient_clip'")
            if index==12:
                s=replace_once(s,'PROJECTION_LR=HEAD_LR=1e-4',f'''PROJECTION_LR=HEAD_LR=1e-4  # HEAD_LR: hanya linear S1.
S2_MLP_LR=S3_MLP_LR={mlp_lr!r}  # Seluruh parameter head MLP S2/S3.
S3_LAYERS_LR=1e-4  # BiGRU, attention pooling, co-attention jika ada, norm dan proyeksi ringkasan.''')
                s=replace_once(s,'head_lr=HEAD_LR,','head_lr=HEAD_LR,\n                s2_mlp_lr=S2_MLP_LR,s3_mlp_lr=S3_MLP_LR,s3_layers_lr=S3_LAYERS_LR,')
                s=s.replace("s3_pooling='bigru_attention'", "s3_pooling='"+mode+"'")
                s=s.replace("lesson_version='e26_s3_bigru_attention_gpu_v1'", "lesson_version='"+experiment.lower()+"_e26_gpu_v1'")
                s=replace_once(s,"'LR encoder':None,'LR head':HEAD_LR", "'LR encoder':None,'LR head':S2_MLP_LR")
                s=replace_once(s,"'LR encoder':S3_ENCODER_LR,'LR head':HEAD_LR", "'LR encoder':S3_ENCODER_LR,'LR head':S3_MLP_LR")
                s=s.replace("'Tiga input seluruh skenario; LR encoder 5e-6; proyeksi/head 1e-4'", "'LR awal sebelum scheduler: linear S1 dan MLP S2/S3 dipisahkan dari pooling/proyeksi'")
                s='\n'.join(line for line in s.splitlines() if not line.startswith("print('E26 |"))+'\n'
                s+=f"print('{experiment} | {label} | LR MLP S2/S3:',S2_MLP_LR,S3_MLP_LR,'| LR layer S3:',S3_LAYERS_LR,flush=True)\n"
            if experiment=='E30':
                if index==11:
                    s+="\nLABELS.update({'reference_content_kept':'Token isi referensi dibaca','answer_position':'Posisi Token Jawaban','answer_token':'Token Jawaban','reference_position':'Posisi Token Referensi','reference_token':'Token Referensi','alignment_weight':'Bobot Kecocokan'})\n"
                    s=replace_once(s,"'token_index','position','window')", "'token_index','position','window','answer_position','reference_position')")
                if index==20:
                    s=replace_once(s,'def pair_content_windows(',shared['reference_context_positions']+'\n\ndef pair_content_windows(')
                    s=add_reference_layout(s)
                if index==30:
                    s=replace_once(s,'# Satu head satu nilai;',shared['reference_token_mask']+'\n\n'+extra['AnswerBiGRUCoAttentionPool']+'\n\n# Satu head satu nilai;')
                    s=replace_once(s,'self.sequence_pool = AnswerBiGRUAttention(sequence_hidden)','self.sequence_pool = AnswerBiGRUCoAttentionPool(sequence_hidden)')
                    s=replace_once(s,'return_features=False, answer_mask=None):','return_features=False, answer_mask=None, reference_mask=None):')
                    s=replace_once(s,'pooled = self.sequence_pool(hidden, answer_mask)','pooled = self.sequence_pool(hidden, answer_mask, reference_mask)')
                    s=replace_once(s,"        if return_features and self.model.scenario == 'S3':", "        ref_mask = reference_token_mask(windows, mapping, details, self.tokenizer.padding_side) if self.model.scenario == 'S3' else None\n"
                        "        if self.model.scenario == 'S3':\n            self.model.sequence_pool.capture_alignment = return_features\n"
                        "        if return_features and self.model.scenario == 'S3':")
                    s=replace_once(s,'            self.pooling_batch = (windows, mask)','            self.pooling_batch = (windows, mask)\n            self.coattention_batch = (windows, mask, ref_mask)')
                    s=replace_once(s,'return_features=return_features, answer_mask=mask)','return_features=return_features, answer_mask=mask, reference_mask=ref_mask)')
                if index==32:s=s.replace('BiGRU + Attention Pooling','BiGRU + Co-Attention respons-referensi + Attention Pooling')
                if index==43:s=export_both_attention(s)
            if index==36:s=replace_once(s,definitions(s)['optimizer_groups'],optimizer)
            if index==45:
                s=replace_once(s,"settings={'s3_fusion'", "settings={'s2_mlp_lr':'LR MLP S2','s3_mlp_lr':'LR MLP S3','s3_layers_lr':'LR pooling dan norm S3','s3_fusion'")
                s=s.replace("'head_lr':'Learning rate head'", "'head_lr':'LR head linear S1'")
                if experiment=='E30':
                    s=replace_once(s,"('14_S3_fitur_setelah_training.xlsx','bigru_hidden128','Ringkasan BiGRU Attention S3')]",
                        "('14_S3_fitur_setelah_training.xlsx','bigru_hidden128','Ringkasan CoAttn Pooling S3'),\n"
                        "              ('14_S3_fitur_setelah_training.xlsx','Bobot_coattention_respons','Bobot CoAttention Respons S3'),\n"
                        "              ('14_S3_fitur_setelah_training.xlsx','Bobot_token_referensi','Bobot Referensi S3'),\n"
                        "              ('14_S3_fitur_setelah_training.xlsx','Pencocokan_token_top1','Pasangan Token Top1 S3')]")
                    s=s.replace('Output token BiGRU dua arah128 diberi attention, diringkas128 lalu diproyeksikan768. Attention bekerja atas urutan token, bukan satu state akhir.',
                        'Output token BiGRU128 dicocokkan dengan isi referensi128 melalui co-attention; additive attention pooling E26 merangkum hasil matching128 lalu Linear768. Dua tahap berbeda, tanpa ringkasan ujung E27.')
                    s=s.replace("' + BiGRU + Attention Pooling'", "' + BiGRU + Co-Attention + Attention Pooling'")
                    s=replace_once(s,"['Batas perbandingan','Jangan", "['Dua tahap attention','Bobot Attention S3 adalah pooling respons SETELAH matching. Bobot CoAttention Respons S3 adalah rata-rata referensi→respons; Bobot Referensi S3 adalah rata-rata respons→referensi. Masing-masing jumlah1 per window, bukan skor guru/penjelasan kausal. Top1 hanya ringkasan kecocokan, bukan matriks lengkap.'],\n        ['Batas perbandingan','Jangan")
                else:s=s.replace("' + BiGRU + Attention Pooling'", "' + BiGRU + Attention Pooling; LR MLP S2/S3 5e-6'")
            cell['outputs']=[];cell['execution_count']=None
        cell['source']=s
    change=('**E29:** arsitektur E26 identik; hanya LR seluruh `head.*` MLP S2 dan S3 turun menjadi5e-6. Linear S1 tetap1e-4. '
            'BiGRU, attention pooling, LayerNorm dan proyeksi tetap1e-4.' if experiment=='E29' else
            '**E30:** E26 + co-attention respons–referensi, kemudian attention pooling E26. Ada dua tahap attention; '
            'LR MLP S2/S3 dan semua layer baru tetap1e-4. Tidak menggabungkan penurunan LR E29.')
    n['cells'][0]['source']=f'''# {experiment} · {label}

{change}

Dasar source **E26**, commit `{PIN}`. Tiga input semua skenario: **soal, jawaban_referensi, response_text**. Label: **rater**, satu skor guru0–100; tidak masuk encoder. S1 RoBERTa full fine-tuning + linear; S2 BGE frozen + MLP; S3 RoBERTa full fine-tuning + BGE frozen + BiGRU/fusion MLP.

**Web Colab biasa → Runtime → Change runtime type → T4 GPU → Run all → unggah fix_dataset_300.xlsx.** GPU wajib; AMP bf16/fp16 otomatis. Tidak perlu clone, ekstensi, file Python proyek atau mount Drive. Semua S1/S2/S3 dilatih baru.

LR encoder RoBERTa S1/S3 **5e-6**; linear S1 **1e-4**; MLP S2/S3 **{mlp_lr:g}**; BiGRU, attention pooling, norm, proyeksi BGE dan layer co-attention jika ada **1e-4**. Nilai tersebut LR awal/puncak sebelum warmup+linear scheduler. MSE target0–1; maksimal25epoch, patience5, lima fold pada240development,60test; microbatch1/efektif8; seed42/split2026.

Folder lokal `{name}` terpisah. Checkpoint dua slot per tahap, tanpa autosave Drive. Resume hanya resep yang sama dan folder lokal yang masih ada; runtime dihapus berarti progres hilang. Lima Excel dan ZIP ringkas diunduh pada cell terakhir, tidak termasuk bobot model. Jangan menyalin checkpoint E26/E27/E28 ke resep baru.

Eksperimen ini belum memiliki hasil penelitian. Pilih konfigurasi melalui development/CV, bukan test. Test lama yang telah berulang kali dilihat bersifat eksploratif; perubahan LR/layer tidak dijamin meningkatkan hasil.
'''
    n['cells'][8]['source']=f'## 3 Konfigurasi {experiment} · tiga input dan kelompok LR\n\nUbah sebelum training. MLP S2/S3 dipisah dari linear S1 dan layer pooling/norm.\n'
    n['cells'][29]['source']=('''## 10 E30 · co-attention lalu attention pooling

Token respons RoBERTa768 → BiGRU64/arah → urutan128. Token isi referensi RoBERTa768 → Linear128. Skor dot product/√128 membentuk dua softmax respons↔referensi. Fitur `[R, AK, R−AK, R⊙AK, A(BR)]`640 → Linear128+tanh. **Additive attention128→64→1 E26** memberi bobot pada token hasil matching, weighted sum128 → Linear768 → LayerNorm → concat BGE256 → MLP1024→256→1.

Co-attention mencocokkan dua urutan; attention pooling merangkum satu urutan hasil pencocokan. Ini bukan E27 yang memakai ujung dua arah E23. Mask memakai posisi layout dan offset dengan ID tokenizer diverifikasi identik; soal/prefix/special/padding bukan query/key/pooling langsung, tetapi ketiga teks tetap dibaca self-attention RoBERTa. Referensi mengikuti anggaran E26; jika seluruh isi terpotong, notebook berhenti dengan penjelasan.

Rujukan: [Zhang & Litman2018](https://aclanthology.org/W18-0549/), DOI10.18653/v1/W18-0549, dan [kode penulis](https://github.com/Rokeer/co-attention). Paper tingkat kalimat/CNN-LSTM/teks sumber; kombinasi token RoBERTa-BiGRU-BGE ini adaptasi, bukan reproduksi atau jaminan RMSE.
''' if experiment=='E30' else '''## 10 E29 · E26 sama, LR kedua MLP berubah

Arsitektur S3 E26: token respons768 → BiGRU64/arah → urutan128 → attention128→64→1 → weighted sum128 → Linear768 → LayerNorm → concat BGE256 → MLP1024→256→1. S2: BGE1024 → MLP1024→256→1. Tidak ada co-attention tambahan.

Hanya parameter `head.*` pada **S2 dan S3** memakai5e-6. Linear S1 tetap1e-4; encoder RoBERTa S1/S3 tetap5e-6; BiGRU/attention/norm/proyeksi tetap1e-4. `optimizer_groups()` memastikan seluruh parameter trainable tercakup tepat sekali. S2 kini juga perlakuan eksperimen; S1 tetap kontrol. Penurunan20× dapat memperlambat pembelajaran head baru; hasil belum diketahui.
''')
    if experiment=='E30':n['cells'][1]['source']+='\nE30 menambah co-attention dari Zhang & Litman2018 (W18-0549), dengan attention pooling E26 tetap. Ini adaptasi token-level, bukan reproduksi paper.\n'
    codes=[source(c).removeprefix('%%aes\n') for c in n['cells'] if c['cell_type']=='code']
    for code in codes:ast.parse(code)
    fingerprint=hashlib.sha256('\n'.join(codes).encode()).hexdigest()
    for cell in n['cells']:cell['source']=source(cell).replace('__NOTEBOOK_CODE_SHA256__',fingerprint)
    n['metadata']['colab']['name']=name+'.ipynb';n['metadata']['aes_standalone']['code_sha256']=fingerprint
    n['metadata']['aes_experiment'].update(name=experiment,base_commit=PIN,base_path=BASE,base_sha256=hashlib.sha256(raw).hexdigest(),
        s3=label,s3_pooling=mode,s2_mlp_lr=mlp_lr,s3_mlp_lr=mlp_lr,s3_layers_lr=1e-4,head_lr=1e-4,
        coattention=experiment=='E30',attention_pooling=True,mlp_lr_scope='S2 and S3 model.head.* only',reused_controls=[])
    path=ROOT/'notebooks/experiments'/(name+'.ipynb');path.write_text(json.dumps(n,ensure_ascii=False,indent=1)+'\n')
    (ROOT/'aes-learning/notebooks/experiments'/path.name).write_bytes(path.read_bytes());print(experiment,path.relative_to(ROOT),fingerprint)
    return path


if __name__=='__main__':
    for experiment in VARIANTS:build(experiment)
