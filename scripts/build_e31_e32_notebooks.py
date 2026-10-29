"""Dua ablasi E25; notebook GPU mandiri, tanpa memodifikasi eksperimen lama."""
import ast,copy,hashlib,json,subprocess
from pathlib import Path
from build_context_pair_notebook import source,replace_once,definitions
from build_e27_e28_notebooks import add_reference_layout
from build_e29_e30_notebooks import export_both_attention
ROOT=Path(__file__).resolve().parents[1]
PIN='bc4713872c7a7ce23f7dad8ae3ea73a2b75d9b86'
BASE='notebooks/experiments/AES_300_E25_Tiga_Input_Attention_GatedFusion_GPU.ipynb'
NAMES={'E31':'AES_300_E31_E25_GatedFusion_MLPLR5e6_GPU','E32':'AES_300_E32_E25_CoAttention_GatedFusion_GPU'}


def build(experiment):
    name=NAMES[experiment];lr=5e-6 if experiment=='E31' else 1e-4;mode='attention' if experiment=='E31' else 'coattention_pooling'
    raw=subprocess.check_output(['git','show',PIN+':'+BASE],cwd=ROOT);n=copy.deepcopy(json.loads(raw));old=n['metadata']['aes_standalone']['code_sha256']
    shared=definitions((ROOT/'scripts/s3_coattention_helpers.py').read_text());extra=definitions((ROOT/'scripts/s3_e25_coattention_helpers.py').read_text())
    optimizer=shared['optimizer_groups_isolated_mlp'].replace('def optimizer_groups_isolated_mlp(','def optimizer_groups(')
    optimizer=replace_once(optimizer,"'head': cfg['s3_mlp_lr'] if model.scenario == 'S3' else cfg['head_lr'],","'head': cfg['head_lr'] if model.scenario == 'S1' else cfg['s2_mlp_lr'] if model.scenario == 'S2' else cfg['s3_mlp_lr'],")
    for index,cell in enumerate(n['cells']):
        s=source(cell).replace(old,'__NOTEBOOK_CODE_SHA256__').replace(Path(BASE).stem,name)
        if cell['cell_type']=='code':
            if index==10:
                s=s.replace("'s3_pooling': 'attention'","'s3_pooling': '"+mode+"'").replace("result['s3_pooling'] != 'attention'","result['s3_pooling'] != '"+mode+"'")
                s=replace_once(s,"'s3_fusion': 'gated_concat',",f"'s2_mlp_lr': {lr!r}, 's3_mlp_lr': {lr!r}, 's3_layers_lr': 1e-4,\n    's3_fusion': 'gated_concat',")
                s=replace_once(s,"'head_lr', 'gradient_clip'","'head_lr', 's2_mlp_lr', 's3_mlp_lr', 's3_layers_lr', 'gradient_clip'")
            if index==12:
                s=replace_once(s,'PROJECTION_LR=HEAD_LR=1e-4',f'PROJECTION_LR=HEAD_LR=1e-4  # Linear S1; bukan LR MLP.\nS2_MLP_LR=S3_MLP_LR={lr!r}\nS3_LAYERS_LR=1e-4  # Pooling, gate, norm dan proyeksi tetap.')
                s=replace_once(s,'head_lr=HEAD_LR,','head_lr=HEAD_LR,s2_mlp_lr=S2_MLP_LR,s3_mlp_lr=S3_MLP_LR,s3_layers_lr=S3_LAYERS_LR,')
                s=s.replace("s3_pooling='attention'","s3_pooling='"+mode+"'").replace("lesson_version='e25_s3_attention_gpu_v1'","lesson_version='"+experiment.lower()+"_e25_gpu_v1'")
                s=replace_once(s,"'LR encoder':None,'LR head':HEAD_LR","'LR encoder':None,'LR head':S2_MLP_LR")
                s=replace_once(s,"'LR encoder':S3_ENCODER_LR,'LR head':HEAD_LR","'LR encoder':S3_ENCODER_LR,'LR head':S3_MLP_LR")
                s='\n'.join(line for line in s.splitlines() if not line.startswith("print('E25 |"))+'\n'
                s+=f"print('{experiment}: E25; MLP S2/S3 {lr:g}; gate/pooling/proyeksi/norm 1e-4; tanpa BiGRU',flush=True)\n"
            if index==36:s=replace_once(s,definitions(s)['optimizer_groups'],optimizer)
            if index==45:
                s=replace_once(s,"settings={'s3_fusion'","settings={'s2_mlp_lr':'LR MLP S2','s3_mlp_lr':'LR MLP S3','s3_layers_lr':'LR pooling gate norm S3','s3_fusion'")
                s=s.replace("'head_lr':'Learning rate head'","'head_lr':'LR head linear S1'")
                s=s.replace("' + Attention Pooling + Gated Fusion'",f"' + Attention Pooling + Gated Fusion; {experiment}'")
            if experiment=='E32':
                if index==11:
                    s+="\nLABELS.update({'reference_content_kept':'Token isi referensi dibaca','answer_position':'Posisi Token Jawaban','answer_token':'Token Jawaban','reference_position':'Posisi Token Referensi','reference_token':'Token Referensi','alignment_weight':'Bobot Kecocokan'})\n"
                    s=replace_once(s,"'token_index','position','window')","'token_index','position','window','answer_position','reference_position')")
                if index==20:
                    s=replace_once(s,'def pair_content_windows(',shared['reference_context_positions']+'\n\ndef pair_content_windows(');s=add_reference_layout(s)
                if index==30:
                    s=replace_once(s,'# Satu head satu nilai;',shared['reference_token_mask']+'\n\n'+extra['AnswerCoAttentionPool']+'\n\n# Satu head satu nilai;')
                    s=replace_once(s,'self.sequence_pool = AnswerAttentionPool(sequence_hidden)','self.sequence_pool = AnswerCoAttentionPool(sequence_hidden)')
                    s=replace_once(s,'return_features=False, answer_mask=None):','return_features=False, answer_mask=None, reference_mask=None):')
                    s=replace_once(s,'pooled = self.sequence_pool(hidden, answer_mask)','pooled = self.sequence_pool(hidden, answer_mask, reference_mask)')
                    s=replace_once(s,"        if return_features and self.model.scenario == 'S3':","        ref_mask = reference_token_mask(windows, mapping, details, self.tokenizer.padding_side) if self.model.scenario == 'S3' else None\n        if self.model.scenario == 'S3':\n            self.model.sequence_pool.capture_alignment = return_features\n        if return_features and self.model.scenario == 'S3':")
                    s=replace_once(s,'            self.pooling_batch = (windows, mask)','            self.pooling_batch = (windows, mask)\n            self.coattention_batch = (windows, mask, ref_mask)')
                    s=replace_once(s,'return_features=return_features, answer_mask=mask)','return_features=return_features, answer_mask=mask, reference_mask=ref_mask)')
                if index==43:s=export_both_attention(s)
                if index==45:
                    s=replace_once(s,"('14_S3_fitur_setelah_training.xlsx','bge_multiplier','Pengali BGE S3')]","('14_S3_fitur_setelah_training.xlsx','bge_multiplier','Pengali BGE S3'),\n              ('14_S3_fitur_setelah_training.xlsx','Bobot_coattention_respons','Bobot CoAttention Respons S3'),\n              ('14_S3_fitur_setelah_training.xlsx','Bobot_token_referensi','Bobot Referensi S3'),\n              ('14_S3_fitur_setelah_training.xlsx','Pencocokan_token_top1','Pasangan Token Top1 S3')]")
            cell['outputs']=[];cell['execution_count']=None
        cell['source']=s
    description=('Arsitektur E25 identik; hanya seluruh head.* MLP S2/S3 menjadi5e-6. Gate/pooling/norm/proyeksi tetap1e-4.' if experiment=='E31' else 'E25 + co-attention sebelum attention pooling; MLP tetap1e-4. Respons/referensi768→128; matching640→768+tanh; pooling E25 tetap768→64→1; gate E25 tetap. Tidak ada BiGRU.')
    n['cells'][0]['source']=f'''# {experiment} · E25 sebagai acuan

{description}

Tiga input semua skenario: soal, jawaban_referensi, response_text. Label tunggal rater0–100. S1 RoBERTa full fine-tuning+linear; S2 BGE frozen+MLP; S3 RoBERTa+attention+gated fusion BGE+MLP. Encoder S1/S3 5e-6; linear S1 1e-4; MLP S2/S3 {lr:g}; seluruh layer lain1e-4.

Web Colab → Runtime → Change runtime type → T4 GPU → Run all → unggah fix_dataset_300.xlsx. Notebook mandiri, GPU wajib, AMP otomatis; tanpa Drive, clone atau file Python pendamping. Semua S1/S2/S3 dilatih baru, tanpa reuse run lama. Penyimpanan/checkpoint hanya runtime lokal; runtime dihapus berarti progres hilang. ZIP lima Excel tidak termasuk bobot.

Source E25 dipin ke `{PIN}`. Maksimal25epoch,patience5,5fold pada240development/60test,microbatch1/efektif8,MSE target0–1,seed42/split2026. Config baru dan folder `{name}` terpisah. Jangan menyalin checkpoint eksperimen lama. LR adalah awal/puncak sebelum warmup+linear scheduler.

Belum ada hasil penelitian {experiment}. Pilih lewat development/CV; test lama sudah sering dilihat, sehingga perbandingan lanjutan eksploratif. E25 unggul pembulatan/toleransi1poin, bukan semua ukuran; E29 S2 memburuk pada LR kecil. Resep ini tidak menjamin peningkatan.
'''
    n['cells'][29]['source']=f'## 10 · {experiment}: arsitektur dan kelompok LR\n\n{description}\n'
    if experiment=='E32':n['cells'][29]['source']+='\nDua tahap tambahan: co-attention mencocokkan respons–referensi, pooling merangkum hasil matching. Self-attention RoBERTa tetap ada. Mask posisi/offset memisahkan isi respons dan referensi; soal/prefix/special/padding bukan query/key langsung. Pooling dan gate berbeda: gate menimbang dua cabang setelah pooling/LayerNorm dan proyeksi BGE. Bobot inspeksi bukan skor guru atau bukti kausal.\n\nRujukan adaptasi: Zhang & Litman2018, https://aclanthology.org/W18-0549/ (tingkat kalimat/teks sumber, bukan reproduksi arsitektur ini). Gate mengikuti E25, inspirasi GMU https://openreview.net/forum?id=Hy-2G6ile .\n'
    codes=[source(c).removeprefix('%%aes\n') for c in n['cells'] if c['cell_type']=='code']
    for code in codes:ast.parse(code)
    fingerprint=hashlib.sha256('\n'.join(codes).encode()).hexdigest()
    for cell in n['cells']:cell['source']=source(cell).replace('__NOTEBOOK_CODE_SHA256__',fingerprint)
    n['metadata']['colab']['name']=name+'.ipynb';n['metadata']['aes_standalone']['code_sha256']=fingerprint
    n['metadata']['aes_experiment'].update(name=experiment,base_commit=PIN,base_path=BASE,base_sha256=hashlib.sha256(raw).hexdigest(),s3=description,s3_pooling=mode,s2_mlp_lr=lr,s3_mlp_lr=lr,s3_layers_lr=1e-4,head_lr=1e-4,coattention=experiment=='E32',reused_controls=[])
    path=ROOT/'notebooks/experiments'/(name+'.ipynb');path.write_text(json.dumps(n,ensure_ascii=False,indent=1)+'\n');(ROOT/'aes-learning/notebooks/experiments'/path.name).write_bytes(path.read_bytes());print(path.relative_to(ROOT),fingerprint);return path

if __name__=='__main__':
    for e in NAMES:build(e)
