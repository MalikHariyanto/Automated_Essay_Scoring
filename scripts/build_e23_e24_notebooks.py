"""Dua ablation S3 dari E21: BiGRU atau attention pooling token respons."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import subprocess
from build_context_pair_notebook import source, replace_once, definitions

ROOT = Path(__file__).resolve().parents[1]
PIN = 'ef712b4c035d96178d9baa67caa7453ac9fd8738'
BASE = 'notebooks/experiments/AES_300_E21_Tiga_Input_GPU.ipynb'
VARIANTS = {
    'E23': ('AES_300_E23_Tiga_Input_BiGRU_GPU', 'bigru', 'AnswerBiGRU'),
    'E24': ('AES_300_E24_Tiga_Input_AttentionPooling_GPU', 'attention', 'AnswerAttentionPool'),
}


def build(experiment):
    name, mode, pool_class = VARIANTS[experiment]
    raw = subprocess.check_output(['git', 'show', PIN + ':' + BASE], cwd=ROOT)
    n = copy.deepcopy(json.loads(raw))
    previous = n['metadata']['aes_standalone']['code_sha256']
    helper = definitions((ROOT / 'scripts/s3_sequence_pooling_helpers.py').read_text())
    pool_source = '\n\n'.join(helper[k] for k in ['answer_token_mask', 'check_answer_pool_inputs', pool_class])
    description = ('BiGRU satu layer, hidden64 per arah; state akhir128 → proyeksi768. Tidak memakai attention pooling.'
                   if mode == 'bigru' else
                   'Additive attention768 →64 →1 pada posisi token jawaban; jumlah berbobot tetap768. Tidak memakai BiGRU.')
    for index, cell in enumerate(n['cells']):
        s = source(cell).replace(previous, '__NOTEBOOK_CODE_SHA256__')
        s = s.replace('AES_300_E21_Tiga_Input_GPU', name)
        if cell['cell_type'] == 'code':
            if index == 10:
                s = replace_once(s, "'vector_source': 'pretrained'", "'s3_pooling': '" + mode + "', 's3_sequence_hidden': 64,\n    'vector_source': 'pretrained'")
                s = replace_once(s, '    return result', "    if result['s3_pooling'] != '" + mode + "' or type(result['s3_sequence_hidden']) is not int or result['s3_sequence_hidden'] < 1:\n        raise ValueError('Jenis/hidden pooling S3 tidak cocok notebook eksperimen ini.')\n    return result")
            if index == 12:
                s = replace_once(s, "INPUT_FORMAT='single_sequence'", "INPUT_FORMAT='single_sequence'\nS3_SEQUENCE_HIDDEN=64  # Ukuran layer baru; jangan memilihnya dari hasil test.")
                s = replace_once(s, "input_format=INPUT_FORMAT,device=", "s3_pooling='" + mode + "',s3_sequence_hidden=S3_SEQUENCE_HIDDEN,input_format=INPUT_FORMAT,device=")
                s = s.replace("lesson_version='three_inputs_format_ablation_local_gpu_v1'", "lesson_version='" + experiment.lower() + "_s3_" + mode + "_e21_gpu_v1'")
                s += "\nprint('" + experiment + " | S3: " + description + " | LR layer baru:',cfg['head_lr'],flush=True)\n"
            if index == 11:
                s += "\nLABELS.update({'attention_weight':'Bobot Attention'})\n"
            if index == 30:
                s = '%%aes\n' + pool_source + '\n\n' + s.removeprefix('%%aes\n')
                s = replace_once(s, 'def __init__(self, scenario, roberta=None, dropout=.1):',
                                 'def __init__(self, scenario, roberta=None, dropout=.1, sequence_hidden=64):')
                marker = "        else:\n            # LayerNorm tambahan pada vektor jawaban RoBERTa, khusus cabang fusion."
                # Module baru dibuat sesudah head, sehingga inisialisasi head E21 tetap sebanding.
                a = s.index(marker)
                tail = s.index('\n    def forward(', a)
                s = s[:tail] + '\n            self.sequence_pool = ' + pool_class + '(sequence_hidden)\n' + s[tail:]
                s = replace_once(s, 'bge=None, return_features=False):', 'bge=None, return_features=False, answer_mask=None):')
                old = "            cls = self.roberta(**windows).last_hidden_state[:, 0, :]\n            # Rata-ratakan vektor CLS semua window milik jawaban yang sama.\n            essay = mean_by_essay(cls, mapping, n_essays)"
                new = """            hidden = self.roberta(**windows).last_hidden_state
            cls = hidden[:, 0, :]
            if self.scenario == 'S3':
                pooled = self.sequence_pool(hidden, answer_mask)
                essay = mean_by_essay(pooled, mapping, n_essays)
                features['roberta_cls_reference'] = mean_by_essay(cls, mapping, n_essays)
                features['answer_pool768'] = essay
                if hasattr(self.sequence_pool, 'last_summary'):
                    features['bigru_hidden128'] = mean_by_essay(self.sequence_pool.last_summary, mapping, n_essays)
            else:
                # Kontrol S1 tetap mengambil CLS, sesuai E21.
                essay = mean_by_essay(cls, mapping, n_essays)"""
                s = replace_once(s, old, new)
                s = s.replace('windows,mapping,_=pair_batch_windows', 'windows,mapping,details=pair_batch_windows')
                marker = "        return self.model(windows=windows, mapping=mapping, n_essays=len(ids), bge=features,"
                s = replace_once(s, marker, """        mask = answer_token_mask(windows, mapping, details, self.tokenizer.padding_side) if self.model.scenario == 'S3' else None
        if return_features and self.model.scenario == 'S3':
            self.pooling_batch = (windows, mask)
""" + marker)
                s = replace_once(s, '                          return_features=return_features)',
                                 '                          return_features=return_features, answer_mask=mask)')
            if index == 31:
                s = replace_once(s, "dropout=cfg['dropout'])", "dropout=cfg['dropout'],sequence_hidden=cfg['s3_sequence_hidden'])")
            if index == 32:
                s = s.replace('RoBERTa 768 → LayerNorm; BGE', 'RoBERTa → ' + ('BiGRU128 → proyeksi768' if mode == 'bigru' else 'attention pooling768') + ' → LayerNorm; BGE')
            if index == 43:
                s = replace_once(s, 'values, window_values, window_ids = {}, [], []', 'values, window_values, window_ids = {}, [], []\n        pooling_rows = []')
                anchor = "                    for name, value in features.items():"
                inspection = """                    if scenario == 'S3' and hasattr(model.sequence_pool, 'last_weights'):
                        windows, mask = engine.pooling_batch
                        weights = model.sequence_pool.last_weights.cpu()
                        for window in range(len(mask)):
                            for position in mask[window].nonzero(as_tuple=False).flatten().tolist():
                                token_id = int(windows['input_ids'][window, position])
                                pooling_rows.append({'row_id':row['row_id'],'window':window,'position':position,
                                    'token_id':token_id,'token':engine.tokenizer.convert_ids_to_tokens([token_id])[0],
                                    'attention_weight':float(weights[window, position])})
"""
                s = replace_once(s, anchor, inspection + anchor)
                s = replace_once(s, "        if window_values:\n", "        if pooling_rows:\n            sheets['Bobot_token_jawaban'] = pd.DataFrame(pooling_rows)\n        if window_values:\n")
            if index == 45:
                s = replace_once(s, "('14_S3_fitur_setelah_training.xlsx','concat','Fusion S3')", "('14_S3_fitur_setelah_training.xlsx','concat','Fusion S3'),\n              ('14_S3_fitur_setelah_training.xlsx','answer_pool768','Pooling Jawaban S3'),\n              ('14_S3_fitur_setelah_training.xlsx','" + ('bigru_hidden128' if mode == 'bigru' else 'Bobot_token_jawaban') + "','" + ('BiGRU S3 State Akhir' if mode == 'bigru' else 'Bobot Attention S3') + "')")
                s = s.replace("['RoBERTa','768 angka:", "['RoBERTa S3','" + description + " Keluaran pooling768 yang ditampilkan, bukan CLS.'],\n        ['RoBERTa','768 angka:")
                s = replace_once(s, "settings={'input_format':'Format input tiga teks'", "settings={'s3_pooling':'Pooling jawaban S3','s3_sequence_hidden':'Hidden pooling S3','input_format':'Format input tiga teks'")
                # Laporan perbandingan menyebut varian S3 secara eksplisit.
                s = s.replace('MODELS.get(r.get(\'Skenario\'))', "(MODELS.get(r.get('Skenario')) + ' + " + ('BiGRU' if mode == 'bigru' else 'Attention Pooling') + "' if r.get('Skenario')=='S3' else MODELS.get(r.get('Skenario')))")
            cell['execution_count'] = None
            cell['outputs'] = []
        cell['source'] = s
    n['cells'][0]['source'] = f'''# {experiment} · E21 + {'BiGRU' if mode == 'bigru' else 'Attention Pooling'} pada S3

Tiga input **soal, jawaban_referensi, response_text**; satu label **rater**. Format input single sequence mengikuti E21, bukan context pair E22.

**S1/S2 tetap mengikuti E21. S3:** {description} Hanya token jawaban yang diringkas oleh layer baru; soal/referensi tetap dibaca self-attention RoBERTa. Tidak memproses768koordinat sebagai urutan. Proyeksi BGE dan concat1024 → MLP256 →1 tetap.

**Web Colab:** Runtime → Change runtime type → T4 GPU, lalu Run all. Pilih fix_dataset_300.xlsx pada Choose Files. Tidak memakai Google Drive; checkpoint dua slot berada di disk runtime lokal. Unduh hasil sebelum runtime berakhir.

LR encoder S1/S3 **5e-6**, proyeksi/head/layer baru **1e-4**; lima fold, maksimum25epoch, patience5; seed42/split_seed2026 mengikuti E21. Ketiga skenario dilatih baru. Tidak ada jaminan skor identik lintas perangkat atau tambahan layer pasti memperbaiki hasil.

Rujukan: [PaSTO-GNN, §3.6 dan Tabel6](https://doi.org/10.3389/frai.2026.1842542); [Dong dkk.2017](https://aclanthology.org/K17-1017/). Ini adaptasi token-level pada RoBERTa, bukan replikasi arsitektur paper. Tidak menambahkan GNN, LoRA, atau head CORAL.
'''
    n['cells'][29]['source'] = f'## Arsitektur {experiment}: S3 dengan {mode}\n\n{description}\n\nS1 linear dan S2 MLP tidak berubah. State/vektor pooling diekspor setelah training; bobot attention bukan bukti sebab-akibat skor.'
    codes = [source(c).removeprefix('%%aes\n') for c in n['cells'] if c['cell_type'] == 'code']
    for code in codes:
        ast.parse(code)
    fingerprint = hashlib.sha256('\n'.join(codes).encode()).hexdigest()
    for cell in n['cells']:
        cell['source'] = source(cell).replace('__NOTEBOOK_CODE_SHA256__', fingerprint)
    n['metadata']['colab']['name'] = name + '.ipynb'
    n['metadata']['aes_standalone']['code_sha256'] = fingerprint
    n['metadata']['aes_experiment'].update(name=experiment, base_commit=PIN, base_path=BASE,
        base_sha256=hashlib.sha256(raw).hexdigest(), s3='e21_three_inputs_fusion_mlp_' + mode,
        s3_pooling=mode, s3_sequence_hidden=64, pooling_tokens='response_only',
        base_cpu_notebook=None, base_cpu_sha256=None, new_layer_lr=1e-4)
    target = ROOT / 'notebooks/experiments' / (name + '.ipynb')
    target.write_text(json.dumps(n, ensure_ascii=False, indent=1) + '\n')
    (ROOT / 'aes-learning/notebooks/experiments' / target.name).write_bytes(target.read_bytes())
    print(json.dumps({'experiment':experiment,'notebook':str(target.relative_to(ROOT)),
                      'code_sha256':fingerprint,'code_cells':len(codes),'pooling':mode}))
    return target


if __name__ == '__main__':
    for experiment in VARIANTS:
        build(experiment)
