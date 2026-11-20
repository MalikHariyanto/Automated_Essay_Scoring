"""Generate the E9 reset notebook with a ProTACT-inspired training package."""

import ast
import hashlib
import json

from build_multitrait_reset_e6_notebook import BASE, GROUP, ROOT, replace
from build_multitrait_reset_e8_notebook import NAME as E8_NAME

NAME = 'AES_Reset_E9_S3_MultiTrait_ProTACTAdaptation.ipynb'
MARKER = '__RESET_E9_CODE_SHA256__'
INTRO = """# E9 reset · S3 multi-trait · Adaptasi ProTACT

Paket E9 memakai attention soal–jawaban per trait, attention antartrait, dan
loss relasi. Ringkasan soal mencari token jawaban yang relevan di setiap
window. Tiap head juga membaca representasi tiga trait lain. Kedua jalur
tambahan mulai dari proyeksi nol. RoBERTa dilatih dan BGE tetap frozen.

Loss = MSE + 0,1 × MSE selisih prediksi antartrait terhadap selisih label.
Pasangan aktif punya korelasi Pearson ≥ 0,7 pada data latih fit saat itu.
Jika tidak ada pasangan, loss relasi nol. Validasi memakai RMSE biasa.

[ProTACT](https://aclanthology.org/2023.findings-acl.98/) memberi ide attention
dan pemilihan pasangan berkorelasi. E9 memakai encoding, split siswa, AdamW,
dan hyperparameter E2. Loss selisih diadaptasi untuk microbatch satu; fitur
topik LDA dan arsitektur POS/LSTM paper berada di luar paket E9.

Pilih T4 GPU lalu Run all dengan workbook semi-final. ZIP ringkas 04–05
tersedia sesudah test. Riwayat 05 mencatat MSE, loss relasi dan pasangan aktif.
"""


def base_model(source):
    helper = (ROOT / 'scripts/reset_e9_protact.py').read_text()
    return replace(source, {
        'class MultiTraitModel(nn.Module):': helper + '\n\nclass MultiTraitModel(nn.Module):',
        'self.head = IndependentTraitHeads(s3_mlp_dropout)': 'self.head = PromptAwareHeads(s3_mlp_dropout)',
        'bge, return_features=False):': 'bge, question_mask, answer_mask, return_features=False):',
        'prediction = self.head(fusion)': 'prediction = self.head(fusion, hidden, question_mask, answer_mask, mapping)',
    })


def build():
    template = BASE.parent / E8_NAME
    notebook = json.loads(template.read_bytes())
    originals = {c['metadata'].get('aes_multitrait_role'): ''.join(c['source'])
                 for c in json.loads(BASE.read_bytes())['cells']}
    restore = {'definition:bge_features', 'bge_execute', 'definition:SingleScoreModel',
               'definition:Engine', 'definition:fit', 'architecture', 'inspection'}
    previous_hash = notebook['metadata']['aes_standalone']['code_sha256']
    for cell in notebook['cells']:
        role = cell['metadata'].get('aes_multitrait_role')
        source = originals[role] if role in restore else ''.join(cell['source'])
        source = (source.replace('Reset_E8', 'Reset_E9').replace('E8 reset', 'E9 reset')
                  .replace('reset_e8_s3_multitrait', 'reset_e9_s3_multitrait')
                  .replace('aes_reset_e8_bge_relevance', 'aes_reset_e9_protact'))
        if role == 'prepare':
            source = replace(source, {"'content_relevance': 'bge_separate_text_log1p_rms_distance',":
                                      "'fusion_variant': 'protact_adaptation', 'trait_corr_threshold': 0.7, 'trait_relation_weight': 0.1,"})
        elif role == 'definition:SingleScoreModel':
            source = base_model(source)
        elif role == 'definition:Engine':
            source = replace(source, {
                '        return self.model(windows=windows': """        question_mask, answer_mask = prompt_answer_masks(
            details, windows['input_ids'].shape[1], self.tokenizer.padding_side, self.device)
        return self.model(question_mask=question_mask, answer_mask=answer_mask, windows=windows""",
            })
        elif role == 'definition:fit':
            source = replace(source, {
                '    seed_everything(seed)': """    pair_indices = correlated_trait_pairs(bundle.y[training], cfg['trait_corr_threshold'])
    active_pairs = [[TRAITS[i][0], TRAITS[j][0]] for i, j in zip(*pair_indices)]
    seed_everything(seed)""",
                '    device = device_name(cfg)': """    device = device_name(cfg)
    pairs = tuple(torch.as_tensor(indices, device=device) for indices in pair_indices)
    print(f'Pasangan trait aktif: {active_pairs}', flush=True)""",
                '        loss_total = 0.': '        loss_total = mse_total = relation_total = 0.',
                '                    loss = torch.mean((prediction-targets)**2)': """                    mse = torch.mean((prediction-targets)**2)
                    relation = relation_error(prediction, targets, pairs)
                    loss = mse + cfg['trait_relation_weight'] * relation""",
                '                loss_total += loss.item()*len(ids)': """                loss_total += loss.item()*len(ids)
                mse_total += mse.item()*len(ids)
                relation_total += relation.item()*len(ids)""",
                "'train_mse_scaled': loss_total/len(training),": """'train_mse_scaled': mse_total/len(training),
               'train_relation_loss': relation_total/len(training),
               'train_total_loss': loss_total/len(training), 'active_trait_pairs': active_pairs,""",
                'loss latih (MSE, target 0–1): {row["train_mse_scaled"]:.6f}':
                    'total loss latih: {row["train_total_loss"]:.6f}',
            })
        elif role == 'architecture':
            source = replace(source, {"    ['Jumlah head', '4'],":
                "    ['Attention soal–jawaban', '4 × 64'],\n    ['Attention antartrait', '4 × 4'],\n    ['Jumlah head', '4'],"})
        elif role == 'report':
            source = source.partition('\n\ndef distance_rows(')[0]
            source = replace(source, {
                'prediction, distance, inspection=None': 'prediction, inspection=None',
                "        'Jarak BGE OOF': distance_rows(bundle, indices, distance),\n": '',
                "        'Jarak BGE Test': distance_rows(bundle, test, distance),\n": '',
            })
        elif role in ('test_execute', 'export_execute'):
            source = source.replace(', bge[:, -2:].numpy()', '')
        if cell['cell_type'] == 'markdown':
            source = source.replace('dengan AdamW dan MSE dari empat target skala 0–1',
                                    'dengan AdamW, MSE target skala 0–1, dan loss relasi antartrait')
        cell['source'] = source.replace(previous_hash, MARKER)

    notebook['cells'][0]['source'] = INTRO
    code = '\n'.join(c['source'] for c in notebook['cells'] if c['cell_type'] == 'code')
    fingerprint = hashlib.sha256(code.encode()).hexdigest()
    for index, cell in enumerate(notebook['cells']):
        cell['id'] = f'reset-e9-{index}'
        cell['source'] = cell['source'].replace(MARKER, fingerprint)
        if cell['cell_type'] == 'code':
            ast.parse(cell['source'])
    notebook['metadata']['colab']['name'] = NAME
    notebook['metadata']['aes_standalone'] = {
        'code_sha256': fingerprint, 'base': BASE.relative_to(ROOT).as_posix(),
        'base_sha256': hashlib.sha256(BASE.read_bytes()).hexdigest(), 'experiment': 'Reset_E9',
    }
    content = json.dumps(notebook, ensure_ascii=False, indent=1) + '\n'
    for folder in (BASE.parent, GROUP / 'notebooks'):
        (folder / NAME).write_text(content)


if __name__ == '__main__':
    build()
