"""Build E29 with one independently learned ScalarMix per scoring trait."""

import hashlib
import json

from build_multitrait_reset_e28_notebook import BASE, GROUP, ROOT, NAME as SOURCE_NAME, replace

NAME = 'AES_Reset_E29_S3_ScalarMix_PerTrait.ipynb'
EXPERIMENT = 'Reset_E29_S3_ScalarMix_PerTrait'
MARKER = '__RESET_E29_CODE_SHA256__'

INTRO = """# E29 · ScalarMix khusus tiap trait

Isi, Organisasi, Kebahasaan dan Mekanik memiliki empat bobot campuran layer
serta gamma masing-masing. Encoder RoBERTa hanya dijalankan sekali per batch.
Empat representasi hasil campuran masuk ke head trait terkait; attention
antartrait tetap memakai empat hidden head seperti E28.

Awalnya semua campuran 0,25 per layer dan gamma 1. E29 memiliki 20 parameter
campuran, tambahan 15 dibanding E28. Setelan training, split, seed dasar 44,
loss dan aturan checkpoint A/B mengikuti E28. Layer baru masuk optimizer LR 1e-4.

Pilih T4, hubungkan Drive ke workbook semi-final, lalu Run all. Lima fit
development langsung menampilkan metrik per fold. Fold selesai tersimpan.
ZIP 04–05 memuat empat metrik per trait, bobot layer/gamma tiap trait dan
perbandingan checkpoint B dengan E28 serta E25 dari arsip agregat run aktual.

Arsitektur ini adaptasi proyek dari campuran layer E43 dan head E25.
Manfaat tiap trait masih harus dibuktikan pada validasi.
"""

HEAD_FORWARD = '''
    def trait_state(self, fusion, tokens, question_mask, answer_mask, mapping, index):
        name = TRAITS[index][0]
        linear, activation, dropout, _ = self.heads[name]
        state = linear(fusion)
        if name != 'organisasi':
            prompt = (tokens * question_mask.unsqueeze(-1)).sum(1) / question_mask.sum(1, keepdim=True)
            query = self.question_query(prompt).reshape(len(tokens), 4, 64)[:, index:index + 1]
            keys = self.answer_key(tokens)
            logits = query.float() @ keys.float().transpose(1, 2) / 8
            weights = logits.masked_fill(~answer_mask[:, None, :], float('-inf')).softmax(-1)
            pooled = (weights.to(tokens.dtype) @ tokens).squeeze(1)
            context = mean_by_essay(pooled, mapping, len(fusion))
            state = state + self.prompt_residual(context)
        return dropout(activation(state))

    def finish(self, states):
        hidden = torch.stack(states, dim=1)
        logits = hidden.float() @ hidden.float().transpose(1, 2) / 8
        diagonal = torch.eye(4, dtype=torch.bool, device=hidden.device)
        weights = logits.masked_fill(diagonal, float('-inf')).softmax(-1)
        updated = hidden + self.trait_residual(weights.to(hidden.dtype) @ hidden)
        return torch.cat([self.heads[name][3](updated[:, i])
                          for i, (name, *_) in enumerate(TRAITS)], dim=1)
'''

MODEL_FORWARD = '''
    def forward(self, windows, mapping, n_essays, bge, question_mask, answer_mask, return_features=False):
        encoded = self.roberta(**windows, output_hidden_states=True, return_dict=True)
        projected = self.projection(bge)
        states, features = [], {}
        for index, (name, *_) in enumerate(TRAITS):
            hidden = self.scalar_mix[name](encoded.hidden_states, windows['attention_mask'])
            cls = mean_by_essay(hidden[:, 0, :], mapping, n_essays)
            fusion = torch.cat([self.roberta_norm(cls), projected], dim=1)
            states.append(self.head.trait_state(fusion, hidden, question_mask, answer_mask, mapping, index))
            if return_features:
                features[f'roberta_{name}'] = cls
                features[f'concat_{name}'] = fusion
        prediction = self.head.finish(states)
        if return_features:
            features.update(bge=bge, projection=projected)
            return prediction, features
        return prediction
'''

COMPARISON = '''
def compare_reference(worksheets, reference):
    current = worksheets['OOF A B per Trait'].query("pilihan == 'B'").set_index('trait')
    rows = []
    for trait_row in reference['metrics']:
        trait = trait_row['trait']
        for metric in ('RMSE', 'Pearson', 'QWK', 'Gwet AC2'):
            value = current.loc[trait, metric]
            baseline = trait_row[metric]
            delta = None if pd.isna(value) or pd.isna(baseline) else value - baseline
            rows.append({'trait': trait, 'metrik': metric, 'E29 (B)': value,
                         'Pembanding (B)': baseline, 'delta E29 minus pembanding': delta})
    return pd.DataFrame(rows)
'''


def snapshot(name):
    run = json.loads((GROUP / f'output/checkpoints/{name}.json').read_text())
    return {
        'split_sha256': run['metadata']['split_sha256'],
        'seed': run['config']['seed'], 'source_sha256': run['source']['sha256'],
        'metrics': [{k: row[k] for k in ('trait', 'RMSE', 'Pearson', 'QWK', 'Gwet AC2')}
                    for row in run['tables']['OOF A B per Trait'] if row['pilihan'] == 'B'],
    }


def build():
    base = BASE.parent / SOURCE_NAME
    raw = base.read_bytes()
    notebook = json.loads(raw)
    previous_hash = notebook['metadata']['aes_standalone']['code_sha256']
    for index, cell in enumerate(notebook['cells']):
        source = ''.join(cell['source']).replace(previous_hash, MARKER)
        role = cell['metadata'].get('aes_multitrait_role')
        if index == 0:
            source = INTRO
        elif role == 'prepare':
            source = replace(source, {
                'Reset_E28_S3_ScalarMix4_MultiTrait': EXPERIMENT,
                "'fusion_variant': 'e25_scalar_mix4'": "'fusion_variant': 'e25_scalar_mix_per_trait'",
                "'scalar_mix_layers': 4,": "'scalar_mix_layers': 4, 'scalar_mix_per_trait': True,",
                'output = run_folder(cfg, bundle)': (
                    f"E28_REFERENCE = {snapshot('E28')!r}\n"
                    f"E25_REFERENCE = {snapshot('E25')!r}\n"
                    "if any(ref['split_sha256'] != manifest['split_sha256'] or ref['seed'] != cfg['seed']\n"
                    "       for ref in (E28_REFERENCE, E25_REFERENCE)):\n"
                    "    raise ValueError('Split dan seed harus cocok dengan E28 dan E25.')\n"
                    'output = run_folder(cfg, bundle)'
                ),
            })
        elif role == 'definition:SingleScoreModel':
            start = source.index('    def forward(', source.index('class PromptAwareHeads'))
            end = source.index('\n\ndef prompt_answer_masks', start)
            source = source[:start] + HEAD_FORWARD + source[end:]
            source = replace(source, {
                'self.scalar_mix = ScalarMix(scalar_mix_layers, gamma_initial)': (
                    'self.scalar_mix = nn.ModuleDict({\n'
                    '            name: ScalarMix(scalar_mix_layers, gamma_initial) for name, *_ in TRAITS\n'
                    '        })'
                ),
            })
            start = source.index('    def forward(', source.index('class MultiTraitModel'))
            source = source[:start] + MODEL_FORWARD
        elif role == 'fold_helpers':
            source = replace(source, {
                "info['scalar_mix_B'] = model.scalar_mix.report(model.roberta.config.num_hidden_layers)": (
                    "info['scalar_mix_B'] = [\n"
                    "            {'trait': name, **row} for name, *_ in TRAITS\n"
                    "            for row in model.scalar_mix[name].report(model.roberta.config.num_hidden_layers)\n"
                    "        ]"
                ),
            })
        elif role == 'validation_report':
            start = source.index('def compare_architecture(')
            end = source.index('def export_validation(', start)
            source = source[:start] + COMPARISON + '\n\n' + source[end:]
            source = replace(source, {
                "        comparison, overview = compare_architecture(worksheets)\n"
                "        worksheets['E28 vs E25 per Trait'] = comparison\n"
                "        worksheets['E28 vs E25 Ringkasan'] = overview": (
                    "        worksheets['E29 vs E28 per Trait'] = compare_reference(worksheets, E28_REFERENCE)\n"
                    "        worksheets['E29 vs E25 per Trait'] = compare_reference(worksheets, E25_REFERENCE)"
                ),
                "'e25_seed44_source_sha256': E25_SEED44_REFERENCE['zip_sha256'],": (
                    "'e25_seed44_source_sha256': E25_REFERENCE['source_sha256'],\n"
                    "        'e28_source_sha256': E28_REFERENCE['source_sha256'],"
                ),
            })
        elif role == 'architecture':
            source = source.replace('4 layer · softmax awal 0,25; gamma awal 1',
                                    '4 layer per trait · softmax awal 0,25; gamma awal 1')
            source = source.replace("'5 skalar terlatih'", "'20 skalar terlatih; +15 dibanding E28'")
        elif role == 'final_report':
            source = replace(source, {
                "        display(sheets['E28 vs E25 Ringkasan'].round(4))\n"
                "        display(sheets['E28 vs E25 per Trait'].round(4))": (
                    "        print('E29 dibanding E28 · delta = E29 minus pembanding')\n"
                    "        display(sheets['E29 vs E28 per Trait'].round(4))\n"
                    "        print('E29 dibanding E25')\n"
                    "        display(sheets['E29 vs E25 per Trait'].round(4))"
                ),
            })
        # Only change run/output names after handling role-specific source.
        source = source.replace('aes_reset_e28_s3_scalarmix4_seed44', 'aes_reset_e29_scalar_mix_per_trait')
        source = source.replace('Hasil_AES_Reset_E28_Progres', 'Hasil_AES_Reset_E29_Progres')
        source = source.replace('Hasil_AES_Reset_E28_Validasi', 'Hasil_AES_Reset_E29_Validasi')
        cell['source'] = source
        cell['id'] = f'reset-e29-{index}'

    fingerprint = hashlib.sha256('\n'.join(c['source'] for c in notebook['cells']
                                            if c['cell_type'] == 'code').encode()).hexdigest()
    for cell in notebook['cells']:
        cell['source'] = cell['source'].replace(MARKER, fingerprint)
        if cell['cell_type'] == 'code':
            compile(cell['source'], cell['id'], 'exec')
    notebook['metadata']['colab']['name'] = NAME
    notebook['metadata']['aes_standalone'] = {
        'code_sha256': fingerprint, 'base': base.relative_to(ROOT).as_posix(),
        'base_sha256': hashlib.sha256(raw).hexdigest(), 'experiment': EXPERIMENT,
    }
    content = json.dumps(notebook, ensure_ascii=False, indent=1) + '\n'
    for folder in (BASE.parent, GROUP / 'notebooks'):
        (folder / NAME).write_text(content, encoding='utf-8')


if __name__ == '__main__':
    build()
