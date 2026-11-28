"""Build the E31-controlled, development-only E33 inter-trait gate experiment."""

import hashlib
import json

from build_multitrait_reset_e31_notebook import BASE, GROUP, ROOT, NAME as SOURCE_NAME, replace
from build_multitrait_reset_e29_notebook import snapshot

NAME = 'AES_Reset_E33_S3_PerTrait_EMA_TraitGate.ipynb'
EXPERIMENT = 'Reset_E33_S3_PerTrait_EMA_TraitGate'
MARKER = '__RESET_E33_CODE_SHA256__'

INTRO = """# E33 · Gerbang informasi antartrait pada E31

E32 memberi trade-off: QWK Isi/Organisasi naik sedikit, tetapi RMSE keduanya
dan Pearson semua trait memburuk. E33 kembali ke objective E31 tanpa ordinal.
Satu perubahan arsitektur: tiap trait mengatur besarnya residual antartrait
berdasarkan hidden-nya sendiri. Rumus: updated_i = hidden_i + gate_i × residual_i,
dengan gate_i = 2 × sigmoid(Linear64→1(hidden_i)). Empat gate menambah 260 parameter.

Bobot dan bias gate mulai nol sehingga gate awal tepat 1 dan fungsi awal sama
dengan E31. Rentang gate 0–2 memungkinkan memperlemah/memperkuat konteks. Gerbang
tidak memisahkan gradien encoder bersama. Hipotesis gangguan antartrait belum
terbukti; tambahan gerbang bisa juga tidak membantu. Residual awal nol mengikuti
E31 sehingga gradien gate baru bermakna setelah jalur residual mulai belajar.

RoBERTa+BGE, ScalarMix per trait, EMA, LR 5e-6/1e-4, dropout 0,2, weight decay
0,01, seed dasar 44, split dan aturan checkpoint A/B dipertahankan. Maksimal
25 epoch, patience 5. Tidak mengulang HPO dan tidak mengevaluasi test.

Pilih T4, Run all, hubungkan Drive dan workbook semi-final seperti E31. Lima
cell fold berjalan berurutan dan menampilkan hasil langsung. Fold selesai
bisa dipakai ulang; fold terputus diulang. ZIP progres diperbarui tiap fold.
ZIP final Hasil_AES_Reset_E33_Validasi_04_05.zip berisi Excel 04 dan 05 serta
manifest seperti E31. Tambahan tabel diagnostik gate memakai validasi OOF B,
diringkas per fold dan trait. Tidak memakai skor guru sebagai input gerbang.

[ProTACT 2023](https://aclanthology.org/2023.findings-acl.98/) memberi konteks relasi
trait; [Wang & Yu 2025](https://aclanthology.org/2025.findings-emnlp.142/) membahas
routing fitur trait. Kuartil sesuai filter penelitian belum terverifikasi.
Rumus gerbang, skala 2 dan inisialisasi adalah desain lokal, bukan resep paper.
"""

INIT = '''
        # Preserve E31's random stream and every shared initial parameter.
        with torch.random.fork_rng(devices=[]):
            self.trait_gates = nn.ModuleList([nn.Linear(64, 1) for _ in TRAITS])
        for gate in self.trait_gates:
            nn.init.zeros_(gate.weight)
            nn.init.zeros_(gate.bias)
'''

GATE_METHODS = '''
    def begin_gate_capture(self):
        self.capture_gate = True
        self.gate_count = 0
        self.gate_sum = self.gate_square_sum = self.gate_min = self.gate_max = None

    def end_gate_capture(self):
        self.capture_gate = False
        if self.gate_count == 0:
            raise RuntimeError('Diagnostik gate membutuhkan prediksi validasi B.')
        mean = self.gate_sum / self.gate_count
        std = (self.gate_square_sum / self.gate_count - mean.square()).clamp_min(0).sqrt()
        return [{'trait': name, 'n': self.gate_count,
                 'gate mean': float(mean[i].item()), 'gate std': float(std[i].item()),
                 'gate min': float(self.gate_min[i].item()), 'gate max': float(self.gate_max[i].item()),
                 'weight norm': float(self.trait_gates[i].weight.detach().float().norm().item()),
                 'bias': float(self.trait_gates[i].bias.detach().float().item())}
                for i, (name, *_) in enumerate(TRAITS)]

    @torch.no_grad()
    def observe_gates(self, gates):
        values = gates.detach().double().squeeze(-1)
        sums, squares = values.sum(0), values.square().sum(0)
        low, high = values.amin(0), values.amax(0)
        if self.gate_count == 0:
            self.gate_sum, self.gate_square_sum = sums, squares
            self.gate_min, self.gate_max = low, high
        else:
            self.gate_sum += sums
            self.gate_square_sum += squares
            self.gate_min = torch.minimum(self.gate_min, low)
            self.gate_max = torch.maximum(self.gate_max, high)
        self.gate_count += len(values)
'''


def build():
    base = BASE.parent / SOURCE_NAME
    raw = base.read_bytes()
    notebook = json.loads(raw)
    previous_hash = notebook['metadata']['aes_standalone']['code_sha256']
    baseline = json.loads((GROUP / 'output/checkpoints/E31.json').read_text())
    folds = [r for r in baseline['tables']['Pilihan Checkpoint'] if r['pilihan'] == 'B']
    for index, cell in enumerate(notebook['cells']):
        source = ''.join(cell['source']).replace(previous_hash, MARKER)
        role = cell['metadata'].get('aes_multitrait_role')
        source = source.replace('E31', 'E33').replace('e31', 'e33')
        if index == 0:
            source = INTRO
        elif role == 'prepare':
            ref_line = next(line for line in source.splitlines() if line.startswith('E29_REFERENCE ='))
            fold_line = next(line for line in source.splitlines() if line.startswith('E29_FOLD_REFERENCE ='))
            source = replace(source, {
                'Reset_E33_S3_PerTrait_EMA': EXPERIMENT,
                "'runtime': runtime_information(),":
                    "'runtime': runtime_information(), 'trait_gate': 'own_hidden_2sigmoid_zero_init', 'ordinal_loss_weight': 0.0,",
                ref_line: f'E31_REFERENCE = {snapshot("E31")!r}',
                fold_line: f'E31_FOLD_REFERENCE = {folds!r}',
            })
            source = source.replace('E29_REFERENCE[', 'E31_REFERENCE[')
            source = source.replace('Split dan seed harus cocok dengan E29.', 'Split dan seed harus cocok dengan E31.')
        elif role == 'definition:SingleScoreModel':
            source = replace(source, {
                '        nn.init.zeros_(self.trait_residual.weight)':
                    '        nn.init.zeros_(self.trait_residual.weight)\n' + INIT,
                '    def finish(self, states):': GATE_METHODS + '\n    def finish(self, states):',
                '        updated = hidden + self.trait_residual(weights.to(hidden.dtype) @ hidden)':
                    "        gates = torch.stack([2 * torch.sigmoid(gate(hidden[:, i]))\n"
                    "                             for i, gate in enumerate(self.trait_gates)], dim=1)\n"
                    "        if getattr(self, 'capture_gate', False):\n"
                    "            self.observe_gates(gates)\n"
                    "        updated = hidden + gates * self.trait_residual(weights.to(hidden.dtype) @ hidden)",
            })
        elif role == 'architecture':
            source = source.replace("['Attention antartrait', 'E25']", "['Attention antartrait', 'E31 + gate 64→1 per trait, rentang 0–2, awal 1']")
            source = source.replace('20 skalar terlatih; +15 dibanding E28', '260 parameter gate tambahan dibanding E31')
        elif role == 'fold_helpers':
            source = replace(source, {
                '        fitted_valid = predictions_array(engine.predict(validation), validation)':
                    "        model.head.begin_gate_capture()\n"
                    "        fitted_valid = predictions_array(engine.predict(validation), validation)\n"
                    "        info['gate_B'] = model.head.end_gate_capture()\n"
                    "        if any(row['n'] != len(validation) for row in info['gate_B']):\n"
                    "            raise ValueError('Jumlah observasi gate harus cocok dengan validasi B.')",
                "    display(pd.DataFrame(options).round(4))":
                    "    display(pd.DataFrame(options).round(4))\n"
                    "    print('Gerbang antartrait pada validasi checkpoint B')\n"
                    "    display(pd.DataFrame(info['gate_B']).round(6))",
            })
        elif role in ('validation_report', 'final_report'):
            source = source.replace('E33 vs E29', 'E33 vs E31').replace('E33 dibanding E29', 'E33 dibanding E31')
            source = source.replace('E29_REFERENCE', 'E31_REFERENCE').replace('E29_FOLD_REFERENCE', 'E31_FOLD_REFERENCE')
            source = source.replace('E29 RMSE', 'E31 RMSE').replace('E29 macro QWK', 'E31 macro QWK').replace('E29 epoch', 'E31 epoch')
            source = source.replace("'e29_source_sha256'", "'e31_source_sha256'")
            if role == 'validation_report':
                source = replace(source, {
                    "            'Konfigurasi': pd.DataFrame":
                        "            'Gate Validasi B': pd.DataFrame([\n"
                        "                {'fold': number, 'checkpoint': 'B', **row}\n"
                        "                for number, result in sorted(fold_results.items())\n"
                        "                for row in result['training']['gate_B']]),\n"
                        "            'Konfigurasi': pd.DataFrame",
                })
        source = source.replace('aes_reset_e33_pertrait_ema', 'aes_reset_e33_pertrait_ema_trait_gate')
        cell['source'], cell['id'] = source, f'reset-e33-{index}'

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
