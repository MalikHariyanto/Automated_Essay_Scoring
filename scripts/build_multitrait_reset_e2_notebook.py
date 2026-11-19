"""Build reset E2 from reset E1, changing only the scoring head and run identity."""

import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "notebooks/multitrait_reset/AES_Reset_E1_S3_MultiTrait.ipynb"
NAME = "AES_Reset_E2_S3_MultiTrait_TraitHeads.ipynb"
PLACEHOLDER = "__RESET_E2_CODE_SHA256__"

TRAIT_HEADS = '''class IndependentTraitHeads(nn.Module):
    """Four independent 1024→64→1 scorers, in the declared trait order."""

    def __init__(self, dropout):
        super().__init__()
        self.heads = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(1024, 64),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(64, 1),
            )
            for name, *_ in TRAITS
        })

    def forward(self, fusion):
        # Input features come from text. The four teacher scores enter the loss.
        return torch.cat(
            [self.heads[name](fusion) for name, *_ in TRAITS], dim=1
        )

'''

INITIALIZE_HEADS = """
        # Preserve the shared E1 initialization and post-initialization RNG state.
        # Replacing self.head unregisters the temporary shared E1 scoring layers.
        with torch.random.fork_rng(devices=[]):
            self.head = IndependentTraitHeads(s3_mlp_dropout)

"""

INTRO = """# E2 reset · S3 multi-trait · Native Learnable Code

Jalankan E2 di runtime GPU Colab tersendiri agar bisa paralel dengan E1 reset.
Hubungkan Drive ke dataset semi-final yang sama, lalu Run all dari atas.

## Yang diuji

E1 reset memakai satu MLP bersama 1024→256→4.
E2 reset memakai empat MLP 1024→64→1, satu untuk isi, organisasi, kebahasaan, dan mekanik.
Masing-masing membaca vektor concat yang sama dari RoBERTa CLS terakhir dan proyeksi BGE frozen.
Urutan kolom keluaran mengikuti urutan empat label pada cell pemetaan.

Empat head mempunyai 262.660 parameter, dibanding 263.428 pada head E1.
Dimensi 64 adalah pilihan rancangan proyek agar anggaran parameter head mendekati E1.
Ide jalur khusus trait mengadaptasi [CTS, Ridley dkk., AAAI 2021](https://github.com/robert1ridley/cross-prompt-trait-scoring/blob/HEAD/models/CTS.py).
CTS menggunakan representasi LSTM per trait dan attention antartait, sedangkan E2 reset menguji pemisahan MLP pada fitur fusion sederhana.

## Resep training

RoBERTa fine-tuned LR5e-6; proyeksi/head/norm LR1e-4; dropout0,2; AdamW/decay0,01;
warmup10%, scheduler linear, gradient clip1, microbatch1/batch efektif8.
Lima fold development, maksimal25epoch/patience5, satu refit dan test.
MSE memakai target normalisasi per trait. Pemilihan epoch memakai RMSE validasi pada skala asli.

## Keluaran

Satu file Hasil_AES_Reset_E2_S3_MultiTrait.xlsx memuat tabel data, token dan coverage,
vektor sebelum training, inspeksi token/kata/kalimat pada satu contoh development,
fitur model final seluruh jawaban, riwayat training, OOF, dan evaluasi test empat trait.
Workbook berisi data siswa dan disimpan secara privat. Folder hasil E2 terpisah dari E1.

Pilih GPU → Run all → hubungkan Drive. Tautan dataset semi-final sama dengan E1 reset.
"""


def build():
    raw = BASE.read_bytes()
    notebook = json.loads(raw)
    base_code_hash = notebook["metadata"]["aes_standalone"]["code_sha256"]

    for index, cell in enumerate(notebook["cells"]):
        source = "".join(cell["source"]).replace(base_code_hash, PLACEHOLDER)
        role = cell["metadata"].get("aes_multitrait_role")
        if role == "definition:SingleScoreModel":
            source = source.replace("E1 reset:", "E2 reset:")
            anchor = "\n    def forward("
            if source.count(anchor) != 1:
                raise ValueError("E1 model forward anchor changed.")
            source = source.replace(anchor, INITIALIZE_HEADS + "    def forward(", 1)
            source = TRAIT_HEADS + source
        elif role == "architecture":
            source = """display(pd.DataFrame([
    ['RoBERTa CLS layer terakhir', '768'],
    ['BGE frozen', '1024'],
    ['Proyeksi BGE', '256'],
    ['Fusion concat', '1024'],
    ['MLP per trait', '1024 → 64 → 1'],
    ['Jumlah head', '4'],
], columns=['Bagian', 'Dimensi']))
"""
        elif role == "prepare":
            source = source.replace(
                "'experiment': 'Reset_E1',",
                "'experiment': 'Reset_E2', 'head_type': 'independent_trait_heads', 'trait_hidden': 64,",
            )
        if cell["cell_type"] == "code":
            source = source.replace("reset_e1_s3_multitrait", "reset_e2_s3_multitrait")
            source = source.replace("'Reset_E1'", "'Reset_E2'")
            source = source.replace("Hasil_AES_Reset_E1", "Hasil_AES_Reset_E2")
            source = source.replace("E1 reset ·", "E2 reset ·")
            source = source.replace("Refit E1 reset", "Refit E2 reset")
        else:
            source = source.replace("E1 reset", "E2 reset")
            source = source.replace(
                "MLP bersama menghasilkan empat skor",
                "empat MLP khusus trait menghasilkan satu skor masing-masing",
            )
            source = source.replace(
                "lalu concat dan MLP bersama menghasilkan empat skor",
                "lalu concat dan empat MLP khusus trait menghasilkan empat skor",
            )
        cell["source"] = source
        cell["id"] = f"reset-e2-{index}"

    notebook["cells"][0]["source"] = INTRO
    fingerprint = hashlib.sha256(
        "\n".join(
            c["source"] for c in notebook["cells"] if c["cell_type"] == "code"
        ).encode()
    ).hexdigest()
    for cell in notebook["cells"]:
        cell["source"] = cell["source"].replace(PLACEHOLDER, fingerprint)
        if cell["cell_type"] == "code":
            ast.parse(cell["source"])
            cell["outputs"] = []
            cell["execution_count"] = None
    notebook["metadata"]["colab"]["name"] = NAME
    notebook["metadata"]["aes_standalone"] = {
        "code_sha256": fingerprint,
        "base": BASE.relative_to(ROOT).as_posix(),
        "base_sha256": hashlib.sha256(raw).hexdigest(),
        "experiment": "Reset_E2",
    }

    serialized = json.dumps(notebook, ensure_ascii=False, indent=1) + "\n"
    for folder in (
        ROOT / "notebooks/multitrait_reset",
        ROOT / "eksperimen/03_tiga_input_multi_trait/reset/notebooks",
    ):
        folder.mkdir(parents=True, exist_ok=True)
        (folder / NAME).write_text(serialized)
    print(f"{NAME} · {fingerprint}")


if __name__ == "__main__":
    build()
