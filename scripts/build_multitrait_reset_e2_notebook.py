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

## Jalankan berurutan

Periksa tabel runtime, hubungkan Drive, kemudian jalankan cell **Siapkan dataset dan split**.
Cell tersebut menyiapkan cfg, bundle, manifest, dan output sekaligus.
Pada runtime tanpa CUDA, tahap model menampilkan status belum dijalankan.
Pilih T4 GPU untuk training. Semua tahap mempunyai heading dan output sendiri.

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

    cells = notebook["cells"]
    # Data and split are prepared together, after the split function is defined.
    split_definition = next(
        c
        for c in cells
        if c["metadata"].get("aes_multitrait_role") == "definition:make_split"
    )
    split_index = cells.index(split_definition)
    if cells[split_index - 1]["cell_type"] == "markdown":
        del cells[split_index - 1 : split_index + 1]
    else:
        cells.remove(split_definition)
    prepare_index = next(
        i
        for i, c in enumerate(cells)
        if c["metadata"].get("aes_multitrait_role") == "prepare"
    )
    cells[prepare_index:prepare_index] = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": "### Definisi split siswa\n\nFungsi ini tersedia sebelum dataset dan manifest disiapkan.",
        },
        split_definition,
    ]
    prepare = next(
        c for c in cells if c["metadata"].get("aes_multitrait_role") == "prepare"
    )
    start = prepare["source"].index("    'runtime':")
    end = prepare["source"].index("\n})", start)
    prepare["source"] = (
        prepare["source"][:start]
        + "    'runtime': runtime_information(),"
        + prepare["source"][end:]
    )
    split = next(
        c for c in cells if c["metadata"].get("aes_multitrait_role") == "split_execute"
    )
    start = split["source"].index("print(")
    prepare["source"] += """
# Selesaikan dataset dan split pada cell yang sama.
manifest = make_split(bundle, cfg)
output = run_folder(cfg, bundle)
dump(output / 'split.json', manifest)
print(f"Split siap: {len(manifest['development'])} development, {len(manifest['test'])} test.")
"""
    split["source"] = (
        """if all(name in globals() for name in ('cfg', 'bundle', 'manifest', 'output')):
"""
        + "\n".join("    " + line for line in split["source"][start:].splitlines())
        + """
else:
    print('Jalankan cell Siapkan dataset dan split terlebih dahulu.')
"""
    )
    prepare_index = cells.index(prepare)
    if cells[prepare_index - 1]["cell_type"] == "markdown":
        cells[prepare_index - 1][
            "source"
        ] = "### Siapkan dataset dan split\n\nCell ini membuat cfg, bundle, manifest, dan output sekaligus. Pembacaan data dapat dijalankan pada CPU."
    device_index = next(
        i
        for i, c in enumerate(cells)
        if c["metadata"].get("aes_multitrait_role") == "definition:device_name"
    )
    cells[device_index + 1 : device_index + 1] = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": "### Periksa runtime\n\nTabel menunjukkan perangkat yang benar-benar tersedia di sesi ini.",
        },
        {
            "cell_type": "code",
            "metadata": {"aes_multitrait_role": "runtime_check"},
            "source": (ROOT / "scripts/reset_e2_runtime.py").read_text(),
            "execution_count": None,
            "outputs": [],
        },
    ]
    stage_requirements = {
        "token_preview": ("Cek tokenisasi", ()),
        "bge_execute": ("Encoding BGE", ()),
        "inspection": ("Inspeksi vektor sebelum training", ("bge",)),
        "cv_execute": ("Lima fold CV", ("bge",)),
        "refit_execute": ("Refit model", ("bge", "cv_summary", "folds")),
        "test_execute": ("Evaluasi test", ("bge", "refit_result")),
        "final_features": ("Inspeksi fitur final", ("bge", "refit_result")),
        "export_execute": (
            "Ekspor workbook",
            (
                "cv_summary",
                "folds",
                "final_info",
                "test_prediction",
                "inspection_sheets",
            ),
        ),
        "download": ("Unduh workbook", ("report_path",)),
    }
    for index, cell in enumerate(cells):
        role = cell["metadata"].get("aes_multitrait_role")
        if role in stage_requirements:
            label, required = stage_requirements[role]
            cell["source"] = (
                f"if model_stage_ready({label!r}, {required!r}):\n"
                + "\n".join("    " + line for line in cell["source"].splitlines())
                + "\n"
            )
        cell["id"] = f"reset-e2-{index}"
    notebook["cells"] = cells

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
