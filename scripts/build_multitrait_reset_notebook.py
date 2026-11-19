"""Build a separate reset-series E1 without altering the earlier experiment series."""

import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "notebooks/experiments/AES_E3_S3_MultiTrait_Dropout02_3Sekolah.ipynb"
NAME = "AES_Reset_E1_S3_MultiTrait.ipynb"


def build():
    n = json.loads(BASE.read_text())
    old = n["metadata"]["aes_standalone"]["code_sha256"]
    remove = {
        "definition:answer_token_mask",
        "definition:check_answer_pool_inputs",
        "definition:AnswerBiGRUAttention",
        "definition:reference_token_mask",
        "definition:AnswerBiGRUCoAttentionMeanPool",
    }
    cells = []
    for c in n["cells"]:
        role = c["metadata"].get("aes_multitrait_role")
        if role in remove:
            if cells and cells[-1]["cell_type"] == "markdown":
                cells.pop()
            continue
        s = "".join(c["source"]).replace(old, "__CODE_HASH__")
        if role == "columns":
            s = s.replace("e43_s3_multitrait_three_schools", "reset_e1_s3_multitrait")
        if role == "dataset_definitions":
            s = s.replace(
                "'text': clean(raw[ANSWER_COLUMN]),",
                "'raw_text': raw[ANSWER_COLUMN], 'text': clean(raw[ANSWER_COLUMN]),",
            )
        if role == "definition:SingleScoreModel":
            s = (ROOT / "scripts/reset_multitrait_model.py").read_text()
        if role == "definition:Engine":
            start = s.index("        mask = answer_token_mask")
            end = s.index("    def predict", start)
            s = (
                s[:start]
                + """        return self.model(windows=windows, mapping=mapping, n_essays=len(ids),
                          bge=features, return_features=return_features)

"""
                + s[end:]
            )
        if role == "definition:make_model":
            s = """def make_model(scenario, cfg):
    return MultiTraitModel(scenario, encoder(cfg,'roberta'),
                           s3_mlp_dropout=cfg['s3_mlp_dropout'])
"""
        if role == "definition:optimizer_groups":
            s = s.replace(", 'scalar_mix': []", "").replace(
                ", 'scalar_mix': cfg['scalar_mix_lr']", ""
            )
            s = s.replace(
                "        elif name.startswith('scalar_mix.'):\n            group = 'scalar_mix'\n",
                "",
            )
        if role == "definition:fit":
            start = s.index("    info['scalar_mix_weights']")
            s = s[:start] + "    return model, engine, info\n"
        if role == "report":
            s = s.replace(
                "info['scalar_mix_weights']", "info.get('scalar_mix_weights', [])"
            )
            s = s.replace(
                "final_info['scalar_mix_weights']",
                "final_info.get('scalar_mix_weights', [])",
            )
            s = s.replace(
                "        'Scalar Mix': pd.DataFrame(mixture),",
                "        **inspection_sheets,",
            )
            s = s.replace(
                "cv_fold, oof_rows, history, mixture, indices, predictions = [], [], [], [], [], []",
                "cv_fold, oof_rows, history, indices, predictions = [], [], [], [], []",
            )
            s = (
                "\n".join(
                    line for line in s.splitlines() if "mixture.extend" not in line
                )
                + "\n"
            )
        if role == "refit_execute":
            s = s.replace(
                "display(pd.DataFrame(final_info['scalar_mix_weights']))",
                "display(pd.DataFrame(final_info['history']))",
            )
        if role == "architecture":
            s = """display(pd.DataFrame([
    ['RoBERTa CLS layer terakhir',768],
    ['BGE frozen',1024],['Proyeksi BGE',256],
    ['Fusion concat',1024],['MLP hidden',256],['Keluaran trait',4],
],columns=['Bagian','Dimensi']))
"""
        if role == "configuration_defaults":
            s = s.replace(
                "'s3_pooling': 'bigru_coattention_mean_pooling'",
                "'s3_pooling': 'last_cls'",
            )
            start = s.index("    if result['s3_pooling']")
            end = s.index("    if result['s3_fusion']", start)
            s = s[:start] + s[end:]
        if role == "hyperparameters":
            s = (
                "\n".join(
                    line
                    for line in s.splitlines()
                    if not line.startswith(
                        ("BIGRU_", "POOLING_ATTENTION_", "SCALAR_MIX_")
                    )
                )
                + "\n"
            )
        if role == "configuration_defaults":
            s = (
                "\n".join(
                    line
                    for line in s.splitlines()
                    if not any(
                        x in line
                        for x in [
                            "'s3_pooling':",
                            "'scalar_mix_gamma_init':",
                        ]
                    )
                )
                + "\n"
            )

        if c["cell_type"] == "markdown":
            s = s.replace("E3", "E1 reset").replace("E43", "baseline sederhana")
            if "ScalarMix" in s or "Scalar Mix" in s or "BiGRU" in s:
                s = "### Model fusion multi-trait\n\nRoBERTa memakai CLS layer terakhir dan mean antarwindow. BGE frozen diproyeksikan ke 256, lalu concat dan MLP bersama menghasilkan empat skor.\n"
        s = (
            s.replace("'E3'", "'Reset_E1'")
            .replace("E3 ·", "E1 reset ·")
            .replace("Refit E3", "Refit E1 reset")
            .replace("Hasil_AES_E3", "Hasil_AES_Reset_E1")
        )
        if role == "prepare":
            s = s.replace(
                "'experiment': 'Reset_E1',",
                "'experiment': 'Reset_E1', 'series': 'multitrait_reset',",
            )
        c["source"] = s
        cells.append(c)
    legacy = json.loads(
        (
            ROOT
            / "notebooks/experiments/AES_300_E43_E42_S3_MaskedMeanPooling_Native_Learnable_GPU.ipynb"
        ).read_text()
    )
    roles = {
        c["metadata"].get("aes43_role"): "".join(c["source"]) for c in legacy["cells"]
    }

    def code(role, source):
        return {
            "cell_type": "code",
            "metadata": {"aes_multitrait_role": role},
            "source": source,
            "execution_count": None,
            "outputs": [],
        }

    at = next(
        i
        for i, c in enumerate(cells)
        if c["metadata"].get("aes_multitrait_role") == "cv_execute"
    )
    definitions = []
    for role in [
        "definition:vector_frame",
        "definition:pair_token_frames",
        "definition:paired_token_vectors",
        "word_helpers",
        "definition:sentence_spans",
        "definition:token_vectors",
        "definition:word_vectors",
        "definition:compare_repetitions",
        "definition:vector_lab",
        "definition:export_word_sentence_vectors",
        "definition:export_tokens",
    ]:
        source = roles[role]
        if role == "definition:vector_lab":
            start = source.index("            report = json.loads")
            end = source.index("            model = wrapper.roberta", start)
            source = source[:start] + """            checkpoint = output/'final.pt'
            payload = torch.load(checkpoint,map_location='cpu',weights_only=True)
            wrapper = make_model('S3',cfg)
            wrapper.load_state_dict(payload['state'],strict=True)
            del payload
""" + source[end:]
        definitions.append(code(role, source))
    definitions.append(
        code(
            "inspection", (ROOT / "scripts/reset_multitrait_inspection.py").read_text()
        )
    )
    cells[at:at] = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": "## Output belajar · data, token, vektor kata dan kalimat sebelum training",
        }
    ] + definitions
    at = next(
        i
        for i, c in enumerate(cells)
        if c["metadata"].get("aes_multitrait_role") == "export_execute"
    )
    cells.insert(
        at,
        code(
            "final_features",
            (ROOT / "scripts/reset_multitrait_features.py").read_text(),
        ),
    )
    # Explain each native helper and keep each execution stage visible.
    presented = []
    for cell in cells:
        role = cell["metadata"].get("aes_multitrait_role", "")
        if cell["cell_type"] == "code" and (
            not presented or presented[-1]["cell_type"] != "markdown"
        ):
            label = role.split(":")[-1].replace("_", " ")
            presented.append(
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": f"### {label}\n\nCell ini menjalankan bagian {label}.",
                }
            )
        presented.append(cell)
    cells = presented
    n["cells"] = cells
    n["cells"][0]["source"] = """# E1 reset · S3 multi-trait · Native Learnable Code

Rangkaian baru menggunakan fusion sederhana. Pilih GPU, Run all, dan hubungkan Drive dataset semi-final.

RoBERTa Indonesia fine-tuned → CLS layer terakhir → mean antarwindow → LayerNorm768.
BGE-M3 frozen → proyeksi256 + LayerNorm.
Concat1024 → MLP256 → GELU → Dropout0,2 → empat skor.

LR encoder5e-6, proyeksi/head1e-4, AdamW decay0,01, warmup10%, batch efektif8/micro1,
maks25epoch/patience5, lima fold, satu refit dan test.
Loss MSE target normalisasi, pemilihan epoch RMSE skala asli.
Satu workbook memuat data, token, fitur, riwayat training, OOF, prediksi test, dan metrik.
"""
    fp = hashlib.sha256(
        "\n".join(c["source"] for c in cells if c["cell_type"] == "code").encode()
    ).hexdigest()
    for i, c in enumerate(cells):
        c["source"] = c["source"].replace("__CODE_HASH__", fp)
        c["id"] = f"reset-e1-{i}"
        if c["cell_type"] == "code":
            ast.parse(c["source"])
            c["outputs"] = []
            c["execution_count"] = None
    n["metadata"]["colab"]["name"] = NAME
    n["metadata"]["aes_standalone"]["code_sha256"] = fp
    folder = ROOT / "notebooks/multitrait_reset"
    folder.mkdir(exist_ok=True)
    text = json.dumps(n, ensure_ascii=False, indent=1) + "\n"
    (folder / NAME).write_text(text)
    organized = ROOT / "eksperimen/03_tiga_input_multi_trait/reset/notebooks"
    organized.mkdir(parents=True, exist_ok=True)
    (organized / NAME).write_text(text)


if __name__ == "__main__":
    build()
