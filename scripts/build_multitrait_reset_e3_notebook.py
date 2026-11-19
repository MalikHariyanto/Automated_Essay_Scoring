"""Build reset E3 from E2: trait-specific answer-token attention pooling."""

import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "notebooks/multitrait_reset/AES_Reset_E2_S3_MultiTrait_TraitHeads.ipynb"
NAME = "AES_Reset_E3_S3_MultiTrait_TraitAttention.ipynb"
PLACEHOLDER = "__RESET_E3_CODE_SHA256__"

INTRO = """# E3 reset · S3 multi-trait · Native Learnable Code

E3 melanjutkan **E2 reset**, memakai dataset semi-final dan split yang sama.
Pilih runtime T4 GPU, lalu jalankan dari atas dan hubungkan Google Drive.
Cell **Siapkan dataset dan split** membuat cfg, bundle, manifest, dan output sekaligus.
Tahap model menampilkan status menunggu GPU jika CUDA belum tersedia.

## Perubahan yang diuji

E2 memberikan vektor CLS yang sama kepada empat MLP khusus trait.
E3 mengganti CLS dengan **trait-specific attention pooling** pada token jawaban
dari layer terakhir RoBERTa. Empat query belajar menghasilkan empat ringkasan
768 dimensi: isi, organisasi, kebahasaan, dan mekanik.
Soal dan referensi tetap masuk RoBERTa sebagai konteks. Pooling mengecualikan
token konteks, prefix, special token, dan padding.

Untuk tiap trait: softmax(query · token / √768) → jumlah token berbobot →
mean antarwindow → LayerNorm → concat dengan proyeksi BGE256 → MLP1024→64→1.
Keempat query menambah **3.072 parameter**; LR-nya1e-4 melalui kelompok pooling/norm.
Query diinisialisasi normal std0,02. Tidak ada dropout attention tambahan.
Bobot token dipelajari melalui MSE terhadap empat skor guru pada data latih;
nama trait tidak menjamin perhatian model langsung sesuai rubrik sejak awal.

## Resep E2 yang dipertahankan

RoBERTa fine-tuned LR5e-6; BGE-M3 frozen; proyeksi/head/norm LR1e-4;
empat MLP1024→64→1/dropout0,2; AdamW/weight decay0,01; warmup10%; linear scheduler;
clip1; microbatch1/batch efektif8. Lima fold development, maksimal25epoch/patience5,
satu refit sesuai median epoch terbaik dan satu evaluasi test.
Target MSE dinormalisasi per trait ke0–1. Seleksi checkpoint memakai RMSE validasi
pada skala skor asli. RMSE, Pearson, QWK, dan Gwet AC2 dilaporkan per trait.
Inisialisasi parameter bersama dan RNG mengikuti E2; folder/checkpoint E3 terpisah.

## Dasar metode

Representasi khusus trait mengadaptasi ide
[CTS, Ridley dkk., AAAI2021](https://github.com/robert1ridley/cross-prompt-trait-scoring/blob/HEAD/models/CTS.py).
CTS memakai LSTM/attention pada tingkat kalimat dan relasi antartait.
[Dong, Zhang, dan Yang2017](https://aclanthology.org/K17-1017/) memakai attention
pada model AES hierarkis. E3 adalah adaptasi token attention untuk backbone
RoBERTa+BGE proyek ini. Rumus query, dimensi, dan inisialisasi di atas adalah
pilihan eksperimen; bukan klaim konfigurasi terbaik dari paper tersebut.

## Satu workbook hasil

Hasil_AES_Reset_E3_S3_MultiTrait.xlsx mencakup data, token/window, vektor awal,
fitur final per trait, riwayat CV/refit, prediksi OOF/test, dan evaluasi empat trait.
Tambahan: query attention final serta bobot token empat trait pada **satu contoh
development**. Bobot tersebut untuk inspeksi checkpoint refit, bukan bukti sebab-akibat
atau prediksi OOF. Empat set fitur final membuat Excel lebih besar daripada E2.
Dataset dan workbook keluaran tetap privat di Drive. Semua kode tersedia di notebook;
setiap tahap mempunyai heading dan output sendiri.
"""


def build():
    raw = BASE.read_bytes()
    notebook = json.loads(raw)
    base_hash = notebook["metadata"]["aes_standalone"]["code_sha256"]
    cells = notebook["cells"]
    changed_engine = False
    for index, cell in enumerate(cells):
        source = "".join(cell["source"]).replace(base_hash, PLACEHOLDER)
        role = cell["metadata"].get("aes_multitrait_role")
        source = (
            source.replace("Reset_E2", "Reset_E3")
            .replace("reset_e2_s3_multitrait", "reset_e3_s3_multitrait")
            .replace("E2 reset", "E3 reset")
        )
        if role == "definition:SingleScoreModel":
            source = (ROOT / "scripts/reset_e3_trait_attention.py").read_text()
        elif role == "prepare":
            source = source.replace(
                "'trait_hidden': 64,",
                "'trait_hidden': 64, 'pooling': 'trait_answer_attention_v1', "
                "'attention_query_std': 0.02, 'attention_score_scale': 'sqrt_768', "
                "'attention_dropout': 0.0, 'window_pooling': 'mean',",
            )
        elif role == "definition:Engine":
            anchor = "        return self.model(windows=windows, mapping=mapping, n_essays=len(ids),\n                          bge=features, return_features=return_features)"
            if anchor not in source:
                raise ValueError(
                    "E2 Engine forward changed; inspect before rebuilding."
                )
            source = source.replace(
                anchor,
                """        mask = answer_token_mask(
            windows, mapping, details, getattr(self.tokenizer, 'padding_side', 'right')
        )
        result = self.model(
            windows=windows, mapping=mapping, n_essays=len(ids), bge=features,
            answer_mask=mask, return_features=return_features,
        )
        if return_features:
            prediction, inspected = result
            inspected.update(input_ids=windows['input_ids'], answer_mask=mask)
            return prediction, inspected
        return result""",
            )
            source = source.replace("Engine E43", "Engine E3 reset")
            changed_engine = True
        elif role == "architecture":
            source = """display(pd.DataFrame([
    ['RoBERTa layer terakhir', 'token × 768'],
    ['Attention jawaban per trait', '4 × 768; 3072 parameter query'],
    ['BGE frozen', '1024'],
    ['Proyeksi BGE bersama', '256'],
    ['Fusion concat per trait', '4 × 1024'],
    ['MLP per trait', '1024 → 64 → 1'],
], columns=['Bagian', 'Dimensi']))
"""
        elif role == "final_features":
            body = (ROOT / "scripts/reset_e3_features.py").read_text()
            source = (
                "if model_stage_ready('Inspeksi fitur final', "
                "('bge', 'refit_result', 'inspection_sheets', 'sample', 'journal')):\n"
                + "\n".join("    " + line for line in body.splitlines())
                + "\n"
            )
        elif role == "token_preview":
            source = source.replace(
                "posisi yang dipakai E43", "posisi jawaban untuk attention E3"
            )
        if cell["cell_type"] == "markdown":
            if source.startswith("### Model fusion multi-trait"):
                next_role = cells[index + 1]["metadata"].get("aes_multitrait_role")
                if next_role == "refit_execute":
                    source = "### Jalankan refit E3 reset\n\nLatih pada seluruh development selama epoch hasil CV, lalu simpan checkpoint final."
                elif next_role == "definition:SingleScoreModel":
                    source = """### Learnable Code · mask, attention per trait, dan head

Mask memilih posisi token jawaban. Empat query belajar memberi bobot token
yang jumlahnya1 per window dan trait. Masing-masing ringkasan RoBERTa768
digabungkan dengan proyeksi BGE256 dan masuk ke MLP trait yang bersangkutan.
"""
                else:
                    source = "## 12 · Learnable Code — model fusion dengan attention per trait"
            elif "Tandai posisi token referensi untuk co-attention" in source:
                source = source.replace(
                    "Tandai posisi token referensi untuk co-attention.",
                    "Tandai posisi referensi untuk tabel inspeksi token.",
                )
        cell["source"] = source
        cell["id"] = f"reset-e3-{index}"
    if not changed_engine:
        raise ValueError("Engine definition missing.")
    cells[0]["source"] = INTRO
    fingerprint = hashlib.sha256(
        "\n".join(c["source"] for c in cells if c["cell_type"] == "code").encode()
    ).hexdigest()
    for cell in cells:
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
        "experiment": "Reset_E3",
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
