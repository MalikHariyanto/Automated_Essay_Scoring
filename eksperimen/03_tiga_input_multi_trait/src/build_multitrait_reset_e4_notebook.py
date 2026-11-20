"""Build reset E4 from E2: attention among four trait-specific hidden vectors."""

import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "notebooks/multitrait_reset/AES_Reset_E2_S3_MultiTrait_TraitHeads.ipynb"
NAME = "AES_Reset_E4_S3_MultiTrait_InterTraitAttention.ipynb"
PLACEHOLDER = "__RESET_E4_CODE_SHA256__"

INTRO = """# E4 reset · S3 multi-trait · Native Learnable Code

E4 berangkat langsung dari **E2 reset** dan dapat dijalankan paralel dengan E3
pada runtime GPU Colab terpisah. Gunakan T4, jalankan dari atas, dan hubungkan
Drive ke dataset semi-final yang sama. Cell **Siapkan dataset dan split** membuat
cfg, bundle, manifest, dan output sekaligus. Runtime tanpa CUDA tetap dapat
membaca dataset dan split; tahap model menampilkan status menunggu GPU.

## Perubahan yang diuji: inter-trait attention

RoBERTa memakai CLS layer terakhir dan mean antarwindow seperti E2.
BGE-M3 tetap frozen, diproyeksikan1024→256, lalu concat768+256 menjadi1024.
Empat jalur Linear1024→64→GELU→Dropout0,2 menghasilkan hidden khusus isi,
organisasi, kebahasaan, dan mekanik.

Untuk setiap jawaban, tiap trait membaca **tiga trait lainnya**:
skor(h_target,h_sumber)=dot(h_target,h_sumber)/√64 → mask diagonal → softmax
→ jumlah hidden sumber berbobot. Attention menghasilkan konteks64 untuk tiap trait.
Hidden sendiri64 + konteks64 digabung menjadi128, lalu Linear128→1 memberi skor.
Attention berukuran4×4 **per jawaban**, tidak mencampur siswa dalam batch.
Tidak ada proyeksi Q/K/V atau dropout attention tambahan.

Head E4 berisi262.916 parameter, bertambah256 dari E2.
Bobot64 untuk hidden sendiri dan bias disalin dari inisialisasi E2; bobot64
untuk konteks trait lain dimulai dari0 dan ikut belajar dengan LR1e-4.
Prediksi awal mengikuti E2, lalu training dapat memanfaatkan jalur konteks.
Inisialisasi parameter bersama dan keadaan RNG juga mengikuti E2.
Skor guru hanya menjadi target loss; nilai trait asli tidak masuk attention/inferensi.

## Resep training E2 dipertahankan

RoBERTa fine-tuned LR5e-6; BGE frozen; proyeksi/head/norm LR1e-4;
AdamW/weight decay0,01; warmup10%; scheduler linear; clip1;
microbatch1/batch efektif8; maksimal25epoch/patience5.
Dataset semi-final dan split kelompok siswa tetap sama, lima fold development,
satu refit selama ceil median best epoch, lalu satu evaluasi test.
Loss MSE memakai empat target normalisasi0–1. Pemilihan epoch memakai RMSE
validasi pada skala asli. RMSE, Pearson, QWK, dan Gwet AC2 dilaporkan per trait.
Folder hasil/checkpoint E4 terpisah dari E2 dan E3.

## Rujukan dan batas adaptasi

[Ridley dkk., Automated Cross-prompt Scoring of Essay Traits, AAAI2021](https://github.com/robert1ridley/cross-prompt-trait-scoring)
memakai attention dari representasi satu trait ke trait lainnya, menggabungkan
representasi sendiri dengan konteks attention, lalu memprediksi skor.
[Kode CTS](https://github.com/robert1ridley/cross-prompt-trait-scoring/blob/HEAD/models/CTS.py)
menjadi dasar ide. E4 mengadaptasinya pada hidden64 fusion RoBERTa+BGE,
dengan pembagi√64 dan inisialisasi jalur konteks0. Rancangan ini bukan reproduksi
CTS lengkap atau klaim hyperparameter terbaik dari paper. Efektivitas E4 belum diketahui.

## Satu file Excel

Hasil_AES_Reset_E4_S3_MultiTrait.xlsx memuat cakupan keluaran E2:
data/label, token/window, vektor awal dan final, inspeksi kata/kalimat,
riwayat training, prediksi OOF/test, serta evaluasi empat trait.
Tambahan: hidden64, konteks64, concat128 per trait, matriks attention antartrait
untuk seluruh jawaban pada checkpoint refit final, dan norma bobot regresor konteks
yang dapat dibandingkan dengan nilai awal0. Inspeksi final tersebut bukan OOF,
bukan bukti sebab-akibat, dan tidak menggantikan metrik penelitian.
Workbook dapat lebih besar daripada E2. Dataset dan hasil tetap privat;
seluruh kode tersedia di notebook dengan heading dan output per tahap.
"""


def build():
    raw = BASE.read_bytes()
    notebook = json.loads(raw)
    base_hash = notebook["metadata"]["aes_standalone"]["code_sha256"]
    cells = notebook["cells"]
    for index, cell in enumerate(cells):
        source = "".join(cell["source"]).replace(base_hash, PLACEHOLDER)
        role = cell["metadata"].get("aes_multitrait_role")
        source = (
            source.replace("Reset_E2", "Reset_E4")
            .replace("reset_e2_s3_multitrait", "reset_e4_s3_multitrait")
            .replace("E2 reset", "E4 reset")
        )
        if role == "definition:SingleScoreModel":
            source = (ROOT / "scripts/reset_e4_intertrait_model.py").read_text()
        elif role == "prepare":
            anchor = "'head_type': 'independent_trait_heads', 'trait_hidden': 64,"
            if anchor not in source:
                raise ValueError("E2 configuration changed; inspect before rebuilding.")
            source = source.replace(
                anchor,
                "'head_type': 'inter_trait_attention_heads', 'trait_hidden': 64, "
                "'trait_output_input': 128, 'inter_trait_attention': {"
                "'type': 'scaled_dot_product', 'exclude_self': True, 'projection': False, "
                "'scale': 'sqrt_64', 'dropout': 0.0, 'context_output_init': 'zero', "
                "'extra_parameters': 256},",
            )
        elif role == "hyperparameters":
            # Remove obsolete single-trait comments; keep all assignment values.
            source = "\n".join(
                line.split("#", 1)[0].rstrip() for line in source.splitlines()
            )
            source = "# Resep training E4 reset mengikuti E2 reset.\n" + source.lstrip()
            source = source.replace(
                "DROPOUT = 0.1",
                "DROPOUT = 0.1  # Konfigurasi legacy; model reset memakai S3_MLP_DROPOUT.",
            )
            source = source.replace(
                "S3_MLP_DROPOUT = 0.2",
                "S3_MLP_DROPOUT = 0.2  # Dropout aktif pada empat jalur hidden64.",
            )
        elif role == "architecture":
            source = """display(pd.DataFrame([
    ['RoBERTa CLS layer terakhir', '768'],
    ['BGE frozen', '1024'],
    ['Proyeksi BGE', '256'],
    ['Fusion concat bersama', '1024'],
    ['Hidden per trait', '4 × 64'],
    ['Attention antartrait per jawaban', '4 × 4; diagonal 0'],
    ['Konteks tiga trait lain', '4 × 64'],
    ['Hidden sendiri + konteks per trait', '4 × 128'],
    ['Regresor per trait', '128 → 1'],
    ['Parameter head', '262916; +256 dari E2'],
], columns=['Bagian', 'Dimensi']))
"""
        elif role == "final_features":
            body = (ROOT / "scripts/reset_e4_features.py").read_text()
            source = (
                "if model_stage_ready('Inspeksi fitur final', "
                "('bge', 'refit_result', 'inspection_sheets', 'sample', 'journal')):\n"
                + "\n".join("    " + line for line in body.splitlines())
                + "\n"
            )
        elif role == "token_preview":
            source = source.replace("posisi yang dipakai E43", "posisi input E4 reset")
        if cell["cell_type"] == "markdown":
            if source.startswith("### Model fusion multi-trait"):
                next_role = cells[index + 1]["metadata"].get("aes_multitrait_role")
                if next_role == "refit_execute":
                    source = "### Jalankan refit E4 reset\n\nLatih pada seluruh development sesuai epoch hasil CV, lalu simpan checkpoint final."
                elif next_role == "definition:SingleScoreModel":
                    source = """### Learnable Code · empat hidden, attention antartrait, dan regresor

Setiap trait membuat hidden64 dari concat yang sama. Attention membaca tiga
hidden trait lain pada jawaban yang sama. Gabungkan hidden sendiri dan konteks
menjadi128, lalu prediksi satu skor per trait. Label guru hanya masuk loss.
"""
                else:
                    source = (
                        "## 12 · Learnable Code — fusion dengan inter-trait attention"
                    )
            elif "Tandai posisi token referensi untuk co-attention" in source:
                source = source.replace(
                    "Tandai posisi token referensi untuk co-attention.",
                    "Tandai posisi referensi untuk tabel inspeksi token.",
                )
        cell["source"] = source
        cell["id"] = f"reset-e4-{index}"
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
        "experiment": "Reset_E4",
    }
    serialized = json.dumps(notebook, ensure_ascii=False, indent=1) + "\n"
    for folder in (
        ROOT / "notebooks/multitrait_reset",
        ROOT / "eksperimen/03_tiga_input_multi_trait/notebooks",
    ):
        folder.mkdir(parents=True, exist_ok=True)
        (folder / NAME).write_text(serialized)
    print(f"{NAME} · {fingerprint}")


if __name__ == "__main__":
    build()
