inspection_sheets = {}


class WorkbookCollector:
    def save(self, name, sheets, *args, **kwargs):
        for key, frame in sheets.items():
            inspection_sheets[name[:8] + "_" + key] = frame


journal = WorkbookCollector()
inspection_sheets["Data dan label"] = pd.DataFrame(
    [
        {
            "id": row["row_id"],
            "soal": row["question"],
            "referensi": row["reference"],
            "jawaban_asli": row["raw_text"],
            "jawaban_bersih": row["text"],
            **dict(zip(TARGET_COLUMNS, bundle.y[i].tolist())),
        }
        for i, row in enumerate(bundle.rows)
    ]
)
inspection_sheets["Split"] = pd.DataFrame(
    [
        {
            "id": bundle.rows[i]["row_id"],
            "bagian": stage,
            "kelompok": int(bundle.groups[i]),
        }
        for stage in ["development", "test"]
        for i in manifest[stage]
    ]
)
export_tokens(bundle, cfg, journal)
values, _, _, _ = paired_pretrained_values(bundle, cfg, "roberta")
inspection_sheets["RoBERTa pretrained"] = vector_frame(
    values, [r["row_id"] for r in bundle.rows]
)
inspection_sheets["BGE frozen"] = vector_frame(
    bge.numpy(), [r["row_id"] for r in bundle.rows]
)
sample = bundle.rows[manifest["development"][0]]
lab = vector_lab(
    sample["text"], cfg, output, context=(sample["question"], sample["reference"])
)
export_word_sentence_vectors(lab, journal, "Kata")
display(pd.DataFrame({"sheet": list(inspection_sheets)}))

inspection_sheets["Anggota fold"] = pd.DataFrame(
    [
        {"fold": fold["fold"], "bagian": stage, "id": bundle.rows[i]["row_id"]}
        for fold in manifest["folds"]
        for stage in ("train", "validation")
        for i in fold[stage]
    ]
)
