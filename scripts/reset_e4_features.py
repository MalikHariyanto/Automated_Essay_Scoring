payload = torch.load(output / "final.pt", map_location="cpu", weights_only=True)
model = make_model("S3", cfg)
model.load_state_dict(payload["state"], strict=True)
del payload
device = device_name(cfg)
model.to(device).eval()
engine = Engine(
    model, bundle.texts, bge, cfg, device, contexts=contexts_from_bundle(bundle)
)
parts = {
    name: []
    for name in (
        "roberta",
        "projection",
        "concat",
        "trait_hidden",
        "trait_context",
        "trait_concat",
    )
}
attention_rows = []
test_indices = set(manifest["test"])
with torch.no_grad():
    for i in range(len(bundle.rows)):
        _, features = engine.forward([i], return_features=True)
        for name in parts:
            parts[name].append(features[name].float().cpu().numpy())
        weights = features["inter_trait_attention"][0].float().cpu().numpy()
        attention_rows.append(
            {
                "id": bundle.rows[i]["row_id"],
                "bagian_data": "test" if i in test_indices else "development",
                "cakupan": "inspeksi checkpoint refit final, bukan OOF",
                **{
                    f"{target}_membaca_{source}": float(weights[t, s])
                    for t, (target, *_) in enumerate(TRAITS)
                    for s, (source, *_) in enumerate(TRAITS)
                },
            }
        )
row_ids = [r["row_id"] for r in bundle.rows]
for name, values in parts.items():
    matrix = np.concatenate(values)
    if matrix.ndim == 3:
        for index, (trait, *_) in enumerate(TRAITS):
            inspection_sheets["Final_" + name + "_" + trait] = vector_frame(
                matrix[:, index, :], row_ids
            )
    else:
        inspection_sheets["Final_" + name] = vector_frame(matrix, row_ids)
inspection_sheets["Attention final antartrait"] = pd.DataFrame(attention_rows)
inspection_sheets["Bobot regresor konteks"] = pd.DataFrame(
    [
        {
            "trait": trait,
            "dimensi_hidden_sendiri": 64,
            "dimensi_konteks_trait_lain": 64,
            "norma_bobot_hidden_sendiri": float(head[3].weight[:, :64].detach().norm()),
            "norma_bobot_konteks_akhir": float(head[3].weight[:, 64:].detach().norm()),
            "norma_bobot_konteks_awal": 0.0,
        }
        for trait, head in model.head.heads.items()
    ]
)
del engine, model, features, parts, matrix
release()
print(
    "Fitur hidden/konteks per trait dan bobot attention final untuk inspeksi. "
    "Metrik penelitian tetap memakai prediksi OOF dan test."
)

lab_final = vector_lab(
    sample["text"],
    {**cfg, "vector_source": "S3"},
    output,
    context=(sample["question"], sample["reference"]),
)
export_word_sentence_vectors(lab_final, journal, "KataFinal")
inspection_sheets["Konfigurasi"] = pd.DataFrame(
    [{"parameter": key, "nilai": value} for key, value in cfg.items()]
)
inspection_sheets["Arsitektur"] = pd.DataFrame(
    [
        {"bagian": "RoBERTa CLS terakhir", "dimensi": 768},
        {"bagian": "BGE frozen", "dimensi": 1024},
        {"bagian": "Proyeksi BGE", "dimensi": 256},
        {"bagian": "Concat bersama", "dimensi": 1024},
        {"bagian": "Hidden per trait", "dimensi": "4 × 64"},
        {"bagian": "Attention antartrait", "dimensi": "4 × 4, diagonal 0"},
        {"bagian": "Konteks tiga trait lain", "dimensi": "4 × 64"},
        {"bagian": "Hidden + konteks per trait", "dimensi": "4 × 128"},
        {"bagian": "Regresor per trait", "dimensi": "128 → 1"},
    ]
)
