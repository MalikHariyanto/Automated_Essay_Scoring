payload = torch.load(output / "final.pt", map_location="cpu", weights_only=True)
model = make_model("S3", cfg)
model.load_state_dict(payload["state"], strict=True)
del payload
device = device_name(cfg)
model.to(device).eval()
engine = Engine(
    model, bundle.texts, bge, cfg, device, contexts=contexts_from_bundle(bundle)
)
parts = {name: [] for name in ["roberta", "projection", "concat"]}
attention_rows = []
sample_index = manifest["development"][0]
with torch.no_grad():
    for i in range(len(bundle.rows)):
        _, features = engine.forward([i], return_features=True)
        for name in parts:
            parts[name].append(features[name].float().cpu().numpy())
        # Inspect one development example rather than exporting every token again.
        if i == sample_index:
            ids = features["input_ids"].cpu().tolist()
            mask = features["answer_mask"].cpu().tolist()
            weights = features["attention"].float().cpu().numpy()
            for window, (token_ids, valid) in enumerate(zip(ids, mask)):
                tokens = engine.tokenizer.convert_ids_to_tokens(token_ids)
                for position, is_answer in enumerate(valid):
                    if is_answer:
                        attention_rows.append(
                            {
                                "id": bundle.rows[i]["row_id"],
                                "cakupan": "contoh development, checkpoint refit final",
                                "window": window + 1,
                                "posisi_dalam_window": position,
                                "token_id": token_ids[position],
                                "token": tokens[position],
                                **{
                                    "bobot_"
                                    + trait: float(weights[window, k, position])
                                    for k, (trait, *_) in enumerate(TRAITS)
                                },
                            }
                        )
for name, values in parts.items():
    matrix = np.concatenate(values)
    row_ids = [r["row_id"] for r in bundle.rows]
    if matrix.ndim == 3:
        for index, (trait, *_) in enumerate(TRAITS):
            inspection_sheets["Final_" + name + "_" + trait] = vector_frame(
                matrix[:, index, :], row_ids
            )
    else:
        inspection_sheets["Final_" + name] = vector_frame(matrix, row_ids)
inspection_sheets["Attention contoh final"] = pd.DataFrame(attention_rows)
inspection_sheets["Query attention final"] = vector_frame(
    model.trait_pool.queries.detach().float().cpu().numpy(),
    [trait for trait, *_ in TRAITS],
)
del engine, model, features, parts, matrix
release()
print(
    "Fitur per trait dan bobot attention final untuk inspeksi. "
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
        {"bagian": "Attention jawaban per trait", "dimensi": "4 × 768"},
        {"bagian": "Query yang belajar", "dimensi": "4 × 768 = 3072 parameter"},
        {"bagian": "BGE frozen", "dimensi": 1024},
        {"bagian": "Proyeksi BGE bersama", "dimensi": 256},
        {"bagian": "Concat per trait", "dimensi": "4 × 1024"},
        {"bagian": "MLP per trait", "dimensi": "1024 → 64 → 1"},
    ]
)
