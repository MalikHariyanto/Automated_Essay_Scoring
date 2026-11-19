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
with torch.no_grad():
    for i in range(len(bundle.rows)):
        _, features = engine.forward([i], return_features=True)
        for name in parts:
            parts[name].append(features[name].float().cpu().numpy())
for name, values in parts.items():
    inspection_sheets["Final_" + name] = vector_frame(
        np.concatenate(values), [r["row_id"] for r in bundle.rows]
    )
del engine, model
release()
print(
    "Fitur model final untuk inspeksi seluruh jawaban. Metrik penelitian tetap memakai OOF dan test."
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
        {"bagian": "Concat", "dimensi": 1024},
        {"bagian": "MLP hidden", "dimensi": 256},
        {"bagian": "Trait", "dimensi": 4},
    ]
)
