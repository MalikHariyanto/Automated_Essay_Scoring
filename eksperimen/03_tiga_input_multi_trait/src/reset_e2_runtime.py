"""Native notebook runtime checks for data preparation and model execution."""


def runtime_information():
    available = torch.cuda.is_available()
    return {
        "torch": str(torch.__version__),
        "transformers": version("transformers"),
        "scikit-learn": version("scikit-learn"),
        "device": "cuda" if available else "cpu",
        "accelerator": torch.cuda.get_device_name(0) if available else None,
    }


def model_stage_ready(stage, required=()):
    """Report unavailable prerequisites before a model stage is executed."""
    missing_data = [
        name
        for name in ("cfg", "bundle", "manifest", "output")
        if name not in globals()
    ]
    if missing_data:
        print(
            f"{stage} belum dijalankan. Jalankan cell Siapkan dataset dan split "
            f'terlebih dahulu. Variabel yang diperlukan: {", ".join(missing_data)}.'
        )
        return False
    if cfg["device"] != "cpu" and not torch.cuda.is_available():
        print(
            f"{stage} belum dijalankan karena GPU CUDA belum tersedia. "
            "Dataset dan split sudah dapat diperiksa. Pilih Runtime → Change runtime type → T4 GPU, "
            "lalu Run all pada runtime GPU."
        )
        return False
    missing_stage = [name for name in required if name not in globals()]
    if missing_stage:
        print(
            f"{stage} belum dijalankan. Selesaikan tahap model sebelumnya. "
            f'Variabel yang diperlukan: {", ".join(missing_stage)}.'
        )
        return False
    return True


runtime_status = runtime_information()
display(pd.DataFrame([runtime_status]))
if not torch.cuda.is_available():
    print(
        "Runtime CPU terdeteksi. Pembacaan workbook dan pembagian data tetap berjalan. "
        'Tahap model menunggu GPU. Untuk uji CPU yang disengaja, set DEVICE = "cpu" '
        "di hyperparameter lalu jalankan ulang konfigurasi dari atas."
    )
