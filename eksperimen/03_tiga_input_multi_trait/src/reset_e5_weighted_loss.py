def training_loss_weights(labels, max_ratio):
    """Buat bobot per kategori dari label training pada satu fit."""
    if labels.ndim != 2 or labels.shape[1] != len(TRAITS) or not len(labels):
        raise ValueError("Label training harus berbentuk [jumlah jawaban, 4].")
    if not math.isfinite(max_ratio) or max_ratio < 1:
        raise ValueError("Batas rasio bobot harus berhingga dan minimal 1.")

    weights = np.empty(labels.shape, dtype=np.float32)
    audit = []
    for column, (name, _, low, high) in enumerate(TRAITS):
        categories = np.arange(low, high + 1)
        scores = labels[:, column]
        if not np.isin(scores, categories).all():
            raise ValueError(f"Label training di luar kategori rubrik: {name}.")
        indices = scores.astype(np.int64) - low
        counts = np.bincount(indices, minlength=len(categories))
        present = counts > 0
        levels = np.zeros(len(categories), dtype=np.float64)
        levels[present] = np.minimum(np.sqrt(counts.max() / counts[present]), max_ratio)
        levels /= np.average(levels, weights=counts)
        weights[:, column] = levels[indices]
        audit.extend(
            {
                'trait': name, 'skor': int(score), 'n_training': int(count),
                'bobot_loss': float(weight) if count else None,
            }
            for score, count, weight in zip(categories, counts, levels)
        )
    return weights, audit
