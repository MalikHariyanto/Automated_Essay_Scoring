class TraitGatedHeads(IndependentTraitHeads):
    """Learn a RoBERTa/BGE mix for each trait from the original concat."""

    def __init__(self, dropout):
        super().__init__(dropout)
        self.gates = nn.ModuleDict({name: nn.Linear(1024, 2) for name, *_ in TRAITS})
        for gate in self.gates.values():
            nn.init.zeros_(gate.weight)
            nn.init.zeros_(gate.bias)

    def forward(self, fusion, return_features=False):
        roberta, bge = fusion.split((768, 256), dim=1)
        predictions, alphas = [], []
        for name, *_ in TRAITS:
            alpha = self.gates[name](fusion).float().softmax(dim=1)
            scales = (2 * alpha).to(fusion.dtype)
            gated = torch.cat([roberta * scales[:, :1], bge * scales[:, 1:]], dim=1)
            predictions.append(self.heads[name](gated))
            alphas.append(alpha)
        prediction = torch.cat(predictions, dim=1)
        return (prediction, torch.stack(alphas, dim=1)) if return_features else prediction


def gate_rows(bundle, indices, alphas, **keys):
    """One row per answer, with the four pairs used for its prediction."""
    alphas = np.asarray(alphas, dtype=np.float64)
    if (alphas.shape != (len(indices), len(TRAITS), 2)
        or not np.isfinite(alphas).all()
        or np.any((alphas < 0) | (alphas > 1))
        or not np.allclose(alphas.sum(axis=2), 1.0, atol=1e-6, rtol=0)):
        raise ValueError('Bobot gate tidak sesuai dengan jawaban dan trait.')
    return [
        {**keys, 'id_jawaban': bundle.rows[i]['row_id'],
         **{f'{name}_{encoder}': float(alphas[row, trait, col])
            for trait, (name, *_) in enumerate(TRAITS)
            for col, encoder in enumerate(('RoBERTa', 'BGE'))}}
        for row, i in enumerate(indices)
    ]


def gate_summary(frame, stage):
    rows = []
    for name, *_ in TRAITS:
        for encoder in ('RoBERTa', 'BGE'):
            values = frame[f'{name}_{encoder}']
            rows.append({'bagian': stage, 'trait': name, 'encoder': encoder, 'n': len(frame),
                         'rerata': values.mean(), 'sd': values.std(ddof=0),
                         'minimum': values.min(), 'maksimum': values.max()})
    return rows
