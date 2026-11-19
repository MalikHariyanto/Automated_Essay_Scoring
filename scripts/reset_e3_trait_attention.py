def answer_token_mask(windows, mapping, details, padding_side="right"):
    """Select answer tokens; exclude context, prefixes, special tokens and padding."""
    valid = windows["attention_mask"].bool()
    mask = torch.zeros_like(valid)
    flat = [
        (essay, positions)
        for essay, info in enumerate(details)
        for positions in info["answer_positions"]
    ]
    if len(flat) != len(valid) or padding_side not in ("left", "right"):
        raise ValueError("Pemetaan window/posisi jawaban tidak cocok.")
    for window, (essay, positions) in enumerate(flat):
        if int(mapping[window]) != essay or not positions:
            raise ValueError("Window harus berisi token jawaban dari esai yang benar.")
        offset = (
            valid.shape[1] - int(valid[window].sum()) if padding_side == "left" else 0
        )
        indices = (
            torch.tensor(positions, dtype=torch.long, device=valid.device) + offset
        )
        if int(indices.min()) < 0 or int(indices.max()) >= valid.shape[1]:
            raise ValueError("Posisi token jawaban keluar dari window.")
        if not bool(valid[window, indices].all()):
            raise ValueError("Token padding tidak boleh menjadi token jawaban.")
        mask[window, indices] = True
    return mask


class TraitAttentionPool(nn.Module):
    """Four learned queries summarize valid answer tokens independently."""

    def __init__(self):
        super().__init__()
        self.queries = nn.Parameter(torch.empty(len(TRAITS), 768))
        nn.init.normal_(self.queries, mean=0.0, std=0.02)

    def forward(self, hidden, answer_mask):
        if answer_mask.shape != hidden.shape[:2]:
            raise ValueError("Mask jawaban harus cocok dengan window RoBERTa.")
        if not bool(answer_mask.any(dim=1).all()):
            raise ValueError("Setiap window harus mempunyai token jawaban.")
        # Float32 softmax and pooling keep the mask stable under mixed precision.
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            values = hidden.float()
            scores = torch.einsum("wtd,kd->wkt", values, self.queries.float())
            scores = scores / math.sqrt(768)
            scores = scores.masked_fill(~answer_mask[:, None, :], float("-inf"))
            weights = torch.softmax(scores, dim=-1)
            pooled = torch.einsum("wkt,wtd->wkd", weights, values)
        return pooled, weights


class IndependentTraitHeads(nn.Module):
    """The E2 scorers now each read their corresponding trait representation."""

    def __init__(self, dropout):
        super().__init__()
        self.heads = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.Linear(1024, 64),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(64, 1),
                )
                for name, *_ in TRAITS
            }
        )

    def forward(self, fusion):
        return torch.cat(
            [
                self.heads[name](fusion[:, index, :])
                for index, (name, *_) in enumerate(TRAITS)
            ],
            dim=1,
        )


class MultiTraitModel(nn.Module):
    """E3 reset: trait-specific answer pooling plus the E2 independent heads."""

    def __init__(self, scenario, roberta, s3_mlp_dropout=0.2):
        super().__init__()
        self.scenario, self.roberta = scenario, roberta
        self.roberta.requires_grad_(True)
        self.roberta_norm = nn.LayerNorm(768)
        self.projection = nn.Sequential(nn.Linear(1024, 256), nn.LayerNorm(256))
        # Reproduce E2's initialization path before replacing the temporary E1 head.
        self.head = nn.Sequential(
            nn.Linear(1024, 256),
            nn.GELU(),
            nn.Dropout(s3_mlp_dropout),
            nn.Linear(256, 4),
        )
        with torch.random.fork_rng(devices=[]):
            self.head = IndependentTraitHeads(s3_mlp_dropout)
            self.trait_pool = TraitAttentionPool()

    def forward(
        self, windows, mapping, n_essays, bge, answer_mask, return_features=False
    ):
        hidden = self.roberta(**windows).last_hidden_state
        pooled_windows, attention = self.trait_pool(hidden, answer_mask)
        # Keep E2's equal averaging across windows, separately for each trait.
        pooled = mean_by_essay(
            pooled_windows.flatten(start_dim=1), mapping, n_essays
        ).reshape(n_essays, len(TRAITS), 768)
        projected = self.projection(bge)
        fusion = torch.cat(
            [
                self.roberta_norm(pooled),
                projected[:, None, :].expand(-1, len(TRAITS), -1),
            ],
            dim=-1,
        )
        prediction = self.head(fusion)
        if not return_features:
            return prediction
        return prediction, {
            "roberta": pooled,
            "bge": bge,
            "projection": projected,
            "concat": fusion,
            "attention": attention,
        }
