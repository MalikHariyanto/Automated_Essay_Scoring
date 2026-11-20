class InterTraitHeads(nn.Module):
    """E2 hidden64 branches, followed by attention to the other three traits."""

    def __init__(self, dropout):
        super().__init__()
        # Construct all E2 heads first, preserving their initialization order.
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
        for head in self.heads.values():
            original = head[3]
            expanded = nn.Linear(128, 1)
            # Start from E2 predictions. The new context weights can then learn.
            with torch.no_grad():
                expanded.weight[:, :64].copy_(original.weight)
                expanded.weight[:, 64:].zero_()
                expanded.bias.copy_(original.bias)
            head[3] = expanded
        self.register_buffer(
            "self_mask", torch.eye(len(TRAITS), dtype=torch.bool), persistent=False
        )

    def forward(self, fusion, return_features=False):
        # Dropout stays at the E2 location, after GELU in each hidden branch.
        hidden = torch.stack(
            [head[2](head[1](head[0](fusion))) for head in self.heads.values()],
            dim=1,
        )
        # Each answer has its own 4×4 matrix. Never mix students in attention.
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            values = hidden.float()
            scores = torch.matmul(values, values.transpose(1, 2)) / math.sqrt(64)
            scores = scores.masked_fill(self.self_mask[None, :, :], float("-inf"))
            attention = torch.softmax(scores, dim=-1)
            context = torch.matmul(attention, values)
        combined = torch.cat([hidden, context], dim=-1)
        prediction = torch.cat(
            [
                head[3](combined[:, index, :])
                for index, head in enumerate(self.heads.values())
            ],
            dim=1,
        )
        if not return_features:
            return prediction
        return prediction, {
            "trait_hidden": hidden,
            "trait_context": context,
            "trait_concat": combined,
            "inter_trait_attention": attention,
        }


class MultiTraitModel(nn.Module):
    """E4 reset: the E2 CLS/BGE backbone with inter-trait attention heads."""

    def __init__(self, scenario, roberta, s3_mlp_dropout=0.2):
        super().__init__()
        self.scenario, self.roberta = scenario, roberta
        self.roberta.requires_grad_(True)
        self.roberta_norm = nn.LayerNorm(768)
        self.projection = nn.Sequential(nn.Linear(1024, 256), nn.LayerNorm(256))
        # Preserve E2's initialization path, including its temporary E1 head.
        self.head = nn.Sequential(
            nn.Linear(1024, 256),
            nn.GELU(),
            nn.Dropout(s3_mlp_dropout),
            nn.Linear(256, 4),
        )
        with torch.random.fork_rng(devices=[]):
            self.head = InterTraitHeads(s3_mlp_dropout)

    def forward(self, windows, mapping, n_essays, bge, return_features=False):
        hidden = self.roberta(**windows).last_hidden_state
        cls = mean_by_essay(hidden[:, 0, :], mapping, n_essays)
        projected = self.projection(bge)
        fusion = torch.cat([self.roberta_norm(cls), projected], dim=1)
        result = self.head(fusion, return_features=return_features)
        if not return_features:
            return result
        prediction, trait_features = result
        return prediction, {
            "roberta": cls,
            "bge": bge,
            "projection": projected,
            "concat": fusion,
            **trait_features,
        }
