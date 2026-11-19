class MultiTraitModel(nn.Module):
    """E1 reset: last-layer CLS, BGE projection, shared four-output MLP."""

    def __init__(self, scenario, roberta, s3_mlp_dropout=0.2):
        super().__init__()
        self.scenario, self.roberta = scenario, roberta
        self.roberta.requires_grad_(True)
        self.roberta_norm = nn.LayerNorm(768)
        self.projection = nn.Sequential(nn.Linear(1024, 256), nn.LayerNorm(256))
        self.head = nn.Sequential(
            nn.Linear(1024, 256),
            nn.GELU(),
            nn.Dropout(s3_mlp_dropout),
            nn.Linear(256, 4),
        )

    def forward(self, windows, mapping, n_essays, bge, return_features=False):
        hidden = self.roberta(**windows).last_hidden_state
        cls = mean_by_essay(hidden[:, 0, :], mapping, n_essays)
        projected = self.projection(bge)
        fusion = torch.cat([self.roberta_norm(cls), projected], dim=1)
        prediction = self.head(fusion)
        features = {
            "roberta": cls,
            "bge": bge,
            "projection": projected,
            "concat": fusion,
        }
        return (prediction, features) if return_features else prediction
