"""E5 scoring head adapted from the inter-trait attention in CTS (Ridley et al., 2021).

Reference implementation:
https://github.com/robert1ridley/cross-prompt-trait-scoring/blob/master/models/CTS.py
E5 learns trait representations from S3 fusion features using 64-unit MLPs.
"""

import torch
from torch import nn


class TraitAttentionHead(nn.Module):
    """Score each trait using its representation and a weighted sum of other traits."""

    def __init__(self, trait_names, dropout=0.2):
        super().__init__()
        self.trait_names = tuple(trait_names)
        self.representations = nn.ModuleDict({
            name: nn.Sequential(nn.Linear(1024, 64), nn.GELU(), nn.Dropout(dropout))
            for name in self.trait_names
        })
        self.scores = nn.ModuleDict({
            name: nn.Linear(128, 1) for name in self.trait_names
        })

    def forward(self, fusion):
        if fusion.ndim != 2 or fusion.shape[1] != 1024:
            raise ValueError('TraitAttentionHead memerlukan fitur fusion [B,1024].')
        traits = torch.stack(
            [self.representations[name](fusion) for name in self.trait_names], dim=1,
        )
        # Dot-product attention CTS. Each query uses only the other three traits.
        # Scores and softmax stay FP32 during mixed-precision encoder training.
        with torch.autocast(device_type=fusion.device.type, enabled=False):
            values = traits.float()
            logits = torch.bmm(values, values.transpose(1, 2))
            diagonal = torch.eye(len(self.trait_names), dtype=torch.bool, device=fusion.device)
            weights = torch.softmax(logits.masked_fill(diagonal, -torch.inf), dim=-1)
            context = torch.bmm(weights, values)
            combined = torch.cat([values, context], dim=-1)
        return torch.cat([
            self.scores[name](combined[:, index])
            for index, name in enumerate(self.trait_names)
        ], dim=1)
