"""Tambahan S3 E25/E26; definisi ditanam langsung ke notebook mandiri."""
import torch
from torch import nn
from s3_sequence_pooling_helpers import check_answer_pool_inputs


class AnswerFusionGate(nn.Module):
    """Gate skalar per jawaban; dimensi concat tetap768+256=1024."""
    def __init__(self, hidden_size=32):
        super().__init__()
        self.hidden = nn.Linear(1024, hidden_size)
        self.score = nn.Linear(hidden_size, 1)
        # g=.5 saat awal: 2*g dan 2*(1-g) sama dengan1, identik concat acuan.
        nn.init.zeros_(self.score.weight)
        nn.init.zeros_(self.score.bias)

    def forward(self, roberta, bge):
        if roberta.ndim != 2 or bge.shape != (len(roberta), 256) or roberta.shape[1] != 768:
            raise ValueError('Gated fusion memerlukan fitur RoBERTa[N,768] dan BGE[N,256].')
        gate = torch.sigmoid(self.score(torch.tanh(self.hidden(torch.cat([roberta, bge], dim=1)))))
        fused = torch.cat([2*gate*roberta, 2*(1-gate)*bge], dim=1)
        return fused, gate


class AnswerBiGRUAttention(nn.Module):
    """E26: urutan token -> BiGRU -> attention atas semua output -> proyeksi768."""
    def __init__(self, hidden_size=64):
        super().__init__()
        self.gru = nn.GRU(768, hidden_size, batch_first=True, bidirectional=True)
        self.attention_hidden = nn.Linear(2*hidden_size, hidden_size)
        self.attention_score = nn.Linear(hidden_size, 1, bias=False)
        self.output = nn.Linear(2*hidden_size, 768)
        self.last_weights = None
        self.last_summary = None

    def forward(self, hidden, mask):
        check_answer_pool_inputs(hidden, mask)
        sequences = [row[keep] for row, keep in zip(hidden, mask)]
        lengths = torch.tensor([len(row) for row in sequences], dtype=torch.long)
        padded = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True)
        packed = torch.nn.utils.rnn.pack_padded_sequence(padded, lengths, batch_first=True, enforce_sorted=False)
        packed_outputs, _ = self.gru(packed)
        outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_outputs, batch_first=True)
        valid = torch.arange(outputs.shape[1], device=outputs.device)[None] < lengths.to(outputs.device)[:,None]
        scores = self.attention_score(torch.tanh(self.attention_hidden(outputs))).squeeze(-1).float()
        weights = torch.softmax(scores.masked_fill(~valid, -torch.inf), dim=1)
        summary = (weights.unsqueeze(-1)*outputs.float()).sum(1).to(outputs.dtype)
        # Pemetaan balik bobot ke posisi token asli, supaya Excel tidak menggeser konteks/padding.
        self.last_weights = torch.zeros_like(mask, dtype=torch.float32).masked_scatter(mask, weights[valid]).detach()
        self.last_summary = summary.detach()
        return self.output(summary)
