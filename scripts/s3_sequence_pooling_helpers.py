"""Layer urutan S3; builder menanam definisi ini ke notebook mandiri."""
import torch
from torch import nn


def answer_token_mask(windows, mapping, details, padding_side='right'):
    """Mask posisi respons saja; konteks/prefix/special/padding tidak dipool."""
    valid = windows['attention_mask'].bool()
    mask = torch.zeros_like(valid)
    flat = [(essay, positions) for essay, info in enumerate(details)
            for positions in info['answer_positions']]
    if len(flat) != len(valid) or padding_side not in ('left', 'right'):
        raise ValueError('Pemetaan window/posisi jawaban tidak cocok.')
    for window, (essay, positions) in enumerate(flat):
        if int(mapping[window]) != essay or not positions:
            raise ValueError('Window harus berisi token jawaban dari esai yang benar.')
        offset = valid.shape[1] - int(valid[window].sum()) if padding_side == 'left' else 0
        indices = torch.tensor(positions, dtype=torch.long, device=valid.device) + offset
        if int(indices.min()) < 0 or int(indices.max()) >= valid.shape[1]:
            raise ValueError('Posisi token jawaban keluar dari window.')
        if not bool(valid[window, indices].all()):
            raise ValueError('Token padding tidak boleh menjadi token jawaban.')
        mask[window, indices] = True
    return mask


def check_answer_pool_inputs(hidden, mask):
    if (hidden.ndim != 3 or mask is None or mask.dtype != torch.bool
            or tuple(mask.shape) != tuple(hidden.shape[:2])
            or mask.device != hidden.device):
        raise ValueError('Pooling memerlukan hidden [window,token,768] dan mask jawaban boolean.')
    if not bool(mask.any(dim=1).all()):
        raise ValueError('Setiap window harus memiliki minimal satu token jawaban.')


class AnswerBiGRU(nn.Module):
    """E23: urutan token jawaban -> BiGRU -> state akhir dua arah -> proyeksi768."""
    def __init__(self, hidden_size=64):
        super().__init__()
        self.gru = nn.GRU(768, hidden_size, batch_first=True, bidirectional=True)
        self.output = nn.Linear(2 * hidden_size, 768)
        self.last_summary = None

    def forward(self, hidden, mask):
        check_answer_pool_inputs(hidden, mask)
        # Ambil token dalam urutan aslinya; 768 adalah dimensi fitur, bukan panjang urutan.
        sequences = [row[keep] for row, keep in zip(hidden, mask)]
        lengths = torch.tensor([len(row) for row in sequences], dtype=torch.long)
        padded = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True)
        packed = torch.nn.utils.rnn.pack_padded_sequence(
            padded, lengths, batch_first=True, enforce_sorted=False)
        _, states = self.gru(packed)
        # pack_padded_sequence menjaga state akhir tidak membaca padding.
        summary = torch.cat([states[-2], states[-1]], dim=-1)
        self.last_summary = summary.detach()
        return self.output(summary)


class AnswerAttentionPool(nn.Module):
    """E24: additive attention bertopeng pada token jawaban; keluaran tetap768."""
    def __init__(self, hidden_size=64):
        super().__init__()
        self.hidden = nn.Linear(768, hidden_size)
        self.score = nn.Linear(hidden_size, 1, bias=False)
        self.last_weights = None

    def forward(self, hidden, mask):
        check_answer_pool_inputs(hidden, mask)
        scores = self.score(torch.tanh(self.hidden(hidden))).squeeze(-1).float()
        # Softmax FP32 dan mask sebelum softmax: bobot non-jawaban tepat nol.
        weights = torch.softmax(scores.masked_fill(~mask, -torch.inf), dim=1)
        self.last_weights = weights.detach()
        return (weights.unsqueeze(-1) * hidden.float()).sum(dim=1).to(hidden.dtype)
