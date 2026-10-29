"""E30: dua tahap attention di atas E26; source ditanam ke notebook mandiri."""
import math
import torch
from torch import nn
from s3_sequence_pooling_helpers import check_answer_pool_inputs
from s3_gated_bigru_attention_helpers import AnswerBiGRUAttention


class AnswerBiGRUCoAttentionPool(AnswerBiGRUAttention):
    """E30: E26 + co-attention respons128 ↔ referensi128, lalu attention pooling128."""
    def __init__(self, hidden_size=64):
        super().__init__(hidden_size)
        dim = 2 * hidden_size
        # Sesudah layer acuan: inisialisasi BiGRU/attention/output E26 tetap sama.
        self.reference_projection = nn.Linear(768, dim)
        self.matching_projection = nn.Linear(5 * dim, dim)
        self.capture_alignment = False
        self.last_cross_attention = None
        self.last_reference_weights = None
        self.last_weights = None
        self.last_response_coattention_weights = None

    def forward(self, hidden, mask, reference_mask):
        check_answer_pool_inputs(hidden, mask)
        check_answer_pool_inputs(hidden, reference_mask)
        if bool((mask & reference_mask).any()):
            raise ValueError('Mask respons dan referensi harus terpisah.')
        responses = [row[keep] for row, keep in zip(hidden, mask)]
        references = [row[keep] for row, keep in zip(hidden, reference_mask)]
        lengths = torch.tensor([len(row) for row in responses], dtype=torch.long)
        ref_lengths = torch.tensor([len(row) for row in references], dtype=torch.long)
        padded = torch.nn.utils.rnn.pad_sequence(responses, batch_first=True)
        packed = torch.nn.utils.rnn.pack_padded_sequence(padded, lengths, batch_first=True, enforce_sorted=False)
        packed_outputs, _ = self.gru(packed)
        response, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_outputs, batch_first=True)
        reference = self.reference_projection(torch.nn.utils.rnn.pad_sequence(references, batch_first=True))
        valid = torch.arange(response.shape[1], device=hidden.device)[None] < lengths.to(hidden.device)[:, None]
        ref_valid = torch.arange(reference.shape[1], device=hidden.device)[None] < ref_lengths.to(hidden.device)[:, None]
        # Softmax/produk skor FP32 juga saat encoder memakai AMP. Kedua urutan punya >=1 token.
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            response32, reference32 = response.float(), reference.float()
            scores = torch.bmm(response32, reference32.transpose(1, 2)) / math.sqrt(response.shape[-1])
            a_to_r = torch.softmax(scores.masked_fill(~ref_valid[:, None, :], -torch.inf), dim=-1)
            a_to_r = a_to_r.masked_fill(~valid[:, :, None], 0.)
            r_to_a = torch.softmax(scores.transpose(1, 2).masked_fill(~valid[:, None, :], -torch.inf), dim=-1)
            r_to_a = r_to_a.masked_fill(~ref_valid[:, :, None], 0.)
            aligned_reference = torch.bmm(a_to_r, reference32)
            # Jalur balik respons→referensi→respons; bukan attention atas satu vektor BGE.
            round_trip = torch.bmm(a_to_r, torch.bmm(r_to_a, response32))
            matching = torch.cat([response32, aligned_reference, response32 - aligned_reference,
                                  response32 * aligned_reference, round_trip], dim=-1)
        matched = torch.tanh(self.matching_projection(matching.to(response.dtype)))
        # Attention pooling E26 sekarang merangkum keluaran matching co-attention.
        pooling_scores = self.attention_score(torch.tanh(self.attention_hidden(matched))).squeeze(-1).float()
        pooling_weights = torch.softmax(pooling_scores.masked_fill(~valid, -torch.inf), dim=1)
        summary = (pooling_weights.unsqueeze(-1) * matched.float()).sum(1).to(response.dtype)
        self.last_weights = torch.zeros_like(mask, dtype=torch.float32).masked_scatter(mask, pooling_weights[valid]).detach()
        # Rata-rata bobot arah referensi→respons untuk inspeksi, terpisah dari pooling.
        response_importance = r_to_a.sum(1) / ref_lengths.to(hidden.device)[:, None]
        self.last_response_coattention_weights = torch.zeros_like(mask, dtype=torch.float32).masked_scatter(
            mask, response_importance[valid]).detach()
        self.last_summary = summary.detach()
        reference_importance = a_to_r.sum(1) / lengths.to(hidden.device)[:, None]
        self.last_reference_weights = torch.zeros_like(reference_mask, dtype=torch.float32).masked_scatter(
            reference_mask, reference_importance[ref_valid]).detach()
        # Matriks hanya ditahan untuk ekspor inspeksi, bukan graph training atau seluruh dataset.
        self.last_cross_attention = a_to_r.detach() if self.capture_alignment else None
        return self.output(summary)
