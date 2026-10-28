"""E27: pencocokan token respons–referensi; ditanam ke Colab mandiri."""
import math
import torch
from torch import nn
from transformers import AutoTokenizer
from s3_sequence_pooling_helpers import check_answer_pool_inputs, answer_token_mask, AnswerBiGRU


def reference_context_positions(context, tok, cfg, info):
    """Posisi isi referensi dari offset karakter; ID token berulang tidak ditebak."""
    prefix = '\nJawaban referensi: '
    text = prefix + context[1]
    try:
        aligned = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    except NotImplementedError:
        # RoBERTa training tetap slow seperti E26; fast hanya membaca offset.
        if not hasattr(tok, '_aes_reference_fast'):
            spec = cfg['roberta']
            tok._aes_reference_fast = AutoTokenizer.from_pretrained(
                spec['checkpoint'], revision=spec['revision'], use_fast=True)
        aligned = tok._aes_reference_fast(text, add_special_tokens=False, return_offsets_mapping=True)
    original = tok.encode(text, add_special_tokens=False)
    if list(aligned['input_ids']) != original or len(aligned['offset_mapping']) != len(original):
        raise ValueError('ID/offset tokenizer referensi tidak identik; posisi tidak ditebak.')
    positions = []
    for index, (start, end) in enumerate(aligned['offset_mapping'][:info['reference_kept']]):
        if not 0 <= start <= end <= len(text):
            raise ValueError('Offset tokenizer referensi tidak sah.')
        # Token yang mencakup teks isi diperbolehkan; token prefix/spasi saja dikecualikan.
        if end > len(prefix) and text[max(start, len(prefix)):end].strip():
            positions.append(info['question_kept'] + index)
    if not positions:
        raise ValueError('Isi referensi habis oleh truncation/prefix. Tambah anggaran konteks sebelum training.')
    return positions


def reference_token_mask(windows, mapping, details, padding_side='right'):
    """Pemetaan posisi referensi memakai offset padding yang sama dengan respons."""
    converted = [{**info, 'answer_positions': info['reference_positions']} for info in details]
    return answer_token_mask(windows, mapping, converted, padding_side)


class AnswerReferenceCoAttention(AnswerBiGRU):
    """E23 + co-attention respons128 ↔ referensi128, lalu ringkasan ujung dua arah."""
    def __init__(self, hidden_size=64):
        super().__init__(hidden_size)
        dim = 2 * hidden_size
        # Sesudah layer acuan: inisialisasi BiGRU/output E23 tetap sama.
        self.reference_projection = nn.Linear(768, dim)
        self.matching_projection = nn.Linear(5 * dim, dim)
        self.capture_alignment = False
        self.last_cross_attention = None
        self.last_reference_weights = None
        self.last_weights = None

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
        # Pola E23: ujung arah maju + ujung arah mundur, setelah pencocokan.
        # Tidak menambahkan additive attention pooling milik E26.
        batch_ids = torch.arange(len(matched), device=hidden.device)
        last_ids = (lengths-1).to(hidden.device)
        half = matched.shape[-1] // 2
        summary = torch.cat([matched[batch_ids, last_ids, :half], matched[:, 0, half:]], dim=-1)
        # Bobot ini untuk inspeksi co-attention referensi→respons, bukan weighted pooling.
        response_importance = r_to_a.sum(1) / ref_lengths.to(hidden.device)[:, None]
        self.last_weights = torch.zeros_like(mask, dtype=torch.float32).masked_scatter(mask, response_importance[valid]).detach()
        self.last_summary = summary.detach()
        reference_importance = a_to_r.sum(1) / lengths.to(hidden.device)[:, None]
        self.last_reference_weights = torch.zeros_like(reference_mask, dtype=torch.float32).masked_scatter(
            reference_mask, reference_importance[ref_valid]).detach()
        # Matriks hanya ditahan untuk ekspor inspeksi, bukan graph training atau seluruh dataset.
        self.last_cross_attention = a_to_r.detach() if self.capture_alignment else None
        return self.output(summary)


def optimizer_groups_isolated_mlp(model, cfg):
    """MLP S3 saja dapat diberi LR berbeda; semua parameter tercakup tepat sekali."""
    buckets = {'roberta': [], 'projection': [], 'head': [], 'pooling_norm': []}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith('roberta.'):
            group = 'roberta'
        elif name.startswith('projection.'):
            group = 'projection'
        elif name.startswith('head.'):
            group = 'head'
        else:
            group = 'pooling_norm'
        buckets[group].append(parameter)
    rates = {'roberta': cfg['s3_encoder_lr'] if model.scenario == 'S3' else cfg['encoder_lr'],
             'projection': cfg['projection_lr'],
             'head': cfg['s3_mlp_lr'] if model.scenario == 'S3' else cfg['head_lr'],
             'pooling_norm': cfg['s3_layers_lr']}
    result = [{'name': name, 'params': params, 'lr': rates[name]} for name, params in buckets.items() if params]
    ids = [id(p) for group in result for p in group['params']]
    expected = {id(p) for p in model.parameters() if p.requires_grad}
    if len(ids) != len(set(ids)) or set(ids) != expected:
        raise ValueError('Optimizer harus mencakup setiap parameter trainable tepat sekali.')
    return result
