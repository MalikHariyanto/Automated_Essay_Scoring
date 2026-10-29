"""E32: co-attention sebelum pooling E25; tanpa BiGRU, gate tetap di model."""
import math
import torch
from torch import nn
from s3_sequence_pooling_helpers import AnswerAttentionPool, check_answer_pool_inputs


class AnswerCoAttentionPool(AnswerAttentionPool):
    """Respons/referensi768→128; matching640→768; pooling E25 tetap768→64→1."""
    def __init__(self, hidden_size=64):
        super().__init__(hidden_size)
        # Pertahankan RNG inisialisasi gate E25 setelah pool.
        with torch.random.fork_rng(devices=[]):
            self.response_projection = nn.Linear(768, 128)
            self.reference_projection = nn.Linear(768, 128)
            self.matching_projection = nn.Linear(5*128, 768)
        self.capture_alignment = False
        self.last_cross_attention = None
        self.last_reference_weights = None
        self.last_response_coattention_weights = None

    def forward(self, hidden, mask, reference_mask):
        check_answer_pool_inputs(hidden, mask)
        check_answer_pool_inputs(hidden, reference_mask)
        if bool((mask & reference_mask).any()):
            raise ValueError('Mask respons dan referensi harus terpisah.')
        responses = [row[m] for row,m in zip(hidden,mask)]
        references = [row[m] for row,m in zip(hidden,reference_mask)]
        lengths = torch.tensor([len(r) for r in responses],device=hidden.device)
        ref_lengths = torch.tensor([len(r) for r in references],device=hidden.device)
        response = self.response_projection(torch.nn.utils.rnn.pad_sequence(responses,batch_first=True))
        reference = self.reference_projection(torch.nn.utils.rnn.pad_sequence(references,batch_first=True))
        valid = torch.arange(response.shape[1],device=hidden.device)[None] < lengths[:,None]
        ref_valid = torch.arange(reference.shape[1],device=hidden.device)[None] < ref_lengths[:,None]
        with torch.autocast(device_type=hidden.device.type,enabled=False):
            a,r = response.float(),reference.float()
            scores = torch.bmm(a,r.transpose(1,2))/math.sqrt(128)
            a_to_r = torch.softmax(scores.masked_fill(~ref_valid[:,None,:],-torch.inf),dim=-1).masked_fill(~valid[:,:,None],0.)
            r_to_a = torch.softmax(scores.transpose(1,2).masked_fill(~valid[:,None,:],-torch.inf),dim=-1).masked_fill(~ref_valid[:,:,None],0.)
            aligned = torch.bmm(a_to_r,r)
            round_trip = torch.bmm(a_to_r,torch.bmm(r_to_a,a))
            matching = torch.cat([a,aligned,a-aligned,a*aligned,round_trip],dim=-1)
        matched = torch.tanh(self.matching_projection(matching.to(response.dtype)))
        # Kembalikan urutan ke posisi asli; pooling E25 memberi bobot pada mask respons.
        expanded = torch.zeros_like(hidden).masked_scatter(mask[:,:,None].expand_as(hidden),matched[valid].reshape(-1).to(hidden.dtype))
        output = super().forward(expanded,mask)
        self.last_response_coattention_weights = torch.zeros_like(mask,dtype=torch.float32).masked_scatter(mask,(r_to_a.sum(1)/ref_lengths[:,None])[valid]).detach()
        self.last_reference_weights = torch.zeros_like(reference_mask,dtype=torch.float32).masked_scatter(reference_mask,(a_to_r.sum(1)/lengths[:,None])[ref_valid]).detach()
        self.last_cross_attention = a_to_r.detach() if self.capture_alignment else None
        return output
