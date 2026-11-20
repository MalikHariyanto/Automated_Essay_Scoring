"""LoRA khusus RoBERTa S3; fusion dan head mengikuti notebook MLP acuan."""
import math

import torch

from .lora_text_runtime import configure_text_only_transformers

configure_text_only_transformers()
from peft import LoraConfig, get_peft_model


def validate_lora_config(cfg):
    """Tolak parameter salah sebelum memuat encoder atau memulai training."""
    if type(cfg['lora_rank']) is not int or cfg['lora_rank'] < 1:
        raise ValueError('Rank LoRA harus bilangan bulat positif.')
    if not math.isfinite(cfg['lora_alpha']) or cfg['lora_alpha'] <= 0:
        raise ValueError('Alpha LoRA harus positif dan terhingga.')
    if not 0 <= cfg['lora_dropout'] < 1:
        raise ValueError('Dropout LoRA harus berada pada rentang 0 sampai kurang dari 1.')
    if not math.isfinite(cfg['lora_lr']) or cfg['lora_lr'] <= 0:
        raise ValueError('Learning rate LoRA harus positif dan terhingga.')
    if cfg['lora_targets'] != ['query', 'value']:
        raise ValueError('Eksperimen ini hanya memasang LoRA pada query dan value attention RoBERTa.')


def attach_s3_lora(model, cfg):
    """Bekukan bobot asli S3, lalu latih penyesuaian kecil pada attention."""
    if model.scenario != 'S3':
        return model
    validate_lora_config(cfg)
    base = model.roberta
    # Pastikan target ada di setiap layer encoder, bukan cocok sebagian diam-diam.
    expected = 2 * base.config.num_hidden_layers
    targets = [name for name, layer in base.named_modules()
               if isinstance(layer, torch.nn.Linear)
               and name.endswith(('.attention.self.query', '.attention.self.value'))]
    if len(targets) != expected:
        raise ValueError('Jumlah layer query/value RoBERTa tidak cocok dengan konfigurasi encoder.')
    # PEFT membekukan bobot asli. A diinisialisasi acak dan B nol: penyesuaian awal nol.
    adapter = LoraConfig(r=cfg['lora_rank'], lora_alpha=cfg['lora_alpha'],
                         lora_dropout=cfg['lora_dropout'], target_modules=cfg['lora_targets'],
                         bias='none', init_lora_weights=True)
    model.roberta = get_peft_model(base, adapter)
    trainable = [name for name, p in model.roberta.named_parameters() if p.requires_grad]
    if not trainable or any('.lora_A.' not in name and '.lora_B.' not in name for name in trainable):
        raise RuntimeError('S3 hanya boleh melatih parameter LoRA di dalam encoder RoBERTa.')
    return model


def training_state(model):
    """S3 menyimpan LoRA + LayerNorm/proyeksi/head; S1/S2 menyimpan state utuh."""
    state = model.state_dict()
    if model.scenario == 'S3':
        names = {name for name, p in model.named_parameters() if p.requires_grad}
        state = {name: value for name, value in state.items() if name in names}
    return {name: value.detach().cpu().clone() for name, value in state.items()}


def restore_training_state(model, state):
    """Bangun ulang base dari revision terkunci, lalu pulihkan seluruh parameter belajar."""
    if model.scenario != 'S3':
        model.load_state_dict(state, strict=True)
        return
    expected = {name for name, p in model.named_parameters() if p.requires_grad}
    if set(state) != expected:
        raise ValueError('Checkpoint S3 tidak memuat tepat seluruh LoRA dan layer prediksi yang dilatih.')
    # Key frozen yang tidak dimuat memang berasal dari checkpoint pretrained terkunci.
    model.load_state_dict(state, strict=False)


def parameter_report(model):
    """Tampilkan bagian yang belajar dan yang frozen dengan istilah singkat."""
    groups = {}
    for name, p in model.named_parameters():
        if name.startswith('roberta.'):
            part = 'LoRA RoBERTa' if '.lora_A.' in name or '.lora_B.' in name else 'Bobot asli RoBERTa'
        elif name.startswith('projection.'):
            part = 'Proyeksi BGE + LayerNorm'
        elif name.startswith('roberta_norm.'):
            part = 'LayerNorm tambahan RoBERTa'
        else:
            part = 'Head regresi'
        row = groups.setdefault(part, {'Skenario': model.scenario, 'Bagian': part,
                                      'Parameter Dilatih': 0, 'Parameter Frozen': 0})
        row['Parameter Dilatih' if p.requires_grad else 'Parameter Frozen'] += p.numel()
    return list(groups.values())
