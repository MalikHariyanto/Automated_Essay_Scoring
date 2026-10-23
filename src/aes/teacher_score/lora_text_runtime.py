"""Batasi backend Transformers worker AES ke PyTorch dan teks saja."""


def configure_text_only_transformers():
    """Hindari import vision ops yang tidak diperlukan oleh RoBERTa/BGE/LoRA."""
    import os
    os.environ['USE_TORCH'] = '1'
    os.environ['USE_TF'] = '0'
    os.environ['USE_FLAX'] = '0'
    from transformers.utils import import_utils
    # Transformers 4.44/4.48 mendeteksi torchvision dari keberadaan paket,
    # meskipun binary ops-nya tidak cocok dengan torch runtime. Notebook ini
    # hanya teks: jangan muat backend gambar melalui import tidak langsung PEFT.
    # Hanya worker AES yang diubah; paket Colab dan operator torch tetap utuh.
    if not hasattr(import_utils, '_torchvision_available'):
        raise RuntimeError('Versi Transformers berbeda dari versi notebook yang didukung.')
    import_utils._torchvision_available = False
    return {'transformers_backend': 'pytorch_text_only', 'torchvision_enabled': False}
