from __future__ import annotations

import torch
import torch.nn.functional as F


def flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False, *args, **kwargs):
    if q.dim() != 4:
        raise NotImplementedError("flash_attn_func stub expects a 4D tensor")
    q = q.permute(0, 2, 1, 3)
    k = k.permute(0, 2, 1, 3)
    v = v.permute(0, 2, 1, 3)
    out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p, is_causal=causal)
    return out.permute(0, 2, 1, 3)


def flash_attn_varlen_func(*args, **kwargs):
    raise NotImplementedError("flash_attn_varlen_func is not used when attn_implementation='eager'")