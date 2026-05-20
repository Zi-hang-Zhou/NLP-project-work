"""KV cache compression methods for Pythia-70M.

Three methods, all done as post-prefill operations on a plain DynamicCache:
- StreamingLLM: keep first N sink tokens + last W window tokens (positional)
- SnapKV: keep top-K by attention from a tail observation window + last W
- PyramidKV: SnapKV with per-layer linearly-decreasing budget

Eval forward uses a manually-constructed 4D causal mask + physical-length
`cache_position` so the model treats the (now shorter) cache correctly.
For PyramidKV, layers end up with different K lengths; we pad shorter layers
at the front with zeros so a single global attention mask matches all layers.
Zero V at padded slots means those slots contribute 0 to the attention output.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F
from transformers import DynamicCache


def streamingllm_compress(
    cache: DynamicCache,
    attentions=None,
    *,
    n_sink: int = 4,
    window_size: int = 508,
) -> DynamicCache:
    for layer_idx in range(len(cache.key_cache)):
        K = cache.key_cache[layer_idx]
        V = cache.value_cache[layer_idx]
        if K.shape[-2] <= n_sink + window_size:
            continue
        cache.key_cache[layer_idx] = torch.cat(
            [K[..., :n_sink, :], K[..., -window_size:, :]], dim=-2
        )
        cache.value_cache[layer_idx] = torch.cat(
            [V[..., :n_sink, :], V[..., -window_size:, :]], dim=-2
        )
    return cache


def _select_topk_keep(
    attn: torch.Tensor,
    seq_len: int,
    budget: int,
    window_size: int,
    kernel_size: int,
) -> torch.Tensor:
    """SnapKV importance scoring + top-k selection per head."""
    obs_attn = attn[..., -window_size:, : seq_len - window_size]
    importance = obs_attn.sum(dim=-2)
    B, H, L = importance.shape
    importance = F.avg_pool1d(
        importance.reshape(B * H, 1, L),
        kernel_size=kernel_size,
        stride=1,
        padding=kernel_size // 2,
    ).reshape(B, H, L)
    k = budget - window_size
    if k <= 0:
        tail = torch.arange(seq_len - budget, seq_len, device=attn.device)
        return tail.view(1, 1, -1).expand(B, H, -1)
    topk_idx = importance.topk(k, dim=-1).indices
    tail = torch.arange(seq_len - window_size, seq_len, device=attn.device)
    tail = tail.view(1, 1, -1).expand(B, H, -1)
    keep = torch.cat([topk_idx, tail], dim=-1)
    keep, _ = keep.sort(dim=-1)
    return keep


def snapkv_compress(
    cache: DynamicCache,
    attentions,
    *,
    budget: int,
    window_size: int = 32,
    kernel_size: int = 7,
) -> DynamicCache:
    for layer_idx in range(len(cache.key_cache)):
        K = cache.key_cache[layer_idx]
        V = cache.value_cache[layer_idx]
        seq_len = K.shape[-2]
        if seq_len <= budget:
            continue
        keep = _select_topk_keep(
            attentions[layer_idx], seq_len, budget, window_size, kernel_size
        )
        keep_expand = keep.unsqueeze(-1).expand(-1, -1, -1, K.shape[-1])
        cache.key_cache[layer_idx] = K.gather(-2, keep_expand)
        cache.value_cache[layer_idx] = V.gather(-2, keep_expand)
    return cache


def pyramidkv_compress(
    cache: DynamicCache,
    attentions,
    *,
    budget: int,
    window_size: int = 32,
    kernel_size: int = 7,
    beta: float = 20.0,
) -> DynamicCache:
    n_layers = len(cache.key_cache)
    half_spread = beta * (n_layers - 1) / 2.0
    per_layer_budget = [
        max(window_size + 1, int(round(budget + half_spread - beta * layer_idx)))
        for layer_idx in range(n_layers)
    ]
    for layer_idx in range(n_layers):
        K = cache.key_cache[layer_idx]
        V = cache.value_cache[layer_idx]
        seq_len = K.shape[-2]
        b = per_layer_budget[layer_idx]
        if seq_len <= b:
            continue
        keep = _select_topk_keep(
            attentions[layer_idx], seq_len, b, window_size, kernel_size
        )
        keep_expand = keep.unsqueeze(-1).expand(-1, -1, -1, K.shape[-1])
        cache.key_cache[layer_idx] = K.gather(-2, keep_expand)
        cache.value_cache[layer_idx] = V.gather(-2, keep_expand)
    pad_caches_to_max(cache)
    return cache


def pad_caches_to_max(cache: DynamicCache) -> None:
    """Pad layers' K/V at the FRONT by repeating the first kept slot.

    Padding with zeros breaks attention because Q·0 = 0 can outscore real
    Q·K when real scores are negative; padded slots then steal softmax mass.
    Repeating real K[0] keeps scores realistic and the bias is concentrated
    on a sink-like first token (often near-uniform attention anyway).
    """
    lens = [K.shape[-2] for K in cache.key_cache]
    max_len = max(lens)
    for layer_idx in range(len(cache.key_cache)):
        K = cache.key_cache[layer_idx]
        V = cache.value_cache[layer_idx]
        L = K.shape[-2]
        if L == max_len:
            continue
        n_pad = max_len - L
        k_pad = K[..., :1, :].expand(-1, -1, n_pad, -1)
        v_pad = V[..., :1, :].expand(-1, -1, n_pad, -1)
        cache.key_cache[layer_idx] = torch.cat([k_pad, K], dim=-2)
        cache.value_cache[layer_idx] = torch.cat([v_pad, V], dim=-2)


def build_cache(method: str, budget: int, window_size: int = 32):
    method = method.lower()
    if method == "dense":
        return DynamicCache(), None
    if method == "streamingllm":
        n_sink = 4
        w = max(1, budget - n_sink)
        return (
            DynamicCache(),
            lambda c, a: streamingllm_compress(c, a, n_sink=n_sink, window_size=w),
        )
    if method == "snapkv":
        return (
            DynamicCache(),
            lambda c, a: snapkv_compress(
                c, a, budget=budget, window_size=window_size
            ),
        )
    if method == "pyramidkv":
        return (
            DynamicCache(),
            lambda c, a: pyramidkv_compress(
                c, a, budget=budget, window_size=window_size
            ),
        )
    raise ValueError(f"unknown method: {method}")


def make_4d_causal_mask(physical_kv_len: int, q_len: int, device, dtype) -> torch.Tensor:
    """4D mask of shape (1, 1, q_len, physical_kv_len + q_len).

    Cache K columns: 0 (visible). New K columns: causal triangle with -inf
    above the diagonal so each new query sees prior new keys but not future ones.
    """
    minv = torch.finfo(dtype).min
    mask = torch.zeros(1, 1, q_len, physical_kv_len + q_len, device=device, dtype=dtype)
    causal = torch.triu(
        torch.full((q_len, q_len), minv, device=device, dtype=dtype), diagonal=1
    )
    mask[:, :, :, physical_kv_len:] = causal
    return mask
