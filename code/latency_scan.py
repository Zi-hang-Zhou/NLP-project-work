"""大 cache 扫描：测 KV 压缩在 Pythia-70M 上的真实 decode 加速。

为什么需要这个：
- run_all.py 的 prefix=1024 太小，attention 计算只占 TPOT 的 ~26%，剩下是 MLP +
  layernorm + kernel launch + Python overhead。所以"压缩前 cache=1024 vs 压缩后
  cache=128"在 TPOT 上看不出差距。
- KV 压缩的真实价值在长上下文（8k-32k），此时 attention 的 Q·K^T 和 attn·V 变
  memory-bound（每步要读完整 K/V），cache 越大 decode 越慢，压缩才能加速。

做法（合成 cache）：
- 跳过 prefill，直接构造一个 (1, 8, N, 64) 的 DynamicCache（K/V 用随机数填充）
- 跑 decode 测 TPOT
- 扫描 cache_size ∈ {128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768}

诚实声明：cache 内容是随机数，**模型输出毫无意义**（位置编码也在 Pythia 训练长度
外），但 GEMM、读 K/V 的 memory bandwidth 完全真实——TPOT 就是"如果 cache 这么大，
decode 一个 token 在 GPU 上要多久"。

这模拟的工业场景：用户给 8k tokens 的 prompt，模型 decode 时每步都要读完整 KV。
"""
from __future__ import annotations
import json
import os
import sys
import time
import torch
from transformers import DynamicCache

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
_ROOT = os.path.dirname(_HERE)
from caches import make_4d_causal_mask
from eval import load_model


# 扫描的 cache 大小（覆盖典型 budget 和典型 prefix）
CACHE_SIZES = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]
GEN_TOKENS = 128  # 每个 cache_size 跑多少 decode 步（取后半 median 算 TPOT）

OUT_PATH = os.path.join(_ROOT, "results", "latency_scan.jsonl")


def build_synthetic_cache(model, cache_size: int, dtype, device):
    """构造一个指定长度的 DynamicCache，K/V 用随机数填充。

    跳过 prefill 的好处：
    1. 任意大 cache_size 都能造（不受模型 context 长度限制——RoPE 会外推，
       但计算量、memory 读写量是真的）
    2. 测的就是"这么大 cache 下 decode 一步要多久"，干净
    """
    n_layers = model.config.num_hidden_layers      # Pythia-70M = 6
    n_heads = model.config.num_attention_heads     # = 8
    head_dim = model.config.hidden_size // n_heads # 512/8 = 64
    cache = DynamicCache()
    for layer in range(n_layers):
        # × 0.1 让数值在合理量级，避免出 inf / nan（我们不看 logits，但避免计算异常）
        K = torch.randn(1, n_heads, cache_size, head_dim, device=device, dtype=dtype) * 0.1
        V = torch.randn(1, n_heads, cache_size, head_dim, device=device, dtype=dtype) * 0.1
        cache.update(K, V, layer)
    return cache


@torch.no_grad()
def measure_decode_tpot(model, cache_size: int, gen_tokens: int = GEN_TOKENS) -> float:
    """对给定 cache_size 测 TPOT（返回毫秒）。"""
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    # ---- Per-measurement warmup：编译这个 cache_size 的 kernel + ramp GPU clock ----
    for _ in range(2):
        wu_cache = build_synthetic_cache(model, cache_size, dtype, device)
        wu_id = torch.tensor([[0]], device=device)
        for _ in range(8):
            pkv = wu_cache.key_cache[0].shape[-2]
            mask = make_4d_causal_mask(pkv, 1, device, dtype)
            pos = torch.tensor([pkv], device=device)
            wu_out = model(wu_id, past_key_values=wu_cache, use_cache=True,
                           attention_mask=mask, cache_position=pos)
            wu_id = wu_out.logits[:, -1:, :].argmax(-1)
        del wu_cache, wu_out
    torch.cuda.synchronize()

    # ---- 实测 ----
    cache = build_synthetic_cache(model, cache_size, dtype, device)
    next_id = torch.tensor([[0]], device=device)
    decode_times = []
    for _ in range(gen_tokens):
        physical_kv_len = cache.key_cache[0].shape[-2]
        attention_mask = make_4d_causal_mask(physical_kv_len, 1, device, dtype)
        cache_position = torch.tensor([physical_kv_len], device=device)
        torch.cuda.synchronize()
        t = time.time()
        out = model(next_id, past_key_values=cache, use_cache=True,
                    attention_mask=attention_mask, cache_position=cache_position)
        torch.cuda.synchronize()
        decode_times.append(time.time() - t)
        next_id = out.logits[:, -1:, :].argmax(-1)

    # 后半段 (gen_tokens//2 → end) 的 median，抗 GPU clock ramping + 偶发慢迭代
    steady = sorted(decode_times[len(decode_times) // 2:])
    return steady[len(steady) // 2] * 1000  # 转毫秒


def kv_size_mb(cache_size: int, n_layers=6, n_heads=8, head_dim=64, bytes_per=4) -> float:
    """理论 KV 显存（fp32）：2 (K+V) × 层 × 头 × seq × dim × bytes。"""
    return 2 * n_layers * n_heads * cache_size * head_dim * bytes_per / 1e6


def main():
    tok, model = load_model()

    # ---- 全局 warmup：跑几次大 cache 把 GPU 时钟拉到 max boost ----
    print("global warmup...")
    for _ in range(3):
        _ = measure_decode_tpot(model, 4096, gen_tokens=32)
    print("global warmup done\n")

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    results = []
    print(f"{'cache_size':>10}  {'tpot_ms':>10}  {'kv_mb':>10}")
    for cs in CACHE_SIZES:
        tpot = measure_decode_tpot(model, cs)
        r = {"cache_size": cs, "tpot_ms": tpot, "kv_mb": kv_size_mb(cs)}
        results.append(r)
        print(f"{cs:>10d}  {tpot:>9.3f}  {kv_size_mb(cs):>9.2f}")

    with open(OUT_PATH, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"\nsaved {OUT_PATH}")

    # ---- 加速比解读 ----
    print("\n--- 压缩加速比（cache=N 压到 budget=B 后 TPOT 变化）---")
    by_size = {r["cache_size"]: r["tpot_ms"] for r in results}
    for budget in [128, 1024]:
        if budget not in by_size:
            continue
        comp_tpot = by_size[budget]
        print(f"\nbudget={budget}  TPOT={comp_tpot:.3f}ms")
        for prefix in [2048, 4096, 8192, 16384, 32768, 65536, 131072]:
            if prefix <= budget or prefix not in by_size:
                continue
            dense_tpot = by_size[prefix]
            speedup = dense_tpot / comp_tpot
            print(f"    cache {prefix:>5d} → {budget:>4d}: "
                  f"{dense_tpot:.3f}ms → {comp_tpot:.3f}ms  ({speedup:.2f}× faster)")


if __name__ == "__main__":
    main()
