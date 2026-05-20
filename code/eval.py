"""评测脚本：在 KV 压缩之后测 PPL、TTFT、TPOT、显存。

整体流程：
1. load_model：加载 Pythia-70m（fp32，eager attention）
2. streaming_ppl：chunked PPL 评测 —— 每个 chunk 喂 prefix → 压缩 → 喂 eval 段算 NLL
3. latency_memory：测延迟和显存 —— prefill → 压缩 → 逐个生成 token 测 TPOT
"""
from __future__ import annotations
import json
import os
import time
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from caches import build_cache, make_4d_causal_mask


# 项目根目录（code/ 的父目录），所有数据路径相对它解析
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 模型路径：默认指向 repo 内 models/pythia-70m；不存在则回退到 HF Hub id
# 也可以用环境变量覆盖：MODEL_PATH=/path/to/pythia-70m python eval.py
_DEFAULT_LOCAL = os.path.join(PROJECT_ROOT, "models", "pythia-70m")
MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    _DEFAULT_LOCAL if os.path.isdir(_DEFAULT_LOCAL) else "EleutherAI/pythia-70m",
)


def load_model(device: str = "cuda:0", dtype=torch.float32):
    """加载模型和 tokenizer。
    - dtype 默认 fp32：Pythia-70m 在 fp16 下前向会 NaN，必须 fp32
    - attn_implementation="eager"：用 PyTorch 原生 attention 而不是 SDPA/flash，
      这样可以通过 output_attentions=True 拿到 attention 概率（SnapKV 选 token 用）
    """
    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=dtype, attn_implementation="eager"
    ).to(device)
    model.eval()  # 关掉 dropout，进入推理模式
    return tok, model


def load_pg19_one_sample() -> str:
    """读取 pg-19 测试集第一本书的文本（约 62k tokens）。"""
    path = os.path.join(PROJECT_ROOT, "data", "pg19_one_sample.json")
    return json.load(open(path))["text"]


def load_wikitext_text() -> str:
    """读取 wikitext-2 测试集，把所有非空行拼成一段长文本。"""
    from datasets import load_dataset

    ds = load_dataset(
        "Salesforce/wikitext",
        "wikitext-2-raw-v1",
        cache_dir=os.path.join(PROJECT_ROOT, "data", "hf_cache"),
        split="test",
    )
    # 过滤空行后用 \n 拼接成一篇长文档
    return "\n".join(x["text"] for x in ds if x["text"].strip())


# --------------------------------------------------------------------------- #
# PPL：chunked streaming 评测
# 每个 chunk = prefix_len + eval_len。先喂 prefix 触发 prefill 和压缩，
# 再喂 eval 段用压缩后的 cache 算 NLL，最后所有 chunk 的 NLL 平均做 PPL。
# --------------------------------------------------------------------------- #
@torch.no_grad()
def streaming_ppl(
    model,
    ids: torch.Tensor,
    method: str,
    budget: int,
    prefix_len: int = 1024,
    eval_len: int = 1024,
    n_chunks: int = 4,
    window_size: int = 32,
) -> float:
    chunk_len = prefix_len + eval_len   # 单 chunk 总长度（用于切片）
    total_nll = 0.0                     # 累积负对数似然
    total_tokens = 0                    # 累积评测的 token 数
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype  # 用于构造 4D mask 的 -inf 值

    for chunk_idx in range(n_chunks):
        # 切出当前 chunk 的 token id（连续段）
        start = chunk_idx * chunk_len
        end = start + chunk_len
        if end > ids.shape[-1]:
            break   # 文本不够长就提前结束
        chunk = ids[:, start:end].to(device)
        prefix = chunk[:, :prefix_len]    # 前 1024：填进 cache 的部分
        eval_part = chunk[:, prefix_len:] # 后 1024：要算 PPL 的部分

        # 每个 chunk 重新建一个空 cache（不同 chunk 之间不共享历史）
        cache, post_fn = build_cache(method, budget, window_size=window_size)
        # SnapKV/PyramidKV 需要 attention 矩阵来打分选 token；
        # StreamingLLM 是纯位置规则不需要；Dense 没压缩函数
        need_attn = post_fn is not None and method != "streamingllm"

        # ---- Prefill：把 prefix 喂进去触发 cache 填充 ----
        out = model(
            prefix,
            past_key_values=cache,           # 把 cache 对象传进去，模型会往里 append K/V
            use_cache=True,                  # 必须开 cache 才会写入 past_key_values
            output_attentions=need_attn,     # 仅 SnapKV/PyramidKV 需要
        )
        # 执行压缩：StreamingLLM 不用 attentions，传 None 也行
        if post_fn is not None:
            post_fn(cache, out.attentions if need_attn else None)
        # 保留 prefix 最后一个 token 的 logit（它对应预测 eval 第 0 个 token）
        first_logit = out.logits[:, -1:, :].clone()
        del out
        torch.cuda.empty_cache()  # 释放 attentions 等中间结果，省显存

        # ---- Eval：用压缩后的 cache 算 NLL ----
        # 压缩后 cache 实际剩多少 token（物理列数）
        physical_kv_len = cache.key_cache[0].shape[-2]
        q_len = eval_part.size(-1)
        # 手写 4D mask：cache 列全可见，新 token 列因果三角
        attention_mask = make_4d_causal_mask(physical_kv_len, q_len, device, dtype)
        # 新 query 的 RoPE 位置 = 压缩后物理长度往后数
        cache_position = torch.arange(
            physical_kv_len, physical_kv_len + q_len, device=device
        )
        eval_out = model(
            eval_part,
            past_key_values=cache,
            use_cache=False,                 # 不往 cache 写入 eval 段 K/V（评测完即丢）
            attention_mask=attention_mask,   # 4D mask，绕开 transformers 的自动推断
            cache_position=cache_position,
        )
        # 拼 logit：first_logit (预测 eval[0]) + eval_out 前 q-1 个（预测 eval[1..q-1]）
        all_logits = torch.cat([first_logit, eval_out.logits[:, :-1, :]], dim=1)

        # 算 NLL（reduction="sum" 累加所有 token 的 loss）
        loss = F.cross_entropy(
            all_logits.reshape(-1, all_logits.size(-1)),
            eval_part.reshape(-1),
            reduction="sum",
        )
        total_nll += loss.item()
        total_tokens += eval_part.numel()

    if total_tokens == 0:
        return float("nan")
    # PPL = exp(平均 NLL)
    return float(torch.tensor(total_nll / total_tokens).exp())


# --------------------------------------------------------------------------- #
# 延迟 + 峰值显存：prefill 长 prompt → 用压缩 cache 生成 N 个 token，
# 测 TTFT（首 token 时延）、TPOT（每 token 时延）、KV 显存。
# --------------------------------------------------------------------------- #
@torch.no_grad()
def latency_memory(
    model,
    ids: torch.Tensor,
    method: str,
    budget: int,
    prefix_len: int = 1024,
    gen_tokens: int = 64,
    window_size: int = 32,
):
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    prefix = ids[:, :prefix_len].to(device)

    cache, post_fn = build_cache(method, budget, window_size=window_size)
    need_attn = post_fn is not None and method != "streamingllm"

    # 先清空显存，再 warmup 暖完直接测，中间不再调 empty_cache（不然 cudaMalloc 又得重做）
    torch.cuda.empty_cache()

    # ---- Warmup：跑两次"完整流程" (prefill + 压缩 + 8 步 decode) ----
    # 之前只 warmup prefill 不够 —— decode 是单 token forward + 不同 K 长度，
    # 首次遇到某个 K 长度范围会触发 kernel JIT 编译，导致 TPOT 偏高 ~0.5ms。
    # 用相同的 (method, budget) 跑完整流程，让 decode 路径上的 kernel 也热好。
    for _ in range(2):
        wu_cache, wu_post = build_cache(method, budget, window_size=window_size)
        wu_out = model(prefix, past_key_values=wu_cache, use_cache=True,
                       output_attentions=need_attn)
        if wu_post is not None:
            wu_post(wu_cache, wu_out.attentions if need_attn else None)
        wu_pkv = wu_cache.key_cache[0].shape[-2]
        for _ in range(8):
            wu_id = wu_out.logits[:, -1:, :].argmax(-1)
            wu_mask = make_4d_causal_mask(wu_pkv, 1, device, dtype)
            wu_pos = torch.tensor([wu_pkv], device=device)
            wu_out = model(wu_id, past_key_values=wu_cache, use_cache=True,
                           attention_mask=wu_mask, cache_position=wu_pos)
            wu_pkv += 1
        del wu_cache, wu_out
    torch.cuda.synchronize()

    # 让 max_memory_allocated 只反映这次测量（不清显存块，仅重置计数器）
    torch.cuda.reset_peak_memory_stats(device)

    # ---- Prefill 计时 ----
    torch.cuda.synchronize()  # 确保前面的 GPU 操作都做完
    t0 = time.time()
    out = model(
        prefix,
        past_key_values=cache,
        use_cache=True,
        output_attentions=need_attn,
    )
    if post_fn is not None:
        post_fn(cache, out.attentions if need_attn else None)
    torch.cuda.synchronize()  # 等 GPU 真的算完
    prefill_time = time.time() - t0

    # 压缩之后实际剩下的 KV 长度（物理）
    physical_kv_len = cache.key_cache[0].shape[-2]
    cached_after_compress = physical_kv_len

    # 在 decode 开始前测 KV 占用，避免被 decode 期间新增的 token 污染
    kv_bytes_per_layer = (
        cache.key_cache[0].numel() + cache.value_cache[0].numel()
    ) * cache.key_cache[0].element_size()
    kv_total_mb = kv_bytes_per_layer * len(cache.key_cache) / 1e6

    # ---- 生成第一个 token（计入 TTFT）----
    torch.cuda.synchronize()
    t1 = time.time()
    # prefill 最后一个 token 的 logit → argmax 出预测的下一个 token id
    next_id = out.logits[:, -1:, :].argmax(-1)
    # 单 token forward 也需要手写 mask（压缩后物理长度和 _seen_tokens 不一致）
    attention_mask = make_4d_causal_mask(physical_kv_len, 1, device, dtype)
    cache_position = torch.tensor([physical_kv_len], device=device)
    out = model(
        next_id,
        past_key_values=cache,
        use_cache=True,                   # 这次要把新 K/V 写回 cache
        attention_mask=attention_mask,
        cache_position=cache_position,
    )
    torch.cuda.synchronize()
    first_token_time = time.time() - t1
    # TTFT = 用户按回车 → 看到第一个 token 的总时间 = prefill + 生成第一个 token
    ttft = prefill_time + first_token_time

    # ---- Decode 循环：测剩下 gen_tokens-1 个 token 的延迟 ----
    # 跑 gen_tokens 步，只取后半段稳态期算 TPOT。原因：dense/streamingllm prefill
    # 较轻 → 进入 decode 时 GPU 时钟未达 max boost；前半段 decode 还在 ramping，
    # 取后半段才反映真实 steady-state 性能（这是 GPU 微基准的标准做法）。
    decode_times = []
    for _ in range(gen_tokens - 1):
        next_id = out.logits[:, -1:, :].argmax(-1)
        # cache 每步会变长（因为 use_cache=True），mask 也要跟着重建
        physical_kv_len = cache.key_cache[0].shape[-2]
        attention_mask = make_4d_causal_mask(physical_kv_len, 1, device, dtype)
        cache_position = torch.tensor([physical_kv_len], device=device)
        torch.cuda.synchronize()
        t = time.time()
        out = model(
            next_id,
            past_key_values=cache,
            use_cache=True,
            attention_mask=attention_mask,
            cache_position=cache_position,
        )
        torch.cuda.synchronize()
        decode_times.append(time.time() - t)
    # 丢弃前半段（GPU 时钟 ramping 期），后半段排序取 median
    steady_state = sorted(decode_times[len(decode_times) // 2:])
    tpot = steady_state[len(steady_state) // 2] if steady_state else 0.0

    # 整个 latency 测量过程中的峰值显存（含模型权重 + cache + 中间激活）
    peak_mem_mb = torch.cuda.max_memory_allocated(device) / 1e6

    return {
        "ttft_ms": ttft * 1000,
        "prefill_ms": prefill_time * 1000,
        "tpot_ms": tpot * 1000,
        "throughput_tok_per_s": 1.0 / max(tpot, 1e-9),
        "peak_mem_mb": peak_mem_mb,
        "kv_cache_mb": kv_total_mb,                 # 已经在 decode 前测好
        "cached_tokens": cached_after_compress,
    }


def run_one(
    model,
    tok,
    text: str,
    dataset_name: str,
    method: str,
    budget: int,
    window_size: int = 32,
    ppl_chunks: int = 3,
    prefix_len: int = 1024,
    eval_len: int = 1024,
    gen_tokens: int = 64,
):
    """跑一组配置（method × budget × dataset），返回所有指标的字典。"""
    # tokenize 整段文本（不截断）
    ids = tok(text, return_tensors="pt").input_ids
    ppl = streaming_ppl(
        model, ids, method, budget,
        prefix_len=prefix_len,
        eval_len=eval_len,
        n_chunks=ppl_chunks,
        window_size=window_size,
    )
    lm = latency_memory(
        model, ids, method, budget,
        prefix_len=prefix_len,
        gen_tokens=gen_tokens,
        window_size=window_size,
    )
    # 把 PPL 和 latency/memory 指标合并成一个 flat dict
    return {
        "dataset": dataset_name,
        "method": method,
        "budget": budget,
        "window_size": window_size,
        "ppl": ppl,
        **lm,
    }
