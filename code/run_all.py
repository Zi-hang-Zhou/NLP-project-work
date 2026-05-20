"""全表实验驱动脚本：4 方法 × 5 budget × 2 数据集 = 32 组实验。

每跑完一组就 append 一行 JSON 到 results/all.jsonl，
中途挂了可以重新运行：已经跑过的 (dataset, method, budget) 会被跳过（断点续跑）。
"""
from __future__ import annotations
import json
import os
import sys
import time

# 让 import eval 能找到同目录下的 eval.py（独立运行时 sys.path 默认不含脚本目录）
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
_ROOT = os.path.dirname(_HERE)

from eval import (
    load_model,
    load_pg19_one_sample,
    load_wikitext_text,
    run_one,
)


# ---- 全局实验配置 ----
OUT_PATH = os.path.join(_ROOT, "results", "all.jsonl")     # 结果写到这里
METHODS = ["dense", "streamingllm", "snapkv", "pyramidkv"]   # 4 个方法
BUDGETS = [128, 256, 512, 768, 1024]                         # 压缩方法的 5 个预算
WINDOW_SIZE = 32     # SnapKV/PyramidKV 的观察窗口大小（也是 StreamingLLM 的 recent 长度上限）
PREFIX_LEN = 1024    # 每 chunk 的 prefix 长度（被压缩的部分）
EVAL_LEN = 1024      # 每 chunk 的 eval 长度（算 PPL 的部分）
PPL_CHUNKS = 3       # 算 PPL 时每条样本用前几个 chunk（共 6144 tokens 评测）
GEN_TOKENS = 128     # latency 测试生成多少 token（前 64 当稳态期 warmup 丢弃，后 64 算 TPOT）


def main():
    # 确保结果目录存在（首次跑会自动创建）
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    # 加载模型一次，所有实验共用
    tok, model = load_model()

    # ---- 全局 warmup：跑两次完整 run_one 让所有路径都热好 ----
    # 单跑几次 model.forward 不够（dispatcher 缓存 + GPU 时钟 ramping 都需要时间）。
    # 跑两次完整流程（每次约 0.6s）让 GPU 时钟稳定在 max boost，结果丢弃。
    print("warmup...")
    for _ in range(2):
        _ = run_one(model, tok, "the " * 4000, "warmup", "streamingllm", 512,
                    window_size=WINDOW_SIZE, ppl_chunks=1,
                    prefix_len=PREFIX_LEN, eval_len=EVAL_LEN, gen_tokens=GEN_TOKENS)
    print("warmup done")

    # 准备两个数据集的原始文本
    datasets = {
        "pg19": load_pg19_one_sample(),
        "wikitext2": load_wikitext_text(),
    }

    # ---- 断点续跑：先读现有结果，构造"已完成 key 集合" ----
    done = set()
    if os.path.exists(OUT_PATH):
        for line in open(OUT_PATH):
            r = json.loads(line)
            done.add((r["dataset"], r["method"], r["budget"]))

    # ---- 主循环：dataset × method × budget ----
    with open(OUT_PATH, "a") as f:   # 追加模式打开
        for ds_name, text in datasets.items():
            for method in METHODS:
                # Dense 没压缩，budget 没意义，用 PREFIX_LEN 当占位（结果会和 budget=1024 一致）
                budgets = [PREFIX_LEN] if method == "dense" else BUDGETS
                for budget in budgets:
                    key = (ds_name, method, budget)
                    # 已经跑过就跳过（断点续跑）
                    if key in done:
                        print(f"skip {key}")
                        continue
                    # 计时整个组（含 PPL + latency）
                    t0 = time.time()
                    r = run_one(
                        model, tok, text, ds_name, method, budget,
                        window_size=WINDOW_SIZE,
                        ppl_chunks=PPL_CHUNKS,
                        prefix_len=PREFIX_LEN,
                        eval_len=EVAL_LEN,
                        gen_tokens=GEN_TOKENS,
                    )
                    dt = time.time() - t0
                    # 写一行 JSON 到文件，立即 flush（防止挂掉丢数据）
                    f.write(json.dumps(r) + "\n")
                    f.flush()
                    # 控制台打印一行汇总，方便边跑边看
                    print(
                        f"[{dt:5.1f}s] {ds_name:10s} {method:13s} "
                        f"b={budget:4d}  ppl={r['ppl']:8.3f}  "
                        f"ttft={r['ttft_ms']:6.1f}ms  tpot={r['tpot_ms']:5.2f}ms  "
                        f"kv={r['kv_cache_mb']:.2f}MB"
                    )


if __name__ == "__main__":
    main()
