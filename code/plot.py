"""从 results/all.jsonl 生成 6 张图。

读取所有实验结果后，画 4 类图：
1. PPL vs budget（PG-19 / WikiText-2 各一张）
2. PPL vs KV size 帕累托图（PG-19 / WikiText-2 各一张）
3. KV cache size vs budget
4. TTFT / TPOT vs budget（双子图合一张）
"""
import json
import os
import matplotlib.pyplot as plt

# 数据源（实验结果 jsonl）和图输出目录（相对项目根，code/ 的父目录）
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(_ROOT, "results", "all.jsonl")
FIG_DIR = os.path.join(_ROOT, "figures")
os.makedirs(FIG_DIR, exist_ok=True)  # 没有就建（已存在也不报错）

# 一次性把 jsonl 全部读进内存（每行一个 dict）
# 32 行数据，内存几乎为 0，没必要流式
rows = [json.loads(l) for l in open(RES)]


def by(method, dataset, key):
    """筛选 + 排序：返回某 (method, dataset) 下按 budget 升序的 (xs, ys)。

    xs = [128, 256, 512, 768, 1024]（budget 列表）
    ys = 对应的 key 值列表（例如 PPL、kv_cache_mb 等）
    给画折线图用，x 已经排好序。
    """
    items = [r for r in rows if r["method"] == method and r["dataset"] == dataset]
    items.sort(key=lambda r: r["budget"])  # 按 budget 升序，避免折线乱跳
    return [r["budget"] for r in items], [r[key] for r in items]


# 4 种方法的统一样式：颜色 + marker，所有图复用，保持视觉一致
METHODS = ["dense", "streamingllm", "snapkv", "pyramidkv"]
COLORS = {"dense": "#444", "streamingllm": "#1f77b4",
          "snapkv": "#2ca02c", "pyramidkv": "#d62728"}
MARKERS = {"dense": "s", "streamingllm": "o", "snapkv": "^", "pyramidkv": "D"}


def plot_ppl_vs_budget(dataset):
    """图 1/2：PPL vs KV budget（每个数据集一张）。

    Dense 没 budget 概念，画成一条横虚线作参考；其余三种方法画折线。
    y 轴用 log 是因为低 budget 时 PPL 飙到 700+，线性轴会把所有差异挤扁。
    """
    plt.figure(figsize=(6, 4))
    # Dense 在 rows 里只有一条记录（budget=1024 占位），取出它的 PPL 当 baseline
    dense_ppl = [r["ppl"] for r in rows if r["dataset"] == dataset and r["method"] == "dense"][0]
    # axhline = 水平参考线（贯穿整张图的 x 范围）
    plt.axhline(dense_ppl, color=COLORS["dense"], linestyle="--",
                label=f"Dense (PPL={dense_ppl:.2f})", linewidth=1.2)
    # 三种压缩方法画折线
    for method in METHODS:
        if method == "dense":
            continue  # Dense 已用横线画过
        x, y = by(method, dataset, "ppl")
        plt.plot(x, y, marker=MARKERS[method], color=COLORS[method],
                 label=method, linewidth=1.5)
    plt.xlabel("KV budget (tokens)")
    plt.ylabel("Perplexity")
    plt.title(f"PPL vs KV budget — {dataset}")
    plt.yscale("log")                         # y 轴对数（处理跨数量级差异）
    plt.grid(alpha=0.3, which="both")         # 主次刻度都画灰格
    plt.legend()
    plt.tight_layout()                        # 自动调边距，防止标签被裁
    out = f"{FIG_DIR}/ppl_{dataset}.png"
    plt.savefig(out, dpi=140)
    plt.close()                               # 关图防止下一张图叠加
    print(f"saved {out}")


def plot_kvsize_vs_budget():
    """图 5：KV cache 显存 vs budget（只画 PG-19，WikiText 趋势完全一样）。

    展示三种压缩方法在不同 budget 下的实际显存占用，验证"线性下降"。
    PyramidKV 因为有 padding，显存比 SnapKV 略高一点点。
    """
    plt.figure(figsize=(6, 4))
    for method in METHODS:
        if method == "dense":
            continue  # Dense 也没必要画（一个点没意义）
        x, y = by(method, "pg19", "kv_cache_mb")
        plt.plot(x, y, marker=MARKERS[method], color=COLORS[method], label=method)
    plt.xlabel("KV budget (tokens)")
    plt.ylabel("KV cache size (MB, fp32)")
    plt.title("KV cache size after compression")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out = f"{FIG_DIR}/kvsize.png"
    plt.savefig(out, dpi=140)
    plt.close()
    print(f"saved {out}")


def plot_latency_vs_budget():
    """图 6：TTFT / TPOT vs budget（左右双子图合并保存）。

    Pythia-70M 太小，所有方法 TTFT 都在 8-11ms、TPOT 在 2.7-3.2ms，差距不明显——
    本图主要是说明"压缩在小模型上没看到明显加速"这个结论。
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))  # 1 行 2 列子图
    for method in METHODS:
        if method == "dense":
            continue
        x, ttft = by(method, "pg19", "ttft_ms")
        _, tpot = by(method, "pg19", "tpot_ms")      # 共享同一组 budget x
        axes[0].plot(x, ttft, marker=MARKERS[method], color=COLORS[method], label=method)
        axes[1].plot(x, tpot, marker=MARKERS[method], color=COLORS[method], label=method)
    # Dense 在两个子图都画横线作参考
    dense_row = [r for r in rows if r["dataset"] == "pg19" and r["method"] == "dense"][0]
    axes[0].axhline(dense_row["ttft_ms"], color=COLORS["dense"], linestyle="--",
                    label=f"Dense ({dense_row['ttft_ms']:.1f}ms)")
    axes[1].axhline(dense_row["tpot_ms"], color=COLORS["dense"], linestyle="--",
                    label=f"Dense ({dense_row['tpot_ms']:.2f}ms)")
    # 子图 0：TTFT
    axes[0].set_xlabel("KV budget"); axes[0].set_ylabel("TTFT (ms)")
    axes[0].set_title("Time to first token")
    axes[0].grid(alpha=0.3); axes[0].legend()
    # 子图 1：TPOT
    axes[1].set_xlabel("KV budget"); axes[1].set_ylabel("TPOT (ms / token)")
    axes[1].set_title("Decode latency")
    axes[1].grid(alpha=0.3); axes[1].legend()
    plt.tight_layout()
    out = f"{FIG_DIR}/latency.png"
    plt.savefig(out, dpi=140)
    plt.close()
    print(f"saved {out}")


def plot_ppl_vs_kvsize(dataset):
    """图 3/4：PPL vs KV size 帕累托权衡图（每个数据集一张）。

    x = 真实 KV 显存（MB），y = PPL。左下角越接近越好（小显存 + 低 PPL）。
    Dense 是右上角的"完美质量但最贵"点；压缩方法画折线展示帕累托前沿。
    这张图比 plot_ppl_vs_budget 更能说明"哪种方法在显存预算下质量最好"。
    """
    plt.figure(figsize=(6, 4))
    # Dense 单点用 scatter 画方块（不画折线，因为它只有一个点）
    dense_row = [r for r in rows if r["dataset"] == dataset and r["method"] == "dense"][0]
    plt.scatter([dense_row["kv_cache_mb"]], [dense_row["ppl"]],
                color=COLORS["dense"], marker="s", s=70, label="Dense", zorder=5)
    # 三种压缩方法：x = 显存（不是 budget），y = PPL
    for method in METHODS:
        if method == "dense":
            continue
        items = [r for r in rows if r["dataset"] == dataset and r["method"] == method]
        items.sort(key=lambda r: r["budget"])   # 按 budget 排让折线从左到右
        x = [r["kv_cache_mb"] for r in items]
        y = [r["ppl"] for r in items]
        plt.plot(x, y, marker=MARKERS[method], color=COLORS[method], label=method)
    plt.xlabel("KV cache size (MB)")
    plt.ylabel("Perplexity")
    plt.title(f"Quality vs memory trade-off — {dataset}")
    plt.yscale("log")
    plt.grid(alpha=0.3, which="both")
    plt.legend()
    plt.tight_layout()
    out = f"{FIG_DIR}/tradeoff_{dataset}.png"
    plt.savefig(out, dpi=140)
    plt.close()
    print(f"saved {out}")


def plot_latency_scan():
    """图 7：长上下文 cache 扫描 — 体现 KV 压缩的真实加速优势。

    数据来自 latency_scan.py：合成不同长度的 KV cache 跑 decode 测 TPOT。
    PG-19/WikiText 实验里 prefix=1024 看不出加速，是因为 attention 计算在 6 层
    8 头的小模型上占总时间比例太低；当 cache 长到 32k+ 后 attention 内存读取
    成为瓶颈，曲线开始陡升 —— 此时压缩才显示出明显加速。
    """
    scan_path = os.path.join(_ROOT, "results", "latency_scan.jsonl")
    if not os.path.exists(scan_path):
        print(f"skip plot_latency_scan: {scan_path} not found")
        return
    scan = [json.loads(l) for l in open(scan_path)]
    scan.sort(key=lambda r: r["cache_size"])
    xs = [r["cache_size"] for r in scan]
    ys = [r["tpot_ms"] for r in scan]

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    # Dense 曲线：随 cache 增长，TPOT 在 16k 后陡升
    ax.plot(xs, ys, marker="o", color="#444",
            label="Dense (cache=N, no compression)", linewidth=1.8)
    # 标几个关键点的 TPOT 值
    for r in scan:
        if r["cache_size"] in (1024, 32768, 65536, 131072):
            ax.annotate(f"{r['tpot_ms']:.2f}ms",
                        (r["cache_size"], r["tpot_ms"]),
                        textcoords="offset points", xytext=(6, 6),
                        fontsize=8.5, color="#444")

    # 压缩后的两条水平参考线
    by_size = {r["cache_size"]: r["tpot_ms"] for r in scan}
    if 128 in by_size:
        ax.axhline(by_size[128], color=COLORS["snapkv"], linestyle="--",
                   linewidth=1.4,
                   label=f"Compressed to budget=128 ({by_size[128]:.2f}ms)")
    if 1024 in by_size:
        ax.axhline(by_size[1024], color=COLORS["pyramidkv"], linestyle="--",
                   linewidth=1.4,
                   label=f"Compressed to budget=1024 ({by_size[1024]:.2f}ms)")

    # 标注最大加速比
    if 131072 in by_size and 128 in by_size:
        speedup = by_size[131072] / by_size[128]
        ax.annotate(f"{speedup:.2f}× faster\n(128k → 128)",
                    xy=(131072, by_size[131072]),
                    xytext=(40000, by_size[131072] - 1.2),
                    fontsize=10, color="#d62728",
                    arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.2))

    ax.set_xscale("log", base=2)
    ax.set_xlabel("KV cache length (tokens)")
    ax.set_ylabel("TPOT (ms / token)")
    ax.set_title("Decode latency vs KV cache length — Pythia-70M\n"
                 "(synthetic cache, no PPL — shows pure GPU cost of KV read)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="upper left", fontsize=9)
    plt.tight_layout()
    out = f"{FIG_DIR}/latency_scan.png"
    plt.savefig(out, dpi=140)
    plt.close()
    print(f"saved {out}")


if __name__ == "__main__":
    # 两个数据集各画两张图（PPL-vs-budget + tradeoff）
    for ds in ["pg19", "wikitext2"]:
        plot_ppl_vs_budget(ds)
        plot_ppl_vs_kvsize(ds)
    # 单图：KV 显存和延迟（PG-19 数据代表，WikiText 趋势一致）
    plot_kvsize_vs_budget()
    plot_latency_vs_budget()
    # 长上下文 cache 扫描（独立实验，体现压缩的真实加速）
    plot_latency_scan()
