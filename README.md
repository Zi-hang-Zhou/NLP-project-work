# Pythia-70M KV Cache Compression

LLM 推理大作业（个人部分）：在 Pythia-70M 上复现 KV 缓存压缩算法，无需训练，
在 PG-19 + WikiText-2 上评估 PPL、TTFT、TPOT、显存。

## 方法

实现 4 条 baseline，全部以 prefill 之后的后处理形式作用在标准 `DynamicCache` 上：

| 方法 | 选什么 | 依赖 |
| --- | --- | --- |
| **Dense** | 不压缩，KV 全留 | — |
| **StreamingLLM** | 头部 `n_sink=4` + 末尾 `window` 个 token（按位置） | 无需 attention |
| **SnapKV** | 用末尾 32-token 观察窗对前文计算 attention 求和，1D 平均池化后 top-K（按重要度） | 需要 `output_attentions=True` |
| **PyramidKV** | 同 SnapKV 选法，但浅层预算大、深层预算小（线性递减，`beta=20`） | 同上 |

文件分工：

- [code/caches.py](code/caches.py)：三种压缩函数 + `make_4d_causal_mask`
- [code/eval.py](code/eval.py)：PPL（chunked streaming）+ 延迟/显存测量
- [code/run_all.py](code/run_all.py)：5 × 4 × 2 全表实验，断点续跑
- [code/plot.py](code/plot.py)：绘图
- [code/smoke.py](code/smoke.py)：4 方法回归测试

## 环境

- Pythia-70M（6 层 / 8 头 / hidden 512 / head_dim 64）
- transformers + PyTorch + matplotlib + datasets，`attn_implementation="eager"`（SnapKV 需要 attention 矩阵）
- 模型用 fp32（fp16 下 Pythia-70M logits 容易 NaN）
- 单卡 GPU，prefix 1024、eval 1024、PPL 取每条样本前 3 个 chunk

数据：

- **PG-19**：[data/pg19_one_sample.json](data/pg19_one_sample.json) 已含测试集第一本书（约 62k tokens，与 deepmind/pg19 官方 test split 字节一致）
- **WikiText-2**：`Salesforce/wikitext` 的 `wikitext-2-raw-v1` test split，由 `datasets` 自动下载到 `data/hf_cache/`（已 gitignore）

## 复现命令

```bash
pip install torch transformers datasets matplotlib

# 模型路径：默认看 models/pythia-70m（本地），找不到就自动从 HF Hub 下载 EleutherAI/pythia-70m
# 也可以显式指定：export MODEL_PATH=/path/to/pythia-70m

cd code
python smoke.py         # ~30s，4 个方法各跑一次 sanity check
python run_all.py       # 主表实验，输出到 ../results/all.jsonl（断点续跑）
python latency_scan.py  # 长上下文加速扫描，输出到 ../results/latency_scan.jsonl
python plot.py          # 7 张图输出到 ../figures/
```

## 结果

完整数字见 [results/all.jsonl](results/all.jsonl)。下面是 PG-19 上的关键对比：

| 方法 | budget | PPL | KV cache (MB) | TTFT (ms) | TPOT (ms) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Dense | 1024 | 29.27 | 25.17 | 7.9 | 2.77 |
| StreamingLLM | 128 | 729.00 | 3.15 | 8.0 | 2.78 |
| StreamingLLM | 512 | 122.10 | 12.58 | 8.1 | 2.72 |
| StreamingLLM | 1024 | 29.27 | 25.17 | 7.9 | 2.73 |
| SnapKV | 128 | 742.02 | 3.15 | 8.2 | 2.73 |
| SnapKV | 512 | 123.94 | 12.58 | 8.4 | 2.71 |
| SnapKV | 1024 | 29.27 | 25.17 | 7.9 | 2.73 |
| PyramidKV | 128 | 511.93 | 4.37 | 8.4 | 2.75 |
| PyramidKV | 512 | 104.84 | 13.81 | 8.4 | 2.74 |
| PyramidKV | 1024 | 29.31 | 25.17 | 8.3 | 2.72 |

WikiText-2 趋势完全一致（PPL 基线 51.8，相同排序）。

### 关键观察

1. **PyramidKV 按 budget 比时全档位领先。** 在 PG-19 上，budget=128 时 PPL 比 SnapKV/StreamingLLM 降 31%（512 vs 729/742），budget=512 时降 15%（105 vs 124/122）。给浅层多留 KV 在小模型上确实有用。但注意 PyramidKV 因为各层长度不齐要 padding，**同 budget 下实际显存比 SnapKV 大 39%**（b=128 时 4.37 vs 3.15 MB），按"相同显存"对比时优势收窄——见 [tradeoff_pg19.png](figures/tradeoff_pg19.png)，三种方法的帕累托前沿其实非常接近。
2. **SnapKV ≈ StreamingLLM。** Pythia-70M 太小（6 层 8 头），attention 选择带来的额外信号优势不明显，attention-based 在 budget 大时甚至略输给纯位置规则；在 WikiText-2 中等 budget 上 SnapKV 才反超。
3. **PPL 随 budget 单调下降，**budget=1024 时三种压缩与 Dense 完全一致（验证流程正确）。
4. **prefix=1024 时 TPOT 看不出差距，但长上下文下加速非常明显。** 主表里所有方法 × 所有 budget 的 TPOT 都落在 2.71-2.79ms（差异 ≤0.08ms，在测量噪声内）——因为 prefix=1024 时 attention 在 6 层 8 头小模型上只占 TPOT 的一小部分，GPU 时间被 layernorm/MLP/kernel launch 主导。但这不是"KV 压缩没用"——见下面[补充实验](#补充实验合成-cache-下的真实加速)，cache 长到 32k+ 后 attention 变 memory-bound，Dense 的 TPOT 飙升到 11.84ms（cache=128k），此时压到 budget=128 后 TPOT 回到 2.87ms，**4.12× 加速**。
5. **KV 显存随 budget 线性下降。** budget=128 vs 1024 时压缩到 8× 小（3.15 vs 25.17 MB）。

### 图

- [figures/ppl_pg19.png](figures/ppl_pg19.png)：PG-19 PPL vs budget
- [figures/ppl_wikitext2.png](figures/ppl_wikitext2.png)：WikiText-2 PPL vs budget
- [figures/tradeoff_pg19.png](figures/tradeoff_pg19.png)：PPL vs KV size 帕累托
- [figures/tradeoff_wikitext2.png](figures/tradeoff_wikitext2.png)：同上
- [figures/kvsize.png](figures/kvsize.png)：KV 显存随 budget
- [figures/latency.png](figures/latency.png)：TTFT/TPOT 随 budget
- [figures/latency_scan.png](figures/latency_scan.png)：长上下文 cache 扫描（补充实验，见下）

## 补充实验：合成 cache 下的真实加速

**动机。** [run_all.py](code/run_all.py) 的 PPL 实验用 prefix=1024，TPOT 看不出方法差异——
不是因为压缩没用，是因为 6 层 8 头的小模型上 attention 计算在 prefix=1024 时只占
TPOT 的一小部分，剩下被 layernorm + MLP + kernel launch + Python overhead 占据。
**KV 压缩的真实价值在长上下文**：cache 越大，每步 decode 要读的 K/V 越多，
attention 变 memory-bound，压缩才能 GPU 时间。

**做法。** [code/latency_scan.py](code/latency_scan.py) 跳过 prefill，直接合成
不同长度的 `DynamicCache`（K/V 用随机数 ×0.1 填充），跑 decode 测 TPOT。
诚实声明：cache 内容随机 + 位置编码超出 Pythia 训练长度 → **模型输出毫无意义**；
但 GEMM、读 K/V 的 memory bandwidth 完全真实——TPOT 就是"如果 cache 这么大，
GPU 上 decode 一个 token 要多久"。

| cache length | TPOT (ms) | KV size (MB, fp32) |
| ---: | ---: | ---: |
| 128 | 2.87 | 3.15 |
| 1024 | 2.85 | 25.17 |
| 8192 | 2.86 | 201.33 |
| 16384 | 2.88 | 402.65 |
| **32768** | **3.52** | 805.31 |
| **65536** | **6.28** | 1610.61 |
| **131072** | **11.85** | 3221.23 |

完整数据见 [results/latency_scan.jsonl](results/latency_scan.jsonl)，
图见 [figures/latency_scan.png](figures/latency_scan.png)。

**结论。** Dense 曲线在 ≤16k 时几乎平的（attention 计算还埋在其他开销里），
**过 32k 开始陡升**——L2 cache 装不下 + memory bandwidth 触顶。
压缩到 budget=128 后 TPOT 恒为 2.87ms，于是：

- cache 32768 → 128：**1.23× faster**（3.52 → 2.87ms）
- cache 65536 → 128：**2.19× faster**（6.28 → 2.87ms）
- cache 131072 → 128：**4.12× faster**（11.85 → 2.87ms）

这模拟的工业场景：用户给 32k-128k tokens 的长 prompt（代码库摘要、长文档问答），
模型 decode 时每步都要扫完整 KV cache，**这种规模下压缩对 decode 延迟有显著加速**。
在 prefix=1024 的标准 benchmark 上看不到加速，不等于压缩在生产场景下没用。

## 实现细节

### 1. 压缩后位置 ID 与 4D 注意力 mask

`DynamicCache` 在压缩之后，物理 K 长度（比如 512）远小于逻辑位置（1024）。
若直接把 logits 当做 logical position 喂回去（用 `cache._seen_tokens` 作 `cache_position`，
让 transformers 自动 expand 2D mask），
`_prepare_4d_causal_attention_mask_with_cache_position` 内的因果判断会失效，
所有 query 看到的 key 都被认为"未来"，causal mask 整体失活，
PPL 跌成一坨 random（任何压缩策略测出来都是 ~327）。

解决：手工构造形状 `(1, 1, q_len, physical + q_len)` 的 4D mask，
其中 cache 列全部为 0（可见），新增 K 列为标准因果三角；
`cache_position = arange(physical, physical + q_len)`，
即让新 query 的 RoPE 与物理 K 长度对齐。

### 2. PyramidKV 各层长度不一致 + 全局 mask

PyramidKV 给不同层不同 budget，但 model 只接受一个全局 mask，K 长度必须一致。
试过两种 padding：

- 零填充：因为 `Q · 0 = 0`，当真实 K 给出负 attention score 时，padding 列反而抢走大量
  softmax 质量（即便 V=0 不污染输出，real token 的相对权重被严重稀释）。
  PPL 从期望的 ~365 升到 1026。
- **复用首个保留 K/V 填充**（当前做法）：padding 列拿到的是真实 score，
  偏向只是把第一个保留 token（通常类似 sink）多复制几份，对输出影响很小。
  实测 PyramidKV 降回 365 并跑赢 SnapKV。

### 3. fp16 NaN

Pythia-70M fp16 直接前向就会 NaN，全程用 fp32。

### 4. PPL 评测时 `use_cache=False`

eval forward 不再 append 新 K（避免把新 token 的 K/V 写入 cache 影响后续 chunk）。
经 verification（`keep=all` PPL 与 dense 一致）确认 `past_key_values` 在
`use_cache=False` 下仍会被消费。

### 5. TPOT 测量需要"稳态采样"，不能直接取均值

第一版实现里 TPOT = `mean(decode_times)`，结果 Dense / StreamingLLM 系列读数 3.2ms、
SnapKV / PyramidKV 系列读数 2.8ms，**明显偏向于 attention-heavy 的方法**。原因不是
算法快慢，是 GPU 状态：

- snapkv/pyramidkv prefill 要分配 attention 概率张量，把 GPU 时钟拉到 max boost，
  进入 decode 时还在峰值；dense/streamingllm prefill 轻量，GPU 时钟没拉满
- 每个 (method, budget) 第一次见到的 cache 长度范围还需要 JIT 编译对应 kernel
- 偶发的"慢迭代"会拉高均值

修复三件套：
1. **每个测量做完整 warmup**（prefill + 压缩 + 8 步 decode，使用同 method/budget）
   —— 编译好 kernel，避免冷启动偏差
2. **gen_tokens=128，只取后 64 步的 median 作为 TPOT** —— 前半段当 GPU 时钟 ramping
   期丢弃，median 还能抗个别慢迭代
3. **全局 warmup 跑两次完整 run_one** —— 让整个流程的 dispatcher cache 都热好

修完之后 TPOT 全部落在 2.71-2.79ms（差异 ≤0.08ms），符合"小模型上 KV 压缩对 decode
速度无明显影响"的物理预期。

