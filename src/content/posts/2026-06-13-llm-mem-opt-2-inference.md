---
title: LLM 训练与推理优化（二）— 推理显存拆解与优化
published: 2026-06-13
description: 以 Llama-3-8B 为例，逐项推导推理时的权重和 KV Cache。讲清楚 KV Cache 公式里为什么是 n_kv 而不是 n_heads，以及 FlashAttention 如何避免把注意力矩阵常驻 HBM。
category: Deep Learning
tags:
- llm
- memory
- llm-memory-optimization-series
- inference
- kv-cache
- flash-attention
draft: false
---

> 这是「LLM 训练与推理优化系列」第 2 篇。如果你还没读过 [第 1 篇 — 显存基础](/posts/2026-06-12-llm-mem-opt-1-fundamentals/)，建议先扫一眼里面「推理三件套」那张图，本篇就是把这三件套逐项拆开。

## 0. 4090 + Llama-3-8B：先算确定账，再测峰值

你有一张 RTX 4090，24 GB 显存，想跑 Llama-3-8B。

第一篇里我们算过：8B 参数 × 2 字节（BF16）= 16 GB。还剩 8 GB——「够了，跑起来！」

仅凭“8B × BF16 = 16 GB”不能判断能否运行。权重和已存 KV Cache 可以精确核算；预填充/解码的临时工作区则随 kernel、请求长度、批量、并发、chunk 设置和框架版本变化，必须在目标引擎上测峰值。因此正确顺序是：先用确定账排除明显不可能的组合，再用真实请求压测。

## 1. 模型参数：第一笔账

### 1.1 公式

第 1 篇已经给过：

$$
M_{\text{params}} = N_{\text{params}} \times b_{\text{param}}
$$

按标称 8B 参数计算，BF16 裸权重约 16 GB，INT8 裸权重约 8 GB，INT4 编码本体约 4 GB。INT4 的 4 GB 是不含 scale、zero-point、group 元数据、对齐和运行时工作区的理论下界。

### 1.2 但「8B 参数」到底是怎么组成的？

光知道结果不够，我们拆一下 Llama-3-8B 的结构，把这 8B 的来源算出来。这对后面理解 KV Cache 和推理工作区都有帮助。

Llama-3-8B 的结构超参：

| 参数 | 符号 | 值 |
|-----|------|----|
| 层数 | $L$ | 32 |
| 注意力 query 头数 | $n_h$ | 32 |
| 注意力 KV 头数 | $n_{kv}$ | 8 (GQA) |
| 隐藏维度 | $d_{\text{model}}$ | 4096 |
| 每头维度 | $d_{\text{head}}$ | 128 |
| FFN 中间维度 | $d_{\text{ff}}$ | 14336 |
| 词表大小 | $V$ | 128256 |

> Llama-3 用 **untied embeddings**，即 token embedding 和 LM head 是两份独立的权重，不共享。

逐块算：

**Embedding 层**：

$$
M_{\text{embed}} = V \times d_{\text{model}} = 128256 \times 4096 \approx 525.34\text{ M}
$$

**每层 Transformer 的注意力部分**：

| 矩阵 | 维度 | 参数量 |
|------|------|-------|
| $W_Q$ | $d_{\text{model}} \times (n_h d_{\text{head}})$ = $4096 \times 4096$ | 16.78 M |
| $W_K$ | $d_{\text{model}} \times (n_{kv} d_{\text{head}})$ = $4096 \times 1024$ | 4.19 M |
| $W_V$ | 同 $W_K$ | 4.19 M |
| $W_O$ | $(n_h d_{\text{head}}) \times d_{\text{model}}$ = $4096 \times 4096$ | 16.78 M |
| **小计** | | **41.94 M** |

注意 $W_K$、$W_V$ 用的是 $n_{kv} d_{\text{head}} = 1024$ 而不是 $n_h d_{\text{head}} = 4096$——这就是 **GQA**（Grouped-Query Attention）：多个 query head 共享一组 K、V，把 KV 投影矩阵砍小了 4 倍。

**每层 Transformer 的 FFN 部分**（Llama 用 SwiGLU，3 个矩阵）：

| 矩阵 | 维度 | 参数量 |
|------|------|-------|
| `gate_proj` | $4096 \times 14336$ | 58.72 M |
| `up_proj`   | $4096 \times 14336$ | 58.72 M |
| `down_proj` | $14336 \times 4096$ | 58.72 M |
| **小计** | | **176.16 M** |

**RMS Norm**：每层有 2 个，每个只有 $d_{\text{model}}=4096$ 个参数，量级可忽略。

**每层总计**：$41.94 + 176.16 \approx 218.1\text{ M}$

**32 层**：$218.1 \times 32 \approx 6.98\text{ B}$

**LM Head**：$d_{\text{model}} \times V = 4096 \times 128256 \approx 525.34\text{ M}$

**全模型**：

$$
M_{\text{total}} \approx 0.525 + 6.98 + 0.525 = 8.03\text{ B} \approx 8\text{ B} \checkmark
$$

数字对得上。这告诉我们：标称“8B”约由 32 层 × 0.218 B/层和头尾两个 0.525 B 组成；RMSNorm 等小项未在上式展开。

> 一个有用的副产品：**FFN 占了 80% 的参数**（176.16 / 218.1），attention 只占 20%。所以 LLM 的"大块头"在 FFN，不是注意力。后面讲 Flash Attention 时回想一下这个比例——优化注意力对吞吐有用，对模型权重显存帮助有限。

## 2. KV Cache：被严重低估的显存杀手

### 2.1 从注意力公式出发

回顾 Transformer Decoder 在第 $t$ 步的注意力：

$$
\text{Attention}(Q_t, K_{\le t}, V_{\le t}) = \mathrm{softmax}\!\left(\frac{Q_t K_{\le t}^{\top}}{\sqrt{d_k}}\right) V_{\le t}
$$

$Q_t$ 只是当前 token 的 query；$K_{\le t}$、$V_{\le t}$ 是 **所有历史 token** 的 K、V。

如果每生成一个 token 都重算所有历史 K、V，第 $t$ 步要 $O(t)$，整个序列 $O(T^2)$，长序列下完全不可接受。所以工业界都用 **KV Cache**：把历史的 $K_i$、$V_i$ 存起来，每步只算当前 token 的 $K_t$、$V_t$ 然后 append。

代价就是把它们常驻显存。

### 2.2 公式

每层需要缓存 K 和 V 各一份。每个 token 每层每个 KV 头需要 $d_{\text{head}}$ 个浮点数。所以总大小：

$$
\boxed{M_{\text{kv}} = 2 \times L \times n_{kv} \times d_{\text{head}} \times b_{\text{param}} \times T \times B}
$$

其中 2 来自 K 和 V 各一份，$T$ 是序列长度，$B$ 是 batch size。

### 2.3 关键：为什么是 $n_{kv}$ 而不是 $n_h$？

注意公式里乘的是 $n_{kv}$（KV 头数），不是 $n_h$（query 头数）。这取决于注意力机制：

- **MHA (Multi-Head Attention)**：每个 query head 有独立的 K、V → $n_{kv} = n_h$
- **GQA (Grouped-Query Attention)**：多个 query head 共享一组 K、V → $n_{kv} < n_h$
- **MQA (Multi-Query Attention)**：所有 query head 共享同一组 K、V → $n_{kv} = 1$

Llama-3-8B 用 GQA，$n_{kv} = 8$。如果它是 MHA（$n_{kv} = 32$），KV Cache 直接膨胀 4 倍。

### 2.4 Llama-3-8B 的具体数字

代入 $L=32$, $n_{kv}=8$, $d_{\text{head}}=128$, BF16（$b_{\text{param}}=2$），单 batch：

$$
M_{\text{kv}} = 2 \times 32 \times 8 \times 128 \times 2 \times T = 131072 \times T \text{ bytes} = 128\, T \text{ KiB}
$$

每个 token 的 KV 占 **131,072 bytes = 128 KiB**。

| $T$ | KV Cache | 占权重比例 |
|-----|---------|-----------|
| 512   | 64 MiB（约 67.1 MB） | 0.4% |
| 2,048 | 256 MiB（约 268.4 MB） | 1.7% |
| 4,096 | 512 MiB（约 536.9 MB） | 3.4% |
| 8,192 | **1 GiB（约 1.074 GB）** | 6.7% |
| 32,768 | **4 GiB（约 4.295 GB）** | 26.8% |
| 131,072 | **16 GiB（约 17.18 GB）** | 107.4% |

原版 Meta-Llama-3-8B 的 `max_position_embeddings` 是 8,192；表中的 32K、128K 是按同一架构参数做的 KV 数学外推，不代表原始 checkpoint 能在该长度上保持可用质量。长上下文部署还要满足模型的位置编码、服务端长度设置和质量验证等条件。

### 2.5 与 batch size 的线性关系

batch size 把 KV Cache 整体放大：

$$
M_{\text{kv}}^{\text{total}} = M_{\text{kv}}(T) \times B
$$

8 个并发请求，每个 32K 上下文 → KV Cache = 32 GiB（约 34.36 GB）。这仍是原版 Llama-3 8K 上下文设置之外的数学外推。

这就是 vLLM 这类推理框架存在的意义：通过 **PagedAttention** 把 KV Cache 切成固定大小的 page，按需分配，多请求共享前缀，消除内部碎片。下一节细讲。

## 3. 推理工作区：必须按引擎实测的动态项

训练中“保存激活供反向使用”的公式，不能拿来估算推理的逐层峰值。推理不保留整条前向的反向激活，但当前层仍会创建 Q/K/V、归一化、FFN 和 attention kernel 的临时张量。它们的峰值取决于 prefill 或 decode、batch/并发、chunk 大小、CUDA 图、内核实现、dtype 和内存分配器，不能从模型结构单独推出一个通用常数。

没有内存高效 attention kernel 时，若某实现为 batch 1、单层显式物化全部注意力分数，其 BF16 张量大小为：

$$
M_{\text{scores}} = n_h \times T^2 \times b = 32 \times 32768^2 \times 2\text{ bytes} = 64\text{ GiB} \approx 68.72\text{ GB}
$$

这约为 64 GiB，即 68.72 GB。

这是“朴素地物化该单层分数张量”的空间量级，不是整次推理的固定峰值。FlashAttention 以分块和 online softmax 避免把完整 $T\times T$ 分数矩阵写入 HBM，从而显著降低注意力的 HBM 驻留和 I/O；精确 attention 的计算量仍是二次量级，并没有变成 $O(T)$。

## 4. 可核算账本与部署检查

对于给定模型、dtype、当前已缓存 token 数和并发数，可先计算：

$$
M_{\text{known}}=M_{\text{weights}}+M_{\text{KV, allocated}}
$$

其中 $M_{\text{known}}$ 只表示可直接核算的常驻部分。

再为运行时工作区、CUDA 上下文、通信/采样缓冲和分配碎片留出实测余量。$M_{\text{known}}$ 不是总峰值，也不能据此断言“某张卡必然 OOM”或“量化后必然可跑”。最小验证是用目标引擎、真实的 prefill/decode 长度与并发，记录峰值显存和延迟。

## 5. 每项的优化边界

### 5.1 模型权重 → 量化

最直接的杠杆。BF16 → INT4：

$$
M_{\text{params}}(\text{INT4}) = \frac{1}{4} M_{\text{params}}(\text{BF16})
$$

标称 8B 模型的编码本体：16 GB → 4 GB；实际节省量应扣除量化元数据和运行时工作区。质量变化与量化算法、group size、校准集、模型和评测集有关，必须针对目标任务验证。

具体怎么把浮点压成 INT4 而不损失太多？这是 **第 3 篇** 的主题——会从最小化误差出发推出对称/非对称量化、GPTQ、AWQ 全套数学。

### 5.2 KV Cache → 三个方向

**架构层（GQA / MQA）**：

$$
\frac{M_{\text{kv}}^{\text{GQA}}}{M_{\text{kv}}^{\text{MHA}}} = \frac{n_{kv}^{\text{GQA}}}{n_{kv}^{\text{MHA}}}
$$

Llama-3-8B 用 GQA 而非 MHA，KV Cache 直接省 4 倍。这是模型设计层面的决策，部署时不可改，但选模型时要看清。

**数值层（KV 量化）**：把 KV Cache 从 BF16 (2 字节) 存成 FP8/INT8 (1 字节) 甚至 INT4 (0.5 字节)：

$$
M_{\text{kv}}(\text{FP8}) = \tfrac{1}{2} M_{\text{kv}}(\text{BF16})
$$

32K 上下文：4 GiB → 2 GiB。vLLM 的 `kv_cache_dtype="fp8"` 就是干这个。

**系统层（PagedAttention）**：

KV Cache 按固定大小的 page 分配，类似操作系统虚拟内存：

- 消除内部碎片（变长序列导致的不连续分配）
- 多请求共享前缀（beam search、system prompt 共享）
- 按需分配（短请求不预占长 cache）

PagedAttention 主要减少按连续大块预留造成的分配浪费；它不会压缩已经真实存储的每个 token 的 K/V。前缀共享只在请求确实共享相同前缀且引擎启用该能力时减少重复 KV。

### 5.3 注意力工作区 → FlashAttention

如第 3 节所述，FlashAttention 避免将完整分数矩阵驻留在 HBM。是否可用以及具体版本取决于硬件、安装包和模型实现，应从目标引擎日志或配置确认。

## 6. 实战配置

### 6.1 vLLM

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    max_model_len=8192,         # 请求可接受的最大上下文长度
    gpu_memory_utilization=0.9, # 引擎的显存预算比例
    kv_cache_dtype="fp8",       # KV Cache FP8 量化（H100+）
)
```

原始 BF16 checkpoint 不能仅靠 `quantization="awq"` 变成已量化 AWQ 模型；应加载兼容的已量化 checkpoint，并按该产物和 vLLM 版本的说明配置量化。`max_model_len` 会约束请求长度，却不是 KV Cache 池大小的唯一决定因素；权重、可用显存比例、并发、block/page 配置和运行时预留同样参与分配。`gpu_memory_utilization` 也不保证固定的 10% 足以覆盖所有框架工作区。

### 6.2 HuggingFace Transformers 原生

```python
from transformers import AutoModelForCausalLM
import torch

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3-8B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="flash_attention_2",  # 避免物化完整注意力分数矩阵
)
```

原生 transformers 没有 PagedAttention，多请求并发吞吐远不如 vLLM。**生产部署用 vLLM/TGI/SGLang，开发调试用 transformers**。

## 7. 小结

| 项目 | 公式 | 关键参数 |
|------|------|---------|
| 模型权重 | $N_{\text{params}} \times b_{\text{param}}$ | 参数量、精度 |
| KV Cache | $2 L\, n_{kv} d_{\text{head}}\, b_{\text{param}}\, T B$ | 层数、KV 头数、序列长度、batch |
| 运行时工作区 | 由目标引擎测量 | prefill/decode、并发、kernel、chunk、dtype |

下次有人问「这个 8B 模型在 4090 上能跑多长上下文？」，先按 KV 公式给出当前 token 数与并发下的**已知下界**，再在目标引擎测量 prefill/decode 峰值。显存数学用于缩小搜索范围；真实引擎压测才给出部署结论。

下一篇 → [LLM 训练与推理优化（三）— 量化深入：从最小化误差到 GPTQ/AWQ](/posts/2026-06-14-llm-mem-opt-3-quantization/)，我们把第 5.1 节里"INT4 量化能省 12 GB"这一句的数学全部展开。

---

## 参考资料

1. Kwon et al. *Efficient Memory Management for Large Language Model Serving with PagedAttention*. SOSP 2023. [arXiv:2309.06180](https://arxiv.org/abs/2309.06180)
2. Dao et al. *FlashAttention: Fast and Memory-Efficient Exact Attention*. NeurIPS 2022. [arXiv:2205.14135](https://arxiv.org/abs/2205.14135)
3. Dao. *FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning*. 2023. [arXiv:2307.08691](https://arxiv.org/abs/2307.08691)
4. Shazeer. *Fast Transformer Decoding: One Write-Head is All You Need*. 2019. [arXiv:1911.02150](https://arxiv.org/abs/1911.02150) — MQA
5. Ainslie et al. *GQA: Training Generalized Multi-Query Transformer Models*. 2023. [arXiv:2305.13245](https://arxiv.org/abs/2305.13245)
6. Meta. *The Llama 3 Herd of Models*. 2024. [arXiv:2407.21783](https://arxiv.org/abs/2407.21783); [Meta-Llama-3-8B config](https://huggingface.co/meta-llama/Meta-Llama-3-8B/blob/main/config.json)
