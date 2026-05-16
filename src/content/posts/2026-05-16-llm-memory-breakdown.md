---
title: 大模型推理显存拆解 — 一步步算清你的显存去哪了
published: 2026-05-16
description: 以 Llama-3-8B 为例，从参数怎么算、KV Cache 公式怎么来的、激活值有多大，到每项怎么优化，一步步推导而不是直接扔给你一个数字。
category: Deep Learning
tags:
- llm
- memory
- deep learning
- inference
draft: false
---

## 0. 前言

你有一张 RTX 4090, 24GB 显存. 你想跑 Llama-3-8B.

问题来了: 8B 参数的模型, 在 BF16 精度下, 光是加载权重就要 16GB. 你还有 8GB 的余量. "够了! 跑吧!"

然后跑起来发现 OOM (Out Of Memory).

为什么? 因为权重只是冰山一角. 推理过程中还有**KV Cache**和**激活值**在悄悄吃显存. 这篇文章我们就来一笔笔算清楚, 每个公式都从原理出发推导出来, 而不是直接扔给你一个数字.

最终你会发现: 显存的去向其实完全可以精确计算, 而且每一笔都有优化的办法.

---

## 1. 模型参数: 第一笔账

这笔账最好算.

### 1.1 基本公式

模型参数占用的显存 = 参数数量 × 每个参数的字节数:

$$
M_{\text{params}} = N_{\text{params}} \times b_{\text{param}}
$$

其中 $b_{\text{param}}$ 取决于精度:
- FP32: 4 字节
- BF16 / FP16: 2 字节
- INT8: 1 字节
- INT4: 0.5 字节

所以对于 8B 参数的模型:

| 精度 | $M_{\text{params}}$ | 计算过程 |
|------|-------------------|---------|
| BF16 | 16 GB | $8 \times 10^9 \times 2 = 16 \times 10^9$ 字节 |
| INT8 | 8 GB | $8 \times 10^9 \times 1 = 8 \times 10^9$ 字节 |
| INT4 | 4 GB | $8 \times 10^9 \times 0.5 = 4 \times 10^9$ 字节 |

### 1.2 更精确地算: 参数从哪来?

"8B 参数"这个数字到底是怎么组成的? 我们以 Llama-3-8B 为例拆一下.

Llama-3-8B 的结构参数:

| 参数 | 符号 | 值 |
|-----|------|----|
| 层数 | $L$ | 32 |
| 注意力头数 | $n_{\text{heads}}$ | 32 |
| 隐藏维度 | $d_{\text{model}}$ | 4096 |
| FFN 中间维度 | $d_{\text{ff}}$ | 14336 |
| 每头维度 | $d_{\text{head}}$ | $d_{\text{model}} / n_{\text{heads}} = 4096 / 32 = 128$ |
| KV 头数 | $n_{\text{kv}}$ | 8 (GQA) |
| 词表大小 | $V$ | 128000 |

逐层细分:

**1. Embedding 层** (词嵌入):

$$
M_{\text{embed}} = V \times d_{\text{model}} = 128000 \times 4096 \approx 524\text{M}
$$

**2. 每层 Transformer** (共 32 层):

注意力部分:
- Q 投影: $d_{\text{model}} \times (n_{\text{heads}} \times d_{\text{head}}) = 4096 \times 4096 = 16.8\text{M}$
- K 投影: $d_{\text{model}} \times (n_{\text{kv}} \times d_{\text{head}}) = 4096 \times 1024 = 4.2\text{M}$
- V 投影: 同 K, $4.2\text{M}$
- O 投影: $(n_{\text{heads}} \times d_{\text{head}}) \times d_{\text{model}} = 4096 \times 4096 = 16.8\text{M}$

每层注意力合计: $16.8 + 4.2 + 4.2 + 16.8 = 42\text{M}$

FFN 部分 (SwiGLU, 3个矩阵):
- gate_proj: $d_{\text{model}} \times d_{\text{ff}} = 4096 \times 14336 = 58.7\text{M}$
- up_proj: 同上, $58.7\text{M}$
- down_proj: $d_{\text{ff}} \times d_{\text{model}} = 14336 \times 4096 = 58.7\text{M}$

每层 FFN 合计: $58.7 \times 3 = 176.1\text{M}$

每层 Transformer 合计: $42 + 176.1 = 218.1\text{M}$
32 层: $218.1 \times 32 \approx 7.0\text{B}$

**3. RMS Norm** (每层有 2 个, 加上最后的):

每个 RMS Norm 只有 $d_{\text{model}}$ 个可训练参数 (= 4096), 可以忽略.

**4. LM Head** (输出层):

$d_{\text{model}} \times V = 4096 \times 128000 = 524\text{M}$

**总计**: $524\text{M} (\text{embed}) + 7.0\text{B} (\text{32层}) + 524\text{M} (\text{head}) \approx 8.0\text{B}$ ✓

这个计算验证了: 8B 参数不是凭空说的, 每一层、每个矩阵的贡献都可以精确计算. 当有人告诉你"这是一个 8B 模型"时, 你可以快速心算: 大概要占 16GB (BF16) / 8GB (INT8) / 4GB (INT4).

---

## 2. KV Cache: 被严重低估的显存杀手

> KV Cache 的原理我在之前的博客 [A Series on LLMs (II)](/posts/2025-02-06-A-Series-on-LLM-Inference-II/) 中已经详细介绍过了, 这里简单回顾一下核心思路, 重点放在"占多少显存"的计算上.

### 2.1 从注意力公式出发

先回顾一下 Transformer Decoder 的注意力计算. 在第 $t$ 步, 模型需要计算:

$$
\text{Attention}(Q_t, K_{\le t}, V_{\le t}) = \text{softmax}\left(\frac{Q_t K_{\le t}^T}{\sqrt{d_k}}\right) V_{\le t}
$$

这里 $Q_t$ 是**当前 token**的 query (大小 $1 \times d_k$), 而 $K_{\le t}$ 和 $V_{\le t}$ 是**所有历史位置**的 key 和 value (大小 $t \times d_k$).

你可以选择:

- **方案 A**: 每次重新算 $K_{\le t}$ 和 $V_{\le t}$ — 第 $t$ 步的计算量是 $O(t \times d_k)$, 累计 $O(T^2 \times d_k)$, 序列长了完全不可接受.
- **方案 B**: 把之前每一步算好的 $K_i$, $V_i$ 存起来, 每次只要算当前 token 的 $K_t$, $V_t$, 然后拼到缓存里.

方案 B 就是 **KV Cache**.

### 2.2 KV Cache 的精确公式

每层需要缓存 K 和 V 两份. 每个 token 每层每头需要的空间是 $d_{\text{head}} \times b_{\text{param}}$.

所以 KV Cache 的总大小:

$$
M_{\text{kv}} = 2 \times L \times n_{\text{kv}} \times d_{\text{head}} \times b_{\text{param}} \times T
$$

其中 $T$ 是序列长度.

**为什么要乘以 $n_{\text{kv}}$ 而不是 $n_{\text{heads}}$?**

这里取决于注意力机制:
- **MHA** (Multi-Head Attention): 每个 query head 有独立的 K, V head → $n_{\text{kv}} = n_{\text{heads}}$
- **GQA** (Grouped-Query Attention): 多个 query head 共享一组 K, V → $n_{\text{kv}} < n_{\text{heads}}$
- **MQA** (Multi-Query Attention): 所有 query head 共享同一组 K, V → $n_{\text{kv}} = 1$

Llama-3-8B 用了 GQA, $n_{\text{kv}} = 8$. 如果它用 MHA ($n_{\text{kv}} = 32$), KV Cache 会大 4 倍!

### 2.3 具体数字

以 Llama-3-8B 为例 ($L=32$, $n_{\text{kv}}=8$, $d_{\text{head}}=128$, BF16, $b_{\text{param}}=2$):

$$
M_{\text{kv}} = 2 \times 32 \times 8 \times 128 \times 2 \times T
$$

简化: $M_{\text{kv}} = 131072 \times T \ \text{字节} = 128 \times T \ \text{KB}$

| $T$ | $M_{\text{kv}}$ | 占权重的比例 |
|-----|----------------|------------|
| 512 | 64 MB | 0.4% |
| 2,048 | 256 MB | 1.6% |
| 4,096 | 512 MB | 3.1% |
| 8,192 | 1 GB | 6.3% |
| 32,768 | 4 GB | 25% |
| 131,072 | 16 GB | 100% |

可以看到: **当序列长度达到 128K 时, KV Cache 的显存开销已经和模型权重本身一样大!**

这就是为什么长上下文推理如此吃显存. 跑 128K 上下文意味着你需要**双倍**的显存——一份装权重, 一份装 KV Cache.

### 2.4 与 batch size 的关系

上面的计算假设 batch size = 1. 如果同时处理 $B$ 个请求:

$$
M_{\text{kv}}(\text{total}) = M_{\text{kv}}(T) \times B
$$

KV Cache 随着 batch size **线性增长**. 如果有 8 个并发请求, 每个 32K 上下文, KV Cache 就要 32GB——已经超过了大多数消费级显卡.

这就是为什么 vLLM 等推理框架如此重要——它们通过 PagedAttention 让多个请求共用显存, 消除了内部碎片.

---

## 3. 激活值: 临时工

与模型参数和 KV Cache 不同, 激活值是**临时**占用的——每步前向传播后就会被释放.

### 3.1 激活值从哪来?

在推理的每一步, 数据流过每一层 Transformer:

```
输入 (hidden_states)
  → RMS Norm → QKV 投影 → Attention 计算 → 残差连接
  → RMS Norm → FFN (gate/up/down) → 残差连接
  → 输出到下一层
```

每一层的**中间结果**都需要占用显存. 具体来说:

- Attention 部分: Q, K, V 投影后的矩阵, attention score ($T \times T$), attention output
- FFN 部分: gate 输出, up 输出, 中间激活, down 输出
- 残差连接: 需要保留输入向量用于加法

### 3.2 估算公式

有个经验公式可以快速估算:

$$
M_{\text{act}} \approx (34 \times d_{\text{model}} + 5 \times d_{\text{ff}}) \times T \times B \times b_{\text{param}}
$$

这个 34 和 5 是怎么来的? 来自每层中 FFN、QKV 投影等中间矩阵的大小之和. **注意: 此公式不包含 $O(T^2)$ 的注意力分数矩阵——后者需要独立计算: $M_{\text{attn}} = n_{\text{heads}} \times T^2 \times b_{\text{param}}$ (见 5.3 节). 若未使用 Flash Attention, 32K 序列的注意力矩阵额外占用约 64 GB, 总激活值将高达 ~77 GB. 下文的优化表中,“原始 (BF16)”行的激活值已假设使用 Flash Attention (否则无法满足 O(T) 的公式), 因此直接使用此公式.** 对于 Llama-3-8B ($d_{\text{model}}=4096$, $d_{\text{ff}}=14336$):

$$
M_{\text{act}} \approx (34 \times 4096 + 5 \times 14336) \times T \times B \times 2
$$

当 $B=1$ 时:

| $T$ | $M_{\text{act}}$ |
|-----|-----------------|
| 512 | $\approx 206\,\text{MB}$ |
| 4,096 | $\approx 1.6\,\text{GB}$ |
| 32,768 | $\approx 13\,\text{GB}$ |

注意: 长序列时激活值的占用也接近模型权重了! 这是因为 $T \times d_{\text{model}}$ 的乘积在变大.

### 3.3 为什么激活值容易被忽略

KV Cache 和模型参数是**常驻**显存的——加载后直到推理结束才释放. 激活值是**临时**的——每算完一层就释放一部分.

所以很多人只关注常驻部分. 但问题在于**峰值**时刻: 当长序列且没有 Flash Attention 时, 每个注意力头都需要创建 $T \times T$ 的分数矩阵 (对 32K 序列就是 $32768^2 \times 2 \approx 2\,\text{GB}$ 每头), 而 Llama-3-8B 有 $n_{\text{heads}}=32$ 个头, 总大小达 $32 \times 2 \approx 64\,\text{GB}$——仅此一项就足以撑爆绝大多数显卡.

这就是为什么 Flash Attention 如此重要——它通过分块计算避免了一次性创建完整的注意力矩阵.

---

## 4. 总账本

对 Llama-3-8B (BF16, batch=1) 做个总账:

### 4.1 短序列 (512 tokens)

| 项目 | 大小 | 占比 |
|------|------|------|
| 模型参数 | 16.0 GB | ~98% |
| KV Cache | 0.0625 GB | 0.4% |
| 激活值 (峰值) | 0.2 GB | 1.2% |
| **总计** | **~16.3 GB** | |

→ 24GB 显卡轻松跑. 主要瓶颈是模型权重.

### 4.2 中等序列 (8K tokens)

| 项目 | 大小 | 占比 |
|------|------|------|
| 模型参数 | 16.0 GB | 74% |
| KV Cache | 1.0 GB | 5% |
| 激活值 (峰值) | 3.2 GB | 15% |
| 其他开销 | ~1.3 GB | 6% |
| **总计** | **~21.5 GB** | |

→ 24GB 显卡刚好够用, 但快满了.

### 4.3 长序列 (32K tokens)

| 项目 | 大小 | 占比 |
|------|------|------|
| 模型参数 | 16.0 GB | 44% |
| KV Cache | 4.0 GB | 11% |
| 激活值 (峰值) | 13 GB | 36% |
| 其他开销 | ~3 GB | 9% |
| **总计** | **~36 GB** | |

→ 24GB 显卡完全不够! 必须优化.

---

## 5. 每项都能优化

既然知道了每一笔账, 就可以针对性地"省钱".

### 5.1 模型参数: 量化

量化就是把 BF16 降到更低位宽:

$$
M_{\text{params}}(\text{INT4}) = \frac{1}{4} M_{\text{params}}(\text{BF16})
$$

8B 模型: 16 GB → 4 GB, 省 12 GB.

代价? 理论上少量精度损失, 实践中 INT4 的 MMLU (Massive Multitask Language Understanding) 损失通常在 1% 以内. 值不值? 对于部署来说, 太值了.

### 5.2 KV Cache: 三个方向

**方向 1: GQA / MQA** (架构层面)

从 MHA 换成 GQA 或 MQA, 直接减少 $n_{\text{kv}}$:

$$
\frac{M_{\text{kv}}^{\text{GQA}}}{M_{\text{kv}}^{\text{MHA}}} = \frac{n_{\text{kv}}^{\text{GQA}}}{n_{\text{kv}}^{\text{MHA}}}
$$

Llama-3-8B 用 GQA ($n_{\text{kv}}=8$) 而非 MHA ($n_{\text{kv}}=32$), KV Cache 直接省 4 倍.

**方向 2: KV Cache 量化** (数值层面)

把 KV Cache 从 FP16 (2 字节) 存成 INT8 (1 字节):

$$
M_{\text{kv}}(\text{INT8}) = \frac{1}{2} M_{\text{kv}}(\text{FP16})
$$

32K 上下文: 4 GB → 2 GB.

**方向 3: PagedAttention** (系统层面)

KV Cache 按固定大小的 page 分配, 类似操作系统虚拟内存分页. 主要收益:
- 消除内部碎片 (不同序列长度导致的不连续分配)
- 方便内存共享 (如 beam search 的多个候选共用前缀)

vLLM 用 PagedAttention 宣称能省 60-80% 的 KV Cache 显存——这个数字来自**碎片消除** + **共享前缀** + **按需分配**的综合效果.

### 5.3 激活值: Flash Attention

标准的注意力实现需要构建 $T \times T$ 的注意力矩阵:

$$
M_{\text{attn}} = n_{\text{heads}} \times T^2 \times b_{\text{param}}
$$

32K 序列: $32 \times 32768^2 \times 2 \approx 64\,\text{GB}$ (32 个头, 每头 $\approx 2$ GB)

Flash Attention 把计算分块, 让注意力矩阵的子块在 SRAM (Static Random-Access Memory) 中处理, 然后累加结果进 HBM (High Bandwidth Memory). 这样在 HBM 层面只需要 $O(T \times d)$ 的显存, 而不是 $O(T^2)$.

收益: 长序列时激活值显存从 $O(T^2)$ 变成 $O(T)$——对于 32K 序列, 可以省数十 GB.

### 5.4 组合优化的效果

对 Llama-3-8B, 32K 上下文, BF16→INT4, 加上各种优化:

| 优化 | 模型参数 | KV Cache | 激活值 | 总计 | 备注 |
|------|---------|---------|--------|------|------|
| 原始 (BF16) | 16 GB | 4 GB | 13 GB | ~36 GB¹ | 不行 |
| + INT4 量化 | 4 GB | 4 GB | 13 GB | ~24 GB¹ | 勉强 |
| + KV Cache INT8 | 4 GB | 2 GB | 13 GB | ~22 GB¹ | OK |
| + Flash Attention | 4 GB | 2 GB | ~11 GB | ~17 GB | 还行 |

这就是 LLM 推理优化的魔力——**通过理解每一笔开销的数学原理, 你可以有针对性地节省几十 GB 的显存**.

> ¹ 包含约 3 GB 的其他开销（CUDA 上下文、临时缓冲区等），Flash Attention 消除了注意力分数矩阵后这部分开销也大幅降低。

---

## 6. 实际部署经验

公式都推导清楚了, 实际操作就简单了:

**短上下文 (<2K)**: 主要瓶颈是模型参数 → 先量化
**长上下文 (>8K)**: 主要瓶颈是 KV Cache → GQA + PagedAttention
**大批量推理**: 激活值和 KV Cache 都线性增长 → Flash Attention + PagedAttention

一个具体的配置:

```python
# vLLM 自动处理大部分优化
from vllm import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Meta-Llama-3-8B",
    max_model_len=8192,         # 限制最大序列长度
    gpu_memory_utilization=0.9, # 使用 90% 显存
    kv_cache_dtype="fp8",       # KV Cache 量化
)
```

HuggingFace 原生:

```python
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3-8B",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="flash_attention_2",  # 省激活值
)
```

---

## 7. 总结

推理显存的数学很简单, 就是加乘:

| 项目 | 公式 | 关键参数 |
|------|------|---------|
| 模型参数 | $N_{\text{params}} \times b_{\text{param}}$ | 参数量和精度 |
| KV Cache | $2 \times L \times n_{\text{kv}} \times d_{\text{head}} \times b_{\text{param}} \times T$ | 层数、头数、序列长度 |
| 激活值 | $\approx (34 \times d_{\text{model}} + 5 \times d_{\text{ff}}) \times T \times B \times b_{\text{param}}$ | 模型宽度、序列长度 |

关键在于: **每一项你都能精确算出, 每算出来一项, 就知道应该从哪下手优化**.

下次别人说"这个 8B 模型跑不起来", 你可以问: 上下文多长? 精度用什么? 用 Flash Attention 了吗? ——而且每一问你都知道他差在哪里.

---

### 参考资料

1. Kwon et al., Efficient Memory Management for Large Language Model Serving with PagedAttention. SOSP 2023. [arXiv:2309.06180](https://arxiv.org/abs/2309.06180) — **vLLM 核心论文**
2. Dao et al., FlashAttention: Fast and Memory-Efficient Exact Attention. NeurIPS 2022. [arXiv:2205.14135](https://arxiv.org/abs/2205.14135)
3. Shazeer, Fast Transformer Decoding: One Write-Head is All You Need. 2019. [arXiv:1911.02150](https://arxiv.org/abs/1911.02150) — **MQA**
4. Ainslie et al., GQA: Training Generalized Multi-Query Transformer Models. 2023. [arXiv:2305.13245](https://arxiv.org/abs/2305.13245)
5. Meta, The Llama 3 Herd of Models. 2024. [arXiv:2407.21783](https://arxiv.org/abs/2407.21783) — **Llama-3 架构细节**
