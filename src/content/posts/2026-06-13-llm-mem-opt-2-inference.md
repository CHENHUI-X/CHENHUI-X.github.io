---
title: 大模型推理：显存容量、KV Cache 与 FlashAttention
published: 2026-06-13
description: 一篇贯通 Llama 参数与 GQA 显存计数、KV Cache 前向、FlashAttention 在线 softmax 和 causal mask，保留可复算例子并附容量计算器与 OOM 清单。
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

做推理优化，需要弄清显存里保存了什么、attention 怎样复用或分块计算，以及运行时的显存峰值出在哪里。本文按「容量 → 计算原理 → OOM 实践」展开；单位、浮点精度和训练状态先看[显存基础](/posts/2026-06-12-llm-mem-opt-1-fundamentals/)。

- **算容量**：第 1—5 节从 Llama 的真实参数数、GQA 缓存和工作区入手，[从模型参数开始](#1-模型参数从矩阵数到精确总数)
- **理解计算（可选数学）**：第 6 节逐步手算 KV 更新与在线 softmax；已有计算基础可以跳过，[从计算原理开始](#cache-computation)
- **解决 OOM**：第 7 节给出引擎配置边界、可运行计算器和实测清单，[直接排查](#oom-checklist)

## 0. 先说明模型和计算边界

下面以**原版 Meta Llama-3-8B** 为例，默认单设备、完整权重、BF16、无 offload（不把权重或缓存移到 CPU 内存等位置），不代表后续 Llama 版本。原版支持 8,192 token 上下文；文中的 32K/128K 只用于同架构数学外推，不能当成该 checkpoint 的部署能力。[Meta 官方说明](https://github.com/meta-llama/llama3#inference)也指出其示例会按最大长度和 batch 预分配缓存。

沿用[显存基础的单位约定](/posts/2026-06-12-llm-mem-opt-1-fundamentals/#1-bitbytegb-与-gib)：$\Psi$ 是参数个数，$M$ 是 bytes；$b_w$、$b_{kv}$、$b_a$ 分别是权重、KV 和某个临时张量每元素的字节数。容量统一以 GiB 对照，必要时另列 GB。

判断某张卡能否运行模型，需要先算清已知的存储量，再测指定引擎的峰值，不能只给一个固定的「能/不能跑」结论。

## 1. 模型参数：从矩阵数到精确总数

### 1.1 结构和符号

| 超参数 | 符号 | Llama-3-8B 的值 |
|--------|------|----------------|
| Transformer 层数 | $L$ | 32 |
| 隐藏维度 | $d$ | 4,096 |
| query 头数 | $n_h$ | 32 |
| KV 头数 | $n_{kv}$ | 8 |
| 每头维度 | $d_h$ | 128，且 $n_hd_h=d$ |
| FFN 中间维度 | $d_{ff}$ | 14,336 |
| 词表大小 | $V$ | 128,256 |

这些结构值可从 [Meta 模型配置](https://github.com/meta-llama/llama-models/blob/main/models/sku_list.py)和[模型实现](https://github.com/meta-llama/llama3/blob/main/llama/model.py)核对。输入 token embedding 与输出 LM head 不共享权重；线性层没有 bias，RMSNorm 各有 $d$ 个缩放参数。RoPE 的频率等 buffer 不属于可训练参数。

### 1.2 注意力与 FFN

以下矩阵用「输入维度 × 输出维度」记法。PyTorch `Linear.weight` 常按相反顺序存储，但参数个数相同。

| 每层矩阵 | 形状 | 参数个数 |
|----------|------|----------|
| $W_Q$ | $d\times n_hd_h=4096\times4096$ | 16,777,216 |
| $W_K$ | $d\times n_{kv}d_h=4096\times1024$ | 4,194,304 |
| $W_V$ | 与 $W_K$ 相同 | 4,194,304 |
| $W_O$ | $n_hd_h\times d=4096\times4096$ | 16,777,216 |
| **注意力小计** | $2dn_hd_h+2dn_{kv}d_h$ | **41,943,040** |
| FFN `gate_proj` | $d\times d_{ff}=4096\times14336$ | 58,720,256 |
| FFN `up_proj` | 与 `gate_proj` 相同 | 58,720,256 |
| FFN `down_proj` | $d_{ff}\times d=14336\times4096$ | 58,720,256 |
| **SwiGLU FFN 小计** | $3dd_{ff}$ | **176,160,768** |
| 两个 RMSNorm | $2d$ | 8,192 |
| **单个 Transformer block** | 上述各项之和 | **218,112,000** |

这里用不同符号区分参数个数和显存：本节只数参数，尚未乘以对应 dtype 的存储字节数。

### 1.3 别漏掉 embedding、LM head 与最后一个 norm

两个词表矩阵各有 $Vd=525,336,576$ 个参数；32 层 block 共 6,979,584,000 个；最后的 RMSNorm 还有 4,096 个。因此：

$$
\begin{aligned}
\Psi
&=2Vd+L\left(2dn_hd_h+2dn_{kv}d_h+3dd_{ff}+2d\right)+d\\
&=8{,}030{,}261{,}248
\end{aligned}
$$

一份未分片、使用同一存储宽度的裸权重满足 $M_w=\Psi b_w$。BF16 的 $b_w=2$，因此：

$$
M_w=2\Psi=16{,}060{,}522{,}496\text{ bytes}
\approx16.0605\text{ GB}\approx14.9575\text{ GiB}
$$

**FFN 占单个 block 约 80.8%，占全模型约 70.2%。** 前者是 $176{,}160{,}768/218{,}112{,}000$；后者是 $32\times176{,}160{,}768/\Psi$。因为全模型还包含两个大词表矩阵，不能把「block 内 80%」直接说成全模型占比。

这也说明 FlashAttention 不会缩小已加载的权重：它优化的是注意力执行方式，权重矩阵本身仍在。

### 1.4 位宽换算与编码本体

| 存储方式 | 字节/参数 | 标称 7B 裸权重 GB | 标称 7B 裸权重 GiB | Llama-3-8B 裸权重 GiB |
|----------|-----------|------------------|--------------------|------------------------|
| FP32 | 4 | 28 | 26.0770 | 29.9151 |
| BF16 / FP16 | 2 | 14 | 13.0385 | 14.9575 |
| FP8 / INT8 | 1 | 7 | 6.5193 | 7.4788 |
| 打包 INT4 | 0.5 | 3.5 | 3.2596 | 3.7394 |

7B 一栏明确采用 $7\times10^9$ 个参数，用于展示标称容量换算；Llama 一栏采用上面逐矩阵得到的真实计数。标称 70B 的同口径数字是 7B 一栏的 10 倍，不能据此把任意名为「70B」的 checkpoint 当成精确参数数。

低位宽行只是所有参数采用同一位宽的**编码本体**，未计 scale、可能的 zero-point、高精度层和对齐，不是 checkpoint 或运行进程的实测大小。分片、offload、混合 dtype 和共享权重还会改变每卡账本；MoE 也不能只按每 token 激活的参数数估计全部已加载权重。

## 2. KV Cache：缓存哪些对象，数哪些头

### 2.1 缓存省掉的是历史位置的重复前向

KV Cache 为后续 token 保存每层历史位置的 K/V；过去的 Query 不属于这份持久缓存。为什么旧结果可复用，以及新 token 如何与旧缓存拼接并得到输出，完整推导与手算例子见[第 6 节的缓存前向机制](#cache-computation)。下面只计算这份缓存需要多少存储。

### 2.2 MHA、GQA、MQA 对应不同 KV 头数

| 机制 | 关系 | 含义 |
|------|------|------|
| MHA（Multi-Head Attention） | $n_{kv}=n_h$ | 每个 query head 有一组 K/V |
| GQA（Grouped-Query Attention） | 常见为 $1<n_{kv}<n_h$ | 每组 query heads 共用一组 K/V |
| MQA（Multi-Query Attention） | $n_{kv}=1$ | 所有 query heads 共用一组 K/V |

通常每组包含 $n_h/n_{kv}$ 个 query heads，要求能均匀分组；GQA 也可广义地包含 MHA 和 MQA 两个端点。Llama-3-8B 是 32 个 query heads、8 个 KV heads，每组 4 个 query heads。[GQA 论文](https://arxiv.org/abs/2305.13245)讨论了这些架构之间的取舍。

共享的是 K/V，不是所有 query 的 attention scores。因此缓存按 $n_{kv}$ 数，显式分数张量仍按 $n_h$ 数。高效实现可直接复用 KV；若 kernel 临时展开 KV 到 query 头数，那是额外工作区，不应误记成架构规定的持久缓存。

### 2.3 从形状推导容量

一层的 K、V 各有逻辑形状 $[B,T,n_{kv},d_h]$，每个元素占 $b_{kv}$ bytes。所有层相同时：

$$
M_{\text{KV,logical}}=2L n_{kv}d_h b_{kv}BT
$$

对于变长请求，把 $BT$ 换成长度和：

$$
M_{\text{KV,logical}}=2L n_{kv}d_h b_{kv}\sum_{i=1}^{B}T_i
$$

$T_i$ 是已写入缓存的 token 数，包括已处理的 prompt 和生成 token；刚从 logits 采样出、尚未送入下一次前向的 token，未必已写入缓存。

这个公式假设无共享、无裁剪、完整注意力，且每层 K/V 的头数、维度、dtype 相同。滑动窗口、混合层结构、压缩潜变量缓存、量化 scale 和特殊架构需要改按实际层/张量求和。Padding 到最大长度的实现也可能占用 $BT_{\max}$，而非有效长度和。

### 2.4 具体数字：权重和 KV 必须用同一单位

单请求 BF16 KV 的每 token 大小为：

$$
2\times32\times8\times128\times2=131{,}072\text{ bytes}=128\text{ KiB}
$$

| 缓存长度 $T$ | 逻辑 KV | 占 BF16 权重的比例 | 加 BF16 权重的小计 |
|--------------|---------|-------------------|----------------------|
| 512 | 64 MiB | 0.42% | 15.0200 GiB |
| 2,048 | 256 MiB | 1.67% | 15.2075 GiB |
| 4,096 | 512 MiB | 3.34% | 15.4575 GiB |
| 8,192 | 1 GiB | 6.69% | 15.9575 GiB |
| 32,768（外推） | 4 GiB，约 4.2950 GB | 26.74% | 18.9575 GiB |
| 131,072（外推） | 16 GiB，约 17.1799 GB | 106.97% | 30.9575 GiB |

后两行超过原版 8K 上下文限制。它们说明缓存随长度增长的规模，不构成扩展上下文后的质量保证。所有小计都还没有工作区与运行时开销。

并发也必须计入：8 个相互独立的等长 8K 请求，逻辑 BF16 KV 共 8 GiB；若由同一模型副本服务，只加一份权重。不能拿单请求账本直接判断并发能力。

## 3. 逻辑缓存不等于实际分配量

### 3.1 静态缓存与分页尾块

静态缓存可能直接为最大 batch 和最大长度分配；动态缓存和分页缓存则采用不同的增长与复用策略。

若分页块长为 $P$ 个 token、每请求独立分配、没有前缀共享，已分给请求的 KV 存储近似为：

$$
M_{\text{KV,blocks}}=2L n_{kv}d_h b_{kv}
\sum_i P\left\lceil\frac{T_i}{P}\right\rceil
$$

这里未计页表、元数据、量化 scale 和引擎预留空闲块。比如 $P=16$，两个请求长 17、33，逻辑上有 50 个 token，却需要 32+48=80 个 token 的块容量。

PagedAttention 让逻辑连续的序列可以映射到不连续物理块，减少连续大块预留的浪费；每个序列的最后一块仍可能没填满，**不能说内部碎片被完全消除**。在上述简单模型中，尾块浪费少于 $BP$ 个 token 槽位。[原论文](https://arxiv.org/abs/2309.06180)还利用块共享与 copy-on-write 管理候选序列。

引擎可能提前建立很大的 KV 池。空闲池块虽暂不含有效请求内容，仍已占显存，所以请求减少时进程显存未必同步下降。

### 3.2 前缀共享有条件

共享的不是任意「差不多的 prompt」：token 前缀、模型及影响 KV 的配置必须匹配，且引擎要启用对应能力。相同前缀的块可以复用；不共享的后缀仍按请求分别保存。不要为每个请求先算一次全部 KV，再把真正共享的物理块重复加进去。[vLLM 的前缀缓存设计](https://docs.vllm.ai/en/v0.9.1/design/v1/prefix_caching.html)说明了按块识别与复用的做法。

### 3.3 多卡不能无条件除以卡数

- **Tensor Parallel（TP）**：常按 head 或矩阵维度切分。只有 KV 头能够按该实现均匀切分、没有复制时，才可按分片数缩小每卡 KV；TP 数大于 KV 头数时，有些引擎会复制 KV heads，有些组合不受支持
- **Pipeline Parallel（PP）**：按层分配，每张卡只缓存自己负责的层。若切层不均或有额外边界/通信数据，各卡峰值会不同
- **Data Parallel（DP）**：不同副本处理不同请求；通常每个副本都保留完整模型，不是把单请求模型状态均分

计算每卡占用时，应按该卡拥有的层、KV heads、请求和副本数求和。参数、KV、临时工作区与通信 buffer 的切分方式也不一定相同。

## 4. Prefill 与 decode：临时工作区为何差这么多

### 4.1 从形状看差别

Prefill 处理一段输入并建立 KV；decode 通常为每个活动请求处理一个新 token，复用已有 KV。若本次新处理的 query 长度是 $S$、可访问的总 key 长度是 $T$：

| 当前层临时对象 | 典型形状 | 影响因素 |
|----------------|----------|----------|
| hidden states | $[B,S,d]$ | 一次调度的 token 数 |
| Q | $[B,S,n_h,d_h]$ | query 头数 |
| 新 K/V | 各 $[B,S,n_{kv},d_h]$ | KV 头数 |
| FFN gate/up 等 | 各约 $[B,S,d_{ff}]$ | 融合、覆盖写和复用策略 |
| 显式 attention scores | $[B,n_h,S,T]$ | 是否物化以及实际 dtype |
| 完整 logits（若保留） | $[B,S,V]$ | 是否只计算所需位置、是否分块 |

从零开始的整段 prefill 有 $S=T$；常规单步 decode 有 $S=1$。Chunked prefill 让 $S$ 变小，但历史 key 长度 $T$ 仍会增长，完整请求的最终 KV 也不会凭空减少。Speculative decoding 一次可验证多个新位置，不总是 $S=1$。

只知道这些形状还不能把它们全部相加当峰值：张量可能顺序创建、共用内存、融合计算、升为 FP32，或被引擎预分配。推理通常不需要为反向保留所有层的激活，因此也不应机械乘上层数 $L$。

<!-- concept-figure:kv-prefill-decode -->

![Hugging Face 的 KV cache 原图：当前 will 的单个 query 读取历史蓝色 K/V 和当前新增的白色 K/V，输出下一个 token be](/images/llm-series/sourced/huggingface-kv-cache.png)

*图：单步 decode 中，蓝色 K/V 来自历史缓存，白色 K/V 本轮新增；当前 `will` 的 query 读取全部可用 K/V，预测 `be`。图中按单个 attention head 简化，$A$ 对应本文 $d_h$，历史长度为 $n$，本轮 $S=1$、$T=n+1$；绿色格表示允许连接的 mask，不表示相同的 attention 权重。原图：Rémi Ouazan Reboul、Arthur Zucker、Luc Georges（Hugging Face），[Continuous batching](https://huggingface.co/blog/continuous_batching#kv-cache)，[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)，未修改。 [查看原尺寸图](/images/llm-series/sourced/huggingface-kv-cache.png)。*

<!-- /concept-figure:kv-prefill-decode -->

### 4.2 几个能算的张量例子

假设 $B=1$、$S=T=32768$、$n_h=32$，某实现显式物化 BF16 scores，则这个单层张量为：

$$
M_{\text{scores}}=Bn_hSTb_a
=32\times32768^2\times2\text{ bytes}
=64\text{ GiB}\approx68.7195\text{ GB}
$$

若实际 scores/softmax 使用 FP32，则同样形状需要 128 GiB。causal mask 只遮住上三角，不代表一个普通稠密张量会自动只存一半。

同一 32K 外推下，一个 BF16 FFN 中间张量 $[1,32768,14336]$ 为 0.875 GiB；有多少个同时存活由实现决定。8K prefill 若保留所有位置的 BF16 logits，$8192\times128256\times2$ bytes 约为 1.9570 GiB；若只需要末位置 logits，实现可能避免保留这份完整张量。

这些是具体张量的大小，**不是通用的激活常数，也不能相加后宣布整次推理必占多少显存**。

### 4.3 FlashAttention 究竟省什么

在容量账本中，FlashAttention 改变的是上面的显式 scores/softmax 中间存储项；权重、持久 KV 和其他算子的工作区仍需计入。不能因为启用它，就把整个推理过程的工作区记成零。

它如何不保留完整分数矩阵，却维持同一个归一化结果，见[第 6.2 节的在线 softmax 合并、两块手算与 causal mask 边界](#62-flashattention-的分块前向)。这里只用这个结论决定是否把完整 scores 张量列入目标实现的账本；实际 kernel 和 decode 专用路径还须核对。

## 5. 三种优化分别改变哪一项

### 5.1 权重量化

第 1.4 节只改变了编码本体的位宽。部署量化模型时，还要计入元数据、高精度层与解包/反量化工作区，并验证任务质量和实际速度。

权重量化并不自动改变 $b_{kv}$ 或激活 dtype。原始 BF16 checkpoint 也不会只靠填一个 `quantization="awq"` 参数就变成正确的 AWQ 产物。量化方法与 checkpoint 必须匹配，详见[量化进阶部分](/posts/2026-05-16-llm-quantization-guide/#advanced-quantization)。

### 5.2 KV：架构、数值、分配三个方向

- **GQA/MQA** 减少 $n_{kv}$。Llama-3-8B 的 GQA 缓存是同头维度、同层数 MHA 假设的四分之一。它是模型架构选择，不能在部署时直接把头数改小而保持原模型行为
- **KV 量化** 减少 $b_{kv}$。若 BF16 的 2 bytes 变成 FP8 的 1 byte，32K 外推的编码本体由 4 GiB 变成 2 GiB；scale、后端支持和质量误差另计
- **分页与共享** 减少空置空间和重复前缀，不改变每个独有 token 必须存的 K/V 元素数

FP8 KV 存储不等于需要或使用 FP8 矩阵乘；支持条件由设备、CUDA/ROCm、内核和版本共同决定，不能笼统写成「H100 以上才可用」。[vLLM 0.9.1 的 KV 量化说明](https://docs.vllm.ai/en/v0.9.1/features/quantization/quantized_kvcache.html)列出了 dtype 与 scale 的配置。

### 5.3 工作区与调度

高效 attention、chunked prefill、限制一次处理的 token 数、减少不需要的 logits，分别针对不同临时对象。改动后仍要复测吞吐、首 token 延迟和每 token 延迟；显存减少不一定同时提高所有性能指标。

<a id="cache-computation"></a>

## 6. 计算原理：KV Cache 与 FlashAttention

这一节是可选的数学推导。第 2 节已经数过持久缓存，第 4 节已经比较过 prefill/decode 的张量形状；下面解释它们为什么这样更新，以及如何避免物化完整 attention 矩阵。

长 prompt 的 prefill 更容易形成较大的矩阵乘；小 batch decode 每步工作量小，却仍要读取权重和历史 KV，常受内存带宽限制。两者不能被固定称为「算力阶段」或「带宽阶段」：batch、上下文、硬件和实现都会改变瓶颈。[SARATHI 论文](https://arxiv.org/abs/2308.16369)据此讨论了 prefill/decode 调度。首 token 延迟、逐 token 延迟和整体吞吐也应分开观察。

以下仍讨论 decoder-only、causal self-attention、普通推理模式。单头手算暂省略 batch、多头拼接、位置变换和输出投影；隐藏维度沿用 $d$，每头维度沿用 $d_h$，不把二者混用。

### 6.1 KV Cache 的前向更新

#### 6.1.1 先分清 K 与 W_K

设某层当前 token 的输入是行向量 $h_t\in\mathbb R^{1\times d}$，单个 attention head 的维度为 $d_h$。这一层的投影参数为：

$$
W_Q,W_K,W_V\in\mathbb R^{d\times d_h}.
$$

当前 token 的 Query、Key、Value 分别是：

$$
q_t=h_tW_Q,\qquad k_t=h_tW_K,\qquad v_t=h_tW_V.
$$

$W_K$ 是模型参数，推理时通常不变；$k_t$ 是对一个 token 算出的向量。把不同位置的 $k_t$ 按行放在一起，才得到随上下文增长的 $K$。**追加新 token 时增长的是 K 和 V，不是 W_K 和 W_V。**

在带 RoPE 的模型中，Query 和 Key 还要按正确的位置进行旋转，通常缓存已经完成位置变换的 Key。具体缓存内容应与模型实现一致，不能为了复用而改变位置编号。

#### 6.1.2 为什么历史结果可以复用？

因果注意力中，旧位置不能看到后来追加的 token。在模型参数、位置编码和旧上下文不变，且关闭训练随机性的条件下，追加 token 不会改变旧位置在各层的结果。

因此，已经处理了 $t-1$ 个 token 时，可以保存每一层的：

$$
K_{1:t-1},V_{1:t-1}\in\mathbb R^{(t-1)\times d_h}.
$$

当前只需计算 $q_t,k_t,v_t$，沿序列维追加：

$$
K_{1:t}=\begin{bmatrix}K_{1:t-1}\\k_t\end{bmatrix},\qquad
V_{1:t}=\begin{bmatrix}V_{1:t-1}\\v_t\end{bmatrix}.
$$

然后计算新位置这一行 attention：

$$
s_t=\frac{q_tK_{1:t}^{\top}}{\sqrt{d_h}}+M_t
\in\mathbb R^{1\times t},\qquad
 a_t=\operatorname{softmax}(s_t),\qquad
 o_t=a_tV_{1:t}\in\mathbb R^{1\times d_h}.
$$

这里需要区分三个对象：

- $q_tK^\top$ 是点积分数
- $s_t$ 是完成缩放并加上 mask 的 attention scores
- $a_t$ 是 softmax 后、和为 1 的 attention weights，**用它与 V 相乘**

$M_t$ 在允许关注的位置取 0，在被屏蔽的位置取 $-\infty$。若缓存中只有过去和当前的有效 token，这一行没有未来位置需要屏蔽；批处理的 padding、预分配缓存的空槽位仍需处理。$o_t$ 只是这一层这个 head 的输出，还要经过剩余网络和词表投影才能预测下一个 token。[Transformer 原论文](https://arxiv.org/abs/1706.03762)给出了 scaled dot-product attention 的定义。

#### 6.1.3 一个可以逐步核算的例子

下面用三个位置的假想隐藏向量演示一次单头计算，不把它们当作某个真实单词的向量。设 $d=d_h=2$：

$$
h_1=[1,0],\qquad h_2=[0,1],\qquad h_3=[1,1],
$$

$$
W_Q=\begin{bmatrix}1&0\\0&1\end{bmatrix},\qquad
W_K=\begin{bmatrix}1&0\\0&2\end{bmatrix},\qquad
W_V=\begin{bmatrix}1&1\\1&0\end{bmatrix}.
$$

前两个位置计算完成后，缓存是：

$$
K_{1:2}=\begin{bmatrix}1&0\\0&2\end{bmatrix},\qquad
V_{1:2}=\begin{bmatrix}1&1\\1&0\end{bmatrix}.
$$

来到第三个位置，只计算新向量 $q_3=[1,1]$、$k_3=[1,2]$、$v_3=[2,1]$，追加后得到：

$$
K_{1:3}=\begin{bmatrix}1&0\\0&2\\1&2\end{bmatrix},\qquad
V_{1:3}=\begin{bmatrix}1&1\\1&0\\2&1\end{bmatrix}.
$$

第三个位置可关注全部三个位置，因此：

$$
s_3=\frac{[1,2,3]}{\sqrt2},\qquad
 a_3\approx[0.140029,\ 0.283995,\ 0.575975],
$$

$$
o_3=a_3V_{1:3}
\approx[1.575975,\ 0.716005].
$$

也可以把三个位置一起重算，在完整 $3\times3$ 的 score 矩阵上加 causal mask，再取第三行，结果相同。KV Cache 省掉的是旧位置的重复投影和重复网络计算；这一行仍然要读取历史 K、V 并做 attention。

核算时先检查投影、缩放、softmax 权重和加权求和四步；把缓存路径与整段重算的第三行对照，可以检查复用是否改变了结果。

#### 6.1.4 缓存的收益与边界

对单层单头而言，处理新位置的 attention 计算量随缓存长度 $t$ 约为 $O(td_h)$。它减少重复工作，但不会让长上下文的 Decode 成为固定成本。每一层都要保留自己的 K/V；怎样按多层、多请求和 KV 头数核算容量，见[第 2 节的容量公式](#2-kv-cache缓存哪些对象数哪些头)。

缓存也不是无条件复用：如果旧 prompt、模型权重、位置处理方式改变，原有缓存可能失效。滑动窗口、GQA/MQA、缓存量化或分页管理会改变保存方式，应分别分析。Prefill/Decode 的缓存形状和更新流程可对照 [Hugging Face Caching 文档](https://huggingface.co/docs/transformers/v4.57.1/en/cache_explanation)。

<details markdown="1">
<summary>思考：为什么通常不缓存历史 Query？</summary>

普通自回归 Decode 要计算的是新位置的输出，所以需要当前 Query 查询历史 K、V。历史 Query 用于计算历史位置的输出，下一轮无需再计算那些输出，自然不需要保存它们供这一步使用。这是当前生成流程下的结论，不代表 Query 在任何算法里都不能缓存。

</details>

### 6.2 FlashAttention 的分块前向

KV Cache 解决“跨生成步骤重复计算旧结果”的问题。**FlashAttention** 解决另一个问题：在一次 attention 计算内部，怎样减少 HBM 与片上存储之间的数据搬运？

它是一种 exact attention 的实现方式：在精确实数运算下计算相同的注意力结果，实际浮点结果可能因运算顺序不同而有小差异。对稠密 attention 而言，它仍需 $O(N^2d_h)$ 量级的计算，**并没有把注意力的算术复杂度变成线性**；主要收益是避免在 HBM 中保存完整 $N\times N$ 中间矩阵，并减少访存。

#### 6.2.1 GPU 内存层级

可以把 HBM 理解为容量较大的显存，把片上寄存器、shared memory 和 cache 理解为离计算单元更近、容量更小的存储。FlashAttention 的核心是把能放下的小块数据搬到片上，在那里多做几步计算，再写回必要结果。

HBM 是高带宽的堆叠 DRAM，并不是 SRAM。SRAM 常用于片上缓存等结构；但“整张 GPU 的所有片上存储容量”和“一个 kernel block 实际能使用的 shared memory”也不是同一个数。

<img src="https://s2.loli.net/2025/12/15/nWKYR5rhVI2QOEz.png" alt="A100 内存层级的简化示意" width="400" height="300" />

图示只用于理解 **A100 40GB 版本**的容量与带宽层级，不能当作所有 A100 或所有 GPU 的统一参数。NVIDIA 对早期 A100 的介绍给出 40GB HBM2、1555GB/s 峰值 HBM 带宽和 40MB L2 cache；这些是不同层级的指标，也不等于实际程序持续达到的速度。详见 [NVIDIA Ampere 架构说明](https://developer.nvidia.com/blog/nvidia-ampere-architecture-in-depth/)。

如果想进一步理解算力、带宽和数据搬运之间的关系，可阅读 [Making Deep Learning Go Brrrr From First Principles](https://horace.io/brrr_intro.html)。

#### 6.2.2 传统 Attention 计算回顾

为简化推导，设单个 head 的 $Q,K,V\in\mathbb R^{N\times d_h}$，Value 维度也取 $d_h$，且不使用 dropout。标准计算为：

$$
S=\frac{QK^\top}{\sqrt{d_h}}+M\in\mathbb R^{N\times N},\qquad
P=\operatorname{softmax}_{\mathrm{row}}(S),\qquad
O=PV\in\mathbb R^{N\times d_h}.
$$

无 mask 时 $M=0$；causal mask 则在未来位置填入 $-\infty$。softmax 按行计算，让每个 Query 对允许访问的 Key 分配权重。

<img src="https://s2.loli.net/2025/12/15/ZHUrvacxpKWusnP.png" alt="传统 attention 分步计算的简化示意" width="800" height="600" />

图中采用简化记号，主要展示矩阵乘法、softmax、再乘 V 的顺序。完整计算应以上面的公式为准，包含 $1/\sqrt{d_h}$ 缩放与任务所需的 mask。

<img src="https://s2.loli.net/2025/12/15/nSxZQFBLgbTt5zP.png" alt="传统 attention 中间结果在显存间传输的简化示意" width="400" height="300" />

这张图表示显式存储中间矩阵的传统分步实现，不表示所有现代 attention kernel 都按同样路径搬运数据。这样的实现通常把 $S$、$P$ 写入 HBM，产生 $O(N^2)$ 规模的中间存储与读写。序列很长时，搬运这些矩阵的开销可能很高。

FlashAttention 要解决的问题是：**不存整行 scores，怎样得到整行 softmax？** 不能简单地“每一块单独做 softmax，再把结果相加”，因为不同块使用的分母不同。

#### 6.2.3 在线 softmax：无 mask 的前向合并

先只看一个 Query 对应的一行。记所有 scores 为 $s_1,\ldots,s_N$，Value 行向量为 $v_1,\ldots,v_N$。输出是：

$$
o=\frac{\sum_{j=1}^{N}e^{s_j}v_j}{\sum_{j=1}^{N}e^{s_j}}.
$$

为了数值稳定，减去行最大值 $m=\max_j s_j$，分子分母同时乘上 $e^{-m}$，比值不变：

$$
\ell=\sum_j e^{s_j-m},\qquad
u=\sum_j e^{s_j-m}v_j,\qquad
o=\frac{u}{\ell}.
$$

其中 $m$、$\ell$ 是标量，$u\in\mathbb R^{1\times d_h}$ 是尚未除以分母的输出累加量。只需要保存这三个量，就能继续读下一块，而不必保存所有 scores。

假设已经处理了一些 Key，现在读到新的非空块 $J$。把旧结果与新块统一到新的最大值：

$$
m_{\mathrm{new}}=\max\left(m,\max_{j\in J}s_j\right),\qquad
\alpha=e^{m-m_{\mathrm{new}}}.
$$

新块用同一个基准计算：

$$
p_j=e^{s_j-m_{\mathrm{new}}},\qquad j\in J.
$$

于是：

$$
\ell_{\mathrm{new}}=\alpha\ell+\sum_{j\in J}p_j,\qquad
u_{\mathrm{new}}=\alpha u+\sum_{j\in J}p_jv_j.
$$

这里的 $p_j$ 还不是最终的 attention weight，而是平移后的指数值。旧分母和旧分子都必须乘以 $\alpha$；如果只修正其中一个，输出就错了。

初始化时取 $m=-\infty,\ell=0,u=0$。第一块有有限 score 时，直接取 $\alpha=0$。后续不断更新 $(m,\ell,u)$，最后输出 $o=u/\ell$。这个过程称为 **online softmax**，“online” 指逐块维护统计量，不是在线训练。相关归一化思想见 [Online normalizer calculation for softmax](https://arxiv.org/abs/1805.02867)。

**为什么合并后仍然正确？** 对旧块中任一 score，都有：

$$
e^{s_j-m_{\mathrm{new}}}
=e^{s_j-m}e^{m-m_{\mathrm{new}}}
=\alpha e^{s_j-m}.
$$

所以把旧分子、旧分母一起缩放，就把全部已处理数据换到同一个指数基准。新块加进来后，仍然满足它们是所有已处理位置的求和。这也是逐块计算与一次性 softmax 相等的原因。

**手算一个两块例子。** 取 scores 为 $[0,\log2,\log3]$，标量 Values 为 $[2,4,8]$：

- 第一块处理前两项：$m=\log2$，$\ell=1/2+1=3/2$，$u=(1/2)\times2+1\times4=5$
- 第二块处理第三项：$m_{\mathrm{new}}=\log3$，$\alpha=2/3$，新块指数值为 1
- 合并得到 $\ell_{\mathrm{new}}=(2/3)(3/2)+1=2$，$u_{\mathrm{new}}=(2/3)5+8=34/3$
- 最终 $o=(34/3)/2=17/3\approx5.666667$

直接计算时，softmax 权重为 $[1/6,2/6,3/6]$，输出 $(2+8+24)/6=17/3$，与分块结果相同。

实际 GPU 会同时处理一块 Query：$Q_I\in\mathbb R^{B_r\times d_h}$，配合 $K_J,V_J\in\mathbb R^{B_c\times d_h}$，临时 scores 只需 $B_r\times B_c$。上面的 $m,\ell$ 变成每行一个统计量，$u$ 变成 $B_r\times d_h$ 的累加器，合并公式按行广播。完整 $N\times N$ 矩阵不需要写入 HBM；输入、输出和 KV Cache 本身仍然占用显存。

以上是解释前向等价性的写法，实际 kernel 的循环顺序、寄存器安排、分块大小会不同。可对照 [FlashAttention 论文](https://arxiv.org/abs/2205.14135)和 [Triton Fused Attention 教程](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html)。CPU 上复算公式不能代替 GPU 性能测量。

#### 6.2.4 Causal mask 与数值边界

causal mask 不改变合并思路，只改变允许参与求和的 Key。若 Query 的绝对位置为 $i$、Key 的绝对位置为 $j$，普通 causal attention 允许 $j\le i$：

$$
s_{ij}=\begin{cases}
q_i k_j^\top/\sqrt{d_h},&j\le i\text{ 且 Key 有效},\\
-\infty,&\text{其他情况}.
\end{cases}
$$

**mask 必须在求最大值和指数之前生效。** 被屏蔽的位置贡献为零，不参与局部最大值与求和。处理分块时，可以跳过完全位于未来的块；跨过对角线的块仍需逐元素屏蔽。

有 KV Cache 时，Query 与 Key 的长度可能不同，不能直接把当前 Query 块的局部行号当作绝对位置。例如已经缓存 5 个位置，再处理 2 个新位置，按零起始编号，Query 的位置是 5、6，而 Key 的位置是 0 到 6：第一行可见 0 到 5，第二行可见 0 到 6。对一个 $2\times7$ 矩阵直接套普通左上角下三角 mask，会错误地只允许前面很少的 Key。

还有两个容易产生 NaN 的边界：

1. **当前块对某一行全被遮蔽。** 这一行应跳过该块并保持旧状态。如果旧状态仍是 $m=-\infty$，机械计算 $-\infty-(-\infty)$ 会得到 NaN；不能靠“最后乘零”补救
2. **整行没有任何有效 Key。** 此时 $\ell=0$，标准 softmax 没有定义。有效的 causal token 通常至少能看见自己；padding query 则应排除计算，或由实现明确约定输出零并确保下游忽略。输出零是接口约定，不是对未定义 softmax 的数学等价结果

下面用 Python 展示一行的合并逻辑。`scores` 已完成缩放，`allowed` 指明该行允许访问的 Key；代码是 CPU 数值参考，不是高性能 FlashAttention kernel：

```python
import math


def online_attention_row(scores, values, allowed, block_size=2):
    if block_size <= 0 or not scores:
        raise ValueError("需要非空 scores 和正 block_size")
    if len(scores) != len(values) or len(scores) != len(allowed):
        raise ValueError("scores、values、allowed 长度必须一致")
    width = len(values[0])
    if width == 0 or any(len(v) != width for v in values):
        raise ValueError("每个 Value 必须具有相同的非零维度")

    m, denom = -math.inf, 0.0
    numer = [0.0] * width
    for start in range(0, len(scores), block_size):
        ids = [j for j in range(start, min(start + block_size, len(scores)))
               if allowed[j]]
        if not ids:
            continue  # 避免全遮蔽块中的 -inf - (-inf)
        if any(not math.isfinite(scores[j]) for j in ids):
            raise ValueError("有效位置的 score 必须有限")
        new_m = max(m, max(scores[j] for j in ids))
        alpha = 0.0 if denom == 0.0 else math.exp(m - new_m)
        weights = [math.exp(scores[j] - new_m) for j in ids]
        denom = alpha * denom + sum(weights)
        numer = [alpha * numer[c] + sum(w * values[j][c]
                 for w, j in zip(weights, ids)) for c in range(width)]
        m = new_m

    if denom == 0.0:
        raise ValueError("整行没有有效 Key，softmax 未定义")
    return [x / denom for x in numer]
```

这里选择对全遮蔽行明确报错，而不是静默返回一个看似合理的结果。代码假设有效 Values 也为有限数值；真实实现还要处理 padding、数据类型、溢出保护和设备限制。单头无 mask、causal mask、非整块长度以及很大正负 scores，都可以用它与稳定的直接 softmax 对照检查。

本文的推导到前向与 mask 边界为止，不是完整训练 kernel。训练中的反向还需利用保存的统计量重算局部 attention，并计算 Q、K、V 的梯度；这部分请直接参照[原论文的反向算法与附录](https://arxiv.org/html/2205.14135v2)。

<a id="6-从账本走到实测"></a>

## 7. 从账本走到实测

### 7.1 先算单请求的确定小计

BF16、已缓存 8,192 token 时：

$$
M_w+M_{\text{KV,logical}}=14.9575+1=15.9575\text{ GiB}
$$

这还不是实际分配池，更不是总峰值。假想 32K 时，类似小计为 18.9575 GiB，但原 checkpoint 的 8K 限制仍需遵守。

### 7.2 一个有版本边界的引擎配置示意

以下按 vLLM 0.9.1 的接口展示配置含义，不是硬件兼容性或峰值实测结果；运行需要已授权的模型访问、兼容的 GPU/CUDA 与安装环境。

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    dtype="bfloat16",
    max_model_len=8192,
    gpu_memory_utilization=0.9,
    kv_cache_dtype="auto",
)
params = SamplingParams(max_tokens=32, temperature=0.0)
outputs = llm.generate(["The main idea of a KV cache is"], params)
print(outputs[0].outputs[0].text)
```

`max_model_len` 约束 prompt 加输出可接受的总长度；`gpu_memory_utilization` 是引擎预算参数，不是「留 10% 就一定安全」的保证。KV 池还受权重、工作区、block 配置和引擎调度影响。[vLLM 版本文档](https://docs.vllm.ai/en/v0.9.1/configuration/engine_args.html)是调整参数的依据。

Transformers 也有多种缓存与 attention 实现，不能概括成「原生一定没有内存高效 attention」或给出无条件的引擎性能排名。应按[目标版本支持的实现](https://huggingface.co/docs/transformers/v4.51.3/en/llm_optims)、服务需求和实测结果选择。

<a id="oom-checklist"></a>

### 7.3 容量计算器与 OOM 排查

前面的公式计算元素数量，这里则根据程序在哪一步耗尽显存来排查。先算确定下界，再按失败时机缩小范围；低于设备容量仍不等于一定能运行。

#### 可直接运行的计算器

先填对输入再运行：`params` 使用真实参数数，`weight_bytes` 使用权重的常驻存储宽度；`lengths` 是每个活动请求已写入缓存的 token 数，容量规划时可填目标 prompt 加生成预算。`kv_heads` 使用 KV 头数，`kv_bytes` 单独读取缓存 dtype。权重量化不会自动量化 KV，函数也不会检查模型的上下文上限。

下面只算单设备、完整权重、无前缀共享的逻辑下界。输入长度是 token 数，不是字数；代码不会加载模型，也不计量化元数据或运行时开销。

```python
from math import isfinite


def estimate_memory(lengths, *, params=8_030_261_248,
                    layers=32, kv_heads=8, head_dim=128,
                    weight_bytes=2, kv_bytes=2):
    """Return raw weights + logical KV, excluding all runtime overhead."""
    lengths = tuple(lengths)
    counts = (params, layers, kv_heads, head_dim)
    if any(type(x) is not int or x <= 0 for x in counts):
        raise ValueError("model counts must be positive integers")
    if not lengths or any(type(t) is not int or t < 0 for t in lengths):
        raise ValueError("lengths must be nonnegative integer token counts")
    for width in (weight_bytes, kv_bytes):
        if isinstance(width, bool) or not isfinite(width) or width <= 0:
            raise ValueError("storage widths must be finite positive bytes")
    weights = params * weight_bytes
    kv = 2 * layers * kv_heads * head_dim * kv_bytes * sum(lengths)
    return {
        "weights_GiB": weights / 2**30,
        "logical_KV_GiB": kv / 2**30,
        "subtotal_GiB": (weights + kv) / 2**30,
    }


print(estimate_memory([8192]))
# weights_GiB: 14.957527..., logical_KV_GiB: 1.0,
# subtotal_GiB: 15.957527...; this is not the process peak.
```

如果算出的单设备常驻下界已超过该设备可用容量，这个配置在不分片、不 offload（把数据移到 CPU 内存等位置）的前提下不可行。反过来，下界小于容量并不保证能运行，还需要测下一项。

#### 按失败时机定位

OOM 出现在哪一步，比模型名称里的「几 B」更能缩小范围。这是检查顺序，不是仅凭症状就能确定的诊断。

| 出现时机或症状 | 先看哪一笔账 | 先做什么 |
|----------------|--------------|----------|
| 模型刚加载就失败 | 权重常驻量、加载期间临时副本、设备已有占用 | 核对真实参数数、加载 dtype、分片和空闲容量 |
| 短 prompt 能跑，长 prompt 失败 | Prefill 工作区、logits、缓存预分配 | 固定并发，逐步增加输入长度并记录峰值 |
| 生成一段后失败，或并发提高后失败 | 已缓存 token、请求副本、KV 池容量 | 固定输入，分别增加生成长度和活动请求数 |
| 反向或第一次 optimizer step 才失败 | 已保存激活、梯度、延迟创建的 optimizer state | 分开测前向、反向、首次更新 |
| 请求结束后进程占用仍高 | 可复用 KV 池、分配器缓存、仍被引用的张量 | 对照引擎池统计与 allocated/reserved，别直接认定泄漏 |

#### 最小测试矩阵与记录要求

1. **先确认任务**：是加载、prefill、decode、反向，还是第一次 optimizer step 时 OOM？它们对应不同的对象和生命周期
2. **核对存储精度**：权重、KV、梯度和 optimizer state 分别是什么 dtype？量化了权重不等于量化 KV
3. **核对长度和并发**：prompt 加生成预算是否超出模型限制？beam、并行采样和多个请求是否增加缓存副本？是否预分配最大长度？
4. **核对实现**：实际用了哪个 attention kernel？是否保存完整 logits？是否开启前缀缓存、chunked prefill、CUDA Graph？不要仅看配置名称
5. **区分统计口径**：PyTorch 的 allocated 是活跃张量占用，reserved 是分配器管理的内存，后者已包含前者；不能把两者相加。进程总占用还可能有 PyTorch 之外的分配
6. **按真实负载测峰值**：覆盖短输入、接近上限的输入、长生成和目标最大并发；记录模型 revision、引擎/驱动版本、dtype、attention backend、并行方式、实际 token 数、峰值显存与输出质量。初始化/热身和稳态都要看，多进程引擎要看实际 worker；按测量方案同步异步 GPU 操作，再记录吞吐、首 token 延迟和生成延迟

推理时还要确认没有意外构建反向图：`model.eval()` 不会关闭梯度记录，应结合 `torch.no_grad()` 或 `torch.inference_mode()`。模式差异见[PyTorch 文档](https://docs.pytorch.org/docs/2.9/notes/autograd.html#locally-disabling-gradient-computation)。

`empty_cache()` 可以归还分配器中未使用的缓存块，但不会释放仍被张量引用的内存；它不能替代减少真正存活的数据。[PyTorch 内存管理说明](https://docs.pytorch.org/docs/2.9/notes/cuda.html#memory-management)是排查 allocated/reserved 差异的入口。

每轮只改一个变量：固定输出和并发增加输入，观察 prefill；固定 prompt，分别增加输出预算和活动请求数，观察 decode 与缓存池。前面已经列出了各类优化改变哪一笔账，测完之后再对照第 5 节选择方案。

如果失败发生在训练阶段，先按[训练状态与生命周期账本](/posts/2026-06-12-llm-mem-opt-1-fundamentals/#5-训练账本先写清配置再谈-16-字节)确认梯度、主权重和 optimizer state，不能直接给此处的推理小计乘上一个固定倍数。[ZeRO](/posts/2026-06-15-llm-mem-opt-4-zero/)处理模型状态分片，[SFT 与 LoRA](/posts/2026-06-16-llm-mem-opt-5-sft/)讨论微调配置；训练激活还要结合 micro-batch、序列长度与 checkpointing 单独测量。

本文只复算了容量与 CPU 参考代码，没有在特定 GPU 上运行以上引擎配置。部署结论仍须通过这套实际负载测试验证。

<a id="7-小结"></a>

## 8. 小结

- 权重按真实参数数乘存储字节；Llama-3-8B 的 BF16 裸权重约 14.9575 GiB
- KV 按 KV 头数、缓存 token 数和缓存精度计数；GQA 的 query 头数不能直接代入 KV 公式
- 逻辑内容、已分配块与整个缓存池是不同层次，多卡还要看复制与切分
- KV Cache 复用每层历史 K/V；当前 Query 仍需读取并计算历史 attention
- FlashAttention 在线维护最大值、分母和输出累加量；缩放、mask 和归一化必须保持一致，计算量仍是稠密 attention 的二次量级
- Prefill/decode 的工作区形状不同，容量下界与运行峰值须分别验证

判断优化时，先说清省掉了哪些计算、存储或传输，再在目标 batch、长度和硬件上讨论收益。

下一步读 [大模型量化](/posts/2026-05-16-llm-quantization-guide/)：前半介绍编码与误差，后半按需阅读 GPTQ/AWQ 的推导。

## 参考资料

### 原论文与实现文档

1. Vaswani et al., [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
2. Dao et al., [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)
3. Milakov and Gimelshein, [Online normalizer calculation for softmax](https://arxiv.org/abs/1805.02867)
4. [Hugging Face Caching，v4.57.1](https://huggingface.co/docs/transformers/v4.57.1/en/cache_explanation)
5. [Triton Fused Attention 教程](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html)
6. [NVIDIA Ampere Architecture In-Depth](https://developer.nvidia.com/blog/nvidia-ampere-architecture-in-depth/)
7. Agrawal et al., [SARATHI: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills](https://arxiv.org/abs/2308.16369)

### 延伸阅读

8. [KV Caching by Hand](https://iaee.substack.com/p/kv-caching-by-hand)
9. [FlashAttention - Tri Dao | Stanford MLSys #67](https://www.youtube.com/watch?v=gMOAud7hZg4)
10. [Flash Attention 原理详解（含代码讲解）](https://zhuanlan.zhihu.com/p/676655352)
11. [Making Deep Learning Go Brrrr From First Principles](https://horace.io/brrr_intro.html)
12. [Hugging Face Flash Attention](https://huggingface.co/docs/text-generation-inference/conceptual/flash_attention)
