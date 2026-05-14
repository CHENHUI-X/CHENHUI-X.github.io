---
title: A Series on LLMs (II)
published: 2025-02-06
description: 对 LLM 推理中的提效技术进行学习，如 KV-Cache、Flash-Attention 的原理、实现及其如何避免注意力机制中的重复计算。
category: Deep Learning
tags:
- reinforcement learning
- deep learning
- nlp
- llm
draft: false
---
## 0. 前言

本系列主要是对 **LLM**(Large Language Models) 中涉及到的一些训练方法、技术进行学习.

本篇博客主要对 LLM 中的一些提效技术进行学习记录.

> 阅读前, 需要你 : 有高数基础知识, 线代基础知识, 统计学习基础知识, 当然还要有 ML 和 DL 的知识背景.
:::note
:::
## 1. KV-Cache

模型在推理时是逐 token 生成的. 当前已经输出 `How Are` 时，它在预测下一个 token 的注意力机制运作如下：

<img src="https://s2.loli.net/2025/12/06/FJoZvcHCXmUp6RD.png" alt="image.png" width="800" height="600" />

假设这一轮得到的预测结果是 `You`，接下来模型会继续预测下一个 token (假设是句号). **如果此时不使用 KV-Cache**，那么前面的注意力计算将被完整重复一遍:

<img src="https://s2.loli.net/2025/12/06/BG7n8EmzWehbOPv.png" alt="image.png" width="800" height="600" />

不过你肯定已经发现了，图中绿色区域其实是在做重复计算.

> 灰色的注意力矩阵部分可以先不用在意，因为这里使用的是因果注意力，也就是说前面的 token 无法看到后面的 token. 换句话说，这一部分的注意力权重最终必然为 0.
:::note
:::
现在我们只关注 Attention 矩阵本身. 你会发现，`token3` 的注意力权重实际上只是由它自身的 `query` 与 $W_{k}$ 相乘得出，而此时的 $W_{k}$ 正是上一轮的 $W_{k}$ (绿色) **再加上** `token3` 对应的 `key` (黄色) 后形成的更新结果.

<img src="https://s2.loli.net/2025/12/06/U9OawIixLJCXG6g.png" alt="image.png" width="800" height="600" />

基于这一点，我们就能顺理成章地进行优化：将之前所有 token 生成的 Key 矩阵缓存起来. 每当一个新 token 到来，只需把它的 key 追加到缓存的 Key 矩阵中，然后用当前 token 的 query 与更新后的 Key 矩阵做一次 attention，就能得到它的注意力权重. 整个过程不需要重新计算已有 token 的结果，从而在推理阶段实现高效的增量计算.

> Value 矩阵的处理方式也是一样的. 缓存已有 token 的 Value，新的 token 到来时生成它的 value 并追加到缓存中；随后用新的 attention score 与更新后的 Value 矩阵相乘，就能得到该 token 的最终输出.
:::note
:::
上述的过程就是 KV-Cache.

<details markdown="1">
<summary> 思考:  为什么不 Cache Query 矩阵? (展开查看)</summary>

答：因为推理时模型每一步只会对“最新的那个 token”计算它的 Query，而历史 token 的 Query 在后续步骤中根本不会被再次使用. 历史上下文信息完全由缓存下来的 Key/Value 提供，新 token 的 Query 只需要与缓存的 Key/Value 做一次注意力计算即可获得完整上下文. 因此，缓存 K/V 是必要的，而缓存 Q 没有任何用途.

</details>

## 2. Flash Attention

标准注意力的 O(N²) 代价会在长序列任务中迅速失控，如何高效利用 GPU 资源并降低计算复杂度显得尤为重要.

### 2.1 准备

### 2.1.1 前置知识

HBM（High Bandwidth Memory）和 SRAM（Static Random-Access Memory）是两种不同类型的计算机内存。

HBM 是一种面向 3D 堆叠 SDRAM 的高带宽内存接口，特点是带宽极高、能效更优，主要用于 GPU 等加速器的主存储。

SRAM 是静态随机存取存储器，通常用于高速缓存等片上存储，访问速度更快、延迟更低，但成本较高且占用较多芯片面积。

下图展示了 GPU A100 的内存层级与分布结构：

<img src="https://s2.loli.net/2025/12/15/nWKYR5rhVI2QOEz.png" alt="image.png" width="400" height="300" />

> 推荐阅读 Horace He 的博客([click here](https://horace.io/brrr_intro.html))，能让你快速了解深度学习中的计算、内存和开销.
:::note
:::
### 2.2.1 前置知识 传统 Attention 计算回顾

给定输入序列 $Q, K, V \in \mathbb{R}^{N \times d}$ , 其中 $N$ 表示序列长度，$d$ 表示每个注意力头（head）的维度，我们希望计算注意力输出 $O \in \mathbb{R}^{N \times d}$ :

$S = QK^\top \in \mathbb{R}^{N \times N}, \; P = softmax(S) \in \mathbb{R}^{N \times N}, \; O = PV \in \mathbb{R}^{N \times d}$，
其中 softmax 是按行（row-wise）应用的。

算法如下:

<img src="https://s2.loli.net/2025/12/15/ZHUrvacxpKWusnP.png" alt="image.png" width="800" height="600" />

上述计算逻辑在 GPU 的几个内存件之间的传输过程如下:
<img src="https://s2.loli.net/2025/12/15/nSxZQFBLgbTt5zP.png" alt="image.png" width="400" height="300" />

标准的 attention 实现会将矩阵 $S$ 和 $P$ 写入 HBM，这需要 $O(N^2)$ 的内存。 通常 $N \gg d$（例如在 GPT-2 中，$N = 1024$，$d = 64$）。

一方面矩阵在 HBM 与 SRAM 之间频繁传输带来了显著的时间开销；另一方面，还需要在 HBM 中存储一个空间复杂度为 $O(N^2)$ 的 Attention 矩阵。综合来看，传统的 Attention 计算在时间和内存开销上都较为昂贵。

### 2.2 计算推导

### 2.2.1 前向传播-no mask

### 2.2.2 前向传播-with mask

## Reference

[1] [https://iaee.substack.com/p/kv-caching-by-hand](https://iaee.substack.com/p/kv-caching-by-hand)

[2] [FlashAttention - Tri Dao | Stanford MLSys #67](https://www.youtube.com/watch?v=gMOAud7hZg4)

[3] [Flash Attention 原理详解(含代码讲解)](https://zhuanlan.zhihu.com/p/676655352)

[4] [Making Deep Learning Go Brrrr From First Principles](https://horace.io/brrr_intro.html)

[4] [ \[Hugging Face\] Flash Attention](https://huggingface.co/docs/text-generation-inference/conceptual/flash_attention)
