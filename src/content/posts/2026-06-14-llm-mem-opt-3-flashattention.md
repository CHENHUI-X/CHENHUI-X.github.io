---
title: 大模型显存（五）：FlashAttention 为什么能省显存
published: 2026-06-14
description: 从普通 attention 的大矩阵和 GPU 存储层级出发，逐步手算 FlashAttention 的分块前向、在线 softmax 递推与因果掩码，并用 CPU 参考代码核对结果。
category: Deep Learning
tags:
- llm
- memory
- llm-memory-optimization-series
- flash-attention
draft: false
---

这是大模型显存系列的第五篇。上一篇[Prefill、Decode 与 KV Cache](/posts/2026-06-13-llm-mem-opt-2-inference/)讲到，KV Cache 可以复用历史位置的 K/V，但不会消除一次 attention 计算内部的中间矩阵。这一篇先看看这些大矩阵从哪里来，再解释 FlashAttention 怎样通过分块计算和合并结果来处理它们。

只需知道：Query（Q）与 Key（K）比较得到分数，softmax 把分数变成权重，再用这些权重对 Value（V）加权。下面从单头公式开始，逐步手算分块合并，再用 CPU 参考代码核对结果；GPU kernel 的性能仍需在目标设备上实测。

KV Cache 解决的是不同生成步骤之间重复计算旧结果的问题。**FlashAttention** 则关注另一个问题：一次 attention 计算中，怎样减少 HBM 和片上存储之间的数据搬运？

本文先讨论 query 和 key 长度同为 $N$ 的整段 self-attention，$N$ 是 token 数，$d_h$ 是每个 head 的向量维度。公式中的大写 $S$ 表示 **scores 分数矩阵**。复杂度里的 $O(\cdot)$ 描述计算量随规模增长的量级，省略常数因子，别与后面记作 $O$ 的输出矩阵混淆。

它是一种 exact attention 的实现方式：在精确实数运算下计算相同的注意力结果，实际浮点结果可能因运算顺序不同而有小差异。对稠密 attention 而言，它仍需 $O(N^2d_h)$ 量级的计算，**并没有把注意力的算术复杂度变成线性**；主要收益是避免在 HBM 中保存完整 $N\times N$ 中间矩阵，并减少访存。

<span id="attention-definition"></span>

<span id="622-传统-attention-计算回顾"></span>

<span id="622-先看完整保存中间矩阵的算法"></span>

## 1. 普通 attention 为什么产生大矩阵

为简化推导，设单个 head 的 $Q,K,V\in\mathbb R^{N\times d_h}$，Value 维度也取 $d_h$，且不使用 dropout。**Q 的一行是一个查询位置，K 和 V 的一行是同一个可被读取的位置**；每行有 $d_h$ 个数。$K^\top$ 把 K 的行转成列，所以 $QK^\top$ 的第 $i$ 行、第 $j$ 列，就是位置 $i$ 的 Query 与位置 $j$ 的 Key 的点积。

标准计算为：

$$
S=\frac{QK^\top}{\sqrt{d_h}}+M\in\mathbb R^{N\times N},\qquad
P=\operatorname{softmax}_{\mathrm{row}}(S),\qquad
O=PV\in\mathbb R^{N\times d_h}.
$$

无 mask 时 $M=0$；causal mask 则在未来位置填入 $-\infty$。softmax 按行计算，让每个 Query 对允许访问的 Key 分配权重。

**为什么输入只有 $N$ 行向量，中间却出现了 $N\times N$ 个数？** 因为每个 Query 都要与全部 Key 比较：纵向数 Query 位置，横向数 Key 位置。

这里 $S$ 是分数矩阵，$P$ 是逐行归一化的权重矩阵。例如 $P_{ij}$ 表示“第 $i$ 个 Query 给第 $j$ 个 Value 多大权重”，所以输出第 $i$ 行为 $O_i=\sum_jP_{ij}V_j$。它不是把所有 Query 混成一个输出，而是每一行独立归一化、独立汇总。

这里讨论的是**完整保存中间矩阵的分步实现**，不表示所有现代 attention kernel 都这样执行。若把 $S$、$P$ 写入 HBM，后续 softmax 和与 V 相乘又需要读它们，就产生 $O(N^2)$ 规模的中间存储与读写。例如 batch 为 1、32 个 Query heads、$N=32768$，一份 BF16 分数张量就有 $32\times32768^2\times2$ bytes，即 64 GiB（约 68.7195 GB）；FP32 则是 128 GiB。这对应一份完整分数张量，不是模型权重或 KV Cache。这里沿用上一篇的架构外推，不代表原版 Llama-3-8B 支持 32K 上下文。因果 mask 遮住未来位置，也不会让普通稠密张量自动只存一半。

<span id="gpu-memory"></span>

<span id="621-gpu-内存层级"></span>

<span id="621-数据在显存和片上存储之间怎样搬运"></span>

## 2. 数据在 HBM 和片上存储之间怎样搬运

先把两种存储分开：**HBM 是 GPU 上容量较大的显存，片上寄存器和 shared memory 是计算单元附近容量小得多的存储。** Q/K/V 等数据通常从 HBM 读入，计算时会使用片上资源；把中间结果写回 HBM、下一步再读进来，也要花时间和带宽。

HBM 是高带宽的堆叠 DRAM，并不是 SRAM。片上还有由硬件管理的 cache，但它不能与 kernel 显式安排的 shared memory、寄存器混为一谈。一个计算块实际可用的容量，受硬件资源与 kernel 配置限制，不能拿“整张 GPU 的片上存储总量”当作它的预算。不同 GPU 的容量和带宽也不同，具体例子可见 [NVIDIA Ampere 架构说明](https://developer.nvidia.com/blog/nvidia-ampere-architecture-in-depth/)。

FlashAttention 的做法不是把整张注意力矩阵塞进片上，而是**只搬入能放下的一小块，算完并合入输出后，复用这块临时空间**。上面的分步算法把完整中间矩阵写进 HBM，FlashAttention 则改变这些中间结果的保存方式。

如果想进一步理解算力、带宽和数据搬运之间的关系，可阅读 [Making Deep Learning Go Brrrr From First Principles](https://horace.io/brrr_intro.html)。

<span id="tiled-attention"></span>

## 3. 只保留小块，怎样接着算下去

**如果只保留一个小分数块，已经算过的部分该存在哪里？** FlashAttention 不保存每个旧分数，而是把它们合入每个 Query 的几个累计量。

![Tri Dao 的 FlashAttention 两块前向原图：红色为第一块，蓝色为第二块，橙色虚线框为不写入 HBM 的片上中间结果，右侧重缩放累计输出](/images/llm-series/annotated/foundation-flashattention-two-blocks-zh.png)

*图：FlashAttention 的两块前向与输出重缩放。原图作者 Tri Dao，来自 Stanford CRFM [FlashAttention-2：FlashAttention Recap](https://crfm.stanford.edu/2023/07/17/flash2.html#flashattention-recap)，与 [FlashAttention-2 论文 Fig. 1](https://tridao.me/publications/flash2/flash2.pdf#page=4) 对应；中文编号与引线为辅助标注，原作者内容与标注请结合下文阅读。[查看未加中文标注的原图](/images/llm-series/sourced/tri-dao-flashattention-two-blocks.png)。*

**先认图例，再沿红色一列向下看，最后看蓝色一列。** 左下角浅蓝底、蓝色点线边框表示存放在 HBM 的数据；白底、橙色虚线框表示在片上 SRAM 中计算、不把这些中间矩阵写入 HBM。红色和蓝色的公式分别属于第一、第二个 K/V 块，不是在区分缓存与新 token。

- **① 同一批 Query 读取不同块：** Q 的每行是一个查询位置，转置后的 K 的每列是一个可读位置，因此 $S^{(1)}=Q(K^{(1)})^\top$ 的行属于 Query、列只覆盖第一块 Key。第二列同理得到 $S^{(2)}$，没有把另一批 Query 混进来。
- **② 红、蓝是块编号：** 上标 $(1)$、$(2)$ 分别表示第一块、第二块，不是平方，也不是历史缓存与新 token。红色指数块对应红色 $V^{(1)}$，蓝色指数块对应蓝色 $V^{(2)}$，同一块的 Key 与 Value 必须配对。
- **③ 指数值还不是最终权重：** 橙框中的 $A^{(1)}$、$A^{(2)}$ 是分数取指数后的值，尚未除以整行共同分母。不能把每块单独 softmax 后的结果直接相加或平均。
- **④ 分母按每个 Query 行累计：** $l^{(1)}$ 是第一块的逐行指数和，$l^{(2)}$ 是已经合并两块的指数和，不是第二块单独的和。图中 $l$ 对应下一节的 $\ell$；此简化图尚未减行最大值来保证数值稳定。
- **⑤ 旧输出先换成共同分母：** 旧输出原来除以 $l^{(1)}$，合并第二块后应除以 $l^{(2)}$，所以先乘 $l^{(1)}/l^{(2)}$。右侧灰箭头就是这个重缩放，避免旧块仍使用局部分母。
- **⑥ 再加入新块贡献：** 第二块的指数值也除以相同的 $l^{(2)}$，再乘 $V^{(2)}$，加到已经重缩放的旧输出上。这时两个块才处于同一归一化尺度，得到的是同一个 Query 行的累计输出。

**橙色虚线框里的中间结果，不需要组成完整大矩阵再写回 HBM。** 实现时依次算块、更新每行累计量，再复用临时空间；图上并排摆着两个块是为展示关系，不表示必须把它们同时完整留在片上。浅蓝色的输入与输出仍要存储，持久 KV Cache 也没有被消除。

这张原图为了讲清分块合并，省略了减行最大值；图面也没有展开 $1/\sqrt{d_h}$ 缩放和 mask。完整 attention 定义以上文公式为准，稳定的分块合并过程见下一节。不要照图直接计算大数的指数。

把当前 Q 块记作 $Q_I$，即一块 Query 行，把 K/V 块记作 $K_J,V_J$，即一块可读位置。它们产生的临时分数只有 $B_r\times B_c$，其中 $B_r$ 是 Query 块行数，$B_c$ 是 Key/Value 块行数。对每一行，保留最大值 $m$、指数和 $\ell$、加权 Value 累加量 $u$，就能接着处理下一块，最后得到输出。这几个量的含义，下一小节用三个数手算。

所有允许的 Query–Key 配对仍要计算，只是计算结果不再全部长期留在 HBM；片上并没有藏着另一份完整 $N\times N$ 矩阵。Q/K/V 输入、输出和持久 KV Cache 也仍然需要存储。实际 kernel 的循环顺序、分工和状态写回方式可不同；[FlashAttention 原论文](https://arxiv.org/abs/2205.14135)和 [FlashAttention-2](https://tridao.me/publications/flash2/flash2.pdf)给出了具体算法。

剩下的关键问题是：**不同块的分母不同，怎样让分块结果等于整行 softmax？** 不能把每块单独归一化后的结果直接相加或平均。

<span id="online-softmax"></span>

<span id="623-在线-softmax无-mask-的前向合并"></span>

<span id="623-在线-softmax每次读一块怎样合成同一个结果"></span>

## 4. 在线 softmax：每次读一块，怎样合成同一个结果

先暂时不考虑指数溢出，用一个 Query、三个可读位置来理解。取 scores 为 $[0,\log2,\log4]$，标量 Values 为 $[2,4,8]$。这里 $\log$ 是自然对数，所以取指数得到 $[1,2,4]$。softmax 要把它们除以共同的总和 $1+2+4=7$，输出就是 $[1/7,2/7,4/7]$ 对 Values 的加权和。

**如果前两项是一块、最后一项是一块，需要记住哪些数，才能算出同一个输出？** 把它对应到上图的红色第一块与蓝色第二块，先算两份和，再加入数值稳定处理。

**第一步：先合并两份和，最后才做除法。** 第一块贡献分母 $1+2=3$、分子 $1\times2+2\times4=10$；第二块贡献分母 $4$、分子 $4\times8=32$。两份和分别相加，得到：

$$
o=\frac{10+32}{3+4}=\frac{42}{7}=6.
$$

对应原图，第一块 $l^{(1)}=3$、$O^{(1)}=10/3$，合并后 $l^{(2)}=7$。右侧的重缩放就是 $(3/7)(10/3)+(4/7)8=6$。

如果各块先做 softmax，第一块输出 $10/3$，第二块输出 $8$；直接平均得到 $(10/3+8)/2=17/3\ne6$。错误在于给了两块相同权重，而它们的指数总量分别是 3 和 4。记住分子与分母，就保留了合并需要的信息；旧的每一个分数不必继续保存。

**第二步：数值很大时，给两个和统一一个缩放基准。** 直接算 $e^s$ 可能溢出，稳定 softmax 会减去最大值；分子、分母一起缩小相同比例，比值不变。在第一块里，最大值是 $\log2$，指数变成 $[1/2,1]$，因此保存：

$$
m=\log2,\qquad \ell=1/2+1=3/2,\qquad
u=(1/2)\times2+1\times4=5.
$$

读到第二块后，新的最大值变成 $\log4$。旧状态按“除以 2”的基准保存，现在要统一成“除以 4”，所以旧的两个和都再乘 $2/4=1/2$。新块在新基准下的指数是 $e^{\log4-\log4}=1$，于是：

$$
\begin{aligned}
\alpha&=1/2,\\
\ell_{\mathrm{new}}&=(1/2)(3/2)+1=7/4,\\
u_{\mathrm{new}}&=(1/2)5+1\times8=21/2,\\
o&=(21/2)/(7/4)=6.
\end{aligned}
$$

这里 $m$ 是当前已处理位置的最大分数，$\ell$ 是同一基准下的指数和，$u$ 是同一基准下的加权 Value 和。这个例子里的 Value 是一个数，所以 $u$ 也是一个数；实际 Value 是向量时，就对向量每个分量做同样的累计。**状态中的分子、分母必须始终使用同一个最大值作为基准。**

**第三步：把同一个过程写成一般公式。**

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

实际 GPU 会同时处理一块 Query：$Q_I\in\mathbb R^{B_r\times d_h}$，配合 $K_J,V_J\in\mathbb R^{B_c\times d_h}$，临时 scores 只需 $B_r\times B_c$。上面的 $m,\ell$ 变成每行各自一个统计量，$u$ 变成 $B_r\times d_h$ 的累加器，合并公式按行广播。可以把它想成 $B_r$ 份刚才的单行手算同时进行：它们读取同一块 K/V，但每个 Query 都有独立的分数、最大值、分母与输出累加量。完整 $N\times N$ 矩阵不需要写入 HBM；输入、输出和 KV Cache 本身仍然占用显存。

以上写法用于说明分块计算为什么能得到相同结果，实际 kernel 的循环顺序、寄存器安排、分块大小会不同。可对照 [FlashAttention 论文](https://arxiv.org/abs/2205.14135)和 [Triton Fused Attention 教程](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html)。CPU 上复算公式不能代替 GPU 性能测量。

<span id="causal-mask"></span>

<span id="624-causal-mask-与数值边界"></span>

<span id="624-因果掩码位置偏移与全遮蔽行"></span>

<span id="624-加上因果掩码后哪些位置该跳过"></span>

## 5. 加上因果掩码后，哪些位置该跳过

causal mask 不改变合并思路，只改变允许参与求和的 Key。若 Query 的绝对位置为 $i$、Key 的绝对位置为 $j$，普通 causal attention 允许 $j\le i$：

$$
s_{ij}=\begin{cases}
q_i k_j^\top/\sqrt{d_h},&j\le i\text{ 且 Key 有效},\\
-\infty,&\text{其他情况}.
\end{cases}
$$

**mask 必须在求最大值和指数之前生效。** 被屏蔽的位置贡献为零，不参与局部最大值与求和。处理分块时，可以跳过完全位于未来的块；跨过对角线的块仍需逐元素屏蔽。

有 KV Cache 时，Query 与 Key 的长度可能不同，不能直接把当前 Query 块的局部行号当作绝对位置。例如已经缓存 5 个位置，再处理 2 个新位置，按零起始编号，Query 的位置是 5、6，而 Key 的位置是 0 到 6：第一行可见 0 到 5，第二行可见 0 到 6。对一个 $2\times7$ 矩阵直接套普通左上角下三角 mask，会错误地只允许前面很少的 Key。

以下两种情况容易产生 NaN：

1. **当前块对某一行全被遮蔽。** 这一行应跳过该块并保持旧状态。如果旧状态仍是 $m=-\infty$，机械计算 $-\infty-(-\infty)$ 会得到 NaN；不能靠“最后乘零”补救
2. **整行没有任何有效 Key。** 此时 $\ell=0$，标准 softmax 没有定义。有效的 causal token 通常至少能看见自己；padding query 则应排除计算，或由实现明确约定输出零并确保下游忽略。输出零是接口约定，不是对未定义 softmax 的数学等价结果

<span id="cpu-reference"></span>

### 5.1 用 CPU 代码核对一行结果

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

如果一整行都被屏蔽，这段代码会直接报错，方便调用方检查数据和 mask。代码假设有效 Values 也为有限数值；真实实现还要处理 padding、数据类型、溢出保护和设备限制。单头无 mask、causal mask、非整块长度以及很大正负 scores，都可以用它与稳定的直接 softmax 对照检查。

<span id="backward-recomputation"></span>

### 5.2 反向也需要注意力权重，为什么没有把大矩阵存回来

本节沿用前文 dropout 为 0 的条件。$L$ 表示最终损失，$i$ 是 Query 位置，$j$ 是 Key/Value 位置。前向不保存完整 $P$，并不意味着反向不需要它。FlashAttention 保留可供反向读取的 Q、K、V、输出 O 和每行的归一化统计；保留输入并不意味着必须另复制一份。反向需要哪一小块，就用 Q/K 重算该块分数，再还原概率：

$$
P_{ij}=\frac{e^{S_{ij}-m_i}}{\ell_i}.
$$

这里使用的是该行**前向结束后**的最大值 $m_i$ 与指数和 $\ell_i$，不能用某一块自己的统计量。这样无需为了一个概率块重新做整行 softmax。mask 仍按前向的有效位置处理：允许的位置参与计算，被遮蔽的位置概率为零、不贡献梯度；全遮蔽行仍需遵守前文的接口约定。

继续前文 $P=[1/7,2/7,4/7]$、$V=[2,4,8]$、输出 $o=6$ 的单行标量例子。设后续网络传来的输出梯度为 $g=\partial L/\partial o=1$，每个 Value 的梯度是 $gP_j$；分数梯度则是：

$$
\frac{\partial L}{\partial v_j}=gP_j,\qquad
\frac{\partial L}{\partial s_j}=P_jg(v_j-o).
$$

代入三个位置，得到：

$$
\begin{aligned}
\frac{\partial L}{\partial V}&=[1/7,\ 2/7,\ 4/7],\\
\frac{\partial L}{\partial S}
&=[(1/7)(2-6),\ (2/7)(4-6),\ (4/7)(8-6)]\\
&=[-4/7,\ -4/7,\ 8/7].
\end{aligned}
$$

读取前两个位置时，重算它们的概率并累加相应梯度；再读第三个位置，补完剩余贡献。这里只写出整行数字方便核算，执行时仍不需要把整张 $P$ 长期留在 HBM。多个 Query 会共同贡献 K、V 的梯度，因此各块的贡献要累加，不能相互覆盖。

实际 Value 是向量时，$g=\partial L/\partial o$ 也是同维向量，分数梯度写成 $\partial L/\partial s_j=P_j\langle g,v_j-o\rangle$。符号 $\langle\cdot,\cdot\rangle$ 表示两个向量的内积；输出 $o$ 已在前向算好，不需要为每一块重新计算整行输出。再沿 $S=QK^\top/\sqrt{d_h}$ 的链式法则，把局部分数梯度累加到 Q、K 的梯度，缩放因子也要保留。

这是一种针对 attention 反向设计的分块重算。它与[微调篇的 activation checkpointing](/posts/2026-06-16-llm-mem-opt-5-sft/#9-gradient-checkpointing用重算减少保存)都有“用重算换存储”的思想，但作用范围和配置不同：后者可对一层或一段网络重算，FlashAttention 在 attention 算子内部安排分块。组合使用时，attention 之外仍有 FFN 等激活需要处理。输入输出、统计量、梯度与模型状态也仍占内存，省掉大 attention 中间矩阵不等于消除全部训练显存。

含 dropout 时还必须一致地恢复前向随机掩码，不能在反向重新随意采样。上面的 CPU 代码只核对前向；本节反向手算也不是完整 GPU kernel。具体的分块梯度累加和保存状态可对照 [FlashAttention 原论文附录的反向推导与算法](https://arxiv.org/html/2205.14135v2#A2.SS2)。

## 6. 省下了什么，还留下了什么

- 完整 scores / softmax 中间矩阵不必写入 HBM；每次只处理一个分数块，并维护逐行统计量
- 旧分子和旧分母必须一起按新最大值重缩放，才能保持同一归一化尺度
- Q/K/V 输入、输出、模型权重和持久 KV 仍需存储；不能把整个推理工作区记成零
- 稠密 attention 仍有二次量级的计算；减少访存不等于改变注意力定义，也不保证所有设备与负载都有相同收益

回到上一篇的[工作区容量估算](/posts/2026-06-13-llm-mem-opt-2-inference/#4-prefill-与-decode临时工作区为何差这么多)，就能理解为什么启用高效 attention 后不能再无条件加上一份完整分数矩阵，但仍要测真实峰值。

至此，显存子系列从单位与生命周期，走过训练状态分片、微调和推理，再回到算子内部的中间存储。可以回到[大模型系列目录](/posts/2026-06-03-llm-knowledge-map/)选择后续主题；若要把高效 attention 用在训练中，可结合前面的[微调实践](/posts/2026-06-16-llm-mem-opt-5-sft/)与本文的反向计算边界，分别核对激活和模型状态。

## 参考资料

1. Vaswani et al., [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
2. Dao et al., [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)
3. Dao, [FlashAttention-2](https://tridao.me/publications/flash2/flash2.pdf) 与 [Stanford CRFM 讲解](https://crfm.stanford.edu/2023/07/17/flash2.html#flashattention-recap)
4. Milakov and Gimelshein, [Online normalizer calculation for softmax](https://arxiv.org/abs/1805.02867)
5. [Triton Fused Attention 教程](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html)
6. [NVIDIA Ampere Architecture In-Depth](https://developer.nvidia.com/blog/nvidia-ampere-architecture-in-depth/)
7. [FlashAttention - Tri Dao | Stanford MLSys #67](https://www.youtube.com/watch?v=gMOAud7hZg4)
8. [Flash Attention 原理详解（含代码讲解）](https://zhuanlan.zhihu.com/p/676655352)
9. [Making Deep Learning Go Brrrr From First Principles](https://horace.io/brrr_intro.html)
10. [Hugging Face Flash Attention](https://huggingface.co/docs/text-generation-inference/conceptual/flash_attention)
