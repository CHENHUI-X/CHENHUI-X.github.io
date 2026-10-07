---
title: LLM Decoding：从概率分布到生成策略
published: 2026-05-16
description: 从 greedy 到 beam search，一文讲清 LLM decoding 策略的原理与选择。
category: Deep Learning
tags:
- llm
- decoding
- deep learning
- inference
draft: false
---

## 0. 前言

语言模型生成文本时，每一步先输出一个长度为词表大小 $V$ 的 logits 向量，再通过 softmax 得到下一个 token 的概率。省略固定的 prompt，记生成历史为 $w_{<t}$：

$$
P(w_t=i\mid w_{<t})
=\frac{e^{z_{t,i}}}{\sum_{j=1}^{V}e^{z_{t,j}}}.
$$

有了分布，接下来才是 **Decoding** 的问题：选概率最大的 token，还是按概率抽一个？是否同时保留几条生成路径？

需要分清：**模型概率高，不等于事实正确，也不等于整段回答质量高。** 概率描述模型在当前上下文中偏好的续写。Decoding 可以调整搜索范围、随机程度和输出长度，但不能把模型不知道的事实变成可靠答案。

本文用 $T>0$ 表示 Temperature，用 $L$ 表示生成长度，用 $\rho\in(0,1]$ 表示 Top-P 的概率阈值，避免把温度、长度和概率函数混在一起。这里的 token 也不一定是一个完整的词。

## 1. 最自然的想法：选概率最大的

**Greedy Search** 每一步都选条件概率最大的 token：

$$
\hat w_t=\arg\max_w P(w\mid w_{<t}).
$$

它简单、没有采样随机性，但只优化当前一步。下面的概率树展示了局部最优与整条序列最优的区别：

- 第一步：$P(A)=0.6$，$P(B)=0.4$
- 选择 $A$ 后：$P(\mathrm{EOS}\mid A)=0.51$，$P(C\mid A)=0.49$
- 选择 $B$ 后：$P(\mathrm{EOS}\mid B)=0.99$，$P(C\mid B)=0.01$

这里 EOS 是结束符，每一行条件分布都归一化为 1。Greedy 先选 $A$，再选 EOS，完成序列的概率为 $0.6\times0.51=0.306$。另一条完成序列 $B,\mathrm{EOS}$ 的概率却是 $0.4\times0.99=0.396$，更高。

所以 Greedy 不保证找到模型概率最高的完整序列。即使找到了，仍不能据此判断内容正确。某些模型和上下文还会形成重复续写；但并非每次 Greedy 都会重复。

生成也不能无限进行：通常遇到 EOS 就停止，并设置最大生成长度作为兜底。到达长度上限只是截断，不代表模型已经自然说完。

## 2. 另一个思路：按概率随机抽

**Pure Sampling** 按整个分布抽取下一个 token：

$$
\hat w_t\sim P(w\mid w_{<t}).
$$

概率为 0.28 的 token，在相同分布下有 28% 的抽中机会。它允许生成不同续写，但不保证两次结果不同。

问题在于，单个低概率 token 的概率虽然小，很多这样的 token 加起来仍可能占据可观的概率质量。例如尾部有 1000 个 token，每个概率为 $10^{-4}$，尾部总概率就是 0.1。纯采样会以 10% 的概率落入这部分。这个例子说明了尾部风险如何累积，并不是说所有模型都有固定比例的“坏 token”。

低概率也不自动等于错误。罕见人名、专业术语可能正是需要的答案。下面的裁剪方法控制的是候选范围，不能识别真假。

## 3. Temperature：调节分布的尖锐程度

Temperature 在 softmax 之前缩放 logits：

$$
P_T(w_i)=\frac{e^{z_i/T}}{\sum_j e^{z_j/T}},\qquad T>0.
$$

它改变概率之间的差距。比较两个 token：

$$
\frac{P_T(w_i)}{P_T(w_j)}=e^{(z_i-z_j)/T}.
$$

- $0<T<1$：放大高低 logits 的概率差距，分布更集中
- $T=1$：保持原始分布
- $T>1$：缩小差距，分布更平坦

如果最大 logit 唯一，$T\to0^+$ 时该 token 的概率趋于 1，采样趋近 Greedy。如果有多个并列最大值，极限分布在它们之间均分；确定性的 Greedy 还需要并列打破规则。**不能直接把 $T=0$ 代入公式**，一些接口会把零温度单独解释为 Greedy，另一些会拒绝这个值。

当 $T\to\infty$，所有有限 logits 对应的 token 概率趋于相等。若某些 token 已经通过 $-\infty$ 被屏蔽，它们不会重新加入候选集。

正温度缩放不会改变 logits 的排序，因此只做 argmax 时结果不变；在采样或 Top-P 筛选时，概率大小才会影响结果。降低 Temperature 也不会提高模型对事实的认知，只会让输出更集中于原有偏好。

## 4. Top-K：固定保留多少个候选

**Top-K Sampling** 保留概率最高的 $K$ 个 token，其余概率置零，再重新归一化。设保留集合为 $C_K$，则：

$$
q_i=\begin{cases}
\displaystyle\frac{p_i}{\sum_{j\in C_K}p_j},&i\in C_K,\\
0,&i\notin C_K.
\end{cases}
$$

这里 $p_i$ 可以是 Temperature 调整后的概率。条件是 $1\le K\le V$，并且候选集总概率大于零；$K=1$ 对应单候选，$K=V$ 不裁剪。数学上若要求恰好保留 $K$ 个，需要规定并列时如何选择；某些实现按第 $K$ 大分数做阈值筛选，遇到并列会保留更多 token。

固定 $K$ 的局限在于，同样的候选数量在不同分布下覆盖的概率质量不同：

**情况 A：分布很集中。** 前两个 token 概率为 0.85、0.10，其余 1000 个各为 0.00005，总和为 1。Top-K=50 保留前两个和 48 个尾部 token，保留质量为：

$$
0.85+0.10+48\times0.00005=0.9524.
$$

归一化后，尾部的总概率仅为 $0.0024/0.9524\approx0.252\%$。因此不能说此时“基本是在随机词里选”。准确的说法是：固定 $K$ 仍保留了一些尾部候选，但它们的实际影响要看概率质量。

**情况 B：分布很平坦。** 假设有 100 个 token，各占 0.01。Top-K=50 只保留原分布 50% 的质量，Top-K=5 只保留 5%。此时较小的 $K$ 可能过度压缩选择范围。

$K=50$ 或 $K=200$ 都只是参数取值；对于几万甚至更多 token 的词表，不能仅凭 $K=200$ 就称为“接近完整采样”。

## 5. Top-P（Nucleus）：按概率质量决定候选数

**Top-P Sampling** 不预先固定候选数，而是选择覆盖足够概率质量的最小前缀。将概率按降序排列为 $p_{(1)},\ldots,p_{(V)}$：

$$
k=\min\left\{r:\sum_{i=1}^{r}p_{(i)}\ge\rho\right\},\qquad
C_\rho=\{w_{(1)},\ldots,w_{(k)}\}.
$$

随后在 $C_\rho$ 内重新归一化并采样。**刚好使累积概率跨过阈值的 token 也要保留**，否则保留质量可能达不到 $\rho$。并列概率的顺序、浮点边界和最少保留数量可能让具体实现略有差异。

例如分布为 $(0.60,0.25,0.10,0.05)$，$\rho=0.8$ 时保留前两个，质量为 0.85，最终分布为 $(12/17,5/17,0,0)$。

直觉上，这是“保留足够多的概率质量”，而非“保留某个比例的 token 数量”：

- 分布集中时，少量 token 就能达到阈值
- 分布平坦时，需要保留更多 token
- $\rho=1$ 不再主动截掉正概率候选，但之前的其他过滤仍然有效

它也不保证候选正确。Top-P 的价值是随当前分布调整候选数量，而不是判断内容质量。[Nucleus Sampling 论文](https://arxiv.org/abs/1904.09751)讨论了这种动态截断与开放式生成的关系。

## 6. Temperature + Top-K + Top-P：顺序和归一化都重要

下面采用 **Temperature → Top-K → Top-P** 的顺序。不同服务可能采用不同处理顺序，不能只看参数名字就认为最终分布相同。

```text
输入：logits z，正温度 T，候选数 K，概率阈值 rho

1. z_scaled = z / T
2. p = softmax(z_scaled)
3. 只保留 p 中最高的 K 个候选，其余置零
4. 将剩余概率重新归一化，得到 q
5. 在 q 上按降序累加，保留首次达到 rho 的最小前缀
6. 再次归一化，并从最终分布中采样

输出：下一个 token
```

**第 4 步不能省略。** Top-P 的阈值要相对于经过 Top-K 后的分布，而不是相对于已经被截断的旧概率总和。

用一个可以手算的例子：Temperature 处理后的概率是 $(0.40,0.30,0.20,0.06,0.04)$，取 $K=3$、$\rho=0.75$：

1. Top-K 保留前三个，总质量为 0.90
2. 中间归一化得到 $q=(4/9,3/9,2/9,0,0)$
3. 前两个的累积概率为 $7/9\approx0.7778$，已经达到 0.75，因此只保留前两个
4. 最终分布为 $(4/7,3/7,0,0,0)$

如果直接用旧概率累加，前两个只有 0.70，就会错误地保留第三个。更明显的是，若阈值为 0.95，Top-K 后旧概率的总和只有 0.90，根本不可能达到阈值。

实际代码常把被删除的 logits 设为 $-\infty$，Top-P 处理器再对剩余 logits 做 softmax，因此这个“中间归一化”可以隐含在实现内部。Hugging Face Transformers **v4.57.1** 的[处理器顺序](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/generation/utils.py)和 [Top-P 实现](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/generation/logits_process.py)可用来对照这一流程。

下面只展示调用参数，假设已经准备好相互匹配的 `model`、`tokenizer` 和编码后的 `inputs`，不是完整的模型加载脚本：

```python
# 参数语义以 transformers==4.57.1 为例
output = model.generate(
    **inputs,
    do_sample=True,
    num_beams=1,
    temperature=0.7,
    top_k=50,
    top_p=0.9,
    max_new_tokens=512,
)
```

在这一版本的普通生成模式中，`do_sample=False, num_beams=1` 对应 Greedy；`do_sample=False, num_beams>1` 对应 Beam Search。不能笼统地说 `do_sample=False` 就一定是 Greedy。Temperature、Top-K、Top-P 是这里的采样参数，不用于这两种非采样模式；实际配置还应检查模型自带的 generation config。

在没有其他 logits processors 的前提下，如要做不截断的普通采样，可显式设 `temperature=1.0, top_k=0, top_p=1.0, num_beams=1`。其中 `top_k=0` 是该 API 的“关闭 Top-K”约定，与数学上保留零个候选不同。详见[版本化生成文档](https://huggingface.co/docs/transformers/v4.57.1/en/generation_strategies)。

## 7. Beam Search：同时保留多条路径

### 7.1 前缀评分与结束符

Greedy 和普通采样每步都只延续一条路径。**Beam Search** 则保留至多 $B$ 条活跃前缀：

1. 扩展每条活跃前缀，计算下一 token 的条件概率
2. 将原前缀分数与新 token 的对数概率相加
3. 在扩展结果中选择高分的非结束前缀，继续搜索
4. 将产生 EOS 的候选放入已完成集合，不再继续扩展
5. 满足停止条件或达到最大长度后，从候选中选出结果

不计结束候选的额外处理时，每一步约有 $B\times V$ 个扩展。实现可能先取更多候选，再分别处理 EOS 与活跃前缀，不是每轮必然得到 $B$ 条“完整句子”。

固定 prompt 后，长度为 $L$ 的候选分数是：

$$
s(w_{1:L})=\sum_{t=1}^{L}\log P(w_t\mid w_{<t}).
$$

对于自然结束的完整序列，本文把最后一个 EOS 也计入 $L$ 和概率。没有 EOS 的前缀分数与完整序列概率不能混为一谈。

为什么累加 log probability？若 100 步的条件概率都为 0.1，乘积是 $10^{-100}$，float32 会下溢为零，而其自然对数是约 $-230.2585$。对数是严格递增函数，比较对数概率与比较原概率等价，又能避免长乘积下溢。实现通常直接计算稳定的 log-softmax。

### 7.2 长度评分会改变搜索目标

原始对数概率每增加一个 token，都会加上一个非正数。因此同一前缀越延伸，原始分数不会增加；不同长度的候选比较时，可能出现偏短倾向。它不是“短句总比长句分数高”的定理。

一种常见修正是长度归一化：

$$
s_\alpha(w_{1:L})=\frac{s(w_{1:L})}{L^\alpha},\qquad\alpha\ge0.
$$

当 $\alpha=0$ 时回到原始分数。$\alpha>0$ 会相对抬高长候选的负分数，但不会保证生成一定更长。例如两个候选的概率与长度分别为 $(0.25,2)$ 和 $(0.24,4)$，按原始概率前者更高；若 $\alpha=1$，则分数约为 $-0.6931$ 和 $-0.3568$，后者更高。**此时优化目标已经不是最大化原始序列概率。**

不同系统的长度惩罚公式，以及长度是否计入 prompt、EOS，都可能不同，应以具体实现为准。EOS 候选也要按同一评分约定比较。停止策略同样重要：完成了 $B$ 条候选，不代表在任意长度评分下继续搜索都不可能找到更好的候选。[Transformers 的生成配置](https://huggingface.co/docs/transformers/v4.57.1/en/main_classes/text_generation)分别定义了 `length_penalty` 与 `early_stopping`。

### 7.3 Beam width 与适用范围

$B=1$ 在标准逐步保留最高分前缀、相同停止规则下退化为 Greedy。更大的 $B$ 扩大搜索范围，也增加计算与缓存开销。它仍是近似搜索：某些前缀一旦被剪掉，后续的高分续写就无法找回，因此不保证找到全局最高分序列。

Beam Search 常用于翻译等需要候选搜索的任务；开放式对话和故事生成则经常使用采样来获得多样性。这个区别不是硬规则。更大的 beam、更高的模型分数，都不自动意味着更准确、更自然，最终仍应使用具体任务的评价方法。

## 8. 从 Softmax 再看 Temperature

还可以通过导数理解温度的作用。令 $p_i(T)=\operatorname{softmax}(\mathbf z/T)_i$，则：

$$
\frac{\partial p_i}{\partial z_j}
=\frac{1}{T}p_i(\delta_{ij}-p_j).
$$

对自身 logit 的导数是 $p_i(1-p_i)/T$。这里既有 $1/T$，也有随温度变化的概率，不能只看前者就断言“温度越低，梯度越大”。

在固定 logits 且最大值唯一时，$T\to0^+$ 会使分布饱和为 one-hot，其中 $p_i(1-p_i)$ 以指数速度趋于零，足以抵消 $1/T$ 的增长，因此导数也趋于零。若最大值并列，这个结论不再适用。$T\to\infty$ 时，导数则因 $1/T$ 趋于零。

这解释的是概率对 logits 扰动的敏感度。Temperature 缩放不是重新训练，也不是事实置信度校准；推理采样本身不需要计算这些梯度。

## 9. 实践指南

先选与任务匹配的解码方式，再调参数，比同时堆叠所有过滤器更容易解释实验结果：

| 目标 | 可以从哪里开始 | 需要额外检查 |
|------|----------------|--------------|
| 稳定、可复现的短输出 | Greedy | 停止条件、重复、事实或代码正确性 |
| 翻译等候选搜索 | 小规模 Beam Search | 长度评分、EOS 与任务指标 |
| 对话、多种表述 | Temperature + Top-P | 多样性、偏题、重复 |
| 限制候选数量 | 在采样中加入 Top-K | 是否剪掉罕见但正确的 token |
| 事实或科学回答 | 用任务验证选择解码参数 | 检索来源、计算核验，不能靠低温代替 |

例如 $T=0.7,\rho=0.9$ 可以作为一组对照实验参数，但不是跨模型通用最优值。比较时固定 prompt、模型版本、长度限制和评价方法；采样应多运行几次并记录随机种子。即使不采样，跨硬件或不同 kernel 的数值差异也可能影响并列或近似并列的选择。

## 10. 总结

| 策略 | 改变的对象 | 需要记住的边界 |
|------|------------|----------------|
| Greedy | 每步只选最高概率 token | 局部最优，不保证整段最优 |
| Temperature | logits 差距与概率锐度 | 正温度不改变排序，不保证正确性 |
| Top-K | 候选数量 | 固定数量不等于固定概率覆盖率 |
| Top-P | 候选覆盖的概率质量 | 在当前归一化分布上计算阈值 |
| Beam Search | 同时保留的候选路径 | 近似搜索，长度评分与停止规则很重要 |

解码可以决定怎样使用模型给出的分布。理解每一步修改了什么，比记一套万能参数更有用。组合采样时，要明确处理顺序和中间归一化；评价结果时，要分别检查模型概率、输出多样性与内容正确性。

### 参考资料

1. Holtzman et al., [The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751). ICLR 2020. Nucleus Sampling 与开放式生成
2. Fan et al., [Hierarchical Neural Story Generation](https://arxiv.org/abs/1805.04833). ACL 2018. Top-K Sampling 在故事生成中的应用，不据此断言其最早起源
3. Graves, [Sequence Transduction with Recurrent Neural Networks](https://arxiv.org/abs/1211.3711). 2012. 不同模型中的 Beam Search 应用背景
4. Hinton et al., [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531). 2015. Temperature 在知识蒸馏中的使用
5. Vaswani et al., [Attention Is All You Need](https://arxiv.org/abs/1706.03762). NeurIPS 2017
6. Hugging Face [Generation Strategies](https://huggingface.co/docs/transformers/en/generation_strategies)；本文参数说明对应 [v4.57.1](https://huggingface.co/docs/transformers/v4.57.1/en/generation_strategies)，后续版本请核对实际配置与源码
