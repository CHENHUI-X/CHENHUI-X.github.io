---
title: Perplexity：语言模型的评分与比较
published: 2024-04-10
description: 深入浅出地解释 NLP 中困惑度（Perplexity）的概念，理解它如何衡量语言模型对样本的预测能力，以及其与概率的关系。
category: Deep Learning
tags:
- nlp
- perplexity
- deep learning
draft: false
---
## 0. 前言

评价语言模型，可以先问一个具体问题：给它一段真实文本，每次只让它看到前文，它会给接下来真正出现的 token 多大概率？

**困惑度（Perplexity，常写作 PPL）把这些真实 token 的概率汇总成一个数。概率的几何平均越大，困惑度越低。** 这里的“真实”指评测文本中实际出现的 token，不表示文本内容一定符合事实。

因此，PPL 衡量的是模型对这份文本的预测表现。它不是模型对自己生成内容的笼统“信心”，也不直接等于回答准确率。一个模型即使很确信错误答案，也可能给真实目标很低的概率，从而得到很高的困惑度。

:::note
本文讨论自回归语言模型的困惑度。评测时使用真实前文，不把模型刚生成的错误 token 接回去继续评分。BERT 这类遮盖语言模型通常需要另外定义评价方式，不能直接套用这里的序列概率公式。
:::

## 1. 举个例子

假设词表只有 6 个 token：`a`、`the`、`red`、`fox`、`dog` 和 `.`。待评测文本是 `a red fox .`，恰好由 4 个 token 组成。真实模型的 token 不一定对应一个完整单词，这里只是为了便于演示。

记文本为 $W=(w_1,\ldots,w_n)$，模型概率为 $q$。利用概率的链式法则：

$$
q(W)=\prod_{t=1}^{n}q(w_t\mid w_{<t}).
$$

其中 $w_{<t}$ 表示第 $t$ 个 token 之前的真实文本。第一个 token 的概率可理解为在固定起始上下文下的概率，后文省略这个条件。

模型对第一个位置给出如下分布：

<img src="https://s2.loli.net/2024/04/10/IfNJ1tRBwbTH8lP.png" alt="第1个字.png" width="600" height="400" />

因为真实的第一个 token 是 `a`，取出 $q(\text{a})=0.4$。接着把真实的 `a` 作为前文，得到第二个位置的分布：

<img src="https://s2.loli.net/2024/04/10/vgHxO3nFumXrQAc.png" alt="第2个字.png" width="600" height="400" />

真实的第二个 token 是 `red`，所以取 $q(\text{red}\mid\text{a})=0.27$。即使另一个 token 概率更高，也不能用它替代 `red` 的概率。后两个位置同理：

<div style="display: flex;">
    <img src="https://s2.loli.net/2024/04/10/UwFikWIL9tNPJRA.png" alt="Image 1" style="width: 100%;">
    <img src="https://s2.loli.net/2024/04/10/k8DHmfxSJIuOTpY.png" alt="Image 2" style="width: 100%;">
</div>

$$
\begin{aligned}
q(W)
&=q(\text{a})\,q(\text{red}\mid\text{a})\,q(\text{fox}\mid\text{a red})\,q(\text{.}\mid\text{a red fox})\\
&=0.4\times0.27\times0.55\times0.79\\
&=0.046926.
\end{aligned}
$$

### 1.1 这是不是“整句话生成完毕”的概率？

上面的乘积表示前 4 个 token 恰好为 `a red fox .` 的概率，**没有包含“此时停止生成”这一事件**。示例中的句点只是普通 token，不自动等于结束标记。

如果模型使用 EOS 表示结束，那么“生成这段文本，然后结束”的概率应为：

$$
q(W,\mathrm{EOS})=q(W)\,q(\mathrm{EOS}\mid W).
$$

这是在包含 EOS 的词表上定义的概率；六词示例没有给出 EOS 概率，不能据此算出完整终止事件的概率。若评测把 EOS 也作为预测目标，负对数概率和计数分母都要加入这一项，目标数从 $n$ 变为 $n+1$。若只是因为达到长度上限而截断，则属于外部停止规则，不能当作模型预测了 EOS。[1]

### 1.2 为什么要按 token 数取平均？

联合概率中连乘的项数取决于文本长度。为避免仅因句子更长就得到更小的值，可以取每个真实 token 条件概率的几何平均：

$$
\bar q(W)=q(W)^{1/n}.
$$

本例为：

$$
\bar q(W)=0.046926^{1/4}\approx0.465429.
$$

这是每个 token 的平均概率尺度，不是一个对所有句子归一化的概率分布。按长度取平均也不意味着任意两段文本都能公平比较模型，它们本身的难度可能不同。

## 2. 如何计算困惑度

困惑度就是上述几何平均的倒数：

$$
\begin{aligned}
\operatorname{PPL}(W)
&=q(W)^{-1/n}\\
&=\exp\left(-\frac1n\sum_{t=1}^{n}\ln q(w_t\mid w_{<t})\right).
\end{aligned}
$$

本例中，$\operatorname{PPL}(W)=0.046926^{-1/4}\approx2.148556$。

作为参照，如果模型在每个位置都对 6 个 token 均匀分配概率，那么：

$$
\operatorname{PPL}(W)=\left((1/6)^4\right)^{-1/4}=6.
$$

直观地说，困惑度可以看成“达到相同平均预测难度时，相当于在多少个等可能选项中选择”。它不要求是整数，也不是每一步实际候选词的数量。

对于离散 token，PPL 至少为 1，但没有“最多等于词表大小”的保证。若真实目标的概率都只有 $0.01$，PPL 就是 100；若某个计分目标概率为 0，则 PPL 为 $+\infty$。

### 2.1 只统计真正参与评分的目标

实际评测里，不一定每个输入 token 都计入损失：padding 是为了补齐长度，重叠窗口中的一部分 token 只提供上下文；对话模型还可能只对回答部分计分。

令 $S$ 为**完成预测与标签对齐后，实际参与评分的位置集合**，$N=|S|>0$，$c_t$ 为第 $t$ 个目标使用的上下文，则：

$$
\operatorname{PPL}
=\exp\left(-\frac1N\sum_{t\in S}\ln q(w_t\mid c_t)\right).
$$

要点是分子和分母使用同一组目标：[2]

- padding、忽略标签和只作上下文的位置，都不进入分子或分母
- 自回归模型通常用前一位置的输出预测后一位置的 token，要在标签移位后统计有效目标；不能直接拿输入长度当作 $N$
- EOS 若参与评分，就同时进入概率和计数；BOS 若只提供起始上下文，则不计作预测目标
- 如果 $N=0$，这批数据没有可评测目标，PPL 未定义，不能返回 1

### 2.2 整个语料库如何汇总？

设第 $j$ 段文本有 $N_j$ 个有效目标，总负对数似然为 $L_j$。语料库级别应当先累加损失和目标数，再取指数：

$$
\operatorname{PPL}_{\mathrm{corpus}}
=\exp\left(\frac{\sum_jL_j}{\sum_jN_j}\right)
=\prod_j\operatorname{PPL}_j^{N_j/\sum_kN_k}.
$$

不能直接对各句或各批次的 PPL 做算术平均，也不能在有效目标数不同时直接平均每批的平均损失。例如，两段文本分别有 1 个和 3 个目标，PPL 分别为 2 和 8，那么整体 PPL 是 $2^{1/4}8^{3/4}\approx5.656854$，不是 $(2+8)/2=5$。

## 3. 和交叉熵的关系

在一个位置上，模型给出词表上的概率分布 $q_t$；真实 token 可以写成一个只有正确位置为 1、其余位置为 0 的分布 $y_t$。若词表大小是 $V$，该位置的交叉熵为：

$$
H(y_t,q_t)
=-\sum_{i=1}^{V}y_{t,i}\ln q_{t,i}
=-\ln q_t(w_t).
$$

所以，**PPL 是有效目标平均交叉熵的指数**。词表大小 $V$ 是每个位置的分类维度，不是跨 token 取平均时的分母：

$$
\operatorname{PPL}
=\exp\left(\frac1N\sum_{t\in S}H(y_t,q_t)\right).
$$

如果交叉熵用自然对数，取 $\exp$；如果用 $\log_2$，取 $2$ 的幂。只要底数对应，得到的 PPL 相同。平均负对数似然本身是 $\ln\operatorname{PPL}$，不能直接称为困惑度。[1]

### 3.1 PPL 更低，准确率就一定更高吗？

不一定。准确率只看概率最大的选项是否命中，PPL 则关心给真实目标分配了多少概率。

例如有两个二分类位置，模型 A 给真实目标的概率分别是 $0.99$ 和 $0.49$，模型 B 都给 $0.6$：

- A 命中 1 个位置，准确率为 50%，PPL 约为 $1.435768$
- B 命中 2 个位置，准确率为 100%，PPL 约为 $1.666667$

A 的 PPL 更低，准确率却更低。因此，在相同评测设置下，PPL 更低意味着真实目标的平均负对数似然更低。它不单独保证事实正确性、回答质量或概率置信度的可靠性。

## 4. 比较模型前，要固定什么？

只有评测口径一致，PPL 的高低才有明确含义：[2]

1. **同一份测试文本与预处理。** 数据难度、清洗、文档拼接和边界处理都会影响结果，单个句子不能代表整体表现
2. **同一 tokenizer 和计分单位。** 切成不同 token 后，预测事件和分母都变了，原始 token 级 PPL 通常不能直接横比
3. **同一组计分目标。** 是否计 EOS、是否只评回答、是否排除 padding，都要一致
4. **同一上下文评测方案。** 上下文长度、窗口步长、是否重置文档上下文都会改变每次预测所见的信息

有限上下文模型通常需要滑动窗口评测。重叠部分可以继续提供上下文，但每个目标只计一次；相比互不重叠的分块，窗口重叠通常让预测看到更多前文，不过并不保证任意模型、任意文本都取得更低 PPL。报告结果时，应同时说明窗口和计分设置。

如果确实要比较使用不同 tokenizer 的模型，可以考虑在共同的原始文本上报告每字节或每字符的负对数似然，并明确文本编码、预处理和边界口径。这已经换了计量单位，不能直接把一个 token 级 PPL 数值换个名字就当作公平比较。

## Reference

[1] [Jurafsky 与 Martin：Speech and Language Processing，N-gram Language Models](https://web.stanford.edu/~jurafsky/slp3/3.pdf)

[2] [Hugging Face：Perplexity of fixed-length models](https://huggingface.co/docs/transformers/perplexity)

[3] [Two minutes NLP — Perplexity explained with simple probabilities（原示例来源）](https://medium.com/nlplanet/two-minutes-nlp-perplexity-explained-with-simple-probabilities-6cdc46884584)
