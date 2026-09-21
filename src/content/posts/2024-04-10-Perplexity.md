---
title: Perplexity
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

在 NLP 中,  经常可以看到使用"困惑度"来描述一个 LLM 的能力. 那么什么是"困惑度"?

简单理解,  困惑度就是"模型对样本预测结果的信心". 具体的,  模型对这个样本结果的预测概率越高,  表明信心越高,  对应困惑度越低.


:::note
本文介绍的Perplexity 特指 "Perplexity of a probability model".
:::



## 1. 举个栗子

假设我们的 vocabulary 就只有6个单词,  `“a”,  “the”,  “red”,  “fox”,  “dog”,  and “.” `. 模型需要从这里边预测输出句子 `W : "a red fox ."`


$$
\begin{align*}
P(W) & = P(w_1,  w_2,  \ldots,  w_n) \\
& = P(w_n|w_1,  w_2,  \ldots,  w_{n-1}) \times P(w_1,  w_2,  \ldots,  w_{n-1})
\end{align*}
$$


对于这句话就是:


$$
\begin{align*}
P(' a\ red\ fox\ . ') =  P(' a ') \times P(' red ' | ' a ') \times P(' fox ' | ' a\ red ') \times P(' . '|' a\ red\ fox ')
\end{align*}
$$



假设模型,  预测第一个字的概率分布如下 :

<img src="https://s2.loli.net/2024/04/10/IfNJ1tRBwbTH8lP.png" alt="第1个字.png" width="600" height="400" />

则有
$P( ' a ' ) = 0.4$
,  进一步的
$P( w_2 | ' a ' )$
分布如下

<img src="https://s2.loli.net/2024/04/10/vgHxO3nFumXrQAc.png" alt="第2个字.png" width="600" height="400" />

于是
$P( ' red '  |  ' a ' ) = 0.27$
, 同理,  根据以下分布

<div style="display: flex;">
    <img src="https://s2.loli.net/2024/04/10/UwFikWIL9tNPJRA.png" alt="Image 1" style="width: 100%;">
    <img src="https://s2.loli.net/2024/04/10/k8DHmfxSJIuOTpY.png" alt="Image 2" style="width: 100%;">


我们有如下结果:


$$
\begin{align*}
P(' a\ red\ fox\ . ') &=  P(' a ') \times P(' red ' | ' a ') \times P(' fox ' | ' a\ red ') \times P(' . '|' a\ red\ fox ')  \\
&= 0.4 * 0.27 * 0.55 * 0.79 \\
&= 0.046926
\end{align*}
$$


0.046926 是模型赋予整句话的概率。由于它是条件概率的连乘，直接比较不同长度句子的联合概率不合适。可以取每个 token 条件概率的[几何平均数](https://en.wikipedia.org/wiki/Geometric_mean)：


$$
P_{norm}(W) = P(W)^{1/n}
$$



这里的n表示句子的单词(token)数量.于是


$$
\begin{align*}
P_{norm}('a\ red\ fox\ .') &= P('a\ red\ fox\ .')^{1/n} \\
&= 0.046926 ^ {1/4} \\
&\approx 0.4654
\end{align*}
$$


这样我们就可以使用 $P_{norm}$ 来度量模型对不同长度句子的预测输出"信心".

## 2. 如何计算

前边我们提到,  模型与输出的句子,  信心越足,  困惑度越小. 可以看到,  困惑度的计算公式如下:


$$
\begin{align*}
PP(W) &= \frac {1} {P_{norm}(W)} \\
&= \frac {1} {P(W)^{1/n}} \\
&= (\frac {1} {P(W)}) ^{1/n} \\
&= P(W) ^{-1/n}
\end{align*}
$$


对于之前的这个模型,  其 $PP(W) = (1/0.046926)^{1/4} \approx 2.15$。

而假设有另外一个模型, 给定任意条件下, 对下一个单词的预测概率均相等为 1/6 . 那么这个模型的的困惑度为:


$$
PP(W) = (\frac {1} {(1/6)^4}) ^{1/4} = 6
$$


在这一个样本、同一种 token 划分下，均匀预测模型的困惑度更高。单个样本的结果不能直接证明一个模型整体更差，应在相同测试集上比较平均负对数似然。

## 3. 和交叉熵的关系
我们知道,  [香农熵](https://zh.wikipedia.org/zh-hans/%E7%86%B5_(%E4%BF%A1%E6%81%AF%E8%AE%BA))  计算方式为 :


$$
H(p) = -\sum_{i=1}^{V} p_i \log_{2} p_i
$$


交叉熵的计算方式:


$$
H(p, q) = -\sum_{i=1}^{V} p_i \log_{2} q_i
$$


> 事实上:
>
> $$
> KL(p, q) = -H(p) + H(p, q)
> $$


对$PP(W)$进行拆解, 得以下式子:
这里令 $n$ 为测试序列的 token 数量，$q_t$ 为模型在第 $t$ 个位置给出的词表概率分布，$y_t$ 为真实 token 的 one-hot 分布。词表大小 $V$ 是每个位置分类问题的维度，不是困惑度公式中取平均时的分母。


$$
\begin{aligned}
P(W) &= \prod_{t=1}^{n} q_t(w_t), \\
H(y_t,q_t) &= -\sum_{i=1}^{V}y_{t,i}\log_2 q_{t,i}
             =-\log_2 q_t(w_t), \\
PP(W) &= 2^{-\frac{1}{n}\sum_{t=1}^{n}\log_2 q_t(w_t)}
       =2^{\frac{1}{n}\sum_{t=1}^{n}H(y_t,q_t)}.
\end{aligned}
$$


从这个角度来看,  困惑度越小,  交叉熵越小,  预测越准确.

若交叉熵使用自然对数，则 $PP(W)=\exp(\mathrm{NLL}_{\mathrm{avg}})$，其中 $\mathrm{NLL}_{\mathrm{avg}}$ 是平均负对数似然；它本身是 log perplexity，不能直接称为困惑度。比较模型时还需使用相同的测试数据、tokenizer 和计数口径。

## Reference
[1] [Two minutes NLP — Perplexity explained with simple probabilities](https://medium.com/nlplanet/two-minutes-nlp-perplexity-explained-with-simple-probabilities-6cdc46884584)

[2] [Wiki-Perplexity](https://en.wikipedia.org/wiki/Perplexity)

[3] [Perplexity Intuition (and its derivation)](https://webcache.googleusercontent.com/search?q=cache:https://towardsdatascience.com/perplexity-intuition-and-derivation-105dd481c8f3&strip=0&vwsrc=1&referer=medium-parser)
