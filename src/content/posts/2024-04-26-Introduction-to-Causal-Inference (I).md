---
title: Causal Inference Series (I)
published: 2024-04-26
description: 因果推断入门系列第一篇，基于 Brady Neal 的课程，从辛普森悖论出发揭示相关性与因果性的本质区别。
category: Causal Inference
tags:
- machine learning
- mathematics
- statistics
- causal inference
draft: false
---
## 0. 前言

Causal Inference（因果推断）已经在多个领域发挥了很大作用。我早就听说过，但一直没有认真学过，通常只是浅尝辄止。这次在博客开一个系列，记录学习过程，也希望借此督促自己坚持下去。

入门阶段，我选了相对简单的课程和书籍。结合网上的推荐和自己的体验，感觉 Brady Neal 的系列介绍比较合适，所以这个系列会以他的课程为基础。课程链接： [https://www.bradyneal.com/causal-inference-course](https://www.bradyneal.com/causal-inference-course)


:::note
这些是初学者的个人学习笔记，记录和理解难免有错，欢迎指正。
:::


## 1. 第一章 Motivation: Why You Might Care

第一章从辛普森悖论出发，介绍相关性和因果性的联系与区别。

### 1.1 Simpson’s Paradox

因果推断的第一课通常会讲 Simpson’s Paradox（辛普森悖论）。假设针对一种病有 2 个治疗方案，treatment A 和 treatment B。实验中，treatment B 比较稀缺，只有较少的志愿者能用上。例如，接受 treatment A 和 treatment B 的志愿者分别占 73% 和 27%，得到的数据如下：

表中的百分比是接受相应 treatment 后志愿者的死亡率。Mild 组表示轻症，Severe 组表示重症。

<img src="https://s2.loli.net/2024/04/26/OvNDhwKrz68ajnu.png" alt="image.png" width="400" height="300" />

从上表可以看到，无论轻症组还是重症组，treatment B 的死亡率都明显更低。但把所有人合在一起看，Total 列反而是 treatment A 的死亡率更低。到底哪个 treatment 更好呢？

关键在于两组的患者构成：接受 treatment B 的 550 个人中，有 500 个是重症患者。计算总体死亡率时，重症组的权重更大，使 treatment B 的 Total 死亡率接近 20 %。接受 treatment A 的轻症患者更多，最终平均死亡率反而比较低。要判断哪个方案更好，还需要知道其中的因果关系。


先看受试者的 Condition 影响 treatment 的情况。例如，医生根据病情选择治疗方案：轻症通常安排 treatment A，重症安排 treatment B。它们之间的关系如下：

<img src="https://s2.loli.net/2024/04/26/I7lOEvoz3VsPmgQ.png" alt="image.png" width="300" height="200" />

这时应比较不同病情下 treatment 的治愈率，显然 treatment B 更好。


再看 treatment 影响 Condition 的情况。比如 treatment B 药效好但稀缺，患者本来可以直接用药，却因为坚持等 treatment B 而耽误了治疗，导致病情恶化；treatment A 没有这个问题。它们之间的关系如下：

<img src="https://s2.loli.net/2024/04/26/oEmGarx15iw4TNM.png" alt="image.png" width="300" height="200" />

这时，尽管 treatment B 药效好，为了提高存活率，我们显然应该选择 treatment A。知道其中的因果关系后，就能解决 Simpson’s Paradox 中的选择问题。


#### 1.2 Correlation Does Not Imply Causation

这里要记住：$相关性 \neq 因果性$。“Nicolas Cage and Pool Drownings”就是一个例子，讨论的是演员尼古拉斯·凯奇出演电影的次数与游泳溺水次数的相关性。

![image.png](https://s2.loli.net/2024/04/26/q9rXRWn3k5ZhlKa.png)


有人发现，把他出演电影的次数和游泳溺水次数拿来计算线性相关性，结果可能显示高度相关。但如果因此认定两者有因果关系，就很离谱了；它们显然没有这样的因果关系。

再看一个例子：有人发现，晚上很晚回来、穿着鞋子睡觉的人，第二天早上醒来会头痛。这两件事确实一起出现，人们会说它们相关。但背后还藏着另一个条件：这些人大概率是喝醉后回来倒头就睡，第二天头痛也八成是喝酒造成的。这个隐藏的条件就叫 **confounder**。

<img src="https://s2.loli.net/2024/04/26/eLzqBX6R4pUKVNc.png" alt="image.png" width="300" height="200" />

confounder 与研究对象的关联称为 **confounding association**。如果想单独探究“穿鞋睡觉 -> 第二天头疼”的因果关系，就必须先排除 confounder 的影响。

## 2. 第二章 Potential Outcomes

第二章介绍因果推断的基础概念。

### 2.1 Potential Outcomes and Individual Treatment Effects

假设有个人心情不太好，有人想送他一只狗。接受这只狗后，他可能会开心起来；如果拒绝了，他会继续不高兴吗？反过来，如果他接受了狗却仍不开心，我们又怎么知道，不送他这只狗，他会不会更开心？


#### 2.1.1  Potential Outcomes

对同一个人采取不同的处理方式，会产生不同的潜在结果。后文用 $\mathit {Y}$ 表示这个潜在输出。

在上面的例子中，$\mathit {Y} = 1$ 表示高兴，$\mathit {Y} = 0$ 表示不高兴。

用 $\mathit {T}$ 表示 treatment 这个随机变量。$\mathit {T} = 1$ 表示接受狗子，$\mathit {T} = 0$ 表示不接受。

用 $\mathit {Y(1)}$ 表示接受狗子后的潜在输出，$\mathit {Y(0)}$ 表示不接受时的输出。

#### 2.1.2  Individual Treatment Effects

面对多个个体时，我们使用 $\tau_i \triangleq Y_i(1) - Y_i(0)$ 来评估某个个体采取 treatment 之后的潜在输出结果。

:::note
接受狗子后，可以观察 $\mathit {Y(1)}$；不接受时，可以观察 $\mathit {Y(0)}$。但不能同时观察到 $\mathit {Y(1)}$ 和 $\mathit {Y(0)}$。这就是 **Fundamental Problem of Causal Inference**。
:::

### 2.2  Average Treatment Effects

每个人可能存在差异。要客观评估 treatment 的作用，需要对所有人求 treatment 期望：


$$
\tau \triangleq \mathbb{E}[Y_i(1) - Y_i(0)] = \mathbb{E}[Y(1) - Y(0)]
$$


但受 Fundamental Problem of Causal Inference 限制，上式很难直接计算。看下面的表：

<img src="https://s2.loli.net/2024/04/27/9vpO8bP5Uq2dYtS.png" alt="image.png" width="300" height="200" />

对个体 $i$ 采取 treatment 0 时，只能观察到 $Y_i(0)$，观察不到 $Y_i(1)$。也就是说，以下式子不成立：


$$
\mathbb{E}[Y(1) - Y(0)] = \mathbb{E}[Y(1)] - \mathbb{E}[Y(0)] \neq \mathbb{E}[Y(1) | T = 1 ] - \mathbb{E}[Y(0) | T = 0]
$$


treatment 0 对应的集合只是总体的一部分，不能代表全部结果；$\mathbb{E}[Y(1)]$ 应对应图中最右边的结果（Intervening）。

<img src="https://s2.loli.net/2024/04/27/vtybfejVDn4QNKA.png" alt="image.png" width="600" height="400" />


#### 2.2.1 Ignorability and Exchangeability

那么，在什么条件或假设下，上式能够成立呢？

> Assumption 2.1 Ignorability / Exchangeability
>
:::note

$$
(Y(1) , Y(0)) \amalg T
$$

:::

这里的 Ignorability（可忽略性）不是说“可以忽视缺失的数据”，而是说处理分配 $T$ 与两种潜在结果独立。再结合一致性（实际接受 $T=t$ 时，观测结果 $Y=Y(t)$）和两组都有样本的条件，才能把潜在结果的均值换成观测组的均值：


$$
\begin{aligned}
\mathbb{E}[Y(1) - Y(0)] &= \mathbb{E}[Y(1)] - \mathbb{E}[Y(0)]
\newline
&=\mathbb{E}[Y(1) \mid T = 1 ] - \mathbb{E}[Y(0) \mid T = 0] \ (Ignorability)
\newline
&=\mathbb{E}[Y \mid T = 1 ] - \mathbb{E}[Y \mid T = 0] \quad \text{（一致性）}
\end{aligned}
$$


这并不是说 $Y(1)$“只基于 $T=1$”；$Y(1)$ 对每个人都有定义，包括实际未接受处理的人。独立性假设让接受处理组能够代表总体中 $Y(1)$ 的分布。

<img src="https://s2.loli.net/2024/04/27/crp69WgbIqMOvX5.png" alt="image.png" width="400" height="300" />


这个假设也叫 `Exchangeability`，即 $\mathbb{E}[Y(0) \mid T = 0] = \mathbb{E}[Y(0) \mid T = 1] = \mathbb{E}[Y(0) \mid t ]$。也就是说，让 group A 和 group B 交换 treatment group 与 control group 的角色，输出结果只与 treatment 有关，与属于哪一组无关，尤其不受 confounder 影响。这里假设两组除了 treatment 方式不同，不受其他因素影响。

> Definition 2.1 Identifiability
>
:::note
causal quantity (e.g. $\mathbb{E}[Y(t)]$) is Identifiable if we can compute it from a purely statistical quantity (e.g. $\mathbb{E}[Y \mid t]$)
:::

这里的 Identifiability 是说，可以用 $\mathbb{E}[Y \mid t]$ 代替 $\mathbb{E}[Y(t)]$。

#### 2.2.2 Conditional Exchangeability and Unconfoundedness

在实际问题中，直接假设 group A 和 group B 只在 treatment 方式上有区别、不受其他因素影响，显然不太现实。可以换一个思路：控制一些条件，让两组除了 treatment 方式以外没有其他区别。

> Assumption 2.2 Conditional Exchangeability / Unconfoundedness
>
:::note

$$
(Y(1) , Y(0)) \amalg T \mid X
$$

:::

满足 Conditional Exchangeability 假设时，也就是控制了 confounder X，使各组处于相同的 confounder 条件下，再施加 treatment，treatment 就直接作用于 outcome，而不会由潜在的 confounder 影响 outcome。如下图所示：

<div style="display: flex;">
    <img src="https://s2.loli.net/2024/04/27/HcwmUCh7oAPdjir.png" alt="Image 1" style="width: 100%;">
    <img src="https://s2.loli.net/2024/04/27/2Gt8Ugf9d6WSomB.png" alt="Image 2" style="width: 100%;">

于是有：


$$
\begin{aligned}
\mathbb{E}[Y(1) - Y(0) \mid X] &= \mathbb{E}[Y(1) \mid X] - \mathbb{E}[Y(0) \mid X]
\newline
&=\mathbb{E}[Y(1) \mid T = 1, X] - \mathbb{E}[Y(0) \mid T = 0, X] \ (Ignorability)
\newline
&=\mathbb{E}[Y \mid T = 1, X ] - \mathbb{E}[Y \mid T = 0, X] \ (fix confounder)
\end{aligned}
$$


进而得到：


$$
\begin{aligned}
\mathbb{E}[Y(1) - Y(0) ] &= \mathbb{E}_X[\mathbb{E}[Y(1) \mid X] - \mathbb{E}[Y(0) \mid X]]
\newline
&=\mathbb{E}_X[\mathbb{E}[Y(1) \mid T = 1, X] - \mathbb{E}[Y(0) \mid T = 0, X]] \ (Ignorability)
\newline
&=\mathbb{E}_X[\mathbb{E}[Y \mid T = 1, X ] - \mathbb{E}[Y \mid T = 0, X] ]\ (expect \ confounder)
\end{aligned}
$$


Conditional exchangeability (Assumption 2.2) is a core assumption for
causal inference and goes by many names. For example, the following are reasonably commonly used to **refer to the same assumption: unconfoundedness, conditional ignorability, no unobserved confounding,
selection on observables, no omitted variable bias**, etc.

$\textit{We will use the name “unconfoundedness” a fair amount throughout this book.}$


> Theorem 2.1  (Adjustment Formula) Given the assumptions of **unconfoundedness**, **positivity**, **consistency**, and **no interference**, we can identify the
average treatment effect:
>
:::note

$$
\mathbb{E}[Y(1) - Y(0) ] = \mathbb{E}_X[\mathbb{E}[Y \mid T = 1, X ] - \mathbb{E}[Y \mid T = 0, X] ]
$$

:::

这个识别公式依赖于已控制足够的**处理前**混杂变量，还要求各类 $X$ 中接受或不接受处理的概率都大于 0。若遗漏了同时影响处理选择和结果的变量，仍会存在“混杂变量 $\rightarrow$ 处理”和“混杂变量 $\rightarrow$ 结果”两条路径，观察组均值之差就不能直接解释为因果效应。


## Reference
