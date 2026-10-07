---
title: Causal Inference Series (II)
published: 2025-02-05
description: 因果推断系列第二篇，基于因果推断综述论文，系统介绍因果推断的基本符号、关键假设和核心定义。
category: Causal Inference
tags:
- machine learning
- mathematics
- statistics
- causal inference
draft: false
---
## 0. 前言

Causal Inference（因果推断）已经在多个领域发挥出重要作用。我早已听说过它，但一直没有系统学习，往往浅尝辄止。于是开这个博客系列，记录学习过程，也希望借此督促自己。

本篇记录我对 [A Survey on Causal Inference](https://arxiv.org/abs/2002.02770) 的学习。

:::note
这是初学者的个人笔记，内容和理解难免有误，欢迎指正。
:::


## 1. BASIC OF CAUSAL INFERENCE

本节介绍因果推断的背景知识，包括数学符号和相关假设。

### 2.1 Definitions

**Definition 1.**
_Unit. A unit is the **atomic** research object in the treatment effect study._

> 例如一个患者、一个独立的人，也可以是一组人，如一个班级或一个市场中的人群。

**Definition 2.**
_Treatment. Treatment refers to the action that applies (exposes, or subjects) to a unit._

> 假设给某个病人开药，“开 A 药”和“开 B 药”分别是一个 treatment。令 W (W $\in$ {0, 1, 2, . . . , N_w }) 表示某个 treatment，这里 $N_w + 1 $ 是 treatment 的个数。

**Definition 3.**
_Potential outcome. For each unit-treatment pair, the outcome of that treatment when applied on
that unit is the potential outcome._

> treatment $w$ 对应的 potential outcome 表示为 $Y(W \ = \ w)$
>
> 潜在结果指不同处理条件下可能产生的结果（不同的处理有不同的结果），更像是在描述一种“薛定谔式”的状态。

**Definition 4.**
_Observed outcome. The observed outcome is the outcome of the treatment that is actually applied._

> observed outcome 有时也称为 factual outcome（事实结果），用 $Y^F$ 表示，其中 F 指 factual。
>
> 观察到的结果 $Y^F$ 是实际实施某个处理 $w$ 后的确定结果。

:::note
观测结果（$Y^F$）和潜在结果（$Y(W \ = \ w)$）的关系是：当某个 treatment $w$ **真正被施加**时，无论具体是哪种 treatment，都有：$Y^F = Y(W \ = \ w)$
:::


**Definition 5.**
_Counterfactual outcome: Counterfactual outcome is the outcome if the unit had taken another
treatment._

> counterfactual outcome 称为反事实结果，指实际接受的 treatment $w$ 之外，其他 treatment 对应的潜在结果。
>
> 一个 unit 只能接受某一个 treatment（比如 A），也只能观测到对应的潜在结果。若想观测另一个 treatment（比如 B）在同一 unit 上的结果，就只能回到之前的时间点，或者去平行世界，因为当前 unit 的状态已经改变了。因此，这些结果称为反事实结果。


**Definition 6.**
_Pre-treatment variables: Pre-treatment variables are the variables that will not be affected by the
treatment._

> 处理前变量也称为 background variables（背景变量），**通常用 $X$ 表示**。**它们不会被任何 treatment 影响，但 $X$ 可能影响 treatment 的选择。** 比如，某个人的性别不受“发广告”这个 treatment 的影响，但性别可能影响广告是否下发。

**Definition 7.**
_Post-treatment variables: The post-treatment variables are the variables that are affected by the
treatment._

> 处理后变量指**会被 treatment 影响的变量**。比如，某个人打开软件的次数会受“发广告”这个 treatment 影响。

**Definition 8.**
_Individual Treatment Effect (ITE)_

对于 unit i，其 _ITE_ 计算公式为：


$$
\text{ITE}_i = Y_i(W=1) - Y_i(W=0)
$$


其中，$Y_i(W=1)$ 和 $Y_i(W=0)$ 分别是 unit $i$ 分配到实验组和对照组时的输出。

**Definition 9.**
_Average Treatment Effect (ATE)_

ATE 关注整个 population 上的 treatment 效果，计算公式为：


$$
\text{ATE} = \mathbb{E}[Y(W=1) - Y(W=0)]
$$


其中，$Y(W=1)$ 和 $Y(W=0)$ 分别是整个 population 在实验组和对照组时的输出。


**Definition 10.**
_Average Treatment effect on the Treated group (ATT)_

ATT 只关注 `subgroup = treatment_group` 的 treatment 效果，计算公式为：


$$
\text{ATT} = \mathbb{E}[Y(W=1)|W=1] - \mathbb{E}[Y(W=0)|W=1]
$$


**Definition 11.**
_Conditional Average Treatment Effect (CATE)_

CATE 关注整个 population 在某个确定性条件 `X = x` 下的 treatment 效果，计算公式为：


$$
\text{CATE} = \mathbb{E}[Y(W=1)|X=x] - \mathbb{E}[Y(W=0)|X=x]
$$


:::note
CATE 关注不同条件（condition or background variables）下的 treatment 效果，因此也被称为 **heterogeneous（异质）treatment effect**。
:::


### 2.2 Assumptions

评估因果效应需要一些基本假设。

**Assumption 1.**
_Stable Unit Treatment Value Assumption (SUTVA)._ The potential outcomes for any unit
do not vary with the treatment assigned to other units, and, for each unit, there are no different forms or versions of each treatment level, which lead to different potential outcomes.

> 1. 一个个体的潜在结果不受其他个体接受何种处理的影响（无干扰）；这**不等于**要求不同个体的观测数据在统计上独立。2. 同一处理水平没有会改变结果的隐藏版本。


**Assumption 2.**
_Ignorability._ Ttreatment assignment $W$ is independent of the potential outcomes, i.e., $W \perp Y(W = 0), Y(W = 1)$.

> 如果满足 Ignorability，以下 2 个结果成立：
>
> 1. 潜在结果与实际接受的处理在总体中独立。随机分配是一种可能使这个假设成立的设计方法，但不能从观察到“可忽略性”反推出研究本来就是随机分配的；
>
> 2. 这种独立性称为可交换性：两组的潜在结果分布可比较，并不要求每个人的其他条件完全相同。

> 实际中，背景变量 $X$ 的干扰使 $W$ 和 $Y$ 之间产生联系。例如，假设 $W$ 表示“藏私房钱是否被老婆发现”，$Y(W = 1)$ 表示“被老婆揍一拳之后的疼痛值”；$W = 0$ 表示没有被揍，$W = 1$ 表示被揍。
> 此时 $W = 1$ 与 $Y(W = 1)$ 有联系，至少存在这样的关系：当 $X = 1$，即藏私房钱被发现时，$W = 1$ 的概率会升高，同时 $Y(W = 1)$ 会变大。
>
:::note
因此考虑给定背景变量 $X$ 的情况，此时有 $W \perp Y(W = 0), Y(W = 1) \mid X$。
:::


## Reference
