---
title: Causal Inference Series (II)
date: 2025-02-05 12:30:00 +0800
categories: [Causal Inference]
tags: [machine learning,  mathematics, statistics, causal inference]     # TAG names should always be lowercase
math: true
---


## 0. 前言

Causal Inference (因果推断) 已经在多个领域发挥出巨大作用, 尽管早已经听说过其大名, 但是从未步入这个领域好好学习一番, 通常是浅尝辄止. 为此在博客开一个系列, 一是用于记录学习, 二是希望能够起到监督作用...

本篇博客主要是对 [A Survey on Causal Inference](https://arxiv.org/abs/2002.02770) 进行学习和记录.

> 由于这是个人学习笔记, 我作为初学者, 在博客中记录的内容和理解难免会有错误. 希望各位能够指正, 并请不吝赐教, 在下将不胜感激.
{: .prompt-info }


## 1. BASIC OF CAUSAL INFERENCE

本节主要给出因果推断的背景知识, 包括数学符号、相关假设等.

### 2.1 Definitions

**Definition 1.**
_Unit. A unit is the **atomic** research object in the treatment effect study._

> 比如一个患者, 一个独立的人, 或者一系列的人, 如一个班级, 一个市场内部的人群等等.

**Definition 2.**
_Treatment. Treatment refers to the action that applies (exposes, or subjects) to a unit._

> 假设给某个病人开药, “开A药”就是一个 treatment, “开B药”也是一个 treatment. 令 W (W $\in$ {0, 1, 2, . . . , N_w }) 表示某个treatment, 这里 $N_w + 1 $ 表示treatment的个数

**Definition 3.**
_Potential outcome. For each unit-treatment pair, the outcome of that treatment when applied on
that unit is the potential outcome._

> 基于treatment $w$ 得到的 potential outcome 表示为 $Y(W \ = \ w)$
>
> 潜在结果是指在不同处理条件下可能产生的结果(不同的处理有不同的结果), 更像是在描述一个状态, 是薛定谔的.

**Definition 4.**
_Observed outcome. The observed outcome is the outcome of the treatment that is actually applied._

> observed outcome 有时候也称为 factual outcome(事实结果), 我们使用 $Y^F$ 来表示事实结果(F 指 factual ).
>
> 观察到的结果 $Y^F$ 是指实际实施某个处理 $w$ 后的结果, 是一个确定的.

> 观测结果($Y^F$)和潜在结果($Y(W \ = \ w)$)的关系是: 当某个treatment $w$ **真正被apply**的时候(具体什么treatment无所谓), 此时有: $Y^F = Y(W \ = \ w)$
{: .prompt-info }


**Definition 5.**
_Counterfactual outcome: Counterfactual outcome is the outcome if the unit had taken another
treatment._

> counterfactual outcome 称为反事实结果, 指除被 treatment $w$ 作用之外的, 其他treatment作用的潜在结果.
>
> 因为一个unit只能被某一个treatment作用(比如A), 并且只能观测到这一个具体的潜在结果. 如果想观测另外一个treatment(比如B)在这个unit的结果, 就只能时光回溯到之前的时间点或者去平行世界(因为当前unit的状态已经被改变了), 所以称这些结果为反事实结果.


**Definition 6.**
_Pre-treatment variables: Pre-treatment variables are the variables that will not be affected by the
treatment._

> 处理前变量也称为 background variables(背景变量),  **通常使用 $X$ 表示**. **他们不会被任何的treatment影响, 但 $X$ 可能会影响treatment的选择!!**. 比如某个人的性别不受 “发广告” 这个treatment的影响, 但反过来, 性别可能会影响广告是否下发.

**Definition 7.**
_Post-treatment variables: The post-treatment variables are the variables that are affected by the
treatment._

> 处理后变量指 、**会被treatment影响的变量**. 比如某个人的打开软件的次数, 会受 “发广告” 这个treatment影响.

**Definition 8.**
_Individual Treatment Effect (ITE)_

对于unit i , 其 _ITE_ 计算公式为:

$$\text{ITE}_i = Y_i(W=1) - Y_i(W=0)$$

其中, $Y_i(W=1) 和 Y_i(W=0)$ 分别是 unit $i$ 分配到实验组和对照组时的输出.

**Definition 9.**
_Average Treatment Effect (ATE)_

ATE关注整个population上的treatment效果, 其计算公式为:

$$\text{ATE} = \mathbb{E}[Y(W=1) - Y(W=0)]$$

其中, $Y(W=1) 和 Y(W=0)$ 分别是整个population 在实验组和对照组时的输出.


**Definition 10.**
_Average Treatment effect on the Treated group (ATT)_

ATT则只关注 `subgroup = treatment_group` 的treatment效果,  其计算公式为:

$$\text{ATT} = \mathbb{E}[Y(W=1)|W=1] - \mathbb{E}[Y(W=0)|W=1]$$


**Definition 11.**
_Conditional Average Treatment Effect (CATE)_

CATE关注整个population, 在某个确定性条件 `X = x` 下的treatment效果, 其计算公式为:

$$\text{CATE} = \mathbb{E}[Y(W=1)|X=x] - \mathbb{E}[Y(W=0)|X=x]$$

> 因为CATE关注的是, 不同条件(condition or background variables)下的treatment效果, 因此也被称为 **heterogeneous(异质) treatment effect.**
{: .prompt-info }


### 2.2 Assumptions

为了评估因果效应, 需要做一些基本的假设或者前提.

**Assumption 1.**
_Stable Unit Treatment Value Assumption (SUTVA)._ The potential outcomes for any unit
do not vary with the treatment assigned to other units, and, for each unit, there are no different forms or versions of each treatment level, which lead to different potential outcomes.

> 1. unit之间是独立的. 2. 一个treatment只有一种表达形式, 可以理解为一一对应的.


**Assumption 2.**
_Ignorability._ Ttreatment assignment $W$ is independent of the potential outcomes, i.e., $W \perp Y(W = 0), Y(W = 1)$.

> 如果满足 Ignorability, 那么会有 2 个 结果成立 :
>
> 1. $Y(W = 1)$ 和 $Y(W = 0)$ 的结果与具体施加的treatment 独立无关, 因此我们可以随机的对 groups 施加 treatment;
>
> 2. 表明 group 之间是可以交换的, 即 _exchangeability_, group 交换其 treatment, $Y(W = 1)$ 和 $Y(W = 0)$ 的结果不变, 意味着此时 group 之间是可比的, 换句话说, 除了 treatment 不同, 其他条件都会相同.

> 实际中, 受到背景变量 $X$ 干扰, 导致 $W$ 和 $Y$ 之间有联系. 举个例子,假设 $W$ 表示 "藏私房钱是否被老婆发现",  $Y(W = 1)$ 表示 "被老婆揍一拳之后的疼痛值"; $W = 0$ 表示没有被揍, $W = 1$ 表示被揍.
> 此时 $W = 1$ 与 $Y(W = 1)$ 是有联系的, 起码我们知道这样的关系存在: 当 $X = 1$, 即藏私房钱被发现时, $W = 1$ 的取值概率会升高, 同时 $Y(W = 1)$ 会变大.
>
> 因此会考虑以下情况: 给定背景变量 $X$ , 此时有 $W \perp Y(W = 0), Y(W = 1) \mid X$.
{: .prompt-info }



## Reference
