---
title: Sampling Method
published: 2024-04-11
description: 介绍蒙特卡洛采样方法及其在参数估计中的应用，涵盖逆变换采样、拒绝采样等核心技术的原理与多臂老虎机场景下的实践。
category: Mathematics
tags:
- machine learning
- mathematics
- statistical
draft: false
---
## 0. 前言

知道一个概率分布，并不等于知道怎样从中抽样。本文讨论**如何用容易生成的随机数，得到服从目标分布的样本**。这些样本可以用来模拟随机现象，也可以用于 Monte Carlo 估计，例如用样本平均值估计期望。

先用 Thompson Sampling 说明为什么要对参数抽样，再介绍 Inverse Transform Sampling 和 Rejection Sampling。若你关心的是语言模型如何使用 Temperature、Top-K、Top-P 选下一个 token，请阅读 [LLM Decoding](/posts/2026-05-16-llm-decoding-strategies/)。那里讨论的是如何修改、截断或搜索 token 分布；这里讨论的是给定目标分布后，怎样正确地抽样。

## 1. 一个动机：Beta–Bernoulli Thompson Sampling

假设有 $K$ 台机器，第 $i$ 台每次返回奖励 $r\in\{0,1\}$，成功概率 $p_i$ 未知但固定。采用独立先验：

$$
p_i\sim\operatorname{Beta}(\alpha_i,\beta_i),\qquad \alpha_i,\beta_i>0.
$$

这里每台机器有自己的成功概率，$p_1,\ldots,p_K$ **不需要加起来等于 1**。因此，不能把它们默认当成一个 Dirichlet 分布的概率向量。Beta 是定义在 $[0,1]$ 上的单变量分布；它与二分类概率向量 $(p,1-p)$ 的 Dirichlet 表示有关，但不是“两维随机变量”的意思。

在这个独立 Beta–Bernoulli 模型里，每轮操作如下：

1. 从每台机器当前的后验中抽一个候选成功概率：$\tilde p_i\sim\operatorname{Beta}(\alpha_i,\beta_i)$
2. 选择 $a=\operatorname*{argmax}_i\tilde p_i$，实际尝试这台机器
3. 观察它返回的奖励 $r$，只更新被选择机器的参数：

$$
\alpha_a\leftarrow\alpha_a+r,\qquad
\beta_a\leftarrow\beta_a+(1-r).
$$

抽到最大的 $\tilde p_i$ 不意味着这台机器本轮必定成功，也不是观察到了“另一台机器赢了”。通常只有被选择机器的结果可见。在带有关联先验或额外反馈的其他模型中，更新范围才可能不同。[Thompson Sampling 教程，第 3 章](https://arxiv.org/html/1707.02038v3)

例如，某台机器从 $\operatorname{Beta}(1,1)$ 出发，被选择三次，依次得到 $1,0,1$，后验就是 $\operatorname{Beta}(3,2)$，其均值为 $3/5$。下面是一次决策与更新的 Python 实现：

```python
import random


def thompson_step(alpha, beta, pull, rng):
    # pull(a) 实际尝试第 a 台机器，并返回 0 或 1
    if not alpha or len(alpha) != len(beta):
        raise ValueError("alpha and beta must have the same nonzero length")
    if any(a <= 0 or b <= 0 for a, b in zip(alpha, beta)):
        raise ValueError("Beta parameters must be positive")
    samples = [rng.betavariate(a, b) for a, b in zip(alpha, beta)]
    arm = max(range(len(samples)), key=samples.__getitem__)
    reward = pull(arm)
    if reward not in (0, 1):
        raise ValueError("Bernoulli reward must be 0 or 1")
    alpha[arm] += reward
    beta[arm] += 1 - reward
    return arm, reward
```

这里调用了现成的 Beta 随机数生成器。下面介绍逆变换和拒绝采样，解释这些基础采样方法为什么有效。

## 2. Inverse Transform Sampling

### 2.1 CDF 与 Probability Integral Transform

累积分布函数定义为 $F(x)=P(X\le x)$。它单调不减、右连续，取值位于 $[0,1]$，但不一定严格递增，也不一定能取到区间内每一个值。若分布有密度，则用 $f$ 表示密度，不把 $f(x)$ 和点概率 $P(X=x)$ 混为一谈。

**如果 $F$ 连续，那么 $F(X)\sim U(0,1)$。** 这叫 Probability Integral Transform。为便于理解证明，先看 $F$ 在其取值介于 0 与 1 的区间上严格递增的情形。对 $0<u<1$：

$$
P(F(X)\le u)=P(X\le F^{-1}(u))=F(F^{-1}(u))=u.
$$

连续 CDF 即使有平坦区间，这个结论仍成立；那些区间没有概率质量，只是不能直接声称普通反函数处处存在且唯一。对于离散分布，结论一般不成立：若 $P(X=0)=P(X=1)=1/2$，则 $F(X)$ 只能取 $1/2$ 和 $1$，显然不是连续均匀分布。

### 2.2 用广义逆同时处理连续和离散分布

真正用于采样的方向是从均匀分布出发。对任意 CDF，定义其广义逆，也叫 quantile function：

$$
Q(u)=\inf\{x\in\mathbb R:F(x)\ge u\},\qquad 0<u<1.
$$

先生成 $U\sim U(0,1)$，再令 $X=Q(U)$。由 CDF 的单调性和右连续性：

$$
P(Q(U)\le x)=P(U\le F(x))=F(x).
$$

因此结果具有目标分布。**这个方向不要求 $F$ 连续，也不要求有概率密度。** 普通反函数存在时，广义逆与普通反函数一致。[Devroye，Non-Uniform Random Variate Generation，第 II 章](https://luc.devroye.org/chapter_two.pdf)

例如，离散变量取值为 $0,1,2$，对应概率为 $0.2,0.5,0.3$：

$$
Q(u)=\begin{cases}
0,&0<u\le0.2,\\
1,&0.2<u\le0.7,\\
2,&0.7<u<1.
\end{cases}
$$

长度分别为 $0.2,0.5,0.3$ 的区间，恰好产生相应的采样概率。

### 2.3 连续例子：指数分布

设 $X\sim\operatorname{Exponential}(\rho)$，其中 $\rho>0$ 是 rate。对 $x\ge0$，有 $F(x)=1-e^{-\rho x}$，所以：

$$
X=-\frac{\log(1-U)}{\rho}.
$$

下面的实现使用 Python 标准库。`random()` 返回 $[0,1)$ 中的浮点数，避免了 $U=1$ 时的无穷值；$U=0$ 时得到合法的边界值 0。`log1p(-u)` 比直接计算 `log(1-u)` 更适合很小的 $u$。

```python
import math
import random


def sample_exponential(rate, rng):
    if not math.isfinite(rate) or rate <= 0:
        raise ValueError("rate must be positive and finite")
    return -math.log1p(-rng.random()) / rate


rng = random.Random(2024)
samples = [sample_exponential(2.0, rng) for _ in range(30000)]
print(sum(samples) / len(samples))  # 应接近理论均值 1 / 2
```

理论推导使用理想连续随机数；程序使用有限精度伪随机数。对于没有简单 quantile function 的分布，还需数值求逆，并控制数值误差。

## 3. Rejection Sampling

### 3.1 已归一化的目标密度

目标密度为 $f$，选择一个容易采样的 proposal density $g$，并找到有限常数 $c$，使：

$$
f(y)\le c g(y)
$$

在整个目标支撑上成立。特别地，$f(y)>0$ 的地方不能被 $g$ 漏掉。不能只在若干测试点检查这个上界，就认定条件成立。

每次独立生成 $Y\sim g$ 和 $U\sim U(0,1)$。若：

$$
U\le\frac{f(Y)}{c g(Y)},
$$

就接受 $Y$；否则重新生成。设接受事件为 $A$，对于任意可测集合 $B$：

$$
P(Y\in B,A)=\int_B g(y)\frac{f(y)}{c g(y)}\,dy
=\frac1c\int_B f(y)\,dy.
$$

令 $B=\mathbb R$ 得 $P(A)=1/c$，所以：

$$
P(Y\in B\mid A)=\int_B f(y)\,dy.
$$

被接受样本的分布正是目标分布。直觉上，先在包络曲线 $cg$ 下均匀取点，再保留落在 $f$ 下的点：候选的密度 $g$ 与接受概率中的 $1/g$ 恰好抵消。

![Rejection Sampling 包络与接受区域示意](https://s2.loli.net/2024/04/11/XO3GehobsnckrNQ.png)

### 3.2 不知道归一化常数也能采样

假设只知道非负函数 $h$，目标密度为：

$$
f(y)=\frac{h(y)}Z,\qquad 0<Z=\int h(y)\,dy<\infty.
$$

若能够找到已知的有限常数 $M$，满足 $h(y)\le M g(y)$，则接受概率可以直接写成：

$$
a(y)=\frac{h(y)}{M g(y)}.
$$

无需在算法中计算 $Z$。同样积分可得：

$$
P(A)=\frac ZM,\qquad
p(y\mid A)=\frac{g(y)a(y)}{Z/M}=\frac{h(y)}Z.
$$

因此，拒绝采样并不要求后验与先验共轭。难点通常在于找到有效且足够贴近目标的包络；如果 $Z/M$ 很小，大量候选会被丢弃。[UC Berkeley Data 102：Rejection Sampling](https://data102.org/fa21/assets/slides/lecture8.pdf)

例如，取 $h(y)=y\mathbf 1_{[0,1]}(y)$、$g$ 为 $U(0,1)$ 的密度、$M=1$。接受概率为 $y$，接受率为 $Z/M=1/2$，结果密度为 $f(y)=2y$，均值为 $2/3$。

```python
import random


def sample_linear_density(n, rng):
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")
    samples, proposals = [], 0
    while len(samples) < n:
        y = rng.random()
        u = rng.random()
        proposals += 1
        if u < y:
            samples.append(y)
    return samples, n / proposals


samples, acceptance = sample_linear_density(30000, random.Random(2024))
print(sum(samples) / len(samples))  # 接近 2 / 3
print(acceptance)                  # 接近 1 / 2
```

## 4. 后验分布与下一步

贝叶斯后验常写为：

$$
p(\theta\mid\mathcal D)=
\frac{p(\mathcal D\mid\theta)p(\theta)}
{\int p(\mathcal D\mid v)p(v)\,dv}.
$$

共轭性让后验落在熟悉的分布族里，便于更新参数，但不是一切采样方法的前提。逆变换需要能够计算 quantile function；拒绝采样只需可计算的未归一化目标及有效包络。即使概率密度写得出来，这些操作也未必高效。

如果包络难找，可以考虑 MCMC。例如 Metropolis–Hastings 从满足 $h(\theta)>0$ 的当前位置 $\theta$ 提议 $\theta'\sim q(\cdot\mid\theta)$，其接受概率为：

$$
\min\left(1,
\frac{h(\theta')q(\theta\mid\theta')}
{h(\theta)q(\theta'\mid\theta)}\right),\qquad
h(\theta)=p(\mathcal D\mid\theta)p(\theta).
$$

后验的归一化常数在比值中抵消；**对称 proposal 的两个 $q$ 项相同，可以约去；一般 proposal 不能省略这个比值**。MCMC 还需要检查链的收敛与相关性，不能把每次输出默认看成独立的精确后验样本。[Hastings, 1970](https://doi.org/10.1093/biomet/57.1.97)

## Reference

1. [Russo et al., A Tutorial on Thompson Sampling](https://arxiv.org/abs/1707.02038)，第 3 章的 Beta–Bernoulli 模型
2. [Luc Devroye, Non-Uniform Random Variate Generation](https://luc.devroye.org/rnbookindex.html)，第 II、III 章的逆变换与拒绝采样
3. [Jacob Steinhardt, Data 102 Lecture 8](https://data102.org/fa21/assets/slides/lecture8.pdf)，未归一化目标与 MCMC 的衔接
4. [Hastings, Monte Carlo Sampling Methods Using Markov Chains and Their Applications](https://doi.org/10.1093/biomet/57.1.97)，Metropolis–Hastings 的原始文献
