---
title: Gamma、Beta 与 Dirichlet 分布
published: 2024-04-10
description: 总结 Gamma 分布、Beta 分布和 Dirichlet 分布的定义、性质及其之间的内在联系，帮助梳理概率分布之间的脉络关系。
category: Mathematics
tags:
- machine learning
- mathematics
- reinforcement learning
draft: false
---
## 0. 前言


这篇文章整理 Gamma、Beta 和 Dirichlet 分布，并梳理它们之间的关系。


## 1. Gamma Function

- 定义

<img src="https://s2.loli.net/2024/04/10/CP5UfaszbRTnBi8.png" alt="Gamma函数定义.png" width="400" height="300" />

可以这样记：用伽马刀对 $t$ 动一刀，指数就变成 $\alpha-1$；动完刀，还要扶着梯子 $-t$ 走下来。

- 性质

<img src="https://s2.loli.net/2024/04/10/KRSeYOvgoc73aQ5.png" alt="Gamma函数性质.png" width="400" height="300" />

## 2. Gamma Distribution

把 Gamma 函数等式两边同时除以 $\Gamma(\alpha)$，得到：

<img src="https://s2.loli.net/2024/04/10/1OSXgUjPNF92b5C.png" alt="Gamma分布.png" width="400" height="300" />

把被积函数作为概率密度，就得到一个简单的 Gamma 分布：

<img src="https://s2.loli.net/2024/04/10/36U2R9JErO7vaiA.png" alt="Gamma分布密度函数.png" width="400" height="300" />

再作变换 $t=\beta x$，就得到 Gamma 分布的一般形式：

<img src="https://s2.loli.net/2024/04/10/7So1LMAUml2OEkr.png" alt="Gamma分布.png" width="400" height="300" />


> 其中，$\alpha$ 称为 shape parameter，主要决定分布曲线的形状；$\beta$ 称为 rate parameter 或 inverse scale parameter（$\frac {1} {\beta}$ 称为 scale parameter），主要决定曲线有多陡。

<img src="https://s2.loli.net/2024/04/10/zTtJFMWeSVPYlGr.png" alt="image.png" width="400" height="300" />

## 3. Binomial Distribution

二项分布描述 n 次独立的是/非试验中成功次数的离散概率分布，每次试验的成功概率均为 p。
> 已知 p 时，可以求出指定成功次数对应的概率，通常使用 $C(m, n, p)$ 计算。


n 次试验中恰好成功 k 次的概率，由下面的概率质量函数给出：
<img src="https://s2.loli.net/2024/04/10/HVzXuTD9Ka5UYlM.png" alt="image.png" width="400" height="300" />

:::note
参数 p 本身也有分布。如何估计这个分布？可以参考后面的 Beta 分布。
:::

## 4. Binomial Distribution 的另一个理解

<img src="https://s2.loli.net/2024/04/10/tq31VBQ9alucOEf.png" alt="image.png" width="400" height="300" />

如上图，向长度为 1 的桌子上扔一个红球，它会落在 0 到 1 的范围内。把这个位置记为 x，也就是上面定义中的 p。再扔一个白球，它落在红球左边的概率就是 x（或 p）。如果独立地扔 n 个白球，其中落在红球左边的有 k 个，那么给定 x（或 p）时，次数 k 的分布为：

<img src="https://s2.loli.net/2024/04/10/B6849qkCnEarUwd.png" alt="image.png" width="400" height="300" />

结果就是二项分布。如果**不关注**具体的概率 p，而要计算总体上成功 k 次的概率，就需要对 p 积分，在这里即对 x 积分：

<img src="https://s2.loli.net/2024/04/10/CQsWx1A2lDktw8V.png" alt="image.png" width="400" height="300" />

这个积分比较难算，可以换个思路：P(K=k) 表示**随机放置 n 个白球和 1 个红球后，红球左边恰好有 k 个白球的概率**。这里红球的位置未知。

假设 k=1，相当于把 n+1 个球中的第 2 个涂成红色，左边就有 1 个白球：$P = \frac {1} {n+1}$

假设 k=2，就把第 3 个球涂成红色，左边有 2 个白球：$P = \frac {1} {n+1}$

...

同理可得：

<img src="https://s2.loli.net/2024/04/10/CGDNJqa5zn3S1AT.png" alt="image.png" width="400" height="300" />

## 5. Beta Function

在上式中，令 $k = \alpha - 1$、$n - k = \beta - 1$，则 $n = \alpha + \beta - 2$。把积分变量换为 $t$，得到：

<img src="https://s2.loli.net/2024/04/10/STYmzEcR9wQs4pV.png" alt="image.png" width="400" height="300" />

- 定义 $Beta$ 函数

<img src="https://s2.loli.net/2024/04/10/b8duz7lpE5vGVCs.png" alt="image.png" width="400" height="300" />

根据前面 $Gama(\alpha)$ 函数的定义，$Beta(\alpha, \beta)$ 可以表示为：

<img src="https://s2.loli.net/2024/04/10/pkcCZHqhrFNwxmV.png" alt="image.png" width="400" height="300" />

## 6. Beta Distribution

- 定义 $Beta$ 分布
<img src="https://s2.loli.net/2024/04/10/5PQtVqLlzJxsRCD.png" alt="image.png" width="400" height="300" />

其中，$Beta(\alpha, \beta)$ 起归一化作用。

:::note
在贝叶斯推断中，Beta 分布是 Bernoulli、二项、负二项和几何分布的共轭先验；这些分布看起来都与成功和失败有关。[conjugate prior](https://stats.stackexchange.com/questions/58564/help-me-understand-bayesian-prior-and-posterior-distributions/58792#58792) 这篇回答解释了共轭先验的含义。
:::

- 性质

$Beta$ 分布与前面的 `Bernoulli` 分布（0–1 分布）、`Binomial` 分布（即二项分布，$C(m, n, p)$）构成共轭分布。

- [1] `Binomial` 分布（二项分布）描述给定成功概率 $p$ 和试验次数 $n$ 时，成功 $k$ 次的概率分布。


- [2] Beta 分布描述未知成功概率 $p$ 的不确定性。$\alpha,\beta$ 是分布的形状参数，**不直接等于**观察到的成功和失败次数。如果先验为 $\mathrm{Beta}(\alpha_0,\beta_0)$，观察到 $k$ 次成功、$n-k$ 次失败后，后验为 $\mathrm{Beta}(\alpha_0+k,\beta_0+n-k)$；这就是 Beta 先验与二项分布似然的共轭关系。

## 7. Multinomial Distribution
- 定义

<img src="https://s2.loli.net/2024/04/10/ZX3Y9TQ8Sx2OIF4.png" alt="image.png" width="400" height="300" />

> [1] 这里 p 已知，我们要求指定成功次数对应的概率。参数 p 本身的分布，可以参考后面的 Dirichlet Distribution。
>
:::note
[2] Multinomial Distribution 即多项分布，可以理解为多维的“二项分布”。
:::

用 $\Gamma$ 函数表示上式：

<img src="https://s2.loli.net/2024/04/10/beKZFcOQVUYS5Dp.png" alt="image.png" width="400" height="300" />

## 8. Multinomial Distribution的另一种理解

沿用前面的小球例子，可以这样理解多项分布：

<img src="https://s2.loli.net/2024/04/10/e4uyMEb7FdQNKUg.png" alt="image.png" width="400" height="300" />


将 $n$ 个球放入不同的箱子，求每个箱子分别分到 $x_1, x_2\ ...\ x_n$ 个球的概率。也可以把小球一字排开，在中间放隔板，隔板位置按一定概率 $p_i$ 选择，使各间隔内的球数为 $x_1, x_2\ ...\ x_n$，得到：


$$
f(x_1 ,  \ ... \ , x_k,  n ,  p_1, p_2 \ ... \ p_n)\ = \ C*p_1^{x_1} * \ ... \ p_n^{x_n}
$$


系数 C 可以这样理解：我们只关心每个箱子里有多少球，不关心箱内的顺序。因此，可以先按有序情况计数，再去掉箱内顺序造成的重复，得到不同放置情况的数量。

- 先考虑有顺序的情况，n 个小球的全排列共有 $n!$ 种
- 再分别去掉每个箱子内部的顺序，也就是除以 $x_i{!}$，便得到上面的结果


##  9. Dirichlet Distribution

- 定义

<img src="https://s2.loli.net/2024/04/10/JLZRe6PzWrCmAS5.png" alt="image.png" width="400" height="300" />

其中，$Beta(\alpha)$ 仍起归一化作用，$x$ 就是参数 $p$，并且：

<img src="https://s2.loli.net/2024/04/10/jRQwx6fGzbdcmT1.png" alt="image.png" width="400" height="300" />

同理，用 $\Gamma$ 函数可以得到 $Beta(\alpha)$ 的另一种表示：

<img src="https://s2.loli.net/2024/04/10/1mqPgNeivSyKR8r.png" alt="image.png" width="400" height="300" />

:::note
多项分布给定 $K$ 个互斥类别的概率向量 $p$，描述各类别计数的分布；Dirichlet 分布描述未知概率向量 $p$ 的不确定性。若先验为 $\mathrm{Dirichlet}(\alpha_1,\ldots,\alpha_K)$，观测到各类别计数 $m_1,\ldots,m_K$ 后，后验为 $\mathrm{Dirichlet}(\alpha_1+m_1,\ldots,\alpha_K+m_K)$。形状参数 $\alpha_i$ 不直接等于观测计数。这与 Beta–二项分布的共轭关系类似。
:::



- 性质

以一次试验只产生一个类别（例如掷骰子）的场景为例，令 $X$ 为各类别的出现次数。独立老虎机各自的成功率不要求总和为 1，不能直接当作同一个多项分布的类别概率。

<img src="https://s2.loli.net/2024/04/11/DUZz4WqcwAkCMIg.png" alt="image.png" width="100" height="130" />

令各互斥类别的概率为

<img src="https://s2.loli.net/2024/04/11/d8biLJZpeyh6N9r.png" alt="image.png" width="100" height="130" />

随机变量 $X$ 服从下面的多项分布：

<img src="https://s2.loli.net/2024/04/11/aiI1wQcuFTCPGDZ.png" alt="image.png" width="400" height="300" />

可以为概率向量 $p$ 选择 Dirichlet 先验。

- 应用

下面的例子与 Bandit 类似，只不过使用的是骰子。

<img src="https://s2.loli.net/2024/04/11/NSyKDk29Fu8Vql6.png" alt="image.png" width="400" height="300" />

假设不知道每个 bandit 的成功概率，在这里就是不知道骰子每个面朝上的概率。要通过试验估计这些概率，可以写成：

<img src="https://s2.loli.net/2024/04/11/wlBVrdCth17y6Go.png" alt="image.png" width="400" height="300" />

其中：

<img src="https://s2.loli.net/2024/04/11/OhzY7IS8T3LK6mE.png" alt="image.png" width="400" height="300" />

相应发生次数的概率计算如下：

<img src="https://s2.loli.net/2024/04/11/bpCENXRI2jQkr4Z.png" alt="image.png" width="400" height="300" />

根据试验结果估计潜在的 $p$，也就是求后验分布，可以使用 Bayes rule：

<img src="https://s2.loli.net/2024/04/11/fK29OLBqdgm4vR5.png" alt="image.png" width="400" height="300" />

其中，分母 $P(X=m)$ 为：

![image.png](https://s2.loli.net/2024/04/11/g32j7IYrqBasc9F.png)

分子中的 $f_p(P)$ 就是前面的 Dirichlet 先验分布，整体代入得到：

<img src="https://s2.loli.net/2024/04/11/Omz684tcdPBZixp.png" alt="image.png" width="400" height="300" />

这里 c 起归一化作用：

<img src="https://s2.loli.net/2024/04/11/ebsgrm3vEq57QYO.png" alt="image.png" width="400" height="300" />

可以看到，**潜在参数 $p$ 的后验分布仍然是 Dirichlet distribution**：

<img src="https://s2.loli.net/2024/04/11/Gl8dEhwBxR3TKHg.png" alt="image.png" width="400" height="300" />

即：

<img src="https://s2.loli.net/2024/04/11/DcgUu9rbPnyZIAL.png" alt="image.png" width="400" height="300" />

因此，对于每个 Bandit 或骰子，先初始化 Dirichlet distribution，再根据试验结果更新相应的概率分布。然后按当前分布进行下一步采样，也就是选择当前认为获胜概率更高的 Bandit，观察结果，再继续更新分布。

:::note
如何根据已有分布采样？可以接着看 [Sampling Method](/posts/2024-04-11-sampling-method/)。
:::



## Reference

[1] [https://zhuanlan.zhihu.com/p/37976562](https://zhuanlan.zhihu.com/p/37976562)

[2] [https://zhuanlan.zhihu.com/p/69606875](https://zhuanlan.zhihu.com/p/69606875)

[3] [https://readmedium.com/en/https:/towardsdatascience.com/dirichlet-distribution-the-underlying-intuition-and-python-implementation-59af3c5d3ca2](https://readmedium.com/en/https:/towardsdatascience.com/dirichlet-distribution-the-underlying-intuition-and-python-implementation-59af3c5d3ca2)







