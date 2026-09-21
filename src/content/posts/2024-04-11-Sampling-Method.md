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


比如在老虎机场景,  我们想知道哪一台老虎机的赢面更大,  通常是给定所有老虎机 "赢" 的参数分布 ,  比如 Dirichlet distribution,  初始化 $\alpha1 \ \alpha2 \ …$  ,  然后根据实际数据采样,  更新 Dirichlet distribution 的参数即可.

具体采样流程(通常使用在类似多臂老虎机场景) :

[1] 首先假设 参数p的先验分布 (比如 beta 分布 $B(m, n)$,  Dirichlet 分布 $D(a, b, c, ..., z)$)

[2] 然后 **基于该分布 ,   采样一组参数(就是各个机器的成功概率)** ,  然后基于当前的参数抽卡,  并选择最大的p对应的老虎机作为成功case ,  然后观察其结果,  并更新对应参数(比如实际是另外一个老虎机赢了). 重复此步骤.


:::note
**这里就会涉及到一个问题,  对参数采样,  怎么采才能尽可能的符合、或者接近参数本身的分布**？
:::


## 1. 基于Monte-Carlo的方法

- 引理1

> 设 X 是一个随机变量，其分布函数$f(x)$,  累积分布函数 (CDF,  Cumulative distribution function) 为 F(x) ,  该函数是一个单调递增的函数,  其值域为[ 0 ,   1 ]. 现在定义一个新的随机变量$Y = F(X) $ ,  则 随机变量 $Y$ 的分布是均匀分布.

- 证明


> 对于任意实数$y$ ,  我们有:
>
> 
$$
P(Y<=y)  = P(F(X) <= y) = P(X <= F^{-1}(y) = F( F^{-1}(y))
$$

>
> 由于F(x)是单调递增函数, 因此$F^{-1}(y)$具有唯一解 $x$ , 令$x = F^{-1}(y)$ , 则有 $F(x) = y$ .
>
> 因此
>
> 
$$
P( Y <= y) = F(F^{-1}(y)) = F(x) = y
$$

>
> 即有
>
> 
$$
P( Y <= y)  = y
$$

>
> 即 Y是均匀分布


### 1.1 逆变换采样法

设 X 是一个随机变量，其分布函数$f(x)$,  累积分布函数 (CDF,  Cumulative distribution function) 为 F(x). 则依据如下采样过程,  得到的x是服从分布$f(x)$的.

1. 从均匀分布 U(0,  1) 中生成一个随机数 u
2. 计算 F(x) = u 的解 x
3. 输出 x 作为采样结果

- 证明

> 根据引理1容易知道,  如果从均匀分布 $U(0,  1) $ 中生成一个随机数 u，并令 $x = F^{-1} (u)$，则 $x$ 服从原分布$ F(x)$。(理解为本身这个$F$就是我们想采样的 $f$ 对应的 $F$,  那反函数求解出来的 $x$ 自然就是 满足 $f(x)$ 和 $F(x)$ ) ,  即为 逆变换方法 ,  几个具体实现: [https://lwz322.github.io/2019/06/02/ITM.html](https://lwz322.github.io/2019/06/02/ITM.html)

### 1.2 拒绝采样法

- 准备工作
    1. 已知 概率密度函数$f(y)$,  我们需要依据这个分布进行抽样
    2. 找一个能够直接采样、且在 $f(y)>0$ 的地方也满足 $g(y)>0$ 的提议分布 $g(y)$（如在有界支撑上选合适的均匀分布）
    3. 找一个常数 $c$,  满足对 $\forall y$ ,  均有 $c \times g(y) >= f(y)$,  即 $c$ 是函数 $\frac {f(y)} {g(y)}$ 的上界 或者 $c \times g(y)$ 能够覆盖 $f(y)$

- 抽样流程
    1. 从 $g(y)$ 中中随机采样一个样本 $y_i$
    2. 从均匀分布 $U(0, 1)$ 中采样一个随机数 $u_i$
    3. 如果 $u_i \le \frac{f(y_i)}{c g(y_i)}$，则保留样本，否则返回第 1 步。被保留样本的密度为 $f$，接受率为 $1/c$（假设 $f,g$ 均已归一化）

- 证明

令 $Y\sim g$、$U\sim U(0,1)$ 且二者独立，接受事件为 $A=\{U\le f(Y)/(c g(Y))\}$。对连续变量，$g(y)$ 是**密度**，不是点事件 $P(Y=y)$ 的概率。由全概率公式：

$$
\begin{aligned}
P(A)&=\int g(t)\frac{f(t)}{c g(t)}\,dt=\frac1c,\\
P(Y\le y,A)&=\int_{-\infty}^{y}g(t)\frac{f(t)}{c g(t)}\,dt
=\frac{F(y)}c.
\end{aligned}
$$

因此 $P(Y\le y\mid A)=P(Y\le y,A)/P(A)=F(y)$，即接受后的样本服从目标分布。这个证明同时说明：必须有 $c g(y)\ge f(y)$，否则所谓的接受概率可能超过 1。


- 直觉理解

假设复杂分布 $P(z)$ ,  存在常数 $k$ 与 任意分布 $q(z)$ ,  以 $z_0$ 点为例,  画直线,  任意从均匀分布抽取一个点 $u_i$,  可以理解为在 $x = z_0$ 这条直线上取一点: 就是 $u_i  * k * q(z_0)$,  其处于阴影即拒绝 (即 $U * k * q(z_0) > p(z_0)$) , 处于白色区域即接受( $U * k * q(z_0) <= p(z_0)$ ) ,  这样从 $z_0$ 出来的点对应的最大概率就是$ f(z_0) $ , 等价于是从 $f(x)$ 抽样出来的


![image.png](https://s2.loli.net/2024/04/11/XO3GehobsnckrNQ.png)

----
上述2个方法都属于Monte-Carlo 方法,  并且是已知 $P(\theta)$ 的情况下 ,  然后在某些特殊场景下,  已知了 参数的后验分布 和 先验分布 的关系(比如之前提到的共轭) , 才能得到一个比较简易的形式 ,  直接对后验分布更新. (当我们面临无法得到具体形式的非共轭后验分布时，我们无法采用这种算法。)

然而,  面对一些复杂的分布,  即使我们已知了 $P(\theta)$  ,  再利用贝叶斯公式的时候 ,  其分母涉及到积分, 往往也是很难求解的


$$
P(\theta|X) = \frac {P(X|\theta)P(\theta)} {\int P(X|\theta)P(\theta) d \theta}
$$


上述提到分母有时候很难进行积分，对于这个问题，一个直观的想法就是 ，能不能通过某个手段把 分母去掉？


$$
P(\theta_a|X) = \frac {P(X|\theta_a)P(\theta_a)} {P(X)}
$$


$$
P(\theta_b|X) = \frac {P(X|\theta_b)P(\theta_b)} {P(X)}
$$


二者做比值


$$
\gamma = \frac {P(\theta_a|X)}{P(\theta_b|X)} = \frac {P(X|\theta_a)P(\theta_a)}{P(X|\theta_b)P(\theta_b)}
$$


这样避免了分母的积分，这里 $P(\theta_a)$ 可以参考 Dirichlet Distribution （多维）或者 Beta Distribution （二维）. 思想是这样的,  不过需要一点点其他知识.

:::note
未完待续...
:::


##  Reference
