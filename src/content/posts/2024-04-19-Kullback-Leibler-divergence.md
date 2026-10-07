---
title: KL 散度：定义、直觉与高斯分布推导
published: 2024-04-19
description: 介绍 KL 散度（Kullback-Leibler divergence）的定义、离散与连续版本的公式，以及其在衡量两个概率分布差异性中的核心作用。
category: Deep Learning
tags:
- machine learning
- mathematics
- statistics
- deep learning
draft: false
---
## 0. 前言

假设真实数据来自分布 $P$，而我们用分布 $Q$ 去预测。怎样衡量这套预测和真实分布之间的差异？

先从一次观测看：结果 $x$ 发生时，预测概率越小，负对数损失 $-\ln Q(x)$ 越大。如果换成真实分布，损失就是 $-\ln P(x)$。**KL 散度比较的是两种损失之差，再按照真实分布 $P$ 求平均。**

这也解释了为什么要写清方向：$D_{\mathrm{KL}}(P\|Q)$ 按 $P$ 加权，换成 $D_{\mathrm{KL}}(Q\|P)$ 就换了参照分布，通常得到不同答案。

本文统一使用自然对数 $\ln$，单位是 nat；改用 $\log_2$ 时单位为 bit，数值相差一个固定比例。

## 1. 定义

### 1.1 离散版本与零概率

设 $P$ 和 $Q$ 定义在同一个离散样本空间 $\mathcal X$ 上：

$$
\begin{aligned}
D_{\mathrm{KL}}(P\|Q)
&=\sum_{x\in\mathcal X}P(x)\ln\frac{P(x)}{Q(x)}\\
&=\mathbb E_{X\sim P}\left[\ln\frac{P(X)}{Q(X)}\right].
\end{aligned}
$$

遇到零概率时，不能直接按普通除法计算：[1]

- 若 $P(x)=0$，这一项按约定取 0，包括 $P(x)=Q(x)=0$ 的情况；可以直接只对 $P(x)>0$ 的位置求和
- 若 $P(x)>0$ 但 $Q(x)=0$，则 $D_{\mathrm{KL}}(P\|Q)=+\infty$：真实可能发生的结果，被预测模型判定为绝不可能

因此，要让 KL 有限，$Q$ 必须覆盖所有在 $P$ 下有正概率的位置。对于有限样本空间，这个条件足够；对于无限样本空间，即使每个这样的 $Q(x)$ 都大于 0，求和仍可能发散。

例如，$P=(1,0)$、$Q=(1/2,1/2)$ 时，$D_{\mathrm{KL}}(P\|Q)=\ln2$，但反过来的 KL 是 $+\infty$。方向不仅影响大小，还可能影响是否有限。

### 1.2 连续版本与绝对连续

若 $P,Q$ 相对于同一坐标空间上的体积元素都有密度 $p,q$，定义为：

$$
D_{\mathrm{KL}}(P\|Q)
=\int p(x)\ln\frac{p(x)}{q(x)}\,dx.
$$

这里的 $p(x),q(x)$ 是密度，不是单点概率。判断是否遗漏了真实可能发生的结果，要看一整块集合承载的概率，不能仅凭“某个点的密度为 0”下结论。

严格地说，这个条件是：对于任意可测集合 $A$，只要 $Q(A)=0$，就有 $P(A)=0$。这称为 **$P$ 对 $Q$ 绝对连续**，记为 $P\ll Q$。如果这个条件不成立，KL 为 $+\infty$；条件成立也不保证积分一定有限。

## 2. 它为什么不是距离？

### 2.1 非负性与取零的条件

KL 满足：

$$
D_{\mathrm{KL}}(P\|Q)\ge0,
\qquad
D_{\mathrm{KL}}(P\|Q)=0\iff P=Q.
$$

离散情形的 $P=Q$ 表示每个位置的概率都相同；密度形式中表示 $p=q$ 几乎处处，即允许在零测集上不同，但它们仍定义同一个概率分布。[1]

下面在有限离散空间上证明。令 $S=\{x:P(x)>0\}$；先假设 $Q(x)>0$ 对所有 $x\in S$ 成立。由于 $-\ln$ 是严格凸函数，使用 Jensen 不等式：

$$
\begin{aligned}
D_{\mathrm{KL}}(P\|Q)
&=\sum_{x\in S}P(x)\left[-\ln\frac{Q(x)}{P(x)}\right]\\
&\ge-\ln\left(\sum_{x\in S}P(x)\frac{Q(x)}{P(x)}\right)\\
&=-\ln\left(\sum_{x\in S}Q(x)\right)\ge0.
\end{aligned}
$$

这里不能丢掉期望的权重 $P(x)$。若某处 $P(x)>0$ 而 $Q(x)=0$，KL 为 $+\infty$，非负性仍成立。

若最后等于 0，上述两个不等式必须同时取等号：第一步要求 $Q(x)/P(x)$ 在 $S$ 上为常数，第二步要求 $\sum_{x\in S}Q(x)=1$。两者合起来得到常数为 1，且 $Q$ 在 $S$ 外没有概率质量，所以 $P=Q$。反过来，把 $P=Q$ 代入定义自然得到 0。

### 2.2 不对称

取 $P=(0.1,0.9)$、$Q=(0.2,0.8)$：

$$
D_{\mathrm{KL}}(P\|Q)\approx0.036690,
\qquad
D_{\mathrm{KL}}(Q\|P)\approx0.044403.
$$

所以 KL 一般不满足对称性。不过，这并不表示对所有 $P,Q$ 都有“不等”；某些特殊分布的两个方向也会相等。

### 2.3 不满足三角不等式

仍取上面的 $P,Q$，再令 $R=(0.3,0.7)$。直接计算可得：

$$
\begin{aligned}
D_{\mathrm{KL}}(P\|R)&\approx0.116322,\\
D_{\mathrm{KL}}(P\|Q)+D_{\mathrm{KL}}(Q\|R)
&\approx0.036690+0.025732\\
&\approx0.062422.
\end{aligned}
$$

左边大于右边，给出了三角不等式不成立的具体反例。因此 KL 可以衡量分布差异，但不是数学意义上的距离。

## 3. 与熵、交叉熵和互信息的关系

### 3.1 固定真实分布时，为什么用交叉熵训练？

对离散分布，熵和交叉熵分别为：

$$
H(P)=-\sum_xP(x)\ln P(x),
\qquad
H(P,Q)=-\sum_xP(x)\ln Q(x).
$$

在这些量有限时，展开定义得到：

$$
\begin{aligned}
D_{\mathrm{KL}}(P\|Q)
&=\sum_xP(x)\ln P(x)-\sum_xP(x)\ln Q(x)\\
&=H(P,Q)-H(P).
\end{aligned}
$$

如果真实分布 $P$ 固定，只调整模型 $Q$，$H(P)$ 就是不随模型参数变化的常数。因此最小化交叉熵，等价于最小化这个方向的 KL。实践中通常用数据样本的平均负对数似然估计交叉熵。

这里有两个边界：若 $P$ 也随待优化参数变化，就不能随意丢掉 $H(P)$；在无限或连续空间里，也不能把未定义的“无穷减无穷”当作合法推导。连续情形需使用微分熵，并保证相应积分和差值有意义。

:::note
如果想先理解熵为什么采用负对数，可以看我在 B站 的[介绍视频](https://www.bilibili.com/video/BV1kg411u7RP)。语言模型中的交叉熵与困惑度，可接着阅读 [Perplexity](/posts/2024-04-10-perplexity/)。
:::

### 3.2 互信息是特定两个分布之间的 KL

知道 $X$ 后，能减少多少对 $Y$ 的不确定性？这由互信息描述。此时比较的两种分布必须明确：一边是实际联合分布 $P_{XY}$，另一边是把两者视作独立时的边缘分布乘积 $P_XP_Y$。[1]

在有限离散情形下：

$$
\begin{aligned}
I(X;Y)
&=D_{\mathrm{KL}}(P_{XY}\|P_XP_Y)\\
&=\sum_{x,y}p(x,y)\ln\frac{p(x,y)}{p(x)p(y)}\\
&=\sum_{x,y}p(x,y)\ln\frac{p(y\mid x)}{p(y)}\\
&=H(Y)-H(Y\mid X).
\end{aligned}
$$

条件熵 $H(Y\mid X)$ 是在不同 $X$ 取值下的熵再求平均。根据 KL 取零的条件：

$$
I(X;Y)=0
\iff P_{XY}=P_XP_Y
\iff X\text{ 与 }Y\text{ 独立}.
$$

这是“互信息等于某个 KL”的特殊构造。任意两个分布的 KL 并不能都解释为两个变量的互信息；减小任意 $D_{\mathrm{KL}}(P\|Q)$，也不能直接解释为“让两个变量更独立”或“只让两个熵的数值接近”。熵相同的两个分布，仍然可能有很大的 KL。

## 4. 两个多元高斯分布之间的 KL

高斯分布之间的差异来自均值和协方差。均值不同意味着分布中心偏移；协方差不同意味着各方向的伸展程度不同。下面把这两部分写成可计算的表达式。

设：

$$
P=\mathcal N(\boldsymbol\mu_p,\Sigma_p),
\qquad
Q=\mathcal N(\boldsymbol\mu_q,\Sigma_q),
$$

其中 $\boldsymbol\mu_p,\boldsymbol\mu_q\in\mathbb R^k$ 为列向量，$\Sigma_p,\Sigma_q\in\mathbb R^{k\times k}$ 均为对称正定矩阵。正定条件保证逆矩阵与正的行列式存在；以下密度和公式不能直接套在奇异协方差上。

### 4.1 从密度展开

$P$ 的密度为：

$$
p(\mathbf x)
=\frac{1}{(2\pi)^{k/2}|\Sigma_p|^{1/2}}
\exp\left[-\frac12(\mathbf x-\boldsymbol\mu_p)^T\Sigma_p^{-1}(\mathbf x-\boldsymbol\mu_p)\right].
$$

把两个对数密度相减，再对 $\mathbf X\sim P$ 求期望：

$$
\begin{aligned}
D_{\mathrm{KL}}(P\|Q)
={}&\frac12\ln\frac{|\Sigma_q|}{|\Sigma_p|}\\
&-\frac12\mathbb E_P\left[(\mathbf X-\boldsymbol\mu_p)^T\Sigma_p^{-1}(\mathbf X-\boldsymbol\mu_p)\right]\\
&+\frac12\mathbb E_P\left[(\mathbf X-\boldsymbol\mu_q)^T\Sigma_q^{-1}(\mathbf X-\boldsymbol\mu_q)\right].
\end{aligned}
$$

注意，这两个二次型都在 $P$ 下求期望，不能把最后一项的期望分布换成 $Q$。

### 4.2 怎样计算二次型的期望？

先考虑一个更通用的式子。设 $A$ 是固定的 $k\times k$ 矩阵，$\mathbf a$ 是固定向量。令 $\boldsymbol\delta=\boldsymbol\mu_p-\mathbf a$，把 $\mathbf X-\mathbf a$ 拆成 $(\mathbf X-\boldsymbol\mu_p)+\boldsymbol\delta$。

因为 $\mathbb E_P[\mathbf X-\boldsymbol\mu_p]=0$，展开后二次型的交叉项期望为 0，剩下：

$$
\mathbb E_P\left[(\mathbf X-\mathbf a)^TA(\mathbf X-\mathbf a)\right]
=\operatorname{tr}(A\Sigma_p)
+(\boldsymbol\mu_p-\mathbf a)^TA(\boldsymbol\mu_p-\mathbf a).
$$

其中 $\operatorname{tr}$ 表示矩阵对角线元素之和。协方差项的推导是：

$$
\begin{aligned}
\mathbb E_P\left[(\mathbf X-\boldsymbol\mu_p)^TA(\mathbf X-\boldsymbol\mu_p)\right]
&=\operatorname{tr}\left(A\,\mathbb E_P[(\mathbf X-\boldsymbol\mu_p)(\mathbf X-\boldsymbol\mu_p)^T]\right)\\
&=\operatorname{tr}(A\Sigma_p).
\end{aligned}
$$

这个恒等式只需要均值和协方差存在，不依赖高斯假设。分别代入 $A=\Sigma_p^{-1},\mathbf a=\boldsymbol\mu_p$ 和 $A=\Sigma_q^{-1},\mathbf a=\boldsymbol\mu_q$，得到：

$$
\begin{aligned}
\mathbb E_P[(\mathbf X-\boldsymbol\mu_p)^T\Sigma_p^{-1}(\mathbf X-\boldsymbol\mu_p)]&=k,\\
\mathbb E_P[(\mathbf X-\boldsymbol\mu_q)^T\Sigma_q^{-1}(\mathbf X-\boldsymbol\mu_q)]
&=\operatorname{tr}(\Sigma_q^{-1}\Sigma_p)
+(\boldsymbol\mu_p-\boldsymbol\mu_q)^T\Sigma_q^{-1}(\boldsymbol\mu_p-\boldsymbol\mu_q).
\end{aligned}
$$

### 4.3 最终公式与检查

合并上述各项：[2]

$$
\boxed{
\begin{aligned}
D_{\mathrm{KL}}(P\|Q)=\frac12\Bigl[&\ln\frac{|\Sigma_q|}{|\Sigma_p|}-k
+\operatorname{tr}(\Sigma_q^{-1}\Sigma_p)\\
&+(\boldsymbol\mu_p-\boldsymbol\mu_q)^T\Sigma_q^{-1}(\boldsymbol\mu_p-\boldsymbol\mu_q)\Bigr].
\end{aligned}
}
$$

几个有用的检查：

- 当 $P=Q$，行列式比为 1，迹为 $k$，均值差为 0，所以 KL 为 0
- 当 $\Sigma_p=\Sigma_q=\Sigma$，只剩 $\tfrac12(\boldsymbol\mu_p-\boldsymbol\mu_q)^T\Sigma^{-1}(\boldsymbol\mu_p-\boldsymbol\mu_q)$
- 当 $Q=\mathcal N(\mathbf0,I_k)$，公式简化为

$$
D_{\mathrm{KL}}(P\|Q)
=\frac12\left[\boldsymbol\mu_p^T\boldsymbol\mu_p
+\operatorname{tr}(\Sigma_p)-k-\ln|\Sigma_p|\right].
$$

例如 $P=\mathcal N(1,4)$、$Q=\mathcal N(0,1)$，第二个参数表示方差：

$$
D_{\mathrm{KL}}(P\|Q)
=\frac12(1+4-1-\ln4)
\approx1.306853.
$$

### 4.4 二次型期望不是高斯熵

上面的 $k$ 只是平方偏差经过协方差逆矩阵加权后的期望。它是计算熵时的一部分，但不等于高斯分布的熵。保留密度中的归一化常数，才能得到微分熵：[1]

$$
\begin{aligned}
h(P)
&=-\mathbb E_P[\ln p(\mathbf X)]\\
&=\frac{k}{2}\ln(2\pi)+\frac12\ln|\Sigma_p|+\frac{k}{2}\\
&=\frac12\ln\left((2\pi e)^k|\Sigma_p|\right).
\end{aligned}
$$

微分熵可能为负，并且会随坐标单位改变；KL 仍然非负。不要把离散熵的所有直觉直接套到微分熵上。

## Reference

[1] [John Duchi：Lecture Notes on Statistics and Information Theory，第 2 章](https://stanford.edu/class/stats311/lecture-notes.pdf)

[2] [Zhang 等：On the Properties of Kullback-Leibler Divergence Between Multivariate Gaussian Distributions](https://arxiv.org/abs/2102.05485)

[3] [原高斯推导参考：KL Divergence between 2 Gaussian Distributions](https://mr-easy.github.io/2020-04-16-kl-divergence-between-2-gaussian-distributions/)
