---
title: L1 and L2 Regularization
published: 2024-04-20
description: 从多个角度探讨 L1 和 L2 正则化的原理，解释其为何能有效防止模型过拟合，涵盖公式推导、几何解释和贝叶斯视角。
category: Machine Learning
tags:
- machine learning
- mathematics
- statistics
draft: false
---
## 0. 前言

模型可以把训练集拟合得很好，却把噪声也当成规律。L1 和 L2 Regularization 在训练目标中加入对参数大小的偏好，限制某些过度拟合的方式。本篇分别从优化、几何和贝叶斯角度解释它们，并说明哪些直觉需要附加条件。

:::note
需要的前置知识是导数、向量范数和基本概率。正则化强度仍要通过验证集等方式选择；加入正则项并不保证所有模型的泛化性能都会提高。
:::

## 1. 公式与系数约定

设模型参数 $w\in\mathbb R^d$，数据损失为 $L(w)$。例如：

$$
L(w)=\frac1n\sum_{i=1}^n\ell(f(w,x_i),y_i).
$$

本文使用如下约定，且始终令 $\lambda\ge0$：

$$
J_{\mathrm{L1}}(w)=L(w)+\lambda\|w\|_1,
\qquad
J_{\mathrm{L2}}(w)=L(w)+\lambda\|w\|_2^2.
$$

其中 $\|w\|_1=\sum_j|w_j|$，$\|w\|_2^2=\sum_jw_j^2$。L2 正则项使用的是**范数的平方**。也有资料把它写成 $(\lambda_2/2)\|w\|_2^2$，这样梯度是 $\lambda_2w$；与本文等价时需要 $\lambda_2=2\lambda$。

损失取总和还是平均值也会改变正则项的相对强度，不能脱离损失定义直接比较两个实现中的 `lambda`。

## 2. 为什么限制参数可能有帮助

<img src="https://s2.loli.net/2024/04/20/csCq1bnfWRQ7mg4.png" alt="不同复杂度分类边界的示意" width="300" height="300" />

训练数据含有噪声时，过于曲折的分类边界可能同时记住规律和偶然误差。正则化可以让优化更偏向某些简单解，但“参数小”与“函数平滑”不是普遍等价的。

### 2.1 Taylor 展开能说明什么

先固定参数 $w$，只考虑单变量输入 $x$。若 $f(w,\cdot)$ 在 $a$ 附近三次连续可微，则有有限阶展开：

$$
f(w,x)=f(w,a)+\partial_x f(w,a)(x-a)
+\frac{\partial_x^2f(w,a)}2(x-a)^2+R_3(x).
$$

如果连接 $a$ 与 $x$ 的区间上满足 $|\partial_x^3f(w,\xi)|\le M$，则：

$$
|R_3(x)|\le\frac M6|x-a|^3.
$$

因此，局部导数可以描述变化率、曲率和近似误差。但这个公式并未建立导数与权重大小之间的直接关系，也不意味着任意函数都等于其无穷 Taylor 级数。ReLU 在折点处不满足这里的可微条件，不能直接套用。

在线性模型 $f(w,x)=w^\top x$ 中，Cauchy–Schwarz 不等式给出：

$$
|f(w,x)-f(w,x')|\le\|w\|_2\|x-x'\|_2.
$$

这时限制 $\|w\|_2$ 确实限制了模型对输入扰动的敏感度。对多层网络、带归一化的模型或不同参数化，需要另外分析，不能仅靠一幅曲线或 Taylor 展开得出泛化结论。

## 3. 正则化的三种理解

### 3.1 给权重加约束：什么时候与惩罚形式对应

记 $R(w)=\|w\|_1$ 或 $R(w)=\|w\|_2^2$。约束问题是：

$$
\min_w L(w)\quad\text{subject to}\quad R(w)\le C,\qquad C\ge0.
$$

其 Lagrangian 为：

$$
\mathcal L(w,\lambda)=L(w)+\lambda(R(w)-C),\qquad\lambda\ge0.
$$

对固定 $\lambda$，$-\lambda C$ 与 $w$ 无关，所以最小化 Lagrangian 与最小化 $L(w)+\lambda R(w)$ 得到相同的 $w$。但这不意味着任意给定的 $C$ 与 $\lambda$ 都匹配，更不意味着可以忽略约束，直接令所有变量的导数等于零。

以可微的凸函数 $L:\mathbb R^d\to\mathbb R$ 和上述凸函数 $R$ 为例，若约束问题存在最优解并满足 Slater 条件，即存在 $\bar w$ 使 $R(\bar w)<C$，则最优解可以与非负乘子 $\lambda^*$ 配对，满足 KKT 条件：

$$
\begin{aligned}
R(w^*)&\le C &&\text{（可行性）},\\
\lambda^*&\ge0 &&\text{（乘子符号）},\\
\lambda^*(R(w^*)-C)&=0 &&\text{（互补松弛）},\\
0&\in\nabla L(w^*)+\lambda^*\partial R(w^*)
&&\text{（驻点条件）}.
\end{aligned}
$$

这里 $\partial R$ 是次梯度集合，允许处理 L1 在零点不可导的情况。在这些凸性和约束资格条件下，KKT 是最优性的刻画；一般非凸问题不能据此保证全局最优。[Boyd and Vandenberghe，Convex Optimization，第 5 章](https://www.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf)

对应关系有两个容易忽略的细节：

- 若已经求出惩罚问题的一个**全局**最优解 $w_\lambda$，取 $C=R(w_\lambda)$，则它也是这个约束问题的全局最优解；这个方向不需要凸性
- 反方向需要条件，且通常不能声称 $C$ 与 $\lambda$ 一一对应，或两个问题的整个解集完全相同。约束不活跃时，乘子可能为 0

一个边界反例是：最小化 $(w-1)^2$，约束 $w^2\le0$。唯一可行解是 0；而惩罚目标 $(w-1)^2+\lambda w^2$ 对任何有限 $\lambda\ge0$ 的最优解都是 $1/(1+\lambda)$，不等于 0。这里严格可行点不存在。

### 3.2 让权重衰减：先限定优化器

对本文的 L2 目标做普通梯度下降：

$$
\begin{aligned}
w_{t+1}
&=w_t-\eta_t\bigl(\nabla L(w_t)+2\lambda w_t\bigr)\\
&=(1-2\eta_t\lambda)w_t-\eta_t\nabla L(w_t).
\end{aligned}
$$

当 $0<2\eta_t\lambda<1$ 时，第一项确实把旧权重按比例缩小。注意总更新还包含数据梯度，因此不保证每一步的参数范数都下降。

这个代数等价适用于不带 momentum、没有自适应缩放的普通 SGD。将 L2 梯度加入 momentum 或 Adam 的状态，与独立衰减权重一般不同，详见 [Weight Decay 与 L2 Regularization](/posts/2024-04-20-weight-decay-and-l2-regularization/)。

### 3.3 参数先验：MLE 与 MAP 的区别

对于固定数据 $\mathcal D$，贝叶斯公式为：

$$
p(w\mid\mathcal D)=\frac{p(\mathcal D\mid w)p(w)}{p(\mathcal D)}.
$$

MLE 只最大化似然 $p(\mathcal D\mid w)$；MAP 则最大化后验密度，等价于：

$$
\hat w_{\mathrm{MAP}}=
\operatorname*{argmin}_w\left[-\log p(\mathcal D\mid w)-\log p(w)\right].
$$

这里证据 $p(\mathcal D)$ 与 $w$ 无关，可以从优化目标中去掉。

若各坐标独立服从零均值 Gaussian 先验 $w_j\sim\mathcal N(0,\sigma^2)$，则：

$$
p(w_j)=\frac1{\sqrt{2\pi}\sigma}
\exp\left(-\frac{w_j^2}{2\sigma^2}\right),\qquad
-\log p(w)=\frac{\|w\|_2^2}{2\sigma^2}+\text{constant}.
$$

若各坐标独立服从 Laplace 先验 $w_j\sim\operatorname{Laplace}(0,b)$，则：

$$
p(w_j)=\frac1{2b}\exp\left(-\frac{|w_j|}{b}\right),\qquad
-\log p(w)=\frac{\|w\|_1}{b}+\text{constant}.
$$

当 $L$ 是**总**负对数似然时，本文 L2 的 $\lambda=1/(2\sigma^2)$，L1 的 $\lambda=1/b$；若 $L$ 是 $n$ 个样本的平均负对数似然，则对应系数分别要除以 $n$。这是从 MAP 角度解释正则项，并不意味着训练得到的权重必须服从这个先验，也不涉及对整个后验做积分。

## 4. 为什么 L1 更容易产生零系数

### 4.1 密度形状：先把尺度写清楚

Laplace 在零点有尖角，Gaussian 在零点平滑；但“哪个峰更高”还取决于尺度。下面列出具体密度，避免根据标签或比例不准确的示意曲线作判断。

| 分布 | 密度 $p(w)$ | $p(0)$ | $p(1.5)$ | $p(3)$ |
| --- | --- | --- | --- | --- |
| $\operatorname{Laplace}(0,b=1.5)$ | $\frac13e^{-\lvert w\rvert/1.5}$ | $1/3\approx0.333333$ | $e^{-1}/3\approx0.122626$ | $e^{-2}/3\approx0.045112$ |
| $\mathcal N(0,\sigma^2=2.25)$ | $\frac1{1.5\sqrt{2\pi}}e^{-w^2/4.5}$ | $0.265962$ | $0.161314$ | $0.035994$ |

Gaussian 这一行的标准差是 1.5，**不是 Standard Normal**；标准正态必须是 $\mathcal N(0,1)$。Laplace 的 $b=1.5$ 意味着峰值为 $1/(2b)=1/3$，不是 0.2。两行尺度参数数值相同，不代表方差相同。密度定义可对照 [SciPy Laplace 文档](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.laplace.html) 与 [SciPy Normal 文档](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.norm.html)。

密度曲线只提供直觉。零系数的直接原因，要看惩罚项的次梯度。

### 4.2 L1 在零点的次梯度

绝对值函数在零点没有普通导数，但次梯度为：

$$
\partial|w_j|=\begin{cases}
\{1\},&w_j>0,\\
[-1,1],&w_j=0,\\
\{-1\},&w_j<0.
\end{cases}
$$

因此，对可微凸损失，L1 目标的最优性条件是：

$$
0\in\nabla L(w^*)+\lambda\partial\|w^*\|_1.
$$

在零坐标上，只要 $|\partial_jL(w^*)|\le\lambda$，就能选到一个次梯度抵消数据梯度。这允许一段范围内的数据梯度对应**恰好为零**的系数。相较之下，L2 的梯度 $2\lambda w_j$ 在 $w_j=0$ 时也是 0，通常只是连续缩小系数。L1 也不保证任何数据、任何 $\lambda$ 下都产生零；普通次梯度迭代更不保证每一步得到精确零。

### 4.3 可以手算的 Soft Thresholding

考虑最简单的平方损失：

$$
\min_w\frac12\|w-z\|_2^2+\lambda\|w\|_1.
$$

各坐标独立，最优解是 Soft Thresholding：

$$
w_j^*=\operatorname{sign}(z_j)\max(|z_j|-\lambda,0).
$$

取 $z=(3,-0.4,0.2)$、$\lambda=0.5$，得到 $w^*=(2.5,0,0)$。而在同一个平方损失上加入本文定义的 L2 项，解为：

$$
w^*=\frac{z}{1+2\lambda}=(1.5,-0.2,0.1).
$$

这不是说一般 LASSO 都能对原始观测值直接做一次阈值操作；这里的闭式解依赖于平方损失可按坐标分离。一般设计矩阵下需要另外求解。[Ryan Tibshirani：Subgradients and Soft Thresholding](https://www.stat.cmu.edu/~ryantibs/convexopt-F13/lectures/06-sgs.pdf)

### 4.4 几何性质

![L1 与 L2 约束和损失等高线的示意](https://miro.medium.com/v2/resize:fit:1600/format:webp/1*_e8BLNA749W_7yxi7hz-DA.gif)

二维 L1 球 $\{w:\|w\|_1\le C\}$ 是菱形；三维是八面体；更高维称为 **cross-polytope**。它不是高维立方体，立方体对应的是 $\|w\|_\infty\le C$。L2 球 $\{w:\|w\|_2\le r\}$ 则是通常的欧氏球；若写成 $\|w\|_2^2\le C$，半径为 $\sqrt C$。

L1 边界有许多坐标为零的角和面，损失等高线与这些位置接触时会得到稀疏解。L2 边界平滑，没有同样的坐标轴偏好。几何图能帮助理解次梯度条件，但不能据此断言“L1 的最优解永远位于顶点”。

![正则化约束的几何示意](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*GdOo-X5Mq2CYLzci6reoZw.png)

LASSO 因而可以用于特征选择。不过系数被压到 0 只说明该特征在当前模型、数据缩放和正则强度下没有被选中，不等于证明它在所有场景中都没有用。特别是特征高度相关时，选择结果可能不稳定。

## Reference

1. [Boyd and Vandenberghe, Convex Optimization](https://www.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf)，第 5 章的强对偶、Slater 条件与 KKT
2. [Ryan Tibshirani, Subgradients](https://www.stat.cmu.edu/~ryantibs/convexopt-F13/lectures/06-sgs.pdf)，零点次梯度与 Soft Thresholding
3. [SciPy：Laplace](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.laplace.html) 与 [Normal](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.norm.html)，密度及尺度参数
4. [Loshchilov and Hutter, Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101)，L2 与 Weight Decay 的区别
