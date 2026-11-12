---
title: L1 和 L2 正则化有什么区别
published: 2024-04-20
description: 看 L1 和 L2 怎样惩罚参数、改变梯度更新，再用软阈值、次梯度和几何图解释 L1 为什么容易把系数变成零。
category: Machine Learning
tags:
- machine learning
- mathematics
- statistics
draft: false
---
<span id="0-前言"></span>

<span id="0-正则化如何改变训练目标"></span>

## 0. 为什么要给参数加惩罚

模型可能把训练集拟合得很好，却把噪声也当成了规律。L1 和 L2 正则化会在训练目标中加入参数惩罚，减少模型为了拟合训练数据而使用过大参数的倾向。下面先看它们怎样改变参数更新，再解释正则化为什么可以写成约束或贝叶斯先验，以及 L1 为什么更容易得到零系数。

:::note
阅读需要导数、向量和基本概率知识；第 3.1 节的严格条件还需要凸优化，第 3.3 节需要似然与贝叶斯公式。第一次读可按“第 1 节 → 第 3.2 节 → 第 4.2–4.3 节”的顺序，先看惩罚怎样改变一个参数，再回看约束与先验。正则化强度仍要通过验证集等方式选择；加入正则项并不保证所有模型的泛化性能都会提高。
:::

<span id="1-公式与系数约定"></span>

## 1. L1 和 L2 的惩罚项怎么写

设模型参数 $w\in\mathbb R^d$，数据损失为 $L(w)$。例如：

$$
L(w)=\frac1n\sum_{i=1}^n\ell(f(w,x_i),y_i).
$$

$n$ 是训练样本数，$(x_i,y_i)$ 是第 $i$ 个输入及标签，$f(w,x_i)$ 是预测，$\ell$ 计算这条预测的损失，$L$ 是所有样本的平均损失。$d$ 是参数个数，下标 $i$ 表示样本编号，下标 $j$ 表示参数编号。例如线性回归可取 $f(w,x_i)=w^\top x_i$、$\ell=\frac12(f-y_i)^2$。

本文使用如下约定，且始终令 $\lambda\ge0$：

$$
J_{\mathrm{L1}}(w)=L(w)+\lambda\|w\|_1,
\qquad
J_{\mathrm{L2}}(w)=L(w)+\lambda\|w\|_2^2.
$$

其中 $\|w\|_1=\sum_j|w_j|$，$\|w\|_2^2=\sum_jw_j^2$。L2 正则项使用的是**范数的平方**。也有资料把它写成 $(\lambda_2/2)\|w\|_2^2$，这样梯度是 $\lambda_2w$；与本文等价时需要 $\lambda_2=2\lambda$。

损失取总和还是平均值也会改变正则项的相对强度，不能脱离损失定义直接比较两个实现中的 `lambda`。

例如两个候选模型的数据损失分别为 $0.10$ 和 $0.15$，平方范数分别为 $10$ 和 $1$。当 $\lambda=0.01$ 时，它们的 L2 总目标为 $0.20$ 和 $0.16$，于是训练更偏好第二个：它在训练集上的损失稍大，但参数小了很多。这个选择能否让模型在新数据上表现更好，还要用验证集检查。

参数大小依赖输入单位。例如把“米”改成“厘米”，保持同一预测所需的线性系数会缩小到原来的 $1/100$，受到的惩罚也变了。因此常先统一特征尺度；标准化所需的均值和方差只能从训练集估计。截距是否惩罚也应单独约定，以下 $w$ 只包括选定要惩罚的参数。

<span id="2-为什么限制参数可能有帮助"></span>

## 2. 参数大小与模型对输入的敏感度


训练数据含有噪声时，过于曲折的分类边界可能同时记住规律和偶然误差。正则化可以让优化更偏向某些简单解，但“参数小”与“函数平滑”不是普遍等价的。

<span id="21-taylor-展开能说明什么"></span>

<span id="21-taylor-展开的条件与局限"></span>

### 2.1 参数小，函数就一定更平滑吗

先固定参数 $w$，只考虑单变量输入 $x$。若 $f(w,\cdot)$ 在 $a$ 附近三次连续可微，则有有限阶展开：

$$
f(w,x)=f(w,a)+\partial_x f(w,a)(x-a)
+\frac{\partial_x^2f(w,a)}2(x-a)^2+R_3(x).
$$

如果连接 $a$ 与 $x$ 的区间上满足 $|\partial_x^3f(w,\xi)|\le M$，则：

$$
|R_3(x)|\le\frac M6|x-a|^3.
$$

这个展开式用一阶、二阶导数描述函数在附近的变化；如果还能限制三阶导数的大小，就能给出近似误差的上界。但要用它解释正则化，还需要知道这些导数怎样受权重影响。另外，任意函数未必等于自己的无穷 Taylor 级数。ReLU 在折点处不满足这里的可微条件，不能直接套用。

在线性模型 $f(w,x)=w^\top x$ 中，Cauchy–Schwarz 不等式给出：

$$
|f(w,x)-f(w,x')|\le\|w\|_2\|x-x'\|_2.
$$

在这个例子中，限制 $\|w\|_2$ 确实限制了模型对输入扰动的敏感度。但多层网络、带归一化的模型或不同的参数化方式，需要另外分析；不能仅靠一幅曲线或 Taylor 展开就得出泛化结论。

## 3. 正则化的三种理解

<span id="31-给权重加约束什么时候与惩罚形式对应"></span>

<span id="31-参数约束与惩罚项的对应条件"></span>

### 3.1 限制参数大小，和给参数加惩罚一样吗

记 $R(w)=\|w\|_1$ 或 $R(w)=\|w\|_2^2$。约束问题是：

$$
\min_w L(w)\quad\text{subject to}\quad R(w)\le C,\qquad C\ge0.
$$

约束形式用 $C$ 限定 $R(w)$ 的上界，要求在满足限制的参数中寻找数据损失最小的解。惩罚形式则用 $\lambda$ 调节参数惩罚与数据损失的相对权重。前者直接划定允许范围，后者在损失中加入惩罚。什么条件下，它们会选出同一个解？

对应的拉格朗日函数（Lagrangian）为：

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

互补松弛可以这样理解：如果 $R(w^*)<C$，约束还没达到上界，那么只能有 $\lambda^*=0$；如果 $\lambda^*>0$，就必须有 $R(w^*)=C$。驻点条件则表示数据损失的下降方向与惩罚的作用相互抵消。

这里 $\partial R$ 是次梯度集合，允许处理 L1 在零点不可导的情况。满足上面的凸性和约束条件时，可以用 KKT 条件判断全局最优解。换成一般的非凸问题，即使满足 KKT 条件，也未必是全局最优。[Boyd and Vandenberghe，Convex Optimization，第 5 章](https://www.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf)

对应关系有两个容易忽略的细节：

- 若已经求出惩罚问题的一个**全局**最优解 $w_\lambda$，取 $C=R(w_\lambda)$，则它也是这个约束问题的全局最优解；这个方向不需要凸性
- 反方向需要条件，且通常不能声称 $C$ 与 $\lambda$ 一一对应，或两个问题的整个解集完全相同。约束不活跃时，乘子可能为 0

一个边界反例是：最小化 $(w-1)^2$，约束 $w^2\le0$。唯一可行解是 0；而惩罚目标 $(w-1)^2+\lambda w^2$ 对任何有限 $\lambda\ge0$ 的最优解都是 $1/(1+\lambda)$，不等于 0。这里严格可行点不存在。

<span id="32-让权重衰减先限定优化器"></span>

### 3.2 普通 SGD 中的 L2 与权重衰减

这里 $t$ 表示第几次参数更新，$w_t$ 是本次更新前的参数，$\eta_t>0$ 是本次学习率；$\nabla L(w_t)$ 是各参数偏导数组成的梯度。普通梯度下降从旧参数中减去“学习率 × 梯度”，下面只是把正则项的梯度也加进去。

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

<span id="33-参数先验mle-与-map-的区别"></span>

### 3.3 从贝叶斯先验看 L1 和 L2

对于固定数据 $\mathcal D$，贝叶斯公式为：

$$
p(w\mid\mathcal D)=\frac{p(\mathcal D\mid w)p(w)}{p(\mathcal D)}.
$$

MLE 只最大化似然 $p(\mathcal D\mid w)$；MAP 则最大化后验密度，等价于：

$$
\hat w_{\mathrm{MAP}}=
\operatorname*{argmin}_w\left[-\log p(\mathcal D\mid w)-\log p(w)\right].
$$

似然是“假如参数取 $w$，观察到这批数据有多可能”；先验是看到数据之前对参数的偏好；后验把两者结合。$\operatorname{argmin}$ 返回使目标最小的参数，而不是最小的目标数值。取负对数后，乘积变成加法、最大化变成最小化。分母 $p(\mathcal D)$ 也叫证据，它不随 $w$ 变化，所以优化时可以去掉。

若样本在给定 $w$ 后独立，$p(\mathcal D\mid w)=\prod_i p(y_i\mid x_i,w)$，因此负对数似然就是逐样本负对数概率之和。所以，以负对数似然作为数据损失时，MAP 目标就可以写成前面的“数据损失加正则项”。

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

<span id="41-密度形状先把尺度写清楚"></span>

<span id="41-laplace-与-gaussian-的密度形状"></span>

### 4.1 先比较 Laplace 和 Gaussian 的形状

Laplace 在零点有尖角，Gaussian 在零点平滑；但“哪个峰更高”还取决于尺度。下面固定两种分布的尺度，比较它们在几个位置的密度。

| 分布 | 密度 $p(w)$ | $p(0)$ | $p(1.5)$ | $p(3)$ |
| --- | --- | --- | --- | --- |
| $\operatorname{Laplace}(0,b=1.5)$ | $\frac13e^{-\lvert w\rvert/1.5}$ | $1/3\approx0.333333$ | $e^{-1}/3\approx0.122626$ | $e^{-2}/3\approx0.045112$ |
| $\mathcal N(0,\sigma^2=2.25)$ | $\frac1{1.5\sqrt{2\pi}}e^{-w^2/4.5}$ | $0.265962$ | $0.161314$ | $0.035994$ |

Gaussian 这一行的标准差是 1.5；作为对照，标准正态（Standard Normal）是 $\mathcal N(0,1)$，标准差为 1。Laplace 的 $b=1.5$ 则给出峰值 $1/(2b)=1/3$。两行尺度参数数值相同，不代表方差相同。密度定义可对照 [SciPy Laplace 文档](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.laplace.html) 与 [SciPy Normal 文档](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.norm.html)。

密度曲线只提供直觉。零系数的直接原因，要看惩罚项的次梯度。

### 4.2 L1 在零点的次梯度

绝对值函数在零点左右的斜率分别为 −1 和 1，没有唯一普通导数。对于凸函数，次梯度允许用一组“从下方支撑函数图像的直线斜率”代替单一导数；在这个尖角处，这组斜率恰好是区间 $[-1,1]$。因此：

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

<span id="43-可以手算的-soft-thresholding"></span>

### 4.3 平方损失下的软阈值解

考虑最简单的平方损失：

$$
\min_w\frac12\|w-z\|_2^2+\lambda\|w\|_1.
$$

这里 $z$ 是一个固定的目标向量，不是新的模型参数；若没有惩罚，最优解就是 $w=z$。损失与惩罚都按坐标求和，因而每个坐标可分别最小化 $\frac12(w_j-z_j)^2+\lambda|w_j|$：

- 若 $w_j>0$，导数为 $w_j-z_j+\lambda$，置零得 $w_j=z_j-\lambda$，它只有在 $z_j>\lambda$ 时才符合正号假设。
- 若 $w_j<0$，导数为 $w_j-z_j-\lambda$，置零得 $w_j=z_j+\lambda$，要求 $z_j<-\lambda$。
- 剩下 $|z_j|\le\lambda$ 时，零点左右的斜率允许在这里达到最小值，取 $w_j=0$。

合并这三种情况，得到软阈值（Soft Thresholding）解：

$$
w_j^*=\operatorname{sign}(z_j)\max(|z_j|-\lambda,0).
$$

取 $z=(3,-0.4,0.2)$、$\lambda=0.5$，得到 $w^*=(2.5,0,0)$。而在同一个平方损失上加入本文定义的 L2 项，解为：

$$
w^*=\frac{z}{1+2\lambda}=(1.5,-0.2,0.1).
$$

L2 的结果也可直接验算：每个坐标的导数为 $w_j-z_j+2\lambda w_j$，置零后得到 $(1+2\lambda)w_j=z_j$。L1 的“减去固定门槛”与 L2 的“按比例缩小”，在这个简单问题里便能看得很清楚。

LASSO 是在线性回归的平方损失上加入 L1 惩罚的模型。这不是说一般 LASSO 都能对原始观测值直接做一次阈值操作；这里的闭式解依赖于平方损失可按坐标分离。一般设计矩阵下需要另外求解。[Ryan Tibshirani：Subgradients and Soft Thresholding](https://www.stat.cmu.edu/~ryantibs/convexopt-F13/lectures/06-sgs.pdf)

<span id="44-几何性质"></span>

<span id="44-l1-与-l2-约束的几何形状"></span>

### 4.4 从几何图看 L1 为什么容易得到零系数

![L1、L2 与 elastic-net 惩罚等于 1 的轮廓](/images/llm-series/sourced/sklearn-penalties.png)

先看深蓝菱形落在坐标轴上的四个尖角：每个尖角都有一个参数为零。青色圆在这些位置仍然平滑；这个形状差异帮助理解 L1 的稀疏性，但是否真的取到零仍要结合数据损失判断。

图源：[scikit-learn 官方 SGD: Penalties 示例](https://scikit-learn.org/stable/auto_examples/linear_model/plot_sgd_penalties.html)，The scikit-learn developers，[BSD-3-Clause 许可](/images/llm-series/sourced/sklearn-LICENSE.txt)，原图未修改。横、纵坐标分别表示两个参数。深蓝色 L1 轮廓是菱形，青色 L2 轮廓是圆；橙色 elastic-net 轮廓展示两种惩罚混合后的形状。三条线表示各自惩罚值为 1 的边界，内部是惩罚不超过 1 的区域。图中没有画数据损失的等高线，也没有标出某个训练问题的最优解。

二维 L1 球 $\{w:\|w\|_1\le C\}$ 是菱形；三维是八面体；更高维称为 **cross-polytope**。它不是高维立方体，立方体对应的是 $\|w\|_\infty\le C$。L2 球 $\{w:\|w\|_2\le r\}$ 则是通常的欧氏球；若写成 $\|w\|_2^2\le C$，半径为 $\sqrt C$。

L1 边界有许多坐标为零的角和面，损失等高线与这些位置接触时会得到稀疏解。L2 边界平滑，没有同样的坐标轴偏好。几何图能帮助理解次梯度条件，但不能据此断言“L1 的最优解永远位于顶点”。


由于 LASSO 可以把部分系数压到 0，我们也可以用它筛选特征。不过系数被压到 0 只说明该特征在当前模型、数据缩放和正则强度下没有被选中，不等于证明它在所有场景中都没有用。特别是特征高度相关时，选择结果可能不稳定。

<span id="reference"></span>

![损失椭圆分别与 L2 圆形、L1 菱形约束相接](/images/llm-series/sourced/lasso-ridge-astroml-original.png)

再把损失函数的等高线加进来，才能看到“为什么会等于零”。图中左边是 ridge（L2），右边是 lasso（L1）；横纵轴分别是两个系数。椭圆围绕未加约束的最小二乘解，向中心收缩代表损失更小。寻找仍与约束区域接触的最小椭圆：左图接触圆弧，两个系数通常都非零；右图接触竖轴上的菱形尖角，所以横轴对应的系数恰好为零。这是一个二维示意，不能推出每个 L1 问题都一定得到零系数，也不能仅靠尖角判断有多少个零。

原图未改动：[astroML，Ridge and Lasso: Geometric Interpretation](https://www.astroml.org/book_figures/chapter8/fig_lasso_ridge.html)，Jake VanderPlas，教材图 8.3；作者示例标注 [BSD 许可](/images/llm-series/sourced/astroml-LICENSE.txt)。[原图链接](https://www.astroml.org/_images/fig_lasso_ridge_1.png)。

## 参考资料

1. [Boyd and Vandenberghe, Convex Optimization](https://www.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf)，第 5 章的强对偶、Slater 条件与 KKT
2. [Ryan Tibshirani, Subgradients](https://www.stat.cmu.edu/~ryantibs/convexopt-F13/lectures/06-sgs.pdf)，零点次梯度与 Soft Thresholding
3. [SciPy：Laplace](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.laplace.html) 与 [Normal](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.norm.html)，密度及尺度参数
4. [Loshchilov and Hutter, Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101)，L2 与 Weight Decay 的区别
