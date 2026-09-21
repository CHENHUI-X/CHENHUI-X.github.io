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

在机器学习或深度学习中，无论是分类、回归还是其他场景，通常都是利用模型去拟合一个函数。在这个过程中，正则化是一种常用的手段，用来防止过拟合。本篇博客主要从几个角度探讨正则化的理解，并解释它为何能够防止过拟合。


:::note
阅读前, 需要你 : 有高数基础知识, 线代基础知识, 统计学习基础知识, 当然还要有 ML和 DL 的知识背景.
:::

## 1. 公式

给定输入 $x_1,x_2...x_n$ 和输出 $y_1,y_2...y_n$，我们通过一个模型 $f(w,x)$ 来映射输入输出之间的关系，其中 $w$ 表示模型参数。参数的求解通过优化以下损失函数：


$$
L = \sum_{i} L(x_i,y_i)  + R(w)
$$


这里 $R(w)$ 是关于参数 $w$ 的一个函数.对于 L1 Regularization


$$
R(w) = \lambda \|w\|_1
$$


对于 L2 Regularization


$$
R(w) = \lambda {\|w\|_2}^2
$$


## 2. 理解

> 从式子上看, Regularization 看起来就是想让参数 $w$ 的范数小一点 , 下面来看为什么 $w$ 的范数小一点, 就能减缓过拟合.


首先我们来看过拟合是什么? 定义这里就不说了, 直观看个图吧.

<img src="https://s2.loli.net/2024/04/20/csCq1bnfWRQ7mg4.png" alt="image.png" width="300" height="300" />

上图中,我们有蓝色和红色,2组类别的数据点, 想训练一个分类器f(w,x)去将蓝色点和红色点分开.

可以看到, 绿色的线($f_1$)近乎完美的对数据进行了拟合, 黑色($f_2$)的看起来差一些.

:::note
但是啊, 我是说有没有一种可能, 这个数据集他有异常点(比如加粗的那几个), 如果你拟合的太好, 反而会把噪声也拟合了, 导致你的模型泛化性能不好. 反观黑色的线, 就看起来更加不错.
:::

那么如何才能让模型从绿色变成黑色的线呢? 即怎么把函数的"弯弯绕绕"给他拿走.

我们对函数 $f(x)$ 在某个点进行泰勒展开:


$$
f(w,x) = f(w,a) + f'(w,a)(x - a) + \frac{f''(w,a)}{2!}(x - a)^2 + \cdots
$$


高阶导数可以描述局部曲率，但它们与参数范数之间**没有普遍的一一对应关系**。只有在特定模型和参数化方式下，限制参数大小才可能限制函数变化幅度；不能仅由泰勒展开就推出“权重越小，函数一定越平滑”。

> 正则化是对参数大小施加偏好，以降低模型对训练数据噪声的敏感性；它是否改善泛化需要结合模型、数据和正则化强度验证。

## 3. 等价形式

### 3.1 给权重 $w$ 加约束

> 让 $w$ 小一点等价于让 $w$ 不太大 - 鲁迅

所以优化目标可以变为:


$$
minimize \ L(w,x) , \ s.t. {\|w\|_2}^2 \leq C
$$


使用拉格朗日乘数法, 上述问题变为:


$$
\mathop{minimize}\limits_{w} \  \mathop{maximize}\limits_{\lambda} \ L(w,\lambda,x) = L(w) + \lambda ( {\|w\|_2}^2 -  C)
$$


剩下过程就是,求导等于0, 然后计算相应的 $w$ 和 $\lambda$ 即可. 不过这里想说的是, 在对 $w$ 求导的时候, 你会发现其实并没有 $C$ 的事情 :


$$
\frac{\partial J}{\partial w} = \frac{\partial L}{\partial w} + 2 * \lambda w
$$


于是不妨直接 $minimize$ 下式:


$$
minimize \ L(w,x) + \lambda {\|w\|_2}^2
$$


> 1范数同理, 不再赘述.

### 3.2 让权重 $w$ 衰减


$$
minimize \ J = \ L(w,x) + \lambda {\|w\|_2}^2
$$


梯度下降:


$$
\begin{align*}
w_{t+1} &= w_t-\eta\left(\nabla L(w_t)+2\lambda w_t\right) \\
&=(1-2\lambda\eta)w_t-\eta\nabla L(w_t) \\
\end{align*}
$$


当 $2 \lambda \eta \in (0,1)$ 时，每次更新权重都是在上一次权重衰减后的基础上进行的。


### 3.3 给权重 $w$ 限定分布

从统计学上来看, $f(w,x)$ 输出的是一个分布去拟合 y 的分布 , 使用贝叶斯公式:


$$
p(w\mid \mathcal D)=\frac{p(\mathcal D\mid w)p(w)}{p(\mathcal D)}
$$


其中 $\mathcal D$ 是观测数据；对固定数据优化 $w$ 时，证据 $p(\mathcal D)$ 与 $w$ 无关。


极大似然估计核心公式为:

$$
\hat w_{\mathrm{MLE}}=\operatorname*{argmax}_w p(\mathcal D\mid w)
$$

> 极大似然估计只最大化数据的似然 $p(\mathcal D\mid w)$，不引入参数先验。它与最大化后验概率不是同一个问题。


最大后验估计核心公式为:

$$
\hat w_{\mathrm{MAP}}=\operatorname*{argmax}_w p(\mathcal D\mid w)p(w)
$$

> 最大后验估计在似然之外，还纳入参数的先验分布 $p(w)$。

OK , 基于最大后验估计, 取 log 得到:








$$
\begin{align*}
\hat w_{\mathrm{MAP}}
&=\operatorname*{argmax}_w\left[\log p(\mathcal D\mid w)+\log p(w)\right]
\end{align*}
$$


我们不看前半部分,只看后半部分.

- 假设 $w \sim N(0 , \sigma ^ 2)$


$$
p(w_j) = \frac{1}{\sqrt{2\pi\sigma^2}}
\exp\left(-\frac{w_j^2}{2\sigma^2}\right)
$$


则


$$
-\log p(w)=\frac{\|w\|_2^2}{2\sigma^2}+C
$$


:::note
若各参数 $w_j$ 独立服从零均值、方差 $\sigma^2$ 的高斯先验，MAP 的负对数目标会增加 $\|w\|_2^2/(2\sigma^2)$。因此 L2 惩罚对应高斯先验；其方差由正则化系数及似然目标的缩放共同决定，不必是标准正态分布。
:::


- 假设 $w \sim Laplace(0 , b)$


$$
f(w) = \frac {1} {2b} exp(- \frac{|w|}{b})
$$


则


$$
-\log p(w)=\frac{\|w\|_1}{b}+C
$$


:::note
若各参数 $w_j$ 独立服从零均值、尺度为 $b$ 的拉普拉斯先验，MAP 的负对数目标会增加 $\|w\|_1/b$。注意这里是 L1 范数，**没有平方**。
:::


## 4. 区别


### 4.1 函数性质


我们可以从标准正态分布和拉普拉斯分布的函数性质,来窥探L1 Regularization 和 L2 Regularization 的区别.

![untitled.png](https://s2.loli.net/2024/04/20/nhvpas6JESRMAUf.png)

根据上图可以看到, L1 Regularization (拉普拉斯分布) 在 0 附近形状更尖锐, 将 w 推向0的时候更加强硬. 而  L2 Regularization (标准正态分布) 显得更加柔和.


### 4.2 几何性质

此外也可以从几何性质上对 L1 Regularization 和 L2 Regularization 进行分析.

![image.gif](https://miro.medium.com/v2/resize:fit:1600/format:webp/1*_e8BLNA749W_7yxi7hz-DA.gif)

1范数在几何上表现为一个高维的四方体,2范数则是一个高维的球体. 可以从上图看到,在做minimize时候,L1 Regularization 的 "尖儿" 更容易触到靠内的等高线,即 "尖儿"的位置具有更低的值, 而 "尖儿"的位置,就意味着 w 的某个分量就是0. 而2范数因为整个表面都是外凸出的弧,在哪个地方都有可能取得最小值.

![image.png](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*GdOo-X5Mq2CYLzci6reoZw.png)

这也就是为什么说, L1 Regularization 能够比 L2 Regularization 更加的 "Sparsity".所以 L1 正则项的另外一个应用就是能够进行特征选择: [LASSO回归](https://en.wikipedia.org/wiki/Lasso_(statistics))通过在原始损失函数上添加 L1 Regularization,导致特征 $i$ 对应的权重 $w_i$ 为 0, 我们认为, 权重 $w_i=0$ 的特征就是可以去除的.


## Reference

[1] [Why L1 norm creates Sparsity compared with L2 norm](https://satishkumarmoparthi.medium.com/why-l1-norm-creates-sparsity-compared-with-l2-norm-3c6fa9c607f4)

[2] [Regularization Wiki](https://en.wikipedia.org/wiki/Regularization_(mathematics))
