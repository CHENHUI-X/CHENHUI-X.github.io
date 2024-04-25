---
title: Kullback–Leibler divergence
date: 2024-04-19 00:04:00 +0800
categories: [Statistical, Mathematics, Machine Learning, Deep Learning]
tags: [machine learning, mathematics, statistics, deep learning]     # TAG names should always be lowercase
math: true
---


## 0. 前言

我们有2个分布 $P$ 和 $Q$, 如何比较二者之间的差异性? 在数理统计上, K-L 散度是一个常用的方法.


## 1. 定义

### 1.1 离散版本

For discrete probability distributions $P$ and $Q$ defined on the same sample space $\mathcal {X}$ . 
 
$$
D_{KL}(P \ ||\  Q)  = \sum_{x \in \mathcal {X}} P(x) \ log(\frac{P(x)} {Q(x)})
$$

which is equivalent to

$$
D_{KL}(P \ ||\  Q)  = - \ \sum_{x \in \mathcal {X}} P(x) \ log(\frac{Q(x)} {P(x)})
$$

### 1.2 连续版本

$$
D_{KL}(P \ ||\  Q)  = \int_{x \in \mathcal {X}} p(x) \ log(\frac{p(x)} {q(x)}) \ dx
$$


## 2. 理解

### 2.1 公式上

有人会把KL散度理解为一种"距离",不过"距离"需要满足以下几个性质

- 非负性 : 满足

> 证明

$$
\begin{align*}
D_{KL}(P \ ||\  Q)  &= - \ \sum_{x \in \mathcal {X}} P(x) \ log(\frac{Q(x)} {P(x)}) \\
&>= - \ \sum_{x \in \mathcal {X}} log(P(x) \  * \ \frac{Q(x)} {P(x)})  \  (凸函数:E(f(x)) >= f(E(x)))\\
&= - \ \sum_{x \in \mathcal {X}} log(Q(x)) \\
&>= - \ log (\sum_{x \in \mathcal {X}}  Q(x) ) \  (凸函数:Jensen不等式)\\ 
&= 0
\end{align*}
$$

- 同一性 : 满足

> 证明

$$
D_{KL}(P \ ||\  P)  =  \ \sum_{x \in \mathcal {X}} P(x) \ log(\frac{P(x)} {P(x)}) = 0
$$

- 对称性 : 不满足

$$
\begin{align*}
D_{KL}(P \ ||\  Q)  &=  \ \sum_{x \in \mathcal {X}} P(x) \ log(\frac{P(x)} {Q(x)}) \\
&\neq\\
D_{KL}(Q \ ||\  P)  &=  \ \sum_{x \in \mathcal {X}} Q(x) \ log(\frac{Q(x)} {P(x)}) \\
\end{align*}
$$

> 一般来说不等, 所以对称性不满足

- 三角不等式 : 不满足

假设有 `P Q R` 三个分布, 探究
$D_{KL}(P \ ||\  R)$ 与 $D_{KL}(P \ ||\  Q)$ 、$D_{KL}(Q \ ||\  R)$ 的关系。


$$
\begin{align*}
D_{KL}(P \ ||\  R)  &=  \ \sum_{x \in \mathcal {X}} P(x) \ log(\frac{P(x)} {R(x)}) \\
&=  \ \sum_{x \in \mathcal {X}} P(x) \ log(\frac{P(x)} {R(x)} * \frac{Q(x)} {Q(x)} ) \\
&=  \ \sum_{x \in \mathcal {X}} P(x) \ log(\frac{P(x)} {Q(x)} * \frac{Q(x)} {R(x)} ) \\
&=  D_{KL}(P \ ||\  Q) + \ \sum_{x \in \mathcal {X}} P(x) \ log(\frac{Q(x)} {R(x)} ) \\
\end{align*}
$$

那就需要看后者 

$$\sum_{x \in \mathcal {X}} P(x) \ log(\frac{Q(x)} {R(x)} )$$ 

与 

$$D_{KL}(Q \ ||\  R)$$ 

之间的大小关系, 但是很遗憾, 二者大小无法判定. 因此有可能出现以下情况, 所以三角不等式不满足.

$$
D_{KL}(P \ ||\  R) > D_{KL}(P \ ||\  Q) + D_{KL}(Q \ ||\  R) 
$$


> 因此称其为“距离”是不合适的， 充其量只能说其可以度量两个分布之间的差异性。
{: .prompt-info }

### 2.2 从熵的角度

"熵"通常指的是[香农熵(Shannon entropy)](https://zh.wikipedia.org/zh-hans/%E7%86%B5_(%E4%BF%A1%E6%81%AF%E8%AE%BA)). 原来是信息论里边的东西, 其公式大家很熟悉:

$$
H(X) = \sum_{x \in \mathcal {X}} P(x) \ log\ \frac{1} {P(x)}
$$

> 如果你不知道这个公式为什么是这样, 而不是那样, 可以看我在 B站 发的视频 : [https://www.bilibili.com/video/BV1kg411u7RP](https://www.bilibili.com/video/BV1kg411u7RP)
{: .prompt-info }

然后来看以下推导:

$$
\begin{align*}
I(X,Y) &= D_{KL}(p(x,y) \ ||\  p(x)\ \times \ p(y))  (定义)\\
&=  \ \sum_{x , y} p_{xy} \ log(\frac{p_{xy}} {p_x \times p_y})  \\ 
& = \ \sum_{x , y} p_{xy} \ log(\frac{p_{xy}} {p_x}) - \ \sum_{x , y} p_{xy} \ log \ p_{y} \\
& = \ \sum_{x , y} p_{xy} \ log \ p(y|x) - \ \sum_{y}  \ (\sum_{x} p_{xy}) \ log \ p_{y} \\
& = \ \sum_{x , y} p_{x}p_{y|x} \ log \ p(y|x) - \ \sum_{y}  \ p_{y} \ log \ p_{y} \\
& = \ \sum_{x} p_{x} \ \sum_{y } p_{y|x} \ log \ p(y|x) - \ \sum_{y}  \ p_{y} \ log \ p_{y} \\
& = - \ \sum_{x } p_{x} \ H(y|X=x) - \ \sum_{y}  \ p_{y} \ log \ p_{y} \\
& = -H(Y|X) +  H(Y) \\
& = H(Y) - H(Y|X)  \\
\end{align*}
$$

其中, $I(X,Y)$ 称为随机变量 X 与 Y 的互信息量,
$H(Y\ |\ X)$ 
表示在已知随机变量X的情况下,Y的熵,也称条件熵.

假设 $P = p(x,y), Q =  p(x) * p(y)$
我们从 $I(X,Y) = D_{KL}(P\ || \ Q) = H(Y) - H(Y|X)$ 
的角度来看, 当
$I(X,Y) = 0$ 
就是想说 
$H(Y) - H(Y|X) = 0$

> 表明已知信息 X , 仍然有 $H(Y\|X) = H(Y)$, 即 X 对 Y 的熵降低无任何作用 <=> X 和 Y 独立 <=> P = Q
{: .prompt-info }

这样看来, $minimize \ KL(P,Q)$ 就是想让二者从`信息量上`尽量的相近

## 3. 应用

$$
\begin{align*}
D_{KL}(P \ ||\  Q)  &= \sum_{x \in \mathcal {X}} P(x) \ log \ \frac{P(x)} {Q(x)} \\
&= \sum_{x \in \mathcal {X}} P(x) \ log \ P(x) - \sum_{x \in \mathcal {X}} P(x) \ log \ Q(x) \\
&=  - H(P) + \sum_{x \in \mathcal {X}} P(x) \ log \ \frac{1}{Q(x)} \\
&=  - H(P) + H(P,Q)
\end{align*}
$$

>这里 H(P,Q) 称为 分布 P 和 分布 Q 的交叉熵, 通常我们在机器学习或者深度学习中, 可以把 P 分布理解为真实的概率分布(未知,但是固定) , 因此 - H(P) 就是个常数 ; Q 为我们模型输出的概率分布, 所以可以通过 
$minimize \ H(P,Q)$ 
去等价 
$minimize \ D_{KL}(P \ ||\  Q) $ 
. 即 交叉熵 损失函数.
{: .prompt-info }


## 4. 补充

KL divergence between two multivariate Gaussian distributions. 

Probabilty density function of multivariate Normal distribution is given by:

$$
p(\mathbf{x}) = \frac{1}{(2\pi)^{k/2}|\Sigma|^{1/2}} \exp\left(-\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^T\Sigma^{-1}(\mathbf{x}-\boldsymbol{\mu})\right)
$$

假设2个分布分别为 $\mathcal{N}(\boldsymbol{\mu_p},\,\Sigma_p) $ 和 $\mathcal{N}(\boldsymbol{\mu_q},\,\Sigma_q)$ , 其中 $\mu$ 为 $k$ 维 列向量.

$$
\begin{aligned}
D_{KL}(p||q) & = \mathbb{E}_p\left[\log(p) - \log(q)\right]
\newline
& = \mathbb{E}_p\left[\frac{1}{2}\log\frac{|\Sigma_q|}{|\Sigma_p|} - \frac{1}{2}(\mathbf{x}-\boldsymbol{\mu_p})^T\Sigma_p^{-1}(\mathbf{x}-\boldsymbol{\mu_p}) + \frac{1}{2}(\mathbf{x}-\boldsymbol{\mu_q})^T\Sigma_q^{-1}(\mathbf{x}-\boldsymbol{\mu_q})\right]
\newline
& = \frac{1}{2}\mathbb{E}_p\left[\log\frac{|\Sigma_q|}{|\Sigma_p|}\right] - \frac{1}{2}\mathbb{E}_p\left[(\mathbf{x}-\boldsymbol{\mu_p})^T\Sigma_p^{-1}(\mathbf{x}-\boldsymbol{\mu_p})\right] + \frac{1}{2}\mathbb{E}_p\left[(\mathbf{x}-\boldsymbol{\mu_q})^T\Sigma_q^{-1}(\mathbf{x}-\boldsymbol{\mu_q})\right]
\newline
& = \frac{1}{2}\log\frac{|\Sigma_q|}{|\Sigma_p|} - \frac{1}{2}\mathbb{E}_p\left[(\mathbf{x}-\boldsymbol{\mu_p})^T\Sigma_p^{-1}(\mathbf{x}-\boldsymbol{\mu_p})\right] + \frac{1}{2}\mathbb{E}_p\left[(\mathbf{x}-\boldsymbol{\mu_q})^T\Sigma_q^{-1}(\mathbf{x}-\boldsymbol{\mu_q})\right] 
\end{aligned}
$$

其中 
$(\mathbf{x}-\boldsymbol{\mu_p})^T\Sigma_p^{-1}(\mathbf{x}-\boldsymbol{\mu_p})$ 
是一个实数. 所以可以重新写为 : 

$$
tr \left\{(\mathbf{x}-\boldsymbol{\mu_p})^T\Sigma_p^{-1}(\mathbf{x}-\boldsymbol{\mu_p})\right\}
$$

其中 tr{} 表示  trace operator , 利用 trace trick (轮换性) , 可以将上式修改为:

$$
tr \left\{(\mathbf{x}-\boldsymbol{\mu_p})(\mathbf{x}-\boldsymbol{\mu_p})^T\Sigma_p^{-1}\right\}
$$

于是第2项可修改为:

$$
\frac{1}{2}\mathbb{E}_p\left[tr\left\{(\mathbf{x}-\boldsymbol{\mu_p})(\mathbf{x}-\boldsymbol{\mu_p})^T\Sigma_p^{-1}\right\}\right]
$$

然后, 将 expectation 和 trace 交换位置, 且 $\Sigma_p^{-1}$ 是常数矩阵:

$$

\begin{aligned}
& = \frac{1}{2}tr\left\{\mathbb{E}_p\left[(\mathbf{x}-\boldsymbol{\mu_p})(\mathbf{x}-\boldsymbol{\mu_p})^T\Sigma_p^{-1}\right]\right\}
\newline
& = \frac{1}{2}tr\left\{\mathbb{E}_p\left[(\mathbf{x}-\boldsymbol{\mu_p})(\mathbf{x}-\boldsymbol{\mu_p})^T\right]\Sigma_p^{-1}\right\}
\newline
& = \frac{1}{2}tr\left\{\Sigma_p\Sigma_p^{-1}\right\}
\newline
& = \frac{1}{2}tr\left\{I_k\right\}
\newline
& = \frac{k}{2}
\end{aligned}

$$

而第3项(证明在最后) :

$$
\mathbb{E}_p\left[(\mathbf{x}-\boldsymbol{\mu_q})^T\Sigma_q^{-1}(\mathbf{x}-\boldsymbol{\mu_q})\right] = (\boldsymbol{\mu_p}-\boldsymbol{\mu_q})^T\Sigma_q^{-1}(\boldsymbol{\mu_p}-\boldsymbol{\mu_q}) + tr\left\{\Sigma_q^{-1}\Sigma_p\right\}
$$

于是:

$$
\mathbb{E}_p\left[(\mathbf{x}-\boldsymbol{\mu_q})^T\Sigma_q^{-1}(\mathbf{x}-\boldsymbol{\mu_q})\right] = (\boldsymbol{\mu_p}-\boldsymbol{\mu_q})^T\Sigma_q^{-1}(\boldsymbol{\mu_p}-\boldsymbol{\mu_q}) + tr\left\{\Sigma_q^{-1}\Sigma_p\right\}
$$

当 $q \sim \mathcal{N}(0,\,I)$ :

$$
D_{KL}(p||q) = \frac{1}{2}\left[\boldsymbol{\mu_p}^T\boldsymbol{\mu_p} + tr\left\{\Sigma_p\right\} - k - \log|\Sigma_p|\right]

$$

---

关于第3项的证明:

$$
\begin{equation}\begin{aligned} 
\mathbb{E}_{\boldsymbol{x}\sim p(\boldsymbol{x})}\left[(\boldsymbol{x}-\boldsymbol{\mu}_q)^{\top}\boldsymbol{\Sigma}_q^{-1}(\boldsymbol{x}-\boldsymbol{\mu}_q)\right]=&\,\mathbb{E}_{\boldsymbol{x}\sim p(\boldsymbol{x})}\left[\text{Tr}\left((\boldsymbol{x}-\boldsymbol{\mu}_q)^{\top}\boldsymbol{\Sigma}_q^{-1}(\boldsymbol{x}-\boldsymbol{\mu}_q)\right)\right]\\ 
=&\,\mathbb{E}_{\boldsymbol{x}\sim p(\boldsymbol{x})}\left[\text{Tr}\left(\boldsymbol{\Sigma}_q^{-1}(\boldsymbol{x}-\boldsymbol{\mu}_q)(\boldsymbol{x}-\boldsymbol{\mu}_q)^{\top}\right)\right]\\ 
=&\,\text{Tr}\left(\boldsymbol{\Sigma}_q^{-1}\mathbb{E}_{\boldsymbol{x}\sim p(\boldsymbol{x})}\left[(\boldsymbol{x}-\boldsymbol{\mu}_q)(\boldsymbol{x}-\boldsymbol{\mu}_q)^{\top}\right]\right)\\ 
=&\,\text{Tr}\left(\boldsymbol{\Sigma}_q^{-1}\mathbb{E}_{\boldsymbol{x}\sim p(\boldsymbol{x})}\left[\boldsymbol{x}\boldsymbol{x}^{\top}-\boldsymbol{\mu}_q\boldsymbol{x}^{\top} - \boldsymbol{x}\boldsymbol{\mu}_q^{\top} +  \boldsymbol{\mu}_q\boldsymbol{\mu}_q^{\top}\right]\right)\\ 
=&\,\text{Tr}\left(\boldsymbol{\Sigma}_q^{-1}\left(\boldsymbol{\Sigma}_p + \boldsymbol{\mu}_p\boldsymbol{\mu}_p^{\top}-\boldsymbol{\mu}_q\boldsymbol{\mu}_p^{\top} - \boldsymbol{\mu}_p\boldsymbol{\mu}_q^{\top} +  \boldsymbol{\mu}_q\boldsymbol{\mu}_q^{\top}\right)\right)\\ 
=&\,\text{Tr}\left(\boldsymbol{\Sigma}_q^{-1}\boldsymbol{\Sigma}_p + \boldsymbol{\Sigma}_q^{-1}(\boldsymbol{\mu}_p-\boldsymbol{\mu}_q)(\boldsymbol{\mu}_p-\boldsymbol{\mu}_q)^{\top}\right)\\ 
=&\,\text{Tr}\left(\boldsymbol{\Sigma}_q^{-1}\boldsymbol{\Sigma}_p\right) + (\boldsymbol{\mu}_p-\boldsymbol{\mu}_q)^{\top}\boldsymbol{\Sigma}_q^{-1}(\boldsymbol{\mu}_p-\boldsymbol{\mu}_q)\\ 
\end{aligned}\end{equation}
$$

注意到当 $\boldsymbol{\mu}_q=\boldsymbol{\mu}_p,\boldsymbol{\Sigma}_q=\boldsymbol{\Sigma}_p$, 上式就是 $n$ , 对应正态分布的熵.


## Reference

[1] [https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence)

[2] [https://zh.wikipedia.org/zh-hans/%E4%BA%92%E4%BF%A1%E6%81%AF](https://zh.wikipedia.org/zh-hans/%E4%BA%92%E4%BF%A1%E6%81%AF)

[3] [https://en.wikipedia.org/wiki/Entropy_(information_theory)](https://en.wikipedia.org/wiki/Entropy_(information_theory))


[4] [https://mr-easy.github.io/2020-04-16-kl-divergence-between-2-gaussian-distributions/](https://mr-easy.github.io/2020-04-16-kl-divergence-between-2-gaussian-distributions/)

[5] [https://kexue.fm/archives/8512](https://kexue.fm/archives/8512)



