---
title: GAN Loss Derivation
date: 2024-04-11 23:39:00 +0800
categories: [Deep Learning , Mathematics, GAN]
tags: [deep learning, mathematics]     # TAG names should always be lowercase
math: true
---

## 前言

[GAN原始paper](https://arxiv.org/abs/1406.2661)中的损失很优美 ( 啊~ 你别说你还真别说 ):

$$
\mathcal{L}_{\text{GAN}} = min_{G} \ max_{D} \  \mathbb{E}_{x \sim p_{\text{data}}(x)}[\log D(x)] + \mathbb{E}_{z \sim p_z(z)}[\log(1 - D(G(z)))]
$$

不过有的同学可能看的一头雾水, 我们来推导一下怎么来的. 

## 推导

为方便推导 , 记 `Generator` 为 `G` , `Discriminator` 为 `D`.

### Generator

Generator 要做的事情呢 , 可以划分为以下几步:

[1] 首先, 从一个 noise 分布 sample 一笔数据 , 不妨假设 $z \sim p_z(z)$ 

[2] 然后 Generator 一顿操作, 输出 $G(z)$ 

[3] 目标: 尽可能的欺骗 Discriminator , 让其认为  $G(Z)$  是真的 , 具体表现为 $D(G(Z))$ 越接近 $1$ 越好

因此, 用交叉熵表示 Generator 要优化的目标是:

<div style="text-align:center">
$$
\begin{align*}
L(G) &=  minimize \ \sum 1 * \frac {1} {log(D(G(z)))} + 0 * \frac {1} {log(1 - D(G(z)))} \\
&= minimize \ \sum 1 * \frac {1} {log(D(G(z)))} \\
&= minimize \ - \sum log(D(G(z))) \\
&= minimize \ \sum log(1 - D(G(z))) \\
\end{align*}
$$
</div>

> 这里有个小trick , 当我们 update Generator 时 , Discriminator 是固定的 , 而 $x \sim p_{\text{data}}(x)$ 也是固定的 (就是我们真实样本训练集), 于是 Generator 有以下**等价优化目标**(当然可以加个 Expectation ~ ).


$$
\begin{align*}
\mathcal{L}_{\text{G}} = min_{G}\  \mathbb{E}_{x \sim p_{\text{data}}(x)}[\log D(x)] + \mathbb{E}_{z \sim p_z(z)}[\log(1 - D(G(z)))]
\end{align*}
$$


### Discriminator

Discriminator 要做的事情呢 , 可以划分为以下几步:

[1] 首先, 从一个 真实 分布 sample 一笔数据 , 不妨假设 $x \sim p_x(x)$ 

[2] 然后, 接受来自 Generator 的输出 $G(Z)$

[3] 将 $x$ 和 $G(Z)$ 都扔给 Discriminator

[4] 目标: 尽力分辨出 $x$ 为真, $G(Z)$ 为假.

因此, 用交叉熵表示 Discriminator 要优化的目标是:

<div style="text-align:left">
$$
\begin{align*}
L(D) &=  minimize \ \sum \{ 1 * \frac {1} {log(D(x))} + 0 * \frac {1} {log(1 - D(x))} \}_{x\  for\  true} \ \\
&+ \ \sum  \{0 * \frac {1} {log(D(G(z)))} + 1 * \frac {1} {log(1 - D(G(z)))} \}_{G(z)\  for \ false} \\
&= minimize \ \sum 1 * \frac {1} {log(D(x))} \ + \ \sum 1 * \frac {1} {log(1 - D(G(z)))}  \\
&= minimize \ - \sum log(D(x)) - \sum log(1 - D(G(z))  \\
&= maximize \ \sum log(D(x)) \ + \ \sum log(1 - D(G(z))  \\
\end{align*}
$$
</div>

> 啊, 美化一下, 加个 Expectation~ ,美滋滋

$$
\begin{align*}
\mathcal{L}_{\text{D}} = max_{D}\  \mathbb{E}_{x \sim p_{\text{data}}(x)}[\log D(x)] + \mathbb{E}_{z \sim p_z(z)}[\log(1 - D(G(z)))]
\end{align*}
$$


### 大一统

$Generator$ 要 $minimize$ 下边这个式子, $Discriminator$ 要 $maximize$ 下边这个式子 . 叮~ 任务完成~

$$
\mathcal{L}_{\text{GAN}} = min_{G} \ max_{D} \  \mathbb{E}_{x \sim p_{\text{data}}(x)}[\log D(x)] + \mathbb{E}_{z \sim p_z(z)}[\log(1 - D(G(z)))]
$$

## 算法步骤

贴一个原始paper中的算法步骤, 不过可以看到 , 上边式子那个只是为了美观 , 实际更新的时候, 还是用原始的,

![image.png](https://s2.loli.net/2024/04/12/yS5QvjER17fJ3z4.png)


