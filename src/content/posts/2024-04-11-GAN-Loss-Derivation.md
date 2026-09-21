---
title: GAN Loss Derivation
published: 2024-04-11
description: 从 Generator 与 Discriminator 的目标出发，逐步推导 GAN 损失函数，并说明 min-max 形式背后的直觉。
category: Deep Learning
tags:
- deep learning
- mathematics
- gan
draft: false
---
## 0. 前言

[GAN  原始  paper](https://arxiv.org/abs/1406.2661)  中的损失很优美:


$$
\min_G\max_D\left[
\mathbb{E}_{x\sim p_{\mathrm{data}}}\log D(x)
+\mathbb{E}_{z\sim p_z}\log(1-D(G(z)))\right]
$$


不过有的同学可能看的一头雾水,  我们来推导一下怎么来的.

## 1. 推导

为方便推导 ,  记 `Generator` 为 `G` ,  `Discriminator` 为 `D`.

### 1.1 Generator

Generator 要做的事情呢 ,  可以划分为以下几步:

[1] 首先,  从一个 noise 分布 sample 一笔数据 ,  不妨假设 $z \sim p_z(z)$

[2] 然后 Generator 一顿操作,  输出 $G(z)$

[3] 目标: 尽可能的欺骗 Discriminator ,  让其认为  $G(Z)$  是真的 ,  具体表现为 $D(G(Z))$ 越接近 $1$ 越好

将生成样本的目标标签设为 1，常用的非饱和生成器损失是二元交叉熵：


$$
\mathcal{L}_{G}^{\mathrm{non\text{-}sat}}
=-\mathbb{E}_{z\sim p_z}\log D(G(z)).
$$


原始 GAN 的极小极大目标让生成器最小化 $\mathbb{E}_{z\sim p_z}\log(1-D(G(z)))$。更新 $G$ 时，真实样本项 $\mathbb{E}_{x\sim p_{\mathrm{data}}}\log D(x)$ 与 $G$ 无关，可以去掉。因此，原始目标为：


$$
\mathcal{L}_{G}^{\mathrm{minimax}}
=\mathbb{E}_{z\sim p_z}\log(1-D(G(z))).
$$

两个损失都鼓励 $D(G(z))$ 增大，但**不是等价函数**，梯度也不同。当判别器很容易识别生成样本时，原始极小极大目标可能出现梯度饱和，因此实践中常使用非饱和损失。


### 1.2 Discriminator

Discriminator 要做的事情呢 ,  可以划分为以下几步:

[1] 首先,  从一个 真实 分布 sample 一笔数据 ,  不妨假设 $x \sim p_x(x)$

[2] 然后,  接受来自 Generator 的输出 $G(Z)$

[3] 将 $x$ 和 $G(Z)$ 都扔给 Discriminator

[4] 目标: 尽力分辨出 $x$ 为真,  $G(Z)$ 为假.

令真实样本标签为 1、生成样本标签为 0，判别器要最小化二元交叉熵：

$$
\mathcal{L}_D
=-\mathbb{E}_{x\sim p_{\mathrm{data}}}\log D(x)
-\mathbb{E}_{z\sim p_z}\log(1-D(G(z))).
$$


等价地，判别器最大化 $V(D,G)=-\mathcal{L}_D$：


$$
V(D,G)=\mathbb{E}_{x\sim p_{\mathrm{data}}}\log D(x)
+\mathbb{E}_{z\sim p_z}\log(1-D(G(z))).
$$


### 1.3 大一统

采用原始极小极大目标时，生成器最小化、判别器最大化同一个 $V(D,G)$：


$$
\min_G\max_D\ V(D,G)
=\min_G\max_D\left[
\mathbb{E}_{x\sim p_{\mathrm{data}}}\log D(x)
+\mathbb{E}_{z\sim p_z}\log(1-D(G(z)))\right].
$$

若采用非饱和生成器损失，判别器目标不变，但双方不再优化同一个标量目标的极小极大形式。


## 2. 算法步骤

下面是原始论文中的算法步骤，其中生成器按极小极大目标更新。论文正文还提出：实际训练时可改为最大化 $\log D(G(z))$，缓解训练初期的梯度饱和。

![image.png](https://s2.loli.net/2024/04/12/yS5QvjER17fJ3z4.png)


## Reference
