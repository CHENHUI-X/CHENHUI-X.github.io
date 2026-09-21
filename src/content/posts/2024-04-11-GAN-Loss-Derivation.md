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

GAN 有两个模型：生成器 $G$ 根据随机输入 $z$ 生成样本，判别器 $D$ 判断样本像不像真实数据。$D(x)$ 越接近 1，表示判别器越倾向于认为 $x$ 是真实样本。[原始论文](https://arxiv.org/abs/1406.2661)把训练目标写成：


$$
\min_G\max_D\left[
\mathbb{E}_{x\sim p_{\mathrm{data}}}\log D(x)
+\mathbb{E}_{z\sim p_z}\log(1-D(G(z)))\right]
$$


这个式子看起来像一个目标，实际训练时却要分别更新 $D$ 和 $G$。下面拆开来看。

## 1. 推导

### 1.1 Generator

生成器想让判别器把生成样本当成真的，也就是让 $D(G(z))$ 变大。更新 $G$ 时先固定 $D$；原始总目标中的真实样本项与 $G$ 无关，因此生成器只需最小化：
$$
\mathcal{L}_{G}^{\mathrm{minimax}}
=\mathbb{E}_{z\sim p_z}\log(1-D(G(z))).
$$

训练刚开始，生成样本通常很假，$D(G(z))$ 可能接近 0。此时判别器已经很确定“这是假的”，原始损失给生成器的改进信号可能很弱。论文还提出另一种训练写法：直接要求生成器提高判别器给生成样本的“真实”分数，即最小化

$$
\mathcal{L}_{G}^{\mathrm{alternative}}
=-\mathbb{E}_{z\sim p_z}\log D(G(z)).
$$

这个替代损失在论文中称为 **non-saturating loss（非饱和损失）**。两种写法都想提高 $D(G(z))$，但它们**不是同一个函数，也不是相同的梯度**；替代写法通常能在训练初期提供更强的改进信号，并不保证所有情况下都不会出现梯度问题。


### 1.2 Discriminator

判别器要把真实样本 $x$ 判为真、生成样本 $G(z)$ 判为假。更新 $D$ 时固定 $G$。

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


### 1.3 合起来看

采用原始极小极大目标时，生成器最小化、判别器最大化同一个 $V(D,G)$：


$$
\min_G\max_D\ V(D,G)
=\min_G\max_D\left[
\mathbb{E}_{x\sim p_{\mathrm{data}}}\log D(x)
+\mathbb{E}_{z\sim p_z}\log(1-D(G(z)))\right].
$$

若生成器采用上面的替代损失，判别器的目标不变，但不能再说双方都在优化同一个极小极大目标。


## 2. 算法步骤

下面是原始论文的算法步骤，图中生成器按原始目标更新；论文正文另外提出了前述替代损失。

![image.png](https://s2.loli.net/2024/04/12/yS5QvjER17fJ3z4.png)


## Reference
