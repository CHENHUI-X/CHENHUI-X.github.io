---
title: AdamW：Weight Decay 与 L2 的区别
published: 2024-04-20
description: 探讨 SGD 与 Adam 优化器下 Weight Decay 和 L2 正则化的等价性差异，引入 AdamW 优化器的设计动机与原理。
category: Deep Learning
tags:
- machine learning
- mathematics
- statistics
- deep learning
draft: false
---
## 0. 前言

在 [L1 and L2 Regularization](/posts/2024-04-20-l1-and-l2-regularization/) 中，我们把普通梯度下降中的 L2 梯度拆成了一个权重缩小项。既然如此，为什么还需要 AdamW？

关键在于**正则项进入优化器的位置**。L2 Regularization 修改损失的梯度；decoupled Weight Decay 直接缩小参数，不把这部分缩小量写进 momentum 和自适应二阶矩。这两种做法对普通 SGD 可以等价，对 momentum 或 Adam 一般不同。[AdamW 原论文](https://arxiv.org/abs/1711.05101)

## 1. 先统一符号与衰减定义

令 $\theta_{t-1}\in\mathbb R^d$ 是第 $t$ 次更新前的参数，$f_t$ 是当前 batch 的数据损失：

$$
g_t=\nabla f_t(\theta_{t-1}),\qquad t=1,2,\ldots
$$

学习率为 $\eta_t>0$。本文区分两个系数：

- L2 系数 $\lambda_2\ge0$：损失增加 $(\lambda_2/2)\|\theta\|_2^2$，梯度增加 $\lambda_2\theta$
- 解耦衰减系数 $\lambda_d\ge0$：每步对旧参数乘 $1-\eta_t\lambda_d$

上一篇的 L2 项写作 $\lambda\|w\|_2^2$，所以对应这里的 $\lambda_2=2\lambda$。另一些论文直接写每步因子 $1-d_t$，那就令 $d_t=\eta_t\lambda_d$。**超参数的名字相同，不代表其定义相同。**

本文采用与 PyTorch AdamW 文档一致的学习率缩放衰减。通常令 $0\le\eta_t\lambda_d<1$，以免衰减步骤将参数翻转符号。[PyTorch AdamW 算法定义](https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html)

## 2. SGD：不带 momentum 时可以等价

### 2.1 将 L2 放进损失

$$
\begin{aligned}
\theta_t
&=\theta_{t-1}-\eta_t\left(g_t+\lambda_2\theta_{t-1}\right)\\
&=(1-\eta_t\lambda_2)\theta_{t-1}-\eta_tg_t.
\end{aligned}
$$

### 2.2 使用解耦 Weight Decay

$$
\theta_t=(1-\eta_t\lambda_d)\theta_{t-1}-\eta_tg_t.
$$

在相同初始参数、batch 顺序和学习率下，取 $\lambda_d=\lambda_2$，两条更新轨迹相同。这里没有 momentum、没有自适应预条件，也没有在其中一条路径上额外修改正则梯度的操作，例如只对加入 L2 后的总梯度做 clipping。

若改用“每步固定缩小 $d$”的定义，匹配条件是 $d=\eta_t\lambda_2$。学习率变化时，固定 $d$ 通常不能再对应固定的 $\lambda_2$。这里并没有要求学习率固定，只是不同约定下的系数需要随 schedule 换算。

## 3. 加入 momentum：看缓存里存了什么

使用不带 dampening、不开 Nesterov 的一个常见 momentum 约定，$0\le\beta<1$，初始缓存为 0。

**L2 进入梯度及缓存：**

$$
\begin{aligned}
m_t^{\mathrm{L2}}
&=\beta m_{t-1}^{\mathrm{L2}}+g_t^{\mathrm{L2}}
+\lambda_2\theta_{t-1}^{\mathrm{L2}},\\
\theta_t^{\mathrm{L2}}
&=\theta_{t-1}^{\mathrm{L2}}-\eta_t m_t^{\mathrm{L2}}.
\end{aligned}
$$

**解耦衰减只修改参数：**

$$
\begin{aligned}
m_t^{\mathrm{WD}}&=\beta m_{t-1}^{\mathrm{WD}}+g_t^{\mathrm{WD}},\\
\theta_t^{\mathrm{WD}}
&=(1-\eta_t\lambda_d)\theta_{t-1}^{\mathrm{WD}}
-\eta_t m_t^{\mathrm{WD}}.
\end{aligned}
$$

这里 $g_t^{\mathrm{L2}}=\nabla f_t(\theta_{t-1}^{\mathrm{L2}})$，$g_t^{\mathrm{WD}}=\nabla f_t(\theta_{t-1}^{\mathrm{WD}})$。轨迹一旦分开，数据梯度也可能不同。

### 3.1 两步就能看出差别

设只有一个参数，初值 $\theta_0=1$，数据损失恒定，因此每步数据梯度为 0。取 $\eta=0.1$、$\beta=0.9$、$\lambda_2=\lambda_d=0.1$：

| 更新 | L2 + momentum | 解耦 Weight Decay + momentum |
| --- | --- | --- |
| 第 1 步缓存 | $m_1=0.1$ | $m_1=0$ |
| 第 1 步参数 | $\theta_1=0.99$ | $\theta_1=0.99$ |
| 第 2 步缓存 | $m_2=0.9(0.1)+0.1(0.99)=0.189$ | $m_2=0$ |
| 第 2 步参数 | $\theta_2=0.99-0.1(0.189)=0.9711$ | $\theta_2=0.99^2=0.9801$ |

两个系数已经按第一步的结果匹配，第二步却仍然不同，因为 L2 的旧梯度还留在缓存中。换一种含 $1-\beta$ 的 momentum 定义会改变系数换算，但不会消除缓存历史的区别。

## 4. 自适应缩放：同一份 L2 梯度会按坐标改变

先暂时不考虑 momentum，用一个**给定**的矩阵 $M_t\in\mathbb R^{d\times d}$ 表示梯度缩放。比较：

$$
\begin{aligned}
\theta_t^{\mathrm{L2}}
&=\theta_{t-1}-\eta_t M_tg_t
-\eta_t\lambda_2M_t\theta_{t-1},\\
\theta_t^{\mathrm{WD}}
&=\theta_{t-1}-\eta_t M_tg_t
-\eta_t\lambda_d\theta_{t-1}.
\end{aligned}
$$

如果要求对**所有**参数向量都相等，必须有：

$$
\lambda_d I_d=\lambda_2M_t.
$$

这是矩阵等式，不能把左边直接写成一个标量再与矩阵比较。对于非零正则强度，如果 $M_t$ 各坐标缩放不同，就不存在一个标量 $\lambda_2$ 同时满足所有坐标；即使每步 $M_t=c_tI_d$，当 $c_t$ 随时间变化时，也通常无法用两个固定系数匹配整条轨迹。

例如 $M_t=\operatorname{diag}(1,10)$、$\theta=(1,1)$、$\eta_t=0.1$、$\lambda_2=\lambda_d=0.1$，仅看正则部分，L2 产生的减量是 $(0.01,0.1)$，解耦衰减产生的减量是 $(0.01,0.01)$。

对某个特殊向量、某个时间点，两者仍可能偶然相同；零衰减时也没有这一区别。因此准确结论是“通常不存在统一的标量系数使两种算法等价”，而不是任何情况下都必然不等。[AdamW 原论文，Appendix A](https://arxiv.org/html/1711.05101)

![AdamW 论文关于自适应缩放的推导摘图](https://s2.loli.net/2024/04/21/afDMybYdESVpQoB.png)

这个矩阵例子解释了坐标缩放的影响，但它不是完整 Adam 的更新式。在 Adam 中，加入 L2 还会改变用于构造缩放的历史状态，所以不能默认两条算法共享同一个 $M_t$。

## 5. 完整的 Adam 与 AdamW

以下考虑最小化目标、不使用 AMSGrad 的版本。初始化 $m_0=v_0=0$，$\beta_1,\beta_2\in[0,1)$、$\epsilon>0$。平方、平方根和除法都按坐标进行。

### 5.1 Adam + L2：正则梯度进入两个状态

$$
\begin{aligned}
\tilde g_t&=\nabla f_t(\theta_{t-1})+\lambda_2\theta_{t-1},\\
m_t&=\beta_1m_{t-1}+(1-\beta_1)\tilde g_t,\\
v_t&=\beta_2v_{t-1}+(1-\beta_2)\tilde g_t^{\odot2},\\
\hat m_t&=\frac{m_t}{1-\beta_1^t},\qquad
\hat v_t=\frac{v_t}{1-\beta_2^t},\\
\theta_t&=\theta_{t-1}
-\eta_t\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}.
\end{aligned}
$$

### 5.2 AdamW：矩估计只使用数据梯度

$$
\begin{aligned}
g_t&=\nabla f_t(\theta_{t-1}),\\
m_t&=\beta_1m_{t-1}+(1-\beta_1)g_t,\\
v_t&=\beta_2v_{t-1}+(1-\beta_2)g_t^{\odot2},\\
\hat m_t&=\frac{m_t}{1-\beta_1^t},\qquad
\hat v_t=\frac{v_t}{1-\beta_2^t},\\
\theta_t&=(1-\eta_t\lambda_d)\theta_{t-1}
-\eta_t\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}.
\end{aligned}
$$

两个框中的状态分别由各自算法维护，不是同一组 $m_t,v_t$。Adam 的矩估计和偏差修正见 [Kingma and Ba, Adam](https://arxiv.org/abs/1412.6980)；解耦衰减的实现约定见 [PyTorch AdamW](https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html)。

同一步中的梯度在旧参数 $\theta_{t-1}$ 上计算，衰减项也乘旧参数。如果先完成整个 Adam 更新，再将**整个结果**乘衰减因子，梯度更新项也会被缩放，不是上面这条更新式。

### 5.3 一个 Adam 的数值例子

令初值为 $\theta_0=(1,2)$，数据梯度为 $(0,0)$，$\eta_1=0.01$、$\lambda_2=\lambda_d=0.1$，$\epsilon=10^{-8}$。采用任意合法的 $\beta_1,\beta_2$，第一步偏差修正后，Adam + L2 有：

$$
\hat m_1=(0.1,0.2),\qquad
\hat v_1=(0.01,0.04).
$$

所以两个坐标都约减少 $0.01$，得到约 $(0.99,1.99)$。AdamW 的矩估计为 0，只做相对比例相同的衰减，得到 $(0.999,1.998)$。前者在这个例子中近似按相同**绝对量**减少，后者按相同**比例**减少。

## 6. 实践时检查三件事

1. **系数定义**：正则项有没有 $1/2$？Weight Decay 是否乘学习率？损失按总和还是平均值计算？
2. **更新路径**：正则项是否进入 momentum、二阶矩或 gradient clipping？一个参数叫 `weight_decay`，不等于它一定实现解耦衰减
3. **参数分组**：哪些参数参与衰减要明确。是否排除 bias、归一化参数等取决于模型与训练配置，不能只从优化器名称推断

AdamW 中的“解耦”指衰减不进入数据梯度的自适应矩估计，**不代表衰减效果与学习率完全无关**。若数据梯度和初始矩状态都为 0，在本文约定下：

$$
\theta_T=\left[\prod_{t=1}^T(1-\eta_t\lambda_d)\right]\theta_0.
$$

因此训练步数与学习率 schedule 仍会影响累计衰减。AdamW 提供了更明确的衰减机制，但不保证在所有任务上都更好。

## Reference

1. [Loshchilov and Hutter, Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101)，SGD 的系数换算、自适应缩放与 AdamW
2. [Kingma and Ba, Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980)，矩估计和偏差修正
3. [PyTorch AdamW 文档](https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html)，学习率缩放的解耦衰减约定
