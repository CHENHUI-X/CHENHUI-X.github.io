---
title: AdamW (part I) — Weight Decay == L2 Regularization?
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

在 [上一篇 Blog](/posts/2024-04-20-l1-and-l2-regularization/#32-让权重-www-衰减) 中探讨了 L1 Regularization 和 L2 Regularization. 我们说到: 对损失函数添加 L2 Regularization , 最后对 w 使用普通梯度下降的时候, 实际是对 w 做了权重衰减.


上述等价性对不带 momentum 的普通 SGD 成立（需要换算系数）。加入 momentum 或使用 [Adam](https://arxiv.org/abs/1412.6980) 时，如果 weight decay 指的是与梯度更新解耦的参数衰减，通常不再等价于把 L2 项加入损失函数。

本篇 Blog 主要探讨在使用 Adam 的时候 Weight Decay 和 L2 Regularization 的关系, 以及当更新参数引入 momentum之后他们之间的关系 , 最后介绍 AdamW 优化器. 文中符号都尽量与 [AdamW paper](https://arxiv.org/abs/1711.05101) 中的一致.

:::note
阅读前, 需要你 : 有高数基础知识, 线代基础知识, 当然还要有 ML和 DL 的知识背景.
:::


## 1. SGD场景下

### 1.1 无 momentum

weight decay 的公式:


$$
\theta_{t+1} = (1 - \lambda ) \theta_{t} - \alpha \nabla f_t(\theta_{t})
$$


这里 $\alpha$ 是学习率 , $\lambda$ 是 weight decay 的系数. 如果对损失函数施加 L2 Regularization :


$$
f_t^{reg}(\theta) =   f_t(\theta) + \frac {\lambda '} {2} {\|\theta\|_2}^2
$$


使用梯度下降:


$$
\begin{align*}
\theta_{t+1} &=   \theta_{t}  - \alpha \nabla f_t^{reg}(\theta_{t}) \\
&=   \theta_{t}  - \alpha \nabla f_t(\theta_{t}) - \alpha \lambda ' \theta_{t}\\
&= (1 - \alpha \lambda ' )  \theta_{t}  - \alpha \nabla f_t(\theta_{t})
\end{align*}
$$


如果想让 weight decay 和 带L2 Regularization 等价 , 则应有
$\alpha \lambda' = \lambda$
, 显然对于SGD我们可以做到这个事情. 也就是说 **在SGD优化器下, weight decay 和 带L2 Regularization 等价.** 不过有个问题, 假设我们存在一个最优的weight decay系数 $\lambda$ , 并且置了 L2 的系数
$\lambda'$
, 这样就会把系统的学习率给固定了. 换句话说, 这时 weight decay 的系数 和 L2 Regularization 的系数是耦合的. 二者会相互影响.


### 1.2 添加 momentum

加入 momentum 后，若把 L2 项加进梯度，更新为：


$$
g_t = \nabla f_t(\theta_t) + \lambda' \theta_t
$$


$$
m_t = \beta_1m_{t-1} + g_t,\qquad
\theta_{t+1}=\theta_t-\alpha m_t
$$


代入后可见，正则化项不仅影响当前参数，还会累积进动量缓存：


$$
\begin{align*}
\theta_{t+1}
&=(1-\alpha\lambda')\theta_t
-\alpha\nabla f_t(\theta_t)-\alpha\beta_1m_{t-1}.
\end{align*}
$$


而解耦的 weight decay 只衰减参数，不把 $\lambda\theta_t$ 放进动量缓存：

$$
\begin{aligned}
\tilde m_t&=\beta_1\tilde m_{t-1}+\nabla f_t(\theta_t),\\
\theta_{t+1}&=(1-\alpha\lambda)\theta_t-\alpha\tilde m_t.
\end{aligned}
$$

因此，普通 SGD 在采用上式的衰减定义时，取 $\lambda=\lambda'$ 即可等价；本文首节将衰减因子写为 $1-\lambda$，对应的换算是 $\lambda=\alpha\lambda'$。但在上述 momentum 定义下，两个动量缓存的历史不同，一般不能只靠重设一个固定系数使整条更新轨迹相同。


## 2. Adam场景下

这里就不敲公式了,给出 [AdamW paper](https://arxiv.org/abs/1711.05101) 附录的证明.

![image.png](https://s2.loli.net/2024/04/21/afDMybYdESVpQoB.png)

我们知道, 在 Adam 优化器中, 学习率是自适应变化的, 上图中 $M_t$ 就表示给学习率乘的自适应系数矩阵. 要想


$$
\lambda \theta_{t}  = \alpha \lambda ' M_t \theta_{t}
$$


就必须让


$$
\lambda   = \alpha \lambda ' M_t
$$


其中 $\lambda \ , \alpha \ ,\lambda' $ 三兄弟都是常数, $M_t$  又是自适应系数, 显然是不能实现上边的目标的,

:::warning
因此对于类似 Adam 这种自适应学习率的算法,  Weight Decay $\neq$ L2 Regularization . 无论加不加 momentum
:::


## Reference
