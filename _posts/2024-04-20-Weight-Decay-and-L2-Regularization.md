---
title: AdmaW(part I) Weight Decay == L2 Regularization?
date: 2024-04-20 20:18:00 +0800
categories: [Deep Learning]
tags: [machine learning,  mathematics, statistics, deep learning]     # TAG names should always be lowercase
math: true
---


## 0. 前言

在 [上一篇 Blog](https://chenhui-x.github.io/posts/L1-and-L2-Regularization/#%E8%AE%A9%E6%9D%83%E9%87%8D-w-%E8%A1%B0%E5%87%8F) 中探讨了 L1 Regularization 和 L2 Regularization. 我们说到: 对损失函数添加 L2 Regularization , 最后对 w 使用梯度下降的时候, 实际是对 w 做了权重衰减.


然而, 上述等价性只在优化器为随机梯度下降（SGD）时成立(下边我们会证明). 在其他情况下, 特别是在训练深度学习模型时, 经常使用[Adam](https://arxiv.org/abs/1412.6980)优化器 , 上述结论不成立.

本篇 Blog 主要探讨在使用 Adam 的时候 Weight Decay 和 L2 Regularization 的关系, 以及当更新参数引入 momentum之后他们之间的关系 , 最后介绍 AdamW 优化器. 文中符号都尽量与 [AdamW paper](https://arxiv.org/abs/1711.05101) 中的一致.

> 阅读前, 需要你 : 有高数基础知识, 线代基础知识, 当然还要有 ML和 DL 的知识背景.
{: .prompt-info }



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

如果在 L2 Regularization 的基础上添加 momentum 项


$$g_t = \nabla f_{t-1}(\theta_{t-1}) + \lambda ' \theta_{t-1}   $$


$$m_t = \beta_{1}m_{t-1} + g_t $$

SGD with momentum and weight decay (L2 Regularization) 式子将会变为:



$$
\begin{align*}
\theta_{t} &=   \theta_{t-1}  - \alpha  m_t \\
&=   \theta_{t-1}  -  \alpha (\beta_{1}m_{t-1} -  \nabla f_{t-1}(\theta_{t-1}) - \lambda ' \theta_{t-1}) \\

&= \underbrace{(1 - \alpha \lambda ' )  \theta_{t-1}}_{weight \ decay}  - \underbrace{\alpha \nabla f_{t-1}(\theta_{t-1})}_{gradient \ descent} -  \underbrace{\alpha \beta_{1}m_{t-1}}_{momentum}

\end{align*}
$$

这里, 学习率 $\alpha$ 和 L2 Regularization 的系数还是耦合, 并且还和 momentum 的系数也耦合上了.

> 耦合归耦合, 但是该说不说, 在SGD场景下, Weight Decay == L2 Regularization 是可以成立的. 无论加不加 momentum
{: .prompt-warning }


## 2. Adam场景下

这里就不敲公式了,给出 [AdamW paper](https://arxiv.org/abs/1711.05101) 附录的证明.

![image.png](https://s2.loli.net/2024/04/21/afDMybYdESVpQoB.png)

我们知道, 在 Adam 优化器中, 学习率是自适应变化的, 上图中 $M_t$ 就表示给学习率乘的自适应系数矩阵. 要想

$$\lambda \theta_{t}  = \alpha \lambda ' M_t \theta_{t}$$

就必须让

$$\lambda   = \alpha \lambda ' M_t  $$

其中 $\lambda \ , \alpha \ ,\lambda' $ 三兄弟都是常数, $M_t$  又是自适应系数, 显然是不能实现上边的目标的,

> 因此对于类似 Adam 这种自适应学习率的算法,  Weight Decay $\neq$ L2 Regularization . 无论加不加 momentum
{: .prompt-warning }




## Reference








