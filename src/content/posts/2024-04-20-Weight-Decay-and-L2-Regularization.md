---
title: AdamW：权重衰减与 L2 正则化的区别
published: 2024-04-20
description: 比较 L2 正则化与解耦权重衰减在 SGD、动量和 Adam 中的更新公式，通过数值例子解释 AdamW 的做法。
category: Deep Learning
tags:
- machine learning
- mathematics
- statistics
- deep learning
draft: false
---
<span id="0-前言"></span>

<span id="0-为什么-l2-正则化不能代替-adamw"></span>

## 0. 有了 L2 正则化，为什么还需要 AdamW

在 [L1 与 L2 正则化](/posts/2024-04-20-l1-and-l2-regularization/) 中，我们把普通梯度下降中的 L2 梯度拆成了一个权重缩小项。既然如此，为什么还需要 AdamW？

关键要看**优化器怎样处理这部分更新**。L2 正则化把额外的梯度加入数据梯度；解耦权重衰减（decoupled Weight Decay）则直接缩小参数，不让衰减量参与动量和自适应二阶矩的计算。这两种做法对普通 SGD 可以等价，对 momentum 或 Adam 一般不同。[AdamW 原论文](https://arxiv.org/abs/1711.05101)

<span id="1-先统一符号与衰减定义"></span>

<span id="1-梯度学习率与衰减系数"></span>

## 1. 先说清梯度、学习率和衰减系数

令 $\theta_{t-1}\in\mathbb R^d$ 是第 $t$ 次更新前的参数向量：$d$ 是参数个数，$\theta$ 是这些参数排成的向量，$\theta_j$ 表示第 $j$ 个参数。$f_t$ 是当前 batch 的数据损失：

$$
g_t=\nabla f_t(\theta_{t-1}),\qquad t=1,2,\ldots
$$

这里 $\nabla f_t$ 是对每个参数求偏导后组成的梯度向量，指出数据损失局部增加最快的方向；优化器沿负方向更新。batch 是本次计算梯度使用的一小批训练数据，$t$ 表示更新次数。学习率为 $\eta_t>0$。

平方范数 $\|\theta\|_2^2=\sum_{j=1}^d\theta_j^2$，就是把所有参数的平方相加。本文区分两个系数：

- L2 系数 $\lambda_2\ge0$：损失增加 $(\lambda_2/2)\|\theta\|_2^2$，梯度增加 $\lambda_2\theta$
- 解耦衰减系数 $\lambda_d\ge0$：每步对旧参数乘 $1-\eta_t\lambda_d$

上一篇的 L2 项写作 $\lambda\|w\|_2^2$，所以对应这里的 $\lambda_2=2\lambda$。另一些论文把每步衰减因子直接写成 $1-d_t$，那就令 $d_t=\eta_t\lambda_d$。比较实现时，需要按这些定义换算系数，不能只比较同名超参数的数值。

本文与 PyTorch AdamW 文档采用相同约定：每步的衰减量为 $\eta_t\lambda_d\theta_{t-1}$，其中包含学习率 $\eta_t$。通常令 $0\le\eta_t\lambda_d<1$，以免衰减步骤将参数翻转符号。[PyTorch AdamW 算法定义](https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html)

<span id="2-sgd不带-momentum-时可以等价"></span>

<span id="2-普通-sgd-中的等价条件"></span>

## 2. 普通 SGD 中，两种做法什么时候相同

<span id="21-将-l2-放进损失"></span>

### 2.1 损失函数加入 L2 正则项

$$
\begin{aligned}
\theta_t
&=\theta_{t-1}-\eta_t\left(g_t+\lambda_2\theta_{t-1}\right)\\
&=(1-\eta_t\lambda_2)\theta_{t-1}-\eta_tg_t.
\end{aligned}
$$

<span id="22-使用解耦-weight-decay"></span>

### 2.2 直接衰减参数

$$
\theta_t=(1-\eta_t\lambda_d)\theta_{t-1}-\eta_tg_t.
$$

在相同初始参数、batch 顺序和学习率下，取 $\lambda_d=\lambda_2$，两种算法每一步都会得到相同的参数。这个结论针对不带 momentum、不做自适应缩放的 SGD；两边也不能额外采用不同的梯度处理，例如只在一边对加入 L2 后的总梯度做 clipping。

若改用“每步都乘以固定因子 $1-d$”的定义，匹配条件是 $d=\eta_t\lambda_2$。学习率变化时，固定 $d$ 通常不能再对应固定的 $\lambda_2$。等价条件允许学习率变化，但不同定义下的衰减系数需要随学习率调度一同换算。

<span id="3-加入-momentum看缓存里存了什么"></span>

## 3. 动量会累积 L2 正则项的梯度

momentum（动量）在缓存中保留历史梯度，并将它们与当前梯度合并，因此每次更新都受到之前 batch 的影响。$m_t$ 是这个缓存，$\beta$ 控制保留多少历史；$\beta=0$ 时就不保留。使用不带 dampening（对新梯度的额外衰减）、不开 Nesterov 的一个常见 momentum 约定，$0\le\beta<1$，初始缓存为 0。

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

这里 $g_t^{\mathrm{L2}}=\nabla f_t(\theta_{t-1}^{\mathrm{L2}})$，$g_t^{\mathrm{WD}}=\nabla f_t(\theta_{t-1}^{\mathrm{WD}})$。两种算法一旦得到不同参数，随后在这些参数上算出的数据梯度也可能不同。

<span id="31-两步就能看出差别"></span>

<span id="31-两次更新的数值比较"></span>

### 3.1 算两步，看看差别出在哪里

设只有一个参数，初值 $\theta_0=1$，数据损失恒定，因此每步数据梯度为 0。取 $\eta=0.1$、$\beta=0.9$、$\lambda_2=\lambda_d=0.1$：

| 更新 | L2 + momentum | 解耦 Weight Decay + momentum |
| --- | --- | --- |
| 第 1 步缓存 | $m_1=0.1$ | $m_1=0$ |
| 第 1 步参数 | $\theta_1=0.99$ | $\theta_1=0.99$ |
| 第 2 步缓存 | $m_2=0.9(0.1)+0.1(0.99)=0.189$ | $m_2=0$ |
| 第 2 步参数 | $\theta_2=0.99-0.1(0.189)=0.9711$ | $\theta_2=0.99^2=0.9801$ |

两个系数已经按第一步的结果匹配，第二步却仍然不同，因为 L2 的旧梯度还留在缓存中。换一种含 $1-\beta$ 的 momentum 定义会改变系数换算，但不会消除缓存历史的区别。

<span id="4-自适应缩放同一份-l2-梯度会按坐标改变"></span>

## 4. 自适应优化器如何缩放 L2 梯度

自适应优化器可以对不同参数使用不同的更新尺度。例如某个参数过去的梯度很大，优化器就可能把它当前的梯度缩小一些。先暂时不考虑 momentum，用一个**给定**的矩阵 $M_t\in\mathbb R^{d\times d}$ 表示梯度缩放。比较：

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

这里 $I_d$ 是 $d\times d$ 单位矩阵：对角线全为 1，其他位置全为 0，乘一个向量后向量保持不变。左边的系数要乘单位矩阵，才能与右边的缩放矩阵比较。对于非零正则强度，如果 $M_t$ 各坐标缩放不同，就不存在一个标量 $\lambda_2$ 同时满足所有坐标；即使每步 $M_t=c_tI_d$，当 $c_t$ 随时间变化时，也通常无法用两个固定系数匹配整条轨迹。

$\operatorname{diag}(1,10)$ 表示对角线为 1、10，其他位置为 0 的对角矩阵；它把向量的第一个坐标乘 1、第二个坐标乘 10。

例如 $M_t=\operatorname{diag}(1,10)$、$\theta=(1,1)$、$\eta_t=0.1$、$\lambda_2=\lambda_d=0.1$，仅看正则部分，L2 产生的减量是 $(0.01,0.1)$，解耦衰减产生的减量是 $(0.01,0.01)$。

对某个特殊向量、某个时间点，两者仍可能偶然相同；零衰减时也没有这一区别。因此准确结论是“通常不存在统一的标量系数使两种算法等价”，而不是任何情况下都必然不等。[AdamW 原论文，Appendix A](https://arxiv.org/html/1711.05101)

这个矩阵例子解释了坐标缩放的影响，但它不是完整 Adam 的更新式。在 Adam 中，加入 L2 还会改变用于构造缩放的历史状态，所以不能默认两条算法共享同一个 $M_t$。

<span id="5-完整的-adam-与-adamw"></span>

## 5. Adam 与 AdamW 的完整更新公式

Adam 同时维护两个历史量：$m_t$ 是梯度的指数移动平均，用来平滑方向；$v_t$ 是梯度平方的指数移动平均，用来估计各坐标的尺度。$\beta_1,\beta_2$ 分别控制保留多少旧值。以下考虑最小化目标、不使用 AMSGrad 变体的版本。初始化 $m_0=v_0=0$，$\beta_1,\beta_2\in[0,1)$、$\epsilon>0$。平方、平方根和除法都按坐标进行；例如 $(a,b)^{\odot2}=(a^2,b^2)$，$\epsilon$ 是避免分母为零的小正数。

因为缓存从 0 开始，初期的移动平均偏小，所以要做“偏差修正”：除以 $1-\beta_1^t$ 或 $1-\beta_2^t$。例如第一步 $m_1=(1-\beta_1)g_1$，修正后 $\hat m_1=g_1$；帽子表示修正后的量。

<span id="51-adam--l2正则梯度进入两个状态"></span>

<span id="51-adam--l2正则梯度参与一阶矩和二阶矩估计"></span>

### 5.1 Adam + L2：把正则梯度一起存进历史状态

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

<span id="52-adamw矩估计只使用数据梯度"></span>

### 5.2 AdamW：历史状态只记录数据梯度，参数单独衰减

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

上面两组公式中的状态分别由各自算法维护，不是同一组 $m_t,v_t$。Adam 的矩估计和偏差修正见 [Kingma and Ba, Adam](https://arxiv.org/abs/1412.6980)；解耦衰减的实现约定见 [PyTorch AdamW](https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html)。

同一步中的梯度在旧参数 $\theta_{t-1}$ 上计算，衰减项也乘旧参数。如果先完成整个 Adam 更新，再将**整个结果**乘衰减因子，梯度更新项也会被缩放，不是上面这条更新式。

下面直接对照原论文的双色算法框。它把两种算法写在一起，**紫色项与绿色项二选一**，不是把两处正则项都执行。

<img src="/images/llm-series/sourced/adamw-algorithm2-original.png" alt="AdamW 原论文 Algorithm 2，紫色 L2 项在第 6 行进入梯度，绿色衰减项在第 12 行直接进入参数更新" width="1000" />

1. 第 6 行紫色项：Adam + L2 先把正则梯度加进 $g_t$，所以第 7、8 行的一阶矩和二阶矩都会受到它影响。
2. 第 12 行绿色项：AdamW 不执行第 6 行的紫色加项；它完成矩估计后，直接对旧参数施加衰减，绕开自适应分母。
3. 原图中，$\alpha$ 是基础学习率，$\eta_t$ 只是调度乘子；二者的乘积，才对应本文的学习率 $\eta_t$。
4. 原图紫色 L2 项的 $\lambda$ 对应本文的 $\lambda_2$；绿色衰减项的 $\lambda$ 则对应“原图的 $\alpha$ 乘本文的 $\lambda_d$”。两种系数要分别换算，不能把同名字母的数值直接套用。

图源：[Loshchilov、Hutter，Decoupled Weight Decay Regularization，Algorithm 2](https://arxiv.org/pdf/1711.05101#page=3)。从论文第 3 页提取完整算法框，未改动公式、行号和双色高亮；版权归原作者，arXiv 页面未声明开放改编许可。

<span id="53-一个-adam-的数值例子"></span>

### 5.3 Adam 与 AdamW 的数值比较

令初值为 $\theta_0=(1,2)$，数据梯度为 $(0,0)$，$\eta_1=0.01$、$\lambda_2=\lambda_d=0.1$，$\epsilon=10^{-8}$。采用任意合法的 $\beta_1,\beta_2$，第一步偏差修正后，Adam + L2 有：

$$
\hat m_1=(0.1,0.2),\qquad
\hat v_1=(0.01,0.04).
$$

所以两个坐标都约减少 $0.01$，得到约 $(0.99,1.99)$。AdamW 的矩估计为 0，只做相对比例相同的衰减，得到 $(0.999,1.998)$。前者在这个例子中近似按相同**绝对量**减少，后者按相同**比例**减少。

<span id="6-实践时检查三件事"></span>

<span id="6-训练配置中的衰减系数与参数分组"></span>

## 6. 配置 weight decay 时要检查什么

1. **系数定义**：正则项有没有 $1/2$？Weight Decay 是否乘学习率？损失按总和还是平均值计算？
2. **梯度处理**：正则项是否进入 momentum、二阶矩或 gradient clipping？一个参数叫 `weight_decay`，不等于它一定实现解耦衰减
3. **参数分组**：哪些参数参与衰减要明确。是否排除 bias、归一化参数等取决于模型与训练配置，不能只从优化器名称推断

AdamW 中的“解耦”指衰减不进入数据梯度的自适应矩估计，**不代表衰减效果与学习率完全无关**。若数据梯度和初始矩状态都为 0，在本文约定下：

$$
\theta_T=\left[\prod_{t=1}^T(1-\eta_t\lambda_d)\right]\theta_0.
$$

因此，训练步数与学习率调度仍会影响累计衰减。AdamW 把参数衰减与梯度的自适应缩放分开处理；是否能改善训练效果，还要看具体任务。

<span id="reference"></span>

## 参考资料

1. [Loshchilov and Hutter, Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101)，SGD 的系数换算、自适应缩放与 AdamW
2. [Kingma and Ba, Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980)，矩估计和偏差修正
3. [PyTorch AdamW 文档](https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html)，学习率缩放的解耦衰减约定
