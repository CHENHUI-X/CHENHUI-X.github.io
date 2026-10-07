---
title: RoPE 旋转位置编码 — 从目标出发，一步步推出旋转矩阵
published: 2026-05-16
description: 从 query 和 key 的相对位置需求出发，理解 RoPE 的旋转、维度与实现，并分清长距离衰减的假设和边界。
category: Deep Learning
tags:
- llm
- transformer
- position encoding
- rope
- deep learning
draft: false
---

## 0. 前言：位置到底影响什么

“我喜欢你”和“你喜欢我”用了相同的词，意思却不同。Self-Attention 需要知道 token 的顺序，不能只比较内容向量。

先回顾 query、key 和 value 的作用：**query 表示当前位置想寻找什么，key 表示一个位置能用什么特征被匹配，value 是匹配后要汇总的信息**。它们都是模型学习出来的向量，不是事先写好的语义标签。如果还不熟悉这部分，可以先读 [Transformer Decoder](/posts/2024-04-13-text-generator-with-transformer-decoder/)。

本文先看一个 attention head。用列向量表示单个 token 的隐藏状态 $\mathbf{x}_m\in\mathbb{R}^{d_{\text{model}}}$，省略投影的 bias：

$$
\mathbf{q}_m=W_Q\mathbf{x}_m,\qquad
\mathbf{k}_n=W_K\mathbf{x}_n,\qquad
W_Q,W_K\in\mathbb{R}^{d\times d_{\text{model}}}.
$$

这里 $m,n$ 是位置，$d$ 是单个 head 的 query/key 维度。$\mathbf{q}_m,\mathbf{k}_n\in\mathbb{R}^d$ 是**本层尚未施加 RoPE** 的向量；它们可能已经包含前面各层处理过的上下文。下面暂时固定这两个内容向量，简写成 $\mathbf{q},\mathbf{k}$。

它们的点积 $\mathbf{q}^{\mathsf T}\mathbf{k}$ 是匹配分数的基础。实际 attention 还要除以 $\sqrt d$、加 mask，再对所有可见 key 做 softmax。**点积、缩放后的 logit、softmax 后的权重，是三个不同的量。**

如果把一个长度为 $S$ 的序列按行排成矩阵，则单个 head 的 $Q,K\in\mathbb{R}^{S\times d}$，分数矩阵 $QK^{\mathsf T}\in\mathbb{R}^{S\times S}$。矩阵的第 $m$ 行、第 $n$ 列，就是 query $m$ 对 key $n$ 的分数。小写表示一个向量，大写表示整批向量。

没有位置编码、位置相关 bias 或 mask，且忽略 dropout 的随机性时，Self-Attention 对 token 排列具有 **permutation equivariance（置换等变性）**：输入换序，输出也按相同顺序换序，并不是输出矩阵完全不变。若 $P$ 是换序矩阵，且 $F$ 表示这样的 attention，则：

$$
F(PX)=PF(X).
$$

这说明它本身没有表达词序的机制。固定的 causal mask 会改变可见关系，因此不能把上述任意换序性质直接套在带 causal mask 的 Decoder 上。

RoPE（Rotary Position Embedding）的思路是：**用位置决定旋转角度，旋转 query 和 key，再计算它们的点积。**

## 1. 设定目标：保留内容，让位置以差值出现

定义位置编码函数 $f_q(\mathbf{q},m)$ 和 $f_k(\mathbf{k},n)$。我们希望：

$$
\left\langle f_q(\mathbf{q},m),f_k(\mathbf{k},n)\right\rangle
=g(\mathbf{q},\mathbf{k},m-n).
$$

也就是说，**固定内容向量后，显式的位置项只依赖相对位置差**。它没有说分数只由距离决定，也没有说两个不同 token 只要距离相同就必然得到同样的分数。

下面构造一个满足目标的方案：在二维平面内做保持长度的旋转。相对位置条件本身并不能唯一确定 RoPE；我们是在选择一组容易计算、能保持向量长度的函数，然后证明它满足条件。RoFormer 论文的 [第 3.1—3.2 节](https://arxiv.org/html/2104.09864v5#S3.SS1) 给出了这一建模思路。

## 2. 从二维旋转开始

### 2.1 两个坐标，也可以写成一个复数

把两个二维向量写成：

$$
\mathbf{q}=\begin{pmatrix}q_0\\q_1\end{pmatrix},\quad
\mathbf{k}=\begin{pmatrix}k_0\\k_1\end{pmatrix},\quad
\widetilde q=q_0+\mathrm{i}q_1,\quad
\widetilde k=k_0+\mathrm{i}k_1.
$$

其中 $\mathrm{i}^2=-1$。复数只是把二维坐标换一种写法；对 key 取共轭，即把虚部的正负号反过来，就能表达原来的实向量点积：

$$
\begin{aligned}
\widetilde q\,\overline{\widetilde k}
&=(q_0k_0+q_1k_1)+\mathrm{i}(q_1k_0-q_0k_1),\\
\mathbf{q}^{\mathsf T}\mathbf{k}
&=\operatorname{Re}\!\left[\widetilde q\,\overline{\widetilde k}\right].
\end{aligned}
$$

### 2.2 用位置乘上一个固定的角速度

选定 $\theta$，表示每前进一个位置旋转多少**弧度**。让 query 在位置 $m$ 旋转 $m\theta$，key 在位置 $n$ 旋转 $n\theta$：

$$
\widetilde q_m=\widetilde q\,e^{\mathrm{i}m\theta},\qquad
\widetilde k_n=\widetilde k\,e^{\mathrm{i}n\theta}.
$$

两者必须使用同一个频率 $\theta$。相乘时，共轭把 key 的相位变成负号，于是：

$$
\operatorname{Re}\!\left[\widetilde q_m\overline{\widetilde k_n}\right]
=\operatorname{Re}\!\left[\widetilde q\,\overline{\widetilde k}\,
e^{\mathrm{i}(m-n)\theta}\right].
$$

位置只通过 $m-n$ 出现。由于 $e^{\mathrm{i}m\theta}$ 的模长是 1，这个操作只改变方向，不改变向量长度。

### 2.3 回到实数矩阵

由欧拉公式 $e^{\mathrm{i}\phi}=\cos\phi+\mathrm{i}\sin\phi$，上面的复数乘法等价于：

$$
R(\phi)=
\begin{pmatrix}
\cos\phi&-\sin\phi\\
\sin\phi&\cos\phi
\end{pmatrix},\qquad
R(m\theta)\mathbf{q}=
\begin{pmatrix}
q_0\cos(m\theta)-q_1\sin(m\theta)\\
q_0\sin(m\theta)+q_1\cos(m\theta)
\end{pmatrix}.
$$

例如 $\mathbf{q}=(1,2)^{\mathsf T}$，旋转 $\pi/2$ 后得到 $(-2,1)^{\mathsf T}$，旋转前后长度都是 $\sqrt5$。公式里的负号很重要：它决定旋转方向。

## 3. 验证相对位置、正负号和长度

二维旋转矩阵有两条性质：

$$
R(\phi)^{\mathsf T}=R(-\phi)=R(\phi)^{-1},\qquad
R(\alpha)R(\beta)=R(\alpha+\beta).
$$

第一条来自 $\cos^2\phi+\sin^2\phi=1$；第二条来自三角函数的和角公式。因此：

$$
\begin{aligned}
\left(R(m\theta)\mathbf{q}\right)^{\mathsf T}
\left(R(n\theta)\mathbf{k}\right)
&=\mathbf{q}^{\mathsf T}R(m\theta)^{\mathsf T}R(n\theta)\mathbf{k}\\
&=\mathbf{q}^{\mathsf T}R((n-m)\theta)\mathbf{k}.
\end{aligned}
$$

**为什么复数形式是 $m-n$，矩阵形式却是 $n-m$？** 因为共轭与转置分别作用在不同的写法里。令 $\Delta=m-n$，展开同一个点积即可核对：

$$
\begin{aligned}
s(\Delta)
&=\mathbf{q}^{\mathsf T}R(-\Delta\theta)\mathbf{k}\\
&=(q_0k_0+q_1k_1)\cos(\Delta\theta)
+(q_0k_1-q_1k_0)\sin(\Delta\theta).
\end{aligned}
$$

这个结果和复数形式取实部完全相同。不要只把 $n-m$ 换成 $m-n$，却忘了同时调整正弦项的符号。

### 一个可以手算的例子

取 $\mathbf{q}=\mathbf{k}=(1,0)^{\mathsf T}$，$\theta=\pi/6$，也就是每个位置旋转 $30^\circ$：

- $m=1,n=3$ 时，两个旋转角分别为 $30^\circ,90^\circ$，点积是 $\cos60^\circ=1/2$
- 把两者同时向后移 10 个位置，$m=11,n=13$，角度变成 $330^\circ,390^\circ$，点积仍是 $1/2$
- 两组位置都有 $m-n=-2$，所以这个结果与公式一致

换成 $\mathbf{q}=(1,0)^{\mathsf T}$、$\mathbf{k}=(0,1)^{\mathsf T}$，仍取 $m=1,n=3$，点积为 $-\sqrt3/2$。第二个例子能检验正负号，而只用余弦例子检验不出来。

一般地，对任意共同平移量 $c$，保持内容、频率不变，就有：

$$
\left\langle R((m+c)\theta)\mathbf{q},R((n+c)\theta)\mathbf{k}\right\rangle
=\left\langle R(m\theta)\mathbf{q},R(n\theta)\mathbf{k}\right\rangle.
$$

而且 $\|R(m\theta)\mathbf{q}\|_2=\|\mathbf{q}\|_2$。**保持长度不等于保持两向量的点积**：只有两者施加同一个旋转时，点积才总是保持原值；不同位置通常会改变它们的夹角。

## 4. 为什么使用不同频率

单个二维旋转只提供一组周期信号。比如 $\theta=0.1$ 弧度/位置，位置 63 的角度是 6.3 弧度，比 $2\pi$ 只多约 0.0168 弧度，所以它的位置旋转接近位置 0。

这说明**单个频率可能让相隔很远的位置具有相近相位**，不是说不同位置的完整 token 表示一定相同。对于整数位置，能否精确重复还取决于频率；“转过一圈”不等于存在一个恰好重复的整数位置周期。

RoPE 让不同维度对使用不同频率。本文采用原始 RoPE 的常见设置：

$$
\theta_i=b^{-2i/d},\qquad
b=10000,\qquad i=0,1,\ldots,d/2-1.
$$

这里 $d$ 必须是偶数。$i$ 是**维度对的编号**，不是 token 的位置；$b$ 是频率基数，并不直接等于模型支持的上下文长度。

- 小编号的维度对旋转快，相邻位置之间的相位变化较大
- 大编号的维度对旋转慢，相位变化覆盖更长的距离尺度

每一对的波长可写成 $\lambda_i=2\pi/\theta_i$，表示连续位置坐标上转一圈对应的距离。多种频率提供多种尺度，但不保证任意长度的外推效果。

![不同频率下的旋转速度示意](/images/llm-series/rope-multi-freq-rotation.png)

> 把不同维度对理解为走得快慢不同的指针即可。它们的具体频率以上面的公式为准；图示不代表精确整数周期或上下文长度上限。

## 5. 从二维扩展到高维

把 $d$ 维向量分成 $d/2$ 对坐标，第 $i$ 对为：

$$
\mathbf{q}^{(i)}=
\begin{pmatrix}q_{2i}\\q_{2i+1}\end{pmatrix}.
$$

<!-- concept-figure:rope-relative-rotation -->

![Su 等人在 RoFormer 论文中的 RoPE 原图：上半部分展示位置 m 使二维 query 或 key 旋转 mθ，下半部分展示不同维度对使用不同频率，多个位置得到各自的旋转表示](/images/llm-series/sourced/roformer-figure-1.svg)

*图：上半部分展示一对坐标随位置旋转；下半部分展示高维向量的各维度对采用不同频率。图中频率下标从 1 开始，正文从 0 开始，二者仅编号不同。原图：Jianlin Su 等，[RoFormer，Figure 1](https://arxiv.org/html/2104.09864v5#S3.F1)，[CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/)，未修改。 [查看原尺寸图](/images/llm-series/sourced/roformer-figure-1.svg)。*

<!-- /concept-figure:rope-relative-rotation -->

分别旋转这些二维向量，就得到分块对角矩阵：

$$
R_m=\operatorname{diag}\left(
R(m\theta_0),R(m\theta_1),\ldots,R(m\theta_{d/2-1})
\right)\in\mathbb{R}^{d\times d}.
$$

“分块对角”表示旋转块沿**主对角线**排列，每块只处理自己的两维。以 $d=4$ 为例，记 $c_i=\cos(m\theta_i)$、$s_i=\sin(m\theta_i)$：

$$
R_m=
\begin{pmatrix}
c_0&-s_0&0&0\\
s_0&c_0&0&0\\
0&0&c_1&-s_1\\
0&0&s_1&c_1
\end{pmatrix}.
$$

取 $d=4,b=10000,m=1$，两组频率是 $1$ 和 $0.01$。向量 $(1,0,1,0)^{\mathsf T}$ 变成：

$$
\begin{pmatrix}\cos1\\\sin1\\\cos0.01\\\sin0.01\end{pmatrix}
\approx
\begin{pmatrix}0.540302\\0.841471\\0.999950\\0.010000\end{pmatrix}.
$$

它的长度仍为 $\sqrt2$。一对转得快，一对转得慢，互不混合。

为什么高维仍满足相对位置条件？因为点积可以按坐标对分组求和：

$$
\begin{aligned}
(R_m\mathbf{q})^{\mathsf T}(R_n\mathbf{k})
&=\sum_{i=0}^{d/2-1}
\left(R(m\theta_i)\mathbf{q}^{(i)}\right)^{\mathsf T}
\left(R(n\theta_i)\mathbf{k}^{(i)}\right)\\
&=\sum_{i=0}^{d/2-1}
(\mathbf{q}^{(i)})^{\mathsf T}R((n-m)\theta_i)\mathbf{k}^{(i)}\\
&=\mathbf{q}^{\mathsf T}R_{n-m}\mathbf{k}.
\end{aligned}
$$

每项都只包含相对位置，所以总和也如此。这是确定性的恒等式，**不需要向量分量独立，也不需要任何概率分布假设**。

本文把整个 head 的 $d$ 维都旋转。若模型只旋转其中偶数个维度，就对那部分使用旋转，对其余维度保持原值；需要按该模型的 rotary dimension 和配置来生成频率，不能直接照搬完整 head 的参数。

## 6. 长距离衰减：哪些成立，哪些不能保证

讨论 RoPE 的长距离性质时，要注意两点：旋转不会缩短向量；**它不保证任意一对 query/key 的分数随距离增大而降低，更不保证 softmax 后的注意力权重单调降低**。

RoFormer [第 3.4.3 节](https://arxiv.org/html/2104.09864v5#S3.SS4.SSS3) 使用复数和的变换与上界来讨论长距离行为。下面给出另一个便于计算的统计例子，帮助理解多频率抵消；它不是对所有真实 attention 的承诺。

### 6.1 先说明期望的假设

把 $\mathbf{q},\mathbf{k}$ 看作有有限二阶矩的随机向量。设它们满足：

$$
\mathbb{E}[q_jk_\ell]=\delta_{j\ell},\qquad
\delta_{j\ell}=
\begin{cases}1,&j=\ell,\\0,&j\ne\ell.\end{cases}
$$

这是关于 query 与 key 的**交叉二阶矩**的条件。若均值不是零，它就不能直接叫作协方差。

一个充分例子是 $\mathbf{q}=\mathbf{k}=\mathbf{x}$，且 $\mathbf{x}$ 的每个分量均值为 0、方差为 1，不同分量不相关。独立也能保证这里的不相关，但要求更强。

这个例子是**充分条件，不是必要条件**。例如 $\mathbf{q}=\mathbf{x}+\boldsymbol\epsilon$、$\mathbf{k}=\mathbf{x}+\boldsymbol\eta$，其中 $\mathbf{x}$ 满足上述条件，两个噪声都零均值，并且与 $\mathbf{x}$ 及彼此独立，也能满足同样的交叉二阶矩条件，而不必有 $\mathbf{q}=\mathbf{k}$。

真实模型通常有 $W_Q\ne W_K$；同一个 token 的 query 与 key 也通常不同。反过来，如果 query 与 key 相互独立且都为零均值，则 $\mathbb{E}[q_jk_\ell]=0$，此时点积期望处处为 0，得不出下面的余弦曲线。

### 6.2 在这个假设下计算

令 $\Delta=m-n$，第 $i$ 对坐标取 $u=2i,v=2i+1$。由二维展开式：

$$
\begin{aligned}
s_i(\Delta)
&=(q_uk_u+q_vk_v)\cos(\Delta\theta_i)\\
&\quad +(q_uk_v-q_vk_u)\sin(\Delta\theta_i).
\end{aligned}
$$

对角项期望为 1，交叉项期望为 0，因此：

$$
\mathbb{E}[s_i(\Delta)]=2\cos(\Delta\theta_i),\qquad
\mathbb{E}[s(\Delta)]=2\sum_{i=0}^{d/2-1}\cos(\Delta\theta_i).
$$

除以零距离处的值 $d$，可以得到便于比较维度的曲线：

$$
\kappa(\Delta)=\frac{\mathbb{E}[s(\Delta)]}{d}
=\frac{2}{d}\sum_{i=0}^{d/2-1}\cos(\Delta\theta_i),\qquad
\kappa(0)=1.
$$

不同频率在零距离时对齐，距离变大后可能相互抵消。在常见维度与有限观察区间内，这能表现出带振荡的衰减趋势。**有限个余弦之和不必单调，也不能据此断言它随距离趋于无穷而收敛到 0。**

最小的反例是 $d=2,b=10000$，此时 $\kappa(\Delta)=\cos\Delta$。整数距离 3 时约为 $-0.9900$，距离 6 时又回升到约 $0.9602$。更远的位置完全可以得到更大的点积。

![RoPE 多频率长距离趋势示意](/images/llm-series/rope-decay.png)

> 长距离图示应结合所选频率、维度、统计假设和观察区间阅读。它不能作为真实模型中每一对 token 的分数或 attention 权重都会下降的证明；本节的统计量由公式明确规定。

### 6.3 更一般的情况

若写 $C_{j\ell}=\mathbb{E}[q_jk_\ell]$，不再要求它等于单位矩阵，就有：

$$
\mathbb{E}[s(\Delta)]
=\sum_{i=0}^{d/2-1}\left[
(C_{2i,2i}+C_{2i+1,2i+1})\cos(\Delta\theta_i)
+(C_{2i,2i+1}-C_{2i+1,2i})\sin(\Delta\theta_i)
\right].
$$

这说明期望也取决于 query 与 key 的统计关系。上面的单位矩阵条件只是一个方便的充分条件，甚至不是得到同一条余弦期望曲线的唯一条件。若比较不同距离时 $C$ 自身也变化，还必须把这种内容变化计算在内。

最后，attention 权重还要与其他 key 竞争：

$$
\alpha_{mn}=\frac{\exp(s_{mn}/\sqrt d)}
{\sum_{j\in\mathcal V_m}\exp(s_{mj}/\sqrt d)},\qquad n\in\mathcal V_m.
$$

$\mathcal V_m$ 是 query $m$ 能看见的 key 集合，假设它非空。一般也不能把分数的期望先代进 softmax，当作权重的期望。RoPE 给出了位置结构，具体该关注近处还是远处，仍取决于模型学到的内容匹配和 mask。

## 7. PyTorch 实现：明确维度和 position IDs

下面实现相邻坐标配对的完整 head 旋转。输入 shape 为 `(batch, seq_len, heads, dim)`；`position_ids` 可以是全 batch 共用的 `(seq_len,)`，也可以是每条序列各自的 `(batch, seq_len)`。

这里直接按 position IDs 计算 sin/cos，避免把长度为 4096 的整张表误广播到只有几个 token 的输入。工程中可预计算并按 position IDs 取表。RoFormer 的[官方示例](https://github.com/ZhuiyiTechnology/roformer#implementation)使用成对坐标变换；Meta 的[官方 Llama 实现](https://github.com/meta-llama/llama/blob/main/llama/model.py)使用等价的复数乘法。

```python
import math
import torch


def apply_rope(x: torch.Tensor, position_ids, base: float = 10000.0):
    # x: (B, S, H, d); adjacent coordinates form a pair.
    if x.ndim != 4 or not x.is_floating_point():
        raise ValueError("x must be a floating tensor with shape (B, S, H, d)")
    B, S, H, d = x.shape
    if d == 0 or d % 2 != 0:
        raise ValueError("rotary dimension must be positive and even")
    if not math.isfinite(base) or base <= 0:
        raise ValueError("base must be finite and positive")

    pos = torch.as_tensor(position_ids, device=x.device)
    if pos.dtype not in (torch.int32, torch.int64):
        raise ValueError("position_ids must contain integers")
    if pos.shape == (S,):
        pos = pos.unsqueeze(0).expand(B, -1)
    if pos.shape != (B, S) or (pos < 0).any().item():
        raise ValueError("position_ids must have shape (S,) or (B, S), and be >= 0")

    # Avoid computing large position angles directly in FP16/BF16.
    work_dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
    pair_index = torch.arange(0, d, 2, device=x.device, dtype=work_dtype)
    inv_freq = base ** (-pair_index / d)                    # (d/2,)
    angles = pos.to(work_dtype)[..., None] * inv_freq       # (B, S, d/2)
    cos = angles.cos().unsqueeze(2)                        # (B, S, 1, d/2)
    sin = angles.sin().unsqueeze(2)

    even = x[..., 0::2].to(work_dtype)                      # (B, S, H, d/2)
    odd = x[..., 1::2].to(work_dtype)
    out = torch.stack((even * cos - odd * sin,
                       even * sin + odd * cos), dim=-1)
    return out.flatten(-2).to(x.dtype)                     # (B, S, H, d)


# A small MHA example; q and k are already projected.
B, S, H, d = 2, 4, 3, 8
q = torch.randn(B, S, H, d)
k = torch.randn(B, S, H, d)
position_ids = torch.arange(S, device=q.device)
q_rotated = apply_rope(q, position_ids).transpose(1, 2)      # (B, H, S, d)
k_rotated = apply_rope(k, position_ids).transpose(1, 2)
logits = q_rotated @ k_rotated.transpose(-2, -1) / math.sqrt(d)
# logits: (B, H, S, S); mask, softmax, and value aggregation follow.
```

这段教学代码只计算旋转和 logits，不包含 mask、softmax、value 汇总或完整 KV Cache。它使用相邻配对 `(0,1), (2,3), …`；有些实现把前后两半配对。两者可以通过一致的坐标重排对应，但不能只换旋转函数、不调整已有模型的权重和布局。

若 query 有 $H_Q$ 个 head，key 有 $H_{KV}$ 个 head，旋转函数可以分别处理它们。但上面最后的直接矩阵乘法以相同 head 数的 MHA 为例；GQA/MQA 还需要正确映射 query head 与 KV head。

> 验证范围：本文数值例子和旋转恒等式使用独立的 NumPy CPU 测试核验。PyTorch 代码做了语法检查，本次环境未安装 PyTorch，因此没有声称完成其运行、梯度或 GPU 测试。

## 8. 生成时的位置偏移与 KV Cache

假设缓存已经包含位置 0 到 99 的 100 个 token。位置 99 的输出先预测位置 100 的 token；把这个新 token 送入下一次前向时，它的新 query 和 key 应使用位置 **100**，而不是因为这一轮只输入一个 token 就重新编号为 0。

对于连续的新片段，可以这样构造位置：

```python
past_len = 100
new_len = 1
position_ids = torch.arange(past_len, past_len + new_len, device=q.device)
```

常见实现会将**已经按各自位置旋转过的 key** 放入 KV Cache。后续生成时，只给新增的 query/key 使用正确位置，再让新 query 与缓存 key 计算分数；不要把缓存 key 当成未旋转向量，再按它的绝对位置旋转一次。value 通常不做这里的 RoPE 旋转。

例如位置 100 的 query 与位置 97 的 key，对应 $\Delta=3$。若错误地把新 query 标成位置 0，它们的位置差就变成 $-97$。缓存能省去重复计算，但仍需正确的位置坐标。

左 padding、不同长度的 batch、滑动窗口或多段样本拼接，会让“缓存槽位编号”和“语义上的 position ID”不再总是一回事：

- 每条序列的 position IDs 必须与它的有效 token 和缓存约定一致，不能一律拿 batch 的 padded 长度当偏移
- padding 位置可用无害的占位 ID，但仍需正确的 attention/loss mask；位置编号不会自动遮蔽 padding
- 拼接独立样本时，仅重置 position IDs 不会阻止样本之间相互 attention，还需要相应的 mask
- 若主动平移一整段缓存的位置坐标，必须让相关 query/key 保持同一个坐标系；只改新 token 的编号不能维持相对位置

缓存细节可以继续读 [推理显存：KV Cache 与 Attention](/posts/2026-06-13-llm-mem-opt-2-inference/)。

## 9. 总结

RoPE 的计算可分为四步：

1. 把一个 head 的向量按两维一组配对
2. 给不同维度对设置不同的旋转频率
3. 用 token 的 position ID 决定每一对的旋转角度
4. 旋转 query/key 后再做点积，得到内容与相对位置共同决定的分数

核心恒等式是 $(R_m\mathbf{q})^{\mathsf T}(R_n\mathbf{k})=\mathbf{q}^{\mathsf T}R_{n-m}\mathbf{k}$。它解释了相对位置，也解释了为什么生成时必须保持正确的位置偏移。

这些性质也有边界：满足相对位置条件不等于方案唯一；保持范数不等于注意力不变；统计例子里的振荡趋势不等于实际 attention 随距离单调下降；能计算更大的位置也不等于模型自然学会更长上下文。

沿主线继续读 [Decoding：如何选下一个 token](/posts/2026-05-16-llm-decoding-strategies/)。如果想进一步理解位置与频率的调整，可选读 [长上下文扩展：PI、NTK-aware 与 YaRN](/posts/2026-05-16-llm-long-context-yarn/)。完整阅读路线见 [LLM 知识地图](/posts/2026-06-03-llm-knowledge-map/)。

### 参考资料

1. Su et al. [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)，arXiv:2104.09864，初稿 2021。本文参考其相对位置目标和旋转构造；统计例子另行说明了假设
2. ZhuiyiTechnology. [RoFormer 官方实现说明](https://github.com/ZhuiyiTechnology/roformer#implementation)，相邻维度配对与 sin/cos 写法
3. Meta. [Llama 官方 model.py](https://github.com/meta-llama/llama/blob/main/llama/model.py)，`precompute_freqs_cis`、`apply_rotary_emb` 与缓存位置处理
4. Vaswani et al. [Attention Is All You Need](https://arxiv.org/abs/1706.03762)，NeurIPS 2017，scaled dot-product attention 与多频率位置编码
