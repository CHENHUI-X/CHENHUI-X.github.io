---
title: 长上下文扩展 — 从 RoPE 出发，一步步推导 PI、NTK 到 YaRN
published: 2026-05-16
description: 从 RoPE 的 θ 公式出发，先想清楚"为什么 RoPE 在训练长度外效果差"，再一步步推出 PI、NTK-aware、YaRN 的改进思路和数学原理。
category: Deep Learning
tags:
- llm
- position encoding
- rope
- yarn
- deep learning
draft: false
---

## 0. 前言

上一篇 [RoPE：旋转位置编码](/posts/2026-05-16-llm-rope-rotary-position-embedding/) 介绍了如何旋转 query 和 key，让内积中的**显式位置项**通过相对位置进入计算。内积仍然取决于 token 内容，并不是距离相同，attention 就一定相同。

这篇继续讨论一个问题：模型原来只在 4K 长度上训练，如果要处理 32K 的文本，RoPE 应该怎么改？

下面依次介绍 Position Interpolation（PI）、NTK-aware scaling 和 YaRN。它们都在调整“位置变化多少，会让向量旋转多少”。区别在于：所有频率一起缩放，还是分别处理高频与低频，以及是否调整 attention 分数的尺度。

先约定符号，区分模型宽度与单个 head 的宽度：

| 符号 | 含义 |
|---|---|
| $m,n$ | 从 0 开始的 token 位置 |
| $d$ | 参与 RoPE 的维数；本文假设整个 attention head 都参与，$d$ 是正偶数 |
| $i$ | 二维旋转对的编号，$i=0,\ldots,d/2-1$ |
| $b$ | RoPE 的 base，数值例子取 $10000$，其他模型未必相同 |
| $\theta_i=b^{-2i/d}$ | 第 $i$ 对的角频率，即每前进一个 token 增加的旋转角度 |
| $\lambda_i=2\pi/\theta_i$ | 波长，即旋转一圈对应的 token 距离 |
| $L_{\text{train}},L_{\text{target}}$ | 原训练上下文长度、希望扩展到的长度 |
| $s=L_{\text{target}}/L_{\text{train}}$ | 扩展比例，本文只讨论有限的 $s\geq1$ |

这里的 4K、32K 分别指 4096、32768 个 token。NTK 的推导还要求 $d>2$。如果模型只旋转 head 的一部分，需要另行区分 rotary dimension 与 attention 分母使用的 head dimension。

## 1. 先定位问题：能计算长位置，不等于能理解长文本

原始 RoPE 在位置 $m$ 使用角度：

$$
\phi_{m,i}=m\theta_i.
$$

这个公式可以计算任意位置，但模型的参数只在有限长度的数据上学过如何使用这些角度。

旋转有周期性。不能简单说“大位置的每个角度模型都没见过”：高频分量在训练区间里可能已经转过很多圈。更准确的问题是，**多个频率的组合、相对距离和文本依赖关系超出了训练时的覆盖范围**，模型未必能正确利用它们。低频分量尤其可能在训练时只覆盖了一个周期的一小段。

缩放方法试图减少这种变化，但它们不能单独保证长文理解能力。训练数据、微调方式、检索任务与长距离推理任务的差异，也会影响最终表现。

## 2. 方案一：Position Interpolation（PI）

### 2.1 把位置压回原范围

PI 将位置换成 $m'=m/s$，于是：

$$
\phi_{m,i}^{\text{PI}}=\frac{m}{s}\theta_i
=m\theta_i^{\text{PI}},
\qquad
\theta_i^{\text{PI}}=\frac{\theta_i}{s}.
$$

也就是说，**位置除以 $s$ 与所有频率除以 $s$，在旋转角度上等价**。

例如从 4K 扩到 32K，$s=8$。位置 16000 使用原来位置 2000 的旋转角度。对任意 $0\leq m<L_{\text{target}}$，都有 $0\leq m/s<L_{\text{train}}$。

这只是把位置坐标压回连续区间，并不是把内容向量也变成了原来的向量，更不表示模型已经见过全部分数位置或长度为 32K 的依赖关系。PI 原论文配合微调验证效果，不能把这一步当成免训练的质量保证。[PI 原论文](https://arxiv.org/abs/2306.15595)

### 2.2 代价：局部角度差也变小

相邻位置的角度差从 $\theta_i$ 降到 $\theta_i/s$。例如最高频对 $i=0$：

$$
\theta_0=1,
\qquad
\theta_0^{\text{PI}}=\frac18=0.125.
$$

角度差从约 $57.30^\circ$ 变成 $7.16^\circ$。这解释了为什么统一缩放可能影响模型对短距离位置差异的利用。

但不能据此断言“相邻 token 的向量几乎重合”：query、key 还含有不同的内容，整个 head 也有许多旋转对。这里比较的是**同一二维内容向量因位置产生的旋转变化**，不是两个任意 token 的最终夹角。

PI 的优点是规则简单，所有频率的相位范围都被压缩；代价是连已经有充分周期覆盖的高频分量也一起改变了。

## 3. 方案二：NTK-aware scaling

### 3.1 先明确想保留什么

频率越高，相邻位置的角度变化越大；频率越低，完成一圈需要的距离越长。NTK-aware 希望最高频保持不变，最低频像 PI 一样缩小 $s$ 倍，中间频率逐渐过渡。

它保留位置 $m$，只把 base 从 $b$ 改成 $b'$：

$$
\theta_i'=b'^{-2i/d}.
$$

这个名称来自 Neural Tangent Kernel 相关启发；下面的推导只需要代数，不需要先学习 NTK 理论。

### 3.2 正确推导新 base

最高频对应 $i=0$，无论怎样修改 base，始终有 $\theta_0'=1$。

最低频对应 $i=d/2-1$。我们要求的是**最低频的波长扩大 $s$ 倍**，不是让它恰好等于目标上下文长度：

$$
\frac{\theta_{d/2-1}'}{\theta_{d/2-1}}
=\left(\frac{b'}{b}\right)^{-(d-2)/d}
=\frac1s.
$$

因此：

$$
\boxed{b'=b\,s^{d/(d-2)}}.
$$

把它代回每一对的频率：

$$
\boxed{\theta_i'=\theta_i\,s^{-2i/(d-2)}}.
$$

可以直接检查两个端点：$i=0$ 时比例为 1；$i=d/2-1$ 时比例为 $1/s$。这也是 YaRN 论文附录 A.1 给出的约束。[推导出处](https://arxiv.org/html/2309.00071v2)

例如 $d=128,b=10000,s=8$，得到 $b'\approx82684.62$。如果简单取 $b'=80000$，最低频比例是 $8^{-126/128}\approx0.12913$，并不严格等于 $0.125$。

最低频对应的是这些旋转对中**最长的波长**，不能称为“最小波长”。

### 3.3 保留最高频，不等于保留所有短文本表现

除 $i=0$ 外，其他频率一般仍会变化。NTK-aware 只是让低频缩放更多；并没有保证整个短文本 attention 保持原样。

而且，它没有像 PI 一样把所有相位压回原区间。对某些高频对，长距离上的相位仍在外推。因此公式里的 $s$ 是缩放参数，不能直接解释成“模型可靠理解长度必然增加 $s$ 倍”。是否需要微调、有效扩展到多长，必须看具体模型与评测。

### 3.4 静态 NTK 与 Dynamic NTK 分开看

上面的推导是**静态缩放**：一次运行选定一个 $s$，所有位置使用同一套频率。

动态缩放则根据当前长度 $\ell$ 调整参数。YaRN 论文给出的通用方式是：

$$
s(\ell)=\max\left(1,\frac{\ell}{L_{\text{train}}}\right).
$$

不同库对 Dynamic NTK 的具体参数化并不完全一样。例如 Transformers 4.44.2 的 `dynamic` 实现使用：

$$
\ell_* = \max(\ell,L_{\text{train}}),
\qquad
b'(\ell)=b\left(f\frac{\ell_*}{L_{\text{train}}}-(f-1)\right)^{d/(d-2)},
$$

其中 $f\geq1$ 是配置中的 `factor`。这里的 $f$ 不能不加区别地当作上面的静态 $s$。当 $\ell\leq L_{\text{train}}$ 时，括号为 1，base 保持原值。[固定版本实现](https://github.com/huggingface/transformers/blob/v4.44.2/src/transformers/modeling_rope_utils.py#L105-L151)

动态调整还会影响 KV Cache：如果旧 key 已按旧频率旋转，新 query 却使用新频率，它们就不再遵循同一套相对位置公式。实现需要保持缓存与当前频率一致，例如保存旋转前的 key 再应用当前旋转，或正确重算缓存。不能只改一个 base 就忽略这件事。

仅重旋转旧 key 可以统一本层显式的位置项，却不保证与按新频率重跑整段上下文完全相同：深层缓存来自旧的隐藏状态，而这些状态也受前面各层的 attention 影响。若需要完整等价性，应在本次生成中固定频率方案，或在频率变化后重新执行相应前向。

## 4. 方案三：YaRN

YaRN（Yet another RoPE extensioN）把两件事组合起来：分段处理频率，以及调整 attention 分数的尺度。它并不是在 NTK 的新 base 上再叠加一次相同缩放。

### 4.1 先看训练区间里能转几圈

对原始频率定义：

$$
\lambda_i=\frac{2\pi}{\theta_i},
\qquad
r_i=\frac{L_{\text{train}}}{\lambda_i}
=\frac{L_{\text{train}}\theta_i}{2\pi}.
$$

$r_i$ 表示按训练长度计算的旋转圈数。圈数多的高频分量保留原频率；圈数少的低频分量使用 PI；中间部分混合两者。

以 $d=128,b=10000,L_{\text{train}}=4096$ 为例：

| 旋转对 $i$ | 波长 $\lambda_i$，单位为 token 距离 | 圈数 $r_i$ |
|---|---:|---:|
| 0 | 6.28 | 651.90 |
| 16 | 62.83 | 65.19 |
| 32 | 628.32 | 6.52 |
| 63 | 54410.14 | 0.0753 |

这些数值有助于理解设计思路；“高频负责局部、低频负责全局”不能理解成每个频率只承担一种功能。

### 4.2 论文的 ramp：对圈数做线性过渡

YaRN 论文 §3.2 用两个圈数阈值 $0<\alpha<\beta$ 定义保留原频率的权重：

$$
\gamma(r)=
\begin{cases}
0, & r\leq\alpha,\\
\dfrac{r-\alpha}{\beta-\alpha}, & \alpha<r<\beta,\\
1, & r\geq\beta.
\end{cases}
$$

再混合原频率与 PI 频率：

$$
\theta_i^{\text{YaRN}}
=\gamma(r_i)\theta_i+
\bigl(1-\gamma(r_i)\bigr)\frac{\theta_i}{s}.
$$

论文对 LLaMA 系列给出的经验值是 $\alpha=1,\beta=32$。因此：

- $r_i\leq1$，即 $\lambda_i\geq L_{\text{train}}$：完全使用 PI
- $r_i\geq32$，即 $\lambda_i\leq L_{\text{train}}/32$：保留原频率
- 两者之间：按圈数线性混合

注意，这既不是以 $L_{\text{train}}/2$ 为默认边界，也不是对波长 $\lambda_i$ 直接做线性插值。[论文定义](https://arxiv.org/html/2309.00071v2#S3.SS2)

### 4.3 实际代码的 ramp：对维度编号做线性过渡

作者发布的代码与 Transformers 4.44.2 使用另一种过渡方式：先把圈数阈值换算成维度编号，再在编号之间线性变化。**它与论文的圈数线性 ramp 方向一致，但中间数值不完全相同。**

由 $r=L_{\text{train}}b^{-2i/d}/(2\pi)$ 解出：

$$
i(r)=\frac{d}{2\ln b}\ln\left(\frac{L_{\text{train}}}{2\pi r}\right).
$$

实现取 `beta_fast` 为高圈数阈值、`beta_slow` 为低圈数阈值，默认分别为 32 与 1。这两个代码字段不要与论文的 $\alpha,\beta$ 同名混用。对本文参数，端点为：

$$
\left\lfloor i(32)\right\rfloor=20,
\qquad
\left\lceil i(1)\right\rceil=46.
$$

在这一有效范围内，令：

$$
w_i=\operatorname{clip}\left(\frac{i-20}{46-20},0,1\right),
\qquad
\theta_i^{\text{impl}}=(1-w_i)\theta_i+w_i\frac{\theta_i}{s}.
$$

这里 $w_i$ 是 PI 的权重，方向与上一节的 $\gamma$ 相反。`clip` 表示把数值限制在 0 到 1 之间。实际实现还处理端点裁剪与重合的情况，不应只复制本例的 20、46。

例如 $i=32,s=8$：

- 论文圈数 ramp 的频率比例约为 $0.28078$
- 上面维度 ramp 的频率比例约为 $0.59615$

差异来自过渡变量不同，不是浮点误差。要复现某个 checkpoint，应遵循它实际采用的实现，而不是把两套公式称为逐项等价。[作者实现](https://github.com/jquesnelle/yarn/blob/master/scaled_rope/LlamaYaRNScaledRotaryEmbedding.py)、[Transformers 4.44.2 实现](https://github.com/huggingface/transformers/blob/v4.44.2/src/transformers/modeling_rope_utils.py#L153-L222)

### 4.4 Attention Temperature：这里是降低温度

先约定 $Q',K'$ 是使用新频率旋转后的 query、key 矩阵，每行对应一个 token，形状为 $T\times d$；$V$ 的形状为 $T\times d_v$。这里 $T$ 是序列长度，$d_v$ 是 value 的维数。单个 head 的 attention 可写为：

$$
\operatorname{Attention}
=\operatorname{softmax}\left(\frac{Q'K'^\top}{t\sqrt d}+M\right)V.
$$

softmax 沿 key 方向计算；$M$ 是可见位置为 0、屏蔽位置为 $-\infty$ 的 mask。温度 $t>0$，越小会把**同一组分数**的差异放大。

YaRN 对 LLaMA/Llama 2 给出的经验关系是：

$$
a(s)=1+0.1\ln s,
\qquad
\boxed{t=\frac{1}{a(s)^2}}.
$$

这里用 $a$ 避免与位置 $m$ 重名；它就是作者代码的 `mscale`，有些说明会把这个增益记成 $m$，写作 $m=1+0.1\ln s$。`ln` 是自然对数。这个经验式适用于这里讨论的扩展范围 $s\geq1$，不是任意缩短上下文的公式，也不是从随机向量方差严格推导出的普遍定律。[论文 §3.4 与附录 A.2](https://arxiv.org/html/2309.00071v2#S3.SS4)

当 $s=8$：

$$
a\approx1.20794,\qquad a^2\approx1.45913,\qquad t\approx0.68534.
$$

所以它会让同一组 attention 分数对应的分布更尖锐，而不是把温度提高到 1 到 2。$s=1$ 时，$a=t=1$，恢复原来的尺度。

实现上不必改 softmax：把用于旋转的 `cos` 和 `sin` 都乘以 $a$，就让整个旋转后的 query、key 各乘以 $a$，从而：

$$
(aQ')(aK')^\top=a^2Q'K'^\top=\frac{Q'K'^\top}{t}.
$$

这一步有意改变向量长度；频率缩放本身仍是保长度的旋转。两者不要混淆。上式也用到了本文“整个 head 都参与旋转”的前提；只旋转部分维度时，不能直接声称整个内积都乘了同一个系数。

### 4.5 不要用错误的“期望内积”来证明温度公式

对任意随机 query、key，并不存在无需条件的 $\mathbb E[\text{内积}]=\sum_i\cos((m-n)\theta_i)$。

例如，若旋转前的 $\mathbf q,\mathbf k$ 相互独立且均值为零，旋转后的期望内积仍为零。一个能得到余弦求和的**充分条件**是：

$$
\mathbb E[\mathbf q\mathbf k^\top]=\sigma^2 I_d.
$$

其中 $\sigma^2\geq0$ 是对应坐标的共同二阶矩，$I_d$ 是 $d$ 维单位矩阵。在此条件下，每个二维对有两个坐标，因而：

$$
\mathbb E\!\left[(R_m\mathbf q)^\top(R_n\mathbf k)\right]
=2\sigma^2\sum_{i=0}^{d/2-1}\cos((m-n)\theta_i).
$$

$R_m,R_n$ 是对应位置的 RoPE 旋转矩阵。令 $\mathbf q=\mathbf k$ 且各坐标二阶矩满足上述关系，是充分情形之一，**不是必要条件**。

即使条件满足，这个有限余弦和仍会振荡，不保证随距离单调下降；真实 attention 还取决于内容、其他 key 与 softmax 归一化。因此不能从“相对位置”直接推出“每个远 token 的注意力都更小”，更不能用它替代 YaRN 温度参数的实验依据。

## 5. 实验对比：只比较有明确口径的数据

下面摘录 YaRN 论文 v2 表 1 的部分列。实验使用 10 篇至少有 128K token 的 Proof-pile 文档，采用步长为 256 的滑动窗口方法计算 perplexity（PPL，越低越好）。三个模型的扩展训练窗口都是 8K；训练 token 数并不相同。[论文表 1](https://arxiv.org/html/2309.00071v2#S4.T1)

| 方法 | 训练 token 数 | 2048 上下文 PPL | 4096 上下文 PPL | 8192 上下文 PPL | 10240 上下文 PPL |
|---|---:|---:|---:|---:|---:|
| PI，$s=2$ | 1B | 3.92 | 3.51 | 3.34 | 8.07 |
| NTK，base 为 20000 | 1B | 4.20 | 3.75 | 3.59 | 6.24 |
| YaRN，$s=2$ | 400M | 3.91 | 3.50 | 3.35 | 6.04 |

这组结果表明，YaRN 在这次实验中使用较少训练 token，取得了与 PI 接近的窗口内 PPL。它不是所有列都最好，例如在 8192 长度上，PI 的 3.34 略低于 YaRN 的 3.35。

这里的数字既不是 LongBench 分数，也不是 8 倍扩展的统一排名。PPL、长文检索与复杂推理衡量不同能力；没有指定模型、训练预算和评测流程，不能笼统说某种缩放方法始终最好。关于 PPL 的计算与比较条件，可接着读 [Perplexity](/posts/2024-04-10-perplexity/)。

## 6. 实现与使用边界

### 6.1 可独立运行的公式检查

下面只依赖 Python 标准库，返回频率与幅度增益，不加载模型。`ramp="cycles"` 对应论文公式；`ramp="dimension"` 对应前述固定版本的默认维度 ramp 计算方式。变量 `dim` 指参与旋转的维数。

```python
import math


def yarn_parameters(dim=128, base=10000.0, scale=8.0,
                    train_length=4096, beta_slow=1.0,
                    beta_fast=32.0, ramp="dimension"):
    if not isinstance(dim, int) or dim <= 2 or dim % 2:
        raise ValueError("dim must be an even integer greater than 2")
    if not math.isfinite(base) or base <= 1:
        raise ValueError("base must be finite and greater than 1")
    if not math.isfinite(scale) or scale < 1:
        raise ValueError("scale must be finite and at least 1")
    if not math.isfinite(train_length) or train_length <= 0:
        raise ValueError("train_length must be positive and finite")
    if not (math.isfinite(beta_slow) and math.isfinite(beta_fast)
            and 0 < beta_slow < beta_fast):
        raise ValueError("require 0 < beta_slow < beta_fast")
    if ramp not in {"cycles", "dimension"}:
        raise ValueError("unknown ramp")

    def clip01(value):
        return max(0.0, min(1.0, value))

    def boundary(rotations):
        return dim * math.log(train_length / (2 * math.pi * rotations)) / (
            2 * math.log(base)
        )

    # Match the boundary convention of Transformers 4.44.2.
    lower = max(0, math.floor(boundary(beta_fast)))
    upper = min(dim - 1, math.ceil(boundary(beta_slow)))
    width = upper - lower if upper != lower else 0.001

    frequencies = []
    for i in range(dim // 2):
        theta = base ** (-2 * i / dim)
        if ramp == "cycles":
            rotations = train_length * theta / (2 * math.pi)
            keep = clip01((rotations - beta_slow) / (beta_fast - beta_slow))
        else:
            keep = 1 - clip01((i - lower) / width)
        frequencies.append(theta * (keep + (1 - keep) / scale))

    amplitude = 1 + 0.1 * math.log(scale)
    return frequencies, amplitude


frequencies, amplitude = yarn_parameters()
assert len(frequencies) == 64
assert math.isclose(frequencies[0], 1.0)
assert math.isclose(frequencies[-1], 10000 ** (-126 / 128) / 8)
assert math.isclose(1 / amplitude**2, 0.6853403266295983)
```

这段程序检查数学关系，不处理张量布局、KV Cache 或模型质量。维度 ramp 的边界裁剪刻意对齐指定实现；不要把它当作所有 YaRN 版本的统一定义。极端参数也需要单独验证，不能只因为函数算出了数字就认为配置适合模型。

### 6.2 固定版本的 Hugging Face 示例

以下示例针对 **Transformers 4.44.2 的 Llama 实现**，仅构造 RoPE 层，用三个位置检查输出形状，不下载模型，也不声称已经完成长上下文训练。`LlamaRotaryEmbedding` 是内部类，升级版本后应重新核对。

```python
# Requires torch and transformers==4.44.2.
import torch
import transformers
from transformers import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

assert transformers.__version__ == "4.44.2"

config = LlamaConfig(
    hidden_size=4096,
    num_attention_heads=32,  # head dimension = 128
    max_position_embeddings=4096,
    rope_theta=10000.0,
    rope_scaling={
        "rope_type": "yarn",
        "factor": 8.0,
        "beta_fast": 32.0,
        "beta_slow": 1.0,
    },
)
rotary = LlamaRotaryEmbedding(config=config, device="cpu")
x = torch.zeros(1, 3, 128)
position_ids = torch.tensor([[0, 1, 32767]], dtype=torch.long)
cos, sin = rotary(x, position_ids)
assert cos.shape == sin.shape == (1, 3, 128)
```

这里保留 `max_position_embeddings=4096`，因为 **4.44.2 的 YaRN 函数直接用这个字段计算原训练长度对应的 ramp**；不能同时把它改成 32768，再期待另一个字段自动纠正。在这一版本的 YaRN 路径里，`original_max_position_embeddings` 不是用于上述计算的替代字段。[配置与计算源码](https://github.com/huggingface/transformers/blob/v4.44.2/src/transformers/modeling_rope_utils.py#L177-L222)

该版本返回的 `attention_factor` 是上文的幅度 $a$，随后乘在 `cos`、`sin` 上；它不是温度 $t$，也不是直接把 logits 乘以 $a$。[Llama 调用位置](https://github.com/huggingface/transformers/blob/v4.44.2/src/transformers/models/llama/modeling_llama.py#L185-L207)

实际使用时，应优先遵循目标 checkpoint 的模型卡、配置与兼容版本，避免覆盖已经训练好的 RoPE 设置。位置 32767 能算出旋转值，只证明公式可计算；有效上下文长度、生成质量、KV Cache 显存以及推理服务的长度上限，都还需要分别确认。

上面的标准库示例可用于 CPU 数值自检；Hugging Face 示例按固定版本源码核对，不作为真实模型或 GPU 的运行实验结果。

## 7. 总结

| 方法 | 对频率做了什么 | 需要记住的边界 |
|---|---|---|
| PI | 所有频率除以 $s$ | 压缩位置范围，也改变短距离角度差 |
| 静态 NTK-aware | 改 base，让最高频不变、最低频除以 $s$ | 其他频率仍变动，部分频率仍外推 |
| Dynamic NTK | 随当前长度更新 NTK 参数 | 具体公式依赖实现，需处理 KV Cache 一致性 |
| YaRN | 高频保留、低频 PI、中间混合，再放大 query/key | 论文 ramp 与代码 ramp 要区分，温度是经验设置 |

位置表示的变化需要适合模型已经学到的规律，并用实际任务验证，并非改得越细就一定越好。读到一个长上下文配置时，先问三件事：原训练长度是多少，频率实际怎样计算，模型在什么条件下验证过这个长度。

### 参考资料

1. Chen et al. [Extending Context Window of Large Language Models via Positional Interpolation](https://arxiv.org/abs/2306.15595)，2023
2. Peng et al. [YaRN: Efficient Context Window Extension of Large Language Models，v2](https://arxiv.org/html/2309.00071v2)，2023；本文对应 §3、附录 A.1/A.2 与表 1
3. [YaRN 作者代码](https://github.com/jquesnelle/yarn/blob/master/scaled_rope/LlamaYaRNScaledRotaryEmbedding.py)，用于核对维度 ramp 与 `mscale`；分支可能更新，固定版本对照见下一项
4. [Transformers v4.44.2：modeling_rope_utils.py](https://github.com/huggingface/transformers/blob/v4.44.2/src/transformers/modeling_rope_utils.py)，用于核对 `linear`、`dynamic`、`yarn` 的区别
5. [Transformers v4.44.2：Llama 实现](https://github.com/huggingface/transformers/blob/v4.44.2/src/transformers/models/llama/modeling_llama.py)，用于核对 `cos`、`sin` 的幅度缩放
6. Su et al. [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)，RoPE 原论文
