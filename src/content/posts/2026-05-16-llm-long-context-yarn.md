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

上一篇文章我们推导了 [RoPE: 旋转位置编码](/posts/2026-05-16-llm-rope-rotary-position-embedding/): 用旋转矩阵给每个位置编码, 让 attention 的内积只依赖相对位置.

但 RoPE 有一个棘手的问题: 模型在训练时只见过 $[0, L_{\text{train}})$ 范围内的位置. 推理时突然要处理 $m \gg L_{\text{train}}$ 的位置——即使 RoPE 的公式在数学上可以计算任意 $m$, **模型"没见过"这么大位置上的频率组合**, 效果会断崖式下跌.

这篇文章就从 RoPE 的频率公式出发, 一步步推导各个改进方案: 从最朴素的 **Position Interpolation**, 到 **NTK-aware scaling**, 再到集大成者的 **YaRN**.

最终你会理解: 这些方法不是在"发明"新东西, 而是在回答一个问题——**当模型需要处理从未见过的长位置时, 怎么把已有的 RoPE 频率知识迁移过去?**

---

## 1. 先定位问题: RoPE 在长位置为什么不行

RoPE 中, 第 $i$ 个维度对的旋转频率是:

$$
\theta_i = 10000^{-2i/d}, \quad i = 0, 1, ..., d/2 - 1
$$

位置 $m$ 的旋转角度是 $m\theta_i$.

**训练时**: 模型只见过 $m \in [0, L_{\text{train}})$ 范围内的 $m\theta_i$ 值. 这些值覆盖了所有 $\theta_i$ 的某个范围.

**推理时**: 当 $m > L_{\text{train}}$, $m\theta_i$ 超出了训练时见过的范围. 尤其是对于高频维度 ($i$ 小, $\theta_i$ 大), 在 $L_{\text{train}}$ 内可能已经转了很多圈; 而对于低频维度 ($i$ 接近 $d/2$, $\theta_i$ 很小), 在 $L_{\text{train}}$ 内可能才转了不到半圈.

模型在训练时学到的是一种**频率组合的"分布"**——当输入第 $m$ 个 token 时, 各维度对以不同的旋转角度协同工作. 超出训练范围后, 这些角度组合不再符合训练时的分布, 模型就"懵"了.

这个认识很重要——问题不在于 RoPE 的数学, 而在于**分布外泛化**. 所以所有改进方案的核心都是: 如何把长位置的旋转角度"拉回"训练时的分布内, 同时尽量保留位置信息.

---

## 2. 方案一: Position Interpolation (PI) — 简单但粗暴

### 2.1 核心思路

PI 的想法非常直接: **既然长位置的 $m\theta_i$ 没见过, 那就把它缩回训练时的范围内**.

做法: 把位置 $m$ 映射到 $m' = m \times \frac{L_{\text{train}}}{L_{\text{infer}}}$.

也就是说, 旋转角度从 $m\theta_i$ 变成:

$$
m'\theta_i = \frac{L_{\text{train}}}{L_{\text{infer}}} \cdot m \cdot \theta_i
$$

例如训练 4K, 推理 32K: 位置 16,000 被当作位置 2,000 来计算旋转. 这意味着 16,000 位置的向量和训练时 2,000 位置的向量**经历完全相同的旋转**.

好处: 所有 $m\theta_i$ 值都落在训练范围内, 模型不会"没见过".

### 2.2 数学上分析 PI 的问题

用 $s = L_{\text{infer}} / L_{\text{train}}$ 表示扩展比 (scale). PI 的等效频率是:

$$
\theta_i^{\text{PI}} = \frac{\theta_i}{s}
$$

相邻位置的角度差从 $\theta_i$ 变成了 $\theta_i / s$.

来算一下这会造成什么后果. 对于高频维度 ($i=0$), $\theta_0 = 1.0$, $d=128$:

原始相邻位置差: $\theta_0 = 1.0$ 弧度, 约 $57.3^\circ$
PI 后相邻位置差 ($s=8$): $\theta_0 / 8 = 0.125$ 弧度, 约 $7.2^\circ$

原来位置 $m$ 和 $m+1$ 的向量方向相差 $57^\circ$, 很容易区分. PI 后只差 $7^\circ$, 几乎重叠——**高频分辨率严重下降**.

对于低频维度 ($i=63$), $\theta_{63} = 10000^{-126/128} \approx 0.0001$:

原始相邻位置差: $\approx 0.0057^\circ$ — 本来相邻位置就很难区分
PI 后: $\approx 0.0007^\circ$ — 更分不清了

但低频维度的作用本来就不是区分相邻位置, 而是感知**大范围距离**. 所以低频损失一些分辨率问题不大. 真正致命的是高频分辨率丢失——它破坏了模型对精细位置关系的建模能力.

### 2.3 PI 的结论

PI 的效果其实还不错——经过几千步微调 (fine-tuning), 可以很好地扩展到 8 倍长度. 但它的问题也明显: **高频信息被均匀压缩, 短距离的区分度下降**. 如果你既想在短序列上保持原有精度, 又想扩展到长序列, PI 不是最优选择.

---

## 3. 方案二: NTK-aware — 保留高频分辨率

### 3.1 直觉

NTK-aware 的直觉和 PI 相反: **高频维度携带精细位置信息, 应该保持分辨率; 低频维度负责大范围感知, 可以拉伸**.

怎么实现? 不缩放位置 $m$, 而是调整频率 $\theta_i$ 本身. 核心是修改 RoPE 的 base 值.

### 3.2 从 base 修改出发

回顾 RoPE 的频率公式:

$$
\theta_i = \text{base}^{-2i/d}
$$

如果我们把 base 从 10000 换成 $\text{base}' > \text{base}$, 会发生什么?

对于高频 ($i$ 小): $\theta_i \approx \text{base}'^{-2i/d}$ — 大的 base 使得 $\theta_i$ 变小(因为指数是负的), 高频频率降低.

不对, 仔细算. $\theta_i = \text{base}^{-2i/d}$. 当 $i$ 很小时($2i/d \approx 0$), $\text{base}^{-2i/d} \approx 1$ 对 base 的变化不敏感. 当 $i$ 接近 $d/2$ 时 ($2i/d \approx 1$), $\theta_{d/2} = \text{base}^{-1}$, 增大 base 会显著降低低频频率.

所以: **增大 base, 高频几乎不变, 低频被压低**. 这正是我们想要的!

NTK-aware 选择:

$$
\text{base}' = \text{base} \times \alpha
$$

其中 $\alpha$ 是跟扩展比相关的值. 推荐的 $\alpha$ 选择:

$$
\alpha = \left(\frac{L_{\text{infer}}}{L_{\text{train}}}\right)^{d/(d-2)}
$$

这个公式的推导思路是: 让**最低频维度**的波长远大于训练长度, 从而把长位置的旋转角度"拉伸"回训练时的范围. 推导如下:

最低频维度 ($i = d/2 - 1$) 的原始波长:

$$
\lambda_{\min} = \frac{2\pi}{\theta_{d/2-1}} = 2\pi \times \text{base}^{(d-2)/d}
$$

我们希望 $\lambda_{\min}$ 拉伸到 $L_{\text{infer}}$ 量级, 所以选择 $\text{base}'$ 使新波长:

$$
\lambda_{\min}' = L_{\text{infer}} \implies \text{base}'^{(d-2)/d} \propto L_{\text{infer}}
$$

解出 $\text{base}' / \text{base} \propto (L_{\text{infer}} / L_{\text{train}})^{d/(d-2)}$.

实际使用中, 可以简化为 $\text{base}' = \text{base} \times s$ (其中 $s = L_{\text{infer}}/L_{\text{train}}$), 或者用经验值 $\text{base}' = \text{base} \times s^{1.2}$ 之类的. 具体哪个最好需要实验验证.

### 3.3 NTK-aware 的逐频率视角

另一种等价的理解方式: NTK-aware 等价于逐频率使用不同的缩放因子.

原始频率 $\theta_i$, NTK 后的频率 $\theta_i'$:

$$
\theta_i' = \text{base}'^{-2i/d} = (\text{base} \cdot \alpha)^{-2i/d} = \text{base}^{-2i/d} \cdot \alpha^{-2i/d}
$$

所以 $\theta_i' = \theta_i \cdot \alpha^{-2i/d}$.

相比 PI 对所有频率乘 $1/s$, NTK-aware 的缩放因子 $\alpha^{-2i/d}$ 是**频率相关的**: 高频 ($i$ 小) 缩放因子接近 1 (几乎不变), 低频 ($i$ 大) 缩放因子显著小于 1 (频率降低, 波长拉长).

这就实现了"高频保留, 低频拉伸"的目标.

### 3.4 NTK-aware 的效果

**不微调也能用!** 这是 NTK-aware 的最大优点——把 base 改大后, 模型在短上下文上的表现几乎不受影响 (因为高频没动), 在长上下文上的表现有显著提升.

原因: 高频维度决定了模型对**相邻位置**的区分能力——只要这个能力保住了, 模型在短序列上的输出就不会大变. 低频维度被拉伸后, 模型虽然可能在长距离依赖上"感觉"不太准, 但至少不会输出乱码.

---

## 4. 方案三: YaRN — 精细化处理

YaRN (Yet another RoPE extensioN) 在 NTK-aware 的基础上做了两个关键改进.

### 4.1 改进一: NTK-by-parts (逐维度差异化处理)

NTK-aware 对所有频率使用了统一的 base 缩放, 这仍然不够精细. YaRN 提出: **应该根据每个维度的波长, 来决定它的缩放方式**.

一个维度的波长:

$$
\lambda_i = \frac{2\pi}{\theta_i} = 2\pi \cdot \text{base}^{2i/d}
$$

对于 $d=128$, base=10000, 各维度的波长范围:

- $i=0$: $\lambda_0 \approx 2\pi \cdot 1 = 6.28$ — 每约 6 个 token 旋转一圈
- $i=32$: $\lambda_{32} \approx 2\pi \cdot 10000^{64/128} = 2\pi \cdot 100 = 628$
- $i=63$: $\lambda_{63} \approx 2\pi \cdot 10000^{126/128} \approx 2\pi \cdot 9341 \approx 58680$

现在来看跟训练长度的关系. 假设 $L_{\text{train}} = 4096$:

- 如果 $\lambda_i \ll L_{\text{train}}$: 维度在训练范围内旋转了很多圈, 携带精细位置信息 → **不缩放**
- 如果 $\lambda_i \gg L_{\text{train}}$: 维度在训练范围内才转了不到半圈, 携带大范围信息 → **用 PI 方式缩放**
- 如果 $\lambda_i \approx L_{\text{train}}$: 介于两者之间 → **平滑过渡**

YaRN 的决策边界:

$$
r_i = \begin{cases}
1, & \lambda_i \leq \frac{L_{\text{train}}}{2} \\
\frac{1}{s}, & \lambda_i \geq L_{\text{train}} \\
1 - (1 - \frac{1}{s}) \cdot \frac{\lambda_i - L_{\text{train}}/2}{L_{\text{train}}/2}, & \text{otherwise}
\end{cases}
$$

其中 $r_i$ 是对 $\theta_i$ 的缩放因子: $\theta_i' = \theta_i \cdot r_i$.

**解释:**

- 当 $\lambda_i \leq L_{\text{train}}/2$: 波长短, 旋转快, 高频 → $r_i = 1$, 完全不缩放
- 当 $\lambda_i \geq L_{\text{train}}$: 波长长, 旋转慢, 低频 → $r_i = 1/s$, 完全用 PI 方式 (等效于位置压缩)
- 中间: $r_i$ 从 1 线性下降到 $1/s$, 平滑过渡

这个"波长 vs 训练长度"的判断标准非常巧妙——它从**物理意义**(旋转一圈需要多少 token)出发, 而不是从**编号**(维度 $i$ 的序号)出发. 同样的维度索引在不同的模型维度 $d$ 下可能需要不同的处理, 但波长是绝对的.

### 4.2 改进二: Attention Temperature 调整

这是 YaRN 一个容易被忽略但非常重要的改进.

**问题**: 当改变频率 (不管是用 PI 还是 NTK) 后, query 和 key 的内积分布会发生变化.

回顾 RoPE 文章中的推导, 对于随机向量 $\mathbf{q}, \mathbf{k}$, RoPE 编码后内积的期望是:

$$
\mathbb{E}[\langle f_q(\mathbf{q}, m), f_k(\mathbf{k}, n) \rangle] = \sum_{i=0}^{d/2-1} \cos((m-n)\theta_i)
$$

当频率 $\theta_i$ 被缩放后, 这个求和的值会发生变化. 具体来说:

- **原始**: $\sum \cos((m-n)\theta_i)$
- **PI 后**: $\sum \cos((m-n)\theta_i / s)$ — 因为位置被压缩了, 所以相对差对应的角度也变了
- **NTK 后**: $\sum \cos((m-n)\theta_i \cdot \alpha^{-2i/d})$ — 每个频率的缩放不同

这个变化导致: **即使对相同的相对距离 $(m-n)$, 内积的绝对大小也变了**. 如果内积整体变小了, softmax 后的 attention 分布就会更"平坦" (温度变高); 如果内积变大了, attention 分布就更"尖锐".

YaRN 的解决方案: 在 attention softmax 中引入一个温度系数 $t$:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d} \cdot t}\right)
$$

$t$ 的选择: YaRN 论文通过分析内积分布的方差变化, 给出 $t \approx \sqrt{1 + \frac{\ln s}{\ln (d/2)}}$ 的参考值. 在实践中, $t$ 通常在 $1.0$ 到 $2.0$ 之间, 需要根据具体模型和扩展比例来调.

### 4.3 YaRN 的完整算法

总结一下 YaRN 的完整流程:

```
输入: 原始 RoPE 频率 θ_i, 训练长度 L_train, 扩展比 s = L_infer / L_train

1. 计算每个维度的波长: λ_i = 2π / θ_i

2. 计算逐维度缩放因子 r_i:
   if λ_i ≤ L_train/2:     r_i = 1
   elif λ_i ≥ L_train:     r_i = 1/s
   else:                   r_i = 1 - (1 - 1/s) * (λ_i - L_train/2) / (L_train/2)

3. 应用缩放: θ_i' = θ_i · r_i

4. 计算 attention 温度系数 t (经验值, 可调)

5. 使用 θ_i' 做 RoPE, 用 t 调节 attention softmax
```

### 4.4 YaRN 的效果为什么更好

| 方案 | 高频 (i=0) | 中频 (i=16) | 低频 (i=63) | 温度调整 |
|------|-----------|------------|------------|---------|
| 直接外推 | 原样 | 原样 | 原样 | 无 |
| PI | 全部 $/s$ | 全部 $/s$ | 全部 $/s$ | 无 |
| NTK | 几乎不变 | 略微降低 | 大幅降低 | 无 |
| YaRN | 完全不变 | 平滑过渡 | PI 缩放 | ✅ |

YaRN 的"精细"之处在于: 它让每个频率的缩放决策有了**物理依据**(波长), 而不是统一的数学公式.

---

## 5. 实验对比

在 LongBench 上的典型结果 (来自 YaRN 论文, 各方法经过微调):

| 方法 | 扩展 8x 后 LongBench 得分 | 短序列质量是否受影响 | 是否需要微调 |
|------|--------------------------|-------------------|------------|
| 直接外推 (无处理) | ~20 | 是 (很差) | 否 (但效果差) |
| PI | ~37 | 轻微下降 | 需微调 |
| NTK-aware | ~34 | 几乎不变 | 可不用微调 |
| YaRN | ~41 | 几乎不变 | 少量微调即可 |

YaRN 是综合效果最好的方案. 如今主流的大模型 (LLaMA-3.1 128K, Mistral 32K, Qwen2.5 128K) 背后的长上下文扩展方案, 基本都基于类似 YaRN 的思路.

---

## 6. HuggingFace 使用示例

```python
from transformers import AutoModelForCausalLM, AutoConfig

model_name = "meta-llama/Llama-2-7b-hf"
config = AutoConfig.from_pretrained(model_name)

# 启用 YaRN
config.rope_scaling = {
    "type": "yarn",
    "factor": 8.0,                              # 4K → 32K
    "original_max_position_embeddings": 4096,
}

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    config=config,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# 现在可以处理 32K 序列
outputs = model.generate(inputs, max_new_tokens=256)
```

手动实现 (核心部分):

```python
def yarn_frequencies(dim, seq_len, base=10000, scale=8.0, L_train=4096):
    """计算 YaRN 的 RoPE 频率"""
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    
    # 波长
    wavelengths = 2 * torch.pi / inv_freq
    
    # 缩放因子: 按波长分配
    ramp = torch.clamp(
        (wavelengths - L_train/2) / (L_train - L_train/2),
        min=0.0, max=1.0
    )
    r = 1 - ramp * (1 - 1/scale)
    
    return inv_freq / r  # 频率 = 1/scale → 缩放
```

---

## 7. 总结

回头看这个演进过程, 每一步都在解决上一步的问题:

| 步骤 | 方法 | 核心洞察 | 解决的问题 |
|------|------|---------|-----------|
| 1 | 直接外推 | — | — |
| 2 | PI | 把长位置缩回训练范围 | 解决了分布外问题 |
| 3 | NTK-aware | 高频分辨率和低频范围不同 | PI 的高频分辨率损失 |
| 4 | YaRN | 波长决定缩放策略 + 温度修正 | NTK 的粗粒度问题 + 分布偏移 |

**核心思想**: 不要把 RoPE 的频率当成固定的, 而是根据目标任务(上下文长度)来调整. 调整的粒度越细(逐维度 vs 全局), 效果越好. 调整后别忘了修正 attention 的热度——因为频率变了, attention 的分布也会变.

---

### 参考资料

1. Chen et al., Extending Context Window of Large Language Models via Positional Interpolation. 2023. [arXiv:2306.15595](https://arxiv.org/abs/2306.15595) — **PI**
2. Peng et al., YaRN: Efficient Context Window Extension of Large Language Models. 2023. [arXiv:2309.00071](https://arxiv.org/abs/2309.00071) — **YaRN**
3. NTK-aware RoPE scaling. Reddit r/LocalLLaMA by u/emozilla (Jeffrey Quesnelle). [Link](https://www.reddit.com/r/LocalLLaMA/comments/14lz7j9/ntkaware_scaled_rope_allows_llama_models_to_have/)
4. Su et al., RoFormer: Enhanced Transformer with Rotary Position Embedding. Neurocomputing 2022. [arXiv:2104.09864](https://arxiv.org/abs/2104.09864) — **RoPE 原文**
5. Bai et al., LongBench: A Bilingual, Multitask Benchmark for Long Context Understanding. 2023. [arXiv:2308.14508](https://arxiv.org/abs/2308.14508)
