---
title: LLM 训练与推理优化（三）— 量化深入：从最小化误差到 GPTQ/AWQ
published: 2026-06-14
description: 不直接抛公式，而是从"我想把 BF16 压成 INT4，又不想损失太多精度"这个目标出发，一步步推出 scale、zero_point、GPTQ 误差补偿、AWQ 通道保护。
category: Deep Learning
tags:
- llm
- memory
- llm-memory-optimization-series
- quantization
- gptq
- awq
draft: false
---

> 这是「LLM 训练与推理优化系列」第 3 篇。前置知识——bit/byte/精度——已在 [第 1 篇 — 显存基础](/posts/2026-06-12-llm-mem-opt-1-fundamentals/) 讲过。本篇直接从"量化目标"切入。

## 0. 第 2 篇省下的 12 GB 是怎么来的

[第 2 篇](/posts/2026-06-13-llm-mem-opt-2-inference/) 第 5.1 节里我们说：「BF16 → INT4 把 8B 模型的权重从 16 GB 压到 4 GB」。

但 INT4 只有 16 个量化级别，BF16 能表示几亿个数。**怎么把几亿个浮点数映射到 16 个整数，又不让模型崩掉？**

这就是量化要解决的问题。

它和一般的数据压缩（比如 zip）不一样：我们不需要精确还原每个权重，**只需要最终模型输出的质量尽量不变**。这就给了我们巨大的操作空间。

## 1. 设定目标

假设有一个权重矩阵 $W \in \mathbb{R}^{m \times n}$，原始 BF16，要转成 INT4。整个流程是：

$$
W \xrightarrow{\text{量化}} W_q \in \{-8, \dots, 7\}^{m \times n} \xrightarrow{\text{反量化}} \hat{W}
$$

推理时算的是 $\hat{W}x$ 而不是 $Wx$，所以误差：

$$
\text{error} = \|Wx - \hat{W}x\|
$$

**量化的核心问题**：找一个映射 $Q: \mathbb{R} \to \{-8, \dots, 7\}$ 让反量化后的 $\hat{W}$ 与 $W$ 的误差最小（在某种合适的度量下）。

## 2. 最简单的尝试：直接四舍五入

最直觉的想法：把每个浮点数四舍五入到最近的整数。

$$0.3 \to 0, \quad 1.7 \to 2, \quad -0.8 \to -1$$

但马上就出问题——大部分模型权重的取值范围在 $[-2, 2]$ 左右（实际不同层差异很大），而 INT4 范围是 $[-8, 7]$。这就像用 16 米的尺子去量 2 米的东西，大部分刻度没用上。四舍五入后几乎所有权重变成 0 或 ±1，信息全丢。

显然要先 **缩放**：把数值范围拉到匹配整数范围，再四舍五入。这就是 **scale** 的由来。

## 3. 对称量化：引入 scale

先考虑最简单的情况——数据分布关于 0 对称（权重通常如此）。

### 3.1 推导 scale

把 $[-a, a]$ 范围内的浮点数映射到 $[-Q_n, Q_p]$ 范围内的整数。对称 INT8 通常用 $Q_n = Q_p = 127$，跳过 $-128$ 以保证关于 0 严格对称（普通有符号 INT8 是 $[-128, 127]$，丢掉一个量化级换取对称性）。

令 $\operatorname{clip}(u, a, b)=\min(\max(u,a),b)$。量化函数应当显式限制到整数表示范围：

$$
x_q = \operatorname{clip}\!\left(\mathrm{round}\!\left(\frac{x}{s}\right), -Q_n, Q_p\right)
$$

反量化：

$$
\hat{x} = x_q \times s
$$

均方误差（MSE）：

$$
\mathrm{MSE}(s) = \frac{1}{N}\sum_{i=1}^N \left(x_i - s \cdot \mathrm{round}\!\left(\frac{x_i}{s}\right)\right)^2
$$

这没有完美的闭式解。一个常见启发式是：**让 scale 刚好覆盖最大幅值**。

$$
s = \frac{\max(|x|)}{Q_p}
$$

当 $\max(|x|)=0$ 时张量全为 0，应直接编码为 0（或使用约定的非零 scale），避免除以 0。

### 3.2 为什么这个 scale 是好的

如果 $s$ 太大 → $x/s$ 普遍很小 → 四舍五入后许多不同的 $x$ 映射到同一整数 → **量化颗粒度太粗**；

如果 $s$ 太小 → 部分数据超出 $[-Q_n s, Q_p s]$ → 发生 **截断（clipping）** → 大值的信息直接丢失。

最优 scale 在两者之间取平衡。对接近均匀分布的数据，覆盖最大幅值就是近似最优。

但模型权重常有长尾：选一个稍小的 scale，让极少数离群点被截断，可能让中间大部分权重的精度更高、整体 MSE 更低。这是一般的量化校准取舍；是否采用裁剪阈值应由校准数据和目标质量指标决定，并不是 AWQ 的核心定义。

### 3.3 一个直观例子

权重 $[0.1, -0.3, 0.7, -0.9, 1.2]$，量化到 INT8（$Q_p=127$）：

$$
s = 1.2 / 127 \approx 0.00945
$$

| 原始值 | $x/s$ | round | 反量化 $\hat{x}$ | 误差 |
|--------|-------|-------|-----------------|------|
| 0.1   | 10.58  | 11   | 0.104  | +0.004 |
| -0.3  | -31.75 | -32  | -0.302 | -0.002 |
| 0.7   | 74.07  | 74   | 0.699  | -0.001 |
| -0.9  | -95.24 | -95  | -0.898 | +0.002 |
| 1.2   | 126.98 | 127  | 1.200  | 0.000 |

误差都在 $s/2 \approx 0.005$ 以内。大值的相对误差 < 1%，小值上的相对误差大一些（0.1 → 0.104，4%）但绝对值很小，对最终的 $Wx$ 贡献有限。

### 3.4 对称量化的局限

如果数据分布不对称（比如 ReLU 后的激活值全是正数 $[0, 5]$），对称量化用 $[0, 127]$ 表示正半轴，$[-127, 0)$ 完全浪费——精度预算等于砍掉一半。需要 **非对称量化** 来补救。

## 4. 非对称量化：引入 zero_point

### 4.1 推导

允许平移的映射。对无符号 $n$ bit 整数，$q\in[0,2^n-1]$：

量化：

$$
x_q = \operatorname{clip}\!\left(\mathrm{round}\!\left(\frac{x}{s}\right) + z, 0, 2^n-1\right)
$$

反量化：

$$
\hat{x} = (x_q - z) \times s
$$

把 $[x_{\min}, x_{\max}]$ 映射到 $[0, 2^n - 1]$（INT8 即 $[0, 255]$）：

$$
s = \frac{x_{\max} - x_{\min}}{2^n - 1}, \quad z = \operatorname{clip}\!\left(\mathrm{round}\!\left(-\frac{x_{\min}}{s}\right), 0, 2^n-1\right)
$$

直观推导：线性映射

$$
x_q = \frac{x - x_{\min}}{x_{\max} - x_{\min}} \times (2^n - 1)
$$

整理后就得到 $s$、$z$ 的表达式。代入 $x = 0$ 验证：未裁剪时 $x_q = -x_{\min}/s = z$，即整数 $z$ 对应原始的 0。若 $x_{\max}=x_{\min}$，张量为常数，不能除以 0；实现通常直接存常数或约定 $s=1$、$z=0$ 并在反量化时恢复常数。

### 4.2 选择对照表

| | 对称量化 | 非对称量化 |
|---|---------|----------|
| 参数 | 1 个 $s$ | 2 个 $s, z$ |
| 计算开销 | 高效（无额外减法） | 多一步 $-z$ |
| 适合分布 | 关于 0 对称（如权重） | 不对称（如 ReLU 后激活） |
| 硬件支持 | 几乎所有 | 部分硬件需特殊路径 |

**实践经验：权重用对称量化，激活用非对称量化。**

## 5. 量化误差从哪里来

写一下反量化后的值：

$$
\hat{x} = s \cdot \mathrm{round}\!\left(\frac{x}{s}\right) = x + s \cdot \underbrace{\left(\mathrm{round}\!\left(\frac{x}{s}\right) - \frac{x}{s}\right)}_{e \in [-0.5, 0.5]}
$$

若 $x/s$ 落在整数范围内、且使用最近舍入而未发生 clipping，则：

$$
\hat{x} = x + s \cdot e, \quad e \in [-0.5, 0.5]
$$

绝对量化误差最大 $0.5\,s$。若发生 clipping，误差还包含超出表示范围的截断项，可能大于 $0.5\,s$。这给我们两个事实：

1. **在未 clipping 的最近舍入区间内，scale 越大，误差上限越大**。范围大的张量量化损失更容易变大。
2. **绝对误差上限固定**，不依赖 $|x|$。

第 2 条藏着一个关键洞察：**小权重的相对误差比大权重高得多**。

但小权重对最终输出的贡献本来就小（$Wx$ 里 $W_{ij} x_j$ 越小贡献越小）。所以"小权重的相对误差大"不一定是大问题。**真正要保护的，是输出贡献大的权重**——也就是要么自身大，要么对应激活大的那些。

这就是 **AWQ** 的出发点。先放着，看完粒度再讲。

## 6. 量化粒度

上面的推导都是对一个张量整体算一个 $s$。但一个 LLM 几十上百层，每层每个矩阵的参数分布差别可能极大。**用一个 $s$ 覆盖全部**，颗粒度肯定太粗。

按多少参数共用一个 $s$ 划分粒度：

| 粒度 | 含义 | 精度 | 元数据开销 |
|------|------|------|-----------|
| **Per-tensor** | 整个矩阵一个 $s$ | 最差 | 极小 |
| **Per-channel** | 每行（输出通道）一个 $s$ | 中等 | 小 |
| **Per-group** | 每 $g$ 个连续参数一个 $s$（常见 $g=32, 64, 128$） | 通常更细 | 较大 |

为什么粒度细就准？信息论视角：每组单独算 $s$ → 每组按自己的实际分布优化精度预算，不被异常值拖累。

代价是元数据：每组多存一个 FP16 的 $s$。group size = 32 时，相当于额外 $16/32 = 0.5$ bit/参数。对 INT4 量化来说，存储开销从 4 bit/参数涨到 4.5 bit/参数，多 12.5%——但精度收益通常远大于这个开销。

> group size 是量化产物和推理内核共同的配置；128 很常见，但不是 GPTQ 或 AWQ 的统一默认值。

## 7. GPTQ：误差补偿式量化

到这里，我们都在 **独立量化每个参数**——一个一个 round，互不影响。GPTQ 的主张是：**量化某个参数时，把它产生的误差"传导"到尚未量化的参数上做补偿**，让总误差更小。

### 7.1 思想来源：Optimal Brain Surgeon (OBS)

经典的网络剪枝问题：删一个权重，如何调整其他权重让损失最小？

二阶泰勒展开（在最优解附近，一阶项为 0）：

$$
\Delta \mathcal{L} \approx \tfrac{1}{2}\, \delta^{\top} H \delta
$$

$H$ 是 Hessian。OBS 推出：移除权重 $w_q$ 的最优补偿量为

$$
\delta = -\frac{w_q}{[H^{-1}]_{qq}}\, [H^{-1}]_{:,q}, \qquad \Delta \mathcal{L} \approx \tfrac{1}{2}\, \frac{w_q^2}{[H^{-1}]_{qq}}
$$

OBQ（Optimal Brain Quantizer）把这个想法推广到 **量化**——把 $w_q$ "移除"换成"量化到 $\bar{w}_q$"，量化误差 $w_q - \bar{w}_q$ 替换上式中的 $w_q$。

但 OBQ 每量化一个参数要重算 Hessian 逆，对 LLM 完全不可行。

### 7.2 GPTQ 的工程优化

GPTQ 在算法不变的情况下做了三个工程优化：

1. **固定顺序**：不再贪心选「影响最小」的参数，而是按列从左到右量化——简单且 GPU 友好。
2. **懒惰更新**：批量量化多列后再统一更新 Hessian 信息，利用 GPU 矩阵乘的并行性。
3. **Cholesky 预分解**：对 $H^{-1}$ 做 Cholesky 分解 $H^{-1} = LL^{\top}$，预先计算下三角因子，后续误差补偿只用 $1/L_{qq}$ 等局部值，避免反复求逆。

GPTQ 的实际量化时间、峰值内存和质量取决于模型、校准数据、group size、实现和硬件；应报告本次量化的设置与目标评测结果，而不是套用固定百分比。

### 7.3 算法伪代码

```text
输入:
  权重矩阵 W (m×n), 校准数据 X (N×n), 量化位宽 b
1. 计算 Hessian: H = 2 X^T X  (n×n)
2. Cholesky 分解: L = chol(H^{-1})
3. for j = 1 to n:                  # 按列从左到右
     a. 量化第 j 列: W_q[:, j]
     b. 计算量化误差 err = W[:, j] - dequant(W_q[:, j])
     c. 把 err 按 L 的信息分摊到第 j+1...n 列
```

步骤 3c 是核心：**前面列的量化误差由后面列"吸收"**。最终未量化列的权重值会偏离原始 BF16 一些，但这种偏离正是为了补偿前面已经量化掉的那些。

### 7.4 GPTQ 的优缺点

| 优点 | 缺点 |
|------|------|
| 可在校准与任务评测通过时部署 INT4 | 需要校准数据与质量验证 |
| 一次量化产物可永久部署 | 量化过程慢（小时量级） |
| 工业级实现成熟（AutoGPTQ） | 对 Hessian 分解的数值稳定性敏感 |

## 8. AWQ：保护重要通道

AWQ（Activation-aware Weight Quantization）的出发点和 GPTQ 不同：**不是所有权重对最终结果影响都一样大**——所以也不该平等对待。

### 8.1 关键观察

AWQ 作者发现：**约 1% 的权重通道对模型质量影响巨大**。这些通道的特征是它们对应的 **激活值幅度特别大**。

回忆 $Wx$ 的展开：每个输出元素是 $\sum_j W_{ij} x_j$。如果某列 $j$ 对应的 $x_j$ 普遍很大，那么这一列权重的微小量化误差会被 $x_j$ 放大——影响远大于其他列。

### 8.2 巧妙做法：放大重要通道再量化

直觉上你会想"那就把这些通道用更高精度存"。AWQ 用了一个更优雅的代数技巧：

1. 用少量校准数据找到激活幅值大的列 $j$
2. 把这些列的权重 **乘以 $s_j > 1$**（放大）
3. 量化放大后的权重
4. 推理时把对应的激活 **除以 $s_j$**（缩小）

数学上：

$$
Wx = \underbrace{(W \cdot \mathrm{diag}(s))}_{\text{放大后量化}} \cdot \underbrace{(\mathrm{diag}(s)^{-1} x)}_{\text{推理时缩小}}
$$

浮点计算中该变换等价；量化后不再严格等价。AWQ 在校准数据上搜索缩放系数，使激活幅度大的重要通道在给定 group 量化下更不易造成输出误差。收益来自“激活感知的通道缩放 + 校准搜索”，不是长尾裁剪，也不能只靠“范围更宽、刻度更细”来解释。

这就像在公共预算里给真正高需求的部门多分配资源——总量没变，分配更合理。

### 8.3 代码层面有多简单

AWQ 的运行时实现取决于导出的量化格式和推理引擎：

- 量化：把 $W \cdot \mathrm{diag}(s)$ 做普通 INT4 量化（per-group + 对称即可）
- 推理：在相应位置把 $x$ 除以 $s$，再使用与该量化产物兼容的 kernel

若模型结构和导出工具允许，可将部分缩放融合进相邻 LayerNorm 或线性层；是否零额外开销需要由具体图优化和 kernel 验证，不能一概而论。

### 8.4 GPTQ vs AWQ

| | GPTQ | AWQ |
|---|------|-----|
| 思路 | 量化误差补偿 | 重要通道保护 |
| 校准数据 | 需要 | 需要（更少） |
| 量化时间 | 数小时 | 几十分钟 |
| 数学依赖 | Hessian 分解 | 激活幅值统计 |
| 推理 kernel | 取决于量化产物与引擎 | 取决于量化产物与引擎 |
| INT4 质量 | 需在目标任务比较 | 需在目标任务比较 |

两者是不同的后训练量化路线；是否组合、导出格式、吞吐和质量均由实现和目标部署栈决定，应以实际产物的评测与压测为准。

## 9. 小结

| 概念 | 一句话回顾 |
|------|-----------|
| Scale $s$ | 把浮点拉伸到整数范围的"尺子刻度" |
| Zero-point $z$ | 让不对称分布也能用上完整整数范围的"原点偏移" |
| 量化误差 $0.5 s$ | 与 $|x|$ 无关——小权重相对误差更大 |
| 量化粒度 | per-tensor / per-channel / per-group，越细越准但元数据更多 |
| GPTQ | 量化误差补偿到未量化列，依靠 Hessian |
| AWQ | 激活感知缩放，保护重要通道 |

**一句话总结量化**：用精度换效率，关键是怎么把「哪里的精度可以省」识别出来。

### 延伸：FP8 量化

本篇聚焦 INT4/INT8 量化，但 [第 1 篇](/posts/2026-06-12-llm-mem-opt-1-fundamentals/) 介绍的 FP8（E4M3 / E5M2）同样重要：
- FP8 保留了浮点的指数/尾数结构，但训练和推理通常仍需按张量维护 scale；Transformer Engine 的 current/delayed scaling 都据 `amax` 设置 scale
- NVIDIA Transformer Engine 支持 FP8 recipe；前向常偏向 E4M3、反向梯度常偏向 E5M2，但格式选择和缩放策略取决于 recipe 与张量分布
- 推理场景下 FP8 KV Cache 量化（`kv_cache_dtype="fp8"`）是 [第 2 篇](/posts/2026-06-13-llm-mem-opt-2-inference/) 提到的省显存手段之一

下一篇我们离开推理，回到训练 → [LLM 训练与推理优化（四）— 训练显存与 ZeRO 优化](/posts/2026-06-15-llm-mem-opt-4-zero/)，把 [第 1 篇](/posts/2026-06-12-llm-mem-opt-1-fundamentals/) 引入的 $16\Psi$ 公式拆开讲清楚，并推导 ZeRO-1/2/3 的精确显存。

---

## 参考资料

1. Frantar et al. *GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers*. ICLR 2023. [arXiv:2210.17323](https://arxiv.org/abs/2210.17323)
2. Lin et al. *AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration*. MLSys 2024. [arXiv:2306.00978](https://arxiv.org/abs/2306.00978)
3. Nagel et al. *A White Paper on Neural Network Quantization*. 2021. [arXiv:2106.08295](https://arxiv.org/abs/2106.08295)
4. Frantar & Alistarh. *Optimal Brain Compression: A Framework for Accurate Post-Training Quantization and Pruning*. NeurIPS 2022. [arXiv:2208.11580](https://arxiv.org/abs/2208.11580) — OBQ
5. Hassibi et al. *Optimal Brain Surgeon and General Network Pruning*. ICNN 1993. — OBS 原文
6. Gray & Neuhoff. *Quantization*. IEEE Transactions on Information Theory, 1998. — 量化信息论经典综述
7. NVIDIA. [Transformer Engine FP8 delayed scaling](https://docs.nvidia.com/deeplearning/transformer-engine/features/low_precision_training/fp8_delayed_scaling/fp8_delayed_scaling.html)
