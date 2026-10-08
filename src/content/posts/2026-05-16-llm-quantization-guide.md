---
title: 大模型量化：从 INT4 编码到 GPTQ、AWQ
published: 2026-05-16
description: 从 bit、scale、zero-point 和量化误差讲起，逐步推导 GPTQ 的二阶误差补偿与 AWQ 的通道缩放，并说明部署时需要检查的格式、精度和性能。
category: Deep Learning
tags:
- llm
- quantization
- deep learning
draft: false
---

阅读顺序：[基础：编码、误差与分组](#quantization-basics) → [进阶：GPTQ / AWQ 推导](#advanced-quantization) → [部署检查](#quantization-deployment)。初学时，读完基础部分即可继续 [SFT](/posts/2026-06-16-llm-mem-opt-5-sft/)；进阶部分需要矩阵和二次型，按需读。

<a id="quantization-basics"></a>

## 0. 为什么 7B 模型的权重能从 14 GB 变成 3.5 GB

一个恰好有 $7\times10^9$ 个参数的模型，如果每个参数用 BF16 存储，权重数据就是 $14\times10^9$ bytes；如果每个参数改用 4 bit，理想的权重编码只需 $3.5\times10^9$ bytes。

减少存储位数可以节省权重空间，接下来还需要回答两个问题：**少了这么多 bit，原来的数怎么表示？模型的输出会变多少？**

本文先介绍回答这两个问题所需的基础，再在[进阶部分](#advanced-quantization)推导 GPTQ 的误差补偿和 AWQ 的通道缩放。若想先看显存除了权重还存什么，可以读[显存基础](/posts/2026-06-12-llm-mem-opt-1-fundamentals/)。

## 1. bit、byte 和数字格式

<span id="11-位数决定编码数量不直接决定数值范围"></span>

<span id="11-位宽编码数量与数值范围"></span>

### 1.1 几个 bit 能表示多少个数

1 bit 只有 0、1 两种状态，8 bits 组成 1 byte。$b$ bit 总共有 $2^b$ 种位模式；这些模式代表什么数，由编码格式决定。

| 格式 | 位数 | 常见解释 |
|---|---:|---|
| 无符号 4-bit 整数 | 4 | $0$ 到 $15$ |
| 有符号 INT4（补码） | 4 | $-8$ 到 $7$ |
| 有符号 INT8（补码） | 8 | $-128$ 到 $127$ |
| FP16 | 16 | 1 位符号、5 位指数、10 位小数部分 |
| BF16 | 16 | 1 位符号、8 位指数、7 位小数部分 |
| FP32 | 32 | 1 位符号、8 位指数、23 位小数部分 |

例如 `1101` 在无符号编码中是 $13$，在 4-bit 补码中是 $-3$。同一组 bit，因为解释规则不同，可以表示不同的数。

浮点数用符号、指数和有效数字共同编码，因此刻度并不均匀。BF16 与 FP32 的指数位数相同，数值范围接近，但 BF16 的有效数字更少；FP16 的有效数字比 BF16 多，指数范围却小得多。

**BF16 共有 $2^{16}=65536$ 种位模式。**其中还包含正负零、无穷和 NaN 编码，因此可表示的不同有限实数还少于 65536 个。4-bit 编码有 16 种，但每组权重可以使用自己的 scale，因此同一个整数编码在不同组里能表示不同的数，而非全模型只能共用 16 个实数。

<span id="12-4-倍压缩是理想权重编码的比例"></span>

### 1.2 BF16 与 INT4 的权重大小对比

本文用十进制单位：$1\,\mathrm{GB}=10^9$ bytes；而 $1\,\mathrm{GiB}=2^{30}$ bytes。

$$
\begin{aligned}
M_{\mathrm{BF16}}&=7\times10^9\times2=14\times10^9\ \mathrm{bytes}\approx13.04\ \mathrm{GiB},\\
M_{\mathrm{INT4,payload}}&=7\times10^9\times\frac48=3.5\times10^9\ \mathrm{bytes}\approx3.26\ \mathrm{GiB}.
\end{aligned}
$$

这里把两个 4-bit 编码打包进一个 byte，而且暂时没有计算 scale、zero-point、对齐开销和未量化层。实际推理还要保存 KV Cache、激活和工作区，所以不能由“权重编码压到四分之一”推出“总显存也压到四分之一”。

<span id="2-从一把刻度尺理解-scale"></span>

## 2. scale：整数相差 1，原来的数相差多少

假设某组权重都很小，直接把它们舍入成整数，会把很多不同的值变成 0。对这组数据来说，整数的刻度间隔 1 太粗了。

我们可以约定：整数编码相差 1，实际数值只相差 $\Delta$。这个正数 $\Delta$ 就是 **scale，也就是量化步长**。本文用 $\Delta$，避免在后半的 AWQ 推导中和通道缩放系数混淆。

先采用严格对称的整数范围 $[-K,K]$。例如 4-bit 存储可使用 $K=7$，舍弃 $-8$ 这个编码；8-bit 可使用 $K=127$。这是一种便于讲解的约定，不代表所有有符号量化格式都舍弃最小整数。

定义 $\operatorname{clip}(u,a,b)=\min(\max(u,a),b)$，那么：

$$
q=\operatorname{clip}\!\left(\operatorname{round}\!\left(\frac{w}{\Delta}\right),-K,K\right),
\qquad \hat w=\Delta q.
$$

- $w$ 是原始权重，$q$ 是存储的整数编码
- $\hat w$ 是反量化后用于表示这个权重的近似值
- round 选择最近整数，clip 保证编码没有超出位宽允许的范围

在恰好位于两个整数中间时，舍入规则需要约定。例如 NumPy 的 `rint` 使用 ties-to-even；下面的例子不涉及恰好落在中点的情况。

<span id="21-一个可手算的例子"></span>

<span id="21-int8-量化与反量化算例"></span>

### 2.1 手算一次 INT8 量化和反量化

取一组权重 $[0.1,-0.3,0.7,-0.9,1.2]$，使用严格对称 INT8，令：

$$
\Delta=\frac{\max_i|w_i|}{127}=\frac{1.2}{127}.
$$

| $w$ | $w/\Delta$ | $q$ | $\hat w$ | $\hat w-w$ |
|---:|---:|---:|---:|---:|
| 0.1 | 10.583333 | 11 | 0.103937 | +0.003937 |
| -0.3 | -31.750000 | -32 | -0.302362 | -0.002362 |
| 0.7 | 74.083333 | 74 | 0.699213 | -0.000787 |
| -0.9 | -95.250000 | -95 | -0.897638 | +0.002362 |
| 1.2 | 127.000000 | 127 | 1.200000 | 0 |

数值没有完整保留，但都还在原值附近。这里没有截断，所以绝对误差不超过 $\Delta/2\approx0.004724$。

<span id="22-覆盖最大值不等于求出了最优-scale"></span>

<span id="22-min-max-校准与量化误差的取舍"></span>

### 2.2 覆盖最大值，为什么未必误差最小

上面的 $\Delta=\max|w|/K$ 是避免范围截断的简单办法。若有极少数离群值很大，它们会把整组的刻度拉粗；缩小覆盖范围、允许这些离群值被截断，有时反而能降低总体误差。

若只关注权重均方误差，可以在候选步长中比较：

$$
\mathrm{MSE}(\Delta)=\frac1N\sum_{i=1}^N
\left[w_i-\Delta\operatorname{clip}\!\left(\operatorname{round}(w_i/\Delta),-K,K\right)\right]^2.
$$

这里的 $N$ 是这组待量化数的数量。实际还可以使用分位数、直方图或层输出误差来选范围。**min-max 是一种范围校准策略，不是对所有分布都成立的 MSE 最优解。**

全零张量需要单独处理：可取约定的正 scale，再把所有编码设为 0，避免出现 $0/0$。

<span id="3-zero-point给整数刻度移动原点"></span>

## 3. zero-point：用哪个整数编码表示零

如果待量化数主要落在正半轴，严格对称的编码会空出很多负刻度。非对称仿射量化允许移动原点，用整数 **zero-point** $z$ 表示实数 0：

$$
q=\operatorname{clip}\!\left(\operatorname{round}(w/\Delta)+z,q_{\min},q_{\max}\right),
\qquad \hat w=\Delta(q-z).
$$

若要求实数 0 可以精确表示，先把校准范围扩展到包含 0：

$$
a=\min(w_{\min},0),\qquad b=\max(w_{\max},0).
$$

当 $b>a$ 时，一种常用选择是：

$$
\Delta=\frac{b-a}{q_{\max}-q_{\min}},\qquad
z=\operatorname{clip}\!\left(\operatorname{round}\!\left(q_{\min}-\frac{a}{\Delta}\right),q_{\min},q_{\max}\right).
$$

例如使用无符号 4-bit 编码 $[0,15]$，表示区间 $[-1,2]$，则 $\Delta=0.2$、$z=5$。编码 0、5、15 分别表示 $-1$、0、2；$w=0.34$ 编码为 7，反量化为 0.4。

这个例子恰好对齐了两个端点。一般情况下 $z$ 需要取整，端点只能近似对齐；“同时精确表示两个端点和零”并不是无条件成立的。全零输入仍需特殊处理；非零常数在扩展到包含 0 后，范围则不再退化。

对称量化的 $z=0$ 可以隐含；非对称格式可能需要保存 $z$，也可能把相关计算融合进 kernel。具体速度不能只凭“多了一个减法”判断。权重和激活都可能采用对称或非对称方案，要同时看分布、格式和硬件支持，不能把“权重一定对称、激活一定非对称”当成规则。[PyTorch 的 observer 实现](https://github.com/pytorch/pytorch/blob/main/torch/ao/quantization/observer.py)展示了这类参数计算和边界处理。

把 scale 和 zero-point 放到同一组数轴上看。黑色标记表示反量化后的实数位置，蓝色标记表示整数编码；下图来自 Qualcomm AI Research 的量化白皮书，采用 8-bit 示例，刻度线只是示意，没有画全 256 个编码。

![量化白皮书原图：有符号、无符号与非对称量化网格，步长 s 和零点 z 决定编码对应的实数位置](/images/llm-series/sourced/mt-quantization-grids-original.png)

*原图内容未改动：Markus Nagel 等，[A White Paper on Neural Network Quantization，Figure 3](https://arxiv.org/html/2106.08295v1#S2.F3)（CC BY 4.0）；[原始 SVG](/images/llm-series/sourced/mt-quantization-grids-original.svg)。*

1. **相邻实数刻度间距是 $s$**：对应本文的 $\Delta$，不是它的倒数；相邻整数编码仍相差 1。
2. **左上采用完整有符号区间 $[-128,127]$**：因此端点并非严格关于零对称。本文前面的严格对称算例使用 $[-127,127]$，两种约定不要混算。
3. **右上从零覆盖非负范围**：图中称它为 symmetric unsigned，这里指不加零点偏移的无符号方案，不意味着非负区间关于零对称。
4. **下图将编码 $z$ 对齐实数零**：蓝色编码 0 对应实数 $-sz$，蓝色编码 255 对应 $s(255-z)$。改变 $z$ 是平移编码与实数的对应关系，不能增加编码总数。

## 4. 舍入误差、截断误差和输出误差

<span id="41-半个步长的上界有适用条件"></span>

<span id="41-舍入误差上界与截断误差"></span>

### 4.1 误差什么时候不超过半个步长

在表示范围内，最近舍入的误差满足：

$$
\hat w-w=\Delta\epsilon,\qquad |\epsilon|\le\frac12.
$$

一旦发生截断，$\Delta/2$ 就不再是总误差上界。例如 $\Delta=0.1$、$K=7$ 时，$w=1$ 被截到 0.7，误差是 0.3，明显大于 0.05。

也可以先把 $w$ 截到可表示的实数端点，得到 $w_c$，再拆成：

$$
\hat w-w=(\hat w-w_c)+(w_c-w).
$$

第一项是范围内的舍入误差，第二项是截断误差。减小步长会改善第一项的上界，但可能加重第二项。

例如步长为 0.1、编码范围为 −7 到 7 时，0.34 舍入到 0.3，绝对误差是 0.04；1.0 则超出可表示范围，被截到 0.7，绝对误差是 0.30，已经超过半个步长。

<span id="42-模型关心的是这些误差如何进入输出"></span>

<span id="42-权重误差对层输出的影响"></span>

### 4.2 权重量化后，层的输出会变多少

对线性层，设 $W\in\mathbb R^{m\times n}$、输入 $x\in\mathbb R^n$，输出 $y=Wx\in\mathbb R^m$。权重量化后的输出差为：

$$
\hat y-y=(\hat W-W)x,\qquad
\hat y_i-y_i=\sum_{j=1}^n(\hat W_{ij}-W_{ij})x_j.
$$

同样大小的权重误差，遇到更大的输入激活会造成更大的输出变化；不同项也可能抵消或叠加。因此，不能只看权重大小或相对误差，就断言某个权重是否重要。

用两个输入分量计算一次：某一行权重误差为 $(0.02,-0.02)$。输入是 $(1,1)^\top$ 时，输出误差为 $0.02-0.02=0$；输入换成 $(10,1)^\top$ 时，误差变成 $0.2-0.02=0.18$。权重的近似误差没有变化，但输入不同，输出误差也不同。这里的「激活」就是模型运行时得到的中间数值，不是另一套可训练权重。

这也解释了为什么量化方法需要校准数据：它提供了“这一层通常收到什么输入”的信息。校准误差小仍不代表所有任务都无损，最终还要比较目标任务质量。

<span id="5-量化粒度哪些值共用一把尺子"></span>

## 5. 量化粒度：哪些权重共用 scale 和 zero-point

设线性层的权重布局仍是 $W\in\mathbb R^{m\times n}$：

| 粒度 | 本文采用的约定 | 主要取舍 |
|---|---|---|
| Per-tensor | 整个矩阵共用一组量化参数 | 元数据少，但容易受全局范围影响 |
| Per-channel | 每个输出通道，即每行，共用一组参数 | 可以适应各行的范围差异 |
| Per-group | 每行沿输入维，把连续的 $g$ 个权重分一组 | 范围通常更局部，但元数据更多 |

使用“channel”时，需要说明对应矩阵的哪一维。这里 per-channel 指输出通道，对应矩阵的行；AWQ 缩放的是输入通道，对应矩阵的列。

更细的分组提供了更多调整空间，但误差和速度还受校准方法、元数据精度和 kernel 支持的影响，不一定随分组变细而改善。group size 也不是 GPTQ 或 AWQ 名称中固定的常数。

假设每组有 $g$ 个权重，权重编码占 $b$ bit，scale 和 zero-point 分别占 $b_\Delta$、$b_z$ bit，忽略尾组与对齐时：

$$
\text{平均 bits/weight}\approx b+\frac{b_\Delta+b_z}{g}.
$$

例如 INT4、$g=32$、每组仅存一个 FP16 scale，则平均为 $4+16/32=4.5$ bit/weight，比纯 INT4 编码多 12.5%。如果同样把 $7\times10^9$ 个参数全按此格式量化，就是约 3.9375 GB；若还存 zero-point、其他参数或未量化层，体积会继续增加。

<span id="6-读懂一个量化配置"></span>

<span id="6-量化对象训练阶段与部署配置"></span>

## 6. 量化哪些数据，什么时候量化

先区分三个对象：

- **权重量化**减少模型参数的存储和读取量
- **激活量化**改变层间或算子内部激活的表示
- **KV Cache 量化**减少生成过程中缓存的 Key/Value，占用仍随 token 数和并发变化

W4A16 通常表示 4-bit 权重、16-bit 激活，不意味着 KV Cache 自动变成 4 bit，也不说明所有算术都用 INT4。低位权重可能在融合 kernel 内分块解码为浮点再计算；累加精度由具体实现决定。反量化可以按计算需要分块进行，不必加载时就把整个模型还原为 BF16 并一直保留。后一种做法会让权重重新占用大量显存。

再区分量化发生的阶段：PTQ（Post-Training Quantization）从已有模型出发，使用统计量、校准或误差补偿；QAT（Quantization-Aware Training）则在训练中考虑量化的影响。GPTQ 和 AWQ 是两条重要的 PTQ 路线，算法名字本身不规定最终文件格式、所有层的精度或推理速度。

部署时至少检查：位宽与粒度、scale/zero-point 格式、哪些层未量化、激活与累加精度、KV Cache dtype，以及目标引擎是否真的支持这个量化产物。质量和速度都需要实测，不能只从“INT4”三个字推断。

<a id="7-下一步为什么-gptq-和-awq-能比直接舍入更好"></a>

<span id="7-从编码误差走向输出误差"></span>

<span id="7-gptq-与-awq-的优化思路"></span>

## 7. GPTQ 和 AWQ 怎样减少输出误差

有了编码、scale、zero-point、舍入和截断的概念，接下来要问：**怎样让近似权重产生的层输出尽量接近原输出？** GPTQ 允许未处理权重补偿已量化权重的误差；AWQ 则用通道缩放改变重要输入通道的有效刻度。

<a id="advanced-quantization"></a>

<span id="8-进阶入口矩阵方向与优化目标"></span>

<span id="8-线性层的矩阵约定与重建目标"></span>

## 8. 用矩阵写出线性层的输出误差

这部分介绍数学推导及其实现条件。算法在不同模型、硬件和数据上的效果，仍需通过测试比较。以下校准输入固定，bias 若不参与量化则保持不变。

<span id="81-本文的矩阵方向"></span>

### 8.1 权重和输入矩阵的行列分别表示什么

考虑一个线性层，忽略不变的 bias：

$$
W\in\mathbb R^{m\times n},\qquad
X=[x^{(1)},\ldots,x^{(N)}]\in\mathbb R^{n\times N},\qquad
Y=WX\in\mathbb R^{m\times N}.
$$

$m$ 是输出维度，$n$ 是输入维度，$N$ 是收集的校准 token/样本向量数。**每个校准向量是 $X$ 的一列**。若代码把输入存成 $N\times n$，那么对应的 Gram 矩阵应写为 `X.T @ X`，不能直接照搬本文的矩阵方向。

沿用基础部分的记号：$q$ 是整数编码，$Q(w)$ 是反量化后的近似值；$\Delta$ 是该组的 scale。下面优化的是层输出误差，不能只看每个权重的独立舍入误差。

下面的 $Q(W)$ 和 GPTQ 伪代码里的量化列，都表示反量化后的数值，可直接与浮点权重比较。

<span id="82-为何不用权重误差作为唯一目标"></span>

### 8.2 为什么要比较输出，而不只比较权重

第 4.2 节已经用单个输入说明权重误差如何传到输出。现在把多个校准输入并成矩阵 $X$，GPTQ 对整组输出使用重建目标：

$$
\min_{\hat W\in\mathcal Q}\;\mathcal L(\hat W)
=\|(\hat W-W)X\|_F^2.
$$

$\mathcal Q$ 是量化网格允许的矩阵集合；Frobenius 范数平方就是把所有输出元素的误差平方后相加。这里没有除以 $N$，除以 $N$ 也不会改变最优解，但会改变 Hessian 的整体比例。

这是给定校准输入下的**局部层目标**，不是整网的训练 loss，也不等于所有生成结果都保持不变。校准数据应覆盖部署输入；是否降低模型质量，仍要用目标任务验证。

<span id="9-gptq固定一个值允许其他值帮它补偿"></span>

## 9. GPTQ 的逐权重量化与误差补偿

<span id="91-二阶信息在这里来自一个精确的二次式"></span>

<span id="91-层重建误差的二次形式与-hessian"></span>

### 9.1 把输出误差展开，得到 Hessian

取 $W$ 的一行，转写成列向量 $w\in\mathbb R^n$，令权重扰动为 $\delta$。这一行输出的重建误差为：

$$
\mathcal L_{\mathrm{row}}(\delta)=\|\delta^\top X\|_2^2
=\delta^\top XX^\top\delta
=\frac12\delta^\top H\delta,\qquad H=2XX^\top.
$$

$H$ 是这个层重建问题的 Hessian，大小为 $n\times n$。它的对角线反映输入通道的平方幅值，非对角线反映通道之间的相关性。

可以先不用「Hessian」这个名称来读式子。若只有两个输入通道，展开误差得到

$$
\mathcal L_{\mathrm{row}}
=\delta_1^2\sum_t X_{1t}^2
+2\delta_1\delta_2\sum_t X_{1t}X_{2t}
+\delta_2^2\sum_t X_{2t}^2.
$$

首尾两项分别计算只改动一个权重时的误差；中间项说明两个权重的误差可能互相补偿。把这些系数排成矩阵就是 $XX^\top$（Gram 矩阵）。对这个误差先求一次导数得到梯度 $H\delta$，再求一次导数得到 $H$，所以称为 Hessian，即二阶导数矩阵。这里的「相关性」指输入乘积之和，没有减均值，严格说不是统计学的相关系数。

**对固定 $X$ 的线性层，这个二次式是精确的。**不需要假设整个神经网络已经训练到梯度严格为零，也没有构造全模型 loss 的巨大 Hessian。

### 9.2 为什么要加 damping

当校准输入不足、某些通道没有被激活或通道高度相关时，$XX^\top$ 可能奇异或病态。奇异表示有些方向无法由校准数据区分，矩阵没有逆；病态表示接近这种情形，小的数值误差也可能使求出的补偿剧烈变化。常见做法是在对角线上加一个小的正数：

$$
H_\lambda=2XX^\top+\lambda I,\qquad \lambda>0.
$$

这里 $I$ 是与 $H$ 同尺寸的单位矩阵，所以 $\lambda I$ 只给对角线加上 $\lambda$。正定表示对任意非零向量 $z$ 都有 $z^\top H_\lambda z>0$。因为 $2XX^\top$ 半正定，加上严格为正的 $\lambda I$ 后，矩阵在精确算术下可逆，也满足 Cholesky 分解的正定条件；有限精度下仍要关注数值稳定性。

从目标看，相当于在该二次问题中加入 $\frac\lambda2\|\delta\|_2^2$，避免在校准数据未充分约束的方向上做很大的补偿。原始 GPTQ 实现把 damping 设为平均对角值的一定比例，并对未激活通道另作处理；它不是一个对所有模型都最佳的固定常数。

下文假定使用的 Hessian 正定。若整个校准矩阵都为零，仅取“平均对角值的百分比”仍会得到零，需要单独处理，否则 Cholesky 分解仍可能失败。

<span id="93-从约束最优解得到-obsobq-补偿式"></span>

### 9.3 固定一个权重后，其他权重该怎么补偿

直觉上，我们要把第 $j$ 个权重固定到网格点 $\bar w_j$，但其他未量化权重仍可移动。设当前还可以调整的权重下标构成集合 $F$，当前权重已是此前约束下的连续最优点。于是新增补偿满足：

$$
\min_{\delta_F}\frac12\delta_F^\top H_F\delta_F,
\qquad e_j^\top\delta_F=d,\qquad d=\bar w_j-w_j.
$$

这里 $H_F=(H_\lambda)_{F,F}$，$e_j$ 是选取当前坐标的单位向量，$d$ 是必须完成的改变量。注意 $H_F^{-1}$ 一般**不等于**完整 $H_\lambda^{-1}$ 的同名子矩阵。

例如自由权重只剩两项，固定第一项时 $e_1=(1,0)^\top$，约束 $e_1^\top\delta_F=d$ 就是「第一项必须改动 $d$，第二项自由」。$F$ 不是一层的新参数，而是尚可调整的下标集合；某个权重量化固定后，就从这个集合移除。下面用 Lagrange 乘子把“必须满足的条件”并入目标，求得其余权重应该怎样一起移动。

引入一个 Lagrange 乘子 $\mu$：

$$
\mathcal J=\frac12\delta_F^\top H_F\delta_F
+\mu(e_j^\top\delta_F-d).
$$

求导并令其为零：$H_F\delta_F+\mu e_j=0$。记 $G_F=H_F^{-1}$，先解出乘子，再代回约束。

具体代入过程是：先由梯度为零得到 $\delta_F=-\mu G_Fe_j$；取第 $j$ 项得到 $d=-\mu(G_F)_{jj}$，因此 $\mu=-d/(G_F)_{jj}$。再代回整个向量，便得到下面的补偿式。符号 $(G_F)_{:,j}$ 表示第 $j$ 列的所有行。

$$
\boxed{\delta_F^*=\frac{d}{(G_F)_{jj}}(G_F)_{:,j}},\qquad
\boxed{\Delta\mathcal L_{\lambda}=\frac{d^2}{2(G_F)_{jj}}}.
$$

如果 $\bar w_j=0$，这就是 OBS 剪掉一个权重时的形式；如果 $\bar w_j$ 是量化网格点，就是 OBQ 使用的推广。分母体现允许其他坐标补偿后的有效曲率，但重要性还取决于分子里的量化距离，不能只比较 $(G_F)_{jj}$ 就给所有权重排好序。

一个两维例子能看出补偿在做什么。取：

$$
H_F=\begin{pmatrix}2&1\\1&2\end{pmatrix},\qquad
w=\begin{pmatrix}0.6\\0.4\end{pmatrix},\qquad \bar w_1=1.
$$

第一个权重必须增加 0.4，最优扰动是 $(0.4,-0.2)^\top$，所以临时权重变为 $(1,0.2)^\top$。二次代价是 0.12；若只改第一个权重、第二个不动，代价是 0.16。这里展示的是**一次约束更新**，第二个权重之后还要量化，此时还没有得到全部量化后的模型。

<span id="94-gptq-如何把这个过程做大"></span>

<span id="94-gptq-的计算效率改进"></span>

### 9.4 GPTQ 怎样减少计算量

对大语言模型逐行、逐参数进行贪心选择，并反复更新逆矩阵，计算开销很大。GPTQ 的主要改进可以理解为：

1. 各输出行共用一个列处理顺序，复用同一份输入二阶信息；也可先按激活统计量排列列，再沿该顺序处理
2. 同一块内仍顺序量化和补偿，块外的权重补偿暂存起来，最后用矩阵乘法一起做
3. 预先计算逆 Hessian 的 Cholesky 因子，把逐次消元所需的信息存下来，避免反复显式更新逆矩阵造成的数值问题

块内各列仍须依次处理：量化后面的列之前，要先应用前面各列产生的补偿。所谓批量更新，指的是推迟并合并块外的更新，而非让块内各列独立舍入。[GPTQ 论文第 3、4 节](https://arxiv.org/html/2210.17323v2)给出了目标、OBQ 更新和这些实现改进。

<span id="10-gptq-的-cholesky-方向与逐列更新"></span>

## 10. GPTQ 的 Cholesky 分解与逐列更新

### 10.1 为什么这里用上三角因子

令：

$$
G=H_\lambda^{-1}=R^\top R,
\qquad R\text{ 为上三角矩阵}.
$$

注意分解对象是**逆 Hessian**。如果库默认返回下三角 $L$，满足 $G=LL^\top$，那么这里取 $R=L^\top$。

Cholesky 分解可以看成正定矩阵的「矩阵平方根」，只不过要用三角矩阵及其转置相乘。这里先算好三角因子，后面就能按列取出补偿需要的数。为什么固定一列后不能直接删除逆矩阵的一行一列？把第一列与剩余坐标分开写成

$$
G=\begin{pmatrix}a&b^\top\\b&C\end{pmatrix},\qquad
R=\begin{pmatrix}r&u\\0&R_2\end{pmatrix}.
$$

由 $G=R^\top R$ 得 $a=r^2$、$b=ru^\top$、$C=u^\top u+R_2^\top R_2$。固定第一坐标后，剩余 Hessian 的逆是 $C-bb^\top/a=R_2^\top R_2$，而不是 $C$。这项相减消除了第一坐标已经不能再参与补偿的影响。重复这个过程，就得到下面的右下角子块公式。

依次固定第 $1,\ldots,j-1$ 列后，剩余坐标的逆 Hessian 等于：

$$
G_F=R_{j:n,j:n}^\top R_{j:n,j:n}.
$$

于是 $(G_F)_{jj}=R_{jj}^2$，而这一阶段的首行满足 $(G_F)_{j,k}=R_{jj}R_{jk}$。代入上一节的约束最优式，得到当前列产生的误差和补偿：

$$
E_{:,j}=\frac{W_{:,j}-\bar W_{:,j}}{R_{jj}},\qquad
W_{:,j:n}\leftarrow W_{:,j:n}-E_{:,j}R_{j,j:n}.
$$

右边是一个列向量和一个行向量的外积。更新当前列时，减掉的正好是 $W_{:,j}-\bar W_{:,j}$；更新后续列时，则按补偿公式调整尚未量化的权重。不能拿下三角因子的第 $j$ 行直接代入向右更新，因为它右侧元素都是零。

<span id="102-与官方实现一致的最简流程"></span>

<span id="102-gptq-逐列量化伪代码"></span>

先对照作者的列块图，区分“当前列”“当前块”和“块外剩余列”。

![GPTQ 作者原图加中文标注：逆 Hessian 的上三角因子、当前白列、粗框列块和块外补偿区](/images/llm-series/annotated/mt-gptq-block-zh.png)

*来源：Elias Frantar 等，[GPTQ，Figure 2](https://arxiv.org/html/2210.17323v2#S4.F2)（CC BY 4.0）；中文为辅助标注；[原始 SVG](/images/llm-series/sourced/mt-gptq-block-original.svg)。*

1. **左侧不是原始 Hessian**：标题中的 Cholesky form 指预先计算的逆 Hessian 三角因子，对应本文 $R$；灰色下三角不是另一组待量化权重。
2. **白色窄列是当前列**：右侧米黄、橙色区域已经量化；白列定下来以后，需要先补偿同一粗框内尚未量化的紫色列，才能继续处理下一列。
3. **粗框外浅紫区域可以稍后更新**：收集当前块的误差后，在块结束时用矩阵乘法统一补偿。它不是独立舍入，也不是把整个粗框一次量化成同一个值。
4. **示意图不规定块宽**：实际 `blocksize` 是实现参数；矩形宽度不代表量化误差或显存占比。

### 10.2 按列执行 GPTQ：伪代码

下面省略分组参数重估、列重排、死通道处理和块更新，保留数值逻辑。`Q_j` 返回第 $j$ 列在对应量化网格上的**反量化值**：

```text
输入 W: m×n, X: n×N；选好量化网格和正的 damping
H = 2 * X @ X.T + lambda * I
G = inverse(H)
R = cholesky(G).T                 # 默认 cholesky 返回下三角
W_work = copy(W)
for j = 0, ..., n-1:
    W_quant[:, j] = Q_j(W_work[:, j])
    err = (W_work[:, j] - W_quant[:, j]) / R[j, j]
    W_work[:, j:] -= outer(err, R[j, j:])
返回 W_quant
```

这里显式写 `inverse` 是为了对照数学定义；实现可先对 $H$ 分解，再用 Cholesky 求逆等数值例程。[原始 `gptq.py`](https://github.com/IST-DASLab/gptq/blob/main/gptq.py)先构造逆矩阵，再调用 `cholesky(..., upper=True)`，其归一化误差除以的是这个上三角因子的对角值。

实际按列块处理时，块内执行同样的顺序更新；收集该块的归一化误差矩阵 $E_{\mathrm{block}}$ 后，再做：

$$
W_{:,\mathrm{later}}\leftarrow W_{:,\mathrm{later}}
-E_{\mathrm{block}}R_{\mathrm{block},\mathrm{later}}.
$$

这延迟的是块外权重更新，不是跳过当前列需要的补偿。固定网格下，两种写法在精确算术中一致；动态分组校准或浮点舍入还需按照具体实现核对。

GPTQ 用顺序近似的方法求解这个离散优化问题，不保证全局最优，也不保证所有模型的 INT4 质量损失都低于某个固定百分比。

<span id="11-awq给重要输入通道更细的有效刻度"></span>

## 11. AWQ 的输入通道缩放

<span id="111-先做一个不改变线性函数的变换"></span>

<span id="111-权重与输入的等价缩放"></span>

先看 AWQ 作者如何说明“重要通道”。下图的 $X$ 是输入激活，$W$ 是权重，$Q$ 表示量化；RTN 是直接舍入到最近量化值，PPL 是困惑度（在相同评测设置下通常越低越好）。

![AWQ 作者原图：直接舍入、重要通道保留 FP16、量化前缩放三个方案](/images/llm-series/annotated/awq-overview-zh.png)

*来源：[原始资料](https://github.com/mit-han-lab/llm-awq)；MIT HAN Lab（MIT 许可，见[许可声明](/images/llm-series/sourced/training-image-license-notices.txt)）。本图添加中文辅助标注与引线；[查看原图](/images/llm-series/sourced/awq-overview.png)。*

1. **直接舍入**：左图把米黄色 FP16 权重直接舍入为蓝色 INT3 编码，没有区分通道的重要性。PPL 43.2 是原图特定评测结果，不是所有模型的必然结果。

2. **激活识别重要通道**：中图较深的红色输入列表示更大的激活幅度，用它识别哪些权重通道的误差更值得保护。原图使用 $XW$，所以这里输入通道对应 $W$ 的行。

3. **混合精度通道**：中图将对应权重行保留为 FP16，示例 PPL 降到 13.0，但混合精度计算会增加硬件实现负担。它是用来解释重要权重的对照方案，不是 AWQ 最终必须保留 FP16 的方案。

4. **量化前缩放**：右图按激活统计选择缩放，改善重要通道的相对量化误差，量化后仍统一存成 INT3。其 PPL 13.0 同样只是作者实验结果；图中未画出的输入反向缩放由下文等式补全。

注意作者画的是 $XW$，输入通道对应 $W$ 的行；下文采用 $Y=WX$，输入通道对应 $W$ 的列。这是矩阵转置约定的差别。图中省略了保持输出不变所需的输入反向缩放，下面用等式补全。

### 11.1 权重放大、输入缩小，先保持输出不变

GPTQ 关注“误差如何补偿”。AWQ 关注另一件事：**量化之前，能否先改变各输入通道使用量化刻度的方式？**

取正的通道缩放系数 $s_1,\ldots,s_n$，组成对角矩阵 $S$：

$$
S=\operatorname{diag}(s_1,\ldots,s_n),\qquad
WX=(WS)(S^{-1}X).
$$

也就是把权重第 $j$ 列乘以 $s_j$，同时把输入第 $j$ 行除以 $s_j$。在实数算术里两者抵消，函数完全相同；实际浮点计算还会有有限精度舍入。

然后只量化变换后的权重：

$$
\hat Y=Q(WS)S^{-1}X,
\qquad
\hat Y-Y=[Q(WS)-WS]S^{-1}X.
$$

量化后通常不再严格相等。这里的 $s_j$ 是**通道缩放系数**，而 $\Delta$ 是组内整数刻度之间的**量化步长**，它们不能混为一谈。

对于输入激活较大的通道，降低有效权重误差往往更有利于减小输出误差。AWQ 由此使用激活统计量来选择缩放，而不是仅凭权重幅值找“重要权重”。论文中保护少量通道的实验是这一思路的证据，不意味着每个模型都恰好有 1% 的通道必须保留高精度。

<span id="112-步长变大也可能改善重要通道"></span>

<span id="112-缩放后的有效步长与误差上界"></span>

### 11.2 缩放后，这个通道的误差上界变小了吗

考虑同一组中的一个权重 $w$。缩放后组内量化步长可能由 $\Delta$ 变成 $\Delta'$。在未截断、忽略 scale 存储舍入的最近舍入模型下，记整数舍入残差为 $\epsilon,\epsilon'\in[-1/2,1/2]$，则折回原坐标的误差为：

$$
e=\Delta\epsilon,\qquad
 e'=\frac{\Delta'}{s_j}\epsilon'.
$$

因此两个**误差上界**为：

$$
|e|\le\frac{\Delta}{2},\qquad
|e'|\le\frac{\Delta'}{2s_j}.
$$

有效步长从 $\Delta$ 变为 $\Delta'/s_j$。只要 $\Delta'/\Delta<s_j$，这个通道的上述舍入误差上界就更小。$\Delta'\approx\Delta$ 且 $s_j>1$ 是易于理解的一种有利情形，**并不要求量化步长必须保持不变**。

不过，上界变小不等于该权重的实际误差必然变小。$\epsilon'$ 与 $\epsilon$ 未必相同；当 $\epsilon\ne0$ 时，真实误差比还包含 $|\epsilon'/\epsilon|$。层输出还受到多个误差项相互叠加或抵消的影响。因此不能把 $\Delta'/(s_j\Delta)$ 不加条件地当成实际输出误差的缩小比例。

<span id="113-一个步长增大但输出误差减小的例子"></span>

<span id="113-两通道缩放与输出误差算例"></span>

### 11.3 算一个两通道的例子

下面是说明原理的两通道简化算例，不是 AWQ 模型质量基准。令：

$$
W=\begin{pmatrix}0.7&1\end{pmatrix},\qquad
x=\begin{pmatrix}10\\1\end{pmatrix},\qquad y=Wx=8.
$$

两个权重共用严格对称 4-bit 网格 $[-7,7]$，步长按最大幅值计算。

**直接量化：**$\Delta=1/7$，编码为 $(5,7)$，反量化为 $(5/7,1)$，于是：

$$
y_{\mathrm{direct}}=\frac57\times10+1=\frac{57}{7}\approx8.142857.
$$

**把第一个通道放大 2 倍：**取 $S=\operatorname{diag}(2,1)$，则 $WS=(1.4,1)$，$S^{-1}x=(5,1)^\top$。新步长为：

$$
\Delta'=\frac{1.4}{7}=0.2>\Delta.
$$

新编码为 $(7,5)$，这两个值恰好都落在新网格上。因此：

$$
y_{\mathrm{scaled}}=1.4\times5+1\times1=8.
$$

第一个通道的有效步长从 $1/7$ 变成 $0.2/2=0.1$。虽然组内步长变大了 1.4 倍，但通道放大了 2 倍，因此上界比例为 $1.4/2=0.7$。这个具体例子里输出误差从 $1/7$ 降到了 0，并非按上界比例乘以 0.7。

第二个通道的有效步长反而从 $1/7$ 增到 0.2，只是在本例中数值 1 恰好仍被精确表示。真实模型里，保护一个通道可能损害同组其他通道，这正是需要校准搜索的原因。

<span id="114-两个边界例子避免过度推论"></span>

<span id="114-实际误差与等比例缩放的反例"></span>

### 11.4 为什么误差不一定变小

**更小的上界仍可能对应更大的实际误差。**例如一组权重为 $(0.3,0.7)$，严格对称 INT4 的步长是 0.1，0.3 原本能精确表示。只把第一个通道乘以 1.1 后，最大幅值仍为 0.7、步长仍为 0.1；0.33 被量化为 0.3，折回原坐标是 $0.3/1.1\approx0.272727$。误差从 0 增到约 0.027273，尽管上界从 0.05 降到了约 0.045455。

**把全组一起等比例放大，通常不会凭空增加精度。**在同一 min-max 对称量化规则、精确算术下，若整组都乘 $c>0$，那么新步长也是 $c\Delta$，有 $Q(cW)/c=Q(W)$。AWQ 的作用来自通道之间的相对缩放，以及对组内所有误差的取舍。

上述分析把 scale 和缩放系数视为精确实数。实际低精度存储、非对称 zero-point、clipping 和融合算子还会引入额外影响，不能直接套用未截断的半步长上界。[AWQ 论文第 3 节](https://arxiv.org/html/2306.00978v5)讨论了缩放、步长变化以及重要和非重要通道间的权衡。

<span id="12-awq-的校准搜索到底在做什么"></span>

<span id="12-awq-通道缩放系数的校准搜索"></span>

## 12. AWQ 怎样用校准数据选择缩放系数

接下来需要选择具体的 $s_j$。先统计每个输入通道在校准数据上的平均绝对激活：

$$
a_j=\frac1N\sum_{t=1}^N|X_{jt}|.
$$

一种搜索方式是用 $s_j(\alpha)\propto a_j^\alpha$ 生成候选缩放，再比较：

$$
\alpha^*=\arg\min_{\alpha\in\mathcal A}
\|Q(WS(\alpha))S(\alpha)^{-1}X-WX\|_F^2.
$$

$\mathcal A$ 是有限候选集合。零激活、极端数值需要正数下限与归一化保护；归一化后并不要求每个 $s_j$ 都大于 1，重要的是通道之间的相对尺度。

这里 $\arg\min$ 返回「哪个候选取得最小误差」，不是返回误差数值。例如两个通道的平均幅值为 $(4,1)$，先试 $\alpha=0$ 得到相对缩放 $(1,1)$，再试 $\alpha=1/2$ 得到 $(2,1)$。每次都重新量化缩放后的权重、算校准输出并与 $WX$ 比较；第二个候选是否更好必须由误差决定，不能只因为第一个通道激活大就直接选它。

[官方 `auto_scale.py`](https://github.com/mit-han-lab/llm-awq/blob/main/awq/quantize/auto_scale.py)的这一路径先由平均绝对激活生成候选，再试量化并比较层或模块输出 MSE。该实现使用 20 个候选指数 $0,0.05,\ldots,0.95$，并对缩放做下限处理和归一化。这是实现细节，不是 AWQ 定义必须采用的常数。

因此，AWQ 虽然不需要反向传播，仍要试量化并比较输出。它不做 GPTQ 式的逐权重二阶补偿，而是使用校准数据选择缩放；权重 clipping 搜索是另一个可配合的步骤，不能代替通道缩放的解释。

部署时，可在图结构允许的位置把 $S^{-1}$ 融合进前面的归一化或线性操作；遇到非线性、残差或多个消费者时，必须保证所有相关路径仍然等价。缩放能否跨算子融合取决于计算图，无法对任意模型都保证零开销。

<a id="quantization-deployment"></a>

<span id="13-把数学结论带回部署"></span>

<span id="13-gptqawq-的部署比较"></span>

## 13. 部署时怎样比较 GPTQ 和 AWQ

| 问题 | GPTQ | AWQ |
|---|---|---|
| 核心调整 | 量化当前列，并补偿尚未量化的列 | 量化前按输入通道缩放，并反向缩放输入 |
| 使用的校准信息 | 输入 Gram/Hessian 与量化残差 | 激活幅值与候选缩放后的输出误差 |
| 需要权衡什么 | 局部目标、量化顺序、阻尼、分组 | 重要通道与其他通道的误差、缩放与裁剪 |
| 最终质量与速度 | 依模型、数据、格式和 kernel 实测 | 依模型、数据、格式和 kernel 实测 |

仅凭算法名称，无法判断哪种方法的误差更小、速度更快，也不能由“权重是 INT4”推断所有矩阵乘和累加都用 INT4。典型 weight-only 部署仍使用较高精度激活，低位权重在 kernel 内解码参与运算；是否获得加速取决于工作负载和实现。

尤其要分清：**GPTQ/AWQ 的权重量化，不会自动量化 KV Cache。**KV Cache dtype、激活量化和权重格式是需要分别验证的配置。更小的模型文件，也不等于同倍率降低端到端延迟或进程峰值显存。

GPTQ 每量化一列，就调整还没量化的列来补偿误差；AWQ 先缩放输入通道，再用校准数据挑出输出误差较小的缩放方案。选好方法后，还要实际测回答质量、速度和显存，才能判断是否适合部署。

沿入门路线可继续 [SFT](/posts/2026-06-16-llm-mem-opt-5-sft/)。若关心训练状态，再按需读 [AdamW](/posts/2024-04-20-weight-decay-and-l2-regularization/) 和 [ZeRO](/posts/2026-06-15-llm-mem-opt-4-zero/)。权重量化主要压缩表示，ZeRO 则从多卡训练中哪些状态需要重复保存入手，是另一条优化路线。


## 参考资料

- Nagel et al. [A White Paper on Neural Network Quantization](https://arxiv.org/abs/2106.08295)，量化参数、粒度与校准的系统介绍
- PyTorch [量化 observer 源码](https://github.com/pytorch/pytorch/blob/main/torch/ao/quantization/observer.py)，用于核对仿射量化参数与边界处理
- Frantar et al. [GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](https://arxiv.org/abs/2210.17323)，ICLR 2023；尤其是局部重建目标与 Algorithm 1
- IST-DASLab [GPTQ 官方实现 `gptq.py`](https://github.com/IST-DASLab/gptq/blob/main/gptq.py)，用于核对 damping、上三角 Cholesky 与块更新
- Lin et al. [AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration](https://arxiv.org/abs/2306.00978)，MLSys 2024
- MIT HAN Lab [AWQ 官方 `auto_scale.py`](https://github.com/mit-han-lab/llm-awq/blob/main/awq/quantize/auto_scale.py) 与 [量化实现](https://github.com/mit-han-lab/llm-awq/blob/main/awq/quantize/quantizer.py)
- Frantar, Singh & Alistarh. [Optimal Brain Compression](https://arxiv.org/abs/2208.11580)，NeurIPS 2022；OBQ 的来源与二阶约束更新
