---
title: 大模型显存（二）：ZeRO 如何分摊训练显存
published: 2026-06-15
description: 从 DDP 的多副本冗余出发，说明特定 Adam 混合精度配置下的 16Ψ 模型状态，并推导 ZeRO-1/2/3 分片后的状态大小与通信开销。
category: Deep Learning
tags:
- llm
- memory
- llm-memory-optimization-series
- distributed-training
- zero
- deepspeed
draft: false
---

<span id="zero-如何节省训练显存状态分片与多卡通信"></span>

这是显存子系列的第二篇。[第一篇](/posts/2026-06-12-llm-mem-opt-1-fundamentals/)讲清了训练时为什么要保存参数、梯度和优化器状态。这一篇沿用其中的 [Adam 存储配置](/posts/2026-06-12-llm-mem-opt-1-fundamentals/#51-一套明确的-adam-混合精度配置)，先看多张卡为什么会重复保存这些数据，再逐步解释 ZeRO 怎样分摊存储，以及为此增加的通信和临时显存。

读本篇前，知道前向、反向、梯度和优化器分别做什么就够了，下面会就地回顾分片需要的状态含义。只做单卡 SFT，可以直接进入[第三篇：微调时怎样节省显存](/posts/2026-06-16-llm-mem-opt-5-sft/)，不必先掌握多卡通信细节。

<span id="0-先理解问题多卡为什么还会装不下"></span>

<span id="0-数据并行中的模型状态副本"></span>

## 0. 为什么加了 GPU，每张卡还是放不下模型

**参数（权重）**是模型要学习的数，**loss（损失）**是衡量预测误差的数，**激活**是计算当前输入时产生的中间结果。一个 batch 是一批训练样本。Data Parallel（DP，数据并行）把一个 batch 分给多张卡处理，每张卡仍执行完整模型的前向和反向。标准 DistributedDataParallel（DDP）通常让每个进程保存完整参数、梯度和优化器状态。因此，增加 GPU 数量可以分摊样本的计算，却不会自动缩小每张卡上的模型状态。

本文按“一张 GPU 对应一个训练进程”来讲。进程编号叫 rank；一个数据并行组就是共同合并梯度的一组进程。每张卡上的模型状态占用要分别核算，8 张 24 GB 卡并不自动变成一张可随意使用的 192 GB 卡。

一次训练更新可以拆成四步：

1. **Forward**：模型读入各自的样本，得到预测与 loss，并保存反向需要的激活。
2. **Backward**：由 loss 计算各参数的梯度。梯度描述 loss 对参数变化的局部敏感程度；优化器再据此决定调整方向和幅度。
3. **梯度同步**：合并不同卡的梯度，使各卡基于同一个全局目标更新。若各卡有效 token 数不同，不能不加权地平均各卡的 token-mean loss；应按有效 token 总数统一归一化。
4. **Optimizer step**：优化器利用同步后的梯度及历史状态更新参数，再开始下一步。

各卡处理的数据不同，同步前的梯度通常也不同。ZeRO 通过分片减少各卡重复保存的优化器状态、梯度和参数；具体分片哪些状态，取决于所用阶段。它不要求各卡同步前的本地梯度相同。

ZeRO（Zero Redundancy Optimizer）的基本思路是：**把模型状态分给不同的卡负责，需要计算时再同步或取回相应数据。** Stage 1 先分优化器状态，Stage 2 再分梯度，Stage 3 再分参数。它保留数据并行的计算方式，并不等于把网络的不同层固定交给不同的卡计算。[DeepSpeed 官方教程](https://www.deepspeed.ai/tutorials/zero/)给出了三个阶段的定义。

<span id="1-先算清楚16ψ16psi16ψ-是哪一种训练配方"></span>

<span id="1-先算清楚每参数-16-字节是哪一种训练配方"></span>

<span id="1-adam-混合精度训练的状态大小"></span>

## 1. Adam 训练需要保存哪些数据，占多少显存

### 1.1 本文符号与单位

- $\Psi$：参与全参数训练的参数数量，单位是“个”，不是字节。
- $N$：同一个 ZeRO 数据并行组中的进程数。以下假定每个进程用一张 GPU；若同时用了 Tensor Parallel 或 Pipeline Parallel，$N$ 不能直接取集群总卡数。
- GB 使用十进制：$1\text{ GB}=10^9$ 字节；GiB 使用二进制：$1\text{ GiB}=2^{30}$ 字节。
- 数值例子统一取 $\Psi=7\times10^9$、$N=8$。这是恰好 70 亿参数的算术例子，不代表所有名称带“7B”的模型参数量完全相同。

<span id="12-为什么训练还要存-adam-状态"></span>

<span id="12-adam-状态与主权重"></span>

### 1.2 Adam 除了参数，还要保存什么

反向传播为每个可训练参数算出梯度；Adam 除了读取这次的梯度，还要保留两份跨训练步更新的统计量：

- $m$：近期梯度的指数移动平均，保留带正负号的平均方向。
- $v$：近期梯度平方的指数移动平均，用来按梯度的量级调整步长。它是未中心化二阶矩，不是梯度的方差。

一个参数对应一个 $m$ 和一个 $v$；一百万个参数就各有一百万个 $m$、$v$ 数值。每一步只更新这些平均值，不需要保存全部历史梯度。**梯度完成更新后可以清理，$m$、$v$ 则要留给下一步。** 完整更新公式、偏差修正和单参数手算已放在[第一篇 §5.3](/posts/2026-06-12-llm-mem-opt-1-fundamentals/#53-为什么常保留-fp32-状态)，也可对照 [Adam 原论文](https://arxiv.org/abs/1412.6980)。

主权重（master weights）是同一组参数的高精度副本：先在这份副本上更新参数，再转成低精度副本供计算，以减少小更新在舍入时丢失的风险。它不是另一套独立学习的模型。下面把主权重与 $m$、$v$ 合称“优化器侧状态”，正是 ZeRO-1 首先分片的对象。

本篇沿用[显存基础的 Adam 混合精度配置](/posts/2026-06-12-llm-mem-opt-1-fundamentals/#51-一套明确的-adam-混合精度配置)：**所有参数都可训练，没有 CPU/NVMe offload**。为对应后面的分片对象，把状态归成三类：

| 状态类别 | 字节/参数 | 7B 占用 | 组成 |
|----------|----------|---------|------|
| 计算用参数 | 2 | 14 GB | BF16 或 FP16 参数副本 |
| 保留梯度 | 2 | 14 GB | BF16 或 FP16 梯度 |
| 优化器侧状态 | 12 | 84 GB | FP32 master weights、Adam $m$、Adam $v$，各 4 字节 |
| **合计** | **16** | **112 GB** | 仅模型状态 |

$$
M_{\text{state}}=(2+2+12)\Psi=16\Psi\text{ 字节}.
$$

FP32 每个数占 4 字节，BF16/FP16 每个数占 2 字节；dtype 就是这种数值类型。这里混合使用两种精度，并保留独立主权重。CPU/NVMe offload 是把状态移到 CPU 内存或固态存储，本表暂不使用它。这里的 $12\Psi$ 是本文统计 ZeRO 优化器侧状态时采用的定义，不表示所有实现都必须有独立 master weights。

<span id="13-18ψ18psi18ψ-与-autocast不要仅凭-bf16-标签猜显存"></span>

<span id="13-每参数-18-字节与-autocast不要仅凭-bf16-标签猜显存"></span>

<span id="13-fp32-梯度与-autocast-对存储的影响"></span>

### 1.3 改用 FP32 梯度或 autocast 后，显存怎么算

本配置若只把保留梯度改为 FP32，其他副本不变，就变为 $(2+4+12)\Psi=18\Psi$ 字节，7B 对应 126 GB；同时存活的转换缓冲仍要另计。

`torch.autocast` 选择算子的计算精度，并不规定参数、梯度和优化器状态怎样常驻。开启 BF16 autocast，不足以证明采用了上面的 $2+2+12$ 配置；[第一篇 §5.2](/posts/2026-06-12-llm-mem-opt-1-fundamentals/#52-autocast-不等于常驻两套权重)解释了常见 AMP 配置的区别。套用后面的分片公式前，应核对实际 dtype、是否有独立主权重，以及哪些副本同时存在。

<span id="2-zero-三阶段切哪些状态没切哪些"></span>

<span id="2-zero-三个阶段的状态分片"></span>

## 2. ZeRO-1、2、3 分别把哪些数据分给各张卡

用两张卡分担参数状态，可以直观看出分片方式：假设参数为 $[w_0,w_1,w_2,w_3]$，两张卡各负责一半，GPU 0 负责前两个，GPU 1 负责后两个。分片是把**同一组参数的位置**拆开：$w_0$ 的主权重、对应的历史摘要 $m^{(0)}$、$v^{(0)}$ 以及同步后的梯度应保存在负责这一分片的同一张卡上。这里用括号上标标识“属于哪个参数”，如 $m^{(0)}$ 属于参数 $w_0$。分片不意味着 GPU 0 只计算前半个模型；数据并行中，每张卡仍用自己的样本经过完整网络。

先约定 $b_p,b_g,b_o$ 分别表示每个参数的计算用副本、保留梯度和优化器侧状态所占的字节数。假设分片均衡、没有 offload、忽略对齐与临时缓冲，则有：

| 配置 | 计算用参数 | 保留梯度 | 优化器侧状态 | 单卡模型状态大小 |
|------|-----------|----------|---------------|------------------|
| DDP / Stage 0 | 完整 | 完整 | 完整 | $(b_p+b_g+b_o)\Psi$ |
| ZeRO-1 | 完整 | 完整 | $1/N$ | $(b_p+b_g)\Psi+b_o\Psi/N$ |
| ZeRO-2 | 完整 | $1/N$ | $1/N$ | $b_p\Psi+(b_g+b_o)\Psi/N$ |
| ZeRO-3 | $1/N$ | $1/N$ | $1/N$ | $(b_p+b_g+b_o)\Psi/N$ |

这里的“保留”描述训练状态的分片布局，不表示梯度在训练的每一刻都已分配，也不表示反向中从未出现较大的未分片张量。表中只算**理想情况下的模型状态**，没有加激活、临时参数和通信缓冲。

下图是 ZeRO 作者的原始分片示意。图中 $\Psi$ 仍表示参数个数，$N_d$ 是数据并行卡数（即本文的 $N$），$K$ 表示每个参数的优化器侧状态字节数；$P_{os}$、$P_{os+g}$、$P_{os+g+p}$ 依次对应 ZeRO-1、2、3。

![ZeRO 原图：蓝色参数、橙色梯度、绿色优化器状态逐阶段分片](/images/llm-series/annotated/zero-partitioning-zh.png)

*来源：[原始资料](https://www.microsoft.com/en-us/research/blog/zero-deepspeed-new-system-optimizations-enable-training-models-with-over-100-billion-parameters/)；Microsoft Research；图片文件来自 [Hugging Face 官方转载教程](https://huggingface.co/blog/zero-deepspeed-fairscale#the-magic-behind-zero)，权利归原作者，来源页未单列该图许可。本图添加中文辅助标注与引线；[查看原图](/images/llm-series/sourced/zero-partitioning.png)。*

1. **各卡完整复制**：最上行每张 GPU 都保存完整的蓝色参数、橙色梯度和绿色优化器状态，因而存在跨卡重复存储。颜色含义按底部原图图例读取。

2. **先分优化器状态**：第二行对应 ZeRO-1：绿色状态分散到不同卡，蓝色参数和橙色梯度仍完整复制。不同卡的绿色窄条位于不同横向位置，表示保存不同片段，不是各卡保留同一小段。

3. **再分梯度与参数**：第三行 ZeRO-2 进一步分梯度，最下行 ZeRO-3 再分参数。这里画的是模型状态的分片布局，不保证训练全过程都没有临时收集或额外缓冲。

4. **7.5B 参数 · 64 卡**：原图取 $\Psi=7.5$B、$N_d=64$、$K=12$，状态容量从 120 GB 降到约 1.9 GB；不要与下文 7B / 8 GPU 算例混用。通信量一列依赖原图通信模型，不是实测耗时，容量也不包含激活、临时参数收集和通信缓冲。

回到正文的 7B / 8 GPU 例子。分别计算三类状态：完整计算参数是 14 GB，完整梯度是 14 GB，优化器侧是 84 GB。ZeRO-1 只把 84 GB 分成 8 份，所以每卡为 $14+14+84/8=38.5$ GB；ZeRO-2 再分梯度，得到 $14+14/8+84/8=26.25$ GB；ZeRO-3 连第一项也分，得到 $(14+14+84)/8=14$ GB。

代入 $b_p=2,b_g=2,b_o=12$，汇总为：

| 配置 | $16\Psi$ 配置下的公式 | 每卡状态大小 |
|------|-------------------------|---------------|
| DDP | $16\Psi$ | 112 GB |
| ZeRO-1 | $4\Psi+12\Psi/N$ | 38.5 GB |
| ZeRO-2 | $2\Psi+14\Psi/N$ | 26.25 GB |
| ZeRO-3 | $16\Psi/N$ | 14 GB |

如果使用上一节的 FP32 梯度配置，$b_g$ 应改为 4：对应 DDP 为 126 GB，ZeRO-1 为 $6\Psi+12\Psi/N=52.5$ GB，ZeRO-2 为 $2\Psi+16\Psi/N=28$ GB，ZeRO-3 为 $18\Psi/N=15.75$ GB。**dtype 改变时，各项状态大小也要重新计算。**

<span id="21-zero-1只分优化器侧状态"></span>

<span id="21-zero-1优化器状态分片"></span>

### 2.1 ZeRO-1：每张卡只更新自己负责的参数

可以把参数按编号分给 8 张卡：每张卡只负责其中一段参数的 master weights、$m$ 和 $v$，也只执行这一段的 optimizer step。但为了前向与反向，各卡仍保存完整计算参数；这一阶段也没有要求把梯度的存储长期缩小到 $1/N$。

下面会用到三个通信名称：Reduce-Scatter 是“合并各卡同位置的数据，然后每卡只收结果的一段”；All-Gather 是“拼起各卡各持有的一段，让每卡拿到完整结果”；All-Reduce 则让每卡都拿到完整合并结果。“归约”在本文主要指按对应位置求和，求平均还要除以合适的总数。第 3 节会用具体数组演示。

一次更新可以按这样的逻辑安排：

1. 各卡完成自己的前向与反向，产生梯度。
2. 将每一段梯度归约到负责该段更新的卡。Reduce-Scatter 是实现这种分发的一种方式。
3. 各卡更新自己负责的 master weights，并生成更新后的低精度参数分片。
4. 通过 All-Gather 或等效广播，把更新后的参数同步回所有卡，供下一步使用。

**一次 Reduce-Scatter 的输出是分片，不代表完整梯度存储已经消失。** Stage 1 和 Stage 2 的区别在于梯度生命周期与存储管理，不能只看通信原语的名字。实际框架也可能用 Reduce、All-Reduce 或不同 bucket 调度实现同步，以上是理解原理的流程，不是所有版本的逐行执行顺序。

<span id="22-zero-2梯度算出来后也只留下自己负责的一段"></span>

<span id="22-zero-2梯度和优化器状态分片"></span>

### 2.2 ZeRO-2：梯度也只保留自己负责的一段

Stage 2 进一步在反向过程中归约梯度并释放不再需要的非本地分片，长期保留的更新梯度与本地优化器状态一一对应。因此梯度项从 $b_g\Psi$ 变成 $b_g\Psi/N$；完整低精度参数仍需在各卡上保存，更新后仍需同步参数。

每张卡仍用自己的样本执行完整模型的反向传播，计算所有参数的梯度；Stage 2 调整的是梯度产生后的归约和释放时机。归约前的梯度 bucket（把多块梯度攒在一起发送的通信缓冲）、打包缓冲和 dtype 转换仍可能占显存。

<span id="23-zero-3参数平时也只留一段用时取回"></span>

<span id="23-zero-3参数分片与按需收集"></span>

### 2.3 ZeRO-3：参数用到时再收集

Stage 3 连计算用参数也分片：执行某个模块之前，先 All-Gather 该模块需要的参数；用完后，按策略释放非本地部分。反向若需要已释放的参数，就要再次取回。算出的梯度再通过 Reduce-Scatter 等方式交给各自负责更新的卡。

分片保存时，每张卡只持有 $1/N$ 的参数；计算时，通常还要临时收集当前模块的完整参数，预取也可能让下一个模块的参数提前驻留。模块划分、参数持久化阈值和重用策略会改变 gather 次数与峰值。[DeepSpeed ZeRO-3 文档](https://deepspeed.readthedocs.io/en/stable/zero3.html#assumptions)也明确要求单个子模块的前向与反向能放进设备显存。

所以 $16\Psi/N$ 只能表示本例的分片状态部分。它不能证明 7B 在 8 张 14 GB 卡上就能训练，也不是固定单卡容量下训练大模型的唯一办法；Tensor Parallel、Pipeline Parallel 和 offload 也可能参与解决容量问题。

<span id="3-通信字节相同也不代表免费"></span>

<span id="3-通信量与通信耗时"></span>

## 3. 每次更新要传多少数据，会花多少时间

先区分三个常用操作：

- **All-Reduce**：把各卡的同位置数值求和（平均还需相应缩放），每张卡拿到完整结果。
- **Reduce-Scatter**：同样归约，但每张卡只拿到结果的一段。
- **All-Gather**：把各卡持有的不同分片拼起来，每张卡拿到完整结果；它本身不做求和。

**两张卡、四个参数的完整小例子。** 为了只看通信，假定每张卡的损失归一化方式和有效样本数相同。两卡分别算出梯度 $g^{(0)}=[1,2,3,4]$、$g^{(1)}=[5,6,7,8]$。这里梯度 $g$ 的括号上标标记 GPU 编号，数组中的四个位置才对应四个参数；前面状态 $m^{(i)},v^{(i)}$ 的上标则标记参数编号：

1. 按位置相加得到 $[6,8,10,12]$；除以 2 得全局平均梯度 $[3,4,5,6]$。All-Reduce 可使两卡都拿到这份完整结果。
2. Reduce-Scatter 则可以让 GPU 0 拿到平均结果的前半段 $[3,4]$，GPU 1 拿到后半段 $[5,6]$。平均缩放可以在归约前或后进行；它不是 All-Gather 做的事。
3. 对于 ZeRO-1/2，各卡用本地对应的主权重和 $m$、$v$ 更新自己的两个参数，得到 $[w'_0,w'_1]$ 与 $[w'_2,w'_3]$。
4. All-Gather 拼成 $[w'_0,w'_1,w'_2,w'_3]$，两卡都拿到新的完整计算权重，开始下一轮。它不再对参数求和，也不需要把 $m$、$v$ 传回所有卡。

ZeRO-3 的区别是平时也只保存本地参数段，在执行需要它们的模块前才收集完整参数，用后按策略释放。在任何阶段，每个参数用于更新的都是合并后的梯度；变化的是谁存它、谁更新它、何时把计算所需的数据拿回来。上述数组是通信概念例子，实际系统会分 bucket、与反向交错执行。

用一个简化模型计数：$N>1$，所有参数和梯度都以相同的 2 字节精度通信，每次 optimizer update 只对应一轮所讨论的梯度同步；gather 出的完整参数用后释放，前向和反向各取回一遍。设完整参数或梯度的通信载荷为 $S=2\Psi$ 字节，按 ring collective 的**单卡发送字节数**来算，则一次 All-Gather 或 Reduce-Scatter 各约为：

$$
C=\frac{N-1}{N}S.
$$

ring collective 指把进程逻辑上排成环，分块传递数据。在这个均衡模型里，完整载荷分成 $N$ 块，每块 $S/N$ 字节，一次 All-Gather 或 Reduce-Scatter 各走 $N-1$ 轮，因此每卡发送 $(N-1)S/N$ 字节。ring All-Reduce 可以拆成一次 Reduce-Scatter 加一次 All-Gather，所以发送量为 $2C$。这是逻辑通信模型，真实通信库还会选择其他算法。

接收量在这个模型中相同；若统计“发送加接收”，各项整体再乘 2，比例不变。

| 配置 | 一次更新的逻辑通信安排 | 理想发送量 |
|------|------------------------|------------|
| DDP | backward 中/结束时对梯度 All-Reduce | $2C$ |
| ZeRO-1/2 | 梯度归约到负责人；更新后 All-Gather 参数 | $C+C=2C$ |
| ZeRO-3 | forward 取参数；backward 再取参数并归约梯度 | $C+C+C=3C$ |

这说明，在上述假设下，ZeRO-1/2 可以达到与 DDP 相同的通信字节量，ZeRO-3 则是 $3/2=1.5$ 倍。这是对[ZeRO 论文第 7 节](https://arxiv.org/html/1910.02054v3#S7)通信分析的简化，不代表所有实现和训练设置都有这个固定成本。

以本篇 7B / 8 GPU 为例，$S=14$ GB，$C=12.25$ GB；DDP、ZeRO-1/2 和 ZeRO-3 分别得到 24.5、24.5、36.75 GB 的单卡发送量。这仍是理论算术，不是测得的带宽或耗时。

实际通信开销还受以下因素影响：

1. **通信 dtype 与存储 dtype 可以不同。** 如果梯度按 FP32 传输、参数按 BF16 传输，两种载荷大小就不相等，不能直接套 1.5 倍。
2. **同步次数不同。** 梯度累积可能改变归约时机；不能假定 DDP 与 ZeRO 在每个 microbatch 上发出相同次数的 collective。
3. **通信能否与计算重叠。** Stage 1/2 的参数同步需要等待对应更新完成；Stage 3 的参数必须在模块计算前到位。时机变了，等待时间也可能变。
4. **bucket 大小与硬件拓扑。** 小消息多会增加延迟，大 bucket 会增加临时显存；跨节点链路、预取和通信计算重叠都会影响吞吐。

因此，“字节量相同”不等于零开销，“Stage 3 是 1.5 倍字节量”也不等于训练慢 50%。应测量完整训练步的时间。

<span id="4-真正会-oom-的是某一时刻的总占用"></span>

<span id="4-训练过程中的显存峰值"></span>

## 4. 哪些数据同时存在，决定显存峰值

再看一次更新的时间顺序：参数与 Adam 历史状态跨步保留；前向会增加待反向使用的激活；反向一边产生梯度、一边释放不再需要的激活；optimizer step 读梯度并更新状态，完成后梯度可以清理。ZeRO-3 还会在前向、反向之间插入参数收集和释放。因此状态表只是各类数据的存储布局，同一时刻还可能保存着临时收集的完整参数和通信缓冲。

在每个时刻把同时占用的显存相加，再取最大值，就是训练峰值：

$$
M_{\text{peak}}
=\max_t\left[
M_{\text{resident}}(t)+M_{\text{activation}}(t)
+M_{\text{transient}}(t)+M_{\text{other}}(t)
\right].
$$

此处 $t$ 表示运行时间，$M_{\text{resident}}$ 是该时刻驻留的模型状态，$M_{\text{activation}}$ 是保存的激活，$M_{\text{transient}}$ 是不与前两项重复计算的临时张量，$M_{\text{other}}$ 是内核、通信库及所选统计口径下的其他占用。不能把不同时间的各项最大值相加，再当作它们一定会同时出现。

重点检查以下几处：

- **Backward**：新梯度、尚未释放的激活和归约 bucket 可能同时存在。
- **Stage 3 参数 gather/预取**：当前模块完整参数与下一模块预取参数可能重叠。
- **Optimizer step**：可能出现 FP32 梯度转换、更新工作区或额外张量副本。
- **初始化与保存**：先构造完整模型再分片，或导出时集中完整权重，可能触发独立于稳定训练步的峰值。

[DeepSpeed 显存估算文档](https://deepspeed.readthedocs.io/en/stable/memory.html)会额外考虑最大层及实现中的缓冲；这也是其估算值可能高于本篇理想状态表的原因。

**ZeRO Stage 1/2/3 不会自动把激活除以 $N$。** 固定每卡 microbatch 和序列长度时，每卡仍处理自己的样本并保存相应激活。固定 global batch 后加卡可能因每卡样本更少而降低激活，这是 batch 分配的变化。Activation Checkpointing、Activation Offloading 或专门的激活分片需要单独配置，不能从“开启 Stage 3”自动推出。ZeRO 原论文还讨论了独立的 ZeRO-R 优化，也不应与三个模型状态分片阶段混为一谈。

<span id="41-怎样做可复现的对比"></span>

<span id="41-不同-zero-阶段的对比测试"></span>

### 4.1 怎样公平比较不同 ZeRO 阶段

上面的 7B 数字是按指定 dtype 推算的模型状态大小。要比较实际训练显存，还需要完整训练脚本，并固定配置、按同一种方式测量。

比较 stage 时，至少固定并记录：

- 模型版本与实际参数量，全参数训练还是 LoRA，参数/梯度/优化器与通信 dtype
- GPU 数量、互联拓扑，PyTorch、DeepSpeed、CUDA 与通信库版本
- 序列长度、每卡 microbatch、梯度累积次数、global batch、有效 token 数
- activation checkpointing、bucket、预取、通信重叠与 offload 设置
- 预热后每个 rank 的峰值，再取各 rank 最大值；区分 allocated、reserved 和设备总占用
- 同步计时后的完整 optimizer update 耗时、有效 tokens/s，以及相同设置下的 loss/收敛表现

还应单独检查初始化和 checkpoint 保存能否通过。某个 batch 是否能运行，仍需通过实际测试确认。

<span id="5-offloadinfinity-与-zero容量和带宽一起看"></span>

<span id="5-offloadinfinity-与-zero-的容量和通信取舍"></span>

## 5. 显存还不够，或通信太慢时怎么办

### 5.1 ZeRO-Offload

[ZeRO-Offload](https://www.deepspeed.ai/tutorials/zero-offload/)把优化器相关状态与计算移到 CPU，GPU 继续承担主要前向与反向计算。这能减少 GPU 驻留状态，但会消耗 CPU RAM、CPU 计算与 CPU–GPU 传输带宽。

微软团队的原图把两个问题放在一起：上面是“状态放在哪”，下面是“什么时候搬、在哪里算”。先看 GPU 与 CPU 之间的边界，再看时间线。

<img src="/images/llm-series/sourced/mt-zero-offload-overview-original.png" alt="微软 ZeRO-Offload 原图：GPU保存计算参数，CPU保存梯度分片和优化器状态，下方展示前后向与更新及传输重叠" style="background: #fff;" />

*原图未改动：DeepSpeed Team，[DeepSpeed: Extreme-scale model training for everyone，Figure 7](https://www.microsoft.com/en-us/research/blog/deepspeed-extreme-scale-model-training-for-everyone/)；[原始图片](https://www.microsoft.com/en-us/research/wp-content/uploads/2020/09/DeepSpeed-3_Figure-2_section-1.png)。图片版权归 Microsoft，网页未单独标明图片开放许可。*

1. **上方蓝条仍在 GPU**：offload 不是让 CPU 执行整个模型。图示的计算用参数留在 GPU，橙色梯度先产生，再按分工传到主机。
2. **点纹块在 CPU 内存**：橙色点纹是 CPU 上的梯度，绿色点纹是优化器状态；它们还按数据并行进程分片。CPU 容量也必须够用。
3. **下方两条时间流要一起读**：蓝色 GPU 区间做前向和反向，绿色 CPU 区间做参数更新；黄色、橙色条表示两个方向的传输。传输可以与部分计算重叠，但依赖关系和链路开销仍在。
4. **这是调度示意，不是计时结果**：条宽不能用来推出固定加速比例；图中没有 NVMe 层，下一节 ZeRO-Infinity 才进一步讨论它。

直观估算：若某条链路每步至少要搬运 $D$ 字节，有效带宽为 $B$ 字节/秒，单是传输就需要约 $D/B$ 秒的链路服务时间。它可以和计算重叠一部分，但不会因 pinned memory 或异步拷贝而消失。必须同时检查主机内存容量、CPU optimizer 耗时和传输瓶颈；CPU 内存足够，并不意味着训练吞吐也能满足要求。

### 5.2 ZeRO-Infinity

ZeRO-Infinity 在 ZeRO-3 的基础上使用 CPU/NVMe 分层存储，进一步扩展可容纳的状态规模。按需读取、预取和缓冲调度决定能否让 GPU 及时拿到数据；NVMe 容量大不代表它具有 GPU 显存带宽。其设计和适用条件见 [ZeRO-Infinity 论文](https://arxiv.org/abs/2104.07857)。

评估时要测存储的持续吞吐、CPU–GPU 链路及它们是否争用，而不是只比较标称延迟。GPU 上仍需给活动模块、激活和通信缓冲留空间；“能把状态存下”与“训练时间可以接受”是两个问题。

### 5.3 ZeRO++

[ZeRO++](https://www.deepspeed.ai/tutorials/zeropp/)组合了量化参数通信、层级分片和量化梯度通信等机制，目标是减少低带宽链路上的传输。它不只是把所有 All-Gather 统一改成 INT8；不同机制也可能改变冗余存储与数值误差。是否适合目标集群，要连同通信拓扑、显存和收敛一起验证。

<span id="6-deepspeed-配置先能解释再去调参"></span>

## 6. DeepSpeed Stage 2 配置示例

下面是 **8 个数据并行进程**的 Stage 2 配置示例。它展示参数含义，需配合模型、数据和训练入口使用；实际状态存储还受优化器与框架实现影响，不一定恰好采用上面的 $16\Psi$ 配置：

```json
{
  "train_micro_batch_size_per_gpu": 2,
  "gradient_accumulation_steps": 4,
  "train_batch_size": 64,
  "bf16": { "enabled": true },
  "zero_optimization": {
    "stage": 2,
    "allgather_partitions": true,
    "allgather_bucket_size": 50000000,
    "overlap_comm": true,
    "reduce_scatter": true,
    "reduce_bucket_size": 50000000,
    "contiguous_gradients": true
  }
}
```

micro-batch 是每张卡一次前向/反向处理的小批次；gradient accumulation 是先累计若干个小批次的梯度，再更新一次参数；global batch 是所有数据并行卡在这次更新中合计处理的样本数。本例每卡一次 2 条、累计 4 次、共 8 张卡，因此 global batch 为 $2\times4\times8=64$ 条样本；序列长度不同或使用 packing 时，样本数相同不代表有效 token 数相同。完整运行仍需模型、数据、优化器、学习率及兼容的训练入口。

| 参数 | 要核对什么 |
|------|------------|
| `stage` | 分片到优化器状态、梯度还是参数 |
| `allgather_partitions` | 更新后用 All-Gather 还是等效广播同步参数 |
| `overlap_comm` | 尝试重叠通信与计算；更高的同时驻留量可能增加峰值 |
| `*_bucket_size` | 单位是元素个数，不是字节；本例 5000 万个 BF16 元素约 100 MB，但总缓冲不一定只有一份 |
| `contiguous_gradients` | 用连续缓冲组织梯度；仍要计算该缓冲的实际占用 |

参数语义以[DeepSpeed 配置文档](https://www.deepspeed.ai/docs/config-json/)与所用版本为准。修改 stage 后应重新验证；Stage 3 还有预取和参数保留相关设置，不能只照搬 Stage 2 的峰值估计。

<span id="7-怎么选-stagefsdp-又是什么关系"></span>

<span id="7-zero-阶段选择与-fsdp"></span>

## 7. 怎样选 ZeRO 阶段，FSDP 又有什么不同

先确定主要瓶颈：模型状态放不下，还是长序列激活太大，或者通信已经占据大部分时间？

1. 固定模型、训练精度、global batch 和数据处理方式，先跑一组基准配置；如果这组配置 OOM，也记下发生在哪一步。
2. 用本篇公式比较模型状态，再实际测试 Stage 1/2/3 的峰值和吞吐。根据容量与速度目标选择配置；仅凭参数量阈值，无法决定合适的阶段。
3. Stage 3 仍 OOM 时，检查活动模块、预取、激活和初始化，而不是只增加分片数。
4. 需要 offload 时，同时评估主机容量和端到端训练时间。$N=1$ 时没有其他卡分摊，单靠分片本身不会减少状态总量。

PyTorch FSDP（Fully Sharded Data Parallel）的全分片用法与 ZeRO-3 解决相似问题：分片参数、梯度和优化器状态，用计算时的参数收集换取较小的常驻量。不同 FSDP API、分片策略和重分片设置，会改变参数何时收集、何时释放；不能把“用了 FSDP”直接替换成同一个峰值公式。[PyTorch FSDP2 教程](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html)展示了按模块分片的使用方式。

DeepSpeed 和 FSDP 的选择应依据现有训练栈、模型兼容性、并行/offload 需求及可复现测试。具体速度和集成成本取决于这些条件。

<span id="8-小结与下一篇前置知识"></span>

<span id="8-小结与回到-sft"></span>

<span id="8-小结与-sft-配置"></span>

## 8. 小结：状态大小、显存峰值和速度要分别算

理解 ZeRO 的关键是三个问题：**哪些状态参与分片？计算时何时需要收集它们？收集会增加多少显存和通信时间？**

- $16\Psi$ 来自明确的参数、梯度、master weights 和 Adam $m$、$v$ 配置；FP32 梯度的同类配置是 $18\Psi$。
- Stage 1/2/3 依次分片优化器侧状态、梯度、参数。分片状态表与运行峰值必须分开。
- 通信字节、通信耗时和训练吞吐是不同指标，激活也不会随 ZeRO stage 自动分片。

下一篇[《大模型显存（三）：微调时怎样节省显存》](/posts/2026-06-16-llm-mem-opt-5-sft/)把重点放到实际 SFT：先检查数据和 loss，再根据瓶颈选择 LoRA、packing、激活重算等方法。ZeRO 在需要多卡分摊状态时才加入这套流程；第三篇第 4 节会解释 LoRA 的可训练参数为什么更少，不能把本篇的全参数 Adam 估算原样套过去。

## 参考资料

1. Rajbhandari et al. [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054)，重点为模型状态与通信分析。
2. DeepSpeed [ZeRO 官方教程](https://www.deepspeed.ai/tutorials/zero/)、[ZeRO-3 配置与限制](https://deepspeed.readthedocs.io/en/stable/zero3.html)、[显存估算](https://deepspeed.readthedocs.io/en/stable/memory.html)。
3. PyTorch [Adam 算法](https://docs.pytorch.org/docs/2.8/generated/torch.optim.Adam.html)与 [AMP/autocast](https://docs.pytorch.org/docs/2.8/amp.html)。
4. DeepSpeed [ZeRO-Offload](https://www.deepspeed.ai/tutorials/zero-offload/)；Rajbhandari et al. [ZeRO-Infinity](https://arxiv.org/abs/2104.07857)；DeepSpeed [ZeRO++](https://www.deepspeed.ai/tutorials/zeropp/)。
5. DeepSpeed [JSON 配置](https://www.deepspeed.ai/docs/config-json/)与 PyTorch [FSDP2 教程](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html)。
