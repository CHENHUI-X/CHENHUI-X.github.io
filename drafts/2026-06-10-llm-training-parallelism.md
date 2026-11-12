---
title: LLM 分布式训练并行策略完全解析：从 DP 到 5D 并行
published: 2026-06-10
description: 系统梳理 LLM 训练中的数据并行、张量并行、流水线并行、专家并行、序列并行、上下文并行，以及它们在 Megatron 框架中的组合策略与选择指南。
category: Deep Learning
tags:
- distributed-training
- llm parallelism
draft: false
---

## 引言：为什么单卡训练已成为过去式

想象你要搬一座图书馆。一个人能搬完一摞书，但要把整座图书馆运到新地址，必然需要分工——有人打包、有人运输、有人上架。LLM 训练也是同样的故事。

当模型参数量突破十亿级别，**单卡训练的时代就结束了**。以 Llama 3 70B 为例，仅模型权重在 BF16 精度下就需要约 140 GB 显存——远超 H100 / A100 单卡 80GB 的容量。

:::note
不熟悉 BF16 / FP16 / FP8 这些精度，或想知道 "70B → 140 GB" 是怎么算出来的？可以先读[模型运行时要存什么](/posts/2026-06-12-llm-mem-opt-1-fundamentals/)；量化方法另见[大模型量化：从 INT4 编码到 GPTQ、AWQ](/posts/2026-05-16-llm-quantization-guide/)。
:::

更何况训练还要存：

- **优化器状态**（Adam 需要 momentum + variance，FP32 下相当于 2× 模型参数）
- **梯度**（与模型参数同样大小）
- **中间激活值**（与序列长度、batch size 成正比，长序列场景下可能比模型本身还大）

把模型放到多张 GPU 上听起来简单，但**不同的拆分方式会带来数量级的性能差异**。有些组合理论上很美但被通信延迟吞噬，有些看似奇怪却能撑起千卡集群。

经过几年实践沉淀，业界形成了一套核心并行策略组合：**DP（数据并行）、TP（张量并行）、PP（流水线并行）、EP（专家并行）、SP（序列并行）、CP（上下文并行）**。本文逐个梳理它们的原理、权衡，以及如何在 Megatron 框架中组合使用。

## 准备知识：通信原语小百科

理解并行策略前，先建立一个直觉：**多 GPU 训练的本质是"算"和"通信"的反复权衡**。下面这些集合通信原语会反复出现，先记住它们能让后文好懂很多。

| 原语 | 通俗解释 |
|------|---------|
| **All-Reduce** | 所有人把手里的数加起来，再每人拿一份完整结果（求和 + 广播） |
| **Reduce-Scatter** | 求和后切片：每人只拿一部分结果 |
| **All-Gather** | 每人把自己手里的一份拼给所有人（不做计算，只收集） |
| **All-to-All** | 每人把不同的部分发给不同的人（路由洗牌） |

直觉上：**All-Reduce ≈ Reduce-Scatter + All-Gather**。这个等式后面就会派上用场——分布式优化器正是把 All-Reduce 拆开做来省显存。

## 数据并行（Data Parallelism, DP）

### 一句话理解

**模型完整复制到每张 GPU，每张卡处理不同的数据。**

> 类比：100 个学生考同一份试卷。每人手里都有完整试题（模型），分到不同的草稿题（数据），考完把答案对一遍（梯度同步）。

![Distributed Data Parallelism](https://docs.nvidia.com/nemo/megatron-bridge/latest/_images/ddp.gif)
*图：DDP 通过 all-reduce 在每轮参数更新前同步所有 GPU 的梯度。*

### 经典 DDP 的工作流程

每张 GPU 独立做前向 + 反向，得到本地梯度。在更新参数前用 **All-Reduce** 把所有 GPU 的梯度求平均，每张卡拿到相同的梯度后独立更新参数——这样所有卡的参数始终一致。

### Distributed Optimizer：DDP 的内存优化版

经典 DDP 的痛点：**每张 GPU 都存了一份完整的优化器状态**。Adam 要存 momentum + variance（FP32），加起来约 8 字节/参数——这部分往往比模型本身还大。

分布式优化器（Distributed Optimizer，又称 ZeRO-1）的关键改进：**把优化器状态和梯度切片，分摊到 DP 组的每张 GPU 上**。

具体流程：
1. 反向传播后，做 **Reduce-Scatter**（不是 All-Reduce），梯度直接分片到各 GPU
2. 每张 GPU 只更新自己分片对应的那部分参数
3. 更新完做 **All-Gather**，让所有 GPU 拿到完整的新参数

总通信量与 All-Reduce 相同，但优化器状态显存降到 $1/N_{\text{DP}}$（$N_{\text{DP}}$ 是 DP 组大小）。

```python
from megatron.bridge.training.config import (
    ConfigContainer,
    DistributedDataParallelConfig,
    OptimizerConfig,
)

optimizer_config = OptimizerConfig(
    optimizer="adam",
    lr=3e-4,
    use_distributed_optimizer=True,   # 启用分布式优化器（ZeRO-1）
)
ddp_config = DistributedDataParallelConfig(use_distributed_optimizer=True)
```

:::tip
分布式优化器只对应 **ZeRO Stage 1**。ZeRO 还有 Stage 2（再切梯度）、Stage 3（连模型参数也切），以及 ZeRO-Offload / ZeRO-Infinity 等扩展。详细原理、显存公式、通信代价、Stage 选择策略我整理成专题：[👉 训练显存与 ZeRO 优化](/posts/2026-06-15-llm-mem-opt-4-zero/)。
:::

### DP 的优缺点

| 优势 | 劣势 |
|------|------|
| 实现最简单，工业级支持成熟 | 每张卡需要完整存模型（模型太大就放不下） |
| 吞吐量近似线性扩展 | 不能减少单步延迟 |
| 通信与计算可重叠 | 单卡装不下时无法独立使用 |

### 适用范围
- 模型能在单卡装下（通常 < 10B 参数）
- 作为其他并行策略的基础维度——DP 几乎总会出现在组合里

## 张量并行（Tensor Parallelism, TP）

### 一句话理解

**把单个层的矩阵切片，让多张 GPU 共同算完一层乘法。**

> 类比：4 个人合作算一道大矩阵乘法。每人算一列（或一行），最后通过通信把结果拼起来。

![Tensor Parallelism](https://docs.nvidia.com/nemo/megatron-bridge/latest/_images/tp1.png)
*图：TP 将单个层的参数分布到多个 GPU 上。*

### Megatron 的精妙设计：列 + 行配对

Megatron-LM 在 Transformer 层的设计极为讲究：**把列并行和行并行配对使用，让一对相邻的线性层只需要一次 All-Reduce**。下面拆开看为什么。

![TP Implementation](https://docs.nvidia.com/nemo/megatron-bridge/latest/_images/tp2.png)
*图：单个 Transformer 层中 TP 的详细切分与同步流程。*

#### 列并行（Column-Parallel）：权重按列切

线性层 $Y = XA$，将权重 $A$ 按列切分：
$$A = [A_1, A_2, \dots, A_n]$$

每张 GPU 拿到完整的 $X$，算 $Y_i = X \cdot A_i$，得到结果的"一部分列"。**这里不需要立刻通信**——结果直接作为下一层的输入即可。

#### 行并行（Row-Parallel）：权重按行切

下一层的输入正好是上一层的列切片 $X = [X_1, \dots, X_n]$。把权重 $A$ 按行切：
$$A = \begin{bmatrix} A_1 \\ A_2 \\ \vdots \\ A_n \end{bmatrix}$$

每张 GPU 算 $Y_i = X_i \cdot A_i$，得到一份"部分和"。最后用一次 **All-Reduce** 把所有 GPU 的部分和加起来，就是完整结果。

> **关键洞察**：列并行 → 行并行 中间不需要通信，只在行并行的尾部做一次 All-Reduce。这就是 Megatron 把 TP 通信压到极低的核心技巧。

#### 在 Transformer 中怎么用

一个标准 Transformer 层的两个核心模块：

- **Attention**：QKV 投影做列并行（每张 GPU 计算不同的注意力头），输出投影做行并行 → Attention 出口一次 All-Reduce
- **FFN**：上投影（up）做列并行，下投影（down）做行并行 → FFN 出口一次 All-Reduce

每层只两次 All-Reduce，通信开销可控。

### TP 的优缺点

| 优势 | 劣势 |
|------|------|
| 单层显存大幅降低 | **每层都得通信**，频率极高 |
| 单步延迟低（推理友好） | 严重依赖 NVLink 等高带宽互联 |
| 帮助削减激活显存（与 SP 配合） | 跨节点部署时性能急剧下降 |

### 关键约束

TP 组内的 GPU 必须在**同一节点**（NVLink 互联），否则跨节点 All-Reduce 会拖垮训练。**工程经验：TP ≤ 8**（即一个节点的 GPU 数）。

### 适用范围
- 单层太大装不下时
- 有 NVLink 的单节点内
- 与 SP（序列并行）搭配削减激活显存

## 流水线并行（Pipeline Parallelism, PP）

### 一句话理解

**按层切模型：不同的 GPU 负责不同的层，数据像流水线一样在卡之间流过。**

> 类比：汽车装配流水线。第 1 站装底盘，第 2 站装引擎，第 3 站装内饰……每站只做自己那一段。

![Pipeline Parallelism](https://docs.nvidia.com/nemo/megatron-bridge/latest/_images/pp.gif)
*图：PP 将模型的连续层分布到多个 GPU 上，以流水线方式处理微批次。*

### Naive PP 的问题：流水线气泡

最朴素的 PP：一个 batch 走完所有 stage 再开始下一个 batch。问题是——前面的 GPU 算完后只能干等，**大部分卡大部分时间都闲着**，称为**流水线气泡（Pipeline Bubble）**。

### 微批次（micro-batch）调度

把全局 batch 切成多个 micro-batch，错峰送入流水线。这样不同 GPU 在同一时刻处理**不同 micro-batch 的不同阶段**——气泡显著减小。

### 1F1B 调度：当前主流

**1F1B（One Forward One Backward）**：每张 GPU 交替执行前向和反向，而不是先把所有前向算完再反向。好处是反向一做完就能丢弃对应的激活值，**激活显存占用大幅降低**，同时气泡也短。

### 交错式流水线（Interleaved PP）

进一步优化：把模型切成更多块（chunks），每张 GPU 负责多个不连续的块。等上游数据时切到别的块去算，几乎榨干所有空闲时间。

![Interleaved PP](https://developer.nvidia.com/blog/wp-content/uploads/2022/09/pp-interleaved-schedule-1.png)
*图：交错式调度（virtual_pipeline_model_parallel_size=2）可显著减少气泡比例。*

```python
from megatron.bridge.models import GPTModelProvider

model_config = GPTModelProvider(
    pipeline_model_parallel_size=4,              # 4 段流水线
    virtual_pipeline_model_parallel_size=2,      # 每段拆 2 块
)
```

### PP 的优缺点

| 优势 | 劣势 |
|------|------|
| 层间几乎不通信（只传激活值） | 流水线气泡降低利用率 |
| 显存利用率高 | 微批次过小气泡过多 |
| 通信少，**适合跨节点部署** | 实现复杂，调度细节多 |

### PP vs TP：怎么选？

工程经验法则：

- **节点内（有 NVLink）**：优先 TP（高带宽足以掩盖每层通信）
- **跨节点（InfiniBand 或以太网）**：优先 PP（少量激活值传输 + 计算掩盖）
- **模型极大（数百层）**：TP + PP 组合

## 专家并行（Expert Parallelism, EP）

### 一句话理解

**MoE 模型专属：把不同的专家分到不同 GPU，每个 token 只激活少数专家。**

> 类比：医院有内科、外科、皮肤科多个专家。每个病人（token）由分诊台（Router）分到 1-2 个专家——不需要让所有专家会诊。

![Expert Parallelism](https://docs.nvidia.com/nemo/megatron-bridge/latest/_images/ep.png)
*图：EP 将 MoE 模型中的不同专家分布到不同 GPU，非专家层保持复制。*

### EP 的工作流程

1. **路由**：Router 为每个 token 选 Top-K 个目标专家
2. **All-to-All**：把 token 发送到对应专家所在的 GPU
3. **专家计算**：每张 GPU 处理分到自己的 token
4. **All-to-All 反送**：结果送回原始 GPU

注意第 2、4 步是 **All-to-All**——这是 EP 独有的通信开销点。

### ETP（Expert Tensor Parallelism）

当单个专家也大到放不下时，可以**在专家内部再做一次 TP**：

```python
model_config = GPTModelProvider(
    num_moe_experts=8,
    expert_model_parallel_size=4,      # 8 个专家分到 4 张卡（每卡 2 个）
    expert_tensor_parallel_size=2,     # 每个专家内部再做 2 路 TP
)
```

### DeepEP / HybridEP：硬件级优化

NVIDIA 为 Hopper / B200 / B300 / GB200 NVL72 提供了优化的 token 分发器：

- **DeepEP**：针对 Ampere / Hopper / B200 / B300 的高效 All-to-All
- **HybridEP**：针对 GB200 NVL72 拓扑设计的混合分发策略

通过把标准 routing 替换为优化的 "flex dispatcher"，MoE 训练吞吐量可提升 10-30%。

### Token Dropping：负载均衡

EP 的典型痛点是**负载不均**——某些专家被路由到的 token 远多于其他。当一个专家收到的 token 超出容量，整个 step 都会被拖慢。

解决方案：设置容量因子（capacity factor），超出容量的 token 直接丢弃（或由后续 layer 兜底）：

```python
from megatron.bridge.training.utils.moe_token_drop import apply_moe_token_drop

apply_moe_token_drop(
    model_config,
    moe_expert_capacity_factor=1.0,
    moe_pad_expert_input_to_capacity=True,
)
```

### EP 的优缺点

| 优势 | 劣势 |
|------|------|
| 稀疏激活，计算量与总参数解耦 | All-to-All 通信开销大 |
| 总参数量可达激活量的 8-64 倍 | 负载不均需要专门处理 |
| 适合极大参数量的高效训练 | 低并发时反而可能更慢 |

## 激活值分区：序列并行 与 上下文并行

讲完了"怎么切模型"，还有一个隐藏的资源大户：**中间激活值**。

为什么这么大？反向传播需要保留前向产生的中间张量（attention 输出、LayerNorm 输出等），它们的大小与 `batch × seq_len × hidden` 成正比。70B 模型 + 8K 序列长度的训练 step，激活值可能占用几十甚至上百 GB——长序列场景下，它会成为压垮显存的最后一根稻草。

### 序列并行（Sequence Parallelism, SP）

LayerNorm、Dropout 这些算子的计算量虽小，但激活值大小与序列长度成正比。如果不并行，它们会成为显存瓶颈。

**SP 的核心思路**：在 LayerNorm / Dropout 处沿**序列维度**切分激活值，让它们共享 TP 的通信域。

![Sequence Parallelism](https://docs.nvidia.com/nemo/megatron-bridge/latest/_images/sp.png)
*图：SP 沿序列维度将 LayerNorm/Dropout 的激活值分布到多个 GPU。*

SP 必须配合 TP 使用（`tensor_model_parallel_size > 1`）：

```python
model_config = GPTModelProvider(
    tensor_model_parallel_size=2,
    sequence_parallel=True,        # 启用序列并行（要求 TP > 1）
)
```

### 上下文并行（Context Parallelism, CP）

序列继续拉长（32K+）后，光靠 SP 还不够。**CP 把所有层的激活值都按序列维度切分**——不只 LayerNorm / Dropout，attention 也要切。

```python
model_config = GPTModelProvider(
    context_parallel_size=2,       # 2 路上下文并行
)
```

CP 的关键挑战：Attention 需要"全局"信息（任意 token 都可能要看任意位置的 KV）。CP 通过 **ring all-gather / reduce-scatter** 让 KV 状态在 CP 组内循环交换，保证 attention 仍然算的是全序列。

### SP vs CP

| 特征 | SP | CP |
|------|----|----|
| 作用范围 | 仅 LayerNorm / Dropout | **所有层**激活值 |
| 依赖 | 必须 TP > 1 | 独立维度 |
| 适用场景 | 中等序列（2K-8K） | 超长序列（8K-1M+） |

## DP 大小是自动算出来的

在 Megatron 中，**DP 不需要手动设置**，由其他维度反推：

$$N_{\text{DP}} = \frac{\text{world\_size}}{N_{\text{TP}} \times N_{\text{PP}} \times N_{\text{CP}}}$$

例：32 张 GPU，TP=2、PP=4、CP=2：
$$N_{\text{DP}} = 32 / (2 \times 4 \times 2) = 2$$

**所有并行维度的乘积必须能整除总 GPU 数**，否则配置非法。

## 组合并行：从 3D 到 5D

实际大规模训练中，这些策略**不是互斥而是组合使用**。按组合维度数，业界俗称 3D / 4D / 5D 并行。

### 3D 并行（TP + PP + DP）

Megatron-LM 最初提出的经典组合，也是大规模 LLM 训练最常见的配置。

![3D Parallelism](https://wilsonwu.me/en/blog/2025/llm-inference-parallelism-in-vllm/2-parallelism.png)
*图：四种核心并行策略的概览（TP/DP/PP/EP），它们可以灵活组合。*

举例（H20 生产环境，TP=4 × PP=2，共 8 GPU）：

```
Stage 0 (PP=0): GPU 0,1,2,3 — Layers 1-20  — TP Group 0
     ↓ 跨节点传激活值
Stage 1 (PP=1): GPU 4,5,6,7 — Layers 21-40 — TP Group 1
```

TP 组内走 NVLink，PP 段间走 InfiniBand——**硬件拓扑与并行策略的边界要对齐**。

### 4D 并行（+ EP）

模型是 MoE 时，3D 基础上叠加 EP：

![Expert Parallelism](https://wilsonwu.me/en/blog/2025/llm-inference-parallelism-in-vllm/6-ep.png)
*图：EP 在 MoE 模型中专门将不同专家分布到不同 GPU。*

```python
from megatron.bridge.models import GPTModelProvider
from megatron.bridge.training.config import ConfigContainer, OptimizerConfig

model_config = GPTModelProvider(
    # 模型并行
    tensor_model_parallel_size=2,
    pipeline_model_parallel_size=4,
    virtual_pipeline_model_parallel_size=2,

    # 激活值分区
    sequence_parallel=True,
    context_parallel_size=2,

    # MoE 专家并行
    num_moe_experts=8,
    expert_model_parallel_size=4,
)

optimizer_config = OptimizerConfig(
    optimizer="adam",
    use_distributed_optimizer=True,
)

config = ConfigContainer(model=model_config, optimizer=optimizer_config)
```

### 5D 并行（+ CP）

超长上下文（如 128K+ tokens）训练中，CP 加入成为独立维度：

$$N_{\text{DP}} = \frac{\text{world\_size}}{N_{\text{TP}} \times N_{\text{PP}} \times N_{\text{CP}}}$$

> 注：EP 与 DP 共享同一组 GPU（专家分布在 DP 维度上），不额外消耗 world_size 的可除性。

## 策略选择指南

### 按模型大小

| 模型规模 | GPU 数 | 推荐起点 |
|----------|--------|---------|
| < 1B | 1-8 | 仅 DP |
| 1-10B | 8-16 | TP=2-4 + DP |
| 10-70B | 16-64 | TP=4-8 + PP=2-4 + DP |
| 70-175B | 64-256 | TP=8 + PP=4-8 + DP |
| 175-500B | 256-1024 | TP=8 + PP=8-16 + CP=2 + DP |

### 按模型架构

**Dense 模型**：

| 参数量 | 推荐布局 |
|--------|---------|
| < 10B | 仅 DP |
| 10-30B | TP=4 + DP |
| 30-70B | TP=8 + PP=2 + DP |
| 70B+ | TP=8 + PP=4-8 + DP |

**MoE 模型**（参数量按 总参/激活 计）：

| 参数量 | 推荐布局 |
|--------|---------|
| < 20B | EP only（TP=1, PP=1） |
| 20-100B | TP=1-2 + PP=2-4 + EP=8-16 |
| 100-500B | TP=2-4 + PP=8-16 + EP=8-32 |
| 500B+ | TP=2 + PP=16 + EP=32-64 |

**Dense vs MoE 的策略优先级**：

```
Dense 模型：
  1. 模型能装单卡 → 仅 DP
  2. 装不下     → PP > TP（TP 通信密集，先用 PP 扩展）

MoE 模型：
  1. 优先最大化 EP（稀疏通信，性价比最高）
  2. 仍装不下   → PP
  3. 最后再考虑 TP（MoE 的 TP 通信开销更大）
```

### 按硬件拓扑

| 硬件拓扑 | 推荐策略 |
|----------|---------|
| 单节点 + NVLink（8 卡） | 最大化 TP（≤8），配合 DP |
| 多节点 + InfiniBand | TP 限制在节点内，PP 跨节点 |
| 多节点 + 以太网 | 最小化 TP，优先 PP 跨节点 |
| GB200 NVL72 | HybridEP + TP + PP 综合调优 |

### 按序列长度

| 序列长度 | 推荐配置 |
|----------|---------|
| < 2K | TP + PP + DP |
| 2K-8K | + SP |
| 8K-32K | + CP=2 |
| 32K+ | + CP=4-8，层级式 CP |

## 显存优化路径（按优先级）

如果某次训练 OOM 了，按这个顺序逐项尝试：

1. **降低 max_length** — 削减激活显存最直接的杠杆
2. **降低 micro_batch_size** — 直接减小一次处理的样本数
3. **启用分布式优化器** — 优化器状态降到 $1/N_{\text{DP}}$
4. **加大 TP** — 模型参数分片
5. **加上 SP / CP** — 长序列必备
6. **激活值重计算（recomputation）** — 用计算换显存
7. **CPU offload** — 优化器状态卸载到 CPU 内存（慢但有效）
8. **LoRA / QLoRA** — 不训完整参数

## 总结

LLM 训练并行策略**没有银弹**。不是哪种"更好"，而是在**模型规模、硬件拓扑、数据分布**三重约束下找到最优工程权衡。

核心要点：

- **DP** 是吞吐基础，分布式优化器让它在大模型场景仍可用
- **TP** 是低延迟关键，但受限于 NVLink 节点边界
- **PP** 是跨节点扩展主力，1F1B 和交错调度大幅减少气泡
- **EP** 是 MoE 专属利器，可与 TP / PP 灵活组合
- **SP / CP** 是激活显存管理的核心，长序列训练不可或缺
- **分布式优化器**几乎是大模型训练的标配
- **所有并行维度乘积必须整除总 GPU 数**

![Parallelism Strategies Overview](https://wilsonwu.me/en/blog/2025/llm-inference-parallelism-in-vllm/1-overview.png)
*图：分布式并行概览——不同策略在模型维度、数据维度和硬件拓扑上的权衡。*

最终，**方案选型不止于"支持哪些并行策略"**，而是在你的模型、你的硬件、你的真实负载下，验证哪种组合才是当下的最优解。

## 参考资料

1. [NVIDIA Megatron Bridge - Parallelisms Guide](https://docs.nvidia.com/nemo/megatron-bridge/latest/parallelisms.html)
2. [Wilson Wu - Comprehensive Analysis of LLM Inference Parallelism: TP/DP/PP/EP](https://wilsonwu.me/en/blog/2025/llm-inference-parallelism-in-vllm/)
3. [NVIDIA - Scaling Language Model Training to a Trillion Parameters](https://developer.nvidia.com/blog/scaling-language-model-training-to-a-trillion-parameters-using-megatron/)
4. [HuggingFace - Transformers Parallelism (FlexFlow)](https://huggingface.co/transformers/v4.12.5/parallelism.html)
5. [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053)
6. [Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM](https://arxiv.org/abs/2104.04473)
7. [Megatron Core Developer Guide](https://docs.nvidia.com/megatron-core/developer-guide/latest/)
