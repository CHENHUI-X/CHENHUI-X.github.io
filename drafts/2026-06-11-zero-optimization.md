---
title: ZeRO 优化全解：从 ZeRO-1 到 ZeRO-3 的显存与通信权衡
published: 2026-06-11
description: 深度解析 DeepSpeed ZeRO 的三个阶段：优化器状态分片（ZeRO-1）、梯度分片（ZeRO-2）、参数分片（ZeRO-3），以及 ZeRO-Offload/Infinity 的扩展机制。含显存公式推导、通信模式对比、H100 实测性能数据与实际选择策略。
category: Deep Learning
tags:
- distributed-training
- llm parallelism
- deepspeed
- zero-optimizer
draft: true
---

## 引言：训练的"多副拷贝"问题

GPT-3 175B 在 FP32 下仅权重就 ~700 GB。Adam 训练时还要配 momentum + variance（FP32）——加起来超过 2 TB。一张 A100 80GB 连权重都放不下，更别说优化器状态。

标准的数据并行（DDP）做法：每张 GPU 各存一份完整模型副本，各自算各自的数据子集，做完梯度 All-Reduce 同步后独立更新参数。这在小模型时代没问题，但大模型时代暴露了一个根本缺陷：

**100 张 GPU 训一个 7B 模型 → 99 份模型状态是纯冗余**。

每一张卡都存了同一份参数、同一份梯度、同一份 Adam 的 m 和 v。不仅浪费，还会让显存成为瓶颈——因为大部分显存消耗不在模型权重上，而在优化器状态上。

ZeRO（Zero Redundancy Optimizer）就是为了消灭这种冗余而生的。它由 Microsoft Research 在 2019 年提出[^1]，核心洞察很简单：

> **每张卡不需要存完整模型状态。只需要存 1/N，用时从其他卡借。**

这篇文章从显存解剖开始，逐步推导 ZeRO 的三个阶段，最终给出实际选择策略。

[^1]: Rajbhandari, S., Rasley, J., Ruwase, O., & He, Y. (2020). ZeRO: Memory optimizations for distributed deep learning training. arXiv:1910.02054.

---

## 1. 训练的显存解剖：$16\Psi$ 公式

搞清楚 ZeRO 在优化什么，必须先看清训练时显存到底被谁吃了。

### 1.1 模型状态（Model States）

| 组件 | 精度 | 每参数字节 | 7B 模型 | 说明 |
|------|------|-----------|---------|------|
| 参数 $\Psi$ | BF16/FP16 | 2 B | 14 GB | 前向+反向时用 |
| 梯度 | BF16/FP16 | 2 B | 14 GB | 反向产生，更新时消费 |
| FP32 参数副本 | FP32 | 4 B | 28 GB | 混合精度的 master copy |
| Adam momentum $m$ | FP32 | 4 B | 28 GB | 一阶矩估计 |
| Adam variance $v$ | FP32 | 4 B | 28 GB | 二阶矩估计 |
| **合计** | | **16 B** | **112 GB** | |

这就是 **$16\Psi$** 公式：每个参数在训练时需要 **16 字节** 的模型状态。

Adam 的 m 和 v 必须用 FP32 存——它们在百万步训练中持续累积小增量，FP16 的动态范围不足以区分大小梯度贡献的比值，小梯度会被四舍五入到 0。FP32 master copy 同理：优化器步产出的参数增量在 FP16 下可能小到直接被舍弃。

### 1.2 残差状态（Residual States）

激活值、临时缓冲区、显存碎片。激活值是为反向传播保留的中间输出，对于 seq_len=2048、batch_size=4 的 transformer，每层激活值可达几 GB。这部分靠 **gradient checkpointing（激活重计算）** 解决：丢掉大部分激活，反向时临时重算。ZeRO 专注模型状态优化，与 checkpointing 互补。

### 1.3 冗余有多严重

256 张 A100 训 7B 模型：

$$256 \times 112\text{ GB} = 28,672\text{ GB 模型状态}$$

其中真实需要只有 112 GB——**28,560 GB 是冗余**，占比 99.6%。

ZeRO 要消灭的就是这 99.6%。

---

## 2. 三种切法：ZeRO 的三个阶段

ZeRO 按分片深度分三层，每层在上层基础上再多切一类数据。下面用一张图建立宏观概念：

![ZeRO 分片结构：Stage 1/2/3 下 Parameters/Gradients/Optimizer States 在各 GPU 的分布](https://assets.mbrenndoerfer.com/notebooks/6_zero_optimization_files/zero-partitioning-structure.png)

> 图片说明：绿色 = 每个 GPU 持有的数据。Stage 1 只在 Optimizer States 上有差异；Stage 2 扩展到 Gradients；Stage 3 连 Parameters 也分片。

### 2.1 ZeRO-1：切优化器状态（$P_S$）

**只切优化器状态**。各 GPU 仍持有完整的参数和梯度副本。

- 前向/反向和标准 DDP 完全一样
- 梯度做完 All-Reduce 后，每张卡仍有完整梯度
- 但 optimizer step 时，每张卡只更新自己负责的那部分参数——只存 1/N 的 $m$ 和 $v$

单卡模型状态显存从 $16\Psi$ 降到 $4\Psi + \frac{12\Psi}{N}$（参数 + 梯度完整，优化器状态分片）。

以 7B 模型 + 8 GPU：从 112 GB/卡 降到 ~60.5 GB/卡。

**与标准 DDP 的通信量相同，零额外通信开销**——只是把 All-Reduce 拆成 Reduce-Scatter + All-Gather 来配合分片。

对于 N=64 的 DP 组，优化器状态冗余从 63 份直接归零。这是性价比最高的 ZeRO 阶段——几乎所有分布式训练场景都值得开。

### 2.2 ZeRO-2：再切梯度（$P_{os+g}$）

**在 ZeRO-1 基础上再切梯度**。每张 GPU 只持有完整参数 + 1/N 的梯度和优化器状态。

实现上很自然：反向传播时直接用 **Reduce-Scatter** 聚合梯度，聚合完每张卡天然就持有 1/N 的梯度分片。不需要额外步骤（"顺便"的事）。

单卡模型状态显存：$4\Psi + \frac{12\Psi}{N}$ → $2\Psi + \frac{14\Psi}{N}$。

7B/8 GPU：~60.5 GB → ~32.3 GB。

通信量仍与标准 DDP 相同。代价是 optimizer step 后需要 **All-Gather 参数**（因为每卡只更新了自己分片内的参数，需要拼回完整），但这一额外通信开销和 All-Reduce 的通信量在同一量级，总通信量不变。

### 2.3 ZeRO-3：全套分片（$P_{os+g+p}$）

**参数也切了**。每张 GPU 只持有 1/N 的参数 + 1/N 的梯度 + 1/N 的优化器状态。

显存降到：$\frac{16\Psi}{N}$（线性缩放）。7B/8 GPU：~14 GB/卡。

但代价来了：**前向和反向时，每张卡需要从其他卡 All-Gather 参数分片来拼出完整模型**，这一步需要额外通信。每层计算前先做一次参数 All-Gather → 计算 → 释放远端参数分片。

ZeRO-3 的通信开销约为标准 DDP 的 **1.5x**（取决于网络带宽和模型结构）。在高带宽互联（NVSwitch ~900 GB/s）上影响可控，但在 PCIe 或以太网环境下可能成为瓶颈。

---

## 3. 通信模式对比

| Stage | 前向通信 | 反向通信 | 参数更新通信 | 总通信量（相对 DDP） |
|-------|---------|---------|-------------|-------------------|
| DDP | 无 | 1× All-Reduce | 无（各卡独立更新） | 1× |
| ZeRO-1 | 无 | 1× Reduce-Scatter | 1× All-Gather | 1× |
| ZeRO-2 | 无 | 1× Reduce-Scatter | 1× All-Gather | 1× |
| ZeRO-3 | 每层 1× All-Gather | 每层 1× All-Gather + Reduce-Scatter | 1× All-Gather | **~1.5×** |

关键结论：

- **ZeRO-1/2 总通信量和 DDP 完全一样**——只是改变了数据的"在哪里"
- **ZeRO-3 的选择本质上是"用通信换显存"**
- 显存宽裕 → ZeRO-1/2；显存紧张 → ZeRO-3

---

## 4. 性能实测数据（H100）

参考 Josh Angel 在 8×H100 上用 Gemma2-9B 做 Continued Pretraining 的实测[^2]：

[^2]: Josh Angel. "ZeRO Optimization Strategies for Large-Scale Model Training". Hugging Face Blog, 2025. https://huggingface.co/blog/josh-a/zero-optimization-strategies

### 4.1 显存占用

| ZeRO Stage | 峰值显存/GPU | 相对节省 | 理论最大模型规模 |
|------------|------------|---------|----------------|
| ZeRO-0 (基线 DDP) | 76 GB | — | ~7B |
| ZeRO-2 | 45 GB | 40% | ~13B |
| ZeRO-3 | 28 GB | 65% | ~30B |

ZeRO-3 将显存需求降到原来的 1/3，使得同一批硬件能训 4 倍大的模型。

### 4.2 吞吐量对比

| Configuration | tokens/sec/GPU | 相对性能 |
|---------------|---------------|---------|
| ZeRO-0 | 2,847 | 100%（基线） |
| ZeRO-2 | 2,698 | 94.7% |
| ZeRO-3 | 2,234 | 78.5% |

ZeRO-3 吞吐损失 ~21%，但这个代价换来的是训得了原本不可能训的模型。

接下来这张图直观展示了显存和吞吐的 trade-off：

![ZeRO 不同 Stage 下显存 vs GPU 数量折线图](https://assets.mbrenndoerfer.com/notebooks/6_zero_optimization_files/zero-memory-savings-by-stage.png)

![不同模型尺寸下各 Stage 的单卡显存对比](https://assets.mbrenndoerfer.com/notebooks/6_zero_optimization_files/zero-memory-by-model-size.png)

![ZeRO 各 Stage 吞吐量随 GPU 数量扩展效率](https://assets.mbrenndoerfer.com/notebooks/6_zero_optimization_files/zero-throughput-scaling.png)

### 4.3 关键发现：Batch Size 影响 ZeRO 效率

Batch size 越大，ZeRO-3 的通信开销分摊效果越好（因为通信频率降低）：

| Effective Batch Size | ZeRO-0 | ZeRO-2 | ZeRO-3 | 最优选择 |
|---------------------|--------|--------|--------|---------|
| 32-64 | 100% | 92% | 72% | ZeRO-0 |
| 96-144 | 100% | 95% | 78% | ZeRO-2 |
| 192-384 | 100% | 96% | 85% | ZeRO-2/ZeRO-3 |
| 512+ | 100% | 97% | 89% | ZeRO-3 可行 |

**实践建议**：调 ZeRO 之前先定 batch size，再根据 batch size 对应的最优 stage 选型。

---

## 5. 扩展：Offload、Infinity 与 ZeRO++

当单节点显存仍然不够时，ZeRO 还有三个重量级扩展：

### 5.1 ZeRO-Offload[^3]

将优化器状态和优化器计算卸载到 CPU 内存。GPU 负责计算密集的前向/反向，CPU 负责 optimizer step：

- CPU 内存通常比 GPU 显存大得多（512 GB-2 TB vs 80 GB）
- 代价是 CPU↔GPU 的数据传输，通常用 pinned memory + 异步拷贝降低开销
- 适合：单卡训 10B+ 模型

[^3]: Ren, J., et al. (2021). ZeRO-Offload: Democratizing Billion-Scale Model Training. USENIX ATC '21.

### 5.2 ZeRO-Infinity[^4]

在 Offload 基础上再加一层 NVMe SSD：

- GPU 显存 → CPU 内存 → NVMe 磁盘，三层存储层次
- 可训万亿参数级别模型
- 代价是 NVMe 的读写延迟（~100 µs vs GPU HBM 的 ~1 µs），需要复杂的预取策略隐藏延迟

[^4]: Rajbhandari, S., et al. (2021). ZeRO-Infinity: Breaking the GPU Memory Wall for Extreme Scale Deep Learning. arXiv:2104.07857.

### 5.3 ZeRO++[^5]

在 ZeRO-3 基础上做通信压缩：

- 用量化技术压缩跨节点的 All-Gather 通信（FP16 → INT8）
- 减少低带宽互联（跨机以太网/InfiniBand）的通信压力
- 在数百 GPU 规模下能提升 20-30% 吞吐

[^5]: Guan, L., et al. (2023). ZeRO++: Extremely Efficient Collective Communication for Giant Model Training. arXiv:2306.10209.

---

## 6. 选择决策树

结合上述分析，给出一个系统化的选择流程：

```text
1. 模型能放进单卡显存吗（标准 DDP）？
   ├── 能 → 用 ZeRO-0，最大吞吐
   └── 不能 ↓

2. 加上优化器+梯度分片（ZeRO-2）能放下吗？
   ├── 能 → 用 ZeRO-2，吞吐损失 ~5%
   └── 不能 ↓

3. 必须用 ZeRO-3 ↓

4. 通信带宽如何？
   ├── NVSwitch/高带宽 InfiniBand → ZeRO-3 开销可接受
   └── PCIe/低带宽以太网 → 考虑 ZeRO-2 + 减小 batch size 或增大 DP 组

5. 是什么场景？
   ├── 研究探索 → 优先模型容量：ZeRO-3
   └── 生产训练 → 优先稳定性：ZeRO-2
```

---

## 7. DeepSpeed ZeRO 配置示例

```json
{
  "train_batch_size": 128,
  "gradient_accumulation_steps": 4,
  "fp16": {
    "enabled": true
  },
  "zero_optimization": {
    "stage": 2,
    "offload_optimizer": {
      "device": "none"
    },
    "allgather_partitions": true,
    "allgather_bucket_size": 5e8,
    "overlap_comm": true,
    "reduce_scatter": true,
    "reduce_bucket_size": 5e8,
    "contiguous_gradients": true
  }
}
```

关键参数说明：

- `stage`：0/1/2/3，选哪个阶段
- `offload_optimizer.device`：`"cpu"` 启用 ZeRO-Offload，`"nvme"` 启用 Infinity
- `overlap_comm`：通信与计算重叠，几乎必开
- `bucket_size`：通信 Buffer 大小，影响 GPU 利用率和延迟的平衡

---

## 8. FSDP vs DeepSpeed ZeRO

PyTorch 的 FSDP（Fully Sharded Data Parallel）和 ZeRO-3 在理念上高度一致——都是参数+梯度+优化器状态全分片。主要差异：

| | DeepSpeed ZeRO-3 | PyTorch FSDP |
|---|---|---|
| 实现 | Microsoft DeepSpeed 库 | PyTorch 原生（torch.distributed） |
| Offload | CPU + NVMe（Infinity） | CPU only |
| 混合并行 | 与 TP/PP 组合灵活 | 主要与 TP 组合 |
| 生态成熟度 | 更成熟，论文验证更多 | 快速追赶中 |
| 调试体验 | 配置复杂但文档丰富 | 原生 PyTorch 生态，调试友好 |

实践建议：如果你已经在用 HuggingFace Trainer → 优先 FSDP（集成更好）。如果用 Megatron 或自定义训练循环 → 优先 DeepSpeed（并行策略组合更成熟）。

---

## 9. 总结

| 核心问题 | 答案 |
|---------|------|
| ZeRO 解决了什么 | 数据并行中 4 类模型状态（参数、梯度、m、v）的冗余存储 |
| ZeRO-1 是什么 | 只切优化器状态 + 梯度上 Reduce-Scatter，无额外通信 |
| ZeRO-2 是什么 | ZeRO-1 + 设异量侧切梯度，通信量不变 |
| ZeRO-3 是什么 | 全切参数+梯度+优化器状态，通信 1.5×，但显存线性缩放 |
| 该用哪个 | 能放下 → ZeRO-0；差一点 → ZeRO-2；实在放不下 → ZeRO-3 |
| 显存极限 | ZeRO-Offload（CPU）/ Infinity（NVMe）/ ++（通信压缩） |

---

## 参考资料

1. Rajbhandari, S., Rasley, J., Ruwase, O., & He, Y. (2020). ZeRO: Memory optimizations for distributed deep learning training. SC '20. [arXiv:1910.02054](https://arxiv.org/abs/1910.02054)
2. Michael Brenndoerfer. "ZeRO Optimization: Stages 1, 2, and 3 Explained". 2026. [mbrenndoerfer.com](https://mbrenndoerfer.com/writing/zero-optimization-stages-optimizer-gradient-parameter-partitioning)
3. Josh Angel. "ZeRO Optimization Strategies for Large-Scale Model Training". Hugging Face Blog, 2025. [huggingface.co](https://huggingface.co/blog/josh-a/zero-optimization-strategies)
4. ele-yufo. "DeepSpeed Stage 2 与 Stage 3 对比". [ele-yufo.com](https://www.ele-yufo.com/archives/1924)
5. luhengshiwo. "大模型训练框架（三）DeepSpeed". [GitHub: LLMForEverybody](https://github.com/luhengshiwo/LLMForEverybody)
6. Ren, J., et al. (2021). ZeRO-Offload: Democratizing Billion-Scale Model Training. USENIX ATC '21.
7. Rajbhandari, S., et al. (2021). ZeRO-Infinity: Breaking the GPU Memory Wall. arXiv:2104.07857.
8. Guan, L., et al. (2023). ZeRO++: Extremely Efficient Collective Communication. arXiv:2306.10209.