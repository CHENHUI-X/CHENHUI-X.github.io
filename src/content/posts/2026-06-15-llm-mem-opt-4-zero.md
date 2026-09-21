---
title: LLM 训练与推理优化（四）— 训练显存与 ZeRO 优化
published: 2026-06-15
description: 从 DDP 的多副本冗余出发，说明特定 Adam 混合精度配置下的 16Ψ 模型状态，并推导 ZeRO-1/2/3 的理想分片账本与通信取舍。
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

> 这是「LLM 训练与推理优化系列」第 4 篇。如果你忘了 $16\Psi$ 的来源，可以回到 [第 1 篇 — 显存基础](/posts/2026-06-12-llm-mem-opt-1-fundamentals/) 第 4.2 节复习一下「训练模型状态」。本篇拆开一种常见的 $16\Psi$ 配方，并推导 ZeRO 三阶段的理想分片账本。

## 0. 100 张 GPU 训 7B 模型，99 份冗余

GPT-3 175B 在 FP32 下仅权重就 ~700 GB。混合精度训练的全套模型状态超 2 TB。一张 A100 80 GB 连权重都装不下。

但即便我们有 100 张 A100，标准 DDP 的做法还是 **每张 GPU 各存一份完整模型副本**，分别算各自的数据子集，反向后做梯度 All-Reduce 同步。

100 张卡训 7B 模型：

> **100 份模型状态里，99 份是纯冗余。**

每张卡都存一份同样的参数、同样的梯度、同样的 Adam $m$ 和 $v$。这种冗余不仅浪费，还让显存成为大模型训练的瓶颈——因为大头不在权重，而在 Adam 优化器状态。

ZeRO（Zero Redundancy Optimizer）就是为消灭这种冗余而生。它的思想一句话：

> **每张卡不需要存完整模型状态，只存 1/N，用时从其他卡借。**

本篇从显存解剖开始，逐步推出 ZeRO-1/2/3，最终给出选择决策树。

## 1. 训练显存解剖：一种 $16\Psi$ 配方

[第 1 篇](/posts/2026-06-12-llm-mem-opt-1-fundamentals/) 讲过一种常见的配置：BF16 参数与梯度、独立 FP32 master copy、以及 FP32 Adam $m/v$。

| 组件 | 精度 | 字节/参数 | 7B | 用途 |
|------|------|---------|-----|------|
| 模型参数 | BF16 | 2 | 14 GB | 前向 + 反向 |
| 梯度 | BF16 | 2 | 14 GB | 反向产出，更新时消费 |
| FP32 master copy | FP32 | 4 | 28 GB | 优化器实际更新对象 |
| Adam momentum $m$ | FP32 | 4 | 28 GB | 一阶矩 |
| Adam variance $v$ | FP32 | 4 | 28 GB | 二阶矩 |
| **合计** | | **16** | **112 GB** | |

$$
\boxed{M_{\text{model state}} = 16\, \Psi \text{ 字节}}
$$

约定：$\Psi$ 表示参数数量（个），16 单位是字节/参数。本系列严格沿用这个约定（旧博客中有把 $\Psi$ 写成字节的，会引起混淆）。

### 1.1 为什么此配方使用 FP32 状态？

[第 1 篇 4.3 节](/posts/2026-06-12-llm-mem-opt-1-fundamentals/) 里讲过：Adam 每步累加的小增量（$1-\beta_2 \approx 0.001$ 量级）可能低于 BF16 的有效分辨率。为降低这类数值风险，该配方用 FP32 保存 master copy 与 $m/v$。这不是“$12\Psi$ 必须 FP32”的定律：8-bit Adam、不同状态精度或无 master copy 等实现会得到不同账本，并需要用收敛实验验证。

### 1.2 残差状态（不在 ZeRO 范围内）

激活值、gather/bucket 临时缓冲、通信工作区和显存碎片不属于 $16\Psi$。激活通常用 **gradient checkpointing**（前向丢弃中间值，反向重算）解决，与 ZeRO 互补，留到 [第 5 篇](/posts/2026-06-16-llm-mem-opt-5-sft/) 讨论。

### 1.3 冗余有多严重

256 张 A100 训 7B 模型：

$$
256 \times 112\text{ GB} = 28{,}672\text{ GB 模型状态}
$$

实际只需要 112 GB → **28,560 GB 是冗余，占 99.6%**。

ZeRO 要消灭的就是这 99.6%。

## 2. 三种切法：ZeRO 的三个阶段

ZeRO 按分片深度分三层，每层在上一层基础上再多切一类数据：

| Stage | 分片对象 | 单卡显存 | 通信开销 |
|-------|---------|---------|---------|
| ZeRO-1 ($P_S$) | 优化器状态（master + $m$ + $v$） | $4\Psi + 12\Psi/N$ | 无额外 |
| ZeRO-2 ($P_{os+g}$) | + 梯度 | $2\Psi + 14\Psi/N$ | 理想传输字节与 DDP 同量级 |
| ZeRO-3 ($P_{os+g+p}$) | + 参数 | $16\Psi/N$ | ~1.5× |

下面逐项推导**模型状态的理想下界**，并以标称 **7B 模型 + 8 GPU**（即 $\Psi=7\times10^9$，$N=8$，单位 GB）实例数字。实际峰值还应加上 activation、参数 gather、通信 bucket、内核工作区和分配器余量；这些项会随 batch、序列长度、实现和重叠策略变化。

### 2.1 ZeRO-1：切优化器状态

只切优化器状态。各 GPU 仍持有完整参数和梯度副本。

- 前向 / 反向和标准 DDP 一样（每张卡都跑全部 forward / backward）
- 反向后梯度不再做完整 All-Reduce，而是 **Reduce-Scatter**——每张卡得到 $1/N$ 的梯度分片
- optimizer step 时每张卡只更新自己负责的那 $1/N$ 参数，对应只存 $1/N$ 的 $m$ 和 $v$
- 更新完后做一次 **All-Gather** 把全模型参数拼回（让下一步前向能用完整参数）

显存计算：参数 + 梯度（不分片，$2 + 2 = 4$ 字节/参数）+ 优化器状态（分片，$12/N$ 字节/参数）：

$$
M_{\text{ZeRO-1}} = 4\Psi + \frac{12\Psi}{N}
$$

7B / 8 GPU：

$$
M_{\text{ZeRO-1}} = 4 \times 7 + \frac{12 \times 7}{8} = 28 + 10.5 = \mathbf{38.5\text{ GB/卡}}
$$

> 来自 112 GB/卡 → **38.5 GB/卡**，省 73.5 GB。

**关键事实：通信总量与标准 DDP 相同。**

直观看似乎多了「Reduce-Scatter + All-Gather」两步，但这两步的通信总量恰好等于一次 All-Reduce（All-Reduce 在底层就常用 Reduce-Scatter + All-Gather 实现）。我们只是 **改变了数据布局**——让 1/N 分片落在不同的卡上——而不是搬运了更多数据。

> 所以 ZeRO-1 是 **零代价** 的优化。除非有特殊原因，分布式训练里几乎都该开。

### 2.2 ZeRO-2：再切梯度

ZeRO-1 反向后已经 Reduce-Scatter 过梯度——每张卡天然就只有 1/N 的梯度分片。**ZeRO-2 干脆不再保留完整梯度，直接用分片版本去更新。** 这一步是"顺便"的，不需要任何额外通信。

显存：参数（不分片，$2\Psi$）+ 梯度（分片，$2\Psi/N$）+ 优化器状态（分片，$12\Psi/N$）：

$$
M_{\text{ZeRO-2}} = 2\Psi + \frac{14\Psi}{N}
$$

7B / 8 GPU：

$$
M_{\text{ZeRO-2}} = 2 \times 7 + \frac{14 \times 7}{8} = 14 + 12.25 = \mathbf{26.25\text{ GB/卡}}
$$

> 来自 38.5 GB → **26.25 GB**，再省 12.25 GB。理想传输字节仍与 DDP 同量级，实际时间需实测。

ZeRO-1 → ZeRO-2 几乎纯收益。生产场景如果显存还差一点，直接升 ZeRO-2 就好。

### 2.3 ZeRO-3：参数也切

最激进的一刀。每张 GPU 只持有 $1/N$ 的参数 + $1/N$ 的梯度 + $1/N$ 的优化器状态：

$$
M_{\text{ZeRO-3}} = \frac{16\Psi}{N}
$$

7B / 8 GPU：

$$
M_{\text{ZeRO-3}} = \frac{16 \times 7}{8} = \mathbf{14\text{ GB/卡}}
$$

显存随 GPU 数量 **线性下降**——这是大模型唯一能在固定单卡显存下训得起的路。

代价：**前向 / 反向的每一层** 都要从其他卡 All-Gather 参数分片，拼出完整参数后做计算，算完释放远端分片。每层多了一次 All-Gather，反向同样多一次。

在常见的 bulk-collective 理想字节模型中，ZeRO-3 的传输字节约为标准 DDP 的 **1.5×**。这是通信量比，不是吞吐或耗时比：bucket 大小、collective 算法、计算通信重叠和互联拓扑都会改变实际时间。

## 3. 通信模式对比

| Stage | 前向通信 | 反向通信 | 参数更新通信 | 理想传输字节 vs DDP |
|-------|---------|---------|--------------|----------------|
| DDP    | 无 | 1× All-Reduce | 无 | 1× |
| ZeRO-1 | 无 | 梯度同步（理想字节与 All-Reduce 等价） | 无单独计入 | 1× |
| ZeRO-2 | 无 | Reduce-Scatter 梯度与参数同步 | 无单独计入 | 1× |
| ZeRO-3 | 每层参数 All-Gather | 每层参数 All-Gather + 梯度 Reduce-Scatter | 无 | **~1.5×** |

要点：

- **ZeRO-1/2 的理想传输字节与 DDP 同量级**——实现中的 bucket 与同步时机会影响实测时间
- **ZeRO-3 是「用通信换显存」**——本质 trade-off
- 先用峰值显存、吞吐和收敛结果比较，再决定 ZeRO stage

## 4. 性能实测（H100）

参考 Josh Angel 在 8×H100 上 Continued Pretraining Gemma2-9B 的实测数据：

### 4.1 显存占用

下表来自 Josh Angel 的一次实验：8× H100 80 GB、NVLink 4.0/NVSwitch、Gemma2-9B continued pretraining、序列长 2,048、BF16、effective batch 144、DeepSpeed 与 Transformers 4.55.0+。它是这组条件下的峰值观测值，不可外推成不同模型、序列长度、batch、bucket 设置或硬件的“最大可训练模型规模”。

| Stage | 峰值显存/GPU | 相对该实验 ZeRO-0 |
|-------|------------|------------------|
| ZeRO-0（DDP 基线） | 76 GB | — |
| ZeRO-2 | 45 GB | -40.8% |
| ZeRO-3 | 28 GB | -63.2% |

### 4.2 吞吐量

| 配置 | tokens/sec/GPU | 相对该实验 ZeRO-0 |
|------|---------------|---------|
| ZeRO-0 | 2,847 | 100% |
| ZeRO-2 | 2,698 | 94.7% |
| ZeRO-3 | 2,234 | 78.5% |

这组实验中 ZeRO-3 相对基线低约 21.5%。它反映的是上述固定配置，不能推出固定吞吐损失或 ZeRO stage 的通用选型规则。batch 会同时改变计算量、激活峰值和通信重叠；应固定模型、序列长度、全局 batch 与数据集后，逐个 stage 实测。

## 5. 扩展：Offload、Infinity、ZeRO++

单节点显存还是不够时，ZeRO 还有三个重量级扩展。

### 5.1 ZeRO-Offload

把优化器状态和优化器计算卸载到 **CPU 内存**。GPU 跑前向 / 反向（计算密集），CPU 跑 optimizer step（带宽密集）。

- CPU 内存通常远大于 GPU 显存（512 GB-2 TB vs 80 GB）
- 代价是 CPU↔GPU 的数据传输，通常用 pinned memory + 异步拷贝降低
- 适合：单卡训 10B+ 模型

### 5.2 ZeRO-Infinity

在 Offload 基础上再加一层 **NVMe SSD**：GPU 显存 → CPU 内存 → NVMe，三级存储层次。

- 可训万亿参数级别模型
- 代价是 NVMe 的读写延迟（~100 µs vs HBM 的 ~1 µs），需要复杂预取策略隐藏延迟
- 实际部署需要 PCIe Gen4/Gen5 NVMe + 多块并行

### 5.3 ZeRO++

ZeRO-3 在低带宽互联（跨节点以太网/IB）下吞吐损失大。ZeRO++ 用 **通信压缩** 缓解：

- 跨节点的 All-Gather 通信用 INT8 量化压缩
- 减少低带宽链路上的数据量
- 实际收益取决于跨节点拓扑、通信模式、压缩设置与训练稳定性，应以目标集群压测为准

## 6. 选择流程

```text
1. 固定模型、精度、序列长度、micro/global batch 和目标硬件，记录 ZeRO-0/2/3 的峰值显存与吞吐。
2. 若 ZeRO-0/2 的实际峰值已满足容量和吞吐目标，优先选择实现更简单、实测更快的方案；若不满足，再评估 ZeRO-3。
3. ZeRO-3 若受通信限制，检查 bucket、重叠、互联拓扑和并行策略；不要仅因某个参数量阈值切换 stage。
4. 单节点仍不满足容量时，再评估 CPU/NVMe offload；同时量化其数据传输和训练时间成本。
```

## 7. DeepSpeed 配置示例

```json
{
  "train_batch_size": 128,
  "gradient_accumulation_steps": 4,
  "bf16": { "enabled": true },
  "zero_optimization": {
    "stage": 2,
    "offload_optimizer": { "device": "none" },
    "allgather_partitions": true,
    "allgather_bucket_size": 5e8,
    "overlap_comm": true,
    "reduce_scatter": true,
    "reduce_bucket_size": 5e8,
    "contiguous_gradients": true
  }
}
```

关键参数：

| 参数 | 作用 |
|------|------|
| `stage` | 0/1/2/3，选哪个阶段 |
| `offload_optimizer.device` | `"cpu"` 启用 ZeRO-Offload，`"nvme"` 启用 Infinity |
| `overlap_comm` | 请求通信与计算重叠；需按内存余量和吞吐测试 |
| `*_bucket_size` | 通信 buffer 大小，平衡 GPU 利用率与延迟 |

## 8. FSDP vs DeepSpeed ZeRO

PyTorch FSDP（Fully Sharded Data Parallel）和 ZeRO-3 在 **理念** 上完全一致——参数 + 梯度 + 优化器状态全分片。差异主要在生态：

| | DeepSpeed ZeRO-3 | PyTorch FSDP |
|---|------------------|--------------|
| 实现 | DeepSpeed 库 | PyTorch 原生（torch.distributed.fsdp） |
| Offload | CPU + NVMe（Infinity） | CPU only |
| 与并行策略组合 | 与 TP/PP 灵活混合 | 主要与 TP（FSDP2 后改善） |
| 生态成熟度 | 论文验证更多 | 快速追赶中，FSDP2 性能已接近 |
| 调试体验 | 配置复杂但文档丰富 | 原生 PyTorch，调试友好 |

实践建议：

- HuggingFace Trainer 用户 → **FSDP**（集成更好）
- Megatron-LM / 自定义训练循环 → **DeepSpeed**（并行策略组合更成熟）
- 新项目从 0 起步 → 直接用 **FSDP2**（PyTorch 2.x 原生支持，生态正在成为标准）

## 9. 小结

| 问题 | 答案 |
|------|------|
| ZeRO 解决了什么 | DDP 中 4 类模型状态（参数 / 梯度 / master / m / v）的多副本冗余 |
| ZeRO-1 | 切优化器状态，理想模型状态 **38.5 GB/卡** (7B/8GPU) |
| ZeRO-2 | + 切梯度，理想模型状态 **26.25 GB/卡** (7B/8GPU) |
| ZeRO-3 | + 切参数，理想模型状态 **14 GB/卡**；理想传输字节约 1.5× |
| 选哪个 | 固定目标配置后，对峰值显存、吞吐和收敛进行对比 |

下一篇我们从理论回到工程 → [LLM 训练与推理优化（五）— SFT 训练实战：Packing 到 Chunked NLL](/posts/2026-06-16-llm-mem-opt-5-sft/)，看 TRL SFTTrainer 在 ZeRO 之外还能用什么招把单卡显存挤干净。

---

## 参考资料

1. Rajbhandari et al. *ZeRO: Memory Optimizations Toward Training Trillion Parameter Models*. SC '20. [arXiv:1910.02054](https://arxiv.org/abs/1910.02054)
2. Ren et al. *ZeRO-Offload: Democratizing Billion-Scale Model Training*. USENIX ATC '21.
3. Rajbhandari et al. *ZeRO-Infinity: Breaking the GPU Memory Wall for Extreme Scale Deep Learning*. SC '21. [arXiv:2104.07857](https://arxiv.org/abs/2104.07857)
4. Wang et al. *ZeRO++: Extremely Efficient Collective Communication for Giant Model Training*. 2023. [arXiv:2306.10209](https://arxiv.org/abs/2306.10209)
5. Zhao et al. *PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel*. VLDB 2023. [arXiv:2304.11277](https://arxiv.org/abs/2304.11277)
6. Josh Angel. *ZeRO Optimization Strategies for Large-Scale Model Training*. Hugging Face Blog, 2025. [huggingface.co/blog/josh-a/zero-optimization-strategies](https://huggingface.co/blog/josh-a/zero-optimization-strategies)
7. Michael Brenndoerfer. *ZeRO Optimization: Stages 1, 2, and 3 Explained*. 2026. [mbrenndoerfer.com](https://mbrenndoerfer.com/writing/zero-optimization-stages-optimizer-gradient-parameter-partitioning)
