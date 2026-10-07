---
title: 训练显存与 ZeRO：模型状态分片和通信
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

> 这是多卡训练专题，不是单卡 SFT 的必修前置。建议先读 [显存基础](/posts/2026-06-12-llm-mem-opt-1-fundamentals/) 和 [Adam / AdamW](/posts/2024-04-20-weight-decay-and-l2-regularization/)，理解前向、loss、反向传播与 optimizer step。本篇解释：多张 GPU 怎样分担模型状态，以及为什么“分片后的大小”不等于训练峰值显存。

## 0. 先理解问题：多卡为什么还会装不下？

Data Parallel（DP）把一个 batch 分给多张卡处理，每张卡仍执行完整模型的前向和反向。标准 DistributedDataParallel（DDP）通常让每个进程保存完整参数、梯度和优化器状态。因此，增加 GPU 数量可以分摊样本的计算，却不会自动缩小每张卡上的模型状态。

一次训练更新可以拆成四步：

1. **Forward**：模型读入各自的样本，得到预测与 loss，并保存反向需要的激活。
2. **Backward**：由 loss 计算各参数的梯度。梯度描述 loss 对参数变化的局部敏感程度；优化器再据此决定调整方向和幅度。
3. **梯度同步**：合并不同卡的梯度，使各卡基于同一个全局目标更新。若各卡有效 token 数不同，不能不加权地平均各卡的 token-mean loss；应按有效 token 总数统一归一化。
4. **Optimizer step**：优化器利用同步后的梯度及历史状态更新参数，再开始下一步。

各卡同步前的梯度一般不同，因为它们处理的数据不同。可减少的是完整梯度存储、同步后的状态副本，以及每张卡重复保存的参数和优化器状态；不是说各卡刚算出的梯度都一样。

ZeRO（Zero Redundancy Optimizer）的基本思路是：**把模型状态分给不同的卡负责，需要计算时再同步或取回相应数据。** Stage 1 先分优化器状态，Stage 2 再分梯度，Stage 3 再分参数。它保留数据并行的计算方式，并不等于把网络的不同层固定交给不同的卡计算。[DeepSpeed 官方教程](https://www.deepspeed.ai/tutorials/zero/)给出了三个阶段的定义。

## 1. 先算清楚：$16\Psi$ 是哪一种训练配方？

### 1.1 本文符号与单位

- $\Psi$：参与全参数训练的参数数量，单位是“个”，不是字节。
- $N$：同一个 ZeRO 数据并行组中的进程数。以下假定每个进程用一张 GPU；若同时用了 Tensor Parallel 或 Pipeline Parallel，$N$ 不能直接取集群总卡数。
- GB 使用十进制：$1\text{ GB}=10^9$ 字节；GiB 使用二进制：$1\text{ GiB}=2^{30}$ 字节。
- 数值例子统一取 $\Psi=7\times10^9$、$N=8$。这是恰好 70 亿参数的算术例子，不代表所有名称带“7B”的模型参数量完全相同。

### 1.2 为什么训练还要存 Adam 状态？

本篇沿用[显存基础的 Adam 混合精度配方](/posts/2026-06-12-llm-mem-opt-1-fundamentals/#51-一套明确的-adam-混合精度配置)：**所有参数都可训练，没有 CPU/NVMe offload**。为对应后面的分片对象，把状态归成三类：

| 状态类别 | 字节/参数 | 7B 占用 | 组成 |
|----------|----------|---------|------|
| 计算用参数 | 2 | 14 GB | BF16 或 FP16 参数副本 |
| 保留梯度 | 2 | 14 GB | BF16 或 FP16 梯度 |
| 优化器侧状态 | 12 | 84 GB | FP32 master weights、Adam $m$、Adam $v$，各 4 字节 |
| **合计** | **16** | **112 GB** | 仅模型状态 |

$$
M_{\text{state}}=(2+2+12)\Psi=16\Psi\text{ 字节}.
$$

$m/v$ 的含义与更新式见[显存基础 §5.1—5.3](/posts/2026-06-12-llm-mem-opt-1-fundamentals/#51-一套明确的-adam-混合精度配置)。这里的 $12\Psi$ 是为 ZeRO 账本约定的“优化器侧”口径，不表示所有实现都必须有独立 master weights。

### 1.3 $18\Psi$ 与 autocast：不要仅凭 BF16 标签猜显存

本配方若只把保留梯度改为 FP32，其他副本不变，就变为 $(2+4+12)\Psi=18\Psi$ 字节，7B 对应 126 GB；同时存活的转换缓冲仍要另计。

`torch.autocast` 控制算子计算精度，不保证常驻两套完整权重。具体区别见[显存基础 §5.2](/posts/2026-06-12-llm-mem-opt-1-fundamentals/#52-autocast-不等于常驻两套权重)。估算前仍须确认参数、梯度和优化器状态的 dtype、是否有独立 master weights，以及副本的生命周期。

## 2. ZeRO 三阶段：切哪些状态，没切哪些？

先约定 $b_p,b_g,b_o$ 分别表示每个参数的计算用副本、保留梯度和优化器侧状态所占的字节数。假设分片均衡、没有 offload、忽略对齐与临时缓冲，则有：

| 配置 | 计算用参数 | 保留梯度 | 优化器侧状态 | 单卡模型状态账本 |
|------|-----------|----------|---------------|------------------|
| DDP / Stage 0 | 完整 | 完整 | 完整 | $(b_p+b_g+b_o)\Psi$ |
| ZeRO-1 | 完整 | 完整 | $1/N$ | $(b_p+b_g)\Psi+b_o\Psi/N$ |
| ZeRO-2 | 完整 | $1/N$ | $1/N$ | $b_p\Psi+(b_g+b_o)\Psi/N$ |
| ZeRO-3 | $1/N$ | $1/N$ | $1/N$ | $(b_p+b_g+b_o)\Psi/N$ |

这里的“保留”描述训练状态的分片布局，不表示梯度在训练的每一刻都已分配，也不表示反向中从未出现较大的未分片张量。这些是**理想模型状态账本，不是程序总峰值**。

<!-- concept-figure:zero-state-shards -->

![四卡 ZeRO 状态布局，只画 GPU0：DDP 每类保存全部四段；Stage1 分优化器状态，Stage2 再分保留梯度，Stage3 再分计算参数，各自只保留本卡编号段](/images/llm-series/concept-zero-state-shards.svg)

*图：为看清布局，单独取 $N=4$，编号 0—3 表示参数片段，图中只展开 GPU 0；下方容量表仍使用正文约定的 $N=8$。每类方框同宽不表示所占字节相同，“优化器侧”沿用本文 master weights 加 Adam $m/v$ 的口径。分片状态图也不含激活、通信缓冲与临时 gather。原创示意；阶段定义依据 [DeepSpeed ZeRO 官方教程](https://www.deepspeed.ai/tutorials/zero/#zero-overview)。*

<!-- /concept-figure:zero-state-shards -->

代入 $b_p=2,b_g=2,b_o=12$，再代入 7B / 8 GPU：

| 配置 | $16\Psi$ 配方下的公式 | 每卡状态大小 |
|------|-------------------------|---------------|
| DDP | $16\Psi$ | 112 GB |
| ZeRO-1 | $4\Psi+12\Psi/N$ | 38.5 GB |
| ZeRO-2 | $2\Psi+14\Psi/N$ | 26.25 GB |
| ZeRO-3 | $16\Psi/N$ | 14 GB |

如果使用上一节的 FP32 梯度配方，$b_g$ 应改为 4：对应 DDP 为 126 GB，ZeRO-1 为 $6\Psi+12\Psi/N=52.5$ GB，ZeRO-2 为 $2\Psi+16\Psi/N=28$ GB，ZeRO-3 为 $18\Psi/N=15.75$ GB。**更换 dtype 后要重算整张表。**

### 2.1 ZeRO-1：只分优化器侧状态

可以把参数按编号分给 8 张卡：每张卡只负责其中一段参数的 master weights、$m$ 和 $v$，也只执行这一段的 optimizer step。但为了前向与反向，各卡仍保存完整计算参数；这一阶段也没有要求把梯度的存储长期缩小到 $1/N$。

一次更新可以按这样的逻辑安排：

1. 各卡完成自己的前向与反向，产生梯度。
2. 将每一段梯度归约到负责该段更新的卡。Reduce-Scatter 是实现这种分发的一种方式。
3. 各卡更新自己负责的 master weights，并生成更新后的低精度参数分片。
4. 通过 All-Gather 或等效广播，把更新后的参数同步回所有卡，供下一步使用。

**一次 Reduce-Scatter 的输出是分片，不代表完整梯度存储已经消失。** Stage 1 和 Stage 2 的区别在于梯度生命周期与存储管理，不能只看通信原语的名字。实际框架也可能用 Reduce、All-Reduce 或不同 bucket 调度实现同步，以上是理解原理的流程，不是所有版本的逐行执行顺序。

### 2.2 ZeRO-2：梯度算出来后，也只留下自己负责的一段

Stage 2 进一步在反向过程中归约梯度并释放不再需要的非本地分片，长期保留的更新梯度与本地优化器状态一一对应。因此梯度项从 $b_g\Psi$ 变成 $b_g\Psi/N$；完整低精度参数仍需在各卡上保存，更新后仍需同步参数。

这不是不再计算其他参数的梯度。每张卡仍用自己的样本跑完整模型的 backward，变化的是梯度产生后的归约和释放时机。归约前的梯度 bucket、打包缓冲和 dtype 转换仍可能占显存。

### 2.3 ZeRO-3：参数平时也只留一段，用时取回

Stage 3 连计算用参数也分片：执行某个模块之前，先 All-Gather 该模块需要的参数；用完后，按策略释放非本地部分。反向若需要已释放的参数，就要再次取回。算出的梯度再通过 Reduce-Scatter 等方式交给各自负责更新的卡。

注意这里不是“每张卡永远只装 $1/N$ 参数”。计算时通常要临时拼出当前模块的完整参数，预取还可能让下一个模块的参数提前驻留。模块划分、参数持久化阈值和重用策略会改变 gather 次数与峰值。[DeepSpeed ZeRO-3 文档](https://deepspeed.readthedocs.io/en/stable/zero3.html#assumptions)也明确要求单个子模块的前向与反向能放进设备显存。

所以 $16\Psi/N$ 只能表示本例的分片状态部分。它不能证明 7B 在 8 张 14 GB 卡上就能训练，也不是固定单卡容量下训练大模型的唯一办法；Tensor Parallel、Pipeline Parallel 和 offload 也可能参与解决容量问题。

## 3. 通信：字节相同，也不代表“免费”

先区分三个常用操作：

- **All-Reduce**：把各卡的同位置数值求和（平均还需相应缩放），每张卡拿到完整结果。
- **Reduce-Scatter**：同样归约，但每张卡只拿到结果的一段。
- **All-Gather**：把各卡持有的不同分片拼起来，每张卡拿到完整结果；它本身不做求和。

用一个简化模型计数：$N>1$，所有参数和梯度都以相同的 2 字节精度通信，每次 optimizer update 只对应一轮所讨论的梯度同步；gather 出的完整参数用后释放，前向和反向各取回一遍。设完整参数或梯度的通信载荷为 $S=2\Psi$ 字节，用 ring collective 的**单卡发送字节数**作口径，则一次 All-Gather 或 Reduce-Scatter 各约为：

$$
C=\frac{N-1}{N}S.
$$

接收量在这个模型中相同；若统计“发送加接收”，各项整体再乘 2，比例不变。

| 配置 | 一次更新的逻辑通信安排 | 理想发送量 |
|------|------------------------|------------|
| DDP | backward 中/结束时对梯度 All-Reduce | $2C$ |
| ZeRO-1/2 | 梯度归约到负责人；更新后 All-Gather 参数 | $C+C=2C$ |
| ZeRO-3 | forward 取参数；backward 再取参数并归约梯度 | $C+C+C=3C$ |

这说明，在上述假设下，ZeRO-1/2 可以达到与 DDP 相同的通信字节量，ZeRO-3 则是 $3/2=1.5$ 倍。这是对[ZeRO 论文第 7 节](https://arxiv.org/html/1910.02054v3#S7)通信分析的简化，不代表所有实现和训练设置都有这个固定成本。

以本篇 7B / 8 GPU 为例，$S=14$ GB，$C=12.25$ GB；三种通信布局分别得到 24.5、24.5、36.75 GB 的单卡发送量。这仍是理论算术，不是测得的带宽或耗时。

实际需要重新核对：

1. **通信 dtype 与存储 dtype 可以不同。** 如果梯度按 FP32 传输、参数按 BF16 传输，两种载荷大小就不相等，不能直接套 1.5 倍。
2. **同步次数不同。** 梯度累积可能改变归约时机；不能假定 DDP 与 ZeRO 在每个 microbatch 上发出相同次数的 collective。
3. **通信能否被计算覆盖。** Stage 1/2 的参数同步需要等待对应更新完成；Stage 3 的参数必须在模块计算前到位。时机变了，等待时间也可能变。
4. **bucket 大小与硬件拓扑。** 小消息多会增加延迟，大 bucket 会增加临时显存；跨节点链路、预取和通信计算重叠都会影响吞吐。

因此，“字节量相同”不等于零开销，“Stage 3 是 1.5 倍字节量”也不等于训练慢 50%。应测量完整训练步的时间。

## 4. 真正会 OOM 的，是某一时刻的总占用

训练峰值可以按生命周期理解为：

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

### 4.1 怎样做可复现的对比？

本文不把缺少完整脚本、dtype 账本和测量口径的显存数字当作基准。上面的 7B 数字仅为可复算示例，不是 H100 或任何 GPU 的实测结果。

比较 stage 时，至少固定并记录：

- 模型版本与实际参数量，全参数训练还是 LoRA，参数/梯度/优化器与通信 dtype
- GPU 数量、互联拓扑，PyTorch、DeepSpeed、CUDA 与通信库版本
- 序列长度、每卡 microbatch、梯度累积次数、global batch、有效 token 数
- activation checkpointing、bucket、预取、通信重叠与 offload 设置
- 预热后每个 rank 的峰值，再取各 rank 最大值；区分 allocated、reserved 和设备总占用
- 同步计时后的完整 optimizer update 耗时、有效 tokens/s，以及相同设置下的 loss/收敛表现

还应单独检查初始化和 checkpoint 保存能否通过。没有实际运行，就不能只凭状态表宣布某个 batch 一定能装下。

## 5. Offload、Infinity 与 ZeRO++：容量和带宽一起看

### 5.1 ZeRO-Offload

[ZeRO-Offload](https://www.deepspeed.ai/tutorials/zero-offload/)把优化器相关状态与计算移到 CPU，GPU 继续承担主要前向与反向计算。这能减少 GPU 驻留状态，但会消耗 CPU RAM、CPU 计算与 CPU–GPU 传输带宽。

直观估算：若某条链路每步至少要搬运 $D$ 字节，有效带宽为 $B$ 字节/秒，单是传输就需要约 $D/B$ 秒的链路服务时间。它可以和计算重叠一部分，但不会因 pinned memory 或异步拷贝而消失。必须同时检查主机内存容量、CPU optimizer 耗时和传输瓶颈；不能只凭“CPU 内存大”就承诺可用吞吐。

### 5.2 ZeRO-Infinity

ZeRO-Infinity 在 ZeRO-3 的基础上使用 CPU/NVMe 分层存储，进一步扩展可容纳的状态规模。按需读取、预取和缓冲调度决定能否让 GPU 及时拿到数据；NVMe 容量大不代表它具有 GPU 显存带宽。其设计和适用条件见 [ZeRO-Infinity 论文](https://arxiv.org/abs/2104.07857)。

评估时要测存储的持续吞吐、CPU–GPU 链路及它们是否争用，而不是只比较标称延迟。GPU 上仍需给活动模块、激活和通信缓冲留空间；“能把状态存下”与“训练时间可以接受”是两个问题。

### 5.3 ZeRO++

[ZeRO++](https://www.deepspeed.ai/tutorials/zeropp/)组合了量化参数通信、层级分片和量化梯度通信等机制，目标是减少低带宽链路上的传输。它不只是把所有 All-Gather 统一改成 INT8；不同机制也可能改变冗余存储与数值误差。是否适合目标集群，要连同通信拓扑、显存和收敛一起验证。

## 6. DeepSpeed 配置：先能解释，再去调参

下面是 **8 个数据并行进程**的 Stage 2 配置示例。它展示参数含义，不是已在本文环境跑通的完整训练脚本，也不保证内部恰好采用上面的 $16\Psi$ 配方：

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

这里 global batch 为 $2\times4\times8=64$ 条样本；序列长度不同或使用 packing 时，样本数相同不代表有效 token 数相同。完整运行仍需模型、数据、优化器、学习率及兼容的训练入口。

| 参数 | 要核对什么 |
|------|------------|
| `stage` | 分片到优化器状态、梯度还是参数 |
| `allgather_partitions` | 更新后用 All-Gather 还是等效广播同步参数 |
| `overlap_comm` | 尝试重叠通信与计算；更高的同时驻留量可能增加峰值 |
| `*_bucket_size` | 单位是元素个数，不是字节；本例 5000 万个 BF16 元素约 100 MB，但总缓冲不一定只有一份 |
| `contiguous_gradients` | 用连续缓冲组织梯度；仍要计算该缓冲的实际占用 |

参数语义以[DeepSpeed 配置文档](https://www.deepspeed.ai/docs/config-json/)与所用版本为准。修改 stage 后应重新验证；Stage 3 还有预取和参数保留相关设置，不能只照搬 Stage 2 的峰值估计。

## 7. 怎么选 stage？FSDP 又是什么关系？

先确定主要瓶颈：模型状态放不下，还是长序列激活太大，或者通信已经占据大部分时间？

1. 固定目标模型、训练精度、global batch 和数据口径，建立能运行的基线；如果基线 OOM，如实记录。
2. 用本篇公式比较模型状态，再实际测试 Stage 1/2/3 的峰值和吞吐。选满足容量与速度目标的配置，不按固定参数量阈值做决定。
3. Stage 3 仍 OOM 时，检查活动模块、预取、激活和初始化，而不是只增加分片数。
4. 需要 offload 时，同时评估主机容量和端到端训练时间。$N=1$ 时没有其他卡分摊，单靠分片本身不会减少状态总量。

PyTorch FSDP（Fully Sharded Data Parallel）的全分片用法与 ZeRO-3 解决相似问题：分片参数、梯度和优化器状态，用计算时的参数收集换取较小的常驻量。不同 FSDP API、分片策略和参数重分片设置有不同生命周期；不能把“用了 FSDP”直接替换成同一个峰值公式。[PyTorch FSDP2 教程](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html)展示了按模块分片的使用方式。

DeepSpeed 和 FSDP 的选择应依据现有训练栈、模型兼容性、并行/offload 需求及可复现测试。本篇不根据框架名称断言哪一个普遍更快或集成更好。

<span id="8-小结与下一篇前置知识"></span>

## 8. 小结与回到 SFT

理解 ZeRO 的关键是三个问题：**哪类状态被切开？什么时候必须拿回？拿回时又多占了多少空间、花了多少时间？**

- $16\Psi$ 来自明确的参数、梯度、master weights 和 Adam $m/v$ 配方；FP32 梯度的同类配方是 $18\Psi$。
- Stage 1/2/3 依次分片优化器侧状态、梯度、参数。分片状态表与运行峰值必须分开。
- 通信字节、通信耗时和训练吞吐是不同指标，激活也不会随 ZeRO stage 自动分片。

理解分片后，可返回 [SFT：数据边界与显存优化](/posts/2026-06-16-llm-mem-opt-5-sft/)，把训练状态与数据、激活优化合起来看。SFT 第 1 节解释 loss 范围，第 4 节解释 LoRA 的可训练参数；不要把本篇全参数 Adam 账本原样套到 LoRA。

## 参考资料

1. Rajbhandari et al. [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054)，重点为模型状态与通信分析。
2. DeepSpeed [ZeRO 官方教程](https://www.deepspeed.ai/tutorials/zero/)、[ZeRO-3 配置与限制](https://deepspeed.readthedocs.io/en/stable/zero3.html)、[显存估算](https://deepspeed.readthedocs.io/en/stable/memory.html)。
3. PyTorch [Adam 算法](https://docs.pytorch.org/docs/2.8/generated/torch.optim.Adam.html)与 [AMP/autocast](https://docs.pytorch.org/docs/2.8/amp.html)。
4. DeepSpeed [ZeRO-Offload](https://www.deepspeed.ai/tutorials/zero-offload/)；Rajbhandari et al. [ZeRO-Infinity](https://arxiv.org/abs/2104.07857)；DeepSpeed [ZeRO++](https://www.deepspeed.ai/tutorials/zeropp/)。
5. DeepSpeed [JSON 配置](https://www.deepspeed.ai/docs/config-json/)与 PyTorch [FSDP2 教程](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html)。
