---
title: SFT 训练实战：数据边界与显存优化
published: 2026-06-16
description: 用 TRL SFTTrainer 讨论 Truncation、Packing、PEFT、Liger Kernel、Chunked NLL、Padding-Free、Activation Offloading 与 Gradient Checkpointing 的显存取舍。
category: Deep Learning
tags:
- llm
- memory
- llm-memory-optimization-series
- sft
- trl
- fine-tuning
draft: false
---

> 本文从单卡 SFT 出发。先读 [显存基础](/posts/2026-06-12-llm-mem-opt-1-fundamentals/)，分清状态与激活；需要多卡全参训练时再读 [ZeRO](/posts/2026-06-15-llm-mem-opt-4-zero/)。SFT 用示范答案做监督微调，不要求先学强化学习。

> **版本与验证范围**：本文保留原始发布日期，API 于 **2026-10-06** 按官方已发布的 [TRL v1.14.1](https://github.com/huggingface/trl/releases/tag/v1.14.1) 及其固定 tag 源码核对。以下默认值、参数和兼容性仅指这一版。示例做过 Python 语法、接口静态检查与独立 CPU 数值验证，未在 GPU 上实际训练；不能据此承诺某张显卡一定跑得动。Transformers、PEFT、bitsandbytes、PyTorch、CUDA 与 attention kernel 也需要记录版本，固定 TRL 并不等于锁定完整环境。

## 0. 消费级单卡 SFT 7B 的工程难题

如果采用 [显存基础](/posts/2026-06-12-llm-mem-opt-1-fundamentals/) 中的一种混合精度 Adam 账本：低精度权重 2 byte、低精度梯度 2 byte、FP32 主权重 4 byte、两个 FP32 动量合计 8 byte，那么模型状态约为 **16 byte/参数**。7B 参数对应 $7\times10^9\times16=112\times10^9$ byte，即 **112 GB，约 104.3 GiB**，还没有加激活、临时张量与通信缓冲。这是带条件的估计，不是所有训练实现的固定开销。

单卡没有其他 GPU 来分摊状态，纯分片的 ZeRO 不能凭空腾出容量；CPU/NVMe offload 是另一种取舍。单卡微调通常先考虑：

- **LoRA / QLoRA**：减少需要训练的参数，以及冻结基模型的权重占用
- **Truncation / Packing / Padding-Free**：控制序列长度，减少无效 padding
- **Chunked NLL / Liger Kernel**：减少损失计算或其他算子的中间张量
- **Gradient Checkpointing / Activation Offloading**：少保存激活，或者把它们暂存到 CPU

这些方法针对不同的显存开销。应先确认瓶颈，再组合使用，不能把各自的“省显存百分比”相加。

## 1. SFT Trainer 快速入门

### 1.1 一个小模型的起点

下面的短训练示例需要兼容的 GPU 和依赖。这里显式选择普通 `nll`，先检查数据和 loss，后面再尝试其他优化方式。

```python
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer

trainer = SFTTrainer(
    model="Qwen/Qwen3-0.6B",
    args=SFTConfig(
        output_dir="sft-smoke",
        max_length=512,
        per_device_train_batch_size=1,
        max_steps=2,
        loss_type="nll",
        report_to="none",
    ),
    train_dataset=load_dataset("trl-lib/Capybara", split="train[:8]"),
)
trainer.train()
```

`SFTTrainer` 可以处理 tokenizer、对话模板、截断和组 batch，但它不会替你判断“哪些文本应当成为训练答案”。本版配置默认启用 BF16（未指定 FP16 时）和 gradient checkpointing；硬件或任务不适合时要显式调整。[配置源码](https://github.com/huggingface/trl/blob/v1.14.1/trl/trainer/base_config.py)

### 1.2 数据格式不等于损失范围

三种常见格式：

```python
# 普通语言建模
{"text": "The sky is blue."}

# 对话形式的语言建模
{"messages": [
    {"role": "user", "content": "What color is the sky?"},
    {"role": "assistant", "content": "It is blue."},
]}

# Prompt-Completion
{"prompt": "The sky is", "completion": " blue."}
```

在 v1.14.1 中：

- `text` 或 `messages` 的语言建模数据，默认在全部有效目标 token 上计算 loss；**出现 assistant 角色不代表默认只训练回复**
- `prompt` + `completion` 数据，默认只对 completion 计算 loss；也可显式设置 `completion_only_loss=True`
- `assistant_only_loss=True` 用于对话数据，只训练助手回复；需要 chat template 能提供助手 token mask，包括正确的回复结束标记

同时使用 completion-only 和 assistant-only 时，参与 loss 的位置必须满足两者。使用模板前，应打印一条实际 tokenized 样本，确认问题、回答、结束符的边界，而不是靠字符长度推算 token 边界。[SFT 数据与损失选项](https://github.com/huggingface/trl/blob/v1.14.1/docs/source/sft_trainer.md)

### 1.3 shift、padding mask 和 response mask 各管什么

令输入 token 为 $x_{b,0},\ldots,x_{b,S-1}$，batch 大小为 $B$、长度为 $S$。位置 $t$ 的输出预测 $x_{b,t+1}$。设 $m_{b,t+1}\in\{0,1\}$ 表示这个**目标位置**是否参与监督，则 token 平均损失为：

$$
\mathcal L_{\mathrm{SFT}}
=-\frac{1}{N_{\mathrm{valid}}}
\sum_{b=1}^{B}\sum_{t=0}^{S-2}
m_{b,t+1}\log p_\theta(x_{b,t+1}\mid x_{b,\le t}),
\qquad
N_{\mathrm{valid}}=\sum_{b=1}^{B}\sum_{t=0}^{S-2}m_{b,t+1}.
$$

这一定义要求 $N_{\mathrm{valid}}>0$。分母不是 $B\times S$，也不是样本数。梯度累积或多卡聚合时，如果希望维持同一个 token 平均目标，应合计有效 token 的损失和与数量；不能不加权地平均有效长度不同的 microbatch loss。

具体看一条右侧 padding 的样本：

```text
input_ids:       [BOS, 问题, 回答1, 回答2, EOS, PAD]
labels:          [-100, -100, 回答1, 回答2, EOS, -100]
attention_mask:  [1,    1,    1,     1,     1,   0]
```

默认因果 LM loss 把 `logits[:, :-1, :]` 与 `labels[:, 1:]` 对齐。因此这里有 3 个有效目标：回答1、回答2、EOS。不要再手动把 labels 移动一次，造成双重 shift。

- **Causal mask**：禁止当前位置偷看未来 token
- **Padding attention mask**：避免真实 token 关注补齐的位置
- **Loss mask**：把无需监督的位置的 label 设为 `-100`，交叉熵忽略它们；prompt 仍留在输入中作为回答的上下文

`-100` 不是 padding token ID，也不会自动把该 token 从 attention 中删除。如果 PAD 和 EOS 共用 ID，应按实际 padding 位置屏蔽，不能把所有等于该 ID 的真实 EOS 都屏蔽。把 prompt 排除在 loss 之外，也不表示 Transformer 不再计算 prompt。

![SFT Figure](https://huggingface.co/datasets/trl-lib/documentation-images/resolve/main/sft_figure.png)
*图：SFT 训练中，预测值与下一 token 的标签对齐；padding 位置不参与损失计算。*

## 2. Truncation：控制单条样本的长度

### 2.1 为什么需要

普通动态 padding 会把 **同一个 batch** 内的序列补到该 batch 最长长度。一条异常长样本会拖累与它同 batch 的短样本，不是把整个数据集都补到全局最长。按长度分组、减小 batch 或截断都可能减轻浪费。

![Truncation](https://huggingface.co/datasets/trl-lib/documentation-images/resolve/main/why_you_should_truncate.png)
*图：序列长度分布不均时，不加截断会造成大量 padding。*

### 2.2 配置与取舍

```python
from trl import SFTConfig

training_args = SFTConfig(max_length=2048)
```

本版默认保留开头、截去尾部。长度应按应用 chat template 后的 token 统计。P90/P95 可以作为观察长度分布的起点，但不是选阈值的规则：长答案、代码结尾和 EOS 可能恰好在被删掉的部分。

截断之后还要检查 **剩余的有效回答 token 数**。如果 prompt 已经占满窗口，这条样本可能只剩 `-100` 标签，无法提供训练信号。应缩短 prompt、提高长度上限、合理切分或过滤该样本。Packing 可以利用剩余空间，但不能恢复被截掉的内容。

## 3. Packing：把短样本放进同一个窗口

### 3.1 先看目的，再看边界

Packing 把多条短序列安排到接近 `max_length` 的块中。例如长度为 900、700、400 的三条序列，放到 2048 的窗口里，总计 2000 token；相比把三条都 pad 到 900，数据排布更紧凑。这个算例只比较 token 槽位，实际计算量还取决于 attention 实现。

![Packing](https://huggingface.co/datasets/trl-lib/documentation-images/resolve/main/packing_3.png)
*图：Packing 把多条序列拼到一个训练样本中，最大化利用上下文窗口。*

**把序列接起来，不等于把样本隔离开。** 普通 causal attention 会让后一个样本看到前一个样本；插入 EOS 本身并不阻止这种可见性。独立样本的 packing 需要注意力后端识别每段边界，也要屏蔽“上一段末尾预测下一段开头”的 loss。

### 3.2 本版策略

```python
training_args = SFTConfig(
    packing=True,
    packing_strategy="bfd",
    max_length=2048,
    model_init_kwargs={"attn_implementation": "flash_attention_2"},
)
```

这是依赖兼容 FlashAttention 安装、模型和硬件的配置片段，不是仅加一个开关就能运行的完整脚本。

| 策略 | 长样本怎么处理 | 要注意什么 |
|------|----------------|------------|
| `bfd` | 超过窗口的序列截断 | 保留各段长度，用于恢复边界 |
| `bfd_split` | 超长序列拆成片段再装箱 | 保留 token 不等于保留跨片段完整上下文 |
| `wrapped` | 拼成 token 流后按窗口切块 | 不保留原始样本隔离语义，可能中途切断样本 |

实际 [trainer 源码](https://github.com/huggingface/trl/blob/v1.14.1/trl/trainer/sft_trainer.py) 对 **`bfd` 和 `bfd_split` 都自动启用 Padding-Free**，而不是只对 `bfd`。Collator 利用 `seq_lengths` 重置每段的 position IDs，并把段首 label 设为 `-100`。兼容的变长 FlashAttention 路径再利用边界分开计算 attention；只重置 position IDs、却使用忽略这些边界的后端，仍可能串样本。

不要把 `wrapped` 当成“自动隔离的 SFT”。若任务确实要把文档当连续流训练，应明确接受这种目标变化；对独立问答优先验证隔离。v1.14.1 也明确拒绝 packing 与 context parallelism（`cp_size > 1`）的组合，因为相应路径不能保留所需的分块 attention 边界。[打包实现](https://github.com/huggingface/trl/blob/v1.14.1/trl/data_utils.py)

## 4. PEFT（LoRA）：只训练少量新增参数

PEFT 是参数高效微调这一类方法，LoRA 是其中一种。对输入列向量 $x\in\mathbb R^{d_{\mathrm{in}}}$，设原线性层权重为 $W\in\mathbb R^{d_{\mathrm{out}}\times d_{\mathrm{in}}}$。标准 LoRA 冻结 $W$，学习一个低秩增量：

$$
W'=W+\frac{\alpha}{r}BA,\qquad
A\in\mathbb R^{r\times d_{\mathrm{in}}},\quad
B\in\mathbb R^{d_{\mathrm{out}}\times r}.
$$

这里 $r$ 是 rank，$\alpha$ 对应 `lora_alpha`；`r=16, lora_alpha=32` 时缩放为 2。默认标准 LoRA 的系数是 **$\alpha/r$**；rsLoRA 等变体使用不同缩放，不能混写。训练时可计算 $Wx+(\alpha/r)B(Ax)$，无须为每一步显式构造完整的 $W'$。[LoRA 论文](https://arxiv.org/abs/2106.09685)

单个矩阵的可训练参数量从 $d_{\mathrm{out}}d_{\mathrm{in}}$ 变成 $r(d_{\mathrm{in}}+d_{\mathrm{out}})$。只有方阵时才能简写为 $2dr$。例如 $d=4096,r=16$，新增参数是 131,072，占原矩阵 16,777,216 参数的 **0.78125%**。这不是整模型显存占比：冻结权重仍然要存，激活仍要为梯度计算服务。

```python
from peft import LoraConfig

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],  # 这里采用 Qwen 的层名
    task_type="CAUSAL_LM",
)
```

梯度和优化器状态主要跟随可训练参数，但 `modules_to_save`、可训练 bias 或其他未冻结层会另外占用。LoRA 的学习率可以从 `1e-4` 这样的量级试起，再看验证集表现；可训练参数少并不能推出“学习率必须更高”。

### 4.1 QLoRA：存储精度与计算精度要分开

QLoRA 在冻结的低比特基模型上训练 LoRA。**4-bit 描述的是基模型的量化存储，不是所有运算都用 4-bit。** 相关权重在需要时按计算路径反量化；前向和反向都可能需要这些权重。`bnb_4bit_compute_dtype=torch.bfloat16` 指定 4-bit 线性层的计算 dtype，不等于整个模型的参数、激活和优化器状态都变成 BF16。[bitsandbytes 文档](https://huggingface.co/docs/transformers/en/quantization/bitsandbytes)

- 基模型：NF4 编码本体为 4 bit/参数，另有量化常数、分组元数据、未量化层和工作区
- LoRA：仍是浮点可训练参数；实际 dtype 由加载、PEFT 和 trainer 处理决定，检查参数而不是只看一个配置项
- 优化器：为可训练参数保存状态，状态 dtype 由优化器实现决定
- 激活：不会因为权重是 NF4 就自动减少为四分之一

QLoRA 论文中的三个技巧：NF4 用适合近似正态权重分布的非均匀编码；Double Quantization 再量化量化常数，论文估计平均额外节省约 0.37 bit/参数；Paged Optimizers 利用 unified memory 迁移分页状态以应对显存压力。这些都不能保证任意配置不会 OOM，分页也不是“发生 OOM 后自动救活整个训练”。论文报告的是 **单张 48 GB GPU 微调 65B 模型**，不是任意 24 GB 卡上的通用结论。[^qlora]

量化原理见 [量化：基础与进阶推导](/posts/2026-05-16-llm-quantization-guide/)，完整的起步配置放在第 12 节。

[^qlora]: Dettmers et al. *QLoRA: Efficient Finetuning of Quantized LLMs*. NeurIPS 2023. [arXiv:2305.14314](https://arxiv.org/abs/2305.14314)

## 5. Liger Kernel：减少算子中间张量

[Liger Kernel](https://github.com/linkedin/Liger-Kernel) 提供面向 LLM 训练的 Triton kernels，例如融合线性投影与交叉熵、RMSNorm、RoPE、SwiGLU 等。核心思路是让数据尽量留在更近的存储层，少写回 GPU 显存，少保留不必要的中间张量。

```python
training_args = SFTConfig(use_liger_kernel=True, loss_type="nll")
```

模型支持、实际替换的算子和库版本共同决定收益。官方材料里的吞吐或显存百分比来自特定 benchmark，不能保证每个配置都有相同收益。本版 `use_liger_kernel=True` 与 `loss_type="chunked_nll"` 不能同时启用；没有显式设置 loss 时，Liger 路径默认选择 `nll`。

## 6. Chunked NLL：不要一次生成所有位置的 logits

### 6.1 大词表为什么占显存

Logits 是模型为每个候选 token 输出的分数，普通 LM head 的输出形状为 `[B, S, V]`，$V$ 是词表大小。以一个近似的大词表规模为例：

$$
B=4,\ S=2048,\ V=152000,\qquad
M_{\mathrm{logits}}=BSV\times2
=2{,}490{,}368{,}000\ \mathrm{byte}.
$$

若 logits 以 BF16 保存，这约为 **2.49 GB / 2.32 GiB**。这只是一个张量；交叉熵可能将部分计算提升到 FP32，反向也需要工作区，因此既不能把它当训练总显存，也不能固定乘二作为真实峰值。不同模型的实际 $V$ 应读配置，而不是用品牌名猜。

### 6.2 分块做同一件事

NLL 是负对数似然；对于这里的 token 分类，它与标准交叉熵是同一个目标。每个目标 token 需要正确词的 logit 和整个词表的归一化项，但不同 token 的损失可以分批算：

1. 先正确 shift hidden states 与 labels，再识别 `labels != -100` 的目标位置
2. 只为有效目标所在的分块做 LM head 投影，避免大量无监督位置生成完整词表 logits
3. 累加各块 loss 的和，再除以总有效 token 数，而不是平均每块的均值
4. 用 checkpoint 在反向重算块内结果，避免 autograd 把每块大张量全部保留到最后

若每块有 $C$ 个 token，**logits 这一项**的峰值量级由 $BSV$ 降为 $CV$，不是整个训练显存都变为 $CV$。在本版实现中，块大小是 256；有效位置会移到前部，最后一块可能夹带被忽略的尾部位置，因此“每一个 `-100` 都绝不会参与任何 matmul”也不够精确。[实现与归一化](https://github.com/huggingface/trl/blob/v1.14.1/trl/trainer/sft_trainer.py)

```python
training_args = SFTConfig(loss_type="chunked_nll", use_liger_kernel=False)
```

### 6.3 兼容性：文档与代码存在分歧

v1.14.1 的 [显存优化指南](https://github.com/huggingface/trl/blob/v1.14.1/docs/source/reducing_memory_usage.md) 仍把 PEFT 和 VLM 列为不兼容，但同版本源码包含 PEFT/VLM 处理路径，[官方测试](https://github.com/huggingface/trl/blob/v1.14.1/tests/test_sft_trainer.py) 也检查了部分模型的 loss 和梯度一致性。因此这里不能沿用“所有 PEFT/VLM 都不支持”，也不能反过来保证所有组合都支持。

可确认的边界是：

- 与 `use_liger_kernel=True` 互斥
- PEFT 路径会对内部基模型打补丁；**若输出 `lm_head` 本身被 LoRA 等 tuner 包装，trainer 会拒绝该配置**，因为直接读取 head 权重会漏掉 adapter 增量
- 对有受支持输出 head 结构的部分 VLM 有实现与测试，不代表任意视觉模型都兼容
- 量化、模型结构、adapter 目标层和并行后端的完整组合仍需要小步测试

本篇 QLoRA 起步示例显式使用 `loss_type="nll"`。若要切成 chunked，先比较同一批数据的 loss 和可训练参数梯度，再测显存。本版官方指南报告 Qwen3-1.7B 的特定测试中单 GPU 峰值约降 30%、FSDP2 × 4 最高约降 50%；这只是来源报告，不是本文实测。

## 7. Padding-Free：减少组 batch 时的补齐

Packing 决定样本如何装入窗口；Padding-Free 决定 batch 如何送进模型。后者把不同长度的序列展平，并通过边界信息让兼容的 attention 后端分别处理，避免每条都补到 batch 最长。它不会恢复截断内容，也不会自动修复样本边界。

![Padding-Free](https://huggingface.co/datasets/trl-lib/documentation-images/resolve/main/padding-free.png)
*图：Padding-Free 把 batch 展平为单条序列，完全消除 padding。*

```python
training_args = SFTConfig(
    packing=False,
    padding_free=True,
    max_length=None,
    model_init_kwargs={"attn_implementation": "flash_attention_2"},
)
```

这个片段假设已事先检查、过滤或切分超长输入。在本版中，`padding_free=True` 且未 packing 时不能保留默认的非空 `max_length`，否则 trainer 报错：该路径不负责按这个上限截断。

使用支持分段边界的 FlashAttention 路径是正确性的条件，不能把警告理解成“换个慢一点的后端也一样”。另外，若设置 `pad_to_multiple_of`，展平后的总长度仍可能为对齐补少量 padding；图中的“完全消除”指不再逐条补到最长这一理想情况。batch 只有一条且未 packing 时，本就没有跨样本长度差导致的补齐，通常不会得到相同收益。

不要把 SFT 的行为推广到其他 trainer；本版指南说明 DPO 的 Padding-Free 暂时会回退到普通 padding。

## 8. Activation Offloading：把部分激活暂存到 CPU

前向时把原本为反向保存的部分 tensor 搬到 CPU，需要时再取回。这减少 GPU 上的激活驻留，但会增加主机内存需求和 CPU↔GPU 传输。

```python
training_args = SFTConfig(activation_offloading=True)
```

其实现利用 PyTorch 的 saved-tensor hooks 管理张量保存与恢复。传输能否和计算重叠，取决于 tensor 大小、带宽、CPU 内存及实现。速度代价可能小，也可能很明显，不能用“轻微下降”或“代价很大”概括所有配置。它不自动卸载基模型权重或优化器状态，也不是任何 OOM 都能解决的兜底开关。[PyTorch saved-tensor hooks](https://pytorch.org/tutorials/intermediate/autograd_saved_tensors_hooks_tutorial.html)

## 9. Gradient Checkpointing：用重算减少保存

前向时只保留选定检查点，反向需要某段中间结果时再算一遍。它主要减少保存的激活，不消除权重、梯度、优化器状态，也未必能解决 checkpoint 区域之外的巨大 logits 分配。

```python
training_args = SFTConfig(gradient_checkpointing=True)
```

本版默认启用。收益与层划分、序列长度、kernel、是否同时分块计算 loss 有关；不能笼统保证从 $O(L)$ 降到 $O(\sqrt L)$，也不能固定写“慢 20%”或“增加 33% 计算”。[经典分段分析](https://arxiv.org/abs/1604.06174) 中的 $L$ 指层数，且结论有特定保存策略的前提。

[重算必须与原前向一致](https://docs.pytorch.org/docs/2.8/checkpoint.html)：会变化的全局状态、随机数处理、原地修改和缓存都可能影响结果。训练时通常关闭推理用的 KV cache；PEFT 的输入梯度需求、checkpoint 的 reentrant 模式与 ZeRO-3 也有组合限制。本版 trainer 对 PEFT + ZeRO-3 有专门处理，别把其他环境的 `use_reentrant=False` 配方无条件照搬。先跑几步，确认所有应训练参数有正确梯度，再测显存收益。

## 10. 其他工程开关与选读支线

### 10.1 pad_to_multiple_of：对齐不是越大越好

```python
training_args = SFTConfig(pad_to_multiple_of=8)
```

对齐可能帮助某些 kernel，但也会增加 token 槽位。例如长度 513 对齐到 8 的倍数是 520；对齐到 2048 就会补到 2048。应比较实际吞吐，不能默认选 2048，更不能说所有 Tensor Core 都因此显著加速。

### 10.2 在线 RL：生成与训练争抢显存

这一节是可跳过的对齐训练支线。在线 RL 会在训练过程中生成回答，再按奖励更新模型；GRPO 是其中一种方法。本篇只指出两个工程开关的冲突，不介绍其目标函数，也不把它当作 SFT 的必修前置。需要算法背景可选读 [Deep Reinforcement Learning](/posts/2024-04-30-deep-reinforcement-learning/)。

若使用 ZeRO-3 和普通 Transformers 生成，生成阶段收集完整权重可能制造峰值。可以考虑：

```python
from trl import GRPOConfig

training_args = GRPOConfig(
    use_vllm=False,
    ds3_gather_for_generation=False,
)
```

这是不开 vLLM 的路线，会有生成速度代价。官方 [GRPOConfig](https://github.com/huggingface/trl/blob/v1.14.1/trl/trainer/grpo_config.py) 明确说明：**关闭 `ds3_gather_for_generation` 不兼容 vLLM 生成**。不要把两个“省显存”开关盲目叠加。

若使用与训练同卡的 vLLM，可以考虑另一条路线：

```python
from trl import GRPOConfig

training_args = GRPOConfig(
    use_vllm=True,
    vllm_mode="colocate",
    vllm_enable_sleep_mode=True,
    ds3_gather_for_generation=True,
)
```

Sleep Mode 在优化阶段释放生成引擎占用，之后唤醒并同步权重。准确的权重卸载/重载行为取决于 sleep level 和集成版本；[vLLM 文档](https://docs.vllm.ai/en/latest/features/sleep_mode/) 中 level 1 将权重备份到 CPU 并**丢弃 KV cache**，不是把所有 KV cache 原封不动搬到 CPU。服务端模式的管理又不同，不能把上述 colocate 配置直接当作 server 操作指南。这些片段不包含数据、奖励函数与分布式启动配置，不是完整 RL 训练脚本。

## 11. 按瓶颈选方法

| 方法 | 主要减少哪一项 | 主要代价或前提 |
|------|----------------|----------------|
| Truncation | 长序列带来的激活与计算 | 可能删除答案和结束标记 |
| Packing | 短序列空槽位 | 检查样本隔离、长样本切分语义 |
| LoRA | 可训练参数的梯度与优化器状态 | 冻结基模型与激活仍占显存 |
| QLoRA | 冻结基模型权重，再配合 LoRA | 量化后端、dtype 与额外工作区 |
| Liger Kernel | 受支持算子的中间张量 | 模型覆盖；与本版 chunked loss 互斥 |
| Chunked NLL | 同时驻留的词表 logits | head/adapter/后端兼容性；逐块重算 |
| Padding-Free | batch 内逐条补齐 | 分段 attention 边界必须正确 |
| Activation Offloading | GPU 驻留激活 | CPU 内存与传输带宽 |
| Gradient Checkpointing | 为反向保存的激活 | 增加重算，核对梯度与缓存行为 |
| ZeRO-1/2/3 | 多卡之间重复的模型状态 | 通信、收集峰值与分片粒度 |

表中的项目不是收益排名。比如 batch 内长度几乎一致时，Packing/Padding-Free 收益可能很小；如果权重本身已占满显卡，单开 checkpointing 也放不下模型。

## 12. 从能验证的小配置开始

### 12.1 单卡 QLoRA 起步示例

先用少量样本、短上下文与普通 loss 验证数据和训练，不要在第一次运行时就把 packing、外部 kernel、offloading 全部打开。以下示例需要支持 BF16 和相应 bitsandbytes 后端的 GPU：

```python
import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

training_args = SFTConfig(
    output_dir="qlora-smoke",
    max_length=512,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    max_steps=2,
    learning_rate=1e-4,
    bf16=True,
    loss_type="nll",
    packing=False,
    gradient_checkpointing=True,
    model_init_kwargs={"dtype": torch.bfloat16},
    report_to="none",
)

trainer = SFTTrainer(
    model="Qwen/Qwen3-8B",
    args=training_args,
    train_dataset=load_dataset("trl-lib/Capybara", split="train[:16]"),
    quantization_config=quantization_config,
    peft_config=LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        task_type="CAUSAL_LM",
    ),
)
trainer.model.config.use_cache = False
trainer.model.print_trainable_parameters()
trainer.train()
```

这里采用对话语言建模的默认监督范围，并非 assistant-only。想只训练助手回答，应先验证模板能产生正确 mask，再加 `assistant_only_loss=True`。本例仅在 Q/V 投影上挂 LoRA，便于起步；目标层选择会影响能力与参数量，不代表复现了 QLoRA 论文的全部配置。

本版 `SFTTrainer` 在 `model` 为字符串时接受独立的 `quantization_config` 参数；如果先自行加载了模型对象，必须在模型加载阶段完成量化，不能指望这个 trainer 参数再量化一次。NF4 + LoRA 才使该示例成为 QLoRA；仅传一个 `LoraConfig` 不是 QLoRA。

### 12.2 每次只改一类变量

1. 解码几条实际输入，检查 prompt、回答和 EOS；确认 shift 后有效 label 数大于零
2. 短跑检查 loss 为有限数、应训练参数获得梯度、保存/加载的 adapter 正确
3. 记录峰值 allocated/reserved 显存与有效 token/s；CUDA 测量要先同步，并包含优化器状态建立后的训练步
4. 再逐项尝试增大长度、packing、chunked loss 或 Liger；更换 loss 实现时比较 loss 与梯度，更换 packing 时检查边界
5. 用独立验证集确认质量没有因截断、mask 或跨样本信息泄漏变差

需要多卡全参微调时，可继续读 [ZeRO 篇](/posts/2026-06-15-llm-mem-opt-4-zero/)，理解分片配置与容量取舍；完整启动流程仍需按训练框架文档配置。模型是 70B，并不意味着任意“ZeRO-3 + packing”组合就足够；还要算临时收集、激活与通信峰值。

<a id="13-系列总结"></a>

## 13. 从这篇走向下一步

单卡 SFT 先验证数据与 loss，保证训练目标和数据边界正确；接着查出哪一项占显存，逐项添加优化，并比较速度、质量与峰值。错误的 mask 或串样本的 attention 也可能跑出平滑下降的 loss，因此不能只看训练是否“能跑”。

需要分摊全参模型状态，读 [ZeRO](/posts/2026-06-15-llm-mem-opt-4-zero/)；需要让回答更符合偏好，读 [LLM 对齐](/posts/2025-12-05-a-series-on-llm-training-i/)。推理阶段的权重/KV 账本见 [推理显存](/posts/2026-06-13-llm-mem-opt-2-inference/)。

## 参考资料

- [TRL v1.14.1 release](https://github.com/huggingface/trl/releases/tag/v1.14.1)
- [固定版本 SFTConfig](https://github.com/huggingface/trl/blob/v1.14.1/trl/trainer/sft_config.py)、[SFTTrainer](https://github.com/huggingface/trl/blob/v1.14.1/trl/trainer/sft_trainer.py)、[SFT 测试](https://github.com/huggingface/trl/blob/v1.14.1/tests/test_sft_trainer.py)
- [固定版本显存优化指南](https://github.com/huggingface/trl/blob/v1.14.1/docs/source/reducing_memory_usage.md)（其 PEFT/VLM 的 chunked loss 描述与实现有分歧，见第 6 节）
- [固定版本 GRPOConfig](https://github.com/huggingface/trl/blob/v1.14.1/trl/trainer/grpo_config.py)
- Hu et al. *LoRA: Low-Rank Adaptation of Large Language Models*. ICLR 2022. [arXiv:2106.09685](https://arxiv.org/abs/2106.09685)
- Dettmers et al. *QLoRA: Efficient Finetuning of Quantized LLMs*. NeurIPS 2023. [arXiv:2305.14314](https://arxiv.org/abs/2305.14314)
- [Transformers bitsandbytes 量化说明](https://huggingface.co/docs/transformers/en/quantization/bitsandbytes)
- [Liger Kernel](https://github.com/linkedin/Liger-Kernel)
- [vLLM Sleep Mode](https://docs.vllm.ai/en/latest/features/sleep_mode/)
