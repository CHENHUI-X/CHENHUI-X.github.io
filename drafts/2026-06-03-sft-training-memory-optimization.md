---
title: TRL SFT 训练与显存优化全指南 — 从 Packing 到 Chunked NLL
published: 2026-06-03
description: 基于 HuggingFace TRL 库，详解 SFT Trainer 的使用方法，以及 Packing、Truncation、PEFT、Liger Kernel、Chunked Cross-Entropy 等显存优化技术的原理与实践。
category: Deep Learning
tags:
- llm
- sft
- memory
- trl
- fine-tuning
draft: true
---

## 0. 前言

大语言模型的微调（Supervised Fine-Tuning, SFT）是将预训练模型适配到特定任务或对话风格的最基础方法。但当你真正打开 Colab 或拿起一张消费级 GPU 准备微调时，第一个撞上的瓶颈总是同一个问题：

**显存不够。**

HuggingFace TRL 库的 `SFTTrainer` 提供了一套完整方案，从基础的截断到高级的 Packing、Chunked NLL 损失、Liger Kernel 等。这篇文章系统梳理这些技术，每项都附上原理和代码，力求让你读完就能上手。

---

## 1. SFT Trainer 快速入门

### 1.1 基本使用

`SFTTrainer` 是 TRL 中用于监督微调的核心类。以下代码用 Qwen 3 0.6B 在 Capybara 数据集上做 SFT：

```python
from trl import SFTTrainer
from datasets import load_dataset

trainer = SFTTrainer(
    model="Qwen/Qwen3-0.6B",
    train_dataset=load_dataset("trl-lib/Capybara", split="train"),
)
trainer.train()
```

就是这么简洁。`SFTTrainer` 会自动处理 tokenizer 加载、padding、truncation 等细节。

### 1.2 支持的数据集格式

SFT 支持两种主流格式：

**语言建模格式（标准）：**
```python
{"text": "The sky is blue."}
```

**对话格式：**
```python
{"messages": [
    {"role": "user", "content": "What color is the sky?"},
    {"role": "assistant", "content": "It is blue."}
]}
```

**Prompt-Completion 格式：**
```python
# 标准
{"prompt": "The sky is", "completion": " blue."}

# 对话式
{"prompt": [{"role": "user", "content": "What color is the sky?"}],
 "completion": [{"role": "assistant", "content": "It is blue."}]}
```

### 1.3 SFT 的损失计算

SFT 的核心是 **token 级别的交叉熵损失**：

$$
\mathcal{L}_{\text{SFT}}(\theta) = - \sum_{t=1}^{T} \log p_\theta(y_t \mid y_{<t})
$$

模型以自回归的方式逐个预测下一个 token，损失只计算非 padding 的位置。

![SFT Figure](https://huggingface.co/datasets/trl-lib/documentation-images/resolve/main/sft_figure.png)

*图：SFT 训练中，输入序列右移一位形成 labels，仅对非 padding token 计算交叉熵损失。*

#### One-Shot Label Shift

训练时，模型通过 **one-token shift** 来学习：输入序列右移一位即为目标 label。Padding token 的位置会被设为 `-100`（在交叉熵中忽略），确保只对有效 token 计算损失。

---

## 2. 显存优化：Truncation（截断）

### 2.1 为什么要截断？

数据集中序列长度差异很大。不做处理时，一个 batch 里的序列会 pad 到最长的那个，导致大量的 padding token 浪费显存。

![Truncation](https://huggingface.co/datasets/trl-lib/documentation-images/resolve/main/why_you_should_truncate.png)

*图：序列长度分布不均时，不加截断会导致大量 padding 浪费。*

### 2.2 如何配置

```python
from trl import SFTConfig

training_args = SFTConfig(..., max_length=2048)
```

SFT 的截断通过 `max_length` 参数控制，作用于整个输入序列。

![Truncation Input IDs](https://huggingface.co/datasets/trl-lib/documentation-images/resolve/main/truncation_input_ids.png)

*图：SFT 中截断应用于 input_ids 序列。*

### 2.3 如何选择 max_length？

- **太小**：大量 token 被丢弃，信息丢失
- **太大**：显存飙升，且很多 padding token 浪费算力

建议先用数据集的长度分布确定合适的阈值。TRL 提供了一个在线工具帮助可视化：https://trl-lib-dataset-length-profiler.hf.space

---

## 3. 显存优化：Packing（序列打包）

### 3.1 问题背景

Truncation 有两个根本缺陷：
1. **信息丢失** — 超长序列的尾部 token 被直接丢弃
2. **效率低下** — 太长的 `max_length` 导致 padding 浪费

**序列打包（Sequence Packing）** 提供了一个更优雅的解法：把多条短序列拼成一条长序列，填充到 `max_length`，几乎消除 padding。

![Packing](https://huggingface.co/datasets/trl-lib/documentation-images/resolve/main/packing_3.png)

*图：Packing 将多条序列拼接到一个训练样本中，最大化利用上下文窗口。*

### 3.2 注意力掩码的问题

但你发现了问题吗？当多条序列拼在一起，模型在计算 Attention 时不能让 token 看到"隔壁"序列的内容。否则就变成了跨样本的信息泄露。

以一个简单的例子说明。我们把三句话用 EOS token 拼接：

```
The cat sat on the mat<|endoftext|>
The dog ate my homework<|endoftext|>
My aunt is a teacher<|endoftext|>
```

标准因果注意力掩码是下三角矩阵。但如果直接这样用，第二句的 token 可以看到第一句的内容——这不正确。

![Standard attention mask](https://cdn-uploads.huggingface.co/production/uploads/6041ff7ff84ebe399f1c85ea/6oIdEuI4N3HpPuY5AaGKX.png)

*图：标准的因果注意力掩码（下三角矩阵），但 pack 之后会出现跨序列的注意力。*

正确的做法是：**在因果掩码的基础上，额外把序列边界之外的 token 掩掉。**

![Corrected attention mask](https://cdn-uploads.huggingface.co/production/uploads/6041ff7ff84ebe399f1c85ea/gd0J4zQedkGw1hNYcwTgX.png)

*图：修正后的注意力掩码——每个序列内的注意力是因果的，但跨序列被完全掩码。*

### 3.3 单条序列的注意力掩码实现

单条 pack 序列（batch size = 1）的实现相对简单：

```python
def get_attention_mask_for_packed_sequence(x, token_id, eos: bool = True):
    T = x.size(0)
    # 找到所有 EOS token 的位置
    eos_indices = (x == token_id).nonzero().squeeze()
    # 计算每条子序列的长度
    reps = torch.cat([eos_indices[[0]] + 1, eos_indices[1:] - eos_indices[:-1]])
    # 重复每个 EOS 索引 n 次（n 是子序列的 token 数）
    repeated_idx = torch.repeat_interleave(eos_indices, reps).view(1, -1).expand(T, -1)
    # 创建从 0 到 T-1 的索引矩阵
    mask_indices = torch.arange(T).view(-1, 1).expand(-1, T)
    # 因果掩码 + 跨序列掩码
    mask = torch.ones(T, T, dtype=torch.bool).tril().expand(-1, -1)
    mask.masked_fill_(mask_indices > repeated_idx, False)
    return mask
```

### 3.4 Position ID 也需要调整

pack 时，每个子序列的 position ID 需要**重置为从 0 开始**，而不是接着上一个子序列的末尾。

```python
eos_indices = (tokenized_sentences == tokenizer.eos_token_id).nonzero().squeeze()
# 计算每条子序列的长度
reps = torch.cat([eos_indices[[0]] + 1, eos_indices[1:] - eos_indices[:-1]])
print(f"EOS indices: {eos_indices}")
print(f"Subsequence lengths (reps): {reps}")

# Position IDs = 全局索引 - 累积偏移量
pos_ids = torch.arange(T) - torch.repeat_interleave(
    torch.cat([torch.tensor([0]), eos_indices + 1], dim=0)[:-1], reps
)
print(f"Position IDs: {pos_ids}")
```

输出：

```
EOS indices: tensor([ 6, 12, 18])
Subsequence lengths (reps): tensor([7, 6, 6])

Position IDs: tensor([0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5])
```

逐 token 看更清楚：

| Token | pos_id |
|-------|--------|
| The           | 0 |
| cat           | 1 |
| sat           | 2 |
| on            | 3 |
| the           | 4 |
| mat           | 5 |
| `<\|endoftext\|>` | 6 |
| **The**       | **0 ← 重置** |
| dog           | 1 |
| ate           | 2 |
| my            | 3 |
| homework      | 4 |
| `<\|endoftext\|>` | 5 |
| **My**        | **0 ← 重置** |
| aunt          | 1 |
| is            | 2 |
| a             | 3 |
| teacher       | 4 |
| `<\|endoftext\|>` | 5 |

可以看到每个子序列的 position ID 都从 0 开始，模型不会误以为某句话的第 8 个 token 和下一句话的第 1 个 token 相邻。

### 3.5 Batch 级别的注意力掩码

我们真正需要的是 batch 级别的实现。先构建 batch：

```python
# 第二个 batch item: pack 另外两句话
sentence4 = "Rome wasn't built in a day"
sentence5 = "My hovercraft is full of eels"

sentences = [sentence4, sentence5]
tokenized_sentences2 = tokenizer(sentences, return_attention_mask=False,
                                 add_special_tokens=False)["input_ids"]
tokenized_sentences2 = torch.tensor(
    [t for s in tokenized_sentences2 for t in s + [tokenizer.eos_token_id]]
)

batch = torch.nn.utils.rnn.pad_sequence(
    [tokenized_sentences, tokenized_sentences2],
    batch_first=True, padding_value=tokenizer.eos_token_id
)
print(f"Batch shape: {batch.shape}")  # [2, 19]
```

用完整的 batch 函数计算注意力掩码：

```python
def get_attention_mask_for_packed_sequence(x, token_id, eos: bool = True):
    B, T = x.shape
    eos_idx = (x.view(-1) == token_id).nonzero(as_tuple=True)[0] + eos
    eos_idx_expanded = torch.cat(
        [eos_idx, torch.arange(0, B * T + 1, T)]
    ).unique().sort()[0]
    normalized_idx = eos_idx_expanded - (eos_idx_expanded // T) * T
    normalized_idx = torch.where(normalized_idx == 0, T, normalized_idx)
    reps = normalized_idx[1:] - normalized_idx[:-1]
    reps = torch.where(reps < 1, normalized_idx[1:], reps)
    repeated_idx = torch.repeat_interleave(
        normalized_idx[1:], reps
    ).view(B, 1, T).expand(-1, T, -1)
    mask_indices = torch.arange(T).view(1, -1, 1).expand(B, -1, T)
    mask = torch.ones(T, T, dtype=torch.bool).tril().expand(B, -1, -1)
    mask = mask.masked_fill(mask_indices >= repeated_idx, False)
    return mask

mask = get_attention_mask_for_packed_sequence(batch, tokenizer.eos_token_id)
print(f"Mask shape: {mask.shape}")  # [2, 19, 19]
```

输出结果（`mask[0]` 和 `mask[1]` 各为 19×19 的矩阵，1 表示允许注意力，0 表示掩码）：

Batch item 0（3 条序列，EOS 在 6/12/18）：
```
[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  ← token 0
[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  ← token 1
[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
[1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  ← EOS(6)
[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  ← seq2 start
[0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]  ← seq3 start
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0]
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0]
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0]
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]  ← EOS(18)
```

注意力矩阵被切成三个**对角块**，每块内部是因果三角形，块之间**完全零掩码**——序列 1 看不到序列 2 和 3，序列 2 看不到序列 3，反之亦然。这就是 packing 的正确注意力模式。

![Batch attention mask](https://cdn-uploads.huggingface.co/production/uploads/6041ff7ff84ebe399f1c85ea/BFQXqx5_s1aVZ94Yw1VMx.png)

*图：Batch packing 的注意力掩码可视化——三个 block-diagonal 的因果三角。*

### 3.6 TRL 中的 Packing 配置

```python
from trl import SFTConfig

training_args = SFTConfig(
    ...,
    packing=True,
    packing_strategy="bfd",  # "bfd" | "bfd_split" | "wrapped"
    max_length=2048,
)
```

TRL 支持三种 packing 策略：

- **`"bfd"`（默认）**：Best-Fit Decreasing 装箱算法，序列超出 `max_length` 的部分被丢弃
- **`"bfd_split"`**：同上，但超出 `max_length` 的长序列会被切分，保留所有 token（来自 [Fewer Truncations Improve Language Modeling](https://huggingface.co/papers/2404.10830)）
- **`"wrapped"`**：所有 token 拼接成一个流，再按 `max_length` 切块。padding 最少，但会打断序列连续性（影响模型性能，见 [Qwen3-Coder-Next Tech Report](https://huggingface.co/papers/2603.00729)）

> **注意**：如果所有序列都短于 `max_length`，`bfd` 和 `bfd_split` 行为完全一致。

> **注意**：Packing 目前仅支持 **SFT** 且需要 **FlashAttention**（或其变体）。

---

## 4. 显存优化：PEFT（参数高效微调）

PEFT 方法（如 LoRA）是最有效的显存优化手段之一。它只训练少量适配器参数，大幅降低显存需求。

```python
from datasets import load_dataset
from peft import LoraConfig
from trl import SFTTrainer

dataset = load_dataset("trl-lib/Capybara", split="train")

trainer = SFTTrainer(
    model="Qwen/Qwen2.5-0.5B",
    train_dataset=dataset,
    peft_config=LoraConfig(),
)
trainer.train()
```

LoRA 可以和量化（4-bit / 8-bit）结合，进一步降低显存。训练适配器时推荐使用更高的学习率（~1e-4）。

---

## 5. 显存优化：Liger Kernel

[Liger Kernel](https://github.com/linkedin/Liger-Kernel) 是 LinkedIn 开源的 Triton kernel 集合，专为 LLM 训练优化：

- 多 GPU 吞吐量提升 **20%**
- 显存减少 **60%**
- 支持 FlashAttention、FSDP、DeepSpeed

```python
from trl import SFTConfig

training_args = SFTConfig(..., use_liger_kernel=True)
```

Liger Kernel 通过对 `lm_head`、MLP、LayerNorm、RoPE 等关键操作定制 Triton kernel，减少了中间激活的显存占用。

---

## 6. 显存优化：Chunked Cross-Entropy（分块交叉熵）

### 6.1 问题

当词表很大时，`[batch × seq_len × vocab]` 大小的 logits 张量是显存的头号杀手。例如 Qwen3-1.7B 的词表约 151K，全量 logits 的显存占用非常恐怖。

### 6.2 解决方案

`loss_type="chunked_nll"` 将一个大的 logits 矩阵拆分成小块计算：

- 在 `lm_head` 矩阵乘法之前**先过滤掉** `labels == -100` 的 token
- 分块计算交叉熵，配合 gradient checkpointing
- 峰值激活显存从 `batch × seq_len × vocab` 降为 `chunk_size × vocab`

```python
from trl import SFTConfig

training_args = SFTConfig(..., loss_type="chunked_nll")
```

**实测效果**（Qwen3-1.7B, vocab≈151K）：
- 单 GPU：峰值显存减少 ~**30%**
- FSDP2 × 4 GPUs：峰值显存减少 ~**50%**
- 训练时间：基本持平或略快

> **重要**：与 `use_liger_kernel=True`、PEFT、VLM 不兼容。
> 如果使用 FSDP2，需要传 `--fsdp_reshard_after_forward false` 给 `accelerate launch`，否则 backward 时每个 chunk 都要重新 gather `lm_head.weight`。

---

## 7. 显存优化：Padding-Free（无填充批处理）

与 Packing 不同，Padding-Free 是把一个 batch 内的所有序列**展平为一条序列**，而不是装箱拼接。所有序列保持完整。

![Padding-free](https://huggingface.co/datasets/trl-lib/documentation-images/resolve/main/padding-free.png)

*图：Padding-free 将 batch 展平为单条序列，完全消除 padding。*

```python
from trl import SFTConfig

training_args = SFTConfig(
    ...,
    padding_free=True,
    model_init_kwargs={"attn_implementation": "kernels-community/flash-attn2"}
)
```

> **强烈建议**配合 **FlashAttention 2/3** 使用，否则会出现 batch 污染问题。

---

## 8. 显存优化：Activation Offloading（激活值卸载）

Activation offloading 通过在 forward 过程中**将激活值临时转移到 CPU 内存**，在 backward 时再取回，来降低 GPU 显存峰值。代价是轻微的训练速度下降。

```python
from trl import SFTConfig

training_args = SFTConfig(..., activation_offloading=True)
```

底层使用 PyTorch 的 [`saved_tensors_hooks`](https://pytorch.org/tutorials/intermediate/autograd_saved_tensors_hooks_tutorial.html#hooks-for-autograd-saved-tensors) 拦截 forward 过程中的激活 tensor，智能判断哪些该卸载、哪些保留在原地。默认启用 CUDA stream 来异步传输，与计算重叠。

---

## 9. 显存优化：Gradient Checkpointing（梯度检查点）

梯度检查点用计算换显存：forward 时不保存中间激活，backward 时重新计算。

```python
from trl import SFTConfig

training_args = SFTConfig(..., gradient_checkpointing=True)
```

> **注意**：TRL 中 gradient checkpointing **默认已启用**。

---

## 10. 其他优化技巧

### 10.1 Pad to Multiple

将序列 padding 到指定值的倍数，对齐到硬件友好的边界：

```python
from trl import SFTConfig

training_args = SFTConfig(..., pad_to_multiple_of=2048)
```

目前支持 SFT 和 Reward 两个 trainer。

### 10.2 DeepSpeed ZeRO-3 下关闭模型收集

使用 DeepSpeed ZeRO-3 时，在线方法（GRPO、PPO 等）需要收集模型权重进行生成，对大模型可能导致 OOM。可以关闭收集：

```python
from trl import GRPOConfig

training_args = GRPOConfig(..., ds3_gather_for_generation=False)
```

代价是生成速度略有下降。

### 10.3 vLLM Sleep Mode

用 vLLM 做在线训练时，可以在优化步骤中将 vLLM 的参数和缓存卸载到 CPU，生成时再加载回 GPU：

```python
from trl import GRPOConfig

training_args = GRPOConfig(..., vllm_enable_sleep_mode=True)
```

---

## 11. 总结

| 优化方法 | 显存节省 | 速度影响 | 适用场景 |
|---------|---------|---------|---------|
| **Truncation** | ⭐⭐⭐ | 中性 | 所有场景，基础操作 |
| **Packing** | ⭐⭐⭐⭐ | 略微加快 | SFT + FlashAttention |
| **PEFT (LoRA)** | ⭐⭐⭐⭐⭐ | 略微加快 | 大模型、单卡场景 |
| **Liger Kernel** | ⭐⭐⭐⭐ | 加快 20% | 多 GPU 训练 |
| **Chunked NLL** | ⭐⭐⭐ | 中性或略快 | 大词表模型（非 PEFT） |
| **Padding-Free** | ⭐⭐⭐ | 略微加快 | SFT/DPO + FlashAttention |
| **Activation Offloading** | ⭐⭐⭐ | 略微变慢 | 显存紧张时兜底 |
| **Gradient Checkpointing** | ⭐⭐⭐⭐ | 变慢 ~20% | 所有场景（默认开启） |

**推荐的组合套路：**
1. **消费级单卡（RTX 4090 ~ 24GB）微调 7B 模型**：LoRA + Packing + Gradient Checkpointing + 4-bit Quantization
2. **多卡训练大模型（70B+）**：Liger Kernel + Packing + FSDP/DeepSpeed + Chunked NLL
3. **在线 RL 训练（GRPO / PPO）**：vLLM Sleep Mode + Activation Offloading + Padding-Free

选择哪种组合取决于你的模型大小、词表大小、硬件配置和训练方法。建议从最基础的 Truncation 开始，逐步叠加各项优化，监控显存和训练速度的变化。

---

## 参考资料

- [TRL SFT Trainer 文档](https://huggingface.co/docs/trl/sft_trainer)
- [TRL Reducing Memory Usage 文档](https://huggingface.co/docs/trl/reducing_memory_usage)
- [Efficient LLM Pretraining: Packed Sequences and Masked Attention](https://huggingface.co/blog/sirluk/llm-sequence-packing)
- [Fewer Truncations Improve Language Modeling](https://huggingface.co/papers/2404.10830)
- [Liger Kernel](https://github.com/linkedin/Liger-Kernel)
- [Transformers Performance Guide](https://huggingface.co/docs/transformers/perf_train_gpu_one)
