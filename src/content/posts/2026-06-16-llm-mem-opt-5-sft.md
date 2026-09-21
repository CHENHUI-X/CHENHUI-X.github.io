---
title: LLM 训练与推理优化（五）— SFT 训练实战：Packing 到 Chunked NLL
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

> 这是「LLM 训练与推理优化系列」第 5 篇（终篇）。前面 [第 4 篇 — 训练显存与 ZeRO](/posts/2026-06-15-llm-mem-opt-4-zero/) 讲了如何用多卡分担模型状态；本篇讨论如何进一步降低每张卡的显存占用。两类方法可以组合，但实际收益取决于训练配置。

## 0. 消费级单卡 SFT 7B 的工程难题

回顾 [第 1 篇](/posts/2026-06-12-llm-mem-opt-1-fundamentals/)：训练 7B 模型完整模型状态 = 112 GB。一张 RTX 4090 24 GB 怎么训？

如果光开 ZeRO 是不够的——单卡场景没有"其他卡"可以分摊。这时需要工程层面的招式：

- **LoRA** 只训练一小部分新增参数，减少梯度和优化器状态占用
- **Packing** 把短样本拼在一起，减少补齐长度的浪费
- **Chunked NLL** 分块计算输出词表的损失，降低峰值显存
- **Gradient Checkpointing** 在反向传播时重算部分中间结果，减少需要保存的激活值
- **Activation Offloading** 把部分激活值暂存到 CPU 内存

Hugging Face **TRL** 的 `SFTTrainer` 提供了相应的配置入口。本篇逐个解释作用、限制和用法。

## 1. SFT Trainer 快速入门

### 1.1 最小可用示例

```python
from trl import SFTTrainer
from datasets import load_dataset

trainer = SFTTrainer(
    model="Qwen/Qwen3-0.6B",
    train_dataset=load_dataset("trl-lib/Capybara", split="train"),
)
trainer.train()
```

`SFTTrainer` 自动处理 tokenizer 加载、padding、truncation、损失 mask 等。

### 1.2 数据格式

支持三种主流格式：

```python
# 语言建模（标准）
{"text": "The sky is blue."}

# 对话格式
{"messages": [
    {"role": "user", "content": "What color is the sky?"},
    {"role": "assistant", "content": "It is blue."}
]}

# Prompt-Completion
{"prompt": "The sky is", "completion": " blue."}
```

### 1.3 损失函数

SFT 是 token 级的交叉熵：

$$
\mathcal{L}_{\text{SFT}}(\theta) = -\sum_{t=1}^{T} \log p_\theta(y_t \mid y_{<t})
$$

因果语言模型用当前位置之前的 token 预测下一个 token；训练实现通常在计算损失时对预测值和标签做错位对齐。Padding 位置的标签设为 `-100`，在交叉熵中被忽略；对话数据还可以进一步只计算助手回复部分的损失。

![SFT Figure](https://huggingface.co/datasets/trl-lib/documentation-images/resolve/main/sft_figure.png)
*图：SFT 训练中，预测值与下一 token 的标签对齐；padding 位置不参与损失计算。*

## 2. Truncation：最朴素的省显存

### 2.1 为什么需要

batch 内序列长度不齐时，所有序列要 pad 到最长那条。如果 1% 的序列特别长，剩下 99% 都被拖着 pad 到那个长度——大量 padding token 浪费显存与算力。

![Truncation](https://huggingface.co/datasets/trl-lib/documentation-images/resolve/main/why_you_should_truncate.png)
*图：序列长度分布不均时，不加截断会造成大量 padding。*

### 2.2 配置

```python
from trl import SFTConfig

training_args = SFTConfig(max_length=2048)
```

### 2.3 选 max_length 的取舍

- 太小：超长样本被截掉尾部，信息丢失
- 太大：padding 占比高，显存浪费

实操做法：先看数据集的长度分布（TRL 提供在线工具 `https://trl-lib-dataset-length-profiler.hf.space`），选 P90 ~ P95 分位数附近的值。

但 truncation 终究是"用信息损失换显存"的笨办法。下面的 Packing 是更优雅的解。

## 3. Packing：拼接序列消除 padding

### 3.1 思路

Truncation 砍尾巴，Packing 是 **拼接**：把多条短序列首尾相接成一条接近 `max_length` 的长序列，几乎消除 padding。

![Packing](https://huggingface.co/datasets/trl-lib/documentation-images/resolve/main/packing_3.png)
*图：Packing 把多条序列拼到一个训练样本中，最大化利用上下文窗口。*

打包后的注意力边界、position ID 和跨样本语义由**所选 packing strategy、模型及 FlashAttention 实现**共同决定；不要用仅按 EOS 扫描的手写 mask 取代 trainer 的实现。自定义 collator 时，应以当前 TRL/模型文档和单元测试确认每个样本的可见范围与位置编码约定。

### 3.2 TRL 配置

```python
training_args = SFTConfig(
    packing=True,
    packing_strategy="bfd",  # "bfd" | "bfd_split" | "wrapped"
    max_length=2048,
)
```

三种策略：

| 策略 | 行为 |
|------|------|
| `bfd`（默认） | Best-Fit Decreasing 装箱算法，超出 `max_length` 的部分被截断 |
| `bfd_split` | 同上但超长序列被切分保留所有 token（来自 [Fewer Truncations Improve LM](https://huggingface.co/papers/2404.10830)） |
| `wrapped` | 所有 token 拼成连续流再按 `max_length` 切块。padding 最少，但会打断序列连续性 |

> 这里使用的 TRL 默认 `bfd` 策略会自动开启 `padding_free`，因此需要兼容的 FlashAttention 2/3。不能据此推断所有 Packing 实现都必须使用 FlashAttention。

> 所有序列都短于 `max_length` 时，`bfd` 和 `bfd_split` 行为完全一致。

## 4. PEFT (LoRA)：只训练一小部分参数

LoRA 是单卡微调大模型最重要的一招。它在原参数旁挂一对低秩矩阵 $A, B$，原参数冻结，只训练 $A, B$：

$$
W' = W + \alpha\, B A, \quad B \in \mathbb{R}^{d \times r}, A \in \mathbb{R}^{r \times d},\ r \ll d
$$

可训练参数从 $d^2$ 降到 $2dr$；实际占比由 rank、目标层和模型结构决定，常见配置可远低于全参微调。

显存收益是连锁的：

- 优化器状态和梯度只需为可训练的 $A,B$ 保存。对单个 $d\times d$ 权重矩阵，LoRA 新增的可训练参数是 $2dr$，而不是 $d^2$；全模型能省多少，还取决于挂载的层、优化器、精度和未冻结的其他参数，不能把这个单层比例直接当作整模型显存比例
- 配合 **QLoRA**（4-bit 量化基模型 + LoRA）可显著降低基模型权重与可训练状态的显存；QLoRA 论文报告的是在**单张 48 GB GPU**上微调 65B 模型，不应改写成 24 GB 的通用结论

```python
from peft import LoraConfig
from trl import SFTTrainer

trainer = SFTTrainer(
    model="Qwen/Qwen3-0.6B",
    train_dataset=load_dataset("trl-lib/Capybara", split="train"),
    peft_config=LoraConfig(),
)
```

LoRA 训练通常用稍高的学习率（~1e-4），因为可训练参数极少，更新需要更明显。

### 4.1 QLoRA：量化基模型 + LoRA

QLoRA 在 LoRA 的基础上，把基模型量化到 4-bit（NF4 格式），只在前向时反量化到 BF16 计算。显存收益叠加：

- 基模型权重：NF4 编码本体为 4 bit/参数；实际占用还包括量化常数、分组元数据、对齐与运行时工作区
- LoRA 适配器：通常仍用 BF16/FP16，参数量取决于 rank 与目标层
- 优化器状态：只针对 LoRA 参数

三个关键技巧（来自 QLoRA 论文 [^qlora]）：

1. **NF4（NormalFloat4）**：一种非均匀量化格式，专门为「正态分布」的权重设计——量化级别在 0 附近更密、两端更疏，比均匀 INT4 精度高
2. **双重量化（Double Quantization）**：连量化常数（scale）自己也做一次 FP8 量化，每个参数额外省 ~0.4 bit
3. **分页优化器（Paged Optimizers）**：用 unified memory 把优化器状态分页到 CPU，OOM 时自动换出

```python
import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

trainer = SFTTrainer(
    model="Qwen/Qwen3-8B",
    args=SFTConfig(max_length=2048),
    train_dataset=load_dataset("trl-lib/Capybara", split="train"),
    quantization_config=bnb_config,
    peft_config=LoraConfig(r=16, lora_alpha=32),
)
```

这里由 `BitsAndBytesConfig` 在加载时量化基模型，并由 `peft_config` 只训练 LoRA adapter；所需显存仍取决于序列长度、batch、target modules、dtype 和内核。量化原理详见 [第 3 篇 — 量化深入](/posts/2026-06-14-llm-mem-opt-3-quantization/)。

[^qlora]: Dettmers et al. *QLoRA: Efficient Finetuning of Quantized LLMs*. NeurIPS 2023. [arXiv:2305.14314](https://arxiv.org/abs/2305.14314)

## 5. Liger Kernel：高效 Triton kernel 集合

[Liger Kernel](https://github.com/linkedin/Liger-Kernel) 是 LinkedIn 开源的 Triton kernel 集合，专为 LLM 训练定制——重写了 `lm_head`、SwiGLU、LayerNorm/RMSNorm、RoPE、Cross-Entropy 等关键算子，融合多个操作减少中间激活。

TRL 文档给出的“吞吐 +20%、显存 -60%”来自其引用的 Liger benchmark；它不是所有模型、序列长度、并行策略和 kernel 组合下的保证，应在目标训练配置上复测。

```python
training_args = SFTConfig(use_liger_kernel=True)
```

TRL 文档列出其与 FlashAttention、FSDP、DeepSpeed 的集成；启用前仍需检查模型算子覆盖、版本与下文的 Chunked NLL 兼容性。

## 6. Chunked Cross-Entropy：解决 logits 显存炸弹

### 6.1 大词表的麻烦

LM head 输出的 logits 形状是 `[batch, seq_len, vocab]`。Qwen3 的 vocab ≈ 152K，Llama-3 是 128K。算一下：

- batch=4, seq=2048, vocab=152K, BF16
- logits 张量 = $4 \times 2048 \times 152000 \times 2 \approx 2.5\,\text{GB}$

这只是 logits 自身的理论大小。反向传播还可能需要保存或生成与 logits 同量级的张量，但具体峰值取决于交叉熵实现、精度、张量是否复用及计算是否融合，不能固定写成“必占 5 GB”。

### 6.2 思路

观察：交叉熵 loss 一次只需要每个位置的"自己那个 label 对应的 logit"和归一化项 $\log \sum_v \exp(\text{logit}_v)$，**不需要同时持有全部位置的全 vocab logits**。

Chunked NLL：

1. 在 LM head 矩阵乘之前 **先过滤掉 `labels == -100`** 的位置（padding/prompt 部分）
2. 把剩下的有效 token 切成 chunk
3. 每个 chunk 单独算 logits → 单独算 loss → 释放
4. 配合 gradient checkpointing 在反向重新计算

峰值显存：`batch × seq × vocab` → `chunk_size × vocab`。

### 6.3 配置

```python
training_args = SFTConfig(loss_type="chunked_nll")
```

TRL 官方 benchmark（Qwen3-1.7B，词表约 152K）报告：

- 单 GPU 峰值 -30%
- FSDP2 × 4 GPU 峰值 -50%
- 训练时间基本持平或略快

> 当前 TRL 中，`chunked_nll` 是 `SFTTrainer` 的默认 loss；`use_liger_kernel=True` 时会自动改用 `nll`，两者不兼容。`chunked_nll` 也不兼容 PEFT 和 VLM。
>
> **如何取舍**：在非 PEFT 场景下，如果大词表产生的 logits 张量是显存瓶颈，可以使用 Chunked NLL。Liger Kernel 优化的是一组算子，是否更合适，要看模型支持情况和实际测试结果。
>
> 30%/50% 是上述官方 benchmark 的结果，不是普遍收益；FSDP 选项、模型结构和词表大小都会改变结果。

## 7. Padding-Free：把 batch 展平为单条序列

和 Packing 不同：Packing 决定如何组织样本，`bfd` 对超过 `max_length` 的部分会截断；Padding-Free 则在计算时**把 batch 展平**，以减少 padding。它不会恢复已被截断的内容。

![Padding-Free](https://huggingface.co/datasets/trl-lib/documentation-images/resolve/main/padding-free.png)
*图：Padding-Free 把 batch 展平为单条序列，完全消除 padding。*

```python
training_args = SFTConfig(
    padding_free=True,
    model_init_kwargs={"attn_implementation": "kernels-community/flash-attn2"},
)
```

> TRL 强烈建议 Padding-Free 配合 FlashAttention 2/3，否则可能出现 batch contamination。当前 `DPOTrainer` 的 `padding_free=True` 暂不可用：会告警并回退为标准 padding；不要把它作为 DPO 的生效优化项。

## 8. Activation Offloading：把激活搬到 CPU

前向时把中间激活临时搬到 CPU 内存，反向时再取回。代价是 CPU↔GPU 拷贝带来的轻微速度下降。

```python
training_args = SFTConfig(activation_offloading=True)
```

底层用 PyTorch 的 [`saved_tensors_hooks`](https://pytorch.org/tutorials/intermediate/autograd_saved_tensors_hooks_tutorial.html) 拦截 forward 中的激活 tensor，智能判断哪些卸载哪些保留，默认开 CUDA stream 异步传输与计算重叠。

显存紧到 OOM 边缘时的"兜底招式"，不到迫不得已不用，因为速度损失可观。

## 9. Gradient Checkpointing：用计算换显存

这个方法用计算时间换显存：前向时只保留选定的检查点，反向传播需要用到中间结果时再算一遍。保留哪些层、每段有多长由实现决定，因此**不能说打开开关后显存一定从 $O(L)$ 降到 $O(\sqrt L)$，或计算开销固定增加 33%**。经典论文给出了特定分段策略下 $O(\sqrt L)$ 的结果；实际训练应以目标配置的显存峰值和吞吐实测为准。

```python
training_args = SFTConfig(gradient_checkpointing=True)
```

> TRL 中 gradient checkpointing **默认已启用**。

## 10. 其他工程开关

### 10.1 pad_to_multiple_of：硬件友好对齐

```python
training_args = SFTConfig(pad_to_multiple_of=2048)
```

把序列 pad 到 64/128/2048 的倍数。Tensor Core 在某些维度对齐下吞吐显著提升。SFT 和 Reward 两个 trainer 支持。

### 10.2 ZeRO-3 + 在线 RL

使用 DeepSpeed ZeRO-3 的在线方法在生成时可能临时 gather 模型权重，大模型因此可能 OOM。对 GRPO 可关闭该 gather（生成速度可能下降）：

```python
from trl import GRPOConfig
training_args = GRPOConfig(ds3_gather_for_generation=False)
```

### 10.3 vLLM Sleep Mode

在线 RL 用 vLLM 做生成时，把 vLLM 的参数和缓存在优化步暂时卸载到 CPU：

```python
training_args = GRPOConfig(vllm_enable_sleep_mode=True)
```

显存腾出来给训练，生成时再加载回 GPU。

## 11. 优化对照表

| 方法 | 显存收益 | 速度影响 | 适用场景 |
|------|---------|---------|---------|
| Truncation | ★★★ | 中性 | 所有场景，基础操作 |
| Packing | ★★★★ | 略快 | SFT + FlashAttention |
| PEFT (LoRA) | ★★★★★ | 略快 | 大模型、单卡 |
| Liger Kernel | 取决于 kernel 覆盖 | 见目标 benchmark | 支持的训练配置 |
| Chunked NLL | 与有效 token 和词表有关 | 官方特定 benchmark 约 30%/50% 降峰值 | 大词表，**非 PEFT/VLM** |
| Padding-Free | 与长度分布有关 | 需实测 | SFT + FlashAttention；DPO 当前不可用 |
| Activation Offloading | ★★★ | 略慢 | OOM 边缘兜底 |
| Gradient Checkpointing | ★★★★ | 慢 ~20% | 默认开启 |
| QLoRA (PEFT + 4-bit) | ★★★★★ | 略慢 | 消费级单卡 |
| ZeRO-2/3（[第 4 篇](/posts/2026-06-15-llm-mem-opt-4-zero/)） | ★★★★ / ★★★★★ | 中性 / 慢 ~20% | 多卡 |

## 12. 推荐组合

### 单卡约 7/8B QLoRA 配置示例（显存以实测为准）

```python
import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

dataset = load_dataset("trl-lib/Capybara", split="train")

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

training_args = SFTConfig(
    max_length=2048,
    packing=True,
    model_init_kwargs={
        "attn_implementation": "kernels-community/flash-attn2",
    },
)

trainer = SFTTrainer(
    model="Qwen/Qwen3-8B",
    args=training_args,
    train_dataset=dataset,
    quantization_config=quantization_config,
    peft_config=LoraConfig(r=16, lora_alpha=32),
)
```

该示例同时传入 4-bit `BitsAndBytesConfig` 和 `LoraConfig`，因此是 QLoRA；packing 还要求安装并验证兼容的 FlashAttention。它没有声称固定显存可运行：开始训练前应以目标 GPU、实际 batch 和数据长度记录峰值显存。

### 多卡训 70B+

```python
training_args = SFTConfig(
    max_length=4096,
    packing=True,
    use_liger_kernel=True,
    gradient_checkpointing=True,
)
# 启动时配 FSDP / DeepSpeed ZeRO-3
# 如果显存瓶颈在 logits 且非 PEFT，可换 loss_type="chunked_nll"（需关掉 use_liger_kernel）
```

### 在线 RL（GRPO）

```python
training_args = GRPOConfig(
    use_vllm=True,
    vllm_enable_sleep_mode=True,
    ds3_gather_for_generation=False,
)
```

`vllm_enable_sleep_mode` 与 `ds3_gather_for_generation` 是在线生成的显存取舍；前者会引入主机与设备间传输，后者可能降低生成速度。不要把 `padding_free` 或 `activation_offloading` 当作当前 `GRPOConfig` 的开关叠加进去。

## 13. 系列总结

回到 [第 1 篇](/posts/2026-06-12-llm-mem-opt-1-fundamentals/) 末尾的"优化技术地图"：

| 显存大头 | 工具 | 在哪一篇 |
|---------|------|---------|
| 推理：模型权重 | 量化（GPTQ / AWQ） | 第 2、3 篇 |
| 推理：KV Cache | GQA、KV 量化、PagedAttention | 第 2 篇 |
| 推理：激活 | Flash Attention | 第 2 篇 |
| 训练：4 类模型状态冗余 | ZeRO-1/2/3 | 第 4 篇 |
| 训练：单机显存极限 | ZeRO-Offload / Infinity | 第 4 篇 |
| 训练：可训参数过多 | PEFT (LoRA / QLoRA) | 第 5 篇 |
| 训练：padding 浪费 | Packing / Padding-Free | 第 5 篇 |
| 训练：logits 显存炸弹 | Chunked NLL | 第 5 篇 |
| 训练：激活峰值 | Gradient Checkpointing / Activation Offloading | 第 5 篇 |
| 训练：kernel 效率 | Liger Kernel | 第 5 篇 |

每一项工具的背后都是一行可计算的显存账。要部署一个新模型，掐着指头照着这张表过一遍——「我这场景对应哪几条？最优组合是什么？」——比靠经验"试一下能不能跑"快得多，也准确得多。

显存优化没有银弹，只有把每一笔账算清楚后的工程组合。希望这 5 篇能让你多算清几笔。

---

## 参考资料

- [TRL SFT Trainer 文档](https://huggingface.co/docs/trl/sft_trainer)
- [TRL Reducing Memory Usage 文档](https://huggingface.co/docs/trl/reducing_memory_usage)
- [Efficient LLM Pretraining: Packed Sequences and Masked Attention](https://huggingface.co/blog/sirluk/llm-sequence-packing)
- *Fewer Truncations Improve Language Modeling*. 2024. [arXiv:2404.10830](https://arxiv.org/abs/2404.10830)
- [Liger Kernel](https://github.com/linkedin/Liger-Kernel)
- Hu et al. *LoRA: Low-Rank Adaptation of Large Language Models*. ICLR 2022. [arXiv:2106.09685](https://arxiv.org/abs/2106.09685)
- Dettmers et al. *QLoRA: Efficient Finetuning of Quantized LLMs*. NeurIPS 2023. [arXiv:2305.14314](https://arxiv.org/abs/2305.14314)
- [Transformers Performance Guide](https://huggingface.co/docs/transformers/perf_train_gpu_one)
