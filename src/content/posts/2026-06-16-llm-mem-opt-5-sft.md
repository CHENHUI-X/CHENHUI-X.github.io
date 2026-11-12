---
title: 大模型显存（三）：微调时怎样节省显存
published: 2026-06-16
description: 从 TRL SFTTrainer 的数据和损失计算讲起，说明 LoRA、QLoRA、Packing、分块损失、激活卸载和重算怎样节省显存，以及各自的配置限制。
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

<span id="sft-训练实战数据处理与显存优化"></span>

这是显存系列的第三篇。[上一篇 ZeRO](/posts/2026-06-15-llm-mem-opt-4-zero/)介绍了多张卡怎样分摊训练状态；这一篇回到微调：从检查数据、标签和 loss 开始，跑通一次 SFT，再根据当前的显存瓶颈选择优化方法。

SFT 是用示范答案做监督微调，仍然需要前向、反向和参数更新，但不要求先学强化学习。我们既可以更新全部参数，也可以冻结基模型，只训练 LoRA 等少量新增参数。本文会分别说明这些选择怎样影响显存占用。

阅读前，先掌握[第一篇](/posts/2026-06-12-llm-mem-opt-1-fundamentals/)里的权重、梯度、优化器状态和激活。**只做单卡微调，不必先掌握 ZeRO 的通信细节**；需要多卡全参训练时再回看第二篇。后面的 Packing 和 Padding-Free 配置对 attention 后端有要求。FlashAttention 怎样减少注意力的中间存储，见[第五篇](/posts/2026-06-14-llm-mem-opt-3-flashattention/)；打包样本、减少 padding 本身能省下什么，以及怎样检查样本边界，仍以下文为准。

> **适用版本**：以下默认值、参数和兼容性以 **2026-10-06** 的 [TRL v1.14.1](https://github.com/huggingface/trl/releases/tag/v1.14.1) 为准。示例用于进行简短的训练测试，显存需求与组合兼容性需要在目标 GPU 上测试。Transformers、PEFT、bitsandbytes、PyTorch、CUDA 与 attention kernel 也需要记录版本，固定 TRL 并不等于锁定完整环境。

<span id="0-消费级单卡-sft-7b-的工程难题"></span>

<span id="0-单卡微调-7b-模型的显存需求"></span>

## 0. 单卡微调 7B，显存花在哪里

如果采用 [显存基础](/posts/2026-06-12-llm-mem-opt-1-fundamentals/) 中的一种混合精度 Adam 存储配置：低精度权重 2 byte、低精度梯度 2 byte、FP32 主权重 4 byte、FP32 一阶矩 $m$ 和二阶矩 $v$ 合计 8 byte，那么模型状态约为 **16 byte/参数**。7B 参数对应 $7\times10^9\times16=112\times10^9$ byte，即 **112 GB，约 104.3 GiB**，还没有加激活、临时张量与通信缓冲。这个数字只对应上述存储配置，换一种训练实现，需要重新计算。

单卡没有其他 GPU 来分摊状态，仅靠 ZeRO 分片无法减少单卡需要保存的状态总量；CPU/NVMe offload 是另一种取舍。单卡微调通常先考虑：

- **LoRA / QLoRA**：减少需要训练的参数，以及冻结基模型的权重占用
- **Truncation / Packing / Padding-Free**：控制序列长度，减少无效 padding
- **Chunked NLL / Liger Kernel**：减少损失计算或其他算子的中间张量
- **Gradient Checkpointing / Activation Offloading**：少保存激活，或者把它们暂存到 CPU

这些方法省下的是不同部分。先查哪一项占得最多，再尝试组合；同一块显存不能重复节省，所以各方法的“省显存百分比”也不能直接相加。

<span id="1-sft-trainer-快速入门"></span>

## 1. 先用 SFTTrainer 跑通一次训练

<span id="11-一个小模型的起点"></span>

### 1.1 小模型训练示例

一次训练更新可以拆成四步：把一批文本变成 token 并做**前向**，得到下一 token 的预测；与示范标签比较得到 loss；做**反向传播**计算参数梯度；最后由 optimizer（优化器）用梯度更新可训练参数。模型权重跨步骤常驻，前向的部分中间结果则要暂存到反向结束，这些是训练激活。后文的优化分别减少这些数据的存储或调整其保存时机。了解各类数据何时被使用，有助于选择合适的方法。

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

`SFTTrainer` 可以处理 tokenizer、对话模板、截断和组 batch，但它不会替你判断“哪些文本应当成为训练答案”。

上例中 `train[:8]` 只取 8 条训练样本，`per_device_train_batch_size=1` 表示每张卡一次前向放一条；`max_steps=2` 限定两次优化器更新，不是训练完整数据两遍。`output_dir` 是日志和训练产物的目录，`report_to="none"` 关闭外部实验记录集成。这样的短训练称为 smoke test，用来检查程序是否能运行、数据是否能读取、梯度是否能正常计算；两步 loss 还不足以评估训练效果。

本版配置默认启用 BF16（未指定 FP16 时）和 gradient checkpointing；硬件或任务不适合时要显式调整。[配置源码](https://github.com/huggingface/trl/blob/v1.14.1/trl/trainer/base_config.py)

<span id="gradient-accumulation-example"></span>

#### 每次放两条，为什么也能完成八条样本的一次更新

每次前向和反向实际放入显卡的小批次称为 **microbatch**。假设每张卡一次处理 $b$ 条，累计 $A$ 次才更新一次参数，有 $D$ 张数据并行卡，且每次都装满，则一次更新覆盖的样本数 $B_{\mathrm{effective}}$ 为：

$$
B_{\mathrm{effective}}=bAD.
$$

单卡取 $b=2,A=4,D=1$，一次更新覆盖 8 条样本。执行顺序是“第 1 小批前向→反向→释放不再需要的激活”，再处理第 2、3、4 小批，最后只做一次 optimizer step。期间参数不更新，参数梯度保留并继续相加。这样不必同时保存 8 条样本的前向图，也不会保存 4 份独立 Adam 状态。

再用两条样本手算累积的是谁：令输入为 $x$，参数为 $w$，输出 $y=wx$，目标为零，每条损失 $L=(wx)^2/2$。在 $w=1$ 时，每次只处理一条，先取 $x=1$，再取 $x=3$；两条的参数梯度 $\partial L/\partial w=wx^2$ 分别为 1 和 9。记两条损失为 $L_1,L_2$，若目标是它们的平均值，累计的梯度应为 $(1+9)/2=5$。先分别反向 $L_1/2$、$L_2/2$ 再更新，和一次计算 $(L_1+L_2)/2$ 的梯度相同。若每条之后都更新 $w$，第二条使用的权重已经改变，就不是同一次大 batch 更新了。

累积主要允许用更小的 microbatch 换取更多轮前后向，不会让权重、参数梯度或 Adam 状态按 $A$ 等比例缩小。固定 microbatch、只把累积次数从 4 改为 8，通常也不会继续降低单个 microbatch 的激活峰值；后续小批前向还会与已存在的梯度缓冲重叠。单条序列都放不下时，不能靠增加累积次数解决。

上面的平均成立条件是每份损失的权重相同。语言模型常按有效目标 token 平均，长度不同时应汇总整个累积窗口的 token 损失，再除以总有效目标数，不能无条件对每个小批的均值再平均。最后不足一个完整窗口、不同设备的 token 数、同步方式也要按框架规则处理。第 1.3 节会具体解释哪些 token 参与监督。[Accelerate 梯度累积文档](https://huggingface.co/docs/accelerate/usage_guides/gradient_accumulation)给出了逐批反向、延迟更新及变长 token 的归一化示例。

这里的等价性还要求逐样本计算与同批计算对应同一个目标；含跨样本统计的操作、不同的随机行为和浮点累加顺序都可能使结果不同，不能把上述小模型的严格相等推广到所有训练配置。

<span id="12-数据格式不等于损失范围"></span>

<span id="12-数据格式与参与监督的-token"></span>

### 1.2 数据怎么写，哪些 token 计算 loss

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

同时使用 completion-only 和 assistant-only 时，参与 loss 的位置必须满足两者。使用模板前，应打印一条实际 tokenized 样本，确认问题、回答和结束符分别对应哪些 token；字符长度无法准确反映这些位置。[SFT 数据与损失选项](https://github.com/huggingface/trl/blob/v1.14.1/docs/source/sft_trainer.md)

<span id="13-shiftpadding-mask-和-response-mask-各管什么"></span>

<span id="13-标签对齐注意力掩码与损失掩码"></span>

### 1.3 标签怎样错位，三种 mask 分别管什么

这里 $\theta$ 表示模型参数，$p_\theta$ 是当前模型给下一 token 的条件概率，$\log$ 取自然对数。

令输入 token 为 $x_{b,0},\ldots,x_{b,S-1}$，batch 大小为 $B$、长度为 $S$。位置 $t$ 的输出预测 $x_{b,t+1}$。设 $m_{b,t+1}\in\{0,1\}$ 表示这个**目标位置**是否参与监督，则 token 平均损失为：

$$
\mathcal L_{\mathrm{SFT}}
=-\frac{1}{N_{\mathrm{valid}}}
\sum_{b=1}^{B}\sum_{t=0}^{S-2}
m_{b,t+1}\log p_\theta(x_{b,t+1}\mid x_{b,\le t}),
\qquad
N_{\mathrm{valid}}=\sum_{b=1}^{B}\sum_{t=0}^{S-2}m_{b,t+1}.
$$

这一定义要求 $N_{\mathrm{valid}}>0$。

损失中的 $p_\theta$ 是模型分给**正确目标 token**的概率。假设只监督两个位置，正确 token 的概率为 $0.5$、$0.25$，损失就是 $-(\log0.5+\log0.25)/2\approx1.040$。若 batch 中还有 10 个 padding，它们既不进入分子，也不进入分母；否则同一个答案只因补齐长度不同，就会得到不同权重。

分母不是 $B\times S$，也不是样本数。梯度累积或多卡聚合时，如果希望维持同一个 token 平均目标，应合计有效 token 的损失和与数量；不能不加权地平均有效长度不同的 microbatch loss。

具体看一条右侧 padding 的样本：

```text
input_ids:       [BOS, 问题, 回答1, 回答2, EOS, PAD]
labels:          [-100, -100, 回答1, 回答2, EOS, -100]
attention_mask:  [1,    1,    1,     1,     1,   0]
```

默认因果 LM loss 把 `logits[:, :-1, :]` 与 `labels[:, 1:]` 对齐。

BOS 表示序列开始，EOS 表示结束，PAD 只用于补齐；实际 token 名称由 tokenizer 决定。`labels` 是“希望被预测出来的答案”，不是另一次模型输入。比如「问题」所在位置的输出预测「回答1」，而「回答2」所在位置的输出预测 EOS，因此训练目标需要错开一格。这里有 3 个有效目标：回答1、回答2、EOS。损失函数已经完成了一次 shift，再手动移动 labels 会导致重复移位。

- **Causal mask**：禁止当前位置偷看未来 token
- **Padding attention mask**：避免真实 token 关注补齐的位置
- **Loss mask**：把无需监督的位置的 label 设为 `-100`，交叉熵忽略它们；prompt 仍留在输入中作为回答的上下文

`-100` 不是 padding token ID，也不会自动把该 token 从 attention 中删除。如果 PAD 和 EOS 共用 ID，应按实际 padding 位置屏蔽，不能把所有等于该 ID 的真实 EOS 都屏蔽。把 prompt 排除在 loss 之外，也不表示 Transformer 不再计算 prompt。

![SFT 的 logits 与下一 token 标签错位对齐](/images/llm-series/annotated/trl-sft-figure-zh.png)

*来源：[原始资料](https://huggingface.co/docs/trl/sft_trainer)；Hugging Face TRL（[Apache-2.0 许可](https://github.com/huggingface/trl/blob/main/LICENSE)）。本图添加中文辅助标注与引线；[查看原图](/images/llm-series/sourced/trl-sft-figure.png)。*

1. **输入 token**：上方蓝格是 input_ids，它们经过模型得到每个位置的输出。训练时仍保留 prompt 作为上下文，即使 prompt 不参与 loss。

2. **每列是词表打分**：橙色矩阵每列对应一个输入位置，每行对应一个词表候选，模型用这一列预测下一个 token。图中词表行数只是示意，不是模型实际词表大小。

3. **错开一格配对**：最右 logits 列和最左 labels 格被划去，余下位置错开一格配对。划线表示 shift 后没有配对的两端，不是 padding mask；不要在 loss 已完成 shift 后再手动移位。

4. **计算交叉熵**：配对后的 logits 与目标标签送入绿色交叉熵，再按有效目标 token 数归一化。prompt、padding 等 loss mask 仍要单独处理，不能仅靠图里的两处划线。

## 2. Truncation：控制单条样本的长度

<span id="21-为什么需要"></span>

<span id="21-长样本造成的-padding-开销"></span>

### 2.1 一条长样本为什么会增加整批开销

普通动态 padding 会把 **同一个 batch** 内的序列补到该 batch 最长长度。一条异常长样本会增加同批短样本的 padding 开销；补齐范围限于当前 batch，并非整个数据集。按长度分组、减小 batch 或截断都可能减轻浪费。

![截断长序列可减少批内补齐，但也会删除真实内容](/images/llm-series/annotated/trl-truncation-zh.png)

*来源：[原始资料](https://huggingface.co/docs/trl/reducing_memory_usage)；Hugging Face TRL（[Apache-2.0 许可](https://github.com/huggingface/trl/blob/main/LICENSE)）。本图添加中文辅助标注与引线；[查看原图](/images/llm-series/sourced/trl-why-you-should-truncate.png)。*

1. **同批最长样本**：上半图第五行最长，普通动态 padding 会让同批其他样本也补到这个长度。补齐只受当前 batch 影响，不是把整个数据集都补到全局最长。

2. **黄色是补齐**：蓝色为真实 token，黄色为 padding；原例上半图的 padding 占 52%。这些格子占输入槽位，却不应计入有效目标 token 的 loss 分母。

3. **截去右侧内容**：下半图统一截掉右侧带叉区域，使示例 padding 占比降到 20%。这两个百分比描述图中样本，不能直接当成显存或速度改善百分比。

4. **真实内容也会丢**：引线指向下半图第五行被叉掉的蓝色 token：截断不仅删除 padding，也会删掉真实文本。阈值应检查回答、代码结尾和 EOS 是否保留，不能只追求更低 padding 比例。

<span id="22-配置与取舍"></span>

### 2.2 设置长度上限，并检查有没有截掉答案

```python
from trl import SFTConfig

training_args = SFTConfig(max_length=2048)
```

本版默认保留开头、截去尾部。统计长度时，应以应用 chat template 后的 token 为准。P90/P95 可以帮助我们观察长度分布，但不能直接作为选择阈值的规则：长答案、代码结尾和 EOS 可能恰好落在被截掉的部分。

截断之后还要检查 **剩余的有效回答 token 数**。如果 prompt 已经占满窗口，这条样本可能只剩 `-100` 标签，无法提供训练信号。应缩短 prompt、提高长度上限、合理切分或过滤该样本。Packing 可以利用剩余空间，但不能恢复被截掉的内容。

## 3. Packing：把短样本放进同一个窗口

<span id="31-先看目的再看边界"></span>

<span id="31-窗口利用率与样本隔离"></span>

### 3.1 把窗口填满，但别让独立样本互相看见

Packing 把多条短序列安排到接近 `max_length` 的块中。例如长度为 900、700、400 的三条序列，放到 2048 的窗口里，总计 2000 token；相比把三条都补到 900，窗口中没有用上的位置更少。这个算例只比较 token 槽位，实际计算量还取决于 attention 实现。

![TRL 的七 token 窗口 packing 示例](/images/llm-series/annotated/trl-packing-zh.png)

*来源：[原始资料](https://huggingface.co/docs/trl/reducing_memory_usage)；Hugging Face TRL（[Apache-2.0 许可](https://github.com/huggingface/trl/blob/main/LICENSE)）。本图添加中文辅助标注与引线；[查看原图](/images/llm-series/sourced/trl-packing-3.png)。*

1. **每色一条样本**：顶部不同颜色代表不同原始样本，长度并不相同。原图窗口为 7 token，与正文 2048 的算例不同。

2. **灰色是空槽位**：左下不开 packing 时，短样本需要补灰格，超长青色样本仅保留开头七格。Packing 的直接目标是减少空槽位，实际计算收益仍取决于 attention 实现。

3. **长样本拆成多块**：右下把 15 个青色 token 分成 7、7、1，展示一种保留 token 的切分结果。并非所有策略都这么做，而且保留 token 不等于保留跨片段完整上下文。

4. **拼接仍需隔离**：右下黄色与橙色、粉色与蓝色共享窗口，颜色交界仍是独立样本边界。原图没有画 attention mask，必须由兼容后端隔离各段，并屏蔽跨段预测 loss；仅插入 EOS 不能实现隔离。

**拼接独立样本时，还需要隔离它们的注意力计算。** 普通 causal attention 会让后一个样本看到前一个样本；插入 EOS 本身并不阻止这种可见性。独立样本的 packing 需要注意力后端识别每段边界，也要屏蔽“上一段末尾预测下一段开头”的 loss。

<span id="32-本版策略"></span>

### 3.2 TRL v1.14.1 的打包策略

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

实际 [trainer 源码](https://github.com/huggingface/trl/blob/v1.14.1/trl/trainer/sft_trainer.py) 对 **`bfd` 和 `bfd_split` 都自动启用 Padding-Free**，而不是只对 `bfd`。Collator 是把若干样本整理成模型输入 batch 的函数；`seq_lengths` 记录各段长度，position IDs 是每个 token 在本段中的位置编号。Collator 利用段长度重置位置，并把段首 label 设为 `-100`。例如两段长 3、2，位置编号为 `[0,1,2,0,1]`，第 0、3 个位置的 label 都设为 `-100`，避免前一段的末尾被要求预测后一段的开头。兼容的变长 FlashAttention 路径再利用边界分开计算 attention；只重置 position IDs、却使用忽略这些边界的后端，仍可能串样本。

`wrapped` 不提供样本的自动隔离。把文档作为连续流训练时，可以采用这种方式，但训练目标也随之改变；独立问答则需要验证样本之间是否隔离。v1.14.1 也明确拒绝 packing 与 context parallelism（`cp_size > 1`）的组合，因为相应路径不能保留所需的分块 attention 边界。[打包实现](https://github.com/huggingface/trl/blob/v1.14.1/trl/data_utils.py)

## 4. PEFT（LoRA）：只训练少量新增参数

PEFT 是参数高效微调这一类方法，LoRA 是其中一种。对输入列向量 $x\in\mathbb R^{d_{\mathrm{in}}}$，设原线性层权重为 $W\in\mathbb R^{d_{\mathrm{out}}\times d_{\mathrm{in}}}$。标准 LoRA 冻结 $W$，学习一个低秩增量：

$$
W'=W+\frac{\alpha}{r}BA,\qquad
A\in\mathbb R^{r\times d_{\mathrm{in}}},\quad
B\in\mathbb R^{d_{\mathrm{out}}\times r}.
$$

这里 $r$ 是 rank（秩），$\alpha$ 对应 `lora_alpha`。

![LoRA 论文原图加中文标注：冻结蓝色主权重，输入经过橙色低秩 A、B 分支后相加](/images/llm-series/annotated/mt-lora-zh.png)

*来源：Edward J. Hu 等，[LoRA，Figure 1](https://arxiv.org/html/2106.09685v2#S1.F1)；中文为辅助标注；[原始 SVG](/images/llm-series/sourced/mt-lora-original.svg)。版权归论文作者；arXiv 页面标注非独占传播许可，不代表图片另获开放再许可。*

1. **从底部输入沿两条路向上读**：左路算 $Wx$，右路先算 $Ax$、再算 $B(Ax)$，最后相加得到输出。图省略缩放系数，本文公式中的 $\alpha/r$ 仍需保留。
2. **窄处的 $r$ 才是低秩瓶颈**：原图用 $d\times d$ 方阵举例；本文公式允许输入、输出维度不同。蓝色矩阵仍要执行计算并占存储，只是不更新它的参数。
3. **$B=0$ 是初始化，不是训练全程等于零**：图中 $A$ 的正态分布也是论文的初始化示意；PEFT 的默认实现通常对 $A$ 用 Kaiming-uniform、对 $B$ 用零初始化。两者都让初始增量 $BA$ 为零，训练后 $B$ 会变化。

低秩具体限制了什么？增量先用 $A$ 把输入投到 $r$ 维，再用 $B$ 映射回输出维度，因此 $BA$ 最多只有 $r$ 个独立变化方向。比如原矩阵是 $4\times4$，取 $r=1$ 时，$A$ 只有一行、$B$ 只有一列，共 8 个参数；它们的外积产生 16 个增量，但这些增量彼此受约束，不是 16 个可以独立调整的数。增大 $r$ 增加表达空间，也增加训练状态。

`r=16, lora_alpha=32` 时缩放为 2。默认标准 LoRA 的系数是 **$\alpha/r$**；rsLoRA 等变体使用不同缩放，不能混写。训练时可计算 $Wx+(\alpha/r)B(Ax)$，无须为每一步显式构造完整的 $W'$。[LoRA 论文](https://arxiv.org/abs/2106.09685)

为什么初始化时只把 $B$ 设为零？这样既让初始增量为零、不改变基模型的输出，又能让低秩分支开始学习。设 $c=\alpha/r$，$G=\partial L/\partial W'$，由链式法则可得：

$$
\frac{\partial L}{\partial A}=cB^\top G,\qquad
\frac{\partial L}{\partial B}=cGA^\top.
$$

初始 $B=0$ 时，这条路径给 $A$ 的损失梯度为零；随机初始化的 $A$ 则可以让 $B$ 得到非零梯度。$B$ 更新后，后续步骤也可以给 $A$ 非零梯度。如果 $A$ 和 $B$ 同时初始化为零，两者通过这条路径得到的梯度都会是零。这里说的是损失梯度；参数是否还会受权重衰减等影响，要看实际优化器的设置。

用一维例子看一次更新：令缩放 $c=1$、冻结的 $W=0$、$a=2$、$b=0$，输入为 1、目标为 1，损失为输出误差平方的一半。初始输出为 0，$\partial L/\partial a=0$、$\partial L/\partial b=-2$。若使用学习率 0.1 的普通 SGD，更新后 $a$ 仍为 2、$b$ 变成 0.2，输出也随之变为 0.4；下一步 $a$ 就能收到非零损失梯度。这只是说明更新过程的玩具例子，实际矩阵的梯度还取决于输入和损失。

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

冻结基模型的意思是“不更新它的权重”，并不是把它从计算中删去。LoRA 的梯度仍要经过基模型的运算回传，所以仍需要其中一些激活；这也是 LoRA 常与 checkpointing 配合的原因。[第一篇的两层链式手算](/posts/2026-06-12-llm-mem-opt-1-fundamentals/#saved-activations-chain-rule)展示了冻结后层参数、仍向前层传梯度的例子，也解释了哪些情况下可以不记录一段运算。

梯度和优化器状态主要跟随可训练参数，但 `modules_to_save`、可训练 bias 或其他未冻结层会另外占用。LoRA 的学习率可以从 `1e-4` 这样的量级试起，再看验证集表现；可训练参数较少并不意味着一定要提高学习率。

<span id="41-qlora存储精度与计算精度要分开"></span>

<span id="41-qlora-的量化存储与浮点计算"></span>

### 4.1 QLoRA：基模型用 4-bit 存储，LoRA 用浮点数训练

QLoRA 在冻结的低比特基模型上训练 LoRA。**4-bit 描述的是基模型的量化存储，不是所有运算都用 4-bit。** 相关权重在需要时按计算路径反量化；前向和反向都可能需要这些权重。`bnb_4bit_compute_dtype=torch.bfloat16` 指定 4-bit 线性层的计算 dtype，不等于整个模型的参数、激活和优化器状态都变成 BF16。[bitsandbytes 文档](https://huggingface.co/docs/transformers/en/quantization/bitsandbytes)

- 基模型：NF4 权重编码为 4 bit/参数，另有量化常数、分组元数据、未量化层和工作区
- LoRA：仍是浮点可训练参数；实际 dtype 由加载、PEFT 和 trainer 处理决定，需要检查加载后的实际参数类型
- 优化器：为可训练参数保存状态，状态 dtype 由优化器实现决定
- 激活：不会因为权重是 NF4 就自动减少为四分之一

QLoRA 论文中的三个技巧：NF4 用适合近似正态权重分布的非均匀编码；Double Quantization 再量化量化常数，论文估计平均额外节省约 0.37 bit/参数；Paged Optimizers 利用 unified memory 迁移分页状态以应对显存压力。这些方法各自减少一部分显存压力，但不能保证所有配置都避免 OOM；分页优化器也无法在 OOM 后自动恢复整个训练。论文报告的是 **单张 48 GB GPU 微调 65B 模型**，不是任意 24 GB 卡上的通用结论。[^qlora]

量化原理见 [量化：基础与进阶推导](/posts/2026-05-16-llm-quantization-guide/)，完整的起步配置放在第 12 节。

[^qlora]: Dettmers et al. *QLoRA: Efficient Finetuning of Quantized LLMs*. NeurIPS 2023. [arXiv:2305.14314](https://arxiv.org/abs/2305.14314)

## 5. Liger Kernel：减少算子中间张量

[Liger Kernel](https://github.com/linkedin/Liger-Kernel) 提供面向 LLM 训练的 Triton kernels，例如融合线性投影与交叉熵、RMSNorm、RoPE、SwiGLU 等。这些 kernel 会把多个计算步骤合在一起，让中间结果尽量留在片上存储，减少写回 GPU 显存的次数，也少保存一些临时张量。

```python
training_args = SFTConfig(use_liger_kernel=True, loss_type="nll")
```

模型支持、实际替换的算子和库版本共同决定收益。官方材料里的吞吐或显存百分比来自特定 benchmark，不能保证每个配置都有相同收益。本版 `use_liger_kernel=True` 与 `loss_type="chunked_nll"` 不能同时启用；没有显式设置 loss 时，Liger 路径默认选择 `nll`。

<span id="6-chunked-nll不要一次生成所有位置的-logits"></span>

## 6. Chunked NLL：分块计算 logits 与损失

### 6.1 大词表为什么占显存

Logits 是模型为每个候选 token 输出的分数，普通 LM head 的输出形状为 `[B, S, V]`，$V$ 是词表大小。以一个近似的大词表规模为例：

$$
B=4,\ S=2048,\ V=152000,\qquad
M_{\mathrm{logits}}=BSV\times2
=2{,}490{,}368{,}000\ \mathrm{byte}.
$$

若 logits 以 BF16 保存，这约为 **2.49 GB / 2.32 GiB**。这只是一个张量；交叉熵可能将部分计算提升到 FP32，反向也需要工作区，因此既不能把它当训练总显存，也不能固定乘二作为真实峰值。不同模型的实际 $V$ 应读配置，而不是用品牌名猜。

<span id="62-分块做同一件事"></span>

<span id="62-分块计算与损失归一化"></span>

### 6.2 每次只算一块 logits，再合并损失

NLL 是负对数似然；对于这里的 token 分类，它与标准交叉熵是同一个目标。每个目标 token 需要正确词的 logit 和整个词表的归一化项，但不同 token 的损失可以分批算：

1. 先正确 shift hidden states 与 labels，再识别 `labels != -100` 的目标位置
2. 只为有效目标所在的分块做 LM head 投影，避免大量无监督位置生成完整词表 logits
3. 累加各块 loss 的和，再除以总有效 token 数，而不是平均每块的均值
4. 用 checkpoint 在反向重算块内结果，避免 autograd 把每块大张量全部保留到最后

若每块有 $C$ 个 token，**logits 这一项**的峰值量级由 $BSV$ 降为 $CV$，不是整个训练显存都变为 $CV$。在本版实现中，块大小是 256；有效位置会移到前部，最后一块可能夹带被忽略的尾部位置，因此“每一个 `-100` 都绝不会参与任何 matmul”也不够精确。[实现与归一化](https://github.com/huggingface/trl/blob/v1.14.1/trl/trainer/sft_trainer.py)

```python
training_args = SFTConfig(loss_type="chunked_nll", use_liger_kernel=False)
```

<span id="63-兼容性文档与代码存在分歧"></span>

### 6.3 哪些模型能用：这一版文档和代码有出入

v1.14.1 的 [显存优化指南](https://github.com/huggingface/trl/blob/v1.14.1/docs/source/reducing_memory_usage.md) 仍把 PEFT 和 VLM 列为不兼容，但同版本源码包含 PEFT/VLM 处理路径，[官方测试](https://github.com/huggingface/trl/blob/v1.14.1/tests/test_sft_trainer.py) 也检查了部分模型的 loss 和梯度一致性。使用时需要按模型结构判断支持范围，不能将任一描述推广到所有 PEFT/VLM 组合。

这一版本可以确认的支持条件与限制如下：

- 与 `use_liger_kernel=True` 互斥
- PEFT 路径会对内部基模型打补丁；**若输出 `lm_head` 本身被 LoRA 等 tuner 包装，trainer 会拒绝该配置**，因为直接读取 head 权重会漏掉 adapter 增量
- 对有受支持输出 head 结构的部分 VLM 有实现与测试，不代表任意视觉模型都兼容
- 量化、模型结构、adapter 目标层和并行后端的完整组合仍需要小步测试

本篇 QLoRA 起步示例显式使用 `loss_type="nll"`。若要切成 chunked，先比较同一批数据的 loss 和可训练参数梯度，再测显存。本版官方指南报告 Qwen3-1.7B 的特定测试中单 GPU 峰值约降 30%、FSDP2 × 4 最高约降 50%；这些数字只适用于该指南所列的测试条件。

<span id="7-padding-free减少组-batch-时的补齐"></span>

## 7. Padding-Free：减少批内 padding

Packing 决定样本如何装入窗口；Padding-Free 决定 batch 如何送进模型。后者把不同长度的序列展平，并通过边界信息让兼容的 attention 后端分别处理，避免每条都补到 batch 最长。已截断的内容无法由此恢复，样本间的隔离仍需正确设置。

![TRL 的 padding-free 批内展平示例](/images/llm-series/annotated/trl-padding-free-zh.png)

*来源：[原始资料](https://huggingface.co/docs/trl/reducing_memory_usage)；Hugging Face TRL（[Apache-2.0 许可](https://github.com/huggingface/trl/blob/main/LICENSE)）。本图添加中文辅助标注与引线；[查看原图](/images/llm-series/sourced/trl-padding-free.png)。*

1. **每批两条样本**：中部和底部都是四个 batch，每批两条原始样本，batch_size 仍为 2。展平改变输入布局，不会把两条样本变成同一段训练上下文。

2. **短样本补灰格**：中部每批短样本补到本批最长，第三批的青色长样本使蓝色短样本产生大量灰格。Padding-Free 主要去掉这类逐条补齐。

3. **有效 token 展平**：下方把每批有效 token 连排，四批分别为 12、13、19、14 个 token。若另外设置 pad_to_multiple_of，展平后的总长度仍可能需要少量对齐 padding，不能据图声称任何配置都完全没有 padding。

4. **保留样本边界**：颜色交界仍是样本边界，需要正确的段长度、位置编号和支持分段边界的 attention 后端。展平本身不提供隔离，也不能恢复此前截断的内容。

```python
training_args = SFTConfig(
    packing=False,
    padding_free=True,
    max_length=None,
    model_init_kwargs={"attn_implementation": "flash_attention_2"},
)
```

这个片段假设已事先检查、过滤或切分超长输入。在本版中，`padding_free=True` 且未 packing 时不能保留默认的非空 `max_length`，否则 trainer 报错：该路径不负责按这个上限截断。

使用支持分段边界的 FlashAttention 路径是正确性的条件，不能把警告理解成“换个慢一点的后端也一样”。另外，若设置 `pad_to_multiple_of`，展平后的总长度仍可能为对齐补少量 padding；图中展示的是不再逐条补到最长的理想情况。batch 只有一条且未 packing 时，本就没有跨样本长度差导致的补齐，通常不会得到相同收益。

不同 trainer 的支持情况也不同；本版指南说明 DPO 的 Padding-Free 暂时会回退到普通 padding。

## 8. Activation Offloading：把部分激活暂存到 CPU

前向时把原本为反向保存的部分 tensor 搬到 CPU，需要时再取回。这减少 GPU 上的激活驻留，但会增加主机内存需求和 CPU↔GPU 传输。

```python
training_args = SFTConfig(activation_offloading=True)
```

其实现利用 PyTorch 的 saved-tensor hooks 管理张量保存与恢复。传输能否和计算重叠，取决于 tensor 大小、带宽、CPU 内存及实现。不同配置下，传输带来的速度损失差别可能很大，需要实测。它不自动卸载基模型权重或优化器状态，因此只能缓解与激活存储有关的部分显存压力。[PyTorch saved-tensor hooks](https://pytorch.org/tutorials/intermediate/autograd_saved_tensors_hooks_tutorial.html)

<span id="9-gradient-checkpointing用重算减少保存"></span>

## 9. Gradient Checkpointing：反向时重算激活

前向时只保留选定检查点，反向需要某段中间结果时再算一遍。

例如一段计算是 $h_0\to h_1\to h_2\to h_3$。普通反向可能要保留中间的 $h_1,h_2$；checkpoint 方案保存这段的输入 $h_0$，反向到来时从它重新算出需要的中间值。这里的 checkpoint 是**激活重算边界**，不是存到磁盘上的模型存档，也不会减少应计算的参数梯度。

PyTorch 的示意图将“保存中间值”和“保存输入后重算”并排比较：

![PyTorch 激活检查点原图加中文说明：左侧保存三个中间张量，右侧保存区域输入并在反向重算](/images/llm-series/annotated/mt-activation-checkpoint-zh.png)

*来源：PyTorch Foundation，[Current and New Activation Checkpointing Techniques](https://pytorch.org/blog/activation-checkpointing-techniques/)；中文为辅助标注；[查看原图](/images/llm-series/sourced/mt-activation-checkpoint-original.png)。图片版权归原作者及相关权利人，网页未单独标明图片许可。*

1. **左侧黑色跨线是保存关系**：橙色方块是操作，跨到反向一侧的线表示把张量留下来供反向使用。图中标出了 3 个保存张量。
2. **右侧黑框是检查点区域**：先留住区域输入，反向执行到这里时重算框内操作，得到需要的中间值。图中“1 tensor”指这个局部例子，不代表整个模型只留一个张量。
3. **3 变 1 不等于显存固定降到三分之一**：张量形状、框外保存项、重算临时工作区都影响峰值；换来的额外计算也要实测。

这种方法主要减少保存的激活，不消除权重、梯度、优化器状态，也未必能解决 checkpoint 区域之外的巨大 logits 分配。

```python
training_args = SFTConfig(gradient_checkpointing=True)
```

本版默认启用。收益与层划分、序列长度、kernel、是否同时分块计算 loss 有关；不能笼统保证从 $O(L)$ 降到 $O(\sqrt L)$，也不能固定写“慢 20%”或“增加 33% 计算”。[经典分段分析](https://arxiv.org/abs/1604.06174) 中的 $L$ 指层数，且结论有特定保存策略的前提。

[重算必须与原前向一致](https://docs.pytorch.org/docs/2.8/checkpoint.html)：会变化的全局状态、随机数处理、原地修改和缓存都可能影响结果。训练时通常关闭推理用的 KV cache；PEFT 的输入梯度需求、checkpoint 的 reentrant 模式与 ZeRO-3 也有组合限制。本版 trainer 对 PEFT + ZeRO-3 有专门处理，其他环境使用的 `use_reentrant=False` 配置在这里未必适用。先跑几步，确认所有需要训练的参数都获得了正确梯度，再测显存收益。

<span id="10-其他工程开关与选读支线"></span>

## 10. 长度对齐与在线 RL 的显存配置

<span id="101-pad_to_multiple_of对齐不是越大越好"></span>

<span id="101-pad_to_multiple_of-的对齐开销"></span>

### 10.1 pad_to_multiple_of 会多补多少 token

```python
training_args = SFTConfig(pad_to_multiple_of=8)
```

对齐可能帮助某些 kernel，但也会增加 token 槽位。例如长度 513 对齐到 8 的倍数是 520；对齐到 2048 就会补到 2048。应比较实际吞吐，不能默认选 2048，更不能说所有 Tensor Core 都因此显著加速。

<span id="102-在线-rl生成与训练争抢显存"></span>

### 10.2 在线 RL 中的生成与训练显存

只做 SFT 可以跳过这一节。在线 RL 会在训练过程中生成回答，再按奖励更新模型；GRPO 是其中一种方法。这里只说明生成和训练共用显卡时，两个配置选项为什么会冲突。需要算法背景可选读 [Deep Reinforcement Learning](/posts/2024-04-30-deep-reinforcement-learning/)。

若使用 ZeRO-3 和普通 Transformers 生成，生成阶段收集完整权重可能制造峰值。可以考虑：

```python
from trl import GRPOConfig

training_args = GRPOConfig(
    use_vllm=False,
    ds3_gather_for_generation=False,
)
```

这一方案不使用 vLLM，生成速度会受到影响。官方 [GRPOConfig](https://github.com/huggingface/trl/blob/v1.14.1/trl/trainer/grpo_config.py) 明确说明：**关闭 `ds3_gather_for_generation` 不兼容 vLLM 生成**。因此，这两个选项不能同时用于该生成流程。

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

<span id="12-从能验证的小配置开始"></span>

## 12. 单卡 QLoRA 配置与验证步骤

### 12.1 单卡 QLoRA 起步示例

首次运行时，先用少量样本、短上下文与普通 loss 检查数据和训练，再逐项尝试 packing、外部 kernel 和 offloading，便于定位问题。以下示例需要支持 BF16 和相应 bitsandbytes 后端的 GPU：

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

这里采用对话语言建模的默认监督范围，并非 assistant-only。

本例的梯度累积配置对应每卡一次 1 条、累计 4 次、单卡运行，即一次更新覆盖 4 条样本。累积的时间顺序、梯度手算与有效 token 归一化限制见[第 1.1 节](#gradient-accumulation-example)。

想只训练助手回答，应先验证模板能产生正确 mask，再加 `assistant_only_loss=True`。本例仅在 Q/V 投影上挂 LoRA，便于起步；目标层选择会影响能力与参数量，不代表复现了 QLoRA 论文的全部配置。

本版 `SFTTrainer` 在 `model` 为字符串时接受独立的 `quantization_config` 参数；如果先自行加载了模型对象，必须在模型加载阶段完成量化，不能指望这个 trainer 参数再量化一次。NF4 + LoRA 才使该示例成为 QLoRA；仅传一个 `LoraConfig` 不是 QLoRA。

<span id="122-每次只改一类变量"></span>

<span id="122-数据梯度与显存的逐项验证"></span>

### 12.2 先检查数据和梯度，再测显存

1. 解码几条实际输入，检查 prompt、回答和 EOS；确认 shift 后有效 label 数大于零
2. 先训练几步，检查 loss 为有限数、应训练参数获得梯度、保存/加载的 adapter 正确
3. 记录峰值 allocated/reserved 显存与有效 token/s；CUDA 测量要先同步，并包含优化器状态建立后的训练步
4. 再逐项尝试增大长度、packing、chunked loss 或 Liger；更换 loss 实现时比较 loss 与梯度，更换 packing 时检查边界
5. 用独立验证集确认质量没有因截断、mask 或跨样本信息泄漏变差

需要多卡全参微调时，可回看[第二篇 ZeRO](/posts/2026-06-15-llm-mem-opt-4-zero/)，了解分片能省下多少模型状态；完整启动流程仍需按训练框架文档配置。模型是 70B，并不意味着任意“ZeRO-3 + packing”组合就足够；还要算临时收集、激活与通信峰值。

<a id="13-系列总结"></a>

<span id="13-从这篇走向下一步"></span>

<span id="13-小结与后续训练方法"></span>

## 13. 小结：先保证训练正确，再减少显存

做单卡 SFT，先解码实际输入、检查标签和 mask，确认模型学的是预期答案。然后测出哪一项最占显存，每次加一种优化，比较显存峰值、训练速度和验证集效果。错误的 mask 或串样本的 attention 也可能跑出平滑下降的 loss，因此不能只看训练是否“能跑”。

可以先用第 12 节的小配置验证自己的数据和模型，再一次只加一种优化。需要多卡分摊全参模型状态时，回看[第二篇 ZeRO](/posts/2026-06-15-llm-mem-opt-4-zero/)；单卡微调可以继续沿着本篇的数据和显存检查流程进行。

微调完成后，模型回答问题时又会怎样占用显存？下一篇[《大模型显存（四）：从 Prefill、Decode 到 KV Cache》](/posts/2026-06-13-llm-mem-opt-2-inference/)会转到推理过程：哪些训练状态不再需要，哪些缓存会随上下文增长，以及输入和生成阶段的峰值为什么不同。想继续研究偏好对齐，也可选读 [RLHF、PPO 与 DPO](/posts/2025-12-05-a-series-on-llm-training-i/)。

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
