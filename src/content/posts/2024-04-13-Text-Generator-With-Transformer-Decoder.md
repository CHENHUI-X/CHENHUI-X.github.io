---
title: Transformer Decoder：从零实现文本生成
published: 2024-04-13
description: 利用 Transformer Decoder 从零实现一个简单的文本生成器，涵盖数据构造、Mask Attention、Positional
  Encoding 等核心组件的代码实现。
category: Deep Learning
tags:
- deep learning
- nlp
- transformer
draft: false
---
## 0. 前言

本文从零实现一个小型字符级 text generator，重点是把三件事接起来：**怎样构造“预测下一个 token”的训练样本，怎样用 mask 防止偷看答案，以及怎样把一次预测接成持续生成。**

这里的模型是 decoder-only：每层包含 causal self-attention 和逐位置的前馈网络，不使用原始 encoder-decoder Transformer 中的 cross-attention。训练语料很小，适合理解计算过程，不能据此期待通用语言能力。

:::note
需要一些 Python、PyTorch、张量和神经网络基础。下面先解释 Q/K/V，再解释 mask；暂时不熟悉 RoPE 也不影响阅读。Tokenization 的背景可先看[文字如何变成 token](/posts/2026-05-16-llm-tokenization-guide/)。
:::

## 1. 准备：先看完整流程

- 数据：文字 → token ID → 输入序列和向右错开一位的标签
- 模型：token embedding + 位置信息 → 多层 Decoder → vocabulary 上的 logits
- 训练：每个有效位置预测下一个 token，PAD 位置不计入 loss
- 生成：选出一个新 token，接到输入末尾，再预测下一步

训练时，后面的真实 token 已经在数据里，所以可以并行计算多个位置的预测；causal mask 保证某一位置看不到自己的答案。生成时，后面的 token 还不存在，必须逐个选出并接回输入。

## 2. 数据构造

### 2.1 字符、token ID 和 embedding

计算机需要数值输入。最简单的办法，是把每个支持的字符对应到一个整数 ID。下面这张图只用于说明“符号和编号一一对应”，不是后续代码的词表：

![字符映射到整数 ID 的示意](https://s2.loli.net/2024/04/14/6nshFYQyPIBq8N1.png)

本文使用字符级 tokenizer。词表包含 26 个小写英文字母、10 个数字、空格、逗号、句号，再加两个特殊 token：

- `<pad>`：只负责补齐长度，不代表真实文字，也不作为正常的预测目标
- `<bos>`：表示序列开始，让空提示词也有一个有效位置来预测首字符

因此 vocabulary 大小是 **26 + 10 + 3 + 2 = 41**。`<bos>` 是一个特殊 ID，不能把字符串 `"<bos>"` 当成 5 个普通字符逐字编码。这里保留旧版普通字符的编号，单独在末尾增加 BOS：PAD 为 0，`a` 为 11，`c` 为 13，空格为 39，BOS 为 40。

```python
class Tokenizer:
    def __init__(self):
        tokens = (
            ['<pad>']
            + list('0123456789')
            + list('abcdefghijklmnopqrstuvwxyz')
            + [',', '.', ' ', '<bos>']
        )
        self.dictionary = {token: i for i, token in enumerate(tokens)}
        self.reverse_dictionary = {i: token for token, i in self.dictionary.items()}
        self.pad_id = self.dictionary['<pad>']
        self.bos_id = self.dictionary['<bos>']

    def tokenize(self, text):
        unsupported = sorted(set(text) - set('0123456789abcdefghijklmnopqrstuvwxyz,. '))
        if unsupported:
            raise ValueError(f'Unsupported characters: {unsupported}')
        return [self.dictionary[c] for c in text]

    def character_to_token(self, character):
        return self.dictionary[character]

    def token_to_character(self, token):
        return self.reverse_dictionary[token]

    def size(self):
        return len(self.dictionary)
```

token ID 只是查表编号，不是带有语义距离的数值。模型还要用可学习的 embedding 表，把 ID 转成 $d_{\text{model}}$ 维向量。真实语言模型通常采用子词或字节级分词；这里的一个 token 恰好对应一个普通字符，不能推广成“token 总是一个字”。

### 2.2 输入和标签错开一位

语料由下面这些简单句子组成。它们用句号和空格连接成一个连续文本，因此模型也会学习句子之间的连接。这里不加入 EOS，生成长度稍后由程序限制。

假设输入长度 $L=3$，从 `cats` 中可以构造 `cat → ats`。这意味着：

- 看到 `c`，预测 `a`
- 看到 `ca`，预测 `t`
- 看到 `cat`，预测 `s`

这三个预测在训练时一起算；每个位置仍然只能使用它左边和自己的输入。只训练最后一个位置 `cat → s` 也是一种做法，但它与给三个位置都计算 loss 的训练目标并不完全相同。

为了从 BOS 开始学习，在语料前放 **$L-1$ 个 PAD + 1 个 BOS**，然后取所有长度为 $L+1$ 的窗口。每个窗口的前 $L$ 个 token 是输入，后 $L$ 个是原始标签。

```python
def tokenize_and_pad_training_data(max_sequence_length, tokenizer, training_data):
    if max_sequence_length < 1:
        raise ValueError('max_sequence_length must be positive')
    return (
        [tokenizer.pad_id] * (max_sequence_length - 1)
        + [tokenizer.bos_id]
        + tokenizer.tokenize(training_data)
    )


def create_training_sequences(max_sequence_length, tokenized_training_data):
    if max_sequence_length < 1:
        raise ValueError('max_sequence_length must be positive')
    # A window has L + 1 tokens. The last legal start index is N - L - 1.
    return [
        tokenized_training_data[i:i + max_sequence_length + 1]
        for i in range(len(tokenized_training_data) - max_sequence_length)
    ]


def split_inputs_and_labels(sequences, pad_id):
    inputs = [window[:-1] for window in sequences]
    # A PAD query must not be trained to predict BOS or any real character.
    labels = [
        [target if source != pad_id else pad_id
         for source, target in zip(window[:-1], window[1:])]
        for window in sequences
    ]
    return inputs, labels


tokenizer = Tokenizer()
max_sequence_length = 20
training_data = '. '.join([
    'cats rule the world',
    'dogs are the best',
    'elephants have long trunks',
    'monkeys like bananas',
    'pandas eat bamboo',
    'tigers are dangerous',
    'zebras have stripes',
    'lions are the kings of the savannah',
    'giraffes have long necks',
    'hippos are big and scary',
    'rhinos have horns',
    'penguins live in the arctic',
    'polar bears are white',
])
stream = tokenize_and_pad_training_data(max_sequence_length, tokenizer, training_data)
sequences = create_training_sequences(max_sequence_length, stream)
input_rows, label_rows = split_inputs_and_labels(sequences, tokenizer.pad_id)
```

辅助函数先定义、再调用，以上代码可按顺序执行。长度为 $N$ 的流，一共有 $\max(0,N-L)$ 个合法窗口；`range(N - L - 1)` 会漏掉最后一个。当前语料有多少普通字符，就会构造多少个窗口，最后一个字符也会出现在最后一个窗口的标签里。

### 2.3 把第一组和最后一组摊开看

用更短的 `cats` 和 $L=3$ 展示完整过程。符号 `PAD`、`BOS` 在表中各代表一个 token：

| 滑动窗口 | 输入 `x` | 原始标签 | 实际训练标签 `y` |
| --- | --- | --- | --- |
| `PAD PAD BOS c` | `PAD PAD BOS` | `PAD BOS c` | `PAD PAD c` |
| `PAD BOS c a` | `PAD BOS c` | `BOS c a` | `PAD c a` |
| `BOS c a t` | `BOS c a` | `c a t` | `c a t` |
| `c a t s` | `c a t` | `a t s` | `a t s` |

第一行的有效学习任务是 **BOS → c**，不是“全部 PAD → c”。原始标签里与 PAD 输入对齐的 BOS 也被忽略；否则模型仍会在无意义的 PAD 位置接受监督。这里把需忽略的标签写成 PAD，稍后通过 `ignore_index` 跳过。

右侧 padding 也遵守同样原则。例如输入 `BOS c a PAD PAD` 可以配标签 `c a t PAD PAD`。padding 的方向可以不同，但输入、标签和有效位置必须始终对齐。

## 3. Mask Attention

### 3.1 先把 Q、K、V 和维度讲清楚

设一层接收的隐藏状态为 $X\in\mathbb{R}^{B\times T\times d_{\text{model}}}$，其中 $B$ 是 batch size，$T$ 是当前输入长度。对每个位置做三组可学习的线性变换：

$$
Q=XW_Q,\qquad K=XW_K,\qquad V=XW_V.
$$

可以把 Q 理解成“这个位置在寻找什么”，K 是“这个位置用什么特征供别人匹配”，V 是“匹配后实际汇总的信息”。这只是直觉；三者都是训练得到的向量，不是固定的词性或语义标签。

采用 $H$ 个 attention head，并令 $d_h=d_{\text{model}}/H$。本文令每个 head 的 Q、K、V 维度都为 $d_h$，拆分后形状如下：

| 张量 | 形状 | 含义 |
| --- | --- | --- |
| Q、K、V | $B\times H\times T\times d_h$ | 每个 head 的查询、匹配和内容向量 |
| $S=QK^\mathsf{T}/\sqrt{d_h}$ | $B\times H\times T\times T$ | softmax 前的 attention scores |
| $A=\operatorname{softmax}(S+M)$ | $B\times H\times T\times T$ | 沿最后一个维度归一化的权重 |
| $AV$ | $B\times H\times T\times d_h$ | 加权汇总的结果 |

矩阵的**行是 query 位置，列是 key 位置**。例如第 2 行第 1 列表示“位置 2 从位置 1 取多少信息”。这里的转置只交换 K 最后两个维度。多头结果合并后回到 $B\times T\times d_{\text{model}}$，再经过输出投影。[Transformer 原论文](https://arxiv.org/html/1706.03762v7#S3.SS2)给出了对应公式。

位置信息也要加入模型。本文后面的最小实现使用可学习的 position embedding，加到 token embedding 上；RoPE 则是在得到 Q/K 后旋转向量，再计算它们的内积。先理解这里 Q/K 的来源，再读 [RoPE 位置编码](/posts/2026-05-16-llm-rope-rotary-position-embedding/)会更容易。

### 3.2 两类 mask 各管什么

先只考虑有效 query：

1. **Causal mask**：位置 $i$ 只能看到 $j\le i$ 的 key，包括自己。当前输入 token 已知，而它要预测的是下一个 token，所以允许看自己不会泄漏答案
2. **Key padding mask**：不能把 PAD 位置当作有效 key。它负责遮列，不会自动删除 PAD query 对应的行

本文用 `1 / True` 表示允许 attention，`0 / False` 表示禁止。无 padding 时，$T=4$ 的 causal allow-mask 是：

| query \ key | 0 | 1 | 2 | 3 |
| --- | --- | --- | --- | --- |
| 0 | 1 | 0 | 0 | 0 |
| 1 | 1 | 1 | 0 | 0 |
| 2 | 1 | 1 | 1 | 0 |
| 3 | 1 | 1 | 1 | 1 |

<!-- concept-figure:causal-mask -->

![Jay Alammar 的因果 attention mask 原图：左侧为四个位置的 softmax 前分数，右侧将主对角线上方的未来位置设为负无穷](/images/llm-series/sourced/jay-alammar-attention-mask.png)

*图：在 softmax 之前，把未来位置的 score 设为 −∞，保留当前及更早位置的分数。原图：Jay Alammar，[The Illustrated GPT-2](https://jalammar.github.io/illustrated-gpt2/)，[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)，未修改。 [查看原尺寸图](/images/llm-series/sourced/jay-alammar-attention-mask.png)。*

<!-- /concept-figure:causal-mask -->

如果使用加性 mask $M$，允许处加 0，禁止处加 $-\infty$，然后再做 softmax。**不能把禁止位置的 score 改成 0 就算屏蔽**：$e^0=1$，它仍可能拿到正的 attention 权重。应当变成 0 的是 softmax 后的禁止位置权重。

:::warning
不同 API 对布尔 mask 的约定相反。PyTorch 的 [scaled_dot_product_attention](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html) 使用 `True = 允许`；[MultiheadAttention](https://docs.pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html) 的布尔 `attn_mask` 和 `key_padding_mask` 使用 `True = 屏蔽`。移植代码时必须检查约定，不能直接照搬同一个布尔矩阵。
:::

### 3.3 PAD query 不能只靠遮 key 处理

输入 `PAD PAD BOS c a` 时，如果只把 causal mask 与非 PAD key 取交集，前两行找不到任何可见 key。一整行 score 都是 $-\infty$，直接 softmax 可能产生 NaN；不能依赖所有 attention 后端都自动处理这种行。

处理方式如下：

- 有效 query：只允许因果范围内的非 PAD key
- PAD query：**整行改为只允许自身位置**，为数值计算保留一个有限 score
- attention 计算后：把 PAD query 的输出清零；它的标签也不计入 loss

于是左 padding 的最终 allow-mask 是：

| query \ key | PAD₀ | PAD₁ | BOS | c | a |
| --- | --- | --- | --- | --- | --- |
| PAD₀ | 1 | 0 | 0 | 0 | 0 |
| PAD₁ | 0 | 1 | 0 | 0 | 0 |
| BOS | 0 | 0 | 1 | 0 | 0 |
| c | 0 | 0 | 1 | 1 | 0 |
| a | 0 | 0 | 1 | 1 | 1 |

PAD 的自连接仅为避免全屏蔽行，不表示它有语言意义。有效 query 仍然看不到任何 PAD key。

右 padding 同样要**替换整行**，不能只在原来的 mask 上补一个对角线。否则右侧 PAD query 仍会看到之前的真实 token。输入 `BOS c a PAD₃ PAD₄` 的最终 mask 应是：

| query \ key | BOS | c | a | PAD₃ | PAD₄ |
| --- | --- | --- | --- | --- | --- |
| BOS | 1 | 0 | 0 | 0 | 0 |
| c | 1 | 1 | 0 | 0 | 0 |
| a | 1 | 1 | 1 | 0 | 0 |
| PAD₃ | 0 | 0 | 0 | 1 | 0 |
| PAD₄ | 0 | 0 | 0 | 0 | 1 |

下面直接构造这一规则。输入为 `(B, T)`，返回 `(B, 1, T, T)` 的 mask，沿 head 维广播：

```python
import torch
from torch import nn
from torch.nn import functional as F


def make_safe_allow_mask(token_ids, pad_id):
    valid = token_ids.ne(pad_id)                  # (B, T)
    length = token_ids.shape[1]
    causal = torch.ones(
        length, length, dtype=torch.bool, device=token_ids.device
    ).tril()
    diagonal = torch.eye(length, dtype=torch.bool, device=token_ids.device)
    real_rows = valid[:, :, None] & valid[:, None, :] & causal[None, :, :]
    pad_rows = (~valid[:, :, None]) & diagonal[None, :, :]
    allowed = real_rows | pad_rows
    return allowed[:, None, :, :], valid
```

这段逻辑还保证：非空序列的每一行都至少有一个允许位置，全 PAD 序列也不例外。不过全 PAD 样本没有训练价值，仍应从数据中排除。上述安全性针对由 mask 导致的全屏蔽行；Q/K/V 本身若已有 NaN 或溢出，需要另行排查。

### 3.4 把 attention 放回 Decoder

下面用一个小型 pre-norm Decoder 把前面串起来。`d_model` 必须能被 `num_heads` 整除；代码不使用 KV Cache，每次对整个窗口重新计算。

```python
class DecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError('d_model must be divisible by num_heads')
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.norm1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, hidden, allowed, valid):
        batch, length, width = hidden.shape
        qkv = self.qkv(self.norm1(hidden))
        qkv = qkv.reshape(batch, length, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        attended = F.scaled_dot_product_attention(
            q, k, v, attn_mask=allowed, dropout_p=0.0, is_causal=False
        )
        attended = attended.transpose(1, 2).reshape(batch, length, width)
        # Clear after the output projection too, since the projection has a bias.
        attended = self.out(attended).masked_fill(~valid[:, :, None], 0.0)
        hidden = hidden + attended
        hidden = hidden + self.ffn(self.norm2(hidden))
        return hidden.masked_fill(~valid[:, :, None], 0.0)


class TinyDecoder(nn.Module):
    def __init__(self, vocab_size, pad_id, max_length, d_model=64,
                 num_heads=4, num_layers=2):
        super().__init__()
        self.pad_id = pad_id
        self.max_length = max_length
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.position_embedding = nn.Embedding(max_length, d_model)
        self.blocks = nn.ModuleList([
            DecoderBlock(d_model, num_heads) for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, token_ids):
        if token_ids.ndim != 2 or not 1 <= token_ids.shape[1] <= self.max_length:
            raise ValueError('Expected (B, T), with 1 <= T <= max_length')
        allowed, valid = make_safe_allow_mask(token_ids, self.pad_id)
        # Count real tokens, so left/right padding does not shift their positions.
        positions = (valid.long().cumsum(dim=1) - 1).clamp_min(0)
        hidden = self.token_embedding(token_ids) + self.position_embedding(positions)
        hidden = hidden.masked_fill(~valid[:, :, None], 0.0)
        for block in self.blocks:
            hidden = block(hidden, allowed, valid)
        return self.lm_head(self.final_norm(hidden))  # (B, T, vocab_size), logits
```

这里已经把 causal 条件包含在 `allowed` 里，所以调用 SDPA 时不再设置 `is_causal=True`。PAD 输出清零发生在 attention 后及每个完整 block 末尾；残差、LayerNorm 和线性层的偏置都不能代替 query mask。模型最后一层可以在 PAD 位置产生非零 logits，但这些位置不会参与 loss，也不会被选来生成。

本例为突出 mask 不使用 dropout。如果给 SDPA 添加 dropout，评估时要显式传入 `dropout_p=0.0`，不能仅依赖 `model.eval()` 自动关闭这个函数的 dropout。

## 4. 训练与文本生成

### 4.1 只在有效标签上计算 loss

模型输出的是未归一化的 logits，形状为 $(B,T,41)$。每个有效标签都是一个类别 ID。下面将前两维展平交给 cross entropy，不要先对 logits 做 softmax。

```python
def next_token_loss(logits, labels, token_ids, pad_id):
    labels = labels.masked_fill(token_ids.eq(pad_id), pad_id)
    count = labels.ne(pad_id).sum()
    if count.item() == 0:
        raise ValueError('This batch has no supervised tokens')
    total = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        labels.reshape(-1),
        ignore_index=pad_id,
        reduction='sum',
    )
    return total / count


if not input_rows:
    raise ValueError('Training text must contain at least one supported character')
inputs = torch.tensor(input_rows, dtype=torch.long)
labels = torch.tensor(label_rows, dtype=torch.long)
model = TinyDecoder(tokenizer.size(), tokenizer.pad_id, max_sequence_length)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

# One training step; repeat with batches to actually train the model.
model.train()
optimizer.zero_grad(set_to_none=True)
loss = next_token_loss(model(inputs), labels, inputs, tokenizer.pad_id)
loss.backward()
optimizer.step()
```

loss 的分母是实际参与监督的 token 数，不是 `B × T`。`ignore_index` 忽略的是指定标签位置，不会从其他位置的 softmax 分母中移除 PAD 这个词表类别；有效位置仍在完整 41 类上训练。这一行为见 [PyTorch CrossEntropyLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)。

这三个动作需要分清：**attention mask 控制能读哪些位置；query 输出清零控制哪些隐藏状态有效；loss mask 控制哪些预测接受监督。** 它们的职责不同，只做其中一个不能替代另外两个。

### 4.2 从一次预测到自回归生成

给定提示词 `cat`，假设输入窗口最大长度为 5，过程如下。表中“选出的字符”只用于演示循环，不是训练效果报告：

| 步骤 | 送入模型的 token | 读取哪个位置的 logits | 假设选出的字符 |
| --- | --- | --- | --- |
| 1 | `PAD BOS c a t` | 最后一个有效位置 `t` | `s` |
| 2 | `BOS c a t s` | `s` | 空格 |
| 3 | `c a t s 空格` | 空格 | `r` |

一次前向会得到每个位置的 logits，但续写只使用**最后一个有效输入位置**的输出。选中的 token 加到末尾，再次运行模型；窗口满了就丢弃最左边的旧 token。本例使用窗口内的位置编号，所以窗口滑动后会重新编号，不代表实现了无限上下文。

空提示词也不是全 PAD：窗口最后一位是 BOS，由它预测第一个字符。下面只处理单条提示词，并统一左 padding，因此最后一个有效位置就是 `-1`。如果改成右 padding 的 batch，不能继续无条件使用 `logits[:, -1]`，要逐条找到最后一个有效位置。

```python
@torch.no_grad()
def generate_greedy(model, tokenizer, prompt, max_new_tokens=30):
    if max_new_tokens < 0:
        raise ValueError('max_new_tokens must be nonnegative')
    model.eval()
    device = next(model.parameters()).device
    token_ids = [tokenizer.bos_id] + tokenizer.tokenize(prompt)
    for _ in range(max_new_tokens):
        context = token_ids[-model.max_length:]
        context = [tokenizer.pad_id] * (model.max_length - len(context)) + context
        inputs = torch.tensor([context], dtype=torch.long, device=device)
        next_logits = model(inputs)[0, -1].clone()
        # PAD/BOS are not ordinary continuation characters.
        next_logits[[tokenizer.pad_id, tokenizer.bos_id]] = float('-inf')
        next_id = int(next_logits.argmax().item())
        token_ids.append(next_id)
    return ''.join(tokenizer.token_to_character(i) for i in token_ids[1:])
```

这里用的是 **greedy decoding**：选择最高分的 token。对同一组有限 logits，softmax 不改变大小顺序，所以求 argmax 不必先计算概率。它不是随机采样，也不保证整段输出拥有最大联合概率。

如果要随机采样，应当先把允许生成的 token 转成归一化概率，再根据这个分布抽样；不是在所有 token 里均匀随机挑一个。两者的区别可参见 [Hugging Face 生成策略文档](https://huggingface.co/docs/transformers/v4.45.2/en/generation_strategies#decoding-strategies)。Temperature、Top-K、Top-P 等留到[生成时如何选下一个 token](/posts/2026-05-16-llm-decoding-strategies/)中展开。

由于这里没有 EOS，函数恰好生成 `max_new_tokens` 个字符后停止。要学习“自然结束”，需要同时把 EOS 加入词表、训练目标和生成的停止条件，不能只在循环里检查一个从未训练过的结束标记。

## 5. 代码与验证边界

上面的 Python 代码块按顺序组成一个教学示例：纯 Python 的数据构造，加上 PyTorch 的最小模型、单步训练和 greedy 生成函数。一次 optimizer step 只演示调用关系，不代表模型已经学会语料。

原始实验仓库仍可参考：[Text-Generator-With-Decoder](https://github.com/CHENHUI-X/Text-Generator-With-Decoder/tree/main)。本文给出的 BOS、窗口边界和安全 mask 修正，不代表该外部仓库已同步修改。

本文的数据构造已用 Python 执行核对，mask 与 attention 数值行为使用 NumPy 等价计算验证，Python 代码块做过语法解析。当前核验环境没有 PyTorch，**不能据此声称已运行 PyTorch 前向、反向传播、GPU 训练或生成质量测试**。这些仍需在安装 PyTorch 的环境中验证。

## Reference

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)：attention、causal mask 与训练并行性的原始说明
- [PyTorch scaled_dot_product_attention](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)：mask 约定、张量形状与 dropout 参数
- [PyTorch MultiheadAttention](https://docs.pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html)：key padding mask 与布尔屏蔽语义
- [PyTorch CrossEntropyLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)：logits、类别标签与 ignore_index
- [Hugging Face generation strategies](https://huggingface.co/docs/transformers/v4.45.2/en/generation_strategies)：greedy 与随机采样
- [Building a language model](https://wingedsheep.com/building-a-language-model/)：原始实现思路的参考文章
