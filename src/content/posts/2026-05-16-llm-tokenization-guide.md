---
title: Tokenization 完全指南 — 从字符到子词，模型到底是怎么"看懂"文字的
published: 2026-05-16
description: 从"模型只能读数字"这个最朴素的问题出发，一步步推出字符级、词级、BPE、WordPiece、Unigram、SentencePiece 的原理和代码实现。
category: Deep Learning
tags:
- llm
- tokenization
- nlp
- deep learning
draft: false
---

## 0. 前言

你有没有好奇过一个问题: **语言模型读的是文字, 但它存的都是数字——中间这一步是怎么转过来的?**

当你输入 "ChatGPT 真厉害" 时, 模型实际上看到的是这样的东西:

```
[1212, 8463, 129, 93127, 4273]
```

这一串数字是怎么来的? 这就是 **Tokenizer** (分词器) 做的事.

但 Tokenizer 远不止"把文字转成数字"这么简单. 选什么样的 Tokenizer, 直接决定了:

- 模型能处理的词表有多大 (几万 vs 几十万)
- 生僻词能不能表示 (会不会出现 OOV: Out-of-Vocabulary)
- 不同语言的表现 (英语好不代表中文也好)
- 序列有多长 (token 越多, 计算越贵)

这篇文章就来完整梳理 Tokenization 的技术演进: 从最朴素的字符级/词级方案, 到 BPE, 到 WordPiece, 再到 Unigram 和 SentencePiece.

---

## 1. 先想清楚: 难点在哪

假设你有一句话: `I love machine learning`.

模型是数学机器, 它只能处理数字. 所以你需要一个映射函数 $f$:

$$
f(\text{"I love machine learning"}) \to [x_1, x_2, ..., x_n]
$$

**方案一: 字符级别 (Character-level)**

把每个字符映射到一个数字:

```python
# ' '→1, 'I'→2, 'a'→3, 'c'→4, 'e'→5, 'g'→6, 'h'→7, 'i'→8, 'l'→9, 'm'→10, 'n'→11, 'o'→12, 'r'→13, 'v'→14
```

结果: `[2, 1, 9, 12, 14, 5, 1, 10, 3, 4, 7, 8, 11, 5, 1, 9, 5, 3, 13, 11, 8, 11, 6]` — 23 个 token (和原始字符串的字符数一样).

词表很小 (26 个字母 + 标点 ≈ 100 左右), 但序列太长, 而且**单个字符没有语义**——模型很难从 'm' 这个字符学会 "machine" 是什么意思.

**方案二: 词级别 (Word-level)**

把每个词映射到一个数字:

```
'I' → 423, 'love' → 156, 'machine' → 8971, 'learning' → 442
```

结果: `[423, 156, 8971, 442]` — 4 个 token, 每个都有明确的语义.

但问题来了: 词表要多大? 英语有几十万个词, 加上专有名词、拼写变体、新造词, 词表轻易突破百万. 而且遇到没见过的词 (OOV: Out-of-Vocabulary) 就只能给 `<UNK>` (未知标记), 信息直接丢失.

**方案三: 子词级别 (Subword-level) — 黄金平衡点**

子词级别做了个巧妙的平衡: 常见的词保持完整 (如 "love"), 不常见的词拆成更小的子词 (如 "tokenization" → "token" + "ization").

这就是当前所有主流模型使用的方案. 接下来详细介绍各个子词方案.

---

## 2. BPE (Byte Pair Encoding) — 最常用的方案

BPE 最早是 1994 年提出的一种数据压缩算法, 被 Sennrich 等人 (2016) 引入 NLP. 现在 GPT 系列、LLaMA、Bloom 等都用它.

### 2.1 核心思路

BPE 的核心是: **从最小的单位(字符)开始, 逐步合并出现频率最高的 Pair, 直到达到目标词表大小**.

这不是直接给你一个词表, 而是一个**合并规则列表**——你知道先合并什么、后合并什么, 就能把任意文本切分成 token.

### 2.2 一步一步的推导 (带例子)

假设我们有一个小语料库, 包含以下 5 个词 (每个重复多次):

```
low low low low low low low low low low   (10 个 "low")
lowest lowest lowest lowest lowest        (5 个 "lowest")
newer newer newer newer newer             (5 个 "newer")
wider wider wider wider wider             (5 个 "wider")
new new                                  (2 个 "new")
```

目标词表大小: 设为 18 个 token. (初始有 11 个独立字符, 每次合并新增 1 个 token, 所以需要做 18 − 11 = 7 次合并.)

**第 1 步: 初始化**

把所有词拆成字符, 加上 `</w>` (词尾标记, 表示一个词的结束):

```
l o w </w>    (10 次)
l o w e s t </w>  (5 次)
n e w e r </w>   (5 次)
w i d e r </w>   (5 次)
n e w </w>       (2 次)
```

初始 token 集合: `{l, o, w, e, s, t, n, r, i, d, </w>}` — 11 个.

**第 2 步: 统计 Pair 频率**

统计所有相邻字符 Pair 的出现次数:

| Pair | 计数 |
|------|------|
| (l, o) | 10+5 = 15 |
| (o, w) | 10+5 = 15 |
| (w, </w>)| 10+2 = 12 |
| (e, s) | 5 |
| (s, t) | 5 |
| (t, </w>) | 5 |
| (n, e) | 5+2 = 7 |
| (w, e) | 5+5 = 10 |
| (e, r) | 5+5 = 10 |
| (r, </w>)| 5+5 = 10 |
| (w, i) | 5 |
| (i, d) | 5 |
| (d, e) | 5 |
| (e, w) | 5+2 = 7 |

**第 3 步: 合并频率最高的 Pair**

最高频的 Pair 是 (l, o) 和 (o, w) 并列第一, 都是 15 次. 选 (l, o) 先合并.

合并后, 所有 "l o" 变成 "lo". 现在词汇变成了:

```
lo w </w>          (10 次)
lo w e s t </w>   (5 次)
n e w e r </w>    (5 次)
w i d e r </w>    (5 次)
n e w </w>        (2 次)
```

加入新 token: `lo`. 当前词表: 12 个.

注意一个关键点: (o, w) 这个 Pair **不再存在了**——因为原来的 "l o" 已经合并成了 "lo", "o" 不再是独立 token, 所以 (o, w) 无法被合并.

所以第二次合并时, 我们要重新统计:

| Pair | 计数 | 来自 |
|------|------|------|
| (lo, w) | 10+5 = 15 | low + lowest |
| (w, </w>)| 10+2 = 12 | low + new |
| (w, e) | 5+5 = 10 | lowest + newer |
| (e, r) | 5+5 = 10 | newer + wider |
| (r, </w>)| 5+5 = 10 | newer + wider |
| (n, e) | 5+2 = 7 | newer + new |
| (e, w) | 5+2 = 7 | newer + new |
| ... | ... | ... |

最高频的是 (lo, w) = 15 次. 合并它:

```
low </w>          (10 次)
low e s t </w>   (5 次)   — "lo w" 变成了 "low"
n e w e r </w>   (5 次)
w i d e r </w>   (5 次)
n e w </w>       (2 次)
```

加入新 token: `low`. 当前词表: 13 个.

就这样反复合并, 直到词表达到目标大小. 每次合并后, 所有序列都会更新, 然后重新统计 Pair 频率.

最终 BPE 学到的一系列合并规则类似:

```
(l, o) → lo
(lo, w) → low
(low, </w>) → low</w>
(e, r) → er
(n, e) → ne
(ne, w) → new
...
```

有了这些规则, 任何新词都可以按同样的方式切分. 比如 "newest":
1. 拆成字符: `n e w e s t </w>`
2. 按规则合并: `ne w e s t </w>` → `new e s t </w>` (不能再合并了)
3. 最终: `['new', 'e', 's', 't', '</w>']`

### 2.3 推理时如何分词

训练完后, BPE 不再统计频率, 而是直接按**学到的合并规则**来切分.

具体步骤:
1. 把输入文本拆成基本单元 (字符或 byte, 取决于具体实现)
2. 从左到右扫描, 看能不能应用学到的合并规则
3. 尽可能合并, 直到不能再合并为止

对于 GPT-2 和 GPT-3 系列, 使用的是 **Byte-level BPE**: 最小单位不是字符, 而是 **byte** (字节). 所以"拆成基本单元"这一步是把每个 Unicode 字符先编码为 1-4 个 byte, 然后再基于 byte 序列做合并. 这样做的好处是: **可以表示任意 Unicode 字符**——不管中文、日文、emoji, 都能拆成 byte 序列来处理, 不会出现 `<UNK>`.

GPT-2 的词表大小是 50,257. LLaMA 系列的词表是 32,000 (LLaMA-1) 或 128,000 (LLaMA-3).

### 2.4 BPE 的优缺点

**优点:**
- 简单直观, 实现容易
- 对常见词保持完整, 对罕见词拆分成子词
- Byte-level 变体可以处理任意文本

**缺点:**
- 基于频率而不是基于**语义相关性**, 可能会把语义相关的词拆开 (如 "unbelievable" → "un" + "believ" + "able", 可能不如 "un" + "believe" + "able" 好)
- 频率统计完全由语料决定, 语料偏斜会导致 token 分布不合理

---

## 3. WordPiece — BERT 的选择

WordPiece 是 Google 为 BERT 开发的 tokenizer, 和 BPE 非常相似, 但合并标准不同.

### 3.1 和 BPE 的区别

BPE 合并的是**频率最高**的 token pair.

WordPiece 合并的是**让语料库似然 (likelihood) 提升最大**的 token pair.

具体来说, WordPiece 计算每个候选 pair 的 score:

$$
\text{score}(x, y) = \frac{\text{freq}(xy)}{\text{freq}(x) \times \text{freq}(y)}
$$

其中 $\text{freq}(xy)$ 是合并后的 token 在语料中的频率, $\text{freq}(x)$ 和 $\text{freq}(y)$ 是两个原始 token 的频率.

为什么要用这个公式? 直观理解:

- 如果 $x$ 和 $y$ 经常一起出现 ($\text{freq}(xy)$ 大), 但各自出现得少 ($\text{freq}(x)$ 和 $\text{freq}(y)$ 小), 说明它们的组合特别有意义, score 高
- 如果 $x$ 和 $y$ 各自出现得很多, 偶尔碰到一起, 那它们的组合可能只是巧合, score 低

这比 BPE 的纯频率更"智能"——它倾向于合并那些**共现特别紧密**的 pair, 而不是单纯出现多的 pair.

### 3.2 例子

继续用上面的语料. 假设目前 token 集包含 `l`, `o`, `w`, `e`, `s` 等字符, 我们需要决定先合并哪一对.

计算 $\text{score}(\text{l}, \text{o})$:

- $\text{freq}(\text{lo})$: "l" 和 "o" 相邻出现 15 次 (low ×10, lowest ×5)
- $\text{freq}(\text{l})$: "l" 本身出现 15 次 (都在词首)
- $\text{freq}(\text{o})$: "o" 出现 15 次

$$\text{score}(\text{l}, \text{o}) = \frac{15}{15 \times 15} = \frac{1}{15} \approx 0.067$$

再看 $(\text{o}, \text{w})$:

- $\text{freq}(\text{ow})$: "o" 和 "w" 相邻出现 15 次 (low ×10, lowest ×5)
- $\text{freq}(\text{o})$: 15
- $\text{freq}(\text{w})$: **27** — 注意 "w" 不仅出现在 low (×10)、lowest (×5), 还出现在 newer (×5)、wider (×5)、new (×2) 中. 所以是 $10+5+5+5+2 = 27$.

$$\text{score}(\text{o}, \text{w}) = \frac{15}{15 \times 27} \approx 0.037$$

这里可以发现区别:

- (l, o) 的 score 0.067 > (o, w) 的 0.037, 因为 `l` 几乎只出现在 `o` 前面——$\text{freq}(\text{l})$ 和 $\text{freq}(\text{(l, o)})$ 几乎相等, 说明 "l" 和 "o" 是近乎绑定在一起的.
- 而 `w` 还可以接 `e` (newer)、`i` (wider)、`</w>` (low, new), 所以 `o` + `w` 不是唯一的组合——score 自然更低.

**这就是 WordPiece 的本质**——它衡量的是 $x$ 和 $y$ 的**共现强度**, 而不是共现频率. 公式 $\text{score}(x, y) = \frac{\text{freq}(xy)}{\text{freq}(x) \cdot \text{freq}(y)}$ 实际上就是 **PMI (Pointwise Mutual Information)** 的变形:

$$\text{PMI}(x, y) = \log \frac{P(x, y)}{P(x) P(y)} \propto \log \frac{\text{freq}(xy)}{\text{freq}(x) \cdot \text{freq}(y)}$$

WordPiece 不取 log (为了保留排序等价性), 但思想完全一样: **如果两个 token 同时出现远高于随机期望, 就应该合并它们**.

### 3.3 实际使用

WordPiece 在 BERT 和 DistilBERT 中使用 (RoBERTa 用的是 Byte-level BPE, 与 GPT-2 相同). 词表大小通常是 30,000 左右. 和 BPE 一样, 训练完成后就是一个确定的合并规则集.

BERT 的 WordPiece 会在 token 前面加 `##` 表示这不是词的开头 (continuation). 比如:

```
"unbelievable" → ["un", "##believe", "##able"]
```

这种标记方式让模型能区分"词首"和"词中"的子词——同一个子词在不同位置可能表示不同含义.

---

## 4. Unigram — 由概率说话

Unigram 是另一种子词分词方案, 和 BPE/WordPiece 的思路完全相反.

### 4.1 BPE/WordPiece 是"自底向上", Unigram 是"自顶向下"

- **BPE/WordPiece**: 从字符开始, 逐步合并 → 自底向上
- **Unigram**: 从一个很大的候选词表开始, 逐步删除"不重要"的 token → 自顶向下

Unigram 的具体做法:

1. 先准备一个很大的候选词表 (比如从语料中提取所有可能的子词序列)
2. 对每个 token, 计算删除它后对语料似然的影响
3. 删除影响最小的那些 token
4. 重复直到达到目标词表大小

### 4.2 损失函数

Unigram 的词表学习和分词不是同时做的, 流程是: 先用 EM 算法迭代估计 token 概率 $P(t)$, 然后用 Viterbi 找最优切分.

给定词表 $V$ 和 token 概率 $P(t)$, 一个句子 $X$ 的最优分词由 Viterbi 算法找到:

$$
\operatorname*{argmax}_{t_1, \ldots, t_k \in V} \sum_{i=1}^{k} \log P(t_i)
$$

即: 穷举所有可能的分词方式, 选出使概率和最大的那一种. Viterbi 用动态规划做这个, 实际实现中用 trie 树限制候选数, 复杂度约 $O(L \cdot M)$, 其中 $M$ 是最大 token 长度 (通常 ≤ 50).

训练过程的核心是 **EM 算法** (标准做法是 soft EM, 即用 forward-backward 算法计算每个 token 在各分词路径上的期望出现次数; 以下描述 hard EM 变体):

1. **E 步**: 对每个句子, 用 Viterbi 找到当前最优分词, 统计每个 token 被选中的次数
2. **M 步**: 根据频率更新 $P(t) = \frac{\text{count}(t)}{\sum_{v \in V} \text{count}(v)}$
3. 重复直到收敛

收敛后, 对每个 token 评估**删除它带来的似然损失**, 删除损失最小的 token, 然后重新训练. 重复直到词表降到目标大小.

### 4.3 和 BPE/WordPiece 的对比

**Unigram 的优势:**
- 能显式控制词表大小 — 不像 BPE 需要逐步合并到目标大小
- 学习到的 token 更"合理" — 概率框架比纯频率统计更鲁棒
- 可以输出多个候选分词路径 (不只是最优的那一种)

**劣势:**
- 需要 Viterbi 解码, 比 BPE 的贪心匹配慢
- 训练过程更复杂

**实际使用:** Unigram 是 SentencePiece 的默认算法, 也用在 T5、ALBERT 等模型中.

---

## 5. SentencePiece — 把一切统一起来

前面讲的 BPE、WordPiece、Unigram, 都有一个共同的前提: **输入文本需要先按空格分成"词"** (pre-tokenization). 这对英语没问题, 但对中文、日文等没有空格的文字来说就尴尬了——"我喜欢你" 应该被切分成什么?

SentencePiece 解决了这个问题: **它把原始文本直接当作 Unicode 字符序列处理, 不需要 pre-tokenization**.

### 5.1 SentencePiece 的核心思想

SentencePiece 的输入不是"词", 而是**原始字符串** (Unicode 字符序列). 它把整个文本看作一个字符流, 然后用 BPE 或 Unigram 算法来训练.

关键特性:

1. **空格也是字符**: 英语中的空格被当作普通字符, 用 `▁` (下划线符号) 表示. 分词结果类似于 `▁I▁love▁machine▁learning`.
2. **不需要语言特定预处理**: 中文、日文、韩文可以直接处理, 不需要先分词
3. **支持 BPE 和 Unigram 两种算法**

### 5.2 对比: SentencePiece BPE vs SentencePiece Unigram

LLaMA、Mistral 使用的是 **SentencePiece + BPE** (字符级别, 不是 byte-level). LLaMA-3 改用了 tiktoken 库的 byte-level BPE, 不再使用 SentencePiece.

T5、ALBERT 使用的是 **SentencePiece + Unigram**.

| 模型 | Tokenizer | 词表大小 | 说明 |
|------|-----------|---------|------|
| GPT-2/3 | Byte-level BPE | 50,257 | 可以表示任意 byte 序列 |
| BERT | WordPiece | 30,000 | 用 ## 标记续词 |
| LLaMA | SentencePiece + BPE | 32,000 | 字符级别, 适合中英文混合 |
| LLaMA-3 | tiktoken BPE (byte-level) | 128,000 | 大词表, 编码效率高 |
| T5 | SentencePiece + Unigram | 32,000 | 可输出多条分词路径 |
| Qwen | tiktoken BPE (byte-level) | 152,000 | 中文优化, 同 GPT-4 架构 |
| GPT-4 | 未知 (推测是 tiktoken BPE) | ≈100K | 未公开 |

---

## 6. 代码: 手把手看 tokenizer 怎么工作

### 6.1 用 HuggingFace `tokenizers` 训练一个 BPE

```python
from tokenizers import Tokenizer, trainers, models

# 创建一个 BPE tokenizer
tokenizer = Tokenizer(models.BPE())

# 准备训练器
trainer = trainers.BpeTrainer(
    vocab_size=5000,
    special_tokens=["<unk>", "<s>", "</s>"],
    min_frequency=2,
    show_progress=True
)

# 准备语料
files = ["corpus.txt"]  # 你的文本文件

# 训练
tokenizer.train(files, trainer)

# 测试
output = tokenizer.encode("I love machine learning")
print(output.tokens)   # 类似 ['I', 'Ġlove', 'Ġmachine', 'Ġlearning'] (不同 tokenizer 输出不同)
print(output.ids)      # [32, 156, 4231, 2156]
```

### 6.2 查看 LLaMA 的 tokenizer

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

# 英文
text = "Tokenization is the first step"
tokens = tokenizer.tokenize(text)
ids = tokenizer.encode(text)
print(tokens)
# ['▁Token', 'ization', '▁is', '▁the', '▁first', '▁step']
print(len(ids))  # 6 个 token

# 中文
text_cn = "大语言模型分词器"
tokens = tokenizer.tokenize(text_cn)
ids = tokenizer.encode(text_cn)
print(tokens)
# ['▁大', '语言', '模型', '分词', '器']
print(len(ids))  # 5 个 token

# 发现 "语言" 是一个完整的 token!
# 这是因为 LLaMA 的训练语料也包含中文, "语言" 常见, 被合并了
```

### 6.3 token 长度分布 (示意)

不同语言的 token 效率不一样 (以下输出为近似值, 使用 LLaMA-2 tokenizer):

```python
texts = [
    ("English", "The quick brown fox jumps over the lazy dog"),
    ("中文", "大语言模型分词器的工作原理是什么"),
    ("混合", "BERT 的 WordPiece 和 LLaMA 的 BPE 有什么区别"),
]

for lang, text in texts:
    ids = tokenizer.encode(text)
    chars_per_token = len(text) / len(ids)
    print(f"{lang}: {len(ids)} tokens, {chars_per_token:.1f} chars/token")
```

输出类似:

```
English: 10 tokens, 4.3 chars/token
中文: 10 tokens, 1.6 chars/token
混合: 15 tokens, 2.4 chars/token
```

中文的 chars/token 更低, 意味着**同样的信息量, 中文需要的 token 更多**. 这就解释了为什么中文模型的上下文窗口"不够用"——同样的 4096 token, 英文能读约 17600 个字符, 中文只能读约 6600 个字符.

### 6.4 自己实现一个简化版 BPE

```python
from collections import defaultdict

def train_bpe(corpus: list[str], vocab_size: int):
    """训练一个简化版 BPE"""

    # 1. 初始化: 把词拆成字符 + 词尾标记
    words = []
    for text in corpus:
        for word in text.split():
            words.append(" ".join(list(word)) + " </w>")

    # 2. 初始词表 (所有字符)
    vocab = set()
    for word in words:
        for char in word.split():
            vocab.add(char)

    # 3. 重复合并
    merges = []
    while len(vocab) < vocab_size:
        # 统计所有 pair 频率
        pairs = defaultdict(int)
        for word in words:
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i+1])] += 1
        
        if not pairs:
            break
        
        # 找频率最高的 pair
        best_pair = max(pairs, key=pairs.get)
        merges.append(best_pair)
        
        # 合并
        new_words = []
        for word in words:
            new_word = word.replace(
                f"{best_pair[0]} {best_pair[1]}",
                f"{best_pair[0]}{best_pair[1]}"
            )
            new_words.append(new_word)
        words = new_words
        
        # 加入词表
        vocab.add(f"{best_pair[0]}{best_pair[1]}")
    
    return merges

def apply_bpe(text: str, merges: list):
    """用学到的合并规则分词"""
    words = text.split()
    result = []
    for word in words:
        tokens = list(word) + ["</w>"]
        # 注意: 简化实现, 实际 BPE 需要更复杂的合并逻辑
        for merge in merges:
            i = 0
            while i < len(tokens) - 1:
                if tokens[i] == merge[0] and tokens[i+1] == merge[1]:
                    tokens = tokens[:i] + [f"{merge[0]}{merge[1]}"] + tokens[i+2:]
                else:
                    i += 1
        result.extend(tokens)
    return result

# 测试
merges = train_bpe(["low low low lowest newer newer wider"], vocab_size=20)
print(apply_bpe("lowest", merges))
```

> 注意: 这是简化版, 实际 BPE 实现需要考虑词频加权、byte-level 编码等.

---

## 7. Tokenizer 选型指南

### 7.1 不同模型的 Tokenizer

| 模型 | Tokenizer 类型 | 词表大小 | 是否支持中文 | 特殊标记 |
|------|--------------|---------|------------|---------|
| GPT-2 | Byte-level BPE | 50,257 | ✅ (通过 byte) | 无 |
| BERT | WordPiece | 30,000 | ❌ (需额外中文版) | ## |
| LLaMA-2 | SentencePiece + BPE | 32,000 | ✅ | ▁ |
| LLaMA-3 | tiktoken BPE (byte-level) | 128,000 | ✅ | 无 (byte-level) |
| T5 | SentencePiece + Unigram | 32,000 | ✅ | ▁ |
| ChatGLM | SentencePiece | 130,000 | ✅ (中文优化) | ▁ |
| Qwen | tiktoken BPE (byte-level) | 152,000 | ✅ (中文优化) | 无 (byte-level) |

### 7.2 怎么选?

- **做英文任务**: Byte-level BPE 或 WordPiece 都行, 词表 30K-50K
- **做多语言任务**: SentencePiece + BPE/Unigram, 词表 100K 以上
- **做中文任务**: 选中文优化的 tokenizer (LLaMA-3, Qwen, ChatGLM), 或者自己训练一个
- **追求效率**: 大词表 → 每个 token 信息量更大 → 序列更短 → 计算更快 (但 embedding 层更大)
- **追求覆盖**: 小词表 → 基本不会遇到 `<UNK>` → 但序列更长

### 7.3 一个重要的权衡

词表大小和序列长度是 trade-off:

计算量
$$\propto \underbrace{L^2 d}_{\text{Self-Attention}} + \underbrace{V d}_{\text{Embedding}}$$

其中 $L$ 是序列长度, $d$ 是隐藏层维度, $V$ 是词表大小. (为简洁, 忽略了 FFN 层的 $L d^2$ 项, 但 trade-off 的结论不变.)

词表越大, 序列越短 (因为每个 token 包含的信息更多), Self-Attention 的 $L^2$ 项显著降低; 但 Embedding 层 $V d$ 变大.

LLaMA-3 选择了 128K 的大词表 (相比 LLaMA-2 的 32K), 序列更短但嵌入层更重. 实测表明好处大于坏处.

---

## 8. 总结

Tokenization 看起来是"把文字转数字"的小事, 但它的设计直接影响模型的能力和效率:

| 方法 | 核心思想 | 典型应用 | 合并标准 |
|------|---------|---------|---------|
| 字符级 | 每个字符映射一个数字 | 理论上可行 | — |
| 词级 | 每个词一个 token | N-gram 时代 | — |
| BPE | 从字符开始, 逐对合并高频 pair | GPT, LLaMA | 纯粹频率 |
| WordPiece | 类似 BPE, 但合并标准用似然提升 | BERT | 似然提升 |
| Unigram | 从大词表开始, 逐步删除不重要 token | T5, ALBERT | 似然损失 |
| SentencePiece | 统一框架, 不需要 pre-tokenization | LLaMA, T5 | 可配 BPE/Unigram |

最核心的一点: **Tokenizer 的好坏, 决定了模型"看到"的是什么**. 一个不好的 Tokenizer 会把语义相关的词拆得七零八落, 或者把不相关的 byte 强行拼在一起——模型学到的是"错位"的语言知识.

下次你再看到 "LLaMA-3 词表 128K" 这种数字时, 就知道它意味着什么了.

---

### 参考资料

1. Sennrich et al., Neural Machine Translation of Rare Words with Subword Units. ACL 2016. [arXiv:1508.07909](https://arxiv.org/abs/1508.07909) — **提出 BPE 用于 NLP**
2. Wu et al., Google's Neural Machine Translation System: Bridging the Gap between Human and Machine Translation. 2016. [arXiv:1609.08144](https://arxiv.org/abs/1609.08144) — **WordPiece 用于 NMT**
3. Kudo, Subword Regularization: Improving Neural Network Translation Models with Multiple Subword Candidates. ACL 2018. [arXiv:1804.10959](https://arxiv.org/abs/1804.10959) — **Unigram**
4. Kudo & Richardson, SentencePiece: A simple and language independent subword tokenizer and detokenizer for Neural Text Processing. EMNLP 2018. [arXiv:1808.06226](https://arxiv.org/abs/1808.06226) — **SentencePiece**
5. Radford et al., Language Models are Unsupervised Multitask Learners. 2019. (GPT-2) — **Byte-level BPE**
6. HuggingFace Tokenizers 文档: [https://huggingface.co/docs/tokenizers](https://huggingface.co/docs/tokenizers)
