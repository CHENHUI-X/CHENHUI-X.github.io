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

语言模型处理的是数字，输入却是文字。这需要先确定：**一段文字应该拆成哪些单位，再给每个单位什么编号？**

例如，我们可以人为规定一个玩具词表：`I → 0`、`love → 1`、`NLP → 2`。于是 `I love NLP` 被编码成 `[0, 1, 2]`。这只是教学例子，不是某个真实模型的输出。

这里要区分三个概念：

- **Token**：切分后的一段内容，可能是一个词、一个字符、一个子词，也可能只对应部分 UTF-8 字节
- **Token ID**：该 token 在词表中的整数编号；编号大不代表含义更复杂
- **Embedding**：模型根据 ID 查到的向量，由模型训练学习，而不是 Tokenizer 直接算出的语义

假设词表大小为 $V$，每个向量有 $d$ 个数，输入有 $L$ 个 token。Embedding 表就是 $E\in\mathbb{R}^{V\times d}$；查出这 $L$ 个 ID 对应的行，得到 $L\times d$ 的输入矩阵。它随后交给 Transformer 处理。

因此，Tokenizer 负责表示方式，模型负责学习这些表示在上下文中的规律。切分得不像人类词语，不等于模型一定学不会；但切分过碎会增加序列长度，丢失字符的未知标记会损失信息。讨论切分方法时，还需要区分**训练词表**和**使用词表分词**这两个阶段。

---

## 1. 先想清楚: 难点在哪

以 `I love machine learning` 为例。

### 1.1 字符级别 (Character-level)

如果每个字符都是一个 token，连空格一起算，这句话有 23 个 token。词表可以只收录语料中出现过的字符，常见英语字符集很小；但多语言、符号和 emoji 会扩大字符集合，不能把“字符级词表”一概理解成约 100 个条目。

字符级模型仍然可以学习词义和句义，只是通常需要用更长的序列表示一句话。另一个细节是，Unicode 码点不等于人眼看到的完整字符：带组合附加符号的字母或某些 emoji 可能由多个码点组成。

### 1.2 词级别 (Word-level)

如果先按空格切词，上面的句子只有 4 个 token。但新词、拼写变体、专有名词会不断出现，有限词表无法全部收录。遇到 OOV（Out-of-Vocabulary，词表外内容）时，简单的词级方案会用 `<UNK>` 代替，原词的信息就丢了。

此外，中文并不能直接照搬英文的空格切词规则。

### 1.3 子词级别 (Subword-level)

子词方案尝试兼顾两边：常见片段可以合成一个 token，不常见的组合可以拆小。例如，在人为设定的词表里，`tokenization` 可以拆成 `token` 和 `ization`；真实切分取决于具体词表。

这也是许多主流语言模型的选择。但“子词”只是历史名称，实际 token 不一定对应词根、词缀或完整字符。

是否出现 `<UNK>`，关键在于**基础单位是否覆盖输入，以及有没有回退机制**，不是词表越小就越安全。包含全部 256 个字节的 Byte-level tokenizer，可以把有效 Unicode 文本编码成 UTF-8 字节再表示；仅包含训练语料字符的子词词表，则可能遇到没见过的字符。

---

## 2. BPE (Byte Pair Encoding) — 最常用的方案

BPE 原本是一种压缩思路。Sennrich 等人把它用于机器翻译中的子词学习：从小单位开始，反复合并高频的相邻单位。这里先使用字符和词尾标记演示，Byte-level 变体稍后再说。参见 [BPE 论文第 3 节](https://aclanthology.org/P16-1162.pdf)。

### 2.1 核心思路

直观地说，如果语料中 `l` 后面经常紧跟 `o`，把它们合成 `lo`，就能少用一个位置。随后，如果 `lo` 又经常紧跟 `w`，就可以继续得到 `low`。

训练时，每一轮都执行：

1. 统计当前 token 序列中相邻 pair 的出现次数，重复词按词频加权
2. 选择频率最高的一对，按固定规则处理并列情况
3. 在 token 边界上合并该 pair 的非重叠出现，记录合并顺序
4. 重新统计，直到满足词表大小或其他停止条件

这里的 pair 是**当前相邻 token**，不一定还是单个字符。训练产物既有词表，也有有序合并规则。

### 2.2 一步一步的推导 (带例子)

用这个小语料：

| 词 | 次数 | 初始 token 序列 |
|---|---:|---|
| low | 10 | `l o w </w>` |
| lowest | 5 | `l o w e s t </w>` |
| newer | 5 | `n e w e r </w>` |
| wider | 5 | `w i d e r </w>` |
| new | 2 | `n e w </w>` |

`</w>` 是本例保留的词尾标记，不是所有 BPE 实现都使用它。初始有 11 种 token：10 个字母加词尾标记。

第一轮中，`(l, o)` 和 `(o, w)` 都出现 $10+5=15$ 次，`(w, </w>)` 出现 $10+2=12$ 次。我们约定并列时选择扫描语料最先遇到的一对，所以先合并 `(l, o)`。

之后 `low` 的表示变成 `lo w </w>`。原来的 `(o, w)` 已经不存在了，不能沿用旧统计。重新计数后，`(lo, w)` 出现 15 次，下一轮得到 `low`。

沿用这个并列规则，做到词表大小 18 时，完整的 7 次合并为：

```text
1. (l, o)       → lo
2. (lo, w)      → low
3. (low, </w>)  → low</w>
4. (e, r)       → er
5. (er, </w>)   → er</w>
6. (n, e)       → ne
7. (ne, w)      → new
```

本例每次都新增一个条目，因此 $11+7=18$。真实训练还要考虑基础字符集、特殊 token、重复候选和停止条件，不能对所有实现都套用“目标词表减字符数就是合并次数”。

### 2.3 推理时如何分词

推理时不重新统计词频，而是遵循训练得到的 **merge rank（合并优先级）**：

1. 按相同的规范化、预分词和基础单位规则处理输入
2. 在当前可合并的相邻 pair 中，优先选择训练时排名最靠前的规则
3. 更新序列，继续执行，直到没有可用规则

对上述从字符逐步训练出来的规则，也可以按训练顺序逐条应用，得到相同结果。第 6.4 节为了直接体现优先级，显式选择当前 rank 最小的 pair。

**这不是“从左到右，看到能合并的就合并”，也不是“每次选词表里最长的片段”。** 例如，假设规则顺序是先合并 `(b, c)`，再合并 `(a, b)`，那么 `a b c` 应当得到 `a bc`；先看最左边就合并，会误得 `ab c`。

<!-- concept-figure:bpe-rank -->

![BPE 合并优先级示意：规则 b+c 的 rank 为 1，a+b 的 rank 为 2，因此 a b c 得到 a bc，而不是 ab c](/images/llm-series/concept-bpe-rank.svg)

*图：这张人为设定的最小规则表，用来单独说明“按已学 rank 合并”；训练语料与真实 tokenizer 的规则会不同。原创示意；流程依据 [Hugging Face BPE 教程的 tokenization 部分](https://huggingface.co/learn/llm-course/en/chapter6/5#tokenization-algorithm)，训练背景见 [Sennrich 等人的论文](https://aclanthology.org/P16-1162/)。*

<!-- /concept-figure:bpe-rank -->

对于上面的 `newest`：

```text
n e w e s t </w>
→ ne w e s t </w>
→ new e s t </w>
```

它的字符都在基础词表中，所以即使整个词没有出现过，仍然能表示出来。

GPT-2 的 Byte-level BPE 则从 UTF-8 字节出发，保留全部 256 种字节作为基础单位；同时有自己的正则预分词规则。它能够表示有效 Unicode 文本，不需要因未见过某个字符而输出 `<UNK>`。不过，单个 token 可能只含一个字符的部分字节，不能保证单独解码每个 token 都是完整字符。参见 [GPT-2 官方编码实现](https://github.com/openai/gpt-2/blob/master/src/encoder.py)。

### 2.4 BPE 的优缺点

- 训练思路直接，能把高频片段压缩成更少的 token
- 切分由语料、预处理、基础单位和合并顺序共同决定
- 高频不等于语义完整，因此 BPE 不保证按词根或词缀切分
- 没有字节覆盖或其他回退机制的字符级 BPE，仍可能遇到未知字符

---

## 3. WordPiece — BERT 的选择

WordPiece 早于 BERT；BERT 使用了它，并不是为了 BERT 才首次提出。理解 WordPiece 时，最好把“如何学词表”和“怎样用词表”分开。前者的公开细节有限，后者可以直接核对 [Google 发布的 BERT 实现](https://github.com/google-research/bert/blob/master/tokenization.py)。

### 3.1 和 BPE 的区别

BPE 的典型训练按相邻 pair 频率合并。WordPiece 的文献描述强调语言模型似然，即选择更能解释训练语料的词表；[GNMT 论文第 4.1 节](https://arxiv.org/pdf/1609.08144)给出了这种方向。

网上常见的相邻频率比值，可以帮助理解“共现紧密”和“出现得多”的区别。但它应当被视为**教学重构中的评分方式，不能当作 Google 原始训练器完整、确定的算法定义**。[Hugging Face 的教学实现](https://huggingface.co/learn/llm-course/en/chapter6/6)也明确说明这一限制。

### 3.2 例子

先看一个简化的统计例子：

- 一对 token 相邻出现 4 次，而它们各自也都只出现 4 次，说明这两段几乎总是一起出现
- 另一对相邻出现 10 次，但两个 token 各自都出现了 100 次，说明它们也经常和别的 token 搭配

用 $c(x)$ 表示当前 token $x$ 的出现次数，$c(x,y)$ 表示相邻 pair $(x,y)$ 的次数，教学评分为：

$$
\operatorname{score}(x,y)=\frac{c(x,y)}{c(x)c(y)}.
$$

前一对得到 $4/(4\times4)=0.25$，后一对是 $10/(100\times100)=0.001$。虽然后一对出现更多，前一对在这种评分下优先。

这个例子只演示统计直觉，不是完整的 WordPiece 训练流程。BERT 风格词表还区分词首 token 和以 `##` 开头的续接 token，训练统计也应对应这些实际 token；不能把所有位置的同一个字母混成一个计数。更不能把“共现紧密”直接解释成“语义更好”。

### 3.3 实际使用：最长匹配，不重放合并规则

BERT 风格 WordPiece 推理依赖**最终词表**，使用 greedy longest-match-first：

1. 对预处理后的一个词，从当前位置开始，寻找词表中最长的匹配片段
2. 词首使用普通条目，后续位置使用带 `##` 的条目
3. 选中后继续处理剩余部分，不做全局最优搜索，也不回溯
4. 如果无法完整切分，原始 BERT 实现会把这个词整体变成 `[UNK]`

假设玩具词表含有 `un`、`##aff` 和 `##able`，而没有更长的可匹配条目，则 `unaffable` 可以切成：

```text
["un", "##aff", "##able"]
```

`##` 表示续接，不是原文真的有两个井号。这个例子用于解释规则，不声称每个 BERT checkpoint 都给出这组结果。

两者的推理规则不同：**BPE 看合并优先级，WordPiece 看当前位置的最长词表匹配。** 两者即使拥有相似的片段，也可能切出不同结果。

---

## 4. Unigram — 由概率说话

### 4.1 从候选词表开始，逐步删减

同一个字符串可能有多种合法切分。假设玩具词表只有 `a`、`b`、`ab`，概率分别为 $0.4$、$0.4$、$0.2$，三者相加为 1。

字符串 `ab` 就有两条路径：

- `[ab]`：路径概率为 $0.2$
- `[a, b]`：路径概率为 $0.4\times0.4=0.16$

Unigram 假设一条路径中各 token 独立出现，以 token 概率的乘积给路径评分。最优路径是 `[ab]`，但另一条路径也可能解释这段文本，训练时不应直接忽略它。

词表学习通常先准备一个较大的候选集合，例如训练语料中的字符和高频子串；反复估计概率，评估删除候选造成的损失，再缩小词表。不是枚举所有可能字符串，也不是保证找到全局最优词表。参见 [Kudo 的 Unigram 论文第 3.2 节](https://aclanthology.org/P18-1007.pdf)。

### 4.2 训练用边缘似然，确定性分词用 Viterbi

设输入文本为 $X$，所有能完整拼回 $X$ 的合法切分组成集合 $\mathcal{S}(X)$。一条切分记为 $s=(t_1,\ldots,t_k)$，其中 $k$ 是这条路径的 token 数，$p(t)$ 是词表中 token $t$ 的概率。那么：

$$
P(s)=\prod_{i=1}^{k}p(t_i),
\qquad
P(X)=\sum_{s\in\mathcal{S}(X)}P(s).
$$

这里的求和是关键：上例字符串 `ab` 的边缘概率为 $0.2+0.16=0.36$，不是只取最大值 $0.2$。

对包含 $N$ 条文本的训练集，记第 $n$ 条为 $X^{(n)}$，最小化的负对数边缘似然是：

$$
\mathcal{L}
=-\sum_{n=1}^{N}\log\left(
\sum_{s\in\mathcal{S}(X^{(n)})}\prod_{t\in s}p(t)
\right).
$$

乘积中的 $t\in s$ 按路径上的每次出现计算，重复 token 也要重复相乘。固定词表后，基本的 **soft EM** 思路是：

1. **E 步**：用 forward-backward 动态规划，对所有合法路径按后验概率加权，得到每个 token 的期望出现次数
2. **M 步**：把这些期望次数归一化，更新 token 概率；具体实现还可能加入平滑等细节
3. 迭代后评估删词造成的似然损失，保留必要的基础单位，剪去影响较小的候选，再重新估计

在玩具例子里，两条路径的后验权重是 $0.2/0.36=5/9$ 和 $0.16/0.36=4/9$。因此 `ab` 的期望次数是 $5/9$，`a`、`b` 各是 $4/9$。只统计最佳路径的整数次数，那是 hard EM 的思路，不能替代这里的 soft EM 描述。

如果使用时只需要一条确定性切分，则取路径概率最大的结果：

$$
s^*=\operatorname*{argmax}_{s\in\mathcal{S}(X)}\sum_{t\in s}\log p(t).
$$

这一步用 **Viterbi 动态规划**完成，不必穷举所有路径。若输入长度为 $L$、token 最大长度为 $M$，用 trie 等结构在每个起点至多沿字符检查 $M$ 步，可以在 $O(LM)$ 时间内构建候选并完成这种简单情形的解码；该界限依赖实现，不能忽略子串复制或匹配的成本。

### 4.3 和 BPE/WordPiece 的对比

Unigram 用概率框架描述多条切分路径，既能选最优路径，也能采样不同切分做 subword regularization。

它并不天然保证语义更合理，也不能只凭“用了 Viterbi”就断言一定比所有 BPE 实现慢。训练、分词的速度还取决于候选数量、数据结构和实现。BPE 同样可以设置目标词表大小，这也不是 Unigram 独有的能力。

---

## 5. SentencePiece — 把一切统一起来

**SentencePiece 是一个 Tokenizer 工具包，不是与 BPE、Unigram 并列的第三种子词学习准则。** 它可以训练 BPE 或 Unigram，并把规范化、空白处理、编码和解码规则一起保存。参见 [SentencePiece 论文](https://aclanthology.org/D18-2012/)。

### 5.1 SentencePiece 的核心思想

前面的玩具例子先按空格切词，这是为了讲解方便，不是所有子词算法的必要条件。SentencePiece 能直接从原始句子训练，无需调用外部的中文分词器或英文分词器。

例如，空格可以在内部表示成 `▁`（U+2581）。假设某个词表包含对应片段，`I love NLP` 的 token 可能显示为 `['▁I', '▁love', '▁NLP']`。这是空格表示方式的示意，不是某个模型的实测切分。

但“不需要外部分词器”不等于“完全不处理文本”：

- 训练配置可以限制是否跨空格、Unicode script 或数字边界形成片段
- 默认规范化采用基于 NFKC 的规则，也可以自定义；空白还可能被整理
- `decode(encode(text))` 是否与原始文本逐字符一致，要看规范化和回退配置，不能仅凭使用 SentencePiece 就保证

这些行为都可在 [SentencePiece 的训练选项](https://github.com/google/sentencepiece/blob/master/doc/options.md)中核对。

### 5.2 SentencePiece BPE 与 Byte-level BPE 不能混为一谈

LLaMA 1/2 使用 SentencePiece BPE，并对未知字符使用 byte fallback；这种“字符子词 + 字节回退”与 GPT-2“从全部字节开始做 BPE”的设计不同。LLaMA 原始论文明确说明了 [SentencePiece 和 UTF-8 字节回退](https://arxiv.org/pdf/2302.13971)。Llama 3 则使用基于 tiktoken 的 BPE，见 [Meta 官方实现](https://github.com/meta-llama/llama3/blob/main/llama/tokenizer.py)。T5 是 Unigram 的常见应用，见 [T5 Tokenizer 文档](https://huggingface.co/docs/transformers/model_doc/t5#transformers.T5Tokenizer)。

另外，`character_coverage=1.0` 表示保留训练数据里观察到的全部字符，不等于预先收录未来可能出现的所有 Unicode 字符。是否启用 byte fallback，需要单独检查配置。

---

## 6. 代码: 手把手看 tokenizer 怎么工作

以下示例分别用于：训练一个小 Tokenizer、检查真实 checkpoint、比较样本文本长度，以及理解 BPE 的核心循环。它们不共享同一份词表。

### 6.1 用 Hugging Face `tokenizers` 训练一个 BPE

仅创建 `models.BPE()` 不会自动得到 GPT-2 风格的 Byte-level 预处理。下面显式配置字节字母表、预分词器和解码器，语料直接写在代码里，不依赖额外的 `corpus.txt`。

```python
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

corpus = [
    "I love machine learning.",
    "I love tokenization.",
    "大语言模型通过 token 处理文字。",
]

tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tokenizer.decoder = decoders.ByteLevel()
trainer = trainers.BpeTrainer(
    vocab_size=300,
    min_frequency=2,
    special_tokens=["<unk>", "<s>", "</s>"],
    initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    show_progress=False,
)
tokenizer.train_from_iterator(corpus, trainer=trainer)

text = "I love machine learning. 你好🙂"
output = tokenizer.encode(text)
print(output.tokens)
print(output.ids)
assert tokenizer.decode(output.ids) == text
assert tokenizer.token_to_id("<unk>") not in output.ids
```

这个例子保留全部字节，所以即使语料没见过 `🙂`，也能回退到字节表示。300 是目标上限，小语料和最小频率限制可能让实际词表更小。声明 `<s>`、`</s>` 为特殊 token，并不会自动在每条序列首尾插入它们；那需要后处理规则。

输出中的可见符号可能是内部字节映射，不能当作原文字符。请以实际运行结果为准，不要把另一套 Tokenizer 的 token 或 ID 填进注释。接口可对照 [Tokenizers 官方 Quicktour](https://huggingface.co/docs/tokenizers/python/latest/quicktour.html)。

### 6.2 查看一个真实 checkpoint 的 tokenizer

下面以公开的 GPT-2 checkpoint 为例，只加载 Tokenizer，不下载模型权重。运行需要安装 `transformers` 和 `tokenizers`，并能访问模型仓库。为了复现实验，应记录包版本，并把 `REVISION` 换成所使用仓库的固定 commit SHA。

```python
from importlib.metadata import version
from transformers import AutoTokenizer

MODEL_ID = "openai-community/gpt2"
REVISION = "main"  # 演示用；正式复现实验请改为固定 commit SHA

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID, revision=REVISION, use_fast=True
)
print("model:", MODEL_ID, "revision:", REVISION)
print("transformers:", version("transformers"))
print("tokenizers:", version("tokenizers"))

for text in ["Tokenization is the first step", "大语言模型分词器"]:
    content_ids = tokenizer.encode(text, add_special_tokens=False)
    model_ids = tokenizer.encode(text, add_special_tokens=True)
    print(repr(text))
    print("tokens:", tokenizer.convert_ids_to_tokens(content_ids))
    print("content IDs:", content_ids)
    print("content count:", len(content_ids))
    print("with special tokens:", len(model_ids))
```

这里不预填切分结果。`add_special_tokens=True` 具体添加什么，由 checkpoint 配置决定，有些 Tokenizer 甚至不会因此增加 token。若换成 LLaMA 2 等默认添加 BOS 的 Tokenizer，直接比较 `tokenize(text)` 和默认 `encode(text)` 的长度，就可能差一个 BOS。

### 6.3 token 长度分布：测量什么，就解释什么

接着上面的代码，可以测量几段文本：

```python
texts = [
    ("English", "The quick brown fox jumps over the lazy dog"),
    ("中文", "大语言模型分词器的工作原理是什么"),
    ("混合", "BERT 的 WordPiece 和 LLaMA 的 BPE 有什么区别"),
]

for lang, text in texts:
    ids = tokenizer.encode(text, add_special_tokens=False)
    chars_per_token = len(text) / len(ids) if ids else 0.0
    print(f"{lang}: {len(ids)} tokens, {chars_per_token:.2f} code points/token")
```

Python 的 `len(text)` 在这里统计 Unicode 码点，不是 UTF-8 字节，也不是信息量。这三句话不是相同内容的翻译，不能据此得出“表达同样的信息，中文固定多花多少 token”。

评估跨语言效率，应使用足够多的可比文本或平行语料，固定 Tokenizer 版本、规范化、特殊 token 和统计口径。上下文窗口限制的是 token 数；可容纳多少汉字、英文词或代码行，取决于具体输入，不能用单条示例比例作保证。

### 6.4 自己实现一个简化版 BPE

下面只使用 Python 标准库，按词训练字符级 BPE。重点是两个边界：**只合并完整的相邻 token，并按学习顺序使用规则。**

如果把 token 序列拼成字符串，再对它做普通的子串替换，`('b', 'c')` 这样的规则可能误改 `('ab', 'c')` 中的后半段。列表或 tuple 能保留边界，避免这个问题。

```python
from collections import Counter

EOW = "</w>"


def split_words(text):
    if EOW in text:
        raise ValueError("</w> is reserved by this teaching tokenizer")
    return [tuple(word) + (EOW,) for word in text.split()]


def merge_pair(symbols, pair):
    """Only merge complete, adjacent, non-overlapping token pairs."""
    merged = []
    i = 0
    while i < len(symbols):
        if i + 1 < len(symbols) and (symbols[i], symbols[i + 1]) == pair:
            merged.append(pair[0] + pair[1])
            i += 2
        else:
            merged.append(symbols[i])
            i += 1
    return tuple(merged)


def train_bpe(corpus, vocab_size):
    words = Counter(word for text in corpus for word in split_words(text))
    vocab = {token for word in words for token in word}
    if vocab_size < len(vocab):
        raise ValueError("vocab_size is smaller than the initial alphabet")

    merges = []
    while len(vocab) < vocab_size:
        pairs = Counter()
        for word, freq in words.items():
            for pair in zip(word, word[1:]):
                pairs[pair] += freq
        if not pairs:
            break

        # Python preserves insertion order: ties use the first encountered pair.
        best_pair = max(pairs, key=pairs.get)
        new_words = Counter()
        for word, freq in words.items():
            new_words[merge_pair(word, best_pair)] += freq
        words = new_words
        merges.append(best_pair)
        vocab.add(best_pair[0] + best_pair[1])
    return vocab, merges


def apply_bpe(text, vocab, merges):
    ranks = {}
    for rank, pair in enumerate(merges):
        ranks.setdefault(pair, rank)  # Keep the earliest rank if repeated.
    result = []
    for tokens in split_words(text):
        if any(token not in vocab for token in tokens):
            raise ValueError("unseen character: this example has no byte fallback")
        while len(tokens) > 1:
            candidates = (
                pair for pair in zip(tokens, tokens[1:]) if pair in ranks
            )
            best_pair = min(candidates, key=ranks.get, default=None)
            if best_pair is None:
                break
            tokens = merge_pair(tokens, best_pair)
        result.extend(tokens)
    return result


corpus = ["low"] * 10 + ["lowest"] * 5 + ["newer"] * 5
corpus += ["wider"] * 5 + ["new"] * 2
vocab, merges = train_bpe(corpus, vocab_size=18)
print(merges)
print(apply_bpe("newest", vocab, merges))
# ['new', 'e', 's', 't', '</w>']
```

这段实现按重复词频加权，并在推理时复用自己训练出来的规则。相邻重复模式也采用非重叠合并，例如把 `a a a` 中的 `(a, a)` 合并为 `aa a`。

它仍不是生产级 Tokenizer：按空白切词会丢失原始空白格式，没有 Unicode 规范化、字节回退、ID 分配和高效索引。遇到未见字符会明确报错。若全部词都已不可再合并，训练会提前停止，词表不一定达到目标大小。

从复杂度看，设规则有 $R$ 条、单个词最初有 $T$ 个基本单位。先建 rank 表需要 $O(R)$；推理每轮扫描相邻 pair，至少减少一个 token，因此这个朴素循环最多进行 $O(T^2)$ 次 token 检查。这里忽略 Python 字符串哈希和拼接成本，不能把这个教学版本的耗时直接推广到有优先队列、缓存等优化的实际 BPE 实现。

---

## 7. Tokenizer 选型指南

### 7.1 比较词表大小，先统一口径

不同文章里的“词表大小”可能指基础子词数、包含特殊 token 的总条目数，或模型 Embedding 的行数。它们不总是相等。

| 具体模型或配置 | Tokenizer | 可核对的数字与口径 |
|---|---|---|
| `openai-community/gpt2` | Byte-level BPE | 模型 `vocab_size` 为 50,257，包含特殊 token |
| `google-bert/bert-base-uncased` | WordPiece | 模型 `vocab_size` 为 30,522；不能代表所有 BERT 版本 |
| LLaMA 1/2 | SentencePiece BPE + byte fallback | 32,000 词表；具体特殊 token 和配置需随 checkpoint 核对 |
| 初代 Llama 3 | 基于 tiktoken 的 Byte-level BPE | 128,000 个基础/可合并条目，加 256 个特殊 token 槽位，共 128,256 |
| `google-t5/t5-small` | Unigram | 模型 `vocab_size` 为 32,128；Tokenizer 还涉及基础 SentencePiece 词表和 100 个 sentinel，不能把模型行数直接当普通子词数 |

数字分别可核对 [GPT-2 配置](https://huggingface.co/openai-community/gpt2/blob/main/config.json)、[BERT 配置](https://huggingface.co/google-bert/bert-base-uncased/blob/main/config.json)、[LLaMA 配置文档](https://huggingface.co/docs/transformers/model_doc/llama#transformers.LlamaConfig)、[Llama 3 的特殊 token 定义](https://github.com/meta-llama/llama3/blob/main/llama/tokenizer.py)及 [BOS 起始 ID 测试](https://github.com/meta-llama/llama3/blob/main/llama/test_tokenizer.py)、[T5 配置](https://huggingface.co/google-t5/t5-small/blob/main/config.json)和 [T5 的 sentinel 配置](https://huggingface.co/google-t5/t5-small/blob/main/tokenizer_config.json)。

“能把中文编码成 ID”也不等于“中文切分高效”，更不等于“模型中文能力好”。字节覆盖解决可表示性，训练语料和模型训练还决定语言能力。

### 7.2 怎么选?

- **使用或微调已有模型**：优先使用该 checkpoint 配套的 Tokenizer、特殊 token 和 chat template；不能随意更换词表，否则 ID 与 Embedding 的对应关系会错位
- **从头训练模型**：在目标语言、领域和代码样本上比较平均长度、长尾长度、未知字符比例与文本还原行为，再决定算法和词表大小
- **多语言需求**：检查各语言在训练语料中的覆盖，不要只用一个总平均值，也没有“必须 100K 以上”的通用门槛
- **效率需求**：把序列长度、模型参数、输出层计算和实际吞吐一起测；更大词表通常有压缩机会，但不是无条件更快

### 7.3 一个重要的权衡

想象把常见的三个小片段合成一个 token：序列变短了，但词表多了一行。这会同时影响模型的参数量和每次计算的运算量，两者需要分别考虑。

令 $V$ 为模型词表行数，$d$ 为隐藏维度，$L$ 为序列长度，$N_{\mathrm{layer}}$ 为层数。输入 Embedding 的参数量是：

$$
P_{\mathrm{embed}}=Vd.
$$

若输出投影不与输入 Embedding 共享权重，输出层还要约 $Vd$ 个参数。**Embedding 查表不是每个 token 都与整个 $V\times d$ 矩阵相乘**，所以不能把 $Vd$ 参数量直接写成整个输入序列的查表计算量。

对稠密 Transformer、全注意力、单条长度 $L$ 的序列，在所有位置都计算词表 logits 的训练前向中，忽略常数、偏置和逐元素操作，主要矩阵运算量可写成下面的量级。这是按矩阵尺寸逐项计算的估计，attention 和 FFN 的结构可对照 [Transformer 原始论文第 3 节](https://arxiv.org/pdf/1706.03762)：

$$
O\!\left(N_{\mathrm{layer}}(Ld^2+L^2d)+LdV\right).
$$

- $Ld^2$：各层线性投影和 FFN，假设 FFN 中间维度与 $d$ 成比例
- $L^2d$：全注意力中的 token 间交互
- $LdV$：在所有位置生成完整词表 logits 的输出投影

使用 KV Cache 一次生成一个新 token 时，若当前共有 $S$ 个可见位置，每步大致对应 $O(N_{\mathrm{layer}}(d^2+Sd)+dV)$，不能继续把整段训练的 $L^2$ 成本照搬过来。实际延迟还受内存带宽、batch size、内核和硬件影响。

例如 $d=4096$ 时，32,000 行的单个 Embedding 表有 131,072,000 个参数，128,000 行则有 524,288,000 个参数。这只是统一口径的算术比较，不是两个真实 checkpoint 的完整参数账本，也不能单凭它判断总体速度。

---

## 8. 总结

| 方法或工具 | 训练时关注什么 | 使用时怎样分词 |
|---|---|---|
| Character-level | 选择字符集合 | 按规定的字符单位切分 |
| Word-level | 选择词和预分词规则 | 查词表，处理未知词 |
| BPE | 逐步合并高频相邻 pair | 按 merge rank 合并 |
| WordPiece | 文献强调似然；原始训练细节不能用教学公式完全代替 | BERT 风格为 greedy longest-match-first |
| Unigram | 概率估计与词表剪枝，目标涉及所有切分的边缘似然 | Viterbi 选最优路径，或按概率采样 |
| SentencePiece | 集成规范化和子词模型的工具包 | 取决于选用的 BPE / Unigram 与配置 |

比较这些方法时，要分清几个概念：**Token 不等于词或字符；训练词表不等于推理分词；可表示、切分高效、模型能力好，是不同的评价维度。**

### 参考资料

1. Sennrich et al. [Neural Machine Translation of Rare Words with Subword Units](https://aclanthology.org/P16-1162/), ACL 2016：BPE 子词学习
2. Wu et al. [Google's Neural Machine Translation System](https://arxiv.org/abs/1609.08144), 2016：WordPiece 的似然训练思路
3. Google [BERT tokenization.py](https://github.com/google-research/bert/blob/master/tokenization.py)：WordPiece 最长匹配和 `[UNK]` 行为
4. Hugging Face [WordPiece 教学实现](https://huggingface.co/learn/llm-course/en/chapter6/6)：明确注明训练算法是基于文献的重构
5. Kudo. [Subword Regularization](https://aclanthology.org/P18-1007/), ACL 2018：Unigram、边缘似然和多路径采样
6. Kudo & Richardson. [SentencePiece](https://aclanthology.org/D18-2012/), EMNLP 2018：不依赖外部分词的工具包
7. Google [SentencePiece 选项](https://github.com/google/sentencepiece/blob/master/doc/options.md)：字符覆盖、规范化和 byte fallback
8. OpenAI [GPT-2 编码实现](https://github.com/openai/gpt-2/blob/master/src/encoder.py)：Byte-level BPE 的基础字节映射与合并顺序
