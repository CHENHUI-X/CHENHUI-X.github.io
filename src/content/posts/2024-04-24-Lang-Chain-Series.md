---
title: LangChain 应用开发：Prompt、检索、工具与 Agent
published: 2024-04-24
description: 用一个文档问答应用串起提示模板、示例选择、RAG、工具调用和 Agent，区分应用层与模型层，并整理旧版 LangChain 示例的迁移边界。
category: Deep Learning
tags:
- deep learning
- llm
- langchain
- nlp
draft: false
---

## 0. 前言

假设要做一个助手，回答「这个项目怎么部署」，必要时读取文档、检查服务状态。模型能写出一段回答，只是其中一环。应用还要决定：给模型看什么资料、怎样保留出处、哪些工具可以调用、失败后怎么办。

本文围绕这些**应用层问题**，按 Prompts、Retrieval、Tools、Agent 四部分展开，重点是理清数据流，以及回答出错时该检查哪一层。

:::note
本文最初发表于 2024 年，2026-10-07 按官方资料重新整理。LangChain v1 已调整包结构，一些旧 chains、retrievers 和缓存组件移入 `langchain-classic`。本文保留可迁移的思路，短代码用于解释接口，不是锁定依赖、完成部署的整套工程；使用前应核对自己的版本与[官方迁移指南](https://docs.langchain.com/oss/python/migrate/langchain-v1)。
:::

先分清四件事：

| 做法 | 改变了什么 | 适合解决什么问题 |
| --- | --- | --- |
| Prompt / few-shot | 本次输入中的规则和例子 | 任务说明、回答格式、示范做法 |
| RAG | 回答时提供的外部证据 | 文档更新、来源可追溯、私有知识问答 |
| 工具调用 / Agent | 应用可以执行的步骤及其选择方式 | 查实时数据、调用服务、完成多步任务 |
| 微调 | 模型参数 | 从训练样本学习任务或行为模式 |

这些方法可以组合。检索不自动等于 Agent，提示模板也不会更新权重。模型如何生成 token，见[解码策略](/posts/2026-05-16-llm-decoding-strategies/)；训练如何改变参数，见[大模型训练篇](/posts/2025-12-05-a-series-on-llm-training-i/)。本篇不重复模型内部原理。

## 1. Prompts

Prompt 可以理解为交给模型的「任务材料包」。其中既有应用规定的任务，也有用户问题、示例、检索材料和对话历史。模板的价值在于把这些部分组织清楚，避免到处拼字符串；它本身不保证模型遵守要求。

### 1.1 PromptTemplate

`PromptTemplate` 适合生成一段文本。固定内容留在模板里，变化的内容通过变量传入。下面是一种构造方法：

```python
from langchain_core.prompts import PromptTemplate

prompt = PromptTemplate.from_template(
    "请用{language}解释以下术语，并给一个例子：{term}"
)
text = prompt.format(language="中文", term="检索增强生成")
```

这一步只生成字符串，没有调用模型。把输入写入模板也不意味着它变成可信指令；尤其不要接受外部用户上传的任意可执行模板。具体模板能力应以对应版本的实现为准。

### 1.2 ChatPromptTemplate

聊天模型的输入通常是**消息列表**，每条消息有角色和内容。可以把系统规则、用户问题、历史回答分开，而不是把角色名写进一大段普通文本。消息还可以包含工具结果，不能把所有输入都归结为最初示例中的三个角色。

下面只检查消息如何组装，不发起远程请求：

```python
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是项目文档助手。证据不足时说明缺少什么，不编造出处。"),
    ("human", "问题：{question}\n\n供参考的资料：\n{context}"),
])
messages = prompt.invoke({
    "question": "这个服务的配置文件在哪里？",
    "context": "[部署手册 §2] 配置文件位于 ~/.example/config.yaml。",
}).to_messages()
```

`from_messages`、变量替换和消息转换可对照[官方 ChatPromptTemplate 源码与示例](https://github.com/langchain-ai/langchain/blob/master/libs/core/langchain_core/prompts/chat.py)。检索到的文档只是资料，即使其中写着「忽略上面的规则」，也不应获得修改应用权限的能力。真正的权限检查必须落在工具执行层。

### 1.3 Example selectors

Few-shot 的意思是：在当前输入里放几个「输入 → 理想输出」示例，让模型参考。例子多了以后，问题不只是选最相似的，还包括长度预算和重复度。

#### Select by length

先扣掉任务说明、当前问题、历史消息和预留输出占用的预算，再决定能放几个示例。某些长度选择器使用字数或简单计数函数；它们不等于目标模型的真实 token 计数。输入更长时通常要减少示例，但不能仅凭字符数断定请求不会超限。可对照[长度选择器实现](https://github.com/langchain-ai/langchain/blob/master/libs/core/langchain_core/example_selectors/length_based.py)。

#### Select by similarity

把输入和示例表示成向量，再找较相近的例子。例如待分类文本是产品退款问题，就优先选择退款类示例。相似度只是选择信号，不代表例子的标签正确，也不意味着所有向量数据库都使用同一种距离。

#### Select by maximal marginal relevance (MMR)

只选最相似的例子，可能得到五个几乎相同的退款案例。MMR（最大边际相关性）同时考虑**相关性与多样性**。设问题为 $q$，已选集合为 $S$，待选例子为 $d$，一种常见评分写法是：

$$
\operatorname{score}(d)=\lambda\,\operatorname{sim}(q,d)
-(1-\lambda)\max_{s\in S}\operatorname{sim}(d,s).
$$

其中 $0\leq\lambda\leq1$。第一项奖励「像当前问题」，第二项惩罚「和已选例子太重复」。$\lambda$ 越大，越看重相关性。首个例子可先按相关性选择；之后逐个更新 $S$。这比「跟整个示例池都尽量不相似」准确，因为要去重的是已经选入上下文的内容。[LangChain 的 MMR 实现](https://github.com/langchain-ai/langchain/blob/master/libs/core/langchain_core/vectorstores/utils.py)使用了这种逐步选择方式。

LangChain 的语义相似度与 MMR 选择器会调用底层 vector store 的搜索接口；具体参数和距离约定见[官方实现](https://github.com/langchain-ai/langchain/blob/master/libs/core/langchain_core/example_selectors/semantic_similarity.py)。同样的 MMR 思路也能用于检索结果去重。

#### Select by n-gram overlap

另一条路线是比较短词串的重合程度。它容易解释，适合表述接近的例子；同义改写可能得分低，字面很像但意思相反的文本也可能得分高。旧示例中的阈值约定属于具体实现，不能当成所有相似度筛选器通用的规则。

<a id="13-few-shot-prompt-templates"></a>

### 1.4 Few-shot prompt templates

把上述选择器接到 few-shot 模板里，流程就是：准备示例池 → 根据当前问题选例子 → 与任务说明、问题一起组装输入。用框架无关的伪代码，可以看清各步职责：

```text
候选 = 从示例池检索与当前输入相关的例子
已选 = 在剩余 token 预算内选择候选，并减少重复
消息 = 系统规则 + 已选输入输出对 + 当前问题
回答 = 调用模型(消息)
```

**检索示例与检索知识不是同一个目的。** 前者通常示范「怎么回答」，后者提供「根据什么回答」。两者都可以用向量搜索，但用于问答的事实证据仍应保留来源和版本，不能因为示例看起来相似就把它当成当前事实。

### 1.5 Partial prompt templates

Partial 的作用是提前填好一部分变量。例如每个请求都用中文，可以先绑定语言，再只传问题。它适合减少重复参数；若绑定的是取时间的函数，还要确认该函数何时求值，以及是否需要为一次请求固定相同时间。具体行为见[模板基类实现](https://github.com/langchain-ai/langchain/blob/master/libs/core/langchain_core/prompts/base.py)。

```python
from langchain_core.prompts import PromptTemplate

base = PromptTemplate.from_template("用{language}解释：{topic}")
chinese = base.partial(language="中文")
text = chinese.format(topic="向量检索")
```

### 1.6 PipelinePrompt

旧文用 `PipelinePrompt` 把介绍、示例、用户问题三段模板连接起来。这里的核心是「分块组装输入」，不必依赖某个旧类名。用明确的中间变量、消息列表或当前版本支持的组合接口，也能完成同样的工作。

组装后至少检查三件事：变量是否遗漏、资料是否被截断、是否给输出留了足够空间。如果要求 JSON，除了提示词，还应使用支持的结构化输出能力并校验字段；合法 JSON 也不代表事实正确。可对照[LangChain 结构化输出文档](https://docs.langchain.com/oss/python/langchain/structured-output)。

## 2. Retrieval

RAG（Retrieval-Augmented Generation，检索增强生成）是在回答时先取得相关资料，再让模型结合资料生成答案。它不要求把知识「写进模型参数」，也不要求检索过程本身必须由 LLM 执行。

以项目文档助手为例，把流程分成两条会更清楚：

- **建库阶段**：读取文档 → 保留来源与权限 → 切块 → 计算表示 → 建索引
- **问答阶段**：接收问题 → 检索允许访问的片段 → 筛选并组装上下文 → 生成回答与出处

这是常见的向量检索路线。已有 SQL、全文搜索或业务 API 时，可以直接利用它们，不必为了叫 RAG 而重新建一套向量库。固定的「先检索、再回答」与由 Agent 动态决定何时检索，是不同的控制方式。[官方检索说明](https://docs.langchain.com/oss/python/deepagents/retrieval)给出了这些区分。

### 2.1 Dataloader

#### Document Loader

Loader 负责把文件、网页等来源转成应用能处理的文档对象，而不是判断文档中的指令该不该执行。`Document` 的正文用于检索，metadata 则适合保存 URL、文件名、章节、更新时间和业务权限标识。

至少保留能回答两个问题的信息：「这段话来自哪里」「这个用户能否看到它」。如果只保存纯文本，即使回答碰巧正确，也很难给出可核对的出处。

原文的自定义 `BaseLoader` 示例同时展示同步、异步和 lazy 读取。它们各有用途：一次读取全部适合小数据；惰性迭代便于逐批处理大数据；异步用于协调 I/O 等待。它们都不自动保证低内存，后续若立刻收集为大列表，仍会把数据全部装入内存。各 loader 的能力应按[对应集成文档](https://docs.langchain.com/oss/python/integrations/document_loaders)检查。

### 2.2 Text Splitters

分块有两个目标：让片段容易被检索命中，以及让最终上下文放得下。块太大，相关的一句话可能埋在无关内容里；块太小，定义、条件和结论又可能分家。不存在对所有文档都最优的固定大小。

#### Split by character

`CharacterTextSplitter` 先按指定分隔符切开，再按长度与 overlap 规则合并。**`chunk_size` 不是完全无效**；问题是，如果一个原始片段本身超过上限，这种 splitter 不一定继续拆开它，因此不能把该值当成严格的最终长度保证。可对照[字符分割文档](https://docs.langchain.com/oss/python/integrations/splitters/character_text_splitter)及[实现中的合并流程](https://github.com/langchain-ai/langchain/blob/master/libs/text-splitters/langchain_text_splitters/character.py)。

#### Recursive Splitter By Character

递归分割器尝试逐级使用更细的分隔符，例如段落、换行、空格、字符。对于中文，应考虑句号、逗号等边界，不能只依赖英文空格。下面是一个起点，不是通用最优配置：

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=80,
    length_function=len,
    separators=["\n\n", "\n", "。", "；", "，", " ", ""],
)
chunks = splitter.split_text("这里替换为需要检索的项目文档。")
```

此处 `len` 计的是 Python 字符数，**不是 token 数**。Overlap 用于尽量保留边界附近上下文，也会带来重复存储和重复检索；不应指望每块都有刚好相同的重叠长度。[递归分割文档](https://docs.langchain.com/oss/python/integrations/splitters/recursive_text_splitter)说明了长度函数、分隔符与 overlap 的关系。

#### Split by tokens

模型按 token 限制输入时，应使用匹配的 tokenizer 检查。分块满足 embedding 模型上限还不够：问题、提示词、多个片段、消息格式以及预留回答，加起来也要满足生成模型的预算。两种模型的 tokenizer 不一定相同。

分词本身的算法和计数差异，统一见[Tokenization](/posts/2026-05-16-llm-tokenization-guide/)。这里关心的是如何用计数结果约束应用输入，不再重复 BPE 的推导。

#### Semantic Chunking

旧文还尝试了语义分块：比较相邻句组的表示，在语义变化较大处切开。它可能改善语义完整性，也会增加 embedding 成本；仍需处理过长片段，不能取代长度校验。

原实验组件里的百分位数、标准差、四分位距等方法，是决定「变化多大才切开」的不同阈值策略。不要把某个旧版本的默认分位数当成文本规律。实际比较时固定文档和测试问题，观察正确片段是否更容易命中，而不只是看分出来的块是否漂亮。

### 2.3 Embedding

检索 embedding 把一段文本变成向量，用于比较或搜索。它通常包含对整段输入的编码处理，**不能简单理解成一次 token embedding 矩阵查表**。同维度的向量也不自动可比，查询和文档表示应使用兼容的模型与约定。

LangChain 区分文档批量编码与查询编码，例如 `embed_documents` 和 `embed_query`；某些模型对两类输入的处理不同。换模型或表示方式后，需要重新检查索引兼容性与检索效果。[官方 embedding 集成说明](https://docs.langchain.com/oss/python/integrations/embeddings/)提供了接口边界。

#### Caching

`CacheBackedEmbeddings` 缓存的是**文本到向量的结果**，不是 Transformer 每一层的 KV Cache，也不是逐 token 查表结果。相同文本再次编码时可以复用已存向量；缓存 namespace 应能区分模型及会改变结果的配置。

文档 embedding 缓存和查询 embedding 缓存还需要分别考虑。查询是否缓存取决于配置，不能因为包装了一层缓存就假定所有请求都会命中。[官方缓存实现](https://github.com/langchain-ai/langchain/blob/master/libs/langchain/langchain_classic/embeddings/cache.py)展示了这一区别。模型推理时的 KV Cache 则见[推理中的缓存与执行](/posts/2026-06-13-llm-mem-opt-2-inference/#cache-computation)。

### 2.4 Vector stores

Vector store 把向量与原始内容、metadata 关联起来，并提供搜索等操作。选择它时，不只看「能存多少向量」，还要看过滤、更新删除、索引持久化和故障恢复是否满足应用要求。

几个容易遗漏的细节：

1. 距离分数可能是越大越近，也可能越小越近；跨库阈值不能直接搬用
2. 文档更新后要同步索引，否则回答会继续引用旧内容
3. 在多用户应用中，必须在资料进入模型上下文前完成访问控制
4. 删除原文件不一定自动删除索引、缓存和日志中的副本

原文的 FAISS 与 Qdrant 示例都遵循「加载 → 切块 → embedding → 搜索」流程。理解其中一种实现后，再按需求查具体集成即可。

#### Asynchronous operations

异步接口有助于处理多个等待网络或磁盘的操作，但不意味着无限并发。设置批量大小、超时、并发上限，并处理供应商限流；不要把一次本机测速写成所有 embedding 请求都能达到的加速倍数。

### 2.5 Retrievers

Retriever 的职责可以概括为「接收查询，返回相关文档」。Vector store 可以包装成 retriever，但 retriever 也可以用全文搜索、数据库或其他来源；高级 retriever 不一定全部建立在同一个 vector-store wrapper 上。[官方 retriever 列表](https://docs.langchain.com/oss/python/integrations/retrievers)能看到这种多样性。

#### Vector store-backed retriever

从最简单的 top-k 开始：先看召回结果，再调整 k、metadata 过滤或分数阈值。k 大了可能找到更多证据，也可能挤掉真正重要的片段。结果重复时可尝试 MMR，其相关性与多样性的取舍见前文。

#### MultiQueryRetriever

当用户表述与文档差距大时，可以生成多个查询视角，分别搜索再去重合并。例如「重启后怎么没反应」可以补充「服务启动」「网关日志」等查询。但改写可能偏离原问题，也会增加调用次数；保留原问题并检查新查询是否引入了用户没有提供的条件。

它适合解决「问法不同导致漏检」，不能替代权限过滤，更不能保证生成的每个查询都正确。旧 `MultiQueryRetriever` 类属于版本敏感实现，迁移时注意 `langchain-classic` 的包边界。

#### Contextual Compression Retriever

检索后还可以压缩上下文。要区分两种操作：

- **过滤、去重、重排**：保留或调整原文片段，不一定改写其文字
- **抽取、摘要**：产生更短的文本，可能遗漏限制条件，需要继续关联原始出处

旧文把两者都装进 compression pipeline，容易误以为「压缩」就是一种固定算法。更清楚的顺序是：先召回 → 检查权限 → 去重与筛选 → 必要时抽取 → 组装上下文。比如检索到整章部署手册，问题只问配置位置，优先保留对应段落及章节标识，而不是把整章丢给生成模型。

### 2.6 怎样知道是检索错了，还是回答错了

用一小组能人工核对的问题检查中间结果。每题记录应该引用的文档、实际召回片段、最终使用的上下文和答案：

| 观察到的问题 | 优先检查 |
| --- | --- |
| 正确文档根本没进库 | loader、更新流程、访问权限 |
| 文档在库里，但正确片段没召回 | 分块、查询表达、检索方法与 k |
| 召回正确，组装后却消失 | 去重、压缩、排序与 token 预算 |
| 证据已经给到，回答仍编造 | 指令、模型行为、证据引用与输出校验 |

至少加入「库里没有答案」与「旧版和新版文档冲突」两类题。只展示一次成功问答，无法判断一个 RAG 应用是否可靠。

## 3. Tools

Tool 是应用提供的一项可执行能力，例如搜索文档、计算数值或查询服务状态。模型通常生成工具名称和参数，**应用执行函数**，再把结果交回模型。模型声称「我执行了」不是执行成功的证据。

### Basic Usage Tutorial

一个工具至少要讲清楚名称、用途、输入结构和实际实现。名称与说明影响模型选择，参数 schema 帮助约束输入；它们都不能代替业务权限验证。工具返回结果也应明确区分成功、无结果和失败。[LangChain 工具文档](https://docs.langchain.com/oss/python/langchain/tools)说明了这些构造入口。

### Custom Tools

#### decorator

简单函数可以用装饰器暴露为工具。以下例子只做本地字符串处理，便于观察接口，不涉及网络或文件写入：

```python
from langchain.tools import tool

@tool
def normalize_issue_key(value: str) -> str:
    """把非空的问题编号去掉首尾空白并转成大写，不查询问题内容。"""
    result = value.strip().upper()
    if not result:
        raise ValueError("问题编号不能为空")
    return result

result = normalize_issue_key.invoke({"value": " docs-42 "})
```

输入 schema 能检查参数形状，但「这个问题是否属于当前用户」之类的规则，仍需由工具服务检查。密钥、身份和权限上下文由应用管理，不应让模型通过可编辑参数自行声明。

<a id="subclassing--basetool-recommend"></a>

#### Subclassing BaseTool

当工具需要维护客户端、定制同步/异步行为或复用复杂实现时，可以继承 `BaseTool`。旧文以 ArXiv 工具为例，其有价值的分层是：工具层提供名称和输入约定；底层客户端负责查询、超时与返回值处理。

继承并不一定比装饰器更合适。先用最简单、可独立测试的函数，确有需要再增加抽象层。

<a id="structuredtool-dataclass"></a>

#### StructuredTool

[`StructuredTool.from_function`](https://github.com/langchain-ai/langchain/blob/master/libs/core/langchain_core/tools/structured.py) 用于把已有函数包装成带结构化参数的工具。它与装饰器、继承方式解决的是相近的问题，不是三个必须按顺序掌握的阶段。旧标题中的「dataclass」也不应当成所有版本的实现保证。

#### Handling Tool Errors

工具失败后，不要把错误文本伪装成正常数据。应用需要决定：

- 输入有问题：给出可修正的信息，而不是无限重试
- 短暂超时或限流：按预算进行有限重试
- 权限不足：停止越权步骤，说明需要什么权限
- 有副作用的操作超时：先核对是否已经执行，避免重复提交

原文的 `ToolException` / `handle_tool_error` 属于工具对象的错误处理方式；LangChain v1 的 Agent 还提供 middleware 拦截工具调用。不要把旧示例直接混入新 Agent 配置，参见[迁移指南的工具错误处理部分](https://docs.langchain.com/oss/python/migrate/langchain-v1#handling-tool-errors)。无论使用哪种接口，都应保留可供排错的内部日志，同时避免把凭证或完整私有内容写进用户可见错误。

## 4. Agent

如果应用明确规定「检索一次，然后回答」，这是固定工作流。当模型要根据前一步结果选择下一步工具，并在多轮中推进任务时，才需要讨论 Agent 的控制循环。Agent 不只是一个 LLM，还包括工具执行、状态、停止条件和权限控制。[官方 Agent 文档](https://docs.langchain.com/oss/python/langchain/agents)把模型与这些外围机制放在一起解释。

### 4.1 把循环写清楚，再选框架

以「检查部署说明与当前服务状态是否一致」为例，可以先画出控制逻辑：

```text
接收用户任务，建立本次状态与预算
循环：
    把允许使用的资料和工具交给模型
    如果模型给出最终答案：检查证据后返回
    否则：验证工具名、参数与权限
    需要审批时暂停，等待用户决定
    执行允许的工具，把真实结果加入状态
    若超过时间、轮次或费用预算：停止并说明未完成部分
```

这只是应用层伪代码，不是可直接部署的 Agent。特别是「验证权限」和「需要审批时暂停」必须在执行前发生，不能寄希望于系统提示词阻止所有危险调用。

LangChain 当前的 `create_agent` 提供模型、工具、提示及 middleware 的组合入口，并基于 LangGraph 实现运行机制。若任务的分支和顺序本来就确定，普通函数或显式工作流可能更容易验证，不必先增加一个自主决策循环。[框架概览](https://docs.langchain.com/oss/python/langchain/overview)

### 4.2 状态、记忆与缓存别混在一起

- **对话状态**：本次任务的消息、工具结果、进行到哪一步
- **长期记忆**：应用选择持久保存、以后再取回的事实或偏好，需要更新和删除规则
- **检索索引 / embedding 缓存**：便于找到资料或避免重复编码
- **模型 KV Cache**：推理引擎保存的中间张量，用来减少重复计算

它们保存的东西和生命周期都不同。把聊天历史全留着不等于做好记忆管理；有了长期记忆，也不能据此推断模型参数已经学习了这些内容。

### 4.3 从能演示到能使用

一个可用的 Agent 至少应检查：

1. 工具范围是否最小化，读操作与写操作是否分开
2. 每次执行有没有真实的状态、结果和可追踪标识
3. 重启后是否会重复提交之前的写操作
4. 工具返回恶意文字时，能否守住权限边界
5. 是否能主动停止，并准确说明已经完成与尚未完成的部分

常驻运行、消息网关、Windows 与 Linux 之间的恢复，则是部署层的问题，放在[Hermes Agent 的 WSL 部署案例](/posts/2026-05-15-hermes-agent-wsl-setup/)里展开。Hermes 是另一个应用实现；阅读该案例不要求先把它当作 LangChain 的插件。

## Reference

- [LangChain v1 迁移指南](https://docs.langchain.com/oss/python/migrate/langchain-v1)：包结构、旧组件与工具错误处理迁移
- [LangChain 检索概览](https://docs.langchain.com/oss/python/deepagents/retrieval)：知识库、检索流程与不同 RAG 控制方式
- [LangChain 文档与检索集成](https://docs.langchain.com/oss/python/integrations/retrievers)：按需求选择实现，而不是复制过时导入路径
- [LangChain Agent 文档](https://docs.langchain.com/oss/python/langchain/agents)：模型、工具、状态与运行循环

以上链接按 2026-10-07 的官方资料核对。API 可能继续变化，概念边界与中间结果的检查方法比记住一个类的完整参数表更重要。
