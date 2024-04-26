---
title: Lang Chain Notes
date: 2024-04-24 19:0000 +0800
categories: [ Deep Learning ,  LLM,  LangChain]
tags: [deep learning , llm ,  langchain]     # TAG names should always be lowercase
math: true
---


## 0. 前言


本篇Blog不是教程,  官方教程[1](https://python.langchain.com/docs/modules/)、[2](https://python.langchain.com/docs/get_started/introduction) 有时候比较抽象,  这里只是在学习LangChain过程中做个记录,  并加入自己的理解和注释. 当然这里也不进行过多的介绍, 直接进入对组件的学习.



## 1. Prompts

Prompts 通常用来"调教"LLM,  比如用来指定LLM的输出格式, 也可以用来给LMM一些例子让他参考等等.LangChain 目前提供了 4 种 Prompts template ,  方便用户构造Prompt.

### 1.1 PromptTemplate

PromptTemplate 是最简单,  最基本的一种 Template. API Reference:[PromptTemplate](https://api.python.langchain.com/en/latest/prompts/langchain_core.prompts.prompt.PromptTemplate.html)

官方给了2种方法来用PromptTemplate.

- 方法一(推荐)

```python
from langchain_core.prompts import PromptTemplate

# Instantiation using from_template (recommended)
prompt = PromptTemplate.from_template("Say {foo}")
prompt.format(foo="bar")
```

结果 : `Say bar`

- 方法二

```python
from langchain_core.prompts import PromptTemplate

# Instantiation using initializer
prompt = PromptTemplate(input_variables=["foo"],  template="Say {foo}")
prompt.format(foo="bar")

```

结果 : `Say bar`

### 1.2 ChatPromptTemplate

 ChatPromptTemplate 通常有 3 种规则: "system",  "ai" and "human". API Reference:[ChatPromptTemplate](https://api.python.langchain.com/en/latest/prompts/langchain_core.prompts.chat.ChatPromptTemplate.html)


```python

from langchain_core.prompts import ChatPromptTemplate

chat_template = ChatPromptTemplate.from_messages(
    [
        ("system",  "You are a helpful AI bot. Your name is {name}."), 
        ("human",  "Hello,  how are you doing?"), 
        ("ai",  "I'm doing well,  thanks!"), 
        ("human",  "{user_input}"), 
    ]
)

messages = chat_template.format_messages(name="Bob",  user_input="What is your name?")
```
输出:

```python

[SystemMessage(content='You are a helpful AI bot. Your name is Bob.'), 
 HumanMessage(content='Hello,  how are you doing?'), 
 AIMessage(content="I'm doing well,  thanks!"), 
 HumanMessage(content='What is your name?')]
```

可以看到, 实际是产生了 `SystemMessage`,  `HumanMessage` and  `AIMessage` 共 3 种 Message. 与之对应的,  在构造这些Message时,  可以使用相应的 Template : `AIMessagePromptTemplate`, `SystemMessagePromptTemplate` and `HumanMessagePromptTemplate`

例子:

```python
from langchain_core.prompts import HumanMessagePromptTemplate, SystemMessagePromptTemplate, AIMessagePromptTemplate

chat_template = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template("你是一个{llm_type}"),  # 使用 template构造 系统的 message
        ('ai', "很高兴帮助您."),  # 直接使用 role 构造 AI的 message
        HumanMessagePromptTemplate.from_template("{text}"), 
    ]
)
messages = chat_template.format_messages(llm_type="AI助手", text = "1 + 1 = ?")
print(messages)

```

输出:

```python
[SystemMessage(content='你是一个AI助手'),  
AIMessage(content='很高兴帮助您.'), 
 HumanMessage(content='1 + 1 = ?')]
```


### 1.3 Example selectors

假设有一些例子, 我们希望 LLM 能够根据输入挑一些合适的例子出来, 方便我们后续的操作, 比如将他们放到一个 prompt 中. [API Reference](https://python.langchain.com/docs/modules/model_io/prompts/example_selectors/)

####  Select by length

简单来说, 这个selector 是根据用户输入的语句长短选择合适的example, 你输入的越短,他给的例子越多, 你输出的越长,他给的例子越少.

例子:

```python
from langchain_core.example_selectors import LengthBasedExampleSelector
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate

# Examples of a pretend task of creating antonyms.
examples = [
    {"input": "happy", "output": "sad"},
    {"input": "tall", "output": "short"},
    {"input": "energetic", "output": "lethargic"},
    {"input": "sunny", "output": "gloomy"},
    {"input": "windy", "output": "calm"},
]

# 用于将 example format
example_prompt = PromptTemplate(
    input_variables=["input", "output"],
    template="Input: {input}\nOutput: {output}",
)
example_selector = LengthBasedExampleSelector(
    examples=examples,
    example_prompt=example_prompt,
    max_length=25,
    # The function used to get the length of a string, which is used
    # to determine which examples to include. It is commented out because
    # it is provided as a default value if none is specified.
    # get_text_length: Callable[[str], int] = lambda x: len(re.split("\n| ", x))
)

```
输入:
```python
example_selector.select_examples({"input": "okay"})
```

输出:

```plaintext
[{'input': 'happy', 'output': 'sad'},
 {'input': 'tall', 'output': 'short'},
 {'input': 'energetic', 'output': 'lethargic'},
 {'input': 'sunny', 'output': 'gloomy'},
 {'input': 'windy', 'output': 'calm'}]

```

输入:
```python
long_string = "big and huge and massive and large and gigantic and tall and much much much much much bigger than everything else"
example_selector.select_examples({"input": long_string})
```

输出:

```plaintext
# 可以看到只给了一个example
[{'input': 'happy', 'output': 'sad'}]
```
#### Select by maximal marginal relevance (MMR)

这个 selector 的思想是, 选择与当前输入 p 相似的(cosine similarity) example $q_j$ , 但是这个 $q_j$ 还要尽量与 example pool 中 $q_i$ 不要太相似, 这是为了多样性. [原始paper](https://arxiv.org/pdf/2211.13892.pdf) 中的 next example 选择公式为:

![image.png](https://s2.loli.net/2024/04/25/2bHXmLKFRyAkxl5.png)

可以看到, 如果下一个 $q_j$ 和 当前的 $p$ 很相似, 但是和其他的 $q_i$ 也非常相似, 那么这个分数也不会太高. 

例子

```python

from langchain_core.example_selectors import MaxMarginalRelevanceExampleSelector
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings

# 用于 format examples pool 
example_prompt = PromptTemplate(
    input_variables=["input", "output"],
    template="Input: {input}\nOutput: {output}",
)

# Examples of a pretend task of creating antonyms.
examples = [
    {"input": "happy", "output": "sad"},
    {"input": "tall", "output": "short"},
    {"input": "energetic", "output": "lethargic"},
    {"input": "sunny", "output": "gloomy"},
    {"input": "windy", "output": "calm"},
]

example_selector = MaxMarginalRelevanceExampleSelector.from_examples(
    # The list of examples available to select from.
    examples,
    # The embedding class used to produce embeddings which are used to measure semantic similarity.
    OpenAIEmbeddings(),
    # The VectorStore class that is used to store the embeddings and do a similarity search over.
    FAISS,
    # The number of examples to produce.
    k=2,
)

```
输出:

``` plaintext
[{'input': 'happy', 'output': 'sad'},
 {'input': 'windy', 'output': 'calm'}]
```

#### Select by similarity

这个就很单纯了, 直接使用 cos similarity 来选择最佳的example, 典型 selector 是 SemanticSimilarityExampleSelector.

例子
```python

from langchain_chroma import Chroma
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate
from langchain_openai import OpenAIEmbeddings

example_prompt = PromptTemplate(
    input_variables=["input", "output"],
    template="Input: {input}\nOutput: {output}",
)

# Examples of a pretend task of creating antonyms.
examples = [
    {"input": "happy", "output": "sad"},
    {"input": "tall", "output": "short"},
    {"input": "energetic", "output": "lethargic"},
    {"input": "sunny", "output": "gloomy"},
    {"input": "windy", "output": "calm"},
]

example_selector = SemanticSimilarityExampleSelector.from_examples(
    # The list of examples available to select from.
    examples,
    # The embedding class used to produce embeddings which are used to measure semantic similarity.
    OpenAIEmbeddings(),
    # The VectorStore class that is used to store the embeddings and do a similarity search over.
    Chroma,
    # The number of examples to produce.
    k=1,
)

example_selector.select_examples({"adjective": "large"})

```
输出

```plaintext
[{'input': 'tall', 'output': 'short'}]
```
#### Select by n-gram overlap

这个是计算输入的 query 和 example 的 similarity(0-1之间), 然后根据给定的阈值, 给出满足条件的example.

阈值为 0.0 表示只排除不相关的 example. 阈值为 -1.0, 表示所有的 example 都会返回. 大于 1.0 表示不返回 example, 默认为 -1.0

例子

```python
from langchain_community.example_selector.ngram_overlap import (
    NGramOverlapExampleSelector,
)
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate

example_prompt = PromptTemplate(
    input_variables=["input", "output"],
    template="Input: {input}\nOutput: {output}",
)

# Examples of a fictional translation task.
examples = [
    {"input": "See Spot run.", "output": "Ver correr a Spot."},
    {"input": "My dog barks.", "output": "Mi perro ladra."},
    {"input": "Spot can run.", "output": "Spot puede correr."},
]
example_selector = NGramOverlapExampleSelector(
    # The examples it has available to choose from.
    examples=examples,
    # The PromptTemplate being used to format the examples.
    example_prompt=example_prompt,
    # The threshold, at which selector stops.
    # It is set to -1.0 by default.
    threshold=-1.0,
    # For negative threshold:
    # Selector sorts examples by ngram overlap score, and excludes none.
    # For threshold greater than 1.0:
    # Selector excludes all examples, and returns an empty list.
    # For threshold equal to 0.0:
    # Selector sorts examples by ngram overlap score,
    # and excludes those with no ngram overlap with input.
)

# 虽然  "My dog barks." 与 输入不相关, 但是还是输出了 
example_selector.select_examples({"sentence": "Spot can run fast."})

```

输出 :

```plaintext
[{'input': 'Spot can run.', 'output': 'Spot puede correr.'},
 {'input': 'See Spot run.', 'output': 'Ver correr a Spot.'},
 {'input': 'My dog barks.', 'output': 'Mi perro ladra.'}]
```

输入:

```python
# 不输出 不相关的 
example_selector.threshold = 0.0
print(dynamic_prompt.format(sentence="Spot can run fast."))

```
输出:
```plaintext
[{'input': 'Spot can run.', 'output': 'Spot puede correr.'},
 {'input': 'See Spot run.', 'output': 'Ver correr a Spot.'}]
```

### 1.3 Few-shot prompt templates

上边介绍了一些 example selector , 现在介绍 
FewShotPromptTemplate + example selector .
二者实现的功能就是, 首先我们有一堆 example (通常是成对儿的输入和输出) , 然后可以自适应的, 根据不同的输入, 能够自动的 select 合适的 example 与 输入合并, 一同变为 LLM 的 prompt.

也就是说, 不同的输入, 会产生不同的 prompt , 我理解是和 Retrieval-augmented generation (RAG) 类似的效果. 

define example:

```python
from langchain_core.prompts.few_shot import FewShotPromptTemplate
from langchain_core.prompts.prompt import PromptTemplate

examples = [
    {
        "question": "Who lived longer, Muhammad Ali or Alan Turing?",
        "answer": """
Are follow up questions needed here: Yes.
Follow up: How old was Muhammad Ali when he died?
Intermediate answer: Muhammad Ali was 74 years old when he died.
Follow up: How old was Alan Turing when he died?
Intermediate answer: Alan Turing was 41 years old when he died.
So the final answer is: Muhammad Ali
""",
    },
    {
        "question": "When was the founder of craigslist born?",
        "answer": """
Are follow up questions needed here: Yes.
Follow up: Who was the founder of craigslist?
Intermediate answer: Craigslist was founded by Craig Newmark.
Follow up: When was Craig Newmark born?
Intermediate answer: Craig Newmark was born on December 6, 1952.
So the final answer is: December 6, 1952
""",
    },
    {
        "question": "Who was the maternal grandfather of George Washington?",
        "answer": """
Are follow up questions needed here: Yes.
Follow up: Who was the mother of George Washington?
Intermediate answer: The mother of George Washington was Mary Ball Washington.
Follow up: Who was the father of Mary Ball Washington?
Intermediate answer: The father of Mary Ball Washington was Joseph Ball.
So the final answer is: Joseph Ball
""",
    },
    {
        "question": "Are both the directors of Jaws and Casino Royale from the same country?",
        "answer": """
Are follow up questions needed here: Yes.
Follow up: Who is the director of Jaws?
Intermediate Answer: The director of Jaws is Steven Spielberg.
Follow up: Where is Steven Spielberg from?
Intermediate Answer: The United States.
Follow up: Who is the director of Casino Royale?
Intermediate Answer: The director of Casino Royale is Martin Campbell.
Follow up: Where is Martin Campbell from?
Intermediate Answer: New Zealand.
So the final answer is: No
""",
    },
]

# format example pool
example_prompt = PromptTemplate(
    input_variables=["question", "answer"], template="Question: {question}\n{answer}"
)

```

define selector:

```python
from langchain_chroma import Chroma
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_openai import OpenAIEmbeddings

example_selector = SemanticSimilarityExampleSelector.from_examples(
    # This is the list of examples available to select from.
    examples,
    # This is the embedding class used to produce embeddings which are used to measure semantic similarity.
    OpenAIEmbeddings(),
    # This is the VectorStore class that is used to store the embeddings and do a similarity search over.
    Chroma,
    # This is the number of examples to produce.
    k=1,
)

```

FewShotPromptTemplate + example + selector : 

```python
prompt = FewShotPromptTemplate(
    example_selector=example_selector,
    example_prompt=example_prompt,
    suffix="Question: {input}",
    input_variables=["input"],
)

print(prompt.format(input="Who was the father of Mary Ball Washington?"))
```

输出:

```plaintext
# 可以看到 prompt 最后只引入了一个 example

Question: Who was the maternal grandfather of George Washington?

Are follow up questions needed here: Yes.
Follow up: Who was the mother of George Washington?
Intermediate answer: The mother of George Washington was Mary Ball Washington.
Follow up: Who was the father of Mary Ball Washington?
Intermediate answer: The father of Mary Ball Washington was Joseph Ball.
So the final answer is: Joseph Ball


Question: Who was the father of Mary Ball Washington?
```

### 1.5 Partial prompt templates

这个功能就是类似函数的参数具有默认值. 

例子:

```python
from langchain_core.prompts import PromptTemplate

prompt = PromptTemplate.from_template("{foo} {bar}")
partial_prompt = prompt.partial(foo="666")
print(partial_prompt.format(bar="baz")) # 输出 666 baz
```

这个也可以使用函数:

```python
from datetime import datetime

def _get_datetime():
    now = datetime.now()
    return now.strftime("%m/%d/%Y, %H:%M:%S")

prompt = PromptTemplate(
    template="Tell me a {adjective} joke about the day {date}",
    input_variables=["adjective", "date"],
)
# date 默认 调用当前时间
partial_prompt = prompt.partial(date=_get_datetime)
# 输出 : Tell me a funny joke about the day 04/25/2024, 23:22:18
print(partial_prompt.format(adjective="funny"))

```

### 1.6 PipelinePrompt

PipelinePrompt 能够把多个 prompt 整到一起. 

```python
from langchain_core.prompts.pipeline import PipelinePromptTemplate
from langchain_core.prompts.prompt import PromptTemplate

# 最终的 prompt
full_template = """{introduction}

{example}

{start}"""
full_prompt = PromptTemplate.from_template(full_template)

# 用于 Introduction
introduction_template = """You are impersonating {person}."""
introduction_prompt = PromptTemplate.from_template(introduction_template)

# example 
example_template = """Here's an example of an interaction:

Q: {example_q}
A: {example_a}"""
example_prompt = PromptTemplate.from_template(example_template)

# 用户的输入
start_template = """Now, do this for real!

Q: {input}
A:"""
start_prompt = PromptTemplate.from_template(start_template)

# 将上边的 prompt 合并到一起
input_prompts = [
    ("introduction", introduction_prompt),
    ("example", example_prompt),
    ("start", start_prompt),
]
# 指定谁是谁
pipeline_prompt = PipelinePromptTemplate(
    final_prompt=full_prompt, pipeline_prompts=input_prompts
)
print(
    pipeline_prompt.format(
        person="Elon Musk",
        example_q="What's your favorite car?",
        example_a="Tesla",
        input="What's your favorite social media site?",
    )
)
```

输出 :

```plaintext

You are impersonating Elon Musk.

Here's an example of an interaction:

Q: What's your favorite car?
A: Tesla

Now, do this for real!

Q: What's your favorite social media site?
A:

```


## 2. Retrieval

Retrieval Augmented Generation (RAG) 可能是目前 LLM 发挥比较大作用的一个应用. 其核心思想是利用外挂的知识库赋予在不同的垂直领域应用能力. 

其核心流程如下:

[1] 首先我们要有相应的资源库, source

[2] 然后针对不同的资源, 我们使用相应的 dataloader 将资源数据读取

[3] 由于资源文档比较长, 通常我们要进行分块, 称为 chunk

[4] 将文档chunk后, 会对每个 chunk 进行 embedding

[5] embedding 之后, 要进行 store, 这个组件一般称为 vector store

[6] 当我们给定输入的时候, LLM 能够根据语义从 store 中抽取有用的资源,这个过程就是 retrieve

![image.png](https://s2.loli.net/2024/04/26/IiHL1MVWNc8QJtG.png)

各种资源就不介绍了, 我们学习不同资源对应的 dataloader

### 2.1 Dataloader

[官方文档](https://python.langchain.com/docs/integrations/document_loaders/)集成了很多第三方 dataloader, 
甚至可以直接从arxiv、GitHub等直接获取数据. 但是常用的可能就是针对 文本 和 csv 的, 而且使用方法类似, 所以这里只学习 文本类型 的.

#### Document Loader

LangChain 给的例子是继承 BaseLoader, 然后将读到的文本初始化为 Document 对象.
内部有 4 个基本方法: 直接读取所有, 异步读取所有, lazy 读取, 异步lazy读取.

![image.png](https://s2.loli.net/2024/04/26/PC8fvdnsaHA9KB6.png){: width="400" height="300" }


例子:

<details markdown="1">
<summary> 详细信息 </summary>

```python
from typing import AsyncIterator, Iterator
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document

class CustomDocumentLoader(BaseLoader):
    """An example document loader that reads a file line by line."""

    def __init__(self, file_path: str) -> None:
        """Initialize the loader with a file path.

        Args:
            file_path: The path to the file to load.
        """
        self.file_path = file_path

    def lazy_load(self) -> Iterator[Document]:  # <-- Does not take any arguments
        """A lazy loader that reads a file line by line.

        When you're implementing lazy load methods, you should use a generator
        to yield documents one by one.
        """
        with open(self.file_path, encoding="utf-8") as f:
            line_number = 0
            for line in f:
                yield Document(
                    page_content=line,
                    metadata={"line_number": line_number, "source": self.file_path},
                )
                line_number += 1

    # alazy_load is OPTIONAL.
    # If you leave out the implementation, a default implementation which delegates to lazy_load will be used!
    async def alazy_load(
        self,
    ) -> AsyncIterator[Document]:  # <-- Does not take any arguments
        """An async lazy loader that reads a file line by line."""
        # Requires aiofiles
        # Install with `pip install aiofiles`
        # https://github.com/Tinche/aiofiles
        import aiofiles

        async with aiofiles.open(self.file_path, encoding="utf-8") as f:
            line_number = 0
            async for line in f:
                yield Document(
                    page_content=line,
                    metadata={"line_number": line_number, "source": self.file_path},
                )
                line_number += 1

with open("./meow.txt", "w", encoding="utf-8") as f:
    quality_content = "meow meow🐱 \n meow meow🐱 \n meow😻😻"
    f.write(quality_content)

loader = CustomDocumentLoader("./meow.txt")

## Test out the lazy load interface
for doc in loader.lazy_load():
    print()
    print(type(doc))
    print(doc)

## Test out the async implementation
async for doc in loader.alazy_load():
    print()
    print(type(doc))
    print(doc)

"""
输出结果一样:

<class 'langchain_core.documents.base.Document'>
page_content='meow meow🐱 \n' metadata={'line_number': 0, 'source': './meow.txt'}

<class 'langchain_core.documents.base.Document'>
page_content=' meow meow🐱 \n' metadata={'line_number': 1, 'source': './meow.txt'}

<class 'langchain_core.documents.base.Document'>
page_content=' meow😻😻' metadata={'line_number': 2, 'source': './meow.txt'}
"""
```
</details>

> load() can be helpful in an interactive environment such as a jupyter notebook.
Avoid using it for production code since eager loading assumes that all the content can fit into memory, which is not always the case, especially for enterprise data.
{: .prompt-info }

### 2.2 Text Splitters

Once you've loaded documents, you'll often want to transform them to better suit your application. The simplest example is you may want to split a long document into smaller chunks that can fit into your model's context window.

[官方文档](https://python.langchain.com/docs/modules/data_connection/document_transformers/)给了多种 Splitter, 最常用的为以下 3 种.

#### Split by character

这个是最简单的 Splitter, 单纯就是使用指定的 character 去切分 text .

例子:

```python
from langchain_text_splitters import CharacterTextSplitter

text_splitter = CharacterTextSplitter(
    separator=" ", # 指定 separator
    chunk_size=5,
    chunk_overlap=2,
    length_function=len,
    is_separator_regex=False,
)

text_splitter.split_text("我是练习时长 达两年半的坤坤")

```
上述代码中, chunk_size 本意是指进行 split 之后, 后续进行store的时候, 最大以多大的size 作为一个整体进行存储. 但是可以看到这个参数对于 CharacterTextSplitter 不生效, 实际上[源码](https://api.python.langchain.com/en/latest/_modules/langchain_text_splitters/character.html#CharacterTextSplitter)中, 就是简单的用 re.split() 对文档按照 separator 进行切割, 不管子句有多长, 直接返回. 

输出:

```plaintext
['我是练习时长', '达两年半的坤坤']
```

> 因为 LLM 通常对输入有长度限制, 因此 CharacterTextSplitter 不太适合, 可能会超出输入尺寸范围, 而下边的 RecursiveCharacterTextSplitter 可以递归切割子句, 直到每个子句都小于 chunk size.
{: .prompt-info }


#### Recursive Splitter By Character

这个看了[源码](https://api.python.langchain.com/en/latest/_modules/langchain_text_splitters/character.html#RecursiveCharacterTextSplitter),它默认的 `separator = ["\n\n", "\n", " ", ""]`, 我的理解是说, 首先会根据`\n\n`进行切割. 一般来说, 2 个换行分割开的通常是 2 篇文章. 所以会先按照这个尺度进行切割. 如果切割之后, 某篇文章还是太长(大于 chunk size), 那么会继续使用 `\n` 进行划分切割, 同理直到其长度小于 chunk size.

例子:

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size=5,
    chunk_overlap=0,
    length_function=len,
    is_separator_regex=False,
    keep_separator = False
)
text_splitter.split_text(
    "我是\n\n练习时长达两年\n半的坤坤"
    )
```
输出:
```plaintext
['我是', '练习时长达', '两年', '半的坤坤']
# 可以看到, 首先用`\n\n`进行分割, 因为"我是"的长度小于5, 所以直接存起来, 
# 但是后边部分太长, 又基于`\n`进行切割, "半的坤坤"是满足要求的,所以一起存了起来.
# 但是"练习时长达两年"的长度还是大于5, 于是进行了继续的切割. 变为"练习时长达" 和 "两年"
```

#### Split by tokens

这个就是使用 NLP 中 token 进行切割, 不同的 tokenizer 有不同的切割方式. 举个例子, 加入一个单词算一个token, 那就按单词切割.

这里使用 OpenAI BPE tokenizer : tiktoken, 是[BPE 算法](https://huggingface.co/learn/nlp-course/chapter6/5)的一个实现.

Split by tokens 的使用方法基于上边 2 种 Splitter, 只是切割时调用方法不同:

```python
# pip install tiktoken
text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
    encoding="cl100k_base", chunk_size=100, chunk_overlap=0
) # CharacterTextSplitter 实际还是受 chunk_size 的约束
texts = text_splitter.split_text("text")

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    model_name="gpt-4",
    chunk_size=100,
    chunk_overlap=0,
) # 能够保证子句全部小于chunk_size
texts = text_splitter.split_text("text")
# 此外 encoding参数 和 model_name参数 效果类似, 具体请参考api
```


#### Semantic Chunking

这个就是字面意思, 基于 text 之间的语义进行切割, 使得语义相近的尽量在一个chunk, 但是这个目前(2024年4月27日)是个实验性功能. 参考[官方文档](https://python.langchain.com/docs/modules/data_connection/document_transformers/semantic-chunker/)

由于这个需要计算语义相似度, 所以需要进行 embedding. 

例子:

```python
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings
texts = text_splitter.split_text("text")
```

这个里边提供了一个 Breakpoints, 用于评估什么时候该切割, 语义相近多少才算相近?

- Percentile

Percentile(百分位数) 是默认的评估标准,  他是计算所有两两句子之间的difference, 如果大于阈值就给他切开.

例子:
```python
text_splitter = SemanticChunker(
    OpenAIEmbeddings(), breakpoint_threshold_type="percentile"
    # breakpoint_threshold_amount : 默认值 
)
```
阅读[源码](https://api.python.langchain.com/en/latest/_modules/langchain_experimental/text_splitter.html#SemanticChunker)可以看到,当`threshold_type = "percentile"` 时, 默认使用 95% 分位数. 表示当 2 个句子差异性大于所有距离的 95% 分位数时, 就进行切割. `breakpoint_threshold_amount` 参数控制分位数具体大小.

- Standard Deviation

用法类似, 不再赘述. 源码中当`threshold_type = "standard_deviation"` 时, 默认使用 `mean + 3 * std` 作为阈值. 表示当 2 个句子差异性大于阈值时, 就进行切割.
`breakpoint_threshold_amount` 参数控制标准差的倍数.
- Interquartile

使用的箱线图方法, 默认使用  `mean + 1.5 * iqr`, 其中 `iqr = q3 - q1`, q3 为 75% 分位数, q1 为 25% 分位数, 
`breakpoint_threshold_amount` 参数控制`q3-q1`的倍数.

## Reference








