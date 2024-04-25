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

假设在Prompt里边给了很多例子, 希望用户输入的时候, 系统能够自动选择合适的例子, 然后和用户输入一起给LLM. [API Reference](https://python.langchain.com/docs/modules/model_io/prompts/example_selectors/)

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
dynamic_prompt = FewShotPromptTemplate(
    # We provide an ExampleSelector instead of examples.
    example_selector=example_selector,
    example_prompt=example_prompt,
    # 下边是用来 format 最终的 prompt
    prefix="Give the antonym of every input",
    suffix="Input: {adjective}\nOutput:",
    input_variables=["adjective"],
)

```
输入:
```python
print(dynamic_prompt.format(adjective="big"))
```

输出:

```plaintext
Give the antonym of every input

Input: happy
Output: sad

Input: tall
Output: short

Input: energetic
Output: lethargic

Input: sunny
Output: gloomy

Input: windy
Output: calm

Input: big
Output:

```

输入:
```python
long_string = "big and huge and massive and large and gigantic and tall and much much much much much bigger than everything else"
print(dynamic_prompt.format(adjective=long_string))
```

输出:

```plaintext
# 可以看到只给了一个example

Give the antonym of every input

Input: happy
Output: sad

Input: big and huge and massive and large and gigantic and tall and much much much much much bigger than everything else
Output:

```
#### Select by maximal marginal relevance (MMR)

参考paper: https://arxiv.org/pdf/2211.13892.pdf

这个 selector 的思想是, 选择与当前输入 p 相似的(cosine similarity) example $q_j$ , 但是这个 $q_j$ 还要尽量与 example pool 中 $q_i$ 不要太相似, 这是为了多样性. paper 中的 next example 选择公式为:

![image.png](https://s2.loli.net/2024/04/25/2bHXmLKFRyAkxl5.png)

可以看到, 如果下一个 $q_j$ 和 当前的 $p$ 很相似, 但是和其他的 $q_i$ 也非常相似, 那么这个分数也不会太高.

例子

```python

from langchain_community.vectorstores import FAISS
from langchain_core.example_selectors import MaxMarginalRelevanceExampleSelector
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate
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
mmr_prompt = FewShotPromptTemplate(
    # We provide an ExampleSelector instead of examples.
    example_selector=example_selector,
    example_prompt=example_prompt,
    # 用于 format 最终的 prompt
    prefix="Give the antonym of every input",
    suffix="Input: {adjective}\nOutput:",
    input_variables=["adjective"],
)
# Input is a feeling, so should select the happy/sad example as the first one
print(mmr_prompt.format(adjective="worried"))

```
输出:

``` plaintext
Give the antonym of every input

Input: happy
Output: sad

Input: windy
Output: calm

Input: worried
Output:

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
similar_prompt = FewShotPromptTemplate(
    # We provide an ExampleSelector instead of examples.
    example_selector=example_selector,
    example_prompt=example_prompt,
    prefix="Give the antonym of every input",
    suffix="Input: {adjective}\nOutput:",
    input_variables=["adjective"],
)
# Input is a measurement, so should select the tall/short example
print(similar_prompt.format(adjective="large"))

```
输出

```plaintext
Give the antonym of every input

Input: tall
Output: short

Input: large
Output:
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
dynamic_prompt = FewShotPromptTemplate(
    # We provide an ExampleSelector instead of examples.
    example_selector=example_selector,
    example_prompt=example_prompt,
    prefix="Give the Spanish translation of every input",
    suffix="Input: {sentence}\nOutput:",
    input_variables=["sentence"],
)
# An example input with large ngram overlap with "Spot can run."
# and no overlap with "My dog barks."
print(dynamic_prompt.format(sentence="Spot can run fast."))
```

输出 :

```plaintext
Give the Spanish translation of every input

Input: Spot can run.
Output: Spot puede correr.

Input: See Spot run.
Output: Ver correr a Spot.

Input: Spot plays fetch.
Output: Spot juega a buscar.

Input: My dog barks.
Output: Mi perro ladra.

Input: Spot can run fast.
Output:

```

输入:

```python

example_selector.threshold = 0.0
print(dynamic_prompt.format(sentence="Spot can run fast."))

```
输出:
```plaintext
# Since "My dog barks." has no ngram overlaps with "Spot can run fast."
# it is excluded.

Give the Spanish translation of every input

Input: Spot can run.
Output: Spot puede correr.

Input: See Spot run.
Output: Ver correr a Spot.

Input: Spot plays fetch.
Output: Spot juega a buscar.

Input: Spot can run fast.
Output:
```

### 1.4 Few-shot prompt templates

FewShotPromptTemplate + example selector 
实现的功能就是, 首先我们有一堆 example (通常是成对儿的输入和输出) , 然后可以自适应的, 根据不同的输入, 能够自动的 select 合适的 example 与 输入合并, 一同变为 LLM 的 prompt.

也就是说, 不同的输入, 会产生不同的 prompt , 我理解是和 Retrieval-augmented generation (RAG) 类似的效果. 

只有 FewShotPromptTemplate:

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

print(example_prompt.format(**examples[0]))

# 构造一个 Template 
prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    # format Template
    suffix="Question: {input}",
    input_variables=["input"],
)

# 当前, 还是所有的 example 都能输出 , 省略
print(prompt.format(input="Who was the father of Mary Ball Washington?"))
```

整个 example selector:

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

# Select the most similar example to the input.
question = "Who was the father of Mary Ball Washington?"
selected_examples = example_selector.select_examples({"question": question})
print(f"Examples most similar to the input: {question}")
for example in selected_examples:
    print("\n")
    for k, v in example.items():
        print(f"{k}: {v}")
```

FewShotPromptTemplate + selector : 

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

```python
You are impersonating Elon Musk.

Here's an example of an interaction:

Q: What's your favorite car?
A: Tesla

Now, do this for real!

Q: What's your favorite social media site?
A:

```


## 




## Reference








