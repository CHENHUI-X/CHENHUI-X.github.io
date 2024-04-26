---
title: Introduction to Causal Inference Series (I)
date: 2024-04-26 14:15:00 +0800
categories: [Statistical, Mathematics, Machine Learning , Causal Inference]
tags: [machine learning,  mathematics, statistics, causal inference]     # TAG names should always be lowercase
math: true
---


## 0. 前言

Causal Inference (因果推断) 已经在多个领域发挥出巨大作用, 尽管早已经听说过其大名, 但是从未步入这个领域好好学习一番, 通常是浅尝辄止. 为此在博客开一个系列, 一是用于记录学习, 二是希望能够起到监督作用...

由于是入门学习, 因此课程和书籍选择了相对简单的. 根据网上的推荐和实际体验, 感觉 Brady Neal 的系列介绍比较合适, 因此这个系列都将会以 Brady Neal 的课程为基础. 课程链接: [https://www.bradyneal.com/causal-inference-course](https://www.bradyneal.com/causal-inference-course)


另外, 由于这是个人学习笔记, 我作为初学者, 在博客中记录的内容和理解难免会有错误. 希望各位能够指正, 并请不吝赐教, 在下将不胜感激。

> 阅读前, 需要你 : 有高数基础知识, 线代基础知识, 当然还要有 ML和 DL 的知识背景. 
{: .prompt-info }



## 1. 第一章 Motivation: Why You Might Care

第一章主要介绍辛普森悖论, 以及向我们初步展示相关性和因果性的联系与区别.

### 1.1 Simpson’s Paradox 

通常因果推断的第一课都是 Simpson’s Paradox (辛普森悖论) . 它说了这么一件事 : 假设现在有种病, 我们有2中治疗方案, treatment A and treatment B. 在做实验的时候, treatment B  比较稀缺, 只有较少的志愿者可以用上 B, 比如 treatment A and treatment B 的志愿者分别为 73% 和 27% . 现在得到这么一组数据 :

表中, 百分比指的是接受相应的 treatment 后志愿者死亡率. Mild组 表示病的不重 , Severe组 表示病的比较严重. 

![image.png](https://s2.loli.net/2024/04/26/OvNDhwKrz68ajnu.png){: width="400" height="300" }

从上表可以看到, 无论是哪个分组, 明显 treatment B 死亡率更低. 但是有趣的是, 当你纵观所有人, 即 Total 列反而是 treatment A 死亡率更高. 那么到低哪个 treatment 更好呢? 

上表有个关键的问题, 总共 550 个人 接受了 treatment B, 但是有 500 个是重病患者. 因此计算最终的死亡率时候, 重病死亡率的权重更大, 导致对于 treatment B 的 Total 死亡率接近 20 %. 同理, 对于 treatment A, 轻症患者更多, 所以最后的平均死亡率反而比较低. 所以, 到底哪个更好?? 实际上, 这个答案是基于因果关系的.


如果受试者的 Condition 影响 treatment. 举个例子, 医生会根据患病情况来给出 treatment, 如果患病情况比较轻, 那么通常会安排 treatment A. 反之, 病重的会安排 treatment B. 他们之间的关系如下:

![image.png](https://s2.loli.net/2024/04/26/I7lOEvoz3VsPmgQ.png){: width="300" height="200" }

那这时, 这需要看不同患病情况下, treatment 的治愈率, 显然这种情况下 treatment B 更好.


如果受试者的 treatment 影响 Condition. 举个例子, 比如 treatment B 比较牛逼但是稀缺, 本来患病了就直接用药即可, 但是由于人们非要等着 treatment B 导致病情恶化. 当然对于 treatment A 是没有这个问题的. 那么这是他们的关系就是:

![image.png](https://s2.loli.net/2024/04/26/oEmGarx15iw4TNM.png){: width="300" height="200" }

此时, 显然尽管 treatment B 药效好, 但是为了存活率, 我们应该选择 treatment A. 总的来说, 当我们有了因果关系之后, 就可以解决 Simpson’s Paradox 了.



#### 1.2 Correlation Does Not Imply Causation

这是一个很关键的思想: $相关性 \neq 因果性$. 有个"Nicolas Cage and Pool Drownings"的例子, 说的是演员尼古拉斯凯奇和发生游泳溺水次数的相关性.

![image.png](https://s2.loli.net/2024/04/26/q9rXRWn3k5ZhlKa.png)


有人发现, 你用这个哥们儿出演电影次数和有人游泳溺水次数算线性相关性, 结果可能显示高度相关, 这明显是很离谱的事情. 显然他们并没有什么因果性. 

再看一个例子, 人们发现一个事情, 那些晚上很晚回来并且穿着鞋子睡觉的人, 第二天早上醒来会头痛. 事实确实发生了, 人们会说他们是相关的. 但是实际隐藏了一个条件, 这些晚上穿着鞋子睡觉的人, 大概率是喝酒喝醉了回来倒头就睡, 第二天头疼也八成是因为喝酒喝的. 我们称背后隐藏的这个条件为 "**confounder**". 

![image.png](https://s2.loli.net/2024/04/26/eLzqBX6R4pUKVNc.png){: width="300" height="200" }

我们称 confounder 与 研究对象 的关联为 "**confounding association**" . 如果想单纯探究 "穿鞋睡觉 -> 第二天头疼" 的因果关系, 我们就必须先断掉 confounder 的影响.

## 2. 第二章 Potential Outcomes

第二章主要介绍基础概念. 

### 2.1 Potential Outcomes and Individual Treatment Effects

考虑一个例子：假设有个人心情不太好，有人想送给他一只狗。如果他接受了这只狗，那他可能会变得开心。但如果他拒绝了呢？他会不会继续感到不高兴呢？反过来想，如果他接受了这只狗，但他仍然感到不高兴，那么我怎么知道，如果不送给他，他是否会变得更开心呢？


#### 2.1.1  Potential Outcomes

根据前面的分析，实际上针对某个人采取不同的处理方式会产生不同的结果，这是一个潜在的结论。我们在之后的讨论中, 称这个潜在的输出为 $\mathit {Y}$ . 

在上边的例子中, $\mathit {Y} = 1$ 表示高兴, $\mathit {Y} = 0$ 为不高兴. 

用 $\mathit {T}$ 表示 treatment 这个随机变量. $\mathit {T} = 1$ 表示接受狗子, $\mathit {T} = 0$ 表示不接受. 

使用 $\mathit {Y(1)}$ 表示接受狗子以后的潜在输出, $\mathit {Y(0)}$ 为采取不接受狗子后的输出.

#### 2.1.2  Individual Treatment Effects

因为有很多人, 我们使用 $\tau_i \triangleq Y_i(1) - Y_i(0)$ 来评估某个个体采取 treatment 之后的潜在输出结果.

> 你可以观察接受狗子之后, 观察 $\mathit {Y(1)}$. 反之, 你可以不接受狗子来观察  $\mathit {Y(0)}$. 但是你不能同时观察到 $\mathit {Y(1)}$ 和 $\mathit {Y(0)}$ !!!! 这个问题就是 "**Fundamental Problem of Causal Inference**"
{: .prompt-info }

#### 2.1.3  Average Treatment Effects

因为每个人可能有些许差异, 实际要想客观的评估 treatment 的作用, 我们要对所有人求 treatment 期望:

$$
\tau \triangleq \mathbb{E}[Y_i(1) - Y_i(0)] = \mathbb{E}[Y(1) - Y(0)]
$$



## Reference








