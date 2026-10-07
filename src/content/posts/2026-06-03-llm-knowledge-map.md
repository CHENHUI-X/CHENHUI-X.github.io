---
title: LLM 学习地图：入门主线、深入专题与应用部署
published: 2026-06-03
description: 从 9 步入门主线开始，再按问题选读优化、对齐与应用专题；重复内容合并到唯一详解页，区分已覆盖与尚未展开。
category: Guide
tags:
  - llm
  - guide
  - learning-path
  - navigation
draft: false
pin: true
---

## 0. 这张图怎么用

先弄清“文字怎样进入模型、模型怎样生成、怎样训练”的完整流程，再挑需要的专题。**学习顺序与发布日期无关**。旧系列标题已经按内容整理，原地址能继续访问；合并掉的文章会直接带到承接内容的位置。

- 第一次系统学习：按下面 9 步走，不必先掌握分布式训练或强化学习
- 遇到具体问题：直接进入深入专题；每个概念只在指定文章完整展开
- 想看其他领域：[全站主题目录](/topics/)收录全部文章，因果推断、分类指标和站务推荐各有独立位置

代码语法或 CPU 算例通过，不等于真实 GPU 训练已经验证。显存估算也不是运行峰值；具体版本、假设与验证范围见对应文章。

## 1. 入门主线：先把流程走通

| 步骤 | 文章 | 这一步要弄清什么 |
| --- | --- | --- |
| 1 | [Tokenization：从文字到 token](/posts/2026-05-16-llm-tokenization-guide/) | 字符、子词与词表；BPE、WordPiece、Unigram 和 SentencePiece 的关系 |
| 2 | [Perplexity：语言模型怎样评分](/posts/2024-04-10-perplexity/) | 条件概率、交叉熵、PPL，以及比较时必须固定的口径 |
| 3 | [Transformer Decoder：从零生成文本](/posts/2024-04-13-text-generator-with-transformer-decoder/) | Q/K/V、causal mask、label shift、最小训练和生成代码 |
| 4 | [RoPE：旋转位置编码](/posts/2026-05-16-llm-rope-rotary-position-embedding/) | 位置怎样进入 Q/K 内积；需要矩阵和三角函数基础 |
| 5 | [Decoding：怎样选下一个 token](/posts/2026-05-16-llm-decoding-strategies/) | Greedy、Temperature、Top-K、Top-P 与 Beam Search |
| 6 | [显存基础：单位、精度与生命周期](/posts/2026-06-12-llm-mem-opt-1-fundamentals/) | 哪些张量长期保留，哪些只在某个阶段存在 |
| 7 | [推理显存：完整算例与 OOM 排查](/posts/2026-06-13-llm-mem-opt-2-inference/) | 权重、GQA/KV 和工作区；同篇包含计算原理、速算器与排查表 |
| 8 | [量化入门：scale、zero-point 与误差](/posts/2026-05-16-llm-quantization-guide/) | 少存 bit 怎样引入近似；后半 GPTQ/AWQ 为数学选读 |
| 9 | [SFT：数据与显存优化实战](/posts/2026-06-16-llm-mem-opt-5-sft/) | 先检查训练目标和样本边界，再组合 LoRA、packing 与内存开关 |

这 9 步只是建议的学习顺序。想先写出最小模型，可以在第 3 步动手；单卡 SFT 不要求先学 ZeRO。遇到 KL、优化器或强化学习的推导，再进入下面的基础和专题。

## 2. 深入专题：按问题选择

| 你的问题 | 唯一详解入口 | 先知道什么 |
| --- | --- | --- |
| RoPE 怎样扩展到更长文本？ | [长上下文：PI、NTK-aware 与 YaRN](/posts/2026-05-16-llm-long-context-yarn/) | RoPE 的频率和旋转 |
| AdamW 的权重衰减和 L2 是否一样？ | [AdamW：更新规则与状态](/posts/2024-04-20-weight-decay-and-l2-regularization/) | 梯度、L1/L2 基础 |
| 多张卡怎样分担训练状态？ | [训练显存与 ZeRO](/posts/2026-06-15-llm-mem-opt-4-zero/) | 显存基础、Adam/AdamW |
| SFT 后怎样学习人类偏好？ | [LLM 对齐：RLHF、PPO 与 DPO](/posts/2025-12-05-a-series-on-llm-training-i/) | SFT、KL；PPO 部分再补通用 RL |

推理的缓存/FlashAttention 推导已并入主线第 7 篇，位于容量账本之后；显存速算与 OOM 表在同篇末尾。GPTQ/AWQ 已并入第 8 篇后半，不再单独维护量化基础。独有推导与数值例子都保留，进阶部分可以按需跳读。

## 3. 基础与应用：需要时再进入

| 入口 | 本文负责的范围 |
| --- | --- |
| [KL 散度](/posts/2024-04-19-kullback-leibler-divergence/) | 比较分布；熵、交叉熵、支撑集和高斯 KL |
| [采样方法](/posts/2024-04-11-sampling-method/) | 逆变换与拒绝采样；LLM 候选分布的选择由 Decoding 负责 |
| [L1/L2 正则化](/posts/2024-04-20-l1-and-l2-regularization/) | 统计、几何、先验和稀疏性；优化器差异由 AdamW 负责 |
| [通用深度强化学习](/posts/2024-04-30-deep-reinforcement-learning/) | MDP、Policy Gradient、GAE、PPO 与 Q-learning；token 级对齐由对齐专题负责 |
| [LLM 应用：Prompt、RAG、Tools 与 Agent](/posts/2024-04-24-lang-chain-series/) | 用 LangChain 组织应用组件；检索 embedding、工具调用和 agent 边界 |
| [Hermes Agent：WSL 部署与排错](/posts/2026-05-15-hermes-agent-wsl-setup/) | 运行环境、系统服务和网关；不再重讲应用框架概念 |

搭建应用不需要先读完整条模型训练路线。先看应用概念，再按实际需要查看部署案例；框架 API 和系统命令都要对照文章注明的版本边界。

## 4. 知识版图：已经覆盖什么，还缺什么

上面列出了可以立即阅读的文章，下面说明各领域讲到了哪里。**部分覆盖**表示已有局部讲解，不代表整项已经讲完。“尚未展开”只记录内容空白，不表示已有发布时间。

### 一、数据表示 (Representation)

已覆盖：分词算法与词表。部分覆盖：token ID、embedding 查表、特殊 token 与对话模板。

尚未展开：embedding 训练专题、完整 EOS/SEP/MASK 设计、ALiBi 与其他位置表示比较。

### 二、模型架构 (Architecture)

已覆盖：最小 Decoder、Q/K/V、causal/padding mask、RoPE。部分覆盖：MHA、GQA/MQA、FFN、LayerNorm 与残差在模型中的位置。

尚未展开：完整归一化和激活函数比较、Cross-Attention、MLA、Sliding Window、MoE 的 router/负载均衡/专家容量，以及 Dense 与 MoE 选型。

### 三、推理 (Inference)

已覆盖：解码策略、KV 前向复用与容量、FlashAttention 前向、量化。部分覆盖：分页缓存、前缀共享和 Chunked Prefill；Chunked Prefill 不保证 TTFT 下降，调度也影响逐 token 延迟与吞吐。

尚未展开：Repetition Penalty、FlashAttention 反向与完整 GPU kernel、Speculative Decoding、PD 分离、Continuous Batching、SplitFuse、Ring/Distributed Attention，以及格式与后端部署比较。

### 四、训练原理 (Training Mechanics)

已覆盖：数值精度、AdamW、SFT 数据边界与 loss、激活优化。部分覆盖：mixed precision、梯度累积、学习率和裁剪配置。

尚未展开：完整 Loss Scaling、FP8 训练、学习率调度比较、8-bit Adam、数据吞吐与分布式采样、profiler 和通信重叠专题。

### 五、分布式训练 (Distributed Training)

已覆盖：指定训练配方下的 ZeRO-1/2/3 分片和通信账本。部分覆盖：DDP、collective 语义、offload 与 FSDP 关系。

尚未展开：TP、PP、EP、SP 的完整实现与组合选型，NCCL/通信拓扑、完整多机启动和容错；没有 H100 实测结论。

### 六、微调与对齐 (Fine-tuning and Alignment)

已覆盖：SFT、LoRA/QLoRA 基础、奖励模型、KL 正则、token 级 PPO、DPO。部分覆盖：在线 RL 与工具兼容性提示。

尚未展开：Prefix/Prompt Tuning、完整 GRPO 训练与推导、ORPO、KTO（Kahneman–Tversky Optimization）、蒸馏与系统性的对齐评测。

### 七、数学与评估基础 (Foundations)

已覆盖：PPL、KL、采样、L1/L2、通用 RL。分类校准和 AUC/GAUC 在全站数学与模型基础目录中，不当作语言生成质量的替代指标。

尚未展开：完整 benchmark 设计、污染检查、LLM-as-a-judge 的偏差控制、事实性与安全性的系统评价。

### 八、工具与框架 (Ecosystem)

已覆盖：应用组件的概念分工与 WSL 部署案例。部分覆盖：TRL、DeepSpeed、PEFT 等在具体文章中的用法；这是有版本边界的示例。

尚未展开：完整 serving 平台、系统性的 RAG 检索评测（指标、数据集与对照实验）、生产 agent 的权限审计、可观测性与回滚设计。

## 按目标选择下一步

先做一件能验证的小事：跑通 Decoder 的最小例子、为自己的模型算清容量，或检查一条 SFT 样本。遇到不明白的概念再回到对应详解页，不用把所有专题一次读完。
