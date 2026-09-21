---
title: 🗺️ LLM 知识体系导航 — 从底层原理到工程实践
published: 2026-06-03
description: 以 LLM 知识体系本身为骨架梳理的完整学习地图。已写的文章可以直接点进去读，空缺的知识点标注了"待补充"，后续按图补全。
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

这张导航页是按 **LLM 知识体系本身** 搭建的，不是按现有博客文章罗列的。每个格子代表一块独立的知识点，标注了：

- ✅ **已写** — 有点击就能读
- 📝 **待补充** — 知识点位置已预留，还没写

建议从上往下读，但同一区块内的知识点可以跳着看。遇到标注了"前置"的概念，跳到对应区块先读。

---

## 一、数据表示 (Representation)

> 模型怎么把文字变成它能处理的数字。

### 1.1 Tokenization

| 知识点                      | 状态                                                                  | 说明                                    |
| --------------------------- | --------------------------------------------------------------------- | --------------------------------------- |
| BPE (Byte-Pair Encoding)    | ✅ [Tokenization 完全指南](/posts/2026-05-16-llm-tokenization-guide/) |                                         |
| WordPiece                   | ✅ 同上                                                               |                                         |
| Unigram                     | ✅ 同上                                                               |                                         |
| SentencePiece               | ✅ 同上                                                               |                                         |
| Special Tokens 设计         | 📝 待补充                                                             | padding / bos / eos / sep / mask 的作用 |
| Vocabulary 与嵌入矩阵的关系 | 📝 待补充                                                             | vocab_size × hidden_dim 怎么来的        |

### 1.2 Position Encoding

| 知识点                           | 状态                                                                          | 说明                                  |
| -------------------------------- | ----------------------------------------------------------------------------- | ------------------------------------- |
| Absolute Position Encoding       | 📝 待补充                                                                     | Transformer 原版 Sinusoidal 编码      |
| RoPE (Rotary Position Embedding) | ✅ [RoPE 旋转位置编码](/posts/2026-05-16-llm-rope-rotary-position-embedding/) |                                       |
| RoPE 外推问题                    | ✅ [长上下文扩展 — PI/NTK/YaRN](/posts/2026-05-16-llm-long-context-yarn/)     | 前置：RoPE                            |
| ALiBi                            | 📝 待补充                                                                     | 另一种位置编码方案                    |
| Context Extension 技术全景       | 📝 待补充                                                                     | PI / NTK-aware / YaRN / Log-N-Spacing |

---

## 二、模型架构 (Architecture)

> 模型内部长什么样，每层在做什么。

### 2.1 Attention 机制

| 知识点                                | 状态          | 说明                                               |
| ------------------------------------- | ------------- | -------------------------------------------------- |
| 缩放点积 Attention                    | 📝 待补充     | Q/K/V 公式，为什么除 √d                            |
| Multi-Head Attention                  | 📝 待补充     | 多头的意义，head_dim 的选择                        |
| **MLA (Multi-head Latent Attention)** | 📝 **待补充** | 通过低秩表示减少 KV 缓存；具体比例依模型结构而定 |
| **Sliding Window Attention**          | 📝 待补充     | Mistral 使用，固定窗口大小                         |
| **Cross-Attention**                   | 📝 待补充     | Encoder-Decoder 和多模态模型的基础                 |
| Causal Mask / Padding Mask            | 📝 待补充     | 两种 mask 的区别                                   |

### 2.2 前馈网络与激活

| 知识点           | 状态          | 说明                                                |
| ---------------- | ------------- | --------------------------------------------------- |
| FFN / MLP 层     | 📝 待补充     | SwiGLU / GeGLU / ReGLU / GELU / ReLU                |
| **GLU 变体对比** | 📝 **待补充** | SwiGLU vs GeGLU vs ReGLU，决定 2/3 参数量的关键结构 |
| **激活函数演进** | 📝 待补充     | ReLU → GELU → SwiGLU 的演进，计算代价与梯度特性     |

### 2.3 归一化与残差连接

| 知识点              | 状态      | 说明                        |
| ------------------- | --------- | --------------------------- |
| LayerNorm / RMSNorm | 📝 待补充 | 为什么现代 LLM 都用 RMSNorm |
| Residual Connection | 📝 待补充 | Pre-Norm vs Post-Norm       |

### 2.4 架构变体

| 知识点                            | 状态          | 说明                                           |
| --------------------------------- | ------------- | ---------------------------------------------- |
| **Transformer 架构选型**          | 📝 **待补充** | Decoder-only vs Encoder-Decoder vs Prefix LM   |
| **MoE (Mixture of Experts) 架构** | 📝 **待补充** | 总参数量 vs 激活参数量的概念，稀疏激活核心思想 |
| └─ Router 设计                    | 📝 待补充     | Top-1 / Top-2 / Noisy Top-k Gating             |
| └─ Load Balancing Loss            | 📝 待补充     | 防止所有 token 路由到同一个专家                |
| └─ Expert Capacity & Token Drop   | 📝 待补充     | 溢出 token 的处理策略                          |
| └─ Dense → MoE 的切换代价         | 📝 待补充     | EP 并行、通信模式、显存分布变化                |

---

## 三、推理 (Inference)

> 模型训练好后怎么跑推理——从单步生成到 Serving 架构。

### 3.1 推理计算

| 知识点                                                             | 状态                                                           | 说明                                     |
| ------------------------------------------------------------------ | -------------------------------------------------------------- | ---------------------------------------- |
| Autoregressive Decoding                                            | 📝 待补充                                                      | 逐 token 生成的基本流程                  |
| Decoding 策略 (Greedy / Beam Search / Top-K / Top-P / Temperature) | ✅ [Decoding 策略](/posts/2026-05-16-llm-decoding-strategies/) |                                          |
| Repetition Penalty                                                 | 📝 待补充                                                      |                                          |
| KV Cache 原理                                                      | ✅ [推理显存拆解](/posts/2026-06-13-llm-mem-opt-2-inference/) | 解释缓存内容与显存计算                   |
| KV Cache 显存计算                                                  | ✅ [推理显存拆解](/posts/2026-06-13-llm-mem-opt-2-inference/)     | 前置：KV Cache、GQA                     |
| GQA / MQA / MHA                                                    | 📝 待补充                                                      | Grouped Query Attention 的原理和显存收益 |

### 3.2 推理加速

| 知识点                       | 状态                                                     | 说明                                    |
| ---------------------------- | -------------------------------------------------------- | --------------------------------------- |
| Flash Attention 原理         | 📝 待补充                                               | [推理显存拆解](/posts/2026-06-13-llm-mem-opt-2-inference/)仅介绍显存收益，尚无完整算法推导 |
| Flash Attention 公式推导     | 📝 待补充                                                | tiling / online softmax / recomputation |
| **PagedAttention / vLLM**    | 📝 **待补充**                                            | 通过虚拟内存管理 KV Cache               |
| Speculative Decoding         | 📝 待补充                                                | Draft model + Verify                    |
| **Weight-Only Quantization** | 📝 **待补充**                                            | GGUF / bitsandbytes                     |
| Quantization (GPTQ / AWQ)    | ✅ [量化深入](/posts/2026-06-14-llm-mem-opt-3-quantization/) | 对称/非对称、scale、GPTQ/AWQ          |

### 3.3 Serving 架构

> 模型上线后的工程架构——请求调度、计算阶段切分、KV Cache 管理。

| 知识点                                     | 状态          | 说明                                                             |
| ------------------------------------------ | ------------- | ---------------------------------------------------------------- |
| **PD 分离 (Prefill-Decode Separation)**    | 📝 **待补充** | 2025-2026 推理系统最大演进，Prefill 和 Decode 分配到不同类型 GPU |
| **Continuous Batching**                    | 📝 **待补充** | vLLM 核心机制，无需等整个 batch 完成即可插入新请求               |
| **In-flight Batching / Dynamic Batching**  | 📝 **待补充** | 推理引擎核心调度策略                                             |
| **Chunked Prefill**                        | 📝 **待补充** | 长 Prefill 切成 chunk 减少 TTFT                                  |
| **SplitFuse**                              | 📝 **待补充** | DeepSpeed-FastGen 方案                                           |
| **Prefix Caching / Context Caching**       | 📝 **待补充** | 共享前缀场景下降低 KV Cache 使用                                 |
| **Ring Attention / Distributed Attention** | 📝 **待补充** | 突破单 GPU 显存限制的长上下文方案                                |

---

## 四、训练原理 (Training Mechanics)

> 模型怎么训练的——一个 Step 的完整生命周期是所有参数的入口地图。

### 4.0 训练 Step 生命周期 ← 所有训练参数的入口

```
DataLoader → Forward → Loss → Backward → AllReduce → Optimizer
```

| 知识点                       | 状态                    | 说明                                     |
| ---------------------------- | ----------------------- | ---------------------------------------- |
| **一个训练 Step 的完整拆解** | 📝 **待补充（高优先）** | 所有训练参数的入口地图，每步控制哪些参数 |

### 4.1 数据管线

| 知识点                      | 状态                                                                             | 说明                     |
| --------------------------- | -------------------------------------------------------------------------------- | ------------------------ |
| Chat Template               | 📝 待补充                                                                        | 从 messages 到格式化文本 |
| Tokenization + Padding      | 📝 待补充                                                                        | 变长序列怎么对齐         |
| Packing / BFD               | ✅ [SFT 训练实战](/posts/2026-06-16-llm-mem-opt-5-sft/) | 掩码原理 + 三种策略     |
| DataLoader / num_workers    | 📝 待补充                                                                        | 数据加载瓶颈排查         |
| Dataset Sharding / Blending | 📝 待补充                                                                        | 分布式训练的数据分配     |

### 4.2 数值精度

| 知识点                   | 状态      | 说明                          |
| ------------------------ | --------- | ----------------------------- |
| FP32 / FP16 / BF16 对比  | 📝 待补充 | 精度 / 范围 / 硬件支持        |
| Mixed Precision Training | 📝 待补充 | Master Weights / Loss Scaling |
| FP8 训练                 | 📝 待补充 | H100 支持的新方案             |

### 4.3 优化器

| 知识点                    | 状态      | 说明                             |
| ------------------------- | --------- | -------------------------------- |
| SGD → SGD+Momentum        | 📝 待补充 | 动量怎么平滑梯度                 |
| Adam / AdamW              | 📝 待补充 | 自适应学习率 + Weight Decay 解耦 |
| Adam 的显存开销           | 📝 待补充 | 2 × fp32 states = 8 bytes/param  |
| 8-bit Adam / AnyPrecision | 📝 待补充 | 优化器状态压缩                   |

### 4.4 学习率调度

| 知识点                              | 状态      | 说明                           |
| ----------------------------------- | --------- | ------------------------------ |
| Warmup 的作用                       | 📝 待补充 | 防止训练初期梯度爆炸           |
| Cosine / Linear / Constant Schedule | 📝 待补充 | 各调度器的使用场景             |
| Min-LR / Max-LR 的选择              | 📝 待补充 | 预训练 vs SFT vs RLHF 的典型值 |
| LR 与 Batch Size 的关系             | 📝 待补充 | Linear Scaling Rule            |

### 4.5 梯度相关

| 知识点                    | 状态      | 说明                                                 |
| ------------------------- | --------- | ---------------------------------------------------- |
| Gradient Accumulation     | 📝 待补充 | micro_batch / global_batch / accumulation_steps 公式 |
| Gradient Clipping         | 📝 待补充 | 防止梯度爆炸                                         |
| Gradient Checkpointing    | 📝 待补充 | 激活值重计算的原理                                   |
| Gradient Sync (AllReduce) | 📝 待补充 | NCCL 通信模式                                        |

### 4.6 损失函数

| 知识点                      | 状态                                                                             | 说明                                            |
| --------------------------- | -------------------------------------------------------------------------------- | ----------------------------------------------- |
| Cross-Entropy Loss (SFT)    | ✅ [SFT 训练实战](/posts/2026-06-16-llm-mem-opt-5-sft/) | 含 Label Shift 和 Loss Mask                     |
| Chunked Cross-Entropy (NLL) | ✅ 同上                                                                          | 大词表场景的显存优化                            |
| KL Divergence (RLHF)        | ✅ [KL 散度](/posts/2024-04-19-kullback-leibler-divergence/)                     | 可约束策略偏离参考模型；不能保证杜绝奖励投机   |
| Pairwise Ranking Loss (DPO) | 📝 待补充                                                                        | 对齐文章尚未纳入站点                            |
| Loss Mask 策略              | 📝 待补充                                                                        | 只算 assistant / 只算最后一轮 / 只算 completion |

### 4.7 显存管理

| 知识点                | 状态                                                                             | 说明                                   |
| --------------------- | -------------------------------------------------------------------------------- | -------------------------------------- |
| 推理显存账本          | ✅ [推理显存拆解](/posts/2026-06-13-llm-mem-opt-2-inference/)                       | 权重 / KV Cache / 临时工作区            |
| **训练模型状态账本**  | ✅ [训练显存与 ZeRO](/posts/2026-06-15-llm-mem-opt-4-zero/)                          | 16Ψ 的适用条件、ZeRO-1/2/3 分片原理    |
| Activation Offloading | ✅ [SFT 训练实战](/posts/2026-06-16-llm-mem-opt-5-sft/) | saved_tensors_hooks，异步传输          |
| **三层防御策略**      | 📝 **待补充**                                                                    | FlashAttn → Recompute → Model Parallel |
| ZeRO 显存优化         | ✅ [训练显存与 ZeRO](/posts/2026-06-15-llm-mem-opt-4-zero/)                          | ZeRO-1/2/3 公式推导 + 实测数据        |

---

## 五、分布式训练 (Distributed Training)

> 模型大到一张 GPU 放不下时，怎么切、怎么通信、怎么选。

### 5.1 并行策略

| 知识点                      | 状态                      | 说明                             |
| --------------------------- | ------------------------- | -------------------------------- |
| Data Parallelism (DP / DDP) | 📝 待补充                 | 每卡完整模型，切分数据           |
| Tensor Parallelism (TP)     | 📝 **待补充（高优先）**   | 层内矩阵切分，AllReduce 通信     |
| Pipeline Parallelism (PP)   | 📝 **待补充（高优先）**   | 层间切分，气泡问题               |
| Expert Parallelism (EP)     | 📝 **待补充（高优先）**   | MoE 模型专用，稀疏通信           |
| Sequence Parallelism (SP)   | 📝 待补充                 | LayerNorm/Dropout 的序列维度切分 |
| 4D 并行全景图               | 📝 待补充                 | TP × PP × DP × SP 的组合排布     |
| **单机并行选型策略**        | 📝 **待补充（实战案例）** | Dense vs MoE 的选型案例          |

### 5.2 通信与框架

| 知识点                                            | 状态          | 说明                                                    |
| ------------------------------------------------- | ------------- | ------------------------------------------------------- |
| NCCL 原语 (AllReduce / AllGather / ReduceScatter) | 📝 待补充     |                                                         |
| Ring AllReduce 原理                               | 📝 待补充     | 通信量 = 2×(N-1)/N × 数据量                             |
| **通信拓扑与带宽分析**                            | 📝 **待补充** | NVLink(900GB/s) vs InfiniBand(400Gb/s) 对并行策略的影响 |
| **通信计算重叠**                                  | 📝 **待补充** | Bucket 化梯度同步，Backward 中启动 AllReduce            |
| DeepSpeed ZeRO-1/2/3                              | ✅ [训练显存与 ZeRO](/posts/2026-06-15-llm-mem-opt-4-zero/)     | 公式 + 通信模式 + H100 实测 + DeepSpeed 配置                            |
| Megatron-LM                                       | 📝 待补充     | TP + PP 的实现架构                                      |
| FSDP (Fully Sharded Data Parallel)                | ✅ [训练显存与 ZeRO](/posts/2026-06-15-llm-mem-opt-4-zero/)     | FSDP vs DeepSpeed ZeRO-3 对比                              |

---

## 六、微调方法 (Fine-tuning Methods)

> 拿到一个预训练模型后，怎么把它适配到你的任务上。

### 6.1 全参数微调

| 知识点                      | 状态                                                                             | 说明                        |
| --------------------------- | -------------------------------------------------------------------------------- | --------------------------- |
| SFT Trainer 使用            | ✅ [SFT 训练实战](/posts/2026-06-16-llm-mem-opt-5-sft/) | 完整入门到优化组合                |
| 全参数微调 vs LoRA 决策     | 📝 待补充                                                                        | 显存对比 + 效果权衡         |
| **Megatron 全参数微调配置** | 📝 **待补充（实战案例）**                                                        | `--finetune false` 参数详解 |
| **训练配置逐行注释**        | 📝 **待补充（实战案例）**                                                        | 一份可复现的训练配置        |

### 6.2 参数高效微调 (PEFT)

| 知识点                                | 状态      | 说明                           |
| ------------------------------------- | --------- | ------------------------------ |
| LoRA 原理                             | 📝 待补充 | A×B 低秩分解，rank 的选择      |
| LoRA 的秩和在哪层插                   | 📝 待补充 | 经验规则：注意力层的 W_Q / W_V |
| QLoRA (4-bit)                         | 📝 待补充 | NF4 量化 + LoRA                |
| AdaLoRA / DoRA                        | 📝 待补充 | 自适应秩分配 / 权重方向分解    |
| P-Tuning / Prefix Tuning              | 📝 待补充 | Soft Prompt 方案               |
| LoRA vs Adapter vs Prompt Tuning 对比 | 📝 待补充 | 完整表格                       |

### 6.3 对齐训练 (Alignment)

| 知识点                                          | 状态                                                | 说明                            |
| ----------------------------------------------- | --------------------------------------------------- | ------------------------------- |
| RLHF 全流程                                     | 📝 待补充                                           | 对齐文章尚未纳入站点            |
| Reward Model 训练                               | 📝 待补充                                           |                                 |
| **PPO 在 LLM 中的应用**                         | 📝 待补充                                           |                                 |
| DPO (Direct Preference Optimization)            | 📝 待补充                                           |                                 |
| GRPO (Group Relative Policy Optimization)       | 📝 待补充                                           | DeepSeek 用的方案               |
| **DAPO (Dynamic Sampling Policy Optimization)** | 📝 **待补充**                                       | 字节/清华提出，GRPO 同族        |
| **RLOO (REINFORCE Leave-One-Out)**              | 📝 **待补充**                                       | 不需要 critic model             |
| **Reinforce++**                                 | 📝 **待补充**                                       | HuggingFace TRL 实现的高效变体  |
| **KTO (Kahneman-Toshev Optimization)**          | 📝 **待补充**                                       | 只需要好/坏二分类标签           |
| **ORPO (Odds Ratio Preference Optimization)**   | 📝 **待补充**                                       | 将对齐融合进 SFT 阶段           |
| **SimPO (Simple Preference Optimization)**      | 📝 **待补充**                                       | 用平均 log-prob 替代隐式 reward |
| **IPO (Identity Preference Optimization)**      | 📝 **待补充**                                       | DPO 改进版，解决过拟合          |
| SFT + DPO + GRPO 显存对比                       | 📝 待补充                                           | 多个模型同时加载的开销          |

### 6.4 训练工程

| 知识点                             | 状态                      | 说明                         |
| ---------------------------------- | ------------------------- | ---------------------------- |
| Checkpoint 结构                    | 📝 待补充                 | 模型/优化器/RNG/断点续训逻辑 |
| **Loss Spike 排查**                | 📝 **待补充（实战问题）** | 常见原因 + 排查步骤          |
| **OOM 排查手册**                   | 📝 **待补充（实战问题）** | 显存账本排查法               |
| 训练监控 (WandB / TensorBoard)     | 📝 待补充                 | 看哪些曲线、怎么看           |
| **多轮 Function Calling 数据构造** | 📝 **待补充（实战问题）** | 数据构造与质量检查           |

---

## 七、数学与评估基础 (Foundations)

> 贯穿所有层的底层工具和评估指标。**不用从头啃，用到什么查什么。**

### 7.1 信息论与概率

| 知识点                        | 状态                                                         | 说明                |
| ----------------------------- | ------------------------------------------------------------ | ------------------- |
| KL 散度                       | ✅ [KL 散度](/posts/2024-04-19-kullback-leibler-divergence/) | RLHF/DPO 的核心概念 |
| 交叉熵                        | 📝 待补充                                                    | SFT 损失的基础      |
| Gamma / Beta / Dirichlet 分布 | ✅ [概率分布](/posts/2024-04-10-gama-beta-dirichlet/)       |                     |
| 概率校准                      | ✅ [概率校准](/posts/2024-04-10-probability-calibration/)   |                     |

### 7.2 优化与正则化

| 知识点          | 状态      | 说明             |
| --------------- | --------- | ---------------- |
| Weight Decay    | ✅ [Weight Decay](/posts/2024-04-20-weight-decay-and-l2-regularization/) | AdamW 的核心改进 |
| L1 / L2 正则化  | ✅ [L1 / L2 正则化](/posts/2024-04-20-l1-and-l2-regularization/) |                  |
| SGD / Adam 原理 | 📝 待补充 |                  |
| 采样方法        | ✅ [采样方法](/posts/2024-04-11-sampling-method/) |                  |

### 7.3 评估指标

| 知识点     | 状态 | 说明             |
| ---------- | ---- | ---------------- |
| Perplexity | ✅ [Perplexity](/posts/2024-04-10-perplexity/) | 语言模型基础评估 |
| AUC / GAUC | ✅ [AUC / GAUC](/posts/2024-04-12-auc-gauc/)    |                  |

---

## 八、工具与框架 (Ecosystem)

> 每一层的工具选型参考。

```
Tokenizer      训练框架        推理框架
─────────────────────────────────────────────
HF Tokenizers  TRL             vLLM
               DeepSpeed       TensorRT-LLM
               Megatron-LM     TGI
               FSDP
```

| 工具                       | 状态                                                                             | 说明                                 |
| -------------------------- | -------------------------------------------------------------------------------- | ------------------------------------ |
| HuggingFace Transformers   | 📝 待补充                                                                        | 核心 API (from_pretrained / Trainer) |
| TRL                        | ✅ [SFT 训练实战](/posts/2026-06-16-llm-mem-opt-5-sft/) | SFTTrainer / DPOTrainer、GRPO              |
| PEFT (LoRA)                | 📝 待补充                                                                        | PeftModel / LoraConfig               |
| DeepSpeed                  | ✅ [训练显存与 ZeRO](/posts/2026-06-15-llm-mem-opt-4-zero/)                       | ZeRO 配置 + Offload/Infinity/ZeRO++                 |
| **Megatron-LM**            | 📝 **待补充（实战案例）**                                                        | TP/PP 配置                           |
| FlashAttention (FA2 / FA3) | 📝 待补充                                                                        |                                      |
| vLLM                       | 📝 待补充                                                                        | PagedAttention / 推理部署            |
| Liger Kernel               | ✅ 同上                                                                          | Triton Kernel 集合                   |

---

## 建议的阅读路径

### 路径 A：快速入门（动手跑通一次）

```
Step 生命周期（先看这张总图）
→ Tokenization → Decoding → KV Cache
→ TRL SFT 上手 → LoRA 实测
```

### 路径 B：训练工程师（系统掌握）

```
路径 A 后 →
数据管线 → 混合精度 → 优化器 → LR Schedule → 梯度累积
→ 训练显存拆解 → 分布式四兄弟 → ZeRO
→ 全参数 vs LoRA → 数据管线
→ 配置逐行注释（实战案例）
```

### 路径 C：分布式专家

```
分布式四兄弟 → NCCL 通信 → Ring AllReduce
→ Megatron TP/PP → DeepSpeed ZeRO
→ 通信拓扑与带宽 → 单机并行选型（实战案例）
→ 通信计算重叠 → 4D 并行全景
```

### 路径 D：数学基础（随时查阅）

```
用到什么查什么，不用从头读：
→ KL 散度（RLHF/DPO）
→ Perplexity（训练评估）
→ Weight Decay（AdamW）
→ 交叉熵（SFT 损失）
```

---

## 按状态筛选

已写、可直接阅读：

- 基础：[分词](/posts/2026-05-16-llm-tokenization-guide/)、[RoPE 位置编码](/posts/2026-05-16-llm-rope-rotary-position-embedding/)、[长上下文扩展](/posts/2026-05-16-llm-long-context-yarn/)、[生成时如何选词](/posts/2026-05-16-llm-decoding-strategies/)
- 显存与训练：[推理显存](/posts/2026-06-13-llm-mem-opt-2-inference/)、[量化](/posts/2026-06-14-llm-mem-opt-3-quantization/)、[SFT 训练](/posts/2026-06-16-llm-mem-opt-5-sft/)
- 数学基础：[KL 散度](/posts/2024-04-19-kullback-leibler-divergence/)、[困惑度](/posts/2024-04-10-perplexity/)、[L1/L2 正则化](/posts/2024-04-20-l1-and-l2-regularization/)、[权重衰减](/posts/2024-04-20-weight-decay-and-l2-regularization/)、[AUC/GAUC](/posts/2024-04-12-auc-gauc/)、[概率分布](/posts/2024-04-10-gama-beta-dirichlet/)、[概率校准](/posts/2024-04-10-probability-calibration/)、[采样方法](/posts/2024-04-11-sampling-method/)

待补充：FlashAttention 的完整推导、对齐训练、单机并行选型、OOM 排查和多轮工具调用数据构造。相关草稿尚未纳入站点，完成核对后再开放入口。

---

_这份导航会随文章更新。只有内容已核对且页面已纳入站点，才将状态改为“已写”。_
