---
title: LLM 对齐入门：RLHF、PPO 与 DPO
published: 2025-12-05
description: 从 SFT 与偏好数据出发，串起奖励模型、KL 正则、token 级 PPO 和 DPO；用可读公式与最小损失代码解释大模型对齐。
category: Deep Learning
tags:
- reinforcement learning
- deep learning
- nlp
- llm
draft: false
---

## 0. 从示范答案到偏好学习

语言模型会续写，不等于它总能按我们的要求回答。SFT（Supervised Fine-Tuning）用示范答案教模型怎么完成任务；偏好学习进一步比较同一个问题的不同回答，告诉模型哪一种更合适。

下文依次介绍 **SFT → 奖励模型与 KL → token 级 PPO → DPO**。RLHF（Reinforcement Learning from Human Feedback）是一套利用人类反馈的训练流程；PPO 是其中可以采用的策略优化算法，DPO 则是直接用偏好对训练模型的方法。这三个名称不处在同一层级。

阅读需要条件概率、交叉熵、反向传播与梯度下降的基础。[通用强化学习](/posts/2024-04-30-deep-reinforcement-learning/) 是补充 Policy Gradient、value function 和 GAE 的选读，不是学习 SFT 的必修课。数据整理、LoRA、packing 与训练显存问题见 [SFT 训练实战](/posts/2026-06-16-llm-mem-opt-5-sft/)；整体顺序见 [LLM 知识体系导航](/posts/2026-06-03-llm-knowledge-map/)。

下面统一用 $x$ 表示 prompt，$y=(y_1,\ldots,y_T)$ 表示回答 token 序列，$\pi_\theta$ 表示参数为 $\theta$ 的语言模型，所有 $\log$ 均为自然对数。回答包含按数据约定计入的结束 token；padding 不是回答的一部分。

## 1. SFT：先学会模仿示范答案

SFT 的数据格式、token 平均损失、attention mask 与 loss mask 已集中在 [SFT 数据与损失范围](/posts/2026-06-16-llm-mem-opt-5-sft/#13-shiftpadding-mask-和-response-mask-各管什么)。这里讨论偏好学习接下来需要的计算：**怎样得到整条回答的概率**。

令 $x$ 为 prompt，$y=(y_1,\ldots,y_T)$ 为包含约定结束 token 的回答，则

$$
\log\pi_\theta(y\mid x)
=\sum_{t=1}^{T}\log\pi_\theta(y_t\mid x,y_{<t}).
$$

下面采用 response-only 计分：prompt 仍作为输入，padding 不属于回答。SFT 用示范答案训练；DPO 则要比较 chosen/rejected 两条回答在当前模型和参考模型下的分数，所以需要保留逐 token log-probability 与有效位置 mask。

### 1.1 从 logits 取出正确的 token log-probability

假设拼接后的 prompt 与回答长度为 $L$，logits 形状为 $[B,L,V]$，labels 为 $[B,L]$，$V$ 是词表大小。labels 中 prompt 和 padding 设为 `-100`，回答位置保留 token ID。位置 $i$ 的 logits 预测位置 $i+1$ 的 label，必须错开一位：

```python
import torch
import torch.nn.functional as F

def response_token_logps(logits, labels):
    # logits: [B, L, V]; labels: [B, L]
    targets = labels[:, 1:]
    mask = targets.ne(-100)
    safe_targets = targets.masked_fill(~mask, 0)
    logps = F.log_softmax(logits[:, :-1, :].float(), dim=-1)
    selected = logps.gather(-1, safe_targets.unsqueeze(-1)).squeeze(-1)
    return selected.masked_fill(~mask, 0.0), mask
```

输出均为 $[B,L-1]$。`-100` 不能直接用于 `gather`；先替换为合法 ID，取出后再置零。模型前向仍要传入正确的 attention mask，输入 token ID 也不能换成 `-100`。至少提供一个可用于预测首个回答 token 的 prompt/BOS 位置，并检查截断后每条样本仍有回答。

对输出在有效回答位置求和，得到每条回答的 $\log\pi_\theta(y\mid x)$；求均值则是另一种长度归一化分数。后面的原始 DPO 目标使用**求和**，不能改成均值。PPO 则保留每个 token 的 log-probability。

## 2. 奖励模型：把偏好变成可优化的分数

### 2.1 偏好对与奖励损失

偏好数据写成 $(x,y_w,y_l)$：在同一个 prompt $x$ 下，标注者更喜欢 $y_w$（chosen），相对不喜欢 $y_l$（rejected）。偏好依据应由任务标准决定，例如正确性、相关性和表达要求，不是仅凭句子“更像人说的”判断。

奖励模型 $r_\phi(x,y)$ 为整条回答给出一个标量。采用 Bradley–Terry 偏好模型时，

$$
P_\phi(y_w\succ y_l\mid x)
=\sigma\!\left(r_\phi(x,y_w)-r_\phi(x,y_l)\right),
\qquad \sigma(z)=\frac{1}{1+e^{-z}},
$$

$$
\mathcal L_{\mathrm{RM}}
=-\mathbb E_{(x,y_w,y_l)\sim\mathcal D}
\left[\log\sigma\!\left(r_\phi(x,y_w)-r_\phi(x,y_l)\right)\right].
$$

奖励差越大，模型认为 chosen 胜出的概率越高，loss 越低。两个分数相等时，一对样本的 loss 为 $\log 2\approx0.6931$。给同一 prompt 的所有回答同时加一个常数，不改变偏好概率，因此这些分数不是天然可跨题比较的“百分制成绩”。实现时用 `-F.logsigmoid(chosen_reward - rejected_reward).mean()`，避免先算 sigmoid 再取 log 的数值问题。

一种常见实现是在 Transformer 的序列表示上接一个 scalar head。奖励模型对**完整回答**评分，value model 则估计**某个前缀之后的未来回报**；即使网络结构相似，这两个角色也不同。经典 SFT、偏好标注、奖励学习、PPO 的阶段划分可参阅 [InstructGPT 原论文 §3.1](https://arxiv.org/abs/2203.02155)。

### 2.2 为什么还需要固定参考模型

如果只追求奖励模型的高分，策略可能利用评分器的盲点，例如堆砌看起来专业却没有帮助的内容。KL 正则给模型偏离参考行为增加代价。令 $\pi_{\mathrm{ref}}$ 为冻结的参考模型，通常取 SFT 检查点，考虑

$$
J(\theta)=\mathbb E_{x\sim\mathcal D_x}\left[
\mathbb E_{y\sim\pi_\theta(\cdot\mid x)}r_\phi(x,y)
-\beta D_{\mathrm{KL}}\!\left(
\pi_\theta(\cdot\mid x)\,\|\,\pi_{\mathrm{ref}}(\cdot\mid x)
\right)\right],\qquad \beta>0.
$$

这里 $\mathcal D_x$ 是 prompt 分布。KL 方向是**当前策略相对参考策略**，期望也在当前策略下：

$$
D_{\mathrm{KL}}(\pi_\theta\|\pi_{\mathrm{ref}})
=\mathbb E_{y\sim\pi_\theta}
\left[\log\pi_\theta(y\mid x)-\log\pi_{\mathrm{ref}}(y\mid x)\right].
$$

只要策略给某个回答正概率，参考模型也应给它正概率，否则这一方向的 KL 会无穷大。更多边界条件见 [KL 散度](/posts/2024-04-19-kullback-leibler-divergence/)。

自回归分解允许把一条回答的 log-ratio 写成 token log-ratio 之和。单个采样 token、甚至单条回答的 log-ratio 都可能为负；非负的是完整分布下的 KL 期望。不能把某一次采样的负数叫作“负 KL 散度”。

在奖励函数和标度固定时，较大的 $\beta$ 意味着偏离参考分布的代价更高。KL 可以缓和过度优化，但不保证消除 reward hacking、事实错误或偏好数据偏差。

## 3. PPO：让奖励影响逐 token 的生成

### 3.1 四种角色和一次 rollout

把生成看成一个有限长度的决策过程：状态 $s_t=(x,y_{<t})$，动作 $a_t=y_t$，策略是 $\pi_\theta(y_t\mid s_t)$。

- **Actor / policy**：生成 token，也是最终要改进的语言模型
- **Reference**：冻结的 $\pi_{\mathrm{ref}}$，用于衡量相对初始行为的偏移
- **Reward model**：本轮 PPO 阶段冻结的 $r_\phi$，对完整回答打分
- **Critic / value model**：学习 $V_\psi(s_t)$，估计从这个前缀继续生成的预期回报

这些是逻辑角色，不强制对应四份独立物理模型。为了算 PPO 的新旧概率比，还要保存生成这批数据时的策略 $\pi_{\mathrm{old}}$ 的 log-probability。它在这批数据的多次更新期间保持固定，下次采样时再刷新；它不是始终冻结的 reference。

最小流程是：用当前策略采样一批回答 → 保存 old log-probability、reference log-probability 和 value → 构造奖励与 advantage → 用小批量更新若干次 → 重新采样。PPO 可以复用最近一批轨迹多个 epoch，但仍通常归为 on-policy 方法，不能因此把它说成可任意复用陈旧数据的 offline RL。

为保持下文概率比的含义一致，先假设按 $\pi_{\mathrm{old}}$ 原分布采样。若实际用了 temperature 或 top-p 等变换，行为分布就可能不同，概率计算及校正必须与该实现一致，不能拿任意解码结果冒充未经变换的策略样本。

### 3.2 从序列奖励到 advantage

下面取有限回答、**不折扣**的设定，即 $\gamma=1$。在一次 rollout 中，一种常见的奖励构造是

$$
\widetilde r_t
=-\beta\left[
\log\pi_{\mathrm{old}}(y_t\mid s_t)
-\log\pi_{\mathrm{ref}}(y_t\mid s_t)\right]
+\mathbf 1_{\{t=T\}}r_\phi(x,y).
$$

每一步有一个采样的 KL 惩罚项，完整回答的奖励在最后一个有效 token 加入一次。这个构造对应 old 策略采样时的正则化回报；PPO 的多次更新是局部 surrogate 优化，不是在每个更新步精确重新计算整个 $J(\theta)$。

Advantage 衡量的是：**这个动作带来的后续回报，比这个前缀下通常能得到的回报高多少？** Critic 提供比较基准，不能简单解释为“状态不好就把动作概率调低”。

先用 rollout 时的 value 估计构造 TD 误差：

$$
\delta_t=\widetilde r_t+V_{\mathrm{old}}(s_{t+1})-V_{\mathrm{old}}(s_t).
$$

GAE（Generalized Advantage Estimation）把后面的误差加权传回前面，可以倒着算：

$$
\widehat A_t=\delta_t+\lambda\widehat A_{t+1},
\qquad \widehat A_{T+1}=0,\qquad 0\le\lambda\le1.
$$

这里假设回答真正终止时 $V_{\mathrm{old}}(s_{T+1})=0$。GAE 的通用推导、$\lambda$ 两端和偏差/方差讨论统一放在[强化学习基础](/posts/2024-04-30-deep-reinforcement-learning/#3510-从-td-error-到-gae)，本篇只应用到回答 token。

若显式采用折扣，TD 中下一个 value 前乘 $\gamma$，递推中乘 $\gamma\lambda$，同时按折扣目标解释回报。达到长度上限不自动等于任务终止：视为截断时要适当 bootstrap；任务就在此结束时要明确终止奖励。GAE 不能跨 padding 或下一条回答继续累加。[GAE 原论文](https://arxiv.org/abs/1506.02438)

### 3.3 PPO clipping 限制的是什么

对同一条采样回答的同一个前缀，定义 token 概率比

$$
\rho_t(\theta)
=\frac{\pi_\theta(y_t\mid s_t)}{\pi_{\mathrm{old}}(y_t\mid s_t)}
=\exp\!\left(\log\pi_\theta(y_t\mid s_t)-\log\pi_{\mathrm{old}}(y_t\mid s_t)\right).
$$

本篇用所有有效回答 token 的平均定义 actor loss：

$$
\mathcal L_{\mathrm{actor}}
=-\frac{1}{N}\sum_{b,t:m_{b,t}=1}
\min\!\left(
\rho_{b,t}\widehat A_{b,t},
\operatorname{clip}(\rho_{b,t},1-\epsilon,1+\epsilon)\widehat A_{b,t}
\right),
\qquad N=\sum_{b,t}m_{b,t}.
$$

$\epsilon>0$ 控制 surrogate 的 clipping 区间。正负 advantage 的分段解释和数值例子见[通用 PPO 推导](/posts/2024-04-30-deep-reinforcement-learning/#24-ppo)；这里保留对齐实现所需的 token 级 loss。必须取两项最小值；clipping 不是概率比的硬约束，也不是硬性的 KL trust region。[PPO 原论文 §3](https://arxiv.org/abs/1707.06347)

```python
def ppo_actor_loss(new_logp, old_logp, advantage, response_mask, eps=0.2):
    # All tensors: [B, L - 1], aligned with the selected next token.
    valid = response_mask.bool()
    if not valid.any():
        raise ValueError("No response tokens")
    ratio = (new_logp[valid] - old_logp.detach()[valid]).exp()
    adv = advantage.detach()[valid]
    unclipped = ratio * adv
    clipped = ratio.clamp(1.0 - eps, 1.0 + eps) * adv
    return -torch.minimum(unclipped, clipped).mean()
```

这里的新 log-probability 重新前向计算并保留梯度；old log-probability、advantage 和回报目标不反向传播。Critic 另外拟合固定的回报目标，例如 $\widehat R_t=\widehat A_t+V_{\mathrm{old}}(s_t)$，最小化有效回答位置的均方误差。实际 PPO 还需要处理 value loss 权重、可选的 entropy bonus、梯度裁剪、更新次数和 KL 监控；上面只展示 actor 核心，不是完整训练器。

Reference KL 与 PPO clipping 的约束对象不同：前者相对固定参考模型限制偏离，后者让每轮新策略相对采样策略的更新更保守。

## 4. DPO：直接从偏好对更新语言模型

PPO 路线需要不断采样、打分并估计 advantage。DPO（Direct Preference Optimization）先找出 KL 正则目标中“奖励与最优策略”的关系，再把奖励差改写为语言模型的 log-probability 差。

这里不需要沿着离散采样结果直接求导。离散采样本身不能像连续张量那样直接反向传播，但期望奖励可以用 score-function / Policy Gradient 估计梯度，PPO 只是可选方法之一。DPO 则对固定偏好样本的可微 log-probability 直接训练。

### 4.1 KL 正则目标的最优分布

固定 prompt $x$，暂时把策略看成可以自由选择的回答分布，奖励 $r(x,y)$ 和 $\pi_{\mathrm{ref}}$ 固定。求

$$
\max_\pi\sum_y\pi(y\mid x)
\left[r(x,y)-\beta\log\frac{\pi(y\mid x)}{\pi_{\mathrm{ref}}(y\mid x)}\right].
$$

假设 $\beta>0$，策略只在参考分布的支撑集上取正概率，并且下式的归一化常数有限。定义

$$
Z(x)=\sum_y\pi_{\mathrm{ref}}(y\mid x)e^{r(x,y)/\beta},
\qquad
\pi^*(y\mid x)=\frac{\pi_{\mathrm{ref}}(y\mid x)e^{r(x,y)/\beta}}{Z(x)}.
$$

为什么这是最优解？把它代入原目标，可直接整理为

$$
J_x(\pi)=\beta\log Z(x)-\beta D_{\mathrm{KL}}(\pi\|\pi^*).
$$

第一项不随 $\pi$ 改变，第二项不大于零，于是在 $\pi=\pi^*$ 时达到最大。这个结论针对固定奖励下的分布优化，不表示有限容量的神经网络、有限偏好数据和有限训练步数必然找到同一个最优点。

例如仅有两个回答，参考概率为 $(0.5,0.5)$，奖励为 $(\log3,0)$，$\beta=1$。未归一化权重为 $(1.5,0.5)$，所以 $Z=2$、$\pi^*=(0.75,0.25)$。高奖励回答的概率提高，但另一回答并未直接变成零。

### 4.2 同一 prompt 的归一化项怎样消掉

对最优分布的表达式取 log 并移项：

$$
r(x,y)=\beta\log\frac{\pi^*(y\mid x)}{\pi_{\mathrm{ref}}(y\mid x)}+\beta\log Z(x).
$$

对**同一个 $x$** 的两个回答取奖励差，$\beta\log Z(x)$ 抵消。将 $\pi^*$ 用可训练的 $\pi_\theta$ 参数化，记

$$
\Delta_\theta(x,y_w,y_l)
=\left[\log\pi_\theta(y_w\mid x)-\log\pi_\theta(y_l\mid x)\right]
-\left[\log\pi_{\mathrm{ref}}(y_w\mid x)-\log\pi_{\mathrm{ref}}(y_l\mid x)\right].
$$

将这个奖励差代入 Bradley–Terry 偏好似然，得到

$$
\mathcal L_{\mathrm{DPO}}
=-\mathbb E_{(x,y_w,y_l)\sim\mathcal D}
\left[\log\sigma\!\left(\beta\Delta_\theta(x,y_w,y_l)\right)\right].
$$

因此 DPO 比较的是“相对 reference，chosen 比 rejected 多获得了多少概率优势”。chosen 和 rejected 不能来自无关的两个 prompt；否则上述归一化项不能这样消掉。参考模型保持冻结，数据中的回答必须在其支撑集内，tokenizer、对话模板和回答边界也要与策略计算对齐。[DPO 原论文 §4 与附录 A](https://arxiv.org/html/2305.18290v3#S4)

注意 $\beta$ 的位置和符号：原奖励目标是**减** $\beta\mathrm{KL}$，最优分布指数中是 $r/\beta$，DPO 的 sigmoid 内则是**乘** $\beta$，并按 chosen 减 rejected 排序。对固定奖励而言增大 $\beta$ 会加强正则；实际 DPO 同时在拟合偏好，训练效果还取决于数据、步数和学习率，不能只凭 $\beta$ 推断每次更新后的 KL。

### 4.3 最小 DPO 损失与方向检查

```python
def dpo_loss(policy_chosen, policy_rejected,
             reference_chosen, reference_rejected, beta=0.1):
    # Each input: [B], SUM of response token log-probabilities.
    policy_gap = policy_chosen - policy_rejected
    reference_gap = (reference_chosen - reference_rejected).detach()
    preference_logit = beta * (policy_gap - reference_gap)
    return -F.logsigmoid(preference_logit).mean()
```

先用第 1 节的函数分别计算 chosen、rejected 的有效 token log-probability，再沿回答维求和。Reference 前向应在禁用梯度的上下文中执行；这里只用 `detach()` 强调损失不更新 reference，事后 detach 并不能省掉已经构造的计算图。做一次参数更新时，保留 policy 的计算图，调用 `loss.backward()` 和 `optimizer.step()`，并按训练循环清理或累积梯度。

记 $z=\beta\Delta_\theta$，一对数据的 loss 为 $-\log\sigma(z)$，其导数为

$$
\frac{\partial\mathcal L}{\partial\Delta_\theta}
=-\beta\sigma(-\beta\Delta_\theta)<0.
$$

因此，优化时希望增大 chosen 相对 rejected 的 log-probability 优势。策略与 reference 完全相同时，$\Delta_\theta=0$，loss 为 $\log2$；例如 $\beta=0.1,\Delta_\theta=2$ 时，loss 约为 $0.5981$，更低。但参数共享、其他样本和归一化都会共同作用，不能保证一次 batch 更新后每条 chosen 的绝对概率都增加。

### 4.4 DPO 省去了什么，又没有省去什么

标准离线 DPO 直接遍历已收集的偏好数据，不需要在每个训练步重新生成回答，也不需要单独的 reward model 和 critic。它仍需要可靠的偏好标签、固定 reference、正确的序列概率计算与独立评估；不能把“省去显式奖励模型”理解为“不需要反馈”。偏好也可来自其他来源，但不能一概称为人类标注。

PPO 的训练样本主要由近期策略生成，DPO 的期望主要由已有偏好数据集决定。这是两条流程的重要差别：一个在训练中不断获取当前策略的回答，另一个受离线数据覆盖范围约束。DPO 的解析关系不是“对任意旧数据都精确等价于在线 RLHF”的保证；参考分布、偏好模型假设、数据质量和模型表达能力都会影响实际结果。

## 5. 实现前的核对清单

1. **概率对象**：PPO ratio 是相同前缀下采样 token 的新旧概率比；DPO 用整段有效回答的 log-probability 之和
2. **mask 与终止**：排除 prompt 的直接损失和 padding；明确 EOS、截断、空回答和多轮 assistant 边界；GAE 不跨样本传播
3. **模型角色**：reference 固定，old 随 rollout 刷新；奖励模型给完整回答评分，critic 估计前缀的未来回报
4. **梯度对象**：更新 policy，PPO 中另训 critic；old log-probability、advantage、reference 及固定奖励目标不反传
5. **数值与评估**：用 `log_softmax`、`logsigmoid`；观察 reward、KL、回答长度和独立质量指标，不用训练 loss 单独证明“对齐成功”

本文代码是损失计算的教学片段。Python 语法和独立 CPU 数值例子可以分别检查，但不代表完成了 PyTorch/GPU 训练、分布式训练或完整 rollout 验证。工程入口可继续读 [SFT 训练实战](/posts/2026-06-16-llm-mem-opt-5-sft/)，通用策略梯度推导见 [Deep Reinforcement Learning](/posts/2024-04-30-deep-reinforcement-learning/)。

## 参考资料

- Ouyang et al., [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155), 2022
- Schulman et al., [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347), 2017
- Schulman et al., [High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438), 2015 / ICLR 2016
- Rafailov et al., [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290), 2023

学习时也参考了 [Umar Jamil 的 RLHF 课程](https://www.youtube.com/watch?v=qGyFrqc34yc)；本文的关键步骤用公式展开，推导依据以上原论文。
