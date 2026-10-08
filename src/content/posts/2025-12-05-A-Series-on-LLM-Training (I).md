---
title: LLM 对齐入门：RLHF、PPO 与 DPO
published: 2025-12-05
description: 从 SFT 与偏好数据讲起，推导奖励模型、KL 正则、token 级 PPO 和 DPO 的训练目标，并给出损失计算代码和数值例子。
category: Deep Learning
tags:
- reinforcement learning
- deep learning
- nlp
- llm
draft: false
---

<span id="0-从示范答案到偏好学习"></span>

## 0. 模型学会回答后，怎样让回答更合适

语言模型虽然能续写文本，回答却未必符合任务要求。SFT（Supervised Fine-Tuning）用示范答案教模型怎么完成任务；偏好学习进一步比较同一个问题的不同回答，告诉模型哪一种更合适。

下文依次介绍 **SFT → 奖励模型与 KL → token 级 PPO → DPO**。RLHF（Reinforcement Learning from Human Feedback）是一套利用人类反馈的训练流程；PPO 是其中可以采用的策略优化算法，DPO 则是直接用偏好对训练模型的方法。本文会分别说明它们用什么数据、训练哪个模型，以及怎样计算损失。

第 1 节先用下一 token 的概率解释训练损失，再进入偏好学习；推导涉及条件概率、反向传播与梯度下降。[通用强化学习](/posts/2024-04-30-deep-reinforcement-learning/) 是补充 Policy Gradient、value function 和 GAE 的选读，不是学习 SFT 的必修课。数据整理、LoRA、packing 与训练显存问题见 [SFT 训练实战](/posts/2026-06-16-llm-mem-opt-5-sft/)；整体顺序见 [LLM 知识体系导航](/posts/2026-06-03-llm-knowledge-map/)。

下面统一用 $x$ 表示 prompt，$y=(y_1,\ldots,y_T)$ 表示回答 token 序列，$\pi_\theta$ 表示参数为 $\theta$ 的语言模型，所有 $\log$ 均为自然对数。回答包含按数据约定计入的结束 token；padding 不是回答的一部分。

<span id="1-sft先学会模仿示范答案"></span>

<span id="1-sft-与回答的序列概率"></span>

## 1. 从单个 token 的概率算到整条回答

先看 token 概率如何决定训练损失。模型在每个位置为词表中的候选输出 logits；softmax 把它们变为和为 1 的概率。若示范要求下一个 token 是「蓝」，模型给它的概率是 $0.2$，这一步的损失就是 $-\log0.2\approx1.609$；概率提高到 $0.8$，损失变成约 $0.223$。反向传播计算哪些参数影响这个损失，梯度下降据此调整参数。SFT 并不要求模型先自由生成出正确答案：训练时把示范前缀送入，逐位置预测其下一个 token，这称为 teacher forcing。

SFT 的数据格式、token 平均损失、attention mask 与 loss mask 见 [SFT 数据与损失范围](/posts/2026-06-16-llm-mem-opt-5-sft/#13-shiftpadding-mask-和-response-mask-各管什么)。这里讨论偏好学习接下来需要的计算：**怎样得到整条回答的概率**。

令 $x$ 为 prompt，$y=(y_1,\ldots,y_T)$ 为包含约定结束 token 的回答，则

$$
\log\pi_\theta(y\mid x)
=\sum_{t=1}^{T}\log\pi_\theta(y_t\mid x,y_{<t}).
$$

下面采用 response-only 计分：prompt 仍作为输入，padding 不属于回答。

例如回答有两个 token，模型给出的条件概率分别是 $0.5$ 和 $0.2$，整条回答概率为 $0.5\times0.2=0.1$，log-probability 为 $\log0.5+\log0.2=\log0.1$。第二个概率的条件已包含第一个 token；不是把两个无条件概率相乘。使用 log 是因为长回答的概率乘积非常小，而取 log 后可以稳定地做加法。

SFT 用示范答案训练；DPO 则要比较 chosen/rejected 两条回答在当前模型和参考模型下的分数，所以需要保留逐 token log-probability 与有效位置 mask。

<span id="11-从-logits-取出正确的-token-log-probability"></span>

### 1.1 从 logits 计算目标 token 的对数概率

假设一个 batch 有 $B$ 条样本，每条拼接后的 prompt 与回答补齐到长度 $L$，logits 形状为 $[B,L,V]$，labels 为 $[B,L]$，$V$ 是词表大小。labels 中 prompt 和 padding 设为 `-100`，回答位置保留 token ID。位置 $i$ 的 logits 预测位置 $i+1$ 的 label，必须错开一位：

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

<span id="2-奖励模型把偏好变成可优化的分数"></span>

## 2. 奖励模型与 KL 正则

<span id="21-偏好对与奖励损失"></span>

### 2.1 用两条回答的偏好训练奖励模型

从这里开始，$\mathbb E_{z\sim\mathcal D}[f(z)]$ 表示按数据分布抽取样本后求平均；训练代码通常用一个 batch 的均值近似它。下标 $\theta$ 表示要改进的语言模型参数，$\phi$ 表示奖励模型参数，后文的 $\psi$ 表示 critic 参数，三个下标分别对应不同模型的权重。

偏好数据写成 $(x,y_w,y_l)$：在同一个 prompt $x$ 下，标注者更喜欢 $y_w$（chosen），相对不喜欢 $y_l$（rejected）。偏好依据应由任务标准决定，例如正确性、相关性和表达要求。仅凭语言是否自然，还不足以作出这些判断。

奖励模型 $r_\phi(x,y)$ 给整条回答打一个分数。采用 Bradley–Terry 偏好模型时，

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

一种常见实现是在 Transformer 的序列表示上接一个输出标量的预测头（scalar head）。奖励模型对**完整回答**评分，value model 则估计**某个前缀之后的未来回报**；即使网络结构相似，这两个角色也不同。经典 SFT、偏好标注、奖励学习、PPO 的阶段划分可参阅 [InstructGPT 原论文 §3.1](https://arxiv.org/abs/2203.02155)。

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

<span id="3-ppo让奖励影响逐-token-的生成"></span>

## 3. PPO 的 token 级策略更新

<span id="31-四种角色和一次-rollout"></span>

### 3.1 策略、参考模型、奖励模型与 Critic

先沿图追踪**同一条生成回答**，再区分各模型的职责。下面采用 RLHF Book 中已明确标出参考模型概率计算的版本，并补上中文定位标注。

![RLHF 数据流中文定位：①策略生成、②参考概率、③奖励评分、④KL 惩罚、⑤更新策略](/images/llm-series/annotated/rlhfbook-overview-annotated.png)

*来源：Nathan Lambert，[RLHF Book：Policy Gradients](https://rlhfbook.com/c/11-policy-gradients)，基于 Hugging Face RLHF 教程的修订图；[未改动原图](/images/llm-series/sourced/rlhfbook-overview.png)。本图添加中文定位标注，按 [CC BY-NC-SA 4.0](/images/llm-series/sourced/rlhfbook-license-chapters.txt)共享。*

- **① 策略生成回答**：中间模型根据问题采样一条回答 $y$；它是最终要更新的模型。
- **② 参考模型只算概率**：左侧冻结模型给这同一条回答计算 log-probability，不另生成一条文本与它比较。
- **③ 奖励模型评分**：右侧模型评价完整回答；图中的 $r_\theta$ 对应本文的 $r_\phi$，两处参数记号不同。
- **④ KL 偏移惩罚**：策略与参考模型的概率用于约束行为偏移；图中的 $\lambda_{\mathrm{KL}}$ 对应本文的 $\beta$。
- **⑤ 合成信号并更新策略**：奖励和惩罚共同影响更新。图省略了 Critic、old policy 快照和 token 级 advantage，不能把它当作完整 PPO 实现；下面补齐这些分工。

把生成看成一个有限长度的决策过程：状态 $s_t=(x,y_{<t})$，动作 $a_t=y_t$，策略是 $\pi_\theta(y_t\mid s_t)$。

- **Actor / policy**：生成 token，也是最终要改进的语言模型
- **Reference**：冻结的 $\pi_{\mathrm{ref}}$，用于衡量相对初始行为的偏移
- **Reward model**：本轮 PPO 阶段冻结的 $r_\phi$，对完整回答打分
- **Critic / value model**：学习 $V_\psi(s_t)$，估计从这个前缀继续生成的预期回报

这是四种分工，不一定要分别加载四份独立模型。为了算 PPO 的新旧概率比，还要保存生成这批数据时的策略 $\pi_{\mathrm{old}}$ 的 log-probability。它在这批数据的多次更新期间保持固定，下次采样时再刷新。Reference 则在整个训练阶段保持冻结。

一次训练循环可以概括为：用当前策略采样一批回答 → 保存 old log-probability、reference log-probability 和 value → 构造奖励与 advantage → 用小批量更新若干次 → 重新采样。PPO 可以复用最近一批轨迹多个 epoch，但仍通常归为 on-policy 方法，不能因此把它说成可任意复用陈旧数据的 offline RL。

为保持下文概率比的含义一致，先假设按 $\pi_{\mathrm{old}}$ 原分布采样。若实际用了 temperature 或 top-p 等变换，行为分布就可能不同，概率计算及校正必须与该实现一致。经过变换的解码结果，不能直接作为原始策略分布的样本计算。

<span id="32-从序列奖励到-advantage"></span>

<span id="32-序列奖励与-token-优势估计"></span>

### 3.2 回答末尾的奖励怎样影响前面的 token

下面取有限回答、**不折扣**的设定，即 $\gamma=1$。在一次 rollout 中，一种常见的奖励构造是

$$
\widetilde r_t
=-\beta\left[
\log\pi_{\mathrm{old}}(y_t\mid s_t)
-\log\pi_{\mathrm{ref}}(y_t\mid s_t)\right]
+\mathbf 1_{\{t=T\}}r_\phi(x,y).
$$

每一步有一个采样的 KL 惩罚项，完整回答的奖励在最后一个有效 token 加入一次。这个构造对应 old 策略采样时的正则化回报；PPO 的多次更新优化的是局部替代目标（surrogate）：用这一批已采样前缀、固定 advantage 和新旧概率比，近似指导当前这轮参数更新，而非在每个更新步精确重新计算整个 $J(\theta)$。

Advantage 衡量的是：**这个动作带来的后续回报，比这个前缀下通常能得到的回报高多少？** Critic 提供这个比较基准。因此，动作是否值得鼓励取决于相对于预期的回报，不能只看状态本身的价值高低。

为什么完整回答最后才打分，前面的 token 仍能学习？一次 token 选择改变了后续生成路径，所以前面动作的评价也要考虑后续回报。最简单地把后面奖励全部相加，就得到该位置的实际回报；减去 critic 对这个前缀的预期，得到“本次回报比预期高多少”。GAE 据此混合多步估计，把后续的估计误差逐步传回前面的 token。这样，每个 token 获得的学习信号可以不同，而非共用一个末尾分数。

TD（Temporal Difference，时序差分）误差比较两种估计：“眼前奖励加下一个前缀的 value”，以及“当前前缀的 value”。先用 rollout 时记录的 value 计算：

$$
\delta_t=\widetilde r_t+V_{\mathrm{old}}(s_{t+1})-V_{\mathrm{old}}(s_t).
$$

GAE（Generalized Advantage Estimation）把后面的误差加权传回前面，可以倒着算：

$$
\widehat A_t=\delta_t+\lambda\widehat A_{t+1},
\qquad \widehat A_{T+1}=0,\qquad 0\le\lambda\le1.
$$

这里假设回答真正终止时 $V_{\mathrm{old}}(s_{T+1})=0$。

手算两个 token：假设已经把 KL 项计入的奖励为 $(0,2)$，两个前缀的旧 value 为 $(0.5,0.8)$，终止后 value 为 0。于是 $\delta_2=2-0.8=1.2$，$\delta_1=0+0.8-0.5=0.3$。取 $\lambda=1$，倒推得 $\widehat A_2=1.2$、$\widehat A_1=0.3+1.2=1.5$；首个动作因此也得到正信号。取 $\lambda=0$ 则只用各自的一步 TD 误差。介于两者之间时，后续误差按距离衰减。

GAE 的通用推导、$\lambda$ 两端和偏差/方差讨论见[强化学习基础](/posts/2024-04-30-deep-reinforcement-learning/#3510-从-td-error-到-gae)，本篇只应用到回答 token。

若显式采用折扣，TD 中下一个 value 前乘 $\gamma$，递推中乘 $\gamma\lambda$，同时按折扣目标解释回报。达到长度上限不自动等于任务终止：若只是截断，就可能还存在未观察到的后续回报，要用截断点的 value 接上估计，这称为 bootstrap；任务就在此结束时要明确终止奖励。GAE 不能跨 padding 或下一条回答继续累加。[GAE 原论文](https://arxiv.org/abs/1506.02438)

<span id="33-ppo-clipping-限制的是什么"></span>

<span id="33-ppo-概率比与裁剪目标"></span>

### 3.3 PPO 的 token 概率比与裁剪目标

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

$\epsilon>0$ 控制 surrogate 的 clipping 区间。

先看要最大化的括号内部分，再记住代码取负号作为 loss。设 $\epsilon=0.2$：

- $\widehat A=2$、$\rho=1.5$ 时，原项是 3，裁剪项是 2.4，取最小值 2.4；继续提高这个正优势动作的概率，不再从本样本获得额外奖励
- $\widehat A=-2$、$\rho=0.5$ 时，原项是 -1，裁剪项是 -1.6，取 -1.6；继续压低这个负优势动作的概率，也不再提高该目标
- $\widehat A=-2$、$\rho=1.5$ 时，取 $\min(-3,-2.4)=-3$；提高负优势动作的概率仍受惩罚，不会被裁剪掉

这里的 $m_{b,t}$ 是回答位置的 0/1 mask，$N$ 是有效 token 总数，$\rho$ 比较的始终是同一前缀下同一个 token 的新旧概率。裁剪限制了单个样本对目标函数的贡献，参数共享却仍可能使某些概率比越过区间，所以它不是概率比或 KL 的硬约束。更完整的分段推导见[通用 PPO 推导](/posts/2024-04-30-deep-reinforcement-learning/#24-ppo)。[PPO 原论文 §3](https://arxiv.org/abs/1707.06347)

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

Reference KL 衡量当前模型偏离固定参考模型有多远；PPO clipping 则比较本轮更新前后的策略，避免在同一批样本上改得太多。两者比较的模型不同。

## 4. DPO：直接从偏好对更新语言模型

采用 PPO 训练时，需要反复采样回答、计算奖励并估计 advantage。DPO（Direct Preference Optimization）先推导 KL 正则目标下奖励与最优策略的关系，再把奖励差改写为语言模型的 log-probability 差。

原论文把这两种训练路径并排画出。图中的 $x$ 是同一个问题，$y_w$ 是人类更偏好的回答，$y_l$ 是较不偏好的回答；奖杯只标记偏好胜者，不是环境返回的数值奖励。

![DPO 论文图的中文阅读标注：①偏好对、②奖励模型、③采样评分循环、④直接偏好训练、⑤省略的参考模型](/images/llm-series/annotated/dpo-rlhf-comparison-annotated.png)

*来源：[Rafailov 等，DPO 论文 Figure 1](https://arxiv.org/html/2305.18290v3#S1.F1)；[原图](/images/llm-series/sourced/dpo-rlhf-comparison.png)，[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)。本图增加中文阅读标注。*

- **① 同题偏好对**：两条回答对应同一个问题 $x$，$y_w$ 优于 $y_l$；奖杯表示偏好标签，不是奖励模型已经给出的数值。右侧 DPO 使用的也是这种比较数据。
- **② 先学奖励模型**：左侧先用偏好数据训练 reward model，使胜者相对败者获得更高的分数。
- **③ 采样与评分循环**：LM policy 采样回答，奖励模型评分，再由强化学习方法更新策略；上下两条弧线分别强调评分和采样的方向。
- **④ 直接学偏好**：右侧把偏好对用于语言模型的可微损失，省去显式奖励模型与这条在线采样训练循环；它不等于只对胜者做 SFT。
- **⑤ 参考模型未画出**：这条标注提示原图省略了训练细节，不是新增的数据路径。标准 DPO 仍需固定 reference 的对数概率；左侧 KL 正则化 RLHF 同样使用参考模型，具体定义见正文。

先读粉色左半边：偏好对经 maximum likelihood（最大似然）训练 reward model；之后 LM policy 采样回答，奖励模型为回答打分，两条弧形箭头表示反复采样和反馈。再读浅蓝右半边：同类偏好对直接用于优化 final LM，不再显式维护这条奖励模型与策略采样循环。

这里“直接”不等于只对胜者做普通 SFT。DPO 的损失还要比较胜者与败者在当前模型、固定参考模型下的对数概率；图省略了参考模型、KL 约束和 PPO 的 Critic，不能拿它代替完整训练定义。下文会逐项推导这些量如何进入损失。

这里不需要沿着离散采样结果直接求导。离散采样本身不能像连续张量那样直接反向传播，但期望奖励可以用 score-function / Policy Gradient 估计梯度，PPO 只是可选方法之一。DPO 则对固定偏好样本的可微 log-probability 直接训练。

<span id="41-kl-正则目标的最优分布"></span>

### 4.1 加上 KL 正则后，最优回答分布是什么

固定 prompt $x$，暂时把策略看成可以自由选择的回答分布，奖励 $r(x,y)$ 和 $\pi_{\mathrm{ref}}$ 固定。求

$$
\max_\pi\sum_y\pi(y\mid x)
\left[r(x,y)-\beta\log\frac{\pi(y\mid x)}{\pi_{\mathrm{ref}}(y\mid x)}\right].
$$

假设 $\beta>0$，策略只在参考分布的支撑集上取正概率，并且下式的归一化常数有限。“支撑集”在这里就是参考模型赋予正概率的回答集合；归一化常数 $Z(x)$ 把重新加权后的数值调整到总和为 1。定义

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

<span id="42-同一-prompt-的归一化项怎样消掉"></span>

### 4.2 用奖励差推导 DPO 损失

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

因此 DPO 比较的是“当前模型对 chosen 与 rejected 的对数概率差，比参考模型的这一差值高多少”。chosen 和 rejected 必须对应同一个 prompt，归一化项才能抵消。参考模型保持冻结，数据中的回答必须在其支撑集内，tokenizer、对话模板和回答边界也要与策略计算对齐。[DPO 原论文 §4 与附录 A](https://arxiv.org/html/2305.18290v3#S4)

注意 $\beta$ 的位置和符号：原奖励目标是**减** $\beta\mathrm{KL}$，最优分布指数中是 $r/\beta$，DPO 的 sigmoid 内则是**乘** $\beta$，并按 chosen 减 rejected 排序。对固定奖励而言增大 $\beta$ 会加强正则；实际 DPO 同时在拟合偏好，训练效果还取决于数据、步数和学习率，不能只凭 $\beta$ 推断每次更新后的 KL。

<span id="43-最小-dpo-损失与方向检查"></span>

<span id="43-dpo-损失实现与梯度方向"></span>

### 4.3 实现 DPO 损失，检查更新方向

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

<span id="44-dpo-省去了什么又没有省去什么"></span>

<span id="44-离线-dpo-的训练流程与数据要求"></span>

### 4.4 离线 DPO 怎样训练，需要哪些数据

标准离线 DPO 直接遍历已收集的偏好数据，不需要在每个训练步重新生成回答，也不需要单独的 reward model 和 critic。它仍需要可靠的偏好标签、固定 reference、正确的序列概率计算与独立评估；省去奖励模型，并没有省去收集和检查反馈数据的工作。偏好也可来自其他来源，但不能一概称为人类标注。

PPO 的训练样本主要由近期策略生成，DPO 的期望主要由已有偏好数据集决定。这是两条流程的重要差别：PPO 在训练中不断生成新回答，离线 DPO 主要学习已有数据中的回答比较。上述解析关系有相应假设，不能据此认为任意离线数据上的 DPO 都与在线 RLHF 精确等价；参考分布、偏好模型假设、数据质量和模型表达能力都会影响实际结果。

<span id="5-实现前的核对清单"></span>

## 5. 写训练代码时要检查什么

1. **概率计算**：PPO ratio 是相同前缀下采样 token 的新旧概率比；DPO 用整段有效回答的 log-probability 之和
2. **mask 与终止**：排除 prompt 的直接损失和 padding；明确 EOS、截断、空回答和多轮 assistant 边界；GAE 不跨样本传播
3. **模型角色**：reference 固定，old 随 rollout 刷新；奖励模型给完整回答评分，critic 估计前缀的未来回报
4. **参与梯度更新的量**：更新 policy，PPO 中另训 critic；old log-probability、advantage、reference 及固定奖励目标不反传
5. **数值与评估**：用 `log_softmax`、`logsigmoid`；观察 reward、KL 和回答长度，并用独立评估检查回答质量，不能只看训练 loss

本文代码展示损失计算；完整训练还需接入模型前向、数据采样、优化器更新与评估，PPO 还需完整的 rollout 流程。训练配置可继续读 [SFT 训练实战](/posts/2026-06-16-llm-mem-opt-5-sft/)，通用策略梯度推导见 [Deep Reinforcement Learning](/posts/2024-04-30-deep-reinforcement-learning/)。

## 参考资料

- Ouyang et al., [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155), 2022
- Schulman et al., [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347), 2017
- Schulman et al., [High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438), 2015 / ICLR 2016
- Rafailov et al., [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290), 2023

学习时也参考了 [Umar Jamil 的 RLHF 课程](https://www.youtube.com/watch?v=qGyFrqc34yc)；本文的关键步骤用公式展开，推导依据以上原论文。
