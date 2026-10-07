---
title: Deep Reinforcement Learning Series
published: 2024-04-30
description: 系统学习强化学习中的策略梯度与价值方法两大类参数更新算法，涵盖 PPO、Q-Learning 等核心方法的原理与公式推导。
category: Deep Learning
tags:
- reinforcement learning
- deep learning
draft: false
---
## 0. 前言

强化学习面对的问题是：一次动作可能很久以后才带来收益，怎样知道现在该增加哪个动作的概率？一种思路直接调整策略，另一种先学会给状态和动作估值，再据此选择动作。本篇从这两条路线出发，串起 Policy Gradient、PPO、Q-Learning、TD(λ) 和 GAE。

阅读需要期望、条件概率、求导和梯度下降的基础，不要求先学过强化学习。第一次读可以先看策略梯度、价值函数和 TD，再回来看 PPO 中的 advantage 是怎样估计出来的。

这是 [LLM 知识体系导航](/posts/2026-06-03-llm-knowledge-map/) 的**通用强化学习选读支线**，可作为理解对齐方法的前置知识。本篇不等于 RLHF 教程：语言模型的奖励构造、参考模型、token 级 KL 和训练流程还需要另外讨论，也不必为了学习 SFT 先读完所有 DP 和 TD 推导。

## 1. 总览和相关概念

### 1.1 总览

- **Policy Based**：直接学习策略 $\pi_\theta(a\mid s)$，它给出各个动作的概率。Actor 是这个策略模型，不保证每次都选最优动作，训练时通常还需要探索。
- **Value Based**：学习状态价值 $V(s)$ 或动作价值 $Q(s,a)$，再根据估值选择动作。它不一定有独立的 Actor 网络。
- **Actor–Critic**：Actor 学策略，Critic 提供价值估计，帮助判断动作好不好。这两种角色可以使用不同网络，也可以共享部分参数。

<img src="https://s2.loli.net/2024/04/30/4Ihfg5X8FtVEbsS.png" alt="强化学习方法分类示意" width="300" height="200" />

原配图来源：[David Silver’s RL Course](https://www.davidsilver.uk/teaching/)。

### 1.2 状态、动作和一段经历

以游戏为例，observation 是你看到的画面，action 是移动或开火，reward 是行动后得到的即时分数。**一帧画面不一定是完整 state**：例如速度可能要结合前后帧才能判断。下文假设 $s_t$ 已包含预测未来所需的信息，满足 Markov 性；部分可观测问题可能需要历史、堆叠帧或记忆模型。

本文统一从 $t=0$ 开始计数：在 $s_t$ 选择 $a_t$，环境返回奖励 $r_t$ 和下一状态 $s_{t+1}$。有些教材将同一个奖励写成 $R_{t+1}$，阅读原配图时要注意这个下标差异。

一次真正结束的游戏是一个 episode，其 trajectory 写作：

$$
\tau=(s_0,a_0,r_0,s_1,a_1,r_1,\ldots,s_{T-1},a_{T-1},r_{T-1},s_T).
$$

这里有 $T$ 次动作。采集器也可能只截取其中一段 rollout；**采样片段结束，不代表任务结束**。

### 1.3 先固定目标和符号

先考虑有限时域 $T$，逐一说明各个折扣项的来源：

$$
R_\gamma(\tau)=\sum_{t=0}^{T-1}\gamma^t r_t,
\qquad
J(\theta)=\mathbb E_{\tau\sim p_\theta}[R_\gamma(\tau)],
\qquad 0\leq\gamma\leq1.
$$

$R_\gamma$ 是从起点计算的整段折扣回报，$\gamma$ 表示目标对较晚奖励的权重；奖励本身可以为正，也可以为负。有限时域且奖励可积时允许 $\gamma=1$。持续任务常取 $\gamma<1$ 并要求奖励有界，以保证回报有限。

从第 $t$ 步往后计算、把该步当作新的时间原点的回报则是：

$$
G_t=\sum_{k=t}^{T-1}\gamma^{k-t}r_k
=r_t+\gamma G_{t+1},\qquad G_T=0.
$$

因此 $R_\gamma=G_0$，但一般 $G_t\neq R_\gamma$。这两个量不能使用同一个符号后随意互换。

有限时域中，同一位置在“还剩十步”和“只剩一步”时价值不同。严格写法是 $\pi_\theta(a\mid s,t)$、$V_t^\pi(s)$、$Q_t^\pi(s,a)$。下文策略省略显式的 $t$ 时，可理解为已把时间并入状态；后面的平稳 Bellman 方程也采用这一约定。若不把时间并入状态，有限时域的方程必须保留 $t$ 与 $t+1$。

## 2. Policy Based

### 2.1 要做什么

Actor 的目标是提高 $J(\theta)$，不是把每次采样的动作都变成“正确标签”。直觉上，在相同处境中，带来较好后续结果的动作应该变得更常见；较差的动作应该变得更少见。

难点是环境通常不可微。我们不对游戏引擎求导，而是对“这段经历由当前策略产生的概率”求导。

### 2.2 理论怎么做

#### 2.2.1 轨迹概率与梯度

设初始分布为 $p_0$，环境转移与奖励分布为 $p(s',r\mid s,a)$。假设它们不随 $\theta$ 改变，则：

$$
p_\theta(\tau)=p_0(s_0)\prod_{t=0}^{T-1}
\pi_\theta(a_t\mid s_t,t)\,p(s_{t+1},r_t\mid s_t,a_t).
$$

策略会通过动作改变未来状态的分布；“环境不含参数 $\theta$”不等于“状态与策略无关”。求给定轨迹的对数概率时，先取 log，再求梯度：

$$
\begin{aligned}
\log p_\theta(\tau)
&=\log p_0(s_0)+\sum_{t=0}^{T-1}
\bigl[\log\pi_\theta(a_t\mid s_t,t)+\log p(s_{t+1},r_t\mid s_t,a_t)\bigr],\\
\nabla_\theta\log p_\theta(\tau)
&=\sum_{t=0}^{T-1}\nabla_\theta\log\pi_\theta(a_t\mid s_t,t).
\end{aligned}
$$

假设可交换求导与期望、策略可微且在需要的支撑集上概率为正，利用 $\nabla p=p\nabla\log p$ 得到：

$$
\begin{aligned}
\nabla_\theta J(\theta)
&=\sum_\tau R_\gamma(\tau)\nabla_\theta p_\theta(\tau)\\
&=\mathbb E_{\tau\sim p_\theta}\left[
R_\gamma(\tau)\sum_{t=0}^{T-1}\nabla_\theta\log\pi_\theta(a_t\mid s_t,t)
\right].
\end{aligned}
$$

离散情况写作求和；连续状态或动作相应使用密度和积分。该推导没有要求知道环境的导数。

#### 2.2.2 因果性：去掉动作发生之前的奖励

$t$ 时刻的动作不会改变已经收到的 $r_0,\ldots,r_{t-1}$。更精确地说，给定动作发生前的历史，策略 score 的条件期望为零：

$$
\mathbb E_{a_t\sim\pi_\theta(\cdot\mid s_t,t)}
[\nabla_\theta\log\pi_\theta(a_t\mid s_t,t)]
=\sum_a\nabla_\theta\pi_\theta(a\mid s_t,t)=0.
$$

因此，过去奖励乘以这个 score 后，对所有轨迹取期望为零。保留从 $t$ 开始的奖励即可，且原目标中的折扣仍然是 $\gamma^k$：

$$
\begin{aligned}
\nabla_\theta J(\theta)
&=\mathbb E_{\tau\sim p_\theta}\left[
\sum_{t=0}^{T-1}\nabla_\theta\log\pi_\theta(a_t\mid s_t,t)
\sum_{k=t}^{T-1}\gamma^k r_k\right]\\
&=\mathbb E_{\tau\sim p_\theta}\left[
\sum_{t=0}^{T-1}\gamma^t G_t\nabla_\theta\log\pi_\theta(a_t\mid s_t,t)
\right].
\end{aligned}
$$

**外面的 $\gamma^t$ 不能无条件丢掉。** 内部 $G_t$ 将折扣原点移到 $t$，所以需要外部系数把它换回整段目标的原点。例如 $\gamma=0.9$、$t=2$，未来两次奖励是 $3,4$，则 $G_2=3+0.9\times4=6.6$；它在上述梯度中的回报权重是 $0.9^2\times6.6=5.346$。

#### 2.2.3 Baseline、advantage 与 batch 估计

同一个动作的回报会受环境随机性和后续动作影响。可以减去一个不依赖当前动作的参考值 $b_t(s_t)$，用来降低梯度估计的波动：

$$
\mathbb E_{a_t\sim\pi_\theta}
[b_t(s_t)\nabla_\theta\log\pi_\theta(a_t\mid s_t,t)\mid s_t]=0.
$$

Baseline 不是为了纠正“奖励一定为正”，也不是任取一个 baseline 都能降低方差。证明要求 baseline 在给定状态和时间后不依赖本次动作；严格讨论无偏估计时，可使用采样前固定的 Critic 或独立数据拟合的 baseline。把含本条样本的 batch 平均回报直接作 baseline，不能自动套用这个独立性结论。

对当前策略独立采集 $N$ 条完整轨迹，REINFORCE 的一个 batch 估计是：

$$
\widehat g=\frac1N\sum_{i=1}^{N}\sum_{t=0}^{T-1}
\gamma^t\bigl(G_t^{(i)}-b_t(s_t^{(i)})\bigr)
\nabla_\theta\log\pi_\theta(a_t^{(i)}\mid s_t^{(i)},t),
\qquad
\theta\leftarrow\theta+\eta\widehat g.
$$

$\widehat g$ 是采样估计，满足条件时其期望等于真实梯度；单个有限 batch 不等于期望。若 episode 提前终止，可将之后视作吸收状态、零奖励并补齐到 $T$；实现也可以直接对各条轨迹的真实长度求和。除以轨迹数 $N$ 与除以总 token/步数是不同的归一化，特别是长度可变时要说明采用哪一种。

一个自然的 baseline 是状态价值：

$$
\begin{aligned}
V_t^\pi(s)&=\mathbb E_\pi[G_t\mid S_t=s],\\
Q_t^\pi(s,a)&=\mathbb E_\pi[G_t\mid S_t=s,A_t=a],\\
A_t^\pi(s,a)&=Q_t^\pi(s,a)-V_t^\pi(s).
\end{aligned}
$$

$Q$ 对应“现在选这个动作，再按策略继续”，$V$ 对应“现在也按策略随机选动作”。所以 advantage 表示这个动作比策略的平均选择好多少。一条采样得到的 $G_t$ 是 $Q_t^\pi$ 的估计，不能把随机样本与条件期望写成逐样本相等。

使用精确的 advantage 可以将梯度写为：

$$
\nabla_\theta J(\theta)=\mathbb E_{\tau\sim p_\theta}\left[
\sum_{t=0}^{T-1}\gamma^t A_t^{\pi_\theta}(s_t,a_t)
\nabla_\theta\log\pi_\theta(a_t\mid s_t,t)\right].
$$

Actor–Critic 用网络估计这些值；估计不准确、bootstrap 或截断都可能引入额外偏差。实际用 $-\log\pi_\theta\,\widehat A_t$ 作 actor loss 时，$\widehat A_t$ 在这个策略梯度项中应停止梯度，Critic 通过单独的价值损失学习。

#### 2.2.4 为什么有些公式看不到外层折扣

一种情况是 $\gamma=1$。另一种是折扣已经放进了采样分布。设：

$$
C_T=\sum_{t=0}^{T-1}\gamma^t,\qquad
d_{\theta,T}(t,s)=\frac{\gamma^t\Pr_\theta(S_t=s)}{C_T}.
$$

这是把时间也包含在内的、归一化后的折扣状态访问分布。于是同一梯度可以写成：

$$
\nabla_\theta J(\theta)=C_T\,
\mathbb E_{(t,s)\sim d_{\theta,T},\ a\sim\pi_\theta(\cdot\mid s,t)}
[A_t^{\pi_\theta}(s,a)\nabla_\theta\log\pi_\theta(a\mid s,t)].
$$

这里没有把状态分布对参数的依赖忽略掉；公式来自前面的完整轨迹推导。对无限时域折扣任务，常见归一化因子是 $1-\gamma$。若从 rollout 中均匀抽时间步，就不是按上述 $d_{\theta,T}$ 采样，不能只引用一句“折扣被分布吸收”来删掉权重。

一些算法用折扣优势近似未折扣目标的梯度，采用另一种更新口径。例如 [GAE 原论文第 2 节](https://arxiv.org/abs/1506.02438) 明确从未折扣任务出发，把 $\gamma$ 引入为估计参数。它与本篇一开始定义的 $J$ 应加以区分。

### 2.3 用旧策略的数据更新

**On-policy / off-policy** 比较的是采样行为策略与学习目标策略是否一致；**online / offline** 比较的是学习时是否继续与环境交互。两组概念不能混用。可以在线交互并采用 off-policy 方法，也可以只在固定历史数据上训练。

#### 2.3.1 Importance Sampling

如果只有 $q$ 的样本，却想计算 $p$ 下的期望，可以给样本乘一个概率比：

$$
\mathbb E_{x\sim p}[f(x)]
=\mathbb E_{x\sim q}\left[\frac{p(x)}{q(x)}f(x)\right].
$$

前提是目标分布需要的区域在采样分布下也有支持，即 $p(x)>0$ 时 $q(x)>0$（更一般地，$p$ 对 $q$ 绝对连续），且相关期望存在。采不到的事件不能靠加权补回来。

保持期望不等于保持方差。若二阶矩存在：

$$
\operatorname{Var}_{q}\left[\frac pq f\right]
=\mathbb E_p\left[\frac pq f^2\right]-\bigl(\mathbb E_p[f]\bigr)^2.
$$

当重要样本在 $q$ 下极罕见时，概率比可能很大，估计方差甚至可能无穷。分布接近有时有助于控制方差，但不能普遍保证这一点。

#### 2.3.2 整条轨迹比与单步概率比

设行为策略为 $\mu$，目标策略为 $\pi_\theta$，初始分布和环境相同。环境项抵消后，整条轨迹的比值为：

$$
W_\theta(\tau)=\frac{p_\theta(\tau)}{p_\mu(\tau)}
=\prod_{t=0}^{T-1}\frac{\pi_\theta(a_t\mid s_t,t)}{\mu(a_t\mid s_t,t)}.
$$

因此，一个精确但可能高方差的梯度表达式是：

$$
\nabla_\theta J(\theta)=\mathbb E_{\tau\sim p_\mu}\left[
W_\theta(\tau)R_\gamma(\tau)
\sum_{t=0}^{T-1}\nabla_\theta\log\pi_\theta(a_t\mid s_t,t)\right].
$$

整条轨迹比不能直接换成某一步的 $\pi_\theta(a_t\mid s_t,t)/\mu(a_t\mid s_t,t)$。单步比只校正**给定状态下的动作分布**；此前动作已经影响了当前状态，之后动作还会影响回报。精确的 off-policy 推导可能需要轨迹比、逐决策的连乘比或状态分布校正，而不是假定状态分布与策略无关。

PPO 使用的则是一个局部 surrogate。固定旧策略 $\pi_{\rm old}$ 及其优势估计，定义：

$$
\rho_t(\theta)=\frac{\pi_\theta(a_t\mid s_t,t)}{\pi_{\rm old}(a_t\mid s_t,t)},
\qquad
L_{\rm surr}(\theta)=\mathbb E_{\tau\sim p_{\rm old}}\left[
\sum_{t=0}^{T-1}\gamma^t\rho_t(\theta)A_t^{\pi_{\rm old}}(s_t,a_t)\right].
$$

旧策略采样分布、旧优势保持固定，在 $\theta=\theta_{\rm old}$ 时，$\rho_t=1$，且 $\nabla\rho_t=\nabla\log\pi_\theta$，所以此处的 surrogate 梯度与前述 $J$ 的梯度一致。离开旧策略后，这不再是 $J(\theta)$ 的精确重写，更新幅度也需要控制。

### 2.4 PPO

PPO 的直觉是：一批数据只能告诉我们旧策略附近的大致改进方向，不要反复优化这一批数据后把策略推得太远。[PPO 原论文](https://arxiv.org/abs/1707.06347) 讨论了 KL penalty 与 clipping 两种形式。

先把本文的 batch 加权约定写清楚：

$$
\widehat{\mathbb E}_w[f_t]=\frac1N\sum_{i=1}^{N}\sum_{t=0}^{T_i-1}\gamma^t f_t^{(i)}.
$$

它沿用第 2.2 节的轨迹平均。实现中常见均匀 timestep/minibatch 平均，应当说明所用目标和权重，不把两种口径默认为相同。

KL penalty 形式可写为：

$$
L_{\rm KL}(\theta)=\widehat{\mathbb E}_w\left[
\rho_t(\theta)\widehat A_t-\beta\,
D_{\rm KL}\bigl(\pi_{\rm old}(\cdot\mid s_t,t)\Vert\pi_\theta(\cdot\mid s_t,t)\bigr)
\right],\qquad\beta>0.
$$

KL 比较的是相同状态下两种动作分布，不是直接比较网络参数。KL 的方向及其与交叉熵的关系可参见 [KL 散度](/posts/2024-04-19-kullback-leibler-divergence/)。

更常见的 PPO-Clip 最大化：

$$
L_{\rm clip}(\theta)=\widehat{\mathbb E}_w\left[
\min\left(\rho_t(\theta)\widehat A_t,
\operatorname{clip}(\rho_t(\theta),1-\epsilon,1+\epsilon)\widehat A_t\right)
\right],\qquad 0<\epsilon<1.
$$

- $\widehat A_t>0$：该动作值得鼓励；当概率比大于 $1+\epsilon$ 时，单个样本的目标不再因继续增加该比值而提高。
- $\widehat A_t<0$：该动作值得抑制；当概率比小于 $1-\epsilon$ 时，单个样本的目标不再因继续降低该比值而提高。
- 朝错误方向变化时，$\min$ 仍保留惩罚，不会把两边一律截成平坦区域。

例如 $\epsilon=0.2$、$\widehat A=2$、$\rho=1.4$，该项为 $\min(2.8,2.4)=2.4$；若 $\widehat A=-2$、$\rho=0.6$，该项为 $\min(-1.2,-1.6)=-1.6$。

**Clip 修改的是目标函数，没有把真实概率比硬限制在区间内。** 参数共享、其他样本及价值/熵损失的梯度都可能让某些比值越界；它也不提供严格的 KL 上界。实现可以另外监控 KL、限制训练轮数或使用提前停止。

一次典型更新是：用旧策略采样，固定旧 log-prob 与优势，做有限次 minibatch 更新，然后重新采样。Critic 通过回报目标学习，优势可用后文的 GAE 估计。PPO 并不因此成为可以无限复用任意陈旧 replay buffer 的算法。

## 3. Value Based

### 3.1 价值函数解决什么问题

第 2 节的 $V$、$Q$、$A$ 都以未来回报为单位，是标量。若有 $m$ 个离散动作，网络可以一次输出长度为 $m$ 的 Q 向量，但其中每个分量才是某个动作的标量价值。

在平稳、折扣 MDP 中，省略时间后：

$$
V^\pi(s)=\sum_a\pi(a\mid s)Q^\pi(s,a),
\qquad
Q^\pi(s,a)=\mathbb E[r_t+\gamma V^\pi(S_{t+1})\mid S_t=s,A_t=a].
$$

这里通过取期望平均掉环境随机性，并不是每一条实际转移都满足同一个等式。最优价值定义为：

$$
V^*(s)=\max_\pi V^\pi(s),\qquad Q^*(s,a)=\max_\pi Q^\pi(s,a).
$$

以下最优性与收敛讨论采用有限状态/动作、奖励有界、$\gamma<1$ 的标准表格型条件。有限时域则通过 $V_T=0$ 反向计算；不能把持续任务中 $\gamma=1$ 的收敛结论直接沿用。

### 3.2 Bellman equation

#### 3.2.1 固定策略：对动作取平均

将回报拆成“本步奖励 + 后续回报”，得到：

$$
\begin{aligned}
V^\pi(s)
&=\mathbb E_\pi[r_t+\gamma G_{t+1}\mid S_t=s]\\
&=\sum_a\pi(a\mid s)\sum_{s',r}p(s',r\mid s,a)
\bigl[r+\gamma V^\pi(s')\bigr],\\
Q^\pi(s,a)
&=\sum_{s',r}p(s',r\mid s,a)
\left[r+\gamma\sum_{a'}\pi(a'\mid s')Q^\pi(s',a')\right].
\end{aligned}
$$

这是任意给定策略的 Bellman expectation equation，不只属于最优策略。终止状态的后续价值设为零。

#### 3.2.2 最优策略：对动作取最大价值

最优决策不再按固定策略平均，而是选择价值最大的动作：

$$
\begin{aligned}
V^*(s)&=\max_a\sum_{s',r}p(s',r\mid s,a)
\bigl[r+\gamma V^*(s')\bigr],\\
Q^*(s,a)&=\sum_{s',r}p(s',r\mid s,a)
\left[r+\gamma\max_{a'}Q^*(s',a')\right].
\end{aligned}
$$

第二行的 $\max_{a'}$ 不能省略。还要区分：$\max_a Q(s,a)$ 返回一个**分数**，$\arg\max_a Q(s,a)$ 返回一个**动作**。奖励可以加上折扣后的分数，不能加上动作编号。

有限时域的对应式是 $V_t^*(s)=\max_a\mathbb E[r_t+\gamma V_{t+1}^*(S_{t+1})\mid s,a]$，以 $V_T^*=0$ 结束。

### 3.3 Dynamic Programming Based

DP 假设已知环境模型，能对可能的下一状态和奖励求期望。DP 不要求使用神经网络。

#### 3.3.1 Policy Evaluation

固定 $\pi$，使用第 $k$ 轮的估值计算第 $k+1$ 轮：

$$
V_{k+1}(s)=\sum_a\pi(a\mid s)\sum_{s',r}p(s',r\mid s,a)
\bigl[r+\gamma V_k(s')\bigr].
$$

迭代是在逼近该策略的价值，不是在这一阶段改变策略。在上述折扣表格条件下，该算子是压缩映射，迭代收敛到唯一的 $V^\pi$。

#### 3.3.2 Policy Improvement

已知精确 $Q^\pi$ 后，让新策略在每个状态选择一个价值最大的动作：

$$
\pi'(s)\in\arg\max_a Q^\pi(s,a).
$$

由策略改进定理，$V^{\pi'}(s)\geq V^\pi(s)$。这次贪心改进不等于立刻找到了全局最优策略；仍要评估新策略再改进。如果策略已对自身的精确价值贪心，满足 Bellman 最优性条件，才可得到最优性结论。使用近似 Critic 时也不能直接保证每次改进都不退步。

#### 3.3.3 Policy Iteration

反复进行“评估当前策略 → 对当前价值作贪心改进”。固定并列最大值的选择规则，可以避免只在等价动作间来回切换。

<img src="https://s2.loli.net/2024/05/03/9s3hY5weflTpL6d.png" alt="Policy Iteration 原配图" width="400" height="300" />

<img src="https://s2.loli.net/2024/05/04/MlOrPbRqevAgFU1.png" alt="策略评估与改进的交替过程" width="300" height="200" />

原配图来源：[Sutton 与 Barto 教材](https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf)。评估和改进不必每次都完全收敛，二者交替作用的思路称为 Generalized Policy Iteration。

#### 3.3.4 Value Iteration

Value Iteration 每轮直接使用最优 Bellman 算子：

$$
V_{k+1}(s)=\max_a\sum_{s',r}p(s',r\mid s,a)
\bigl[r+\gamma V_k(s')\bigr].
$$

它把一次价值更新与贪心选择合在一起。在标准折扣表格条件下收敛到 $V^*$，再由该价值提取贪心策略。

<img src="https://s2.loli.net/2024/05/04/Px6KeON8TqJmHVn.png" alt="Value Iteration 原配图" width="400" height="300" />

原配图来源：[Sutton 与 Barto 教材](https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf)。

#### 3.3.5 DP 的边界

DP 使用模型对后继情况求期望；状态很多或模型未知时，这一步可能难以进行。后面的 MC 和 TD 则可以直接用实际采到的经历学习。

<img src="https://s2.loli.net/2024/05/04/arfKpUyeJ692CsY.png" alt="DP 更新示意原配图" width="400" height="300" />

原配图来源：[Monte Carlo Methods 课程](https://www.youtube.com/watch?v=P0ZvxeQqv0A)。

### 3.4 Monte-Carlo based

#### 3.4.1 从完整回报估计状态价值

MC 的想法是：先把这一局走完，再用实际观察到的折扣回报 $G_t$ 更新估值。无需知道环境模型，也无需用另一个价值估计补齐后半段。

对某状态的第 $n$ 次回报样本 $G^{(n)}$，样本均值可以递推为：

$$
V_n(s)=V_{n-1}(s)+\frac1n\bigl(G^{(n)}-V_{n-1}(s)\bigr).
$$

或者用固定步长 $\alpha$，得到指数加权的估计。First-visit MC 每局对一个状态只用首次访问；every-visit MC 使用每次访问。无需从每个状态专门启动一局，可以从同一完整 episode 中取出访问该状态后的回报。

<img src="https://s2.loli.net/2024/05/04/5QwhUXIr96KiNSP.png" alt="MC 状态价值估计原配图" width="400" height="300" />

<img src="https://s2.loli.net/2024/05/04/1TUgNV5yYWfdeF3.png" alt="MC 完整回报示意原配图" width="400" height="300" />

原配图来源：[Sutton 与 Barto 教材](https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf)、[Monte Carlo Methods 课程](https://www.youtube.com/watch?v=P0ZvxeQqv0A)。

#### 3.4.2 动作价值与探索

同样可以用访问 $(s_t,a_t)$ 后的 $G_t$ 更新 $Q(s_t,a_t)$，再改善策略。只选当前最高估值的动作会漏掉尚未尝试的动作。

一个简单办法是 $\epsilon$-greedy：以概率 $1-\epsilon$ 选择贪心动作，以概率 $\epsilon$ 在全部动作中均匀采样。若有 $m$ 个动作且唯一贪心动作，它的总概率是 $1-\epsilon+\epsilon/m$，其余动作各为 $\epsilon/m$。$\epsilon$ 控制探索，不是 PPO 的 clip 超参数；两处虽惯用同一字母，含义不同。

<img src="https://s2.loli.net/2024/05/04/qiZp7dsTecbukfw.png" alt="MC 动作价值估计原配图" width="400" height="300" />

<img src="https://s2.loli.net/2024/05/04/jf4X5lavU3OVeht.png" alt="带探索的 MC control 原配图" width="400" height="300" />

原配图来源：[Sutton 与 Barto 教材](https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf)。

### 3.5 Temporal-difference based

#### 3.5.1 从完整回报改成一步目标

MC 要等到结束才能拿到完整回报。TD 先走一步，收到 $r_t$，再用当前价值估计补上未来，这个“用估计更新估计”的过程叫 bootstrap。

所有这些更新都可以理解为：旧估计向目标移动一小步。MC 的目标是 $G_t$；TD(0) 的目标是 $r_t+\gamma V(s_{t+1})$。

#### 3.5.2 TD Prediction

对于固定策略，Bellman 方程描述的是条件期望：

$$
V^\pi(s)=\mathbb E_\pi[r_t+\gamma V^\pi(S_{t+1})\mid S_t=s].
$$

一次随机转移不一定满足 $V^\pi(s_t)=r_t+\gamma V^\pi(s_{t+1})$。定义 $d_t=1$ 表示这次转移真正终止，否则为零，则采样 TD error 与更新是：

$$
\delta_t=r_t+\gamma(1-d_t)V(s_{t+1})-V(s_t),
\qquad
V(s_t)\leftarrow V(s_t)+\alpha\delta_t.
$$

奖励是“本次观察到的数值”，不等于奖励机制没有随机性。与 MC 相比，TD 往往更早更新、方差较小，但引入 bootstrap 依赖；不能仅按谁更快就判定所有任务上谁更好。

对神经网络 $V_\phi$，可最小化 $\tfrac12(V_\phi(s_t)-y_t)^2$，其中 $y_t=r_t+\gamma(1-d_t)V_{\bar\phi}(s_{t+1})$ 在这次反向传播中停止梯度，$\bar\phi$ 可表示固定快照。若连目标一侧也一同求导，就不是这里的标准 semi-gradient TD 更新。

<img src="https://s2.loli.net/2024/04/30/QveSp4AUZ7ykamt.png" alt="MC 与 TD 估计差异的原配图" width="400" height="300" />

原配图来源：[DRL Lecture 3: Q-learning](https://youtu.be/o_g9JUMw1Oc?t=924)。有限数据、不同更新目标和函数近似都可能让二者给出不同估计。

#### 3.5.3 SARSA: On-Policy TD Control

SARSA 用当前行为策略实际选出的下一动作 $a_{t+1}$ 构造目标：

$$
Q(s_t,a_t)\leftarrow Q(s_t,a_t)+\alpha
\bigl[r_t+\gamma(1-d_t)Q(s_{t+1},a_{t+1})-Q(s_t,a_t)\bigr].
$$

五个量是 State、Action、Reward、下一 State、下一 Action。On-policy 指估值目标与实际采样动作使用同一个策略，例如同一个 $\epsilon$-greedy 策略。真正终止时不需要再采样 $a_{t+1}$，目标只有 $r_t$。

<img src="https://s2.loli.net/2024/05/05/aZwMWOVS2yA1hKQ.png" alt="SARSA 原配图" width="400" height="300" />

原配图来源：[Sutton 与 Barto 教材](https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf)。

#### 3.5.4 Q-Learning、DQN 与 Double DQN

Q-Learning 仍可用带探索的行为策略采样，但下一步目标使用当前 Q 的贪心价值：

$$
Q(s_t,a_t)\leftarrow Q(s_t,a_t)+\alpha
\left[r_t+\gamma(1-d_t)\max_{a'}Q(s_{t+1},a')-Q(s_t,a_t)\right].
$$

这是 off-policy：目标中的贪心策略与采样行为策略可以不同。这里的贪心策略只是相对于**当前估值**，还不能直接称为已知的最优 $\pi^*$。表格型收敛到 $Q^*$ 需要充分访问各状态动作、合适的递减步长等条件；神经网络版本没有同样的一般收敛保证。

<img src="https://s2.loli.net/2024/05/05/xVeEJQyK5LqDYtA.png" alt="Q-Learning 原配图" width="400" height="300" />

原配图来源：[DRL Lecture 3: Q-learning](https://youtu.be/o_g9JUMw1Oc?t=924)。

DQN 用 online 网络 $Q_\theta$ 学习，用更新较慢的 target 网络 $Q_{\bar\theta}$ 提供目标，并从 replay buffer 取样：

$$
y_t^{\rm DQN}=r_t+\gamma(1-d_t)\max_{a'}Q_{\bar\theta}(s_{t+1},a'),
\qquad
\mathcal L_Q(\theta)=\frac1B\sum_{i=1}^{B}
\frac12\bigl(Q_\theta(s_i,a_i)-\operatorname{stopgrad}(y_i)\bigr)^2.
$$

$B$ 是 transition batch 大小；也常用 Huber loss。交互时通常需要 $\epsilon$-greedy 等探索，不能只靠永远取 argmax 获取所有经验。每隔若干步复制 $\bar\theta\leftarrow\theta$，或采用另行指定的缓慢更新方式。

当同一组带误差的估值既用于选择价值最大的动作，又用于给该动作打分时，max 会偏向被高估的分量。[Double DQN](https://arxiv.org/abs/1509.06461) 将这两个步骤分开：

$$
a_t^{\rm sel}\in\arg\max_{a'}Q_\theta(s_{t+1},a'),
\qquad
y_t^{\rm DDQN}=r_t+\gamma(1-d_t)
Q_{\bar\theta}(s_{t+1},a_t^{\rm sel}).
$$

**Online 网络选动作，target 网络给该动作一个标量分数。** 例如下一状态的 online Q 为 $(5,4)$、target Q 为 $(3,6)$，$r=1$、$\gamma=0.9$ 且未终止：DQN 目标是 $1+0.9\times6=6.4$；Double DQN 先选第一个动作，再得到 $1+0.9\times3=3.7$。真正终止时两者都等于 $1$。

这有助于缓解过估计，不表示 target 网络总会高估，也不保证 Double DQN 的每个估计都无偏。两个 Q 网络也不等于必须存在两个独立 Actor。

#### 3.5.5 SARSA 与 Q-Learning 的取舍

在 cliff-walking 中，如果一直保留随机探索，SARSA 的目标会把未来探索时掉下悬崖的风险计入，往往学到离悬崖更远的路线。Q-Learning 的目标评估未来按贪心动作行动的价值，即使采样时还在探索，也可能偏好贴边的短路线。

<img src="https://s2.loli.net/2024/05/05/Dhf3iG1WpzymvdF.png" alt="Cliff-walking 原配图" width="300" height="200" />

原配图来源：[Sutton 与 Barto 教材](https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf)。这不能简化为“SARSA 天生次优”：要先说清比较的是持续探索时的实际回报，还是关闭探索后贪心执行的回报。

#### 3.5.6 DP、MC 与 TD

- DP：知道环境模型，对后继情况求期望，也使用后继价值进行 bootstrap。
- MC：不要求模型，使用完整采样回报，不 bootstrap。
- TD：不要求模型，使用采样转移与后继估值，进行 bootstrap。

![DP 与 TD 对照原配图一](https://s2.loli.net/2024/05/10/RHEGAv73d5Wru9t.png)

![DP 与 TD 对照原配图二](https://s2.loli.net/2024/05/10/KENkys8WxcmGUVo.png)

关键区别是“采样还是对模型求期望”“是否 bootstrap”，不是有没有神经网络。

#### 3.5.7 Forward View of TD-λ

一步 TD 很依赖下一状态的估值；MC 要等待很久。中间办法是先实际观察 $n$ 步，再用估值补上剩余部分。以下先讨论固定策略的**状态价值预测**，不把 on-policy 的结论直接套到 off-policy Q-Learning。

设同一局在 $T$ 真正终止，$h=T-t$，计算本轮目标时价值快照 $V$ 固定，且 $V(s_T)=0$。$n$ 步回报是：

$$
G_t^{(n)}=\sum_{j=0}^{n-1}\gamma^j r_{t+j}
+\gamma^n V(s_{t+n}),\qquad 1\leq n\leq h.
$$

例如：

$$
\begin{aligned}
G_t^{(1)}&=r_t+\gamma V(s_{t+1}),\\
G_t^{(2)}&=r_t+\gamma r_{t+1}+\gamma^2V(s_{t+2}),\\
G_t^{(3)}&=r_t+\gamma r_{t+1}+\gamma^2r_{t+2}+\gamma^3V(s_{t+3}).
\end{aligned}
$$

中间奖励也要逐项折扣。将各个长度的回报加权，有限 episode 的 λ-return 定义为：

$$
G_t^\lambda=(1-\lambda)\sum_{n=1}^{h-1}\lambda^{n-1}G_t^{(n)}
+\lambda^{h-1}G_t^{(h)},\qquad 0\leq\lambda\leq1.
$$

最后一项接收尚未分配完的全部权重；权重之和为 $(1-\lambda)\sum_{n=1}^{h-1}\lambda^{n-1}+\lambda^{h-1}=1$。当只剩一步时求和为空，唯一目标就是 $G_t^{(1)}$。

- $\lambda=0$：只保留一步 TD 目标。
- $\lambda=1$：只保留直到真正终止的完整回报，即有限 episodic 情况下的 MC 目标，**不是对所有 n-step 回报等权平均**。
- $0<\lambda<1$：在不同 bootstrap 长度间折中。

持续任务在级数收敛、$\lambda<1$ 等条件下可以写作 $(1-\lambda)\sum_{n=1}^{\infty}\lambda^{n-1}G_t^{(n)}$；不能把 $\lambda=1$ 生硬代入这个无限级数表达式就宣布得到 MC。

<img src="https://s2.loli.net/2024/05/05/yB2vdxL4s9eqQKD.png" alt="不同 n-step 回报的原配图" width="300" height="200" />

<img src="https://s2.loli.net/2024/05/07/XJTtUIEQxjAO2mR.png" alt="TD λ forward view 原配图" width="300" height="200" />

原配图来源：[Sutton 与 Barto 教材](https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf)。λ 调整的是不同长度回报的混合方式，不应只解释成“未来越远，影响一定越小”。

#### 3.5.8 Backward View of TD-λ

Forward view 在当前位置向未来看；backward view 在新的 TD error 到来时，把它分配给之前访问过的状态。Eligibility trace 就是每个状态还应获得多少更新的记录。

表格型 accumulating trace 初始化 $e_{-1}(s)=0$，每步先更新 trace，再更新所有状态：

$$
\begin{aligned}
e_t(s)&=\gamma\lambda e_{t-1}(s)+\mathbb 1\{s_t=s\},\\
\delta_t&=r_t+\gamma(1-d_t)V(s_{t+1})-V(s_t),\\
V(s)&\leftarrow V(s)+\alpha\delta_t e_t(s)\quad\text{对每个状态 }s.
\end{aligned}
$$

所有状态使用同一个当前 $\delta_t$，但乘各自的 trace。最近访问会增加 trace，久未访问会衰减；同一个状态重复出现可以累积贡献，所以不能简单用“出现次数少”代替“离当前时间远”。完成终止转移的更新后，在新 episode 开始前清零 trace。

Replacing trace 对当前访问状态将 trace 设为 $1$，其他状态继续衰减。Dutch trace 是另一种带步长/特征修正的 trace，不能仅称为“衰减后加一”，那正是 accumulating trace 的基本形式。

<img src="https://s2.loli.net/2024/05/09/3fkGFpXTNZbI16y.png" alt="Eligibility trace 原配图" width="300" height="200" />

<img src="https://s2.loli.net/2024/05/09/cgxYrhOWDmblPs7.png" alt="TD λ backward view 算法原配图" width="300" height="200" />

<img src="https://s2.loli.net/2024/05/09/ePOLlfxAQijUC42.png" alt="TD error 沿 trace 分配的原配图" width="300" height="200" />

原配图来源：[Sutton 与 Barto 教材](https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf)。这里写的是 on-policy 状态价值预测；off-policy trace 往往还需校正或截断，不能原样替换成任意 Q-Learning。

#### 3.5.9 Forward 与 Backward 何时等价

先固定整局内用于算目标和 TD error 的价值函数 $V$。在未终止的中间步骤写 $\delta_k=r_k+\gamma V(s_{k+1})-V(s_k)$，终点取 $V(s_T)=0$。连续相加时，中间价值项会抵消：

$$
\begin{aligned}
\sum_{j=0}^{n-1}\gamma^j\delta_{t+j}
&=\sum_{j=0}^{n-1}\gamma^j
\bigl[r_{t+j}+\gamma V(s_{t+j+1})-V(s_{t+j})\bigr]\\
&=G_t^{(n)}-V(s_t).
\end{aligned}
$$

将其代入有限 λ-return，$\delta_{t+j}$ 所对应的剩余混合权重为 $\lambda^j$，因而：

$$
G_t^\lambda-V(s_t)=\sum_{j=0}^{T-t-1}(\gamma\lambda)^j\delta_{t+j}.
$$

而 accumulating trace 展开为 $e_k(s)=\sum_{t=0}^{k}(\gamma\lambda)^{k-t}\mathbb1\{s_t=s\}$。若整局计算期间固定 $V$、固定步长 $\alpha$，最后统一应用累积更新，则交换有限求和次序得到：

$$
\begin{aligned}
\Delta V(s)
&=\alpha\sum_{k=0}^{T-1}\delta_k e_k(s)\\
&=\alpha\sum_{t=0}^{T-1}\mathbb1\{s_t=s\}
\sum_{k=t}^{T-1}(\gamma\lambda)^{k-t}\delta_k\\
&=\alpha\sum_{t=0}^{T-1}\mathbb1\{s_t=s\}
\bigl[G_t^\lambda-V(s_t)\bigr].
\end{aligned}
$$

这证明了上述 **offline、固定估值、accumulating trace 的表格型更新**与 forward view 相同。此处 offline 指延迟到 episode 结束应用更新，与“固定历史数据集上的 offline RL”不是同一个分类问题。

当 $\lambda=1$，同样的抵消给出 $\sum_{j=0}^{T-t-1}\gamma^j\delta_{t+j}=G_t-V(s_t)$，所以在这些条件下对应 every-visit MC 的累计更新。当 $\lambda=0$，两种视角都退化为一步 TD；但 online 与整局延迟更新使用的数值可能不同，不能据此说两种更新时机结果永远一样。

普通 online TD(λ) 每一步都会改变 $V$，不同时间的 TD error 不再来自同一份快照，上述证明不能直接沿用。[Harm van Seijen 与 Richard S. Sutton 的 True Online TD(λ)，ICML 2014，PMLR 32(1):692–700](https://proceedings.mlr.press/v32/seijen14.html) 给出了包含估值变化的新 online forward view，并用 Dutch traces 和修正更新精确实现它。该结果涵盖表格/线性函数近似；不是说把任意深度网络的普通 TD(λ) 换个名字就获得了等价保证。

#### 3.5.10 从 TD error 到 GAE

回到 Actor–Critic，这里要估计的是动作比平均选择好多少，不需要重新定义价值函数。

采样来自当前评估策略 $\pi$，若 $V=V^\pi$，则条件期望满足：

$$
\mathbb E[\delta_t\mid s_t,a_t]
=Q^\pi(s_t,a_t)-V^\pi(s_t)=A^\pi(s_t,a_t).
$$

所以单步 TD error 已经是一种 advantage 估计；用近似 Critic 时会有估计误差。多看几步，就得到第 3.5.9 节中的 $G_t^{(n)}-V(s_t)$。将不同长度的估计按 λ 加权混合，就是 [Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438)：

$$
\widehat A_t^{\rm GAE}=\sum_{j=0}^{T-t-1}(\gamma\lambda)^j\delta_{t+j}
=G_t^\lambda-V(s_t).
$$

这里使用同一份 Critic 快照、同一局完整数据，终点是真正终止状态。

- $\lambda=0$：$\widehat A_t=\delta_t$，只用一步信息。
- $\lambda=1$：完整有限 episode 中 $\widehat A_t=G_t-V(s_t)$，没有尾部 bootstrap。即使 baseline 不精确，只要满足动作独立及停止梯度等条件，它仍可用于无偏的策略梯度估计；但其逐状态动作的期望不一定等于真实 $A^\pi$。
- 中间值通常在方差与 Critic 误差引起的偏差间折中；不是 λ 越大或越小就一定更好。

为了计算高效，通常从后往前递推：

$$
\widehat A_t=\delta_t+\gamma\lambda\widehat A_{t+1},\qquad\widehat A_T=0.
$$

**真正终止与采样截断要分开处理。** 若环境任务仍在继续，只是 rollout 在 $H$ 处截断，应保留 $V(s_H)$ 的 bootstrap；此时即使 $\lambda=1$，结果也是 $\sum_{j=0}^{H-t-1}\gamma^j r_{t+j}+\gamma^{H-t}V(s_H)-V(s_t)$，不等于完整 MC 回报。若时间上限本身就是任务定义的有限终点，则按真正终止处理，并把剩余时间纳入状态。

实践中可分开记录两个 mask：$b_t$ 表示下一状态的价值是否应保留，$c_t$ 表示下一条已存储 advantage 是否仍属于当前连续片段：

$$
\delta_t=r_t+\gamma b_t V(s_{t+1})-V(s_t),\qquad
\widehat A_t=\delta_t+\gamma\lambda c_t\widehat A_{t+1}.
$$

真正终止时 $b_t=c_t=0$；被外部截断并 reset 时 $b_t=1,c_t=0$；片段内部通常均为 $1$。最后一条采样的递推尾值设零，但仍按 $b_t$ 决定是否保留 value bootstrap。使用环境自动 reset 时，应取 reset **之前**的最终 observation 计算这个 bootstrap，不能误用新一局初始状态。

例如奖励 $(1,2,3)$、价值快照 $(0.5,1,1.5,0)$，第三步真正终止，$\gamma=0.9$、$\lambda=0.8$。TD errors 是 $(1.4,2.35,1.5)$，从后往前得到 advantages $(3.8696,3.43,1.5)$。取 $\lambda=1$ 时变为 $(4.73,3.7,1.5)$，等于折扣 MC 回报减 baseline。

## 4. 一个可以直接验算的例子

下面的 Python 只需要标准库，展示单条连续 rollout 的 GAE 和 Double DQN 标量目标。它不是完整训练器；输入使用已经固定、停止梯度后的数值，没有执行环境训练或 GPU 推理。

```python
import math


def gae(rewards, values, terminated, gamma=0.9, lam=0.8):
    """One continuous rollout; values includes the final observation's value.

    An external cutoff is not termination. If the last transition merely
    truncates collection, pass terminated[-1] = False and bootstrap values[-1].
    Internal resets are rejected: split them into separate rollouts first.
    """
    n = len(rewards)
    if n == 0 or len(values) != n + 1 or len(terminated) != n:
        raise ValueError("expected n rewards, n+1 values, n termination flags")
    if not (0 <= gamma <= 1 and 0 <= lam <= 1):
        raise ValueError("gamma and lambda must lie in [0, 1]")
    if not all(type(flag) is bool for flag in terminated):
        raise ValueError("termination flags must be boolean")
    if any(terminated[:-1]):
        raise ValueError("split rollouts at terminal transitions")
    if not all(math.isfinite(x) for x in [*rewards, *values]):
        raise ValueError("rewards and values must be finite")
    advantages = [0.0] * n
    tail = 0.0
    for t in reversed(range(n)):
        bootstrap = 0.0 if terminated[t] else values[t + 1]
        delta = rewards[t] + gamma * bootstrap - values[t]
        continuation = 0.0 if terminated[t] else tail
        tail = delta + gamma * lam * continuation
        advantages[t] = tail
    return advantages


def double_dqn_target(reward, online_next, target_next, terminated, gamma=0.9):
    if not online_next or len(online_next) != len(target_next):
        raise ValueError("Q vectors must be nonempty and have equal lengths")
    if not 0 <= gamma <= 1 or type(terminated) is not bool:
        raise ValueError("invalid gamma or termination flag")
    if not all(math.isfinite(x) for x in [reward, *online_next, *target_next]):
        raise ValueError("reward and Q values must be finite")
    if terminated:
        return float(reward)
    action = max(range(len(online_next)), key=online_next.__getitem__)
    return reward + gamma * target_next[action]


advantages = gae([1, 2, 3], [0.5, 1, 1.5, 0], [False, False, True])
print([round(x, 4) for x in advantages])  # [3.8696, 3.43, 1.5]
print(double_dqn_target(1, [5, 4], [3, 6], False))  # 3.7
```

GAE 负责构造优势估计，PPO-Clip 负责使用优势改进策略，DQN/Double DQN 则走动作价值学习路线。把“学习什么”“目标从哪来”“数据由谁采样”分开，公式就更容易核对。

## Reference

1. [David Silver — UCL Course on Reinforcement Learning](https://www.davidsilver.uk/teaching/)
2. [Sutton 与 Barto — Reinforcement Learning: An Introduction，原文引用的教材 PDF](https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf)
3. [Schulman et al. — Proximal Policy Optimization Algorithms，2017](https://arxiv.org/abs/1707.06347)
4. [Schulman et al. — High-Dimensional Continuous Control Using Generalized Advantage Estimation，ICLR 2016](https://arxiv.org/abs/1506.02438)
5. [van Hasselt, Guez 与 Silver — Deep Reinforcement Learning with Double Q-learning，AAAI 2016](https://arxiv.org/abs/1509.06461)
6. [van Seijen 与 Sutton — True Online TD(λ)，ICML 2014，PMLR 32(1):692–700](https://proceedings.mlr.press/v32/seijen14.html)
