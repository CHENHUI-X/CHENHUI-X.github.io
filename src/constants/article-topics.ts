/** Each published post has exactly one topic home; chronology remains in Archive. */
export type TopicArticle = { slug: string; title: string; purpose: string };
export type ArticleTopic = {
	id: string;
	title: string;
	description: string;
	articles: readonly TopicArticle[];
};
export const articleTopics: readonly ArticleTopic[] = [
	{
		id: "llm-start",
		title: "LLM 入门主线",
		description:
			"文本分词、模型结构与解码入门；完整 9 步路线还串联下方显存子系列与量化专题。",
		articles: [
			{
				slug: "2026-05-16-llm-tokenization-guide",
				title: "Tokenization：文字怎样变成 token，分词器怎样训练",
				purpose: "介绍分词算法、词表训练，以及文本与 Token ID 之间的转换。",
			},
			{
				slug: "2024-04-10-perplexity",
				title: "困惑度 Perplexity：怎么算，怎样比较语言模型",
				purpose: "解释困惑度的计算方法，以及比较不同模型时需要统一的评估条件。",
			},
			{
				slug: "2024-04-13-text-generator-with-transformer-decoder",
				title: "用 PyTorch 写一个 Transformer 文本生成器",
				purpose:
					"从注意力计算开始，实现一个能够训练和生成文本的 Decoder 示例。",
			},
			{
				slug: "2026-05-16-llm-rope-rotary-position-embedding",
				title: "RoPE 旋转位置编码：为什么旋转 Q 和 K 就能表示相对位置",
				purpose: "推导旋转位置编码的相对位置性质，并给出实现与计算示例。",
			},
			{
				slug: "2026-05-16-llm-decoding-strategies",
				title: "大模型怎样选下一个 token：Greedy、Top-K、Top-P 与 Beam Search",
				purpose: "介绍温度采样、Top-K、Top-P 和束搜索如何选择下一个 Token。",
			},
			{
				slug: "2026-05-16-llm-quantization-guide",
				title: "大模型量化：从 INT4 编码到 GPTQ、AWQ",
				purpose: "解释量化编码、量化误差和分组方式，进一步推导 GPTQ 与 AWQ。",
			},
		],
	},
	{
		id: "llm-memory",
		title: "大模型 · 显存子系列",
		description:
			"先认识模型保存的数据，再讲训练状态如何分摊、微调怎样省显存，最后进入推理缓存与 FlashAttention。篇次是专题阅读顺序；FlashAttention 和 ZeRO 可按需求选读，单卡 SFT 不要求先学 ZeRO。",
		articles: [
			{
				slug: "2026-06-12-llm-mem-opt-1-fundamentals",
				title: "大模型显存（一）：模型运行时要存什么",
				purpose:
					"先认清权重、梯度、优化器状态、激活和缓存，说明它们为什么占用显存、怎样计算大小。",
			},
			{
				slug: "2026-06-15-llm-mem-opt-4-zero",
				title: "大模型显存（二）：ZeRO 如何分摊训练显存",
				purpose: "多卡选读：沿训练状态的分片方式理解 ZeRO-1/2/3 与通信代价。",
			},
			{
				slug: "2026-06-16-llm-mem-opt-5-sft",
				title: "大模型显存（三）：微调时怎样节省显存",
				purpose:
					"回到微调任务，结合 LoRA、QLoRA、激活检查点和梯度累积选择方案。",
			},
			{
				slug: "2026-06-13-llm-mem-opt-2-inference",
				title: "大模型显存（四）：从 Prefill、Decode 到 KV Cache",
				purpose:
					"沿一次生成过程区分 Prefill 与 Decode，计算权重和 KV 占用，排查推理中的显存不足。",
			},
			{
				slug: "2026-06-14-llm-mem-opt-3-flashattention",
				title: "大模型显存（五）：FlashAttention 为什么能省显存",
				purpose:
					"深入选读：从注意力中间矩阵到分块与在线 softmax，理解临时显存为何减少。",
			},
		],
	},
	{
		id: "llm-deep",
		title: "LLM 进阶专题",
		description:
			"进一步学习长上下文、优化器、多卡训练和偏好对齐，可按需求选择专题。",
		articles: [
			{
				slug: "2026-05-16-llm-long-context-yarn",
				title: "怎样扩展大模型上下文：PI、NTK-aware 和 YaRN",
				purpose: "比较 PI、NTK-aware 和 YaRN 的位置编码调整方法与适用条件。",
			},
			{
				slug: "2024-04-20-weight-decay-and-l2-regularization",
				title: "AdamW：权重衰减与 L2 正则化的区别",
				purpose:
					"比较不同优化器中的 L2 正则化与权重衰减，解释 AdamW 的更新规则。",
			},
			{
				slug: "2025-12-05-a-series-on-llm-training-i",
				title: "LLM 对齐入门：RLHF、PPO 与 DPO",
				purpose: "介绍奖励模型、PPO 和 DPO，解释语言模型如何从偏好数据中学习。",
			},
		],
	},
	{
		id: "applications",
		title: "LLM 应用与部署",
		description:
			"了解提示词、检索与工具调用的配合方式，并学习 Agent 部署和故障排查。",
		articles: [
			{
				slug: "2024-04-24-lang-chain-series",
				title: "LangChain 应用开发：Prompt、检索、工具与 Agent",
				purpose:
					"以文档问答为例，介绍提示模板、检索、工具调用和 Agent 执行流程。",
			},
			{
				slug: "2026-05-15-hermes-agent-wsl-setup",
				title: "在 WSL 上部署 Hermes Agent：接入微信、配置后台服务和浏览器",
				purpose: "介绍 WSL 环境下的安装、消息网关、后台服务和故障定位。",
			},
		],
	},
	{
		id: "foundations",
		title: "数学与模型基础",
		description:
			"介绍概率分布、模型评估、正则化和强化学习中的基础概念与计算方法。",
		articles: [
			{
				slug: "2024-04-19-kullback-leibler-divergence",
				title: "KL 散度是什么，怎么计算",
				purpose:
					"解释 KL 散度与熵、交叉熵的关系，并推导高斯分布之间的 KL 散度。",
			},
			{
				slug: "2024-04-11-sampling-method",
				title: "怎么从概率分布中采样：逆变换、拒绝采样和 Thompson Sampling",
				purpose: "介绍逆变换采样、拒绝采样和它们的计算原理。",
			},
			{
				slug: "2024-04-10-gama-beta-dirichlet",
				title: "Gamma、Beta 和 Dirichlet 分布怎么理解",
				purpose: "介绍三种分布的定义、相互关系，以及如何用它们表示概率。",
			},
			{
				slug: "2024-04-20-l1-and-l2-regularization",
				title: "L1 和 L2 正则化有什么区别",
				purpose: "从优化目标、几何约束和概率先验解释 L1、L2 正则化。",
			},
			{
				slug: "2024-04-30-deep-reinforcement-learning",
				title: "深度强化学习：策略梯度、PPO 与价值学习",
				purpose:
					"介绍智能体如何根据奖励学习动作，推导策略梯度、PPO 和 Q-learning。",
			},
			{
				slug: "2024-04-10-probability-calibration",
				title: "分类模型的概率准不准，怎么校准",
				purpose: "解释概率校准的含义、常用方法，以及它与排序指标的区别。",
			},
			{
				slug: "2024-04-12-auc-gauc",
				title: "AUC 和 GAUC 怎么理解、怎么算",
				purpose: "解释 ROC 曲线、AUC 的计算方法，以及 GAUC 如何按组汇总结果。",
			},
			{
				slug: "2024-04-11-gan-loss-derivation",
				title: "GAN 的损失函数是怎么推出来的",
				purpose: "推导生成器与判别器的损失函数，解释它们各自优化什么。",
			},
		],
	},
	{
		id: "causal",
		title: "因果推断",
		description:
			"介绍潜在结果与处理效应，以及从观测数据推断因果关系所需的假设。",
		articles: [
			{
				slug: "2024-04-26-introduction-to-causal-inference-i",
				title: "因果推断（一）：相关不等于因果，处理效应怎么算",
				purpose: "从辛普森悖论出发，解释潜在结果和平均处理效应。",
			},
			{
				slug: "2025-02-05-introduction-to-causal-inference-ii",
				title: "因果推断（二）：ITE、ATE、ATT 和 CATE 有什么区别",
				purpose:
					"区分不同人群的处理效应，介绍从观测数据识别这些效应所需的假设。",
			},
		],
	},
	{
		id: "guides",
		title: "学习路线与服务推荐",
		description: "查找学习顺序，或了解文中推荐的服务。",
		articles: [
			{
				slug: "2026-06-03-llm-knowledge-map",
				title: "大模型学习路线：从文本生成到显存优化和微调",
				purpose: "提供大模型学习路线，并列出各个专题及其阅读要求。",
			},
			{
				slug: "2026-05-06-recommendations",
				title: "机场与大模型 API 中转服务推荐",
				purpose: "介绍 XMRth 机场和 PackyAPI 大模型 API 中转服务。",
			},
		],
	},
];

export function getArticleTopic(slug: string): ArticleTopic | undefined {
	return articleTopics.find((topic) =>
		topic.articles.some((article) => article.slug === slug),
	);
}
