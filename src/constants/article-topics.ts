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
			"从文字、概率和 Transformer 走到推理与微调。先建立完整流程，深入推导按需选读。",
		articles: [
			{
				slug: "2026-05-16-llm-tokenization-guide",
				title: "Tokenization：从文字到 token",
				purpose: "分词算法、词表训练与编码；不把检索 embedding 混在分词里。",
			},
			{
				slug: "2024-04-10-perplexity",
				title: "Perplexity：语言模型的评分",
				purpose: "平均负对数似然、PPL 与可比条件；分布差异的推导去 KL 专题。",
			},
			{
				slug: "2024-04-13-text-generator-with-transformer-decoder",
				title: "Transformer Decoder：从零生成文本",
				purpose:
					"最小模型、Q/K/V、mask、label shift 与训练循环；生成策略只保留最小示例。",
			},
			{
				slug: "2026-05-16-llm-rope-rotary-position-embedding",
				title: "RoPE：旋转位置编码",
				purpose: "相对位置内积与实现；长度扩展的方法比较放在长上下文专题。",
			},
			{
				slug: "2026-05-16-llm-decoding-strategies",
				title: "Decoding：怎样选下一个 token",
				purpose:
					"Temperature、Top-K、Top-P 与 Beam Search；不重复模型结构和缓存推导。",
			},
			{
				slug: "2026-06-12-llm-mem-opt-1-fundamentals",
				title: "显存基础：单位、精度与生命周期",
				purpose: "统一数值格式和训练/推理账本；具体模型容量去推理显存。",
			},
			{
				slug: "2026-06-13-llm-mem-opt-2-inference",
				title: "推理显存：权重、KV Cache 与工作区",
				purpose:
					"权重与 KV 容量、缓存与 FlashAttention 前向算法，以及速算器和 OOM 排查。",
			},
			{
				slug: "2026-05-16-llm-quantization-guide",
				title: "大模型量化：从整数刻度到 GPTQ/AWQ",
				purpose: "整数编码、误差和粒度，以及按需阅读的 GPTQ/AWQ 进阶推导。",
			},
			{
				slug: "2026-06-16-llm-mem-opt-5-sft",
				title: "SFT：数据与显存优化实战",
				purpose:
					"训练数据、loss mask、LoRA、packing 和算子开关；偏好学习去对齐专题。",
			},
		],
	},
	{
		id: "llm-deep",
		title: "LLM 深入与速查",
		description:
			"遇到具体瓶颈再进入，不是开始学习或运行单卡 SFT 前的统一门槛。",
		articles: [
			{
				slug: "2026-05-16-llm-long-context-yarn",
				title: "长上下文：PI、NTK-aware 与 YaRN",
				purpose: "在 RoPE 基础上比较位置/频率变换、缩放约定和质量边界。",
			},
			{
				slug: "2024-04-20-weight-decay-and-l2-regularization",
				title: "AdamW：优化器中的权重衰减",
				purpose: "SGD、momentum、Adam 与解耦衰减；L1/L2 的统计解释另读基础。",
			},
			{
				slug: "2026-06-15-llm-mem-opt-4-zero",
				title: "训练显存与 ZeRO：分片和通信",
				purpose: "按既定训练配方推导 ZeRO-1/2/3；数值格式定义回显存基础。",
			},
			{
				slug: "2025-12-05-a-series-on-llm-training-i",
				title: "LLM 对齐：RLHF、PPO 与 DPO",
				purpose:
					"奖励模型、参考策略与 token 级偏好目标；通用 RL 推导单独阅读。",
			},
		],
	},
	{
		id: "applications",
		title: "LLM 应用与部署",
		description:
			"一篇负责应用概念，一篇负责运行环境，避免把框架 API 和模型训练混成同一系列。",
		articles: [
			{
				slug: "2024-04-24-lang-chain-series",
				title: "LLM 应用：Prompt、RAG、Tools 与 Agent",
				purpose: "LangChain 作为例子，梳理应用组件、检索边界与工具执行。",
			},
			{
				slug: "2026-05-15-hermes-agent-wsl-setup",
				title: "Hermes Agent：WSL 部署与排错",
				purpose: "有版本边界的部署案例；系统服务、网关与跨层排错。",
			},
		],
	},
	{
		id: "foundations",
		title: "数学与模型基础",
		description: "按问题补知识；分类指标、概率建模和强化学习各有自己的目标。",
		articles: [
			{
				slug: "2024-04-19-kullback-leibler-divergence",
				title: "KL 散度：比较两个分布",
				purpose: "定义、非负性、熵的关系和高斯 KL；PPL 应用单独讲。",
			},
			{
				slug: "2024-04-11-sampling-method",
				title: "采样方法：逆变换与拒绝采样",
				purpose: "从给定分布获得样本；LLM 如何修改候选分布去 Decoding。",
			},
			{
				slug: "2024-04-10-gama-beta-dirichlet",
				title: "Gamma、Beta 与 Dirichlet 分布",
				purpose: "连续分布、共轭关系和概率建模基础。",
			},
			{
				slug: "2024-04-20-l1-and-l2-regularization",
				title: "L1/L2 正则化：统计与几何直觉",
				purpose: "正则目标、约束、先验和稀疏性；优化器差异去 AdamW。",
			},
			{
				slug: "2024-04-30-deep-reinforcement-learning",
				title: "深度强化学习：策略与价值方法",
				purpose: "通用 MDP、Policy Gradient、GAE、PPO 和 Q-learning。",
			},
			{
				slug: "2024-04-10-probability-calibration",
				title: "概率校准：分数能否当作概率",
				purpose: "校准曲线与分类概率；不与 AUC 的排序评价混为一谈。",
			},
			{
				slug: "2024-04-12-auc-gauc",
				title: "AUC 与 GAUC：排序评价",
				purpose: "ROC、正负样本排序和分组汇总。",
			},
			{
				slug: "2024-04-11-gan-loss-derivation",
				title: "GAN：对抗目标的推导",
				purpose: "生成器与判别器的 min-max 目标。",
			},
		],
	},
	{
		id: "causal",
		title: "因果推断",
		description:
			"先理解为什么相关性不等于因果，再检查从观测数据识别效应的条件。",
		articles: [
			{
				slug: "2024-04-26-introduction-to-causal-inference-i",
				title: "因果推断入门：从悖论到潜在结果",
				purpose: "动机、反事实与平均处理效应的直觉。",
			},
			{
				slug: "2025-02-05-introduction-to-causal-inference-ii",
				title: "因果效应识别：估计目标与假设",
				purpose: "ATT/CATE、SUTVA、可交换性、正值性和调整公式。",
			},
		],
	},
	{
		id: "guides",
		title: "导航与站务",
		description: "导航与工具推荐保留独立位置，不混入技术课程。",
		articles: [
			{
				slug: "2026-06-03-llm-knowledge-map",
				title: "LLM 知识地图",
				purpose: "入门路线、专题分工与尚未展开的知识位置。",
			},
			{
				slug: "2026-05-06-recommendations",
				title: "推荐与推广",
				purpose: "工具推荐与推广信息。",
			},
		],
	},
];

export function getArticleTopic(slug: string): ArticleTopic | undefined {
	return articleTopics.find((topic) =>
		topic.articles.some((article) => article.slug === slug),
	);
}
