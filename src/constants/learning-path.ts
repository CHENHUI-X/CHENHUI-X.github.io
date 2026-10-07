/** Learning order is independent of publication dates and legacy series numbers. */
export type LearningPathKind = "mainline" | "branch" | "foundation";

export type LearningPathEntry = {
	slug: string;
	title: string;
	kind: LearningPathKind;
	prerequisites: readonly string[];
	/** Mainline article to return to after an optional entry. */
	returnSlug?: string;
};

export const KNOWLEDGE_MAP_SLUG = "2026-06-03-llm-knowledge-map";

export const learningPathEntries: readonly LearningPathEntry[] = [
	{
		slug: "2026-05-16-llm-tokenization-guide",
		title: "Tokenization · 文字与 token",
		kind: "mainline",
		prerequisites: [],
	},
	{
		slug: "2024-04-10-perplexity",
		title: "Perplexity (PPL) · 语言模型评分",
		kind: "mainline",
		prerequisites: ["2026-05-16-llm-tokenization-guide"],
	},
	{
		slug: "2024-04-19-kullback-leibler-divergence",
		title: "KL Divergence · 比较概率分布",
		kind: "foundation",
		prerequisites: ["2024-04-10-perplexity"],
		returnSlug: "2024-04-10-perplexity",
	},
	{
		slug: "2024-04-13-text-generator-with-transformer-decoder",
		title: "Transformer Decoder · 从零生成文本",
		kind: "mainline",
		prerequisites: [
			"2026-05-16-llm-tokenization-guide",
			"2024-04-10-perplexity",
		],
	},
	{
		slug: "2026-05-16-llm-rope-rotary-position-embedding",
		title: "RoPE · 旋转位置编码",
		kind: "mainline",
		prerequisites: ["2024-04-13-text-generator-with-transformer-decoder"],
	},
	{
		slug: "2026-05-16-llm-decoding-strategies",
		title: "Decoding · 从分布到生成策略",
		kind: "mainline",
		prerequisites: ["2024-04-13-text-generator-with-transformer-decoder"],
	},
	{
		slug: "2026-06-12-llm-mem-opt-1-fundamentals",
		title: "显存基础 · 单位、精度与生命周期",
		kind: "mainline",
		prerequisites: ["2024-04-13-text-generator-with-transformer-decoder"],
	},
	{
		slug: "2026-06-13-llm-mem-opt-2-inference",
		title: "推理 · 容量、缓存计算与 OOM",
		kind: "mainline",
		prerequisites: [
			"2026-06-12-llm-mem-opt-1-fundamentals",
			"2026-05-16-llm-rope-rotary-position-embedding",
		],
	},
	{
		slug: "2026-05-16-llm-quantization-guide",
		title: "量化 · 编码与误差（进阶推导选读）",
		kind: "mainline",
		prerequisites: ["2026-06-13-llm-mem-opt-2-inference"],
	},
	{
		slug: "2024-04-20-weight-decay-and-l2-regularization",
		title: "AdamW · Weight Decay 与 L2",
		kind: "branch",
		prerequisites: ["2024-04-20-l1-and-l2-regularization"],
		returnSlug: "2026-06-16-llm-mem-opt-5-sft",
	},
	{
		slug: "2026-06-15-llm-mem-opt-4-zero",
		title: "训练显存与 ZeRO · 模型状态分片",
		kind: "branch",
		prerequisites: [
			"2026-06-12-llm-mem-opt-1-fundamentals",
			"2024-04-20-weight-decay-and-l2-regularization",
		],
		returnSlug: "2026-06-16-llm-mem-opt-5-sft",
	},
	{
		slug: "2026-06-16-llm-mem-opt-5-sft",
		title: "SFT · 数据边界与显存优化",
		kind: "mainline",
		prerequisites: [
			"2024-04-13-text-generator-with-transformer-decoder",
			"2026-05-16-llm-quantization-guide",
			"2026-06-12-llm-mem-opt-1-fundamentals",
		],
	},
	{
		slug: "2026-05-16-llm-long-context-yarn",
		title: "长上下文 · PI / NTK-aware / YaRN",
		kind: "branch",
		prerequisites: ["2026-05-16-llm-rope-rotary-position-embedding"],
		returnSlug: "2026-05-16-llm-rope-rotary-position-embedding",
	},
	{
		slug: "2025-12-05-a-series-on-llm-training-i",
		title: "LLM 对齐 · RLHF / PPO / DPO",
		kind: "branch",
		prerequisites: [
			"2026-06-16-llm-mem-opt-5-sft",
			"2024-04-19-kullback-leibler-divergence",
			"2024-04-30-deep-reinforcement-learning",
		],
		returnSlug: "2026-06-16-llm-mem-opt-5-sft",
	},
	{
		slug: "2024-04-30-deep-reinforcement-learning",
		title: "通用 RL · Policy / Value / PPO",
		kind: "foundation",
		prerequisites: ["2024-04-19-kullback-leibler-divergence"],
		returnSlug: "2026-06-16-llm-mem-opt-5-sft",
	},
	{
		slug: "2024-04-11-sampling-method",
		title: "采样基础 · Inverse Transform / Rejection",
		kind: "foundation",
		prerequisites: [],
		returnSlug: "2026-05-16-llm-decoding-strategies",
	},
	{
		slug: "2024-04-20-l1-and-l2-regularization",
		title: "L1 / L2 · 正则化基础",
		kind: "foundation",
		prerequisites: [],
		returnSlug: "2024-04-20-weight-decay-and-l2-regularization",
	},
];

export const learningMainline: readonly LearningPathEntry[] =
	learningPathEntries.filter((entry) => entry.kind === "mainline");

export type LearningPathNavigation = {
	entry: LearningPathEntry | undefined;
	isKnowledgeMap: boolean;
	/** One-based on the mainline; zero for the map and optional entries. */
	position: number;
	total: number;
	previous: LearningPathEntry | undefined;
	next: LearningPathEntry | undefined;
	returnTo: LearningPathEntry | undefined;
	prerequisites: readonly LearningPathEntry[];
	optional: readonly LearningPathEntry[];
};

export function getLearningPathEntry(
	slug: string,
): LearningPathEntry | undefined {
	return learningPathEntries.find((entry) => entry.slug === slug);
}

export function getLearningPathNavigation(
	slug: string,
): LearningPathNavigation | undefined {
	const isKnowledgeMap = slug === KNOWLEDGE_MAP_SLUG;
	const entry = getLearningPathEntry(slug);
	if (!entry && !isKnowledgeMap) return undefined;

	const index = learningMainline.findIndex((item) => item.slug === slug);
	const prerequisites = (entry?.prerequisites ?? [])
		.map(getLearningPathEntry)
		.filter((item): item is LearningPathEntry => item !== undefined);

	return {
		entry,
		isKnowledgeMap,
		position: index + 1,
		total: learningMainline.length,
		previous: index > 0 ? learningMainline[index - 1] : undefined,
		next: isKnowledgeMap
			? learningMainline[0]
			: index >= 0
				? learningMainline[index + 1]
				: undefined,
		returnTo: entry?.returnSlug
			? getLearningPathEntry(entry.returnSlug)
			: undefined,
		prerequisites,
		optional: learningPathEntries.filter(
			(item) => item.returnSlug === slug && item.kind !== "mainline",
		),
	};
}
