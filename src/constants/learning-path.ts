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
		title: "Tokenization：文字怎样变成 token，分词器怎样训练",
		kind: "mainline",
		prerequisites: [],
	},
	{
		slug: "2024-04-10-perplexity",
		title: "困惑度 Perplexity：怎么算，怎样比较语言模型",
		kind: "mainline",
		prerequisites: ["2026-05-16-llm-tokenization-guide"],
	},
	{
		slug: "2024-04-19-kullback-leibler-divergence",
		title: "KL 散度是什么，怎么计算",
		kind: "foundation",
		prerequisites: ["2024-04-10-perplexity"],
		returnSlug: "2024-04-10-perplexity",
	},
	{
		slug: "2024-04-13-text-generator-with-transformer-decoder",
		title: "用 PyTorch 写一个 Transformer 文本生成器",
		kind: "mainline",
		prerequisites: [
			"2026-05-16-llm-tokenization-guide",
			"2024-04-10-perplexity",
		],
	},
	{
		slug: "2026-05-16-llm-rope-rotary-position-embedding",
		title: "RoPE 旋转位置编码：为什么旋转 Q 和 K 就能表示相对位置",
		kind: "mainline",
		prerequisites: ["2024-04-13-text-generator-with-transformer-decoder"],
	},
	{
		slug: "2026-05-16-llm-decoding-strategies",
		title: "大模型怎样选下一个 token：Greedy、Top-K、Top-P 与 Beam Search",
		kind: "mainline",
		prerequisites: ["2024-04-13-text-generator-with-transformer-decoder"],
	},
	{
		slug: "2026-06-12-llm-mem-opt-1-fundamentals",
		title: "大模型显存（一）：模型运行时要存什么",
		kind: "mainline",
		prerequisites: ["2024-04-13-text-generator-with-transformer-decoder"],
	},
	{
		slug: "2026-05-16-llm-quantization-guide",
		title: "大模型量化：从 INT4 编码到 GPTQ、AWQ",
		kind: "mainline",
		prerequisites: ["2026-06-12-llm-mem-opt-1-fundamentals"],
	},
	{
		slug: "2024-04-20-weight-decay-and-l2-regularization",
		title: "AdamW：权重衰减与 L2 正则化的区别",
		kind: "branch",
		prerequisites: ["2024-04-20-l1-and-l2-regularization"],
		returnSlug: "2026-06-16-llm-mem-opt-5-sft",
	},
	{
		slug: "2026-06-15-llm-mem-opt-4-zero",
		title: "大模型显存（二）：ZeRO 如何分摊训练显存",
		kind: "branch",
		prerequisites: [
			"2026-06-12-llm-mem-opt-1-fundamentals",
			"2024-04-20-weight-decay-and-l2-regularization",
		],
		returnSlug: "2026-06-16-llm-mem-opt-5-sft",
	},
	{
		slug: "2026-06-16-llm-mem-opt-5-sft",
		title: "大模型显存（三）：微调时怎样节省显存",
		kind: "mainline",
		prerequisites: [
			"2024-04-13-text-generator-with-transformer-decoder",
			"2026-05-16-llm-quantization-guide",
			"2026-06-12-llm-mem-opt-1-fundamentals",
		],
	},
	{
		slug: "2026-06-13-llm-mem-opt-2-inference",
		title: "大模型显存（四）：从 Prefill、Decode 到 KV Cache",
		kind: "mainline",
		prerequisites: [
			"2026-06-16-llm-mem-opt-5-sft",
			"2026-06-12-llm-mem-opt-1-fundamentals",
			"2026-05-16-llm-rope-rotary-position-embedding",
		],
	},
	{
		slug: "2026-06-14-llm-mem-opt-3-flashattention",
		title: "大模型显存（五）：FlashAttention 为什么能省显存",
		kind: "branch",
		prerequisites: ["2026-06-13-llm-mem-opt-2-inference"],
		returnSlug: "2026-06-13-llm-mem-opt-2-inference",
	},
	{
		slug: "2026-05-16-llm-long-context-yarn",
		title: "怎样扩展大模型上下文：PI、NTK-aware 和 YaRN",
		kind: "branch",
		prerequisites: ["2026-05-16-llm-rope-rotary-position-embedding"],
		returnSlug: "2026-05-16-llm-rope-rotary-position-embedding",
	},
	{
		slug: "2025-12-05-a-series-on-llm-training-i",
		title: "LLM 对齐入门：RLHF、PPO 与 DPO",
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
		title: "深度强化学习：策略梯度、PPO 与价值学习",
		kind: "foundation",
		prerequisites: ["2024-04-19-kullback-leibler-divergence"],
		returnSlug: "2026-06-16-llm-mem-opt-5-sft",
	},
	{
		slug: "2024-04-11-sampling-method",
		title: "怎么从概率分布中采样：逆变换、拒绝采样和 Thompson Sampling",
		kind: "foundation",
		prerequisites: [],
		returnSlug: "2026-05-16-llm-decoding-strategies",
	},
	{
		slug: "2024-04-20-l1-and-l2-regularization",
		title: "L1 和 L2 正则化有什么区别",
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
