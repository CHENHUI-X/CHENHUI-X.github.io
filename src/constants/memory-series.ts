/** Editorial reading order, not a prerequisite chain or the beginner mainline. */
export const MEMORY_SERIES_ID = "llm-memory";
export const memorySeriesEntries = [
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
		purpose: "回到微调任务，结合 LoRA、QLoRA、激活检查点和梯度累积选择方案。",
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
] as const;

export type MemorySeriesEntry = (typeof memorySeriesEntries)[number];
export type MemorySeriesNavigation = {
	entry: MemorySeriesEntry;
	position: number;
	total: number;
	previous: MemorySeriesEntry | undefined;
	next: MemorySeriesEntry | undefined;
};

export function getMemorySeriesNavigation(
	slug: string,
): MemorySeriesNavigation | undefined {
	const index = memorySeriesEntries.findIndex((entry) => entry.slug === slug);
	if (index < 0) return undefined;
	return {
		entry: memorySeriesEntries[index],
		position: index + 1,
		total: memorySeriesEntries.length,
		previous: memorySeriesEntries[index - 1],
		next: memorySeriesEntries[index + 1],
	};
}
