<script lang="ts">
import { onMount } from "svelte";

import I18nKey from "../i18n/i18nKey";
import { i18n } from "../i18n/translation";
import { getPostUrlBySlug } from "../utils/url-utils";

export let tags: string[] = [];
export let categories: string[] = [];
export let sortedPosts: Post[] = [];

const params = new URLSearchParams(window.location.search);
tags = params.has("tag") ? params.getAll("tag") : [];
categories = params.has("category") ? params.getAll("category") : [];
const uncategorized = params.get("uncategorized");

interface Post {
	slug: string;
	data: {
		title: string;
		tags: string[];
		category?: string | null;
		published: Date;
	};
}

interface Group {
	year: number;
	posts: Post[];
}

let groups: Group[] = [];

function formatDate(date: Date) {
	const month = (date.getMonth() + 1).toString().padStart(2, "0");
	const day = date.getDate().toString().padStart(2, "0");
	return `${month}-${day}`;
}

onMount(async () => {
	let filteredPosts: Post[] = sortedPosts;

	if (tags.length > 0) {
		filteredPosts = filteredPosts.filter(
			(post) =>
				Array.isArray(post.data.tags) &&
				post.data.tags.some((tag) => tags.includes(tag)),
		);
	}

	if (categories.length > 0) {
		filteredPosts = filteredPosts.filter(
			(post) => post.data.category && categories.includes(post.data.category),
		);
	}

	if (uncategorized) {
		filteredPosts = filteredPosts.filter((post) => !post.data.category);
	}

	const grouped = filteredPosts.reduce(
		(acc, post) => {
			const year = post.data.published.getFullYear();
			if (!acc[year]) {
				acc[year] = [];
			}
			acc[year].push(post);
			return acc;
		},
		{} as Record<number, Post[]>,
	);

	const groupedPostsArray = Object.keys(grouped).map((yearStr) => ({
		year: Number.parseInt(yearStr, 10),
		posts: grouped[Number.parseInt(yearStr, 10)],
	}));

	groupedPostsArray.sort((a, b) => b.year - a.year);

	groups = groupedPostsArray;
});
</script>

<div class="card-base px-5 md:px-8 py-6">
    {#each groups as group}
        <div class="mb-4 last:mb-0">
            <div class="flex flex-row w-full items-center h-14">
                <div class="w-[16%] md:w-[11%] text-xl md:text-2xl font-extrabold text-right text-90 tabular-nums tracking-tight">
                    {group.year}
                </div>
                <div class="w-[14%] md:w-[9%]">
                    <div
                        class="h-3 w-3 bg-none rounded-full outline outline-[var(--primary)] mx-auto -outline-offset-[2px] z-10 outline-[3px]"
                    ></div>
                </div>
                <div class="w-[70%] md:w-[80%] text-left">
                    <span class="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-semibold bg-[var(--btn-regular-bg)] text-[var(--primary)] tabular-nums">
                        {group.posts.length} {i18n(group.posts.length === 1 ? I18nKey.postCount : I18nKey.postsCount)}
                    </span>
                </div>
            </div>

            {#each group.posts as post}
                <a
                    href={getPostUrlBySlug(post.slug)}
                    aria-label={post.data.title}
                    class="group btn-plain !block h-11 w-full rounded-xl hover:text-[initial]"
                >
                    <div class="flex flex-row justify-start items-center h-full">
                        <!-- date -->
                        <div class="w-[16%] md:w-[11%] text-xs md:text-sm text-right text-50 font-medium tabular-nums group-hover:text-[var(--primary)] transition-colors duration-200">
                            {formatDate(post.data.published)}
                        </div>

                        <!-- dot and line -->
                        <div class="w-[14%] md:w-[9%] relative dash-line h-full flex items-center">
                            <div
                                class="transition-[height,background-color,box-shadow] duration-220 ease-[cubic-bezier(0.22,1,0.36,1)] mx-auto w-1.5 h-1.5 rounded-full group-hover:h-5
                                bg-[oklch(0.55_0.06_var(--hue))] group-hover:bg-[var(--primary)]
                                outline outline-4 z-10
                                outline-[var(--card-bg)]
                                group-hover:outline-[var(--btn-plain-bg-hover)]
                                group-active:outline-[var(--btn-plain-bg-active)]"
                            ></div>
                        </div>

                        <!-- post title -->
                        <div
                            class="w-[70%] md:max-w-[58%] md:w-[58%] text-left font-bold text-sm md:text-[0.95rem]
                            group-hover:translate-x-1 transition-[transform,color] duration-220 ease-[cubic-bezier(0.22,1,0.36,1)] group-hover:text-[var(--primary)]
                            text-75 pr-4 whitespace-nowrap overflow-ellipsis overflow-hidden"
                        >
                            {post.data.title}
                        </div>

                        <!-- tag pills -->
                        <div
                            class="hidden md:flex md:w-[22%] justify-end items-center gap-1.5 pr-2 overflow-hidden"
                        >
                            {#each (post.data.tags || []).slice(0, 2) as tag}
                                <span class="px-2 py-0.5 rounded-md text-[0.7rem] font-medium bg-black/[0.035] dark:bg-white/[0.055] text-black/45 dark:text-white/45 group-hover:text-[var(--primary)] group-hover:bg-[var(--btn-regular-bg)] transition-colors whitespace-nowrap truncate max-w-[6.5rem]">
                                    #{tag}
                                </span>
                            {/each}
                        </div>
                    </div>
                </a>
            {/each}
        </div>
    {/each}
</div>
