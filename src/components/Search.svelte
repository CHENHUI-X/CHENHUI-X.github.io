<script lang="ts">
import I18nKey from "@i18n/i18nKey";
import { i18n } from "@i18n/translation";
import Icon, { type IconifyIcon } from "@iconify/svelte";
import { url } from "@utils/url-utils.ts";
import { onMount } from "svelte";
import type { SearchResult } from "@/global";

export let searchIcon: IconifyIcon;

let keywordDesktop = "";
let keywordMobile = "";
let result: SearchResult[] = [];
let isSearching = false;
let pagefindLoaded = false;
let initialized = false;

let desktopInputEl: HTMLInputElement | null = null;
let mobileInputEl: HTMLInputElement | null = null;

const fakeResult: SearchResult[] = [
	{
		url: url("/"),
		meta: {
			title: "开发环境示例结果",
		},
		excerpt: "全文搜索索引仅在生产构建时生成（使用 <mark>pnpm build</mark>）。",
	},
	{
		url: url("/topics/"),
		meta: {
			title: "按专题浏览全部文章",
		},
		excerpt: "可在开发阶段直接通过主题目录或时间归档浏览内容。",
	},
];

const togglePanel = () => {
	const panel = document.getElementById("search-panel");
	panel?.classList.toggle("float-panel-closed");
};

const setPanelVisibility = (show: boolean, isDesktop: boolean): void => {
	const panel = document.getElementById("search-panel");
	if (!panel || !isDesktop) return;

	if (show) {
		panel.classList.remove("float-panel-closed");
	} else {
		panel.classList.add("float-panel-closed");
	}
};

const search = async (keyword: string, isDesktop: boolean): Promise<void> => {
	if (!keyword) {
		setPanelVisibility(false, isDesktop);
		result = [];
		return;
	}

	if (!initialized) {
		return;
	}

	isSearching = true;

	try {
		let searchResults: SearchResult[] = [];

		if (import.meta.env.PROD && pagefindLoaded && window.pagefind) {
			const response = await window.pagefind.search(keyword);
			searchResults = await Promise.all(
				response.results.map((item) => item.data()),
			);
		} else if (import.meta.env.DEV) {
			searchResults = fakeResult;
		} else {
			searchResults = [];
			console.error("Pagefind is not available in production environment.");
		}

		result = searchResults;
		setPanelVisibility(result.length > 0, isDesktop);
	} catch (error) {
		console.error("Search error:", error);
		result = [];
		setPanelVisibility(false, isDesktop);
	} finally {
		isSearching = false;
	}
};

onMount(() => {
	const initializeSearch = () => {
		initialized = true;
		pagefindLoaded =
			typeof window !== "undefined" &&
			!!window.pagefind &&
			typeof window.pagefind.search === "function";
		if (keywordDesktop) search(keywordDesktop, true);
		if (keywordMobile) search(keywordMobile, false);
	};

	const onKeydown = (e: KeyboardEvent) => {
		const target = e.target as HTMLElement | null;
		const inInput =
			!!target &&
			(target.tagName === "INPUT" ||
				target.tagName === "TEXTAREA" ||
				target.isContentEditable);

		if (
			((e.key === "k" || e.key === "K") && (e.metaKey || e.ctrlKey)) ||
			(e.key === "/" && !e.metaKey && !e.ctrlKey && !e.altKey && !inInput)
		) {
			if (desktopInputEl && window.innerWidth >= 1024) {
				e.preventDefault();
				desktopInputEl.focus();
				desktopInputEl.select();
			}
			return;
		}

		if (e.key === "Escape") {
			const panel = document.getElementById("search-panel");
			if (panel && !panel.classList.contains("float-panel-closed")) {
				panel.classList.add("float-panel-closed");
				desktopInputEl?.blur();
				mobileInputEl?.blur();
			}
		}
	};
	window.addEventListener("keydown", onKeydown);

	if (import.meta.env.DEV || window.pagefind?.search) {
		initializeSearch();
	} else {
		document.addEventListener("pagefindready", () => {
			initializeSearch();
		});
		document.addEventListener("pagefindloaderror", () => {
			initializeSearch(); // Initialize with pagefindLoaded as false
		});

		// Fallback in case events are not caught or pagefind is already loaded by the time this script runs
		setTimeout(() => {
			if (!initialized) {
				initializeSearch();
			}
		}, 2000);
	}

	return () => {
		window.removeEventListener("keydown", onKeydown);
	};
});

$: if (initialized && keywordDesktop) {
	(async () => {
		await search(keywordDesktop, true);
	})();
}

$: if (initialized && keywordMobile) {
	(async () => {
		await search(keywordMobile, false);
	})();
}
</script>

<!-- search bar for desktop view -->
<div id="search-bar" class="group hidden lg:flex relative transition-[background-color,box-shadow,border-color] duration-200 items-center h-10 mr-1.5 rounded-[var(--radius-btn)] border border-[var(--border-subtle)]
      bg-black/[0.035] hover:bg-black/[0.055] focus-within:bg-black/[0.045] focus-within:border-[var(--primary)]/40 focus-within:ring-2 focus-within:ring-[var(--primary)]/15
      dark:bg-white/[0.045] dark:hover:bg-white/[0.08] dark:focus-within:bg-white/[0.07]
">
    <Icon icon={searchIcon} class="absolute text-[1.15rem] pointer-events-none ml-3 transition-colors my-auto text-black/40 dark:text-white/40"></Icon>
    <input bind:this={desktopInputEl} placeholder="{i18n(I18nKey.search)}" bind:value={keywordDesktop} on:focus={() => search(keywordDesktop, true)}
           class="transition-[width] duration-250 ease-[cubic-bezier(0.22,1,0.36,1)] pl-9 pr-8 text-sm bg-transparent outline-0
         h-full w-28 active:w-32 focus:w-32 xl:w-32 xl:active:w-44 xl:focus:w-44 text-black/75 dark:text-white/80"
    >
    <kbd class="pointer-events-none absolute right-2.5 text-[10px] font-mono px-1.5 py-0.5 rounded border border-[var(--border-subtle)] text-black/40 dark:text-white/40 group-focus-within:opacity-0 transition-opacity">/</kbd>
</div>

<!-- toggle btn for phone/tablet view -->
<button on:click={togglePanel} aria-label="Search Panel" id="search-switch"
        class="btn-plain scale-animation lg:!hidden rounded-[var(--radius-btn)] w-10 h-10 active:scale-90">
    <Icon icon={searchIcon} class="text-[1.25rem]"></Icon>
</button>

<!-- search panel -->
<div id="search-panel" class="float-panel float-panel-closed search-panel absolute md:w-[30rem]
top-20 left-4 md:left-[unset] right-4 shadow-2xl rounded-2xl p-2">

    <!-- search bar inside panel for phone/tablet -->
    <div id="search-bar-inside" class="flex relative lg:hidden transition-colors items-center h-11 rounded-[var(--radius-btn)]
      border border-[var(--border-subtle)]
      bg-black/[0.04] hover:bg-black/[0.06] focus-within:bg-black/[0.06]
      dark:bg-white/5 dark:hover:bg-white/10 dark:focus-within:bg-white/10
  ">
        <Icon icon={searchIcon} class="absolute text-[1.25rem] pointer-events-none ml-3 transition-colors my-auto text-black/35 dark:text-white/35"></Icon>
        <input bind:this={mobileInputEl} placeholder="{i18n(I18nKey.search)}" bind:value={keywordMobile}
               class="pl-10 pr-3 w-full h-full text-sm bg-transparent outline-0 text-black/75 dark:text-white/80"
        >
    </div>

    <!-- search results -->
    {#each result as item}
        <a href={item.url}
           class="transition first-of-type:mt-2 lg:first-of-type:mt-0 group block
       rounded-xl text-lg px-3 py-2 hover:bg-[var(--btn-plain-bg-hover)] active:bg-[var(--btn-plain-bg-active)]">
            <div class="transition text-90 inline-flex font-bold group-hover:text-[var(--primary)]">
                {item.meta.title}<Icon icon="fa6-solid:chevron-right" class="transition text-[0.75rem] translate-x-1 my-auto text-[var(--primary)]"></Icon>
            </div>
            <div class="transition text-sm text-50">
                {@html item.excerpt}
            </div>
        </a>
    {/each}
</div>

<style>
  input:focus {
    outline: 0;
  }
  .search-panel {
    max-height: calc(100vh - 100px);
    overflow-y: auto;
  }
</style>
