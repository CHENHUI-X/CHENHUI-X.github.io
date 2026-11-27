<script lang="ts">
import I18nKey from "@i18n/i18nKey";
import { i18n } from "@i18n/translation";
import Icon from "@iconify/svelte";
import { getDefaultHue, getHue, setHue } from "@utils/setting-utils";

let hue = getHue();
const defaultHue = getDefaultHue();
const presets = [
	{ name: "玫瑰", value: 345 },
	{ name: "科研蓝", value: 235 },
	{ name: "靛青", value: 195 },
	{ name: "松绿", value: 155 },
	{ name: "琥珀", value: 40 },
];

function resetHue() {
	hue = getDefaultHue();
}

$: if (hue || hue === 0) {
	setHue(hue);
}
</script>

<div id="display-setting" class="float-panel float-panel-closed absolute w-80 right-4 px-4 py-4">
    <div class="flex flex-row gap-2 mb-3 items-center justify-between">
        <div class="flex gap-2 font-bold text-lg text-neutral-900 dark:text-neutral-100 relative ml-3
            before:w-1 before:h-4 before:rounded-md before:bg-[var(--primary)]
            before:absolute before:-left-3 before:top-[0.33rem]"
        >
            {i18n(I18nKey.themeColor)}
            <button aria-label="Reset to Default" class="btn-tonal w-7 h-7 rounded-md"
                    class:opacity-0={hue === defaultHue} class:pointer-events-none={hue === defaultHue} on:click={resetHue}>
                <div class="text-[var(--btn-content)]">
                    <Icon icon="fa6-solid:arrow-rotate-left" class="text-[0.875rem]"></Icon>
                </div>
            </button>
        </div>
        <div class="flex gap-1">
            <div id="hueValue" class="bg-[var(--btn-regular-bg)] border border-[var(--border-subtle)] w-10 h-7 rounded-md flex justify-center
            font-bold text-sm items-center text-[var(--btn-content)] tabular-nums">
                {hue}
            </div>
        </div>
    </div>
    <div class="w-full h-6 px-1 bg-[oklch(0.80_0.10_0)] dark:bg-[oklch(0.70_0.10_0)] rounded-lg select-none mb-3">
        <input aria-label={i18n(I18nKey.themeColor)} type="range" min="0" max="360" bind:value={hue}
               class="slider" id="colorSlider" step="5" style="width: 100%">
    </div>
    <div class="flex items-center justify-between gap-1.5">
        {#each presets as preset}
            <button
                type="button"
                on:click={() => (hue = preset.value)}
                class="flex-1 inline-flex items-center justify-center gap-1.5 h-7 rounded-md text-xs font-medium border transition-[background-color,border-color,color] duration-150"
                class:bg-[var(--btn-regular-bg)]={hue === preset.value}
                class:text-[var(--primary)]={hue === preset.value}
                class:border-[var(--primary)]={hue === preset.value}
                class:border-[var(--border-subtle)]={hue !== preset.value}
                class:text-[var(--text-75)]={hue !== preset.value}
            >
                <span class="w-2 h-2 rounded-full shrink-0" style={`background: oklch(0.68 0.15 ${preset.value})`}></span>
                <span>{preset.name}</span>
            </button>
        {/each}
    </div>
</div>


<style lang="stylus">
    #display-setting
      input[type="range"]
        -webkit-appearance none
        height 1.5rem
        background-image var(--color-selection-bar)
        transition background-image 0.15s ease-in-out

        /* Input Thumb */
        &::-webkit-slider-thumb
          -webkit-appearance none
          height 1rem
          width 1rem
          border-radius 9999px
          background var(--primary)
          border 2px solid #fff
          box-shadow 0 1px 4px rgba(0, 0, 0, 0.3)
          &:hover
            transform scale(1.08)
          &:active
            transform scale(0.95)

        &::-moz-range-thumb
          -webkit-appearance none
          height 1rem
          width 1rem
          border-radius 9999px
          background var(--primary)
          border 2px solid #fff
          box-shadow 0 1px 4px rgba(0, 0, 0, 0.3)

        &::-ms-thumb
          -webkit-appearance none
          height 1rem
          width 1rem
          border-radius 9999px
          background var(--primary)
          border 2px solid #fff
          box-shadow 0 1px 4px rgba(0, 0, 0, 0.3)

</style>
