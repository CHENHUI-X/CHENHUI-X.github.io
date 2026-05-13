---
title: 博客性能优化手记
published: 2026-05-14
description: 记录本站前端性能优化的全过程 — 从背锅的毛玻璃到首帧渐变的 CSS 动画，从 2 秒淡入到 localStorage 壁纸缓存，以及滚动卡顿的排除心法。
category: Meta
tags:
- performance
- optimization
- Astro
- CSS
draft: false
---

## 背景

本站基于 Astro 5.x + Fuwari 主题构建，部署在 Cloudflare Pages。作为一个技术博客，长文章（单篇 37KB+ Markdown）和丰富的交互效果（背景动画、壁纸轮播、毛玻璃卡片、SPA 导航）让页面渲染压力不小。

早期的体验反馈主要集中在这几个问题：
1. **刷新时闪白 / 闪粉红** — 背景渐变动画靠 JS 启动，首帧无内容
2. **壁纸出现太慢** — API 请求 + 2 秒淡入，总等 4~8 秒
3. **滚动卡顿** — 毛玻璃 `backdrop-filter: blur()` 每帧触发 GPU 重采样
4. **评论区加载慢** — unpkg CDN 国内访问不佳

以下是逐一排查和修复的记录。

---

## 1. 背景动画：从 JS 到 CSS 的转移

### 问题

原来的背景渐变动画在 `bg-anim.js` 中通过 `requestIdleCallback` 延迟执行（最多 500ms），然后注入 `<style>` 标签 + 设置 `animation` 属性。这 500ms 里 body 只有 CSS 默认的粉红色，用户刷新时会先看到一片纯色，然后渐变才开始转。

### 解决

把 `@keyframes` 定义和 `#bg-gradient-layer` 的 `animation` 属性移到了 `Layout.astro` 的**内联 `<style>`** 中，随 HTML 同步解析渲染：

```html
<style>
  @keyframes bg-grad-light {
    0%   { background-color: rgb(255,150,170); }
    20%  { background-color: rgb(255,195,130); }
    40%  { background-color: rgb(185,145,255); }
    60%  { background-color: rgb(130,200,255); }
    80%  { background-color: rgb(140,235,200); }
    100% { background-color: rgb(255,150,170); }
  }
  #bg-gradient-layer {
    animation: bg-grad-light 25s ease-in-out infinite;
  }
  .dark #bg-gradient-layer {
    animation: bg-grad-dark 25s ease-in-out infinite;
  }
</style>
```

关键点：暗色模式通过 CSS `.dark` 选择器自动切换，**不再需要 `MutationObserver`**。渐变从**浏览器首次绘制就开始动画**，零 JS 延迟。

---

## 2. 壁纸轮播：localStorage 缓存 + 预加载

### 问题

壁纸经历了两个不必要的延迟：
1. `requestIdleCallback` 延迟 ~500ms 后才发出 API 请求
2. 壁纸淡入时间 2000ms，用户需要看着渐变转 4~7 秒才看到壁纸

### 解决

- **API 请求立即发出** — `fetch()` 放在 IIFE 顶部，不经过 `requestIdleCallback`
- **localStorage 缓存** — 每次获取到壁纸 URL 后缓存，下次刷新时直接预加载显示，**秒出**
- **淡入时间缩短** — `transition: opacity 2s` → `500ms`

时序对比：

| 阶段 | 优化前 | 首次访问 | 后续刷新 |
|------|--------|----------|----------|
| 渐变启动 | 第 1 帧 | 第 1 帧 | 第 1 帧 |
| API 请求 | ~500ms 后才发 | 立即 | 立即 |
| 壁纸出现 | ~4~7s | ~1.5~4s | **~0ms** |
| 淡入 | 2s | 0.5s | instant |

---

## 3. 滚动卡顿：`backdrop-filter` 是头号杀手

### 排查过程

用户反馈「滚动比较卡」，打开 DevTools Performance 面板，发现每帧都有大量 **Paint** 和 **Layer** 活动。

使用 Chrome 的「Rendering → Paint flashing」和「Layers」面板，定位到 `.card-base` 的 `backdrop-filter: blur(8px)` 是主要原因。每次滚动，卡片后面内容变化，GPU 都需要对每个卡片区域的像素重新做高斯模糊采样。

### 解决

移除 `backdrop-filter`，保留半透明 `rgba(255,255,255,0.80)` 底色的视觉通透感：

```css
.card-base {
  background-color: rgba(255, 255, 255, 0.80) !important;
  /* 去掉 backdrop-filter，避免每帧 GPU 高斯模糊采样 */
}
```

额外优化：
- `will-change: transform` 将内容层提升为独立 compositor 层
- 滚动时通过 `.bg-scroll-active` 类**暂停所有背景动画**，释放帧预算
- `content-visibility: auto` 让长文章中视口外的元素跳过渲染

---

## 4. 评论组件：Waline + 骨架屏

### 问题

Waline 评论通过 `unpkg.com` 加载，国内访问速度慢。同时评论加载完成前评论区是一片空白。

### 解决

- CDN 从 `unpkg.com` 切换到 `cdn.jsdelivr.net`
- 添加 `<head>` 中的 `preconnect` / `dns-prefetch`
- **骨架屏**：加载过程中显示头像占位、输入框骨架、评论列表骨架，带脉冲动画
- 10 秒超时兜底，骨架屏自动消失

---

## 5. 性能优化清单

| 优化项 | 涉及文件 | 效果 |
|--------|----------|------|
| 渐变内联 CSS | `Layout.astro` | 零 JS 延迟，首帧即动画 |
| 壁纸缓存 + 预加载 | `bg-anim.js` | 后续刷新秒出壁纸 |
| 滚动暂停动画 | `bg-anim.js`, `Layout.astro` | 滚动流畅度提升 |
| 移除 backdrop-filter | `main.css` | 滚动不再 GPU 重采样 |
| content-visibility | `markdown.css` | 长文章首次渲染加速 |
| 去除过渡动画 | `transition.css` | 减少不必要的 Layout/Paint |
| Waline CDN 切换 | `WalineComment.astro` | 评论加载加速 |
| 评论骨架屏 | `WalineComment.astro` | 用户感知加载更快 |
| 资源预连接 | `Layout.astro` | DNS + TCP 提前建立 |

---

## 结语

性能优化不是一蹴而就的，每次改动后都需要在真实设备上验证效果。Chrome DevTools 的 Performance 面板、Layers 面板和 `content-visibility` 的配合使用是本次优化的主要工具链。

如果你也在做 Astro 博客的性能优化，希望这份记录能给你一些参考。
