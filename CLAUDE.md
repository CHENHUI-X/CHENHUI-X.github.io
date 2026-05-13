# CHENHUI 博客项目指南

## 项目概述
基于 Astro 5.x + Fuwari 主题的个人博客（SSG），部署在 Cloudflare Pages。

## 关键命令
- `pnpm dev --port 4321 --host 0.0.0.0` — 开发服务器（指定端口 4321）
- `pnpm build` — 构建 + Pagefind 索引
- `pnpm lint` — Biome 检查 + 自动格式化

## 目录结构
- `src/config.ts` — 站点主配置（标题、导航、主题色、license 等）
- `src/components/WalineComment.astro` — 评论组件（jsdelivr CDN + 骨架屏）
- `src/components/widget/BlogStats.astro` — 博客统计卡片
- `src/layouts/Layout.astro` — 根布局（Swup 钩子、滚动逻辑、内联渐变 CSS 动画）
- `public/js/bg-anim.js` — 背景动画 + 壁纸轮播（localStorage 缓存，scroll 暂停）
- `src/styles/main.css` — 卡片样式（无 backdrop-filter，无通用 transition）
- `src/styles/transition.css` — Swup SPA 过渡动画（无文章入场阶梯动画）
- `src/styles/markdown.css` — 文章排版（含 content-visibility: auto）
- 文章在 `src/content/posts/`

## 重要模式
1. **Swup SPA 导航**：页面切换无刷新，`visit:start` / `content:replace` / `page:view` 钩子管理生命周期
2. **Waline 评论**：组件用 `data-slug` + 动态 `import()`，避免 `define:vars` + 静态 import 的 Astro IIFE 语法问题
3. **暗色模式**：`class="dark"` 在 `<html>` 上，Waline 配置 `dark: "html.dark"`
4. **动画系统**：`transition.css` 定义 `is-changing` / `is-animating` / `is-rendering` 类
5. **背景动画系统**：
   - 渐变（`@keyframes bg-grad-light/bg-grad-dark`）通过内联 CSS 注入，首帧即生效，零 JS 延迟
   - 壁纸轮播（`bg-anim.js`）通过 `requestIdleCallback` 延迟执行 API 请求 + 形状创建
   - 壁纸 URL 缓存到 `localStorage`（key: `hermes_wallpaper_url`），刷新秒出
   - 滚动时通过 `.bg-scroll-active` 暂停所有背景动画，释放帧预算
   - 壁纸图片插入在 `#bg-gradient-layer` 之后，确保层叠顺序正确

## 性能优化要点
- ✅ 渐变内联 CSS — 无需 JS 注入，首帧即动画
- ✅ 壁纸 `localStorage` 缓存 — 刷新秒出
- ✅ `backdrop-filter` 移除 — 滚动不触发 GPU 重采样
- ✅ `content-visibility: auto` — 长文章视口外元素跳过渲染
- ✅ 滚动暂停背景动画 — `animation-play-state: paused`
- ✅ `will-change: transform` — 内容层提升为 compositor 层
- ✅ Waline CDN 切换至 `cdn.jsdelivr.net` — 国内访问加速
- ✅ 评论骨架屏 — 加载体验优化
- ✅ `scroll` 事件 rAF 节流 + resize 事件防抖
- ✅ 资源 `preconnect` / `dns-prefetch`

## Waline 配置参考
- 服务端：https://waline-comment-smoky.vercel.app
- 管理后台：https://waline-comment-smoky.vercel.app/ui/
- 客户端必须 `requiredMeta: ["nick", "mail"]` 禁止匿名评论
- SECURE_DOMAINS 必须包含 Waline 服务端自身域名才能登录管理后台
- CDN 使用 jsdelivr: `https://cdn.jsdelivr.net/npm/@waline/client@3/dist/waline.js`

## 注意事项
- ⚠️ 修改代码后**不要自动推送**，等用户确认后再推送
- Biome 检查（`pnpm lint`）是 CI 必过项，改代码后必须运行
- 构建命令是 `pnpm build && pagefind --site dist`
- 背景渐变 CSS 在 Layout.astro 的 `<head>` 内联 `<style>` 中，不要误删
- 修改 `bg-anim.js` 后需检查 `localStorage` 缓存键名一致
- Dev server 默认端口 4321，如有冲突先 `kill` 旧进程
