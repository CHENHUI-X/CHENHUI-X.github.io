# CHENHUI 博客项目指南

## 项目概述
基于 Astro 5.x + Fuwari 主题的个人博客（SSG），部署在 Cloudflare Pages。

## 关键命令
- `pnpm dev` — 开发服务器
- `pnpm build` — 构建 + Pagefind 索引
- `pnpm lint` — Biome 检查 + 自动格式化

## 目录结构
- `src/config.ts` — 站点主配置（标题、导航、主题色、license 等）
- `src/components/WalineComment.astro` — 评论组件
- `src/components/widget/BlogStats.astro` — 博客统计卡片
- `src/layouts/Layout.astro` — 根布局（Swup 钩子、滚动逻辑）
- `public/js/bg-anim.js` — 背景动画 + 壁纸轮播
- 文章在 `src/content/posts/`

## 重要模式
1. **Swup SPA 导航**：页面切换无刷新，`visit:start` / `content:replace` / `page:view` 钩子管理生命周期
2. **Waline 评论**：组件用 `data-slug` + 动态 `import()`，避免 `define:vars` + 静态 import 的 Astro IIFE 语法问题
3. **暗色模式**：`class="dark"` 在 `<html>` 上，Waline 配置 `dark: "html.dark"`
4. **动画系统**：`transition.css` 定义 `is-changing` / `is-animating` / `is-rendering` 类，`onload-animation` 首屏入场

## Waline 配置参考
- 服务端：https://waline-comment-smoky.vercel.app
- 管理后台：https://waline-comment-smoky.vercel.app/ui/
- 客户端必须 `requiredMeta: ["nick", "mail"]` 禁止匿名评论
- SECURE_DOMAINS 必须包含 Waline 服务端自身域名才能登录管理后台

## 注意事项
- ⚠️ 修改代码后**不要自动推送**，等用户确认后再推送
- Biome 检查（`pnpm lint`）是 CI 必过项，改代码后必须运行
- 构建命令是 `pnpm build && pagefind --site dist`
