# 博客技术文档

基于 [Fuwari](https://github.com/saicaca/fuwari) 模板构建的个人博客，部署地址：**[xuchenhui.cc](https://xuchenhui.cc)**

---

## 技术栈

| 层级 | 技术 |
|---|---|
| 框架 | [Astro](https://astro.build) 5.x |
| 样式 | [Tailwind CSS](https://tailwindcss.com) 3.x + [Stylus](https://stylus-lang.com) |
| UI 组件 | [Svelte](https://svelte.dev) 5.x |
| 页面切换 | [@swup/astro](https://swup.js.org) — SPA 式导航，自定义淡入缩放动画 |
| 代码块 | [Expressive Code](https://expressive-code.com) — 语法高亮、行号、可折叠区块、自定义复制按钮 |
| 数学公式 | [KaTeX](https://katex.org)，通过 `remark-math` + `rehype-katex` 渲染 |
| 全文搜索 | [Pagefind](https://pagefind.app) — 静态索引，构建时生成 |
| 图片灯箱 | [PhotoSwipe](https://photoswipe.com) 5.x |
| 滚动条 | [OverlayScrollbars](https://kingsora.github.io/OverlayScrollbars) |
| 图标 | [Astro Icon](https://github.com/natemoo-re/astro-icon) — Material Symbols + Font Awesome 6 |
| 字体 | Roboto（界面）· JetBrains Mono Variable（代码），均通过 `@fontsource` 引入 |
| 评论 | [Waline](https://waline.js.org) — 基于 LeanCloud / MongoDB，支持表情反应、阅读量 |
| 访问统计 | Google Analytics（`G-P9JSV1JJEK`） |
| 页面浏览量 | [不蒜子 Busuanzi](https://busuanzi.ibruce.info) |
| 站点地图 | `@astrojs/sitemap` |
| RSS | `@astrojs/rss` |
| 包管理器 | pnpm 9.x |
| 代码规范 | [Biome](https://biomejs.dev) |
| 部署平台 | [Vercel](https://vercel.com) |

---

## 功能特性

### 内容
- Markdown 扩展语法：提示块（admonitions）、GitHub 卡片嵌入、剧透标签、可折叠代码区块
- 行内与块级数学公式（KaTeX）
- 文章阅读时间估算
- 自动生成摘要
- 文章置顶
- 目录（自动生成，滚动同步高亮）
- 图片点击放大（PhotoSwipe）

### 界面 / 交互
- 亮色 / 暗色模式，支持跟随系统 + 手动切换
- 主题色相自定义（持久化到 `localStorage`）
- 动态渐变背景 + 浮动几何形状（`public/js/bg-anim.js`）
- 外部 API 壁纸轮播，每 30 秒切换，交叉淡入淡出
- Apple 风格页面切换：快速离场（150ms ease-in）+ 弹性入场（500ms easeOutExpo）
- 首屏错落入场动画（导航栏 → 侧边栏 → 内容区 → 页脚）
- 卡片 hover 弹性上浮动画（`cubic-bezier(0.34, 1.56, 0.64, 1)`）
- 毛玻璃卡片（backdrop-filter blur + 半透明背景）
- 回到顶部按钮 + 跳到底部按钮（滚动到评论区）
- 响应式布局（移动端 / 平板 / 桌面）
- 自定义覆盖式滚动条

### 技术
- 静态站点生成（SSG），无服务端运行时
- Swup SPA 导航，页面切换无刷新
- Pagefind 搜索索引在构建时生成
- RSS 订阅：`/rss.xml`
- 站点地图：`/sitemap-index.xml`
- 每页自动生成 OG / Twitter Card meta 标签
- 按色彩方案切换 favicon（亮色 / 暗色）

---

## 外部 API

| API | 用途 | 接口地址 |
|---|---|---|
| 夜轻 (yppp.net) | 壁纸 — 横屏 | `https://api.yppp.net/pc.php?return=json` |
| 夜轻 (yppp.net) | 壁纸 — 竖屏 | `https://api.yppp.net/pe.php?return=json` |
| 赫萝 (horosama.com) | 壁纸备用接口 | `https://api.horosama.com/random.php?type=pc&format=json` |
| Google Analytics | 访问分析 | `https://www.googletagmanager.com/gtag/js?id=G-P9JSV1JJEK` |
| 不蒜子 Busuanzi | 页面浏览量 | `https://busuanzi.ibruce.info/busuanzi/2.3/busuanzi.pure.mini.js` |
| Waline | 评论系统 | `@waline/client` 客户端组件嵌入文章布局 |

> 壁纸请求设有 8 秒 `AbortController` 超时，主接口失败时自动切换到赫萝备用接口。

---

## 关键文件路径

```
src/
  config.ts                    # 站点配置：标题、作者、导航、主题色、banner
  content/
    posts/                     # 博客文章（.md）
    spec/
      about.md                 # 关于页内容
  layouts/
    Layout.astro               # 根布局：Swup、滚动逻辑、背景动画
    MainGridLayout.astro       # 主网格布局（侧边栏 + 内容区）
  components/
    control/
      BackToTop.astro          # 回到顶部 + 跳到底部按钮
    misc/
      Markdown.astro           # Markdown 容器 + 复制按钮事件处理
  styles/
    main.css                   # 全局组件样式（卡片、按钮等）
    transition.css             # 页面切换 + 首屏入场动画关键帧
public/
  js/
    bg-anim.js                 # 背景渐变 + 浮动形状 + 壁纸轮播
```

---

## 开发命令

```bash
pnpm dev          # 启动开发服务器
pnpm build        # 构建 + 生成 Pagefind 搜索索引
pnpm preview      # 预览生产构建
pnpm new-post     # 创建新文章
pnpm lint         # Biome 代码检查 + 格式化
```
