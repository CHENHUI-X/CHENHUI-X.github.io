# 博客技术文档

基于 [Fuwari](https://github.com/saicaca/fuwari) 模板构建的个人博客
- 部署地址：**[xuchenhui.cc](https://xuchenhui.cc)**
- 备用域名：**haibarai.dpdns.org**
- 部署平台：Cloudflare Pages（自动从 GitHub main 分支部署）

---

## 技术栈

| 层级 | 技术 |
|---|---|
| 框架 | [Astro](https://astro.build) 5.x（SSG 静态生成） |
| 样式 | [Tailwind CSS](https://tailwindcss.com) 3.x + [Stylus](https://stylus-lang.com) |
| UI 组件 | [Svelte](https://svelte.dev) 5.x |
| 页面切换 | [@swup/astro](https://swup.js.org) — SPA 式无刷新导航 |
| 代码块 | [Expressive Code](https://expressive-code.com) — 语法高亮、行号、可折叠区块 |
| 数学公式 | [KaTeX](https://katex.org)，通过 `remark-math` + `rehype-katex` 渲染 |
| 全文搜索 | [Pagefind](https://pagefind.app) — 构建时生成静态索引 |
| 图片灯箱 | [PhotoSwipe](https://photoswipe.com) 5.x |
| 滚动条 | [OverlayScrollbars](https://kingsora.github.io/OverlayScrollbars) — 自定义覆盖式滚动条 |
| 图标 | [Astro Icon](https://github.com/natemoo-re/astro-icon) — Material Symbols + Font Awesome 6 |
| 字体 | Roboto（界面）· JetBrains Mono Variable（代码），均通过 `@fontsource` 引入 |
| 评论 | [Waline](https://waline.js.org) v1.39.3 + MongoDB Atlas — 托管于 Vercel |
| 访问统计 | Google Analytics（`G-P9JSV1JJEK`） |
| 页面浏览量 | [不蒜子 Busuanzi](https://busuanzi.ibruce.info) |
| 站点地图 | `@astrojs/sitemap` |
| RSS | `@astrojs/rss` |
| 包管理器 | pnpm 9.x |
| 代码规范 | [Biome](https://biomejs.dev) — `pnpm lint` 自动修复 |

---

## 功能特性

### 内容
- Markdown 扩展语法：提示块（admonitions）、GitHub 卡片嵌入、剧透标签、可折叠代码区块
- 行内与块级数学公式（KaTeX）
- 文章阅读时间估算（`remark-reading-time`）
- 自动生成摘要
- 文章置顶
- 目录（自动生成，滚动同步高亮）
- 图片点击放大（PhotoSwipe）
- 文章字数统计 + 与经典书籍字数对比（如"相当于《道德经》的一半"）

### 界面 / 交互
- 亮色 / 暗色模式，支持跟随系统 + 手动切换
- 主题色相自定义（持久化到 `localStorage`，`--hue` CSS 变量）
- 动态渐变背景 + 浮动几何形状（`public/js/bg-anim.js`）
- 外部 API 壁纸轮播，每 30 秒切换，交叉淡入淡出（8 秒 AbortController 超时，自动切换备用接口）
- Apple 风格页面切换：快速离场（150ms ease-in）+ 弹性入场（500ms easeOutExpo）
- 首屏错落入场动画（导航栏 → 侧边栏 → 内容区 → 页脚）
- 卡片 hover 弹性上浮动画（`cubic-bezier(0.34, 1.56, 0.64, 1)`）
- 毛玻璃卡片（backdrop-filter blur + 半透明背景）
- 回到顶部按钮 + 跳到底部按钮（滚动到 Waline 评论区）
- 响应式布局（移动端 / 平板 / 桌面）
- 自定义覆盖式滚动条
- 博客统计卡片：文章数、标签数、全站字数、建站天数、最后更新（`BlogStats.astro`）

### 技术
- 静态站点生成（SSG），无服务端运行时
- Swup SPA 导航，页面切换无刷新
- Pagefind 搜索索引在构建时生成
- RSS 订阅：`/rss.xml`
- 站点地图：`/sitemap-index.xml`
- 每页自动生成 OG / Twitter Card meta 标签
- 按色彩方案切换 favicon（亮色 / 暗色）

---

## 评论系统（Waline）

### 服务端
- **地址**：https://waline-comment-smoky.vercel.app
- **版本**：Waline 1.39.3（ThinkJS 框架）
- **数据库**：MongoDB Atlas（免费 512M 档）
- **管理后台**：https://waline-comment-smoky.vercel.app/ui/
- **环境变量**：MongoDB 连接信息（`MONGO_HOST`/`PORT`/`DB`/`USER`/`PASSWORD`/`REPLICASET` 等）+ `SECURE_DOMAINS` + `MONGO_OPT_SSL=true`
  - 注意：Waline 1.x 不支持 `MONGODB_URI`，需用散装 `MONGO_*` 变量，`MONGO_HOST` 用 JSON 数组格式
  - `SECURE_DOMAINS` 必须包含 Waline 服务端自身域名，否则管理后台无法登录
- **管理员登录**：在管理后台用注册邮箱 + 密码登录

### 客户端组件
- 文件：`src/components/WalineComment.astro`
- 使用 `@waline/client@3` npm 包
- 关键 init 选项：`requiredMeta: ["nick", "mail"]`（强制必填昵称和邮箱，禁止匿名评论）
- 支持表情反应（`reaction: true`）和阅读量（`pageview: true`）
- 暗色模式适配选择器：`dark: "html.dark"`
- 集成 Swup 生命周期：`content:replace` 时销毁，`page:view` 时重新初始化

---

## 外部 API

| API | 用途 | 接口地址 |
|---|---|---|
| 夜轻 (yppp.net) | 壁纸 — 横屏 | `https://api.yppp.net/pc.php?return=json` |
| 夜轻 (yppp.net) | 壁纸 — 竖屏 | `https://api.yppp.net/pe.php?return=json` |
| 赫萝 (horosama.com) | 壁纸备用接口 | `https://api.horosama.com/random.php?type=pc&format=json` |
| Google Analytics | 访问分析 | `https://www.googletagmanager.com/gtag/js?id=G-P9JSV1JJEK` |
| 不蒜子 Busuanzi | 页面浏览量 | `https://busuanzi.ibruce.info/busuanzi/2.3/busuanzi.pure.mini.js` |
| Waline 服务端 | 评论系统 API | `https://waline-comment-smoky.vercel.app` |

> 壁纸请求设有 8 秒 `AbortController` 超时，主接口失败时自动切换到赫萝备用接口。

---

## 关键文件路径

```
根目录/
  CLAUDE.md                     # 项目级 Claude 指令
  BLOG.md                       # 本文档

src/
  config.ts                     # 站点配置：标题、作者、导航、主题色、banner、license
  types/config.ts               # 类型定义

  content/
    posts/                      # 博客文章（.md）
    spec/
      about.md                  # 关于页内容

  layouts/
    Layout.astro                # 根布局：Swup、滚动逻辑、背景动画、Google Analytics
    MainGridLayout.astro        # 主网格布局（侧边栏 + 内容区）

  components/
    control/
      BackToTop.astro           # 回到顶部 + 跳到底部按钮
    misc/
      Markdown.astro            # Markdown 容器 + 复制按钮事件处理
    widget/
      BlogStats.astro           # 博客统计卡片 + 书籍字数对比
    WalineComment.astro         # Waline 评论组件
    Giscus.astro (已删除)        # 已被 Waline 替换

  styles/
    main.css                    # 全局组件样式（卡片、按钮、毛玻璃）
    transition.css              # 页面切换 + 首屏入场动画关键帧

public/
  js/
    bg-anim.js                  # 背景渐变 + 浮动形状 + 壁纸轮播
  css/
    giscus-light.css (已删除)    # 已被 Waline 替换
    giscus-dark.css (已删除)     # 已被 Waline 替换

memory/                         # Claude 持久化记忆
  MEMORY.md                     # 记忆索引
```

---

## 开发命令

```bash
pnpm dev          # 启动开发服务器（默认 http://localhost:4321）
pnpm build        # 构建 + 生成 Pagefind 搜索索引（输出到 dist/）
pnpm preview      # 预览生产构建
pnpm new-post     # 创建新文章
pnpm lint         # Biome 代码检查 + 自动格式化
pnpm check        # Astro 类型检查
pnpm type-check   # tsc 类型检查
```
