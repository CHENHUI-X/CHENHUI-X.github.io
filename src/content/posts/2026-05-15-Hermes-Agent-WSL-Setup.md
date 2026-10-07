---
title: Hermes Agent on WSL：网关、服务恢复与浏览器集成
published: 2026-05-15
description: 一个 Windows 11 与 WSL2 上的 Agent 部署案例：核对微信接入、systemd 服务、跨层健康检查和 Camofox 浏览器配置，并说明历史脚本与验证边界。
category: Technical
tags:
- llm
- agent
- devops
- tool
draft: false
---

## 0. 前言

让 Agent 在终端回答一次问题，与让它在手机发来消息时随时可用，是两件事。后一种用法还涉及消息网关、登录状态、后台服务、宿主机休眠和浏览器进程。

本文以一次 **Windows 11 + WSL2（Ubuntu）** 上的 Hermes Agent 部署案例为例，说明这些环节如何连接，以及没有响应时应该从哪里排查。Prompt、工具调用与 Agent 的概念统一见[LangChain 应用开发篇](/posts/2024-04-24-lang-chain-series/)，这里不再从头解释。

:::note
原稿发表于 2026-05-15，但没有记录当时的 Hermes tag/commit、完整依赖锁及守护脚本。2026-10-07 的整理以当前官方资料核对命令和配置，不能据此追认旧部署的所有行为。文中的恢复测试是建议的验收方法，本次内容整理没有在 Windows/WSL 上重新执行这些测试。
:::

本文仍选择 WSL 路线，并不表示它是唯一的 Windows 安装方式。[Hermes 官方仓库](https://github.com/NousResearch/hermes-agent)目前也提供原生 Windows 安装入口。已经稳定使用某种方式时，先检查其服务管理能力，不必为了照抄文章而切换环境。

## 1. Hermes Agent 是什么

[Hermes Agent](https://github.com/NousResearch/hermes-agent) 是 Nous Research 的开源 Agent 项目，提供终端交互、消息平台接入、工具与记忆等能力。本案例涉及三个部署角色：

1. **Gateway（网关）**：连接消息平台，接收请求并发送回复
2. **Agent 运行逻辑**：根据任务调用模型和允许使用的工具
3. **工具及其依赖**：终端、文件、浏览器或其他服务，可能分别失败

「微信能收到消息」只验证了链路的一部分。模型凭证失效时仍可能收得到消息；浏览器打不开时，普通聊天也可能正常。把这些状态分开观察，比只检查有没有一个 Python 进程更有用。

同时，能从聊天入口调用终端意味着它有实际操作能力。先限定使用者、工具和工作目录，再考虑常驻运行。具体权限与隔离选项参见[Hermes 安全文档](https://hermes-agent.nousresearch.com/docs/user-guide/security)。

## 2. 在 WSL 上安装

### 2.1 先记环境，再装软件

在 Windows PowerShell 中检查：

```powershell
wsl --version
wsl --list --verbose
```

确认目标发行版使用 WSL2，并记下发行版名称。后文的 `Ubuntu` 只是本案例名字，如果实际是 `Ubuntu-24.04`，调用 `wsl -d` 时必须用自己的名称。[Microsoft 的 WSL 命令说明](https://learn.microsoft.com/en-us/windows/wsl/basic-commands)列出了这些参数。

部署记录建议包含：Windows 与 WSL 版本、Linux 发行版、Hermes 安装方式及版本或 commit、模型供应商、浏览器后端版本。只记「安装了最新版」，以后很难判断是配置变化还是升级造成的故障。不要把 API key、网关 token 或浏览器 cookie 放进这份记录。

### 2.2 安装与基本配置

在 WSL 的普通 Linux 用户下，按[官方 WSL 指南](https://hermes-agent.nousresearch.com/docs/user-guide/windows-wsl-quickstart)安装。下面是官方安装器的地址；先下载阅读，再决定是否执行，可以更清楚地知道它会安装什么：

```bash
curl -fsSL https://hermes-agent.nousresearch.com/install.sh -o /tmp/hermes-install.sh
less /tmp/hermes-install.sh
# 阅读并确认安装内容后再执行
bash /tmp/hermes-install.sh
```

这是软件安装操作，不是检查命令。安装后按安装器提示加载环境，再运行：

```bash
hermes setup
hermes --help
hermes gateway --help
```

原部署选用了 OpenRouter。供应商、模型和密钥配置应以安装版本的向导为准，不在文章里固定一个会下线的模型名，也不把凭证直接写进可分享的配置示例。

长期运行时，需要弄清这些限制：任务轮次上限控制一次任务可以推进多久；命令超时约束单条工具调用；它们不是「设置越大，Agent 越可靠」。先采用可观察的小任务，确有需要再调整。旧稿中的 `100` 轮和 `600s` 是当时的取值，不是通用建议或当前默认值。

### 2.3 先在前台跑通

在设置自启动前，先确认终端聊天能完成一次简单请求，然后前台启动 Gateway，观察真实日志。不要一开始就同时启用手工启动、systemd 和自写 watchdog，否则很难知道正在使用哪个进程和哪份配置。

配置和项目文件的路径也要分清：WSL 的 `/home/...`、Windows 的 `C:\...` 和 WSL 看到的 `/mnt/c/...` 不是可直接互换的路径。脚本报 `bad interpreter` 且带 `^M` 时，先检查是否混入 CRLF 换行；不要先怀疑模型或网关。

## 3. 微信网关配置

本案例使用个人微信的 Weixin 适配器，与企业微信 WeCom 是不同接入方式。按[官方 Weixin 文档](https://hermes-agent.nousresearch.com/docs/user-guide/messaging/weixin)，通过下面的向导扫码并在手机确认：

```bash
hermes gateway setup
```

扫码连接的是 iLink bot 身份，不等于获得一个普通微信账号的所有自动化能力。配置完成后先测试私聊，不把群聊作为验收前提。

当前文档中的账号变量是 `WEIXIN_ACCOUNT_ID`，旧稿写的 `WEIXIN_ACCOUNT` 不应直接沿用。个人部署还应明确谁能触发 Agent；以下是需要替换占位值的访问控制示例，凭证仍由向导管理：

```text
WEIXIN_ACCOUNT_ID=向导返回的机器人账号ID
WEIXIN_DM_POLICY=allowlist
WEIXIN_ALLOWED_USERS=已核实的发送者ID
WEIXIN_GROUP_POLICY=disabled
```

机器人账号 ID 与允许发消息的用户 ID 是不同东西。群策略只在 iLink 真正投递对应群事件时才有作用，设置为 `open` 不会凭空获得群消息。群聊与主动发送的限制应检查实际账号类型和平台响应，不能简单归纳成所有版本都「支持」或「不支持」。

首次测试按三步走：

1. 在自己的私聊中发送一条普通问题，核对网关收到了哪个发送者 ID
2. 核对回复是否来自预期的 Hermes 配置，再测试一个只读工具
3. 检查未授权的发送者不会被处理；不要为了排错长期开放全部使用者

如果扫码成功但收不到消息，先看平台日志、访问策略、登录状态和是否存在第二个网关实例，而不是反复更换模型。执行命令或操作浏览器的能力来自 Agent 的工具与权限，不是微信协议本身提供的功能。

## 4. 保活机制

「保活」容易混淆三类事情：

- Gateway 进程退出后重新启动
- Windows 登录后启动 WSL 与相关服务
- 电脑休眠期间仍对外在线

前两类可以由服务管理器和启动机制处理。**普通后台脚本不能让一台已休眠或关机的电脑持续提供服务。** 服务能否恢复，取决于宿主机、网络、WSL 和上游服务是否就绪；实际恢复时间需要测量。

### 4.1 优先使用受支持的服务管理

当前 Hermes 官方网关提供 Linux systemd 服务安装、启动、状态与日志入口。先在 WSL 检查 PID 1：

```bash
ps -p 1 -o comm=
```

如果不是 `systemd`，按 [Microsoft 的 systemd 指南](https://learn.microsoft.com/en-us/windows/wsl/systemd)检查 WSL 版本，并在现有 `/etc/wsl.conf` 中合并所需的 `[boot]` / `systemd=true` 配置，不要覆盖文件里其他设置。应用配置可能需要重启 WSL；`wsl --shutdown` 会停止运行中的发行版，应先保存其他工作。

确认 systemd 可用且前台测试通过后，再安装用户服务：

```bash
hermes gateway install
hermes gateway start
hermes gateway status
journalctl --user -u hermes-gateway -n 100 --no-pager
```

上述 unit 名针对默认配置；多安装目录或不同服务 scope 时，要以本机生成的服务名为准。安装前应停止同一账号的手工前台实例，避免多个网关争用。用户服务退出登录后的行为、是否需要 linger，以及系统级服务的区别，见[官方网关服务说明](https://hermes-agent.nousresearch.com/docs/user-guide/messaging#linux-systemd)。

让 systemd 管理 Gateway，并不自动等于 Windows 登录时会启动 WSL。宿主机的启动触发与 Linux 内的进程恢复，是两个不同问题。

### 第一层: Windows Startup 启动

原方案在 Windows Startup 目录放 VBS，隐藏窗口后启动 PowerShell。要注意：**用户启动目录通常在该用户登录后运行，不是机器通电就必然执行。** 隐藏窗口也只影响显示，不保证进程健康。

如果继续使用这条历史路线，至少应验证脚本路径带空格时的引用、登录用户是否正确、目标 WSL 发行版是否正确，以及脚本能否把失败写到日志。原文使用 `ExecutionPolicy Bypass` 不应成为不加判断的默认选项；按自己的脚本信任与管理策略处理。

对于新部署，优先只保留一个明确的登录入口。官方 [WSL 长期运行说明](https://hermes-agent.nousresearch.com/docs/user-guide/windows-wsl-quickstart#making-wsl-itself-start-on-windows-login)给出了使用 Task Scheduler 启动 WSL 常驻进程的方式。它解决的是启动与维持 WSL 使用状态，不会克服宿主机休眠或断网。

### 第二层: PowerShell 自循环守护脚本

旧方案每 5 分钟检查一次 WSL 侧 Gateway，并将等待拆成 60 秒小段写心跳。这个设计中的**可观察性与分层判断**仍值得参考；脚本则没有附全依赖函数，不能直接复制使用。

需要自建看门狗时，可以采用以下流程，并让它与服务管理器分工，避免两者同时直接启动 Gateway：

```text
取得只允许一个看门狗运行的锁
检查目标 WSL 是否可达，并限制检查耗时
调用服务管理器查询 Gateway 状态
  正常：记录成功检查时间
  退出：请求服务管理器启动一次，再确认结果
  状态不明或超时：记录故障，退避，不盲目再起一个进程
周期性更新看门狗心跳；退出时释放锁
```

心跳只证明「写心跳的代码执行了」。它不证明网关能收消息，甚至不证明主检查流程没有卡住。最好分别记录最近一次循环时间、最近一次成功健康检查时间、最近一次重启及原因。

旧代码里还有几个值得保留为反例的细节：

- `pgrep -f` 可能匹配 shell 包装或其他命令；只检查 `/proc/PID/exe` 是 Python，也不能证明它就是正确实例
- PID 可能被复用，PID 文件必须结合进程身份和生命周期检查
- `D` 是不可中断睡眠状态，不等于进程已退出；把它跳过后立即启动替代进程，可能制造双实例。[Linux 进程状态说明](https://man7.org/linux/man-pages/man5/proc_pid_stat.5.html)
- `flock` 锁与打开的文件描述关联，写一个 PID 不会自动防止锁文件被替换。锁目录权限、稳定的锁对象和持锁范围都要设计，不能声称「flock + PID 文件」已解决所有竞态。[flock 手册](https://man7.org/linux/man-pages/man2/flock.2.html)
- 原文将「已恢复」编码为退出码 `1`，而很多调度器把非零视作失败。应明确约定输出状态与退出码，成功恢复也要让上层正确识别

另外，不建议在找不到指定可执行文件时自动改用 PATH 里的 `hermes`，再改用另一个 Python 模块。它可能启动不同版本、不同环境的程序。更安全的做法是记录明确的安装路径，失败时报告依赖损坏，再修复该安装。

### 第三层: Windows 计划任务(兜底)

计划任务可以在登录时启动入口，也可以定期执行健康检查。两者的 trigger 不一样：原文第一个 `Register-ScheduledTask` 示例没有传入触发器，不能据此声称已经配置了开机自启。[Microsoft 的触发器文档](https://learn.microsoft.com/en-us/powershell/module/scheduledtasks/new-scheduledtasktrigger?view=windowsserver2025-ps)明确区分 `-AtLogOn`、`-AtStartup` 和定时触发。

如果自建兜底流程，需要检查：

1. 它运行的 Windows 用户是否拥有目标 WSL 发行版与所需环境
2. 主看门狗与兜底进程是否使用同一锁、同一实例范围
3. 是否在持锁期间完成「检查 → 决定 → 启动」，而不是先检查、放锁后再启动
4. WSL 卡住时能否超时退出，是否有重试退避和频率限制
5. 故意停止服务维护时，是否会被兜底误认为故障而重新拉起

旧方案的「7 分钟心跳过期」只是结合当时检查间隔选的阈值。它需要大于预期检查耗时并考虑调度延迟，不是通用判断标准；Mutex 与心跳都正常，也仍需要独立验证网关功能。

### 全链路故障恢复

原文提到「63 项压力测试」和「5 分钟内恢复」，但没有附可复验的脚本、日志、版本及测试条件，因此这里不把它们当成已验证的保证。可以参考以下测试覆盖面：

| 故障或操作 | 要观察的结果 |
| --- | --- |
| 在测试环境终止 Gateway 进程 | 服务是否重启；是否只有一个实例；消息是否能再次往返 |
| 终止自建看门狗 | 兜底是否生效；不会与服务管理器重复启动 |
| Windows 退出登录或重启后再登录 | 启动触发是否执行；WSL 用户与环境是否正确 |
| 休眠后唤醒 | 网络、登录状态、Gateway 与浏览器分别何时恢复 |
| 暂时断网或上游服务不可用 | 是否有超时和退避；避免不断重启正常进程 |
| 登录凭证过期 | 是否明确要求重新登录，而不是无限重试 |
| 依赖路径失效 | 能否定位缺失依赖，而不是悄悄启动另一个版本 |
| 人工停止服务维护 | 服务是否保持停止，维护模式是否有效 |

为每项记录注入故障时间、检测时间、服务恢复时间、首次成功消息时间和相关日志。**进程出现了**与**用户真的能用了**，应分开计时。恢复测试尽量在没有进行中的写操作时做，避免人为中断造成文件或任务状态不一致。

## 5. Camoufox 反爬浏览器

原案例还引入了 Camoufox 相关浏览器服务。这里需要区分两个名称：Camoufox 是基于 Firefox 的浏览器项目；[camofox-browser](https://github.com/jo-inc/camofox-browser) 是面向 Agent 提供 REST 接口的包装服务。只安装 Python 的 `camoufox` 包，不等于已经启动后者的控制 API。

浏览器后端可以影响网站兼容性，但不能保证 Google、百度或某个站点永远可访问。IP、账号、网站策略和访问频率都会改变结果；遇到验证码、拒绝访问或登录要求，应使用网站允许的方式处理，不能把「反检测」当成无限制访问的许可。

### 安装过程

使用自己选定版本的 [camofox-browser 官方安装说明](https://github.com/jo-inc/camofox-browser#quick-start)，记录仓库 commit 和依赖。原稿中「某个命令存在已知 Node.js/Playwright 兼容性问题」没有给 issue 或版本，这里不把它当成当前结论。

采用仓库安装路线时，基本步骤仍是克隆、安装依赖、启动服务；实际部署应固定已验证版本，并检查安装脚本会下载哪些浏览器文件。不要把多个同名或相似项目的安装方法拼在一起。

### 启动服务

如果已按该项目说明完成安装，当前服务支持明确的端口和监听地址。只供同一个 WSL 环境内的 Hermes 使用时，可以先限制为回环地址：

```bash
# 在已安装并核对版本的 camofox-browser 目录内
CAMOFOX_BIND_HOST=127.0.0.1 CAMOFOX_PORT=9377 npm start
```

另开一个同环境终端检查：

```bash
curl --fail --max-time 5 http://127.0.0.1:9377/health
```

健康接口返回成功后，还要实际打开一个允许访问的测试页面并读取内容。若浏览器与 Hermes 分别在 Windows、WSL 或容器内，`127.0.0.1` 指向各自环境，不能直接照搬地址。不要为了方便连接就把拥有浏览器控制能力的 API 无鉴权暴露到公网。

### 集成到 Hermes

当前 Hermes 文档中，`CAMOFOX_URL` 只表示服务地址。在已经选择其他后端的配置中，单独设置它不会自动完成切换：

```text
CAMOFOX_URL=http://127.0.0.1:9377
```

再通过 `hermes tools` 的 Browser Automation 选项明确选择 Camofox。当前文档说明该选择写入 `browser.cloud_provider: camofox`；旧稿的 `browser.engine: auto` 不应继续当作通用配置。

会话是否保留登录状态是另一件事：`browser.camofox.managed_persistence` 控制 Hermes 侧的持久身份使用，而服务端也必须支持相应 profile 持久化。确实需要跨任务保留登录时，再有意识地启用并验证；凭证与 cookie 应当按账号访问资料管理。配置和行为可对照[Hermes 浏览器文档](https://hermes-agent.nousresearch.com/docs/user-guide/features/browser/)。

### 效果对比

旧稿把三家网站列为「默认浏览器失败，Camoufox 成功」，但没有保留测试时间、访问条件和日志。它最多是一次环境里的观察，不能支持「默认浏览器在中文互联网上基本不可用」的结论。

更有用的验证顺序是：

1. 服务健康接口可达
2. Hermes 实际选中了预期浏览器后端
3. 测试页面能打开并读取，点击等动作结果可核对
4. 结束任务后会话按预期关闭或保留
5. 重启后再次验证，确认登录、profile 和服务恢复的行为

原文提到的 `camofoxctl.sh` 是本地管理脚本，文章没有附其实现，因此不把 `start/status/stop` 当成安装后自动存在的官方命令。自己的脚本可以保留这类统一入口，但要让状态查询真正检查所管理的服务。

<a id="6-灵魂画手-架构总览"></a>

## 6. 架构总览

一条请求在这套部署中会经过五个交接点：

1. 手机把消息交给平台，平台再交给 Weixin 适配器
2. Gateway 检查入口配置，把请求交给 Agent
3. Agent 调用模型，应用核对工具权限后执行具体操作
4. 浏览器任务通过本机控制 API 交给 Camofox 服务，后者访问目标网站
5. 真实工具结果返回 Agent，最终回复再经 Gateway 发回手机

运行保障则分两层：Windows 侧负责何时启动 WSL，Linux 服务管理器负责已经运行的 WSL 内的 Gateway。浏览器服务有自己的启动和健康检查。自写看门狗只有在这些机制仍缺少具体恢复能力时才值得补充。

因此，不能仅凭「手机没回复」就认定「模型坏了」。先定位消息有没有进网关、任务有没有开始、模型有没有响应、工具有没有完成，最后检查回复是否成功送达。

## 7. 总结

从这个案例中可以总结出几条可复用的部署习惯：

- 先在前台验证单条链路，再安装后台服务与登录入口
- 固定版本、用户、路径和配置来源，避免同一账号启动多个实例
- 区分进程存活、服务健康与消息真正往返
- 保留超时、退避、单实例锁与维护停止的设计
- 只给需要的用户和工具开放权限，妥善处理账号与浏览器状态
- 对恢复时间给出测试条件和日志，不把一次成功扩展成永久保证

如果要让服务在个人电脑睡眠时也持续可用，就需要另行选择保持运行的宿主环境，而不是继续叠加 watchdog。应用本身如何设计工具与停止条件，回到[应用开发篇的 Agent 部分](/posts/2024-04-24-lang-chain-series/#4-agent)；本文负责的是把已经设计好的应用可靠地运行起来。

原版文章曾用交互彩蛋说明 AI 参与了写作。本次整理保留这一背景，但不再沿用固定字数、图片大小等已经不适用的展示。AI 生成了命令，不等于命令已经在你的环境验证；部署仍应以实际日志与验收结果为准。
