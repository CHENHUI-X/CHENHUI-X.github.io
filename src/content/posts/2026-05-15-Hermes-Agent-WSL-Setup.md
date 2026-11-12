---
title: 在 WSL 上部署 Hermes Agent：接入微信、配置后台服务和浏览器
published: 2026-05-15
description: 记录 Windows 11 与 WSL2 上的 Hermes Agent 部署，介绍微信接入、systemd 服务管理、故障检查和 Camofox 浏览器配置，并区分官方功能与自建脚本。
category: Technical
tags:
- llm
- agent
- devops
- tool
draft: false
---

<span id="0-前言"></span>

<span id="0-部署环境与使用场景"></span>

## 0. 让手机发来的消息得到回复，需要配置什么

要让手机发来的消息随时得到回复，仅仅让 Agent 在终端里正常回答还不够：还要配置消息网关、维持登录状态、管理后台服务，并处理宿主机休眠和浏览器进程。

下面以 **Windows 11 + WSL2（Ubuntu）** 上的一次 Hermes Agent 部署为例，说明怎样把这些部分连起来，以及没有回复时该查哪里。Prompt、工具调用与 Agent 的概念见[LangChain 应用开发篇](/posts/2024-04-24-lang-chain-series/)，这里不再从头解释。

:::note
下文命令和配置以 2026-10-07 的官方资料为准。复现时请记录 Hermes tag/commit 和完整依赖版本；自写看门狗部分只说明流程，没有提供完整脚本。配置完成后，可以按第 4 节测试故障恢复，并在日志中记录恢复花了多久。
:::

本文仍选择 WSL 路线，并不表示它是唯一的 Windows 安装方式。[Hermes 官方仓库](https://github.com/NousResearch/hermes-agent)目前也提供原生 Windows 安装入口。已经稳定使用某种方式时，可以先检查其服务管理能力，无需仅为复现本文而切换环境。

## 1. Hermes Agent 是什么

[Hermes Agent](https://github.com/NousResearch/hermes-agent) 是 Nous Research 的开源 Agent 项目，提供终端交互、消息平台接入、工具与记忆等能力。本案例包含三个部分：

1. **Gateway（网关）**：连接消息平台，接收请求并发送回复
2. **Agent 运行逻辑**：根据任务调用模型和允许使用的工具
3. **工具及其依赖**：终端、文件、浏览器或其他服务，可能分别失败

「微信能收到消息」只说明消息接收这一部分正常。模型凭证失效时仍可能收得到消息；浏览器打不开时，普通聊天也可能正常。分别检查消息、模型和工具的状态，才能区分这些故障；仅查看 Python 进程是否存在还不够。

同时，能从聊天入口调用终端意味着它有实际操作能力。先限定使用者、工具和工作目录，再考虑常驻运行。具体权限与隔离选项参见[Hermes 安全文档](https://hermes-agent.nousresearch.com/docs/user-guide/security)。

## 2. 在 WSL 上安装

<span id="21-先记环境再装软件"></span>

<span id="21-wsl-环境检查与版本记录"></span>

### 2.1 检查 WSL，并记下安装版本

![Microsoft 官方 WSL 架构原图：上方 Windows 进程通过 WSL 服务连接下方 Linux 虚拟机和发行版](/images/llm-series/sourced/microsoft-wsl-architecture.png)

*来源：Microsoft，[WSL 开源组件总览](https://learn.microsoft.com/en-us/windows/wsl/opensource)；[官方原图](https://learn.microsoft.com/en-us/windows/wsl/media/wsl-architecture.png)，原样保留。文档按 [CC BY 4.0](https://github.com/MicrosoftDocs/WSL/blob/main/LICENSE)许可。此图说明 WSL 的运行环境，不是 Hermes 内部架构。*

读这张图先找三个位置，不必记住全部进程名：

- **上方 Windows 框**：`wsl.exe` 等入口通过 `wslservice.exe` 管理 WSL。这对应本文在 PowerShell 中查看、启动和停止发行版的操作。
- **下方 Linux 框里的 Linux Distribution**：Ubuntu 属于这一层。本案例的 Hermes、Gateway 和 Linux 工具运行在这里；Windows 终端窗口只是访问它的入口。
- **两个大框之间的 hvsocket 连线**：提示两侧需要通信。Linux 服务能重启自己的进程，不意味着它能保证 Windows 宿主机一直醒着。

图没有展开 systemd，也没有画出 Hermes 服务。第 4 节的 systemd 配置管理的是发行版内的服务生命周期；宿主机休眠、WSL 停止与单个 Gateway 退出，应分别测试。进程级启动细节可对照 [Microsoft 的 WSL2 启动过程](https://github.com/microsoft/WSL/blob/master/doc/docs/technical-documentation/boot-process.md)。

WSL2 是 Windows 运行 Linux 环境的一种方式；Ubuntu 是其中安装的一套 Linux 发行版。本文会在两个终端之间切换：PowerShell 管 Windows 侧的 WSL 启停，Ubuntu 终端运行 Linux 命令和 Hermes。代码块标成 `powershell` 的在前者运行，`bash` 的在后者运行。运行前应确认当前使用的是哪一种 shell。

在 Windows PowerShell 中检查：

```powershell
wsl --version
wsl --list --verbose
```

确认目标发行版使用 WSL2，并记下发行版名称。

看 `wsl --list --verbose` 的 `VERSION` 列，目标那一行应为 `2`；`STATE` 显示它当前是运行还是停止。若尚未安装发行版，先按 [Microsoft 安装指南](https://learn.microsoft.com/en-us/windows/wsl/install)完成 WSL 安装，再继续本文。在 PowerShell 输入 `wsl -d Ubuntu` 会进入该发行版，进入后可运行 `whoami` 和 `pwd` 确认 Linux 用户与当前目录；用 `exit` 返回 PowerShell。

后文的 `Ubuntu` 只是本案例名字，如果实际是 `Ubuntu-24.04`，调用 `wsl -d` 时必须用自己的名称。[Microsoft 的 WSL 命令说明](https://learn.microsoft.com/en-us/windows/wsl/basic-commands)列出了这些参数。

部署记录建议包含：Windows 与 WSL 版本、Linux 发行版、Hermes 安装方式及版本或 commit、模型供应商、浏览器后端版本。只记「安装了最新版」，以后很难判断是配置变化还是升级造成的故障。这份记录不应包含 API key、网关 token 或浏览器 cookie。

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

本案例选用 OpenRouter。供应商、可用模型和密钥配置以安装版本的向导为准；不要把凭证直接写进可分享的配置示例。

长期运行时，需要弄清这些限制：任务轮次上限控制一次任务可以推进多久；命令超时约束单条工具调用；增大这两个上限并不一定提高可靠性。可以先运行便于检查的小任务，再按需要调整。本案例曾使用 `100` 轮和 `600s`，这些取值需要按任务调整，并非通用建议或当前默认值。

<span id="23-先在前台跑通"></span>

<span id="23-前台聊天与网关测试"></span>

### 2.3 先在终端测试聊天和网关

「前台」表示进程占用当前终端、日志直接显示在窗口；「后台服务」则由服务管理器启动和看管，不依赖你一直保留交互窗口。先在 Ubuntu 终端运行 `hermes` 测试聊天，完成第 3 节的平台配置后，再运行 `hermes gateway` 测试消息入口。Gateway 前台运行时，另用手机发送测试消息；结束测试可在该终端按 Ctrl+C。命令入口见[官方 Gateway 命令表](https://hermes-agent.nousresearch.com/docs/user-guide/messaging#gateway-commands)。

先在前台测试，可以直接看到报错。此时先不要同时启用手工启动、systemd 和自写 watchdog，否则很难知道正在使用哪个进程和哪份配置。

配置和项目文件的路径也要分清：WSL 的 `/home/...`、Windows 的 `C:\...` 和 WSL 看到的 `/mnt/c/...` 不是可直接互换的路径。脚本报 `bad interpreter` 且带 `^M` 时，先检查是否混入 CRLF 换行，排除脚本格式问题后再查其他原因。

## 3. 微信网关配置

本案例使用个人微信的 Weixin 适配器，与企业微信 WeCom 是不同接入方式。按[官方 Weixin 文档](https://hermes-agent.nousresearch.com/docs/user-guide/messaging/weixin)，通过下面的向导扫码并在手机确认：

```bash
hermes gateway setup
```

扫码连接的是 iLink bot 身份，不等于获得一个普通微信账号的所有自动化能力。配置完成后，先通过私聊确认能收到并回复消息，再检查群聊是否可用。

当前文档中的账号变量是 `WEIXIN_ACCOUNT_ID`；迁移使用 `WEIXIN_ACCOUNT` 的旧配置时，应按安装版本重新设置。个人部署还应明确谁能触发 Agent；以下是需要替换占位值的访问控制示例，凭证仍由向导管理：

```text
WEIXIN_ACCOUNT_ID=向导返回的机器人账号ID
WEIXIN_DM_POLICY=allowlist
WEIXIN_ALLOWED_USERS=已核实的发送者ID
WEIXIN_GROUP_POLICY=disabled
```

机器人账号 ID 与允许发消息的用户 ID 是不同东西。

上面的 `text` 块是在说明配置键和值，其中的中文占位符需要替换，不能作为命令直接执行。由向导或安装版本支持的配置入口写入真实值；`allowlist` 意味着只有名单中的发送者可以触发任务，`disabled` 意味着关闭该入口。如果在终端临时设置环境变量，它也未必会传给已经运行的后台服务；改完配置应按该服务的说明重新加载，并重新核对访问限制。

群策略只在 iLink 真正投递对应群事件时才有作用，设置为 `open` 不会凭空获得群消息。群聊与主动发送的限制应检查实际账号类型和平台响应，不能简单归纳成所有版本都「支持」或「不支持」。

首次测试按三步走：

1. 在自己的私聊中发送一条普通问题，核对网关收到了哪个发送者 ID
2. 核对回复是否来自预期的 Hermes 配置，再测试一个只读工具
3. 检查未授权的发送者不会被处理；不要为了排错长期开放全部使用者

如果扫码成功但收不到消息，先看平台日志、访问策略、登录状态和是否存在第二个网关实例，再判断是否需要检查模型配置。执行命令或操作浏览器的能力来自 Agent 的工具与权限，不是微信协议本身提供的功能。

<span id="4-保活机制"></span>

<span id="4-网关服务管理与故障恢复"></span>

## 4. 让网关在后台运行，并在出错后恢复

想让服务一直可用，先分清下面三件事：

- Gateway 进程退出后重新启动
- Windows 登录后启动 WSL 与相关服务
- 电脑休眠期间仍对外在线

前两类可以由服务管理器和启动机制处理。**普通后台脚本不能让一台已休眠或关机的电脑持续提供服务。** 服务能否恢复，取决于宿主机、网络、WSL 和上游服务是否就绪；实际恢复时间需要测量。

<span id="41-优先使用受支持的服务管理"></span>

### 4.1 用 systemd 管理 Gateway

systemd 是 Linux 的服务管理器；unit 是它管理某个服务的配置单元，描述启动哪个程序以及如何管理它。PID 是运行中进程的编号，PID 1 是 Linux 启动后的首个进程。所以下面的命令只是在问“当前 WSL 由谁管理”，没有启动或停止任何服务。

当前 Hermes 官方网关提供 Linux systemd 服务安装、启动、状态与日志入口。先在 WSL 检查 PID 1：

```bash
ps -p 1 -o comm=
```

如果不是 `systemd`，按 [Microsoft 的 systemd 指南](https://learn.microsoft.com/en-us/windows/wsl/systemd)检查 WSL 版本，并在现有 `/etc/wsl.conf` 中合并所需的 `[boot]` / `systemd=true` 配置，不要覆盖文件里其他设置。应用配置可能需要重启 WSL；`wsl --shutdown` 会停止运行中的发行版，执行前先保存其他工作。

确认 systemd 可用且前台测试通过后，再安装用户服务：

```bash
hermes gateway install
hermes gateway start
hermes gateway status
journalctl --user -u hermes-gateway -n 100 --no-pager
```

四条命令分别完成：安装服务配置、启动服务、查询状态、查看最近 100 条日志。`journalctl` 是日志查询工具，`--user` 表示当前 Linux 用户的服务，`-u` 选择服务名，`--no-pager` 让结果直接输出。用安装 Hermes 的同一普通 Linux 用户操作，避免切换到 root 后查到另一套用户服务。

上述 unit 名针对默认配置；多安装目录或不同服务 scope 时，要以本机生成的服务名为准。安装前应停止同一账号的手工前台实例，避免多个网关争用。用户服务退出登录后的行为、是否需要 linger，以及系统级服务的区别，见[官方网关服务说明](https://hermes-agent.nousresearch.com/docs/user-guide/messaging#linux-systemd)。

让 systemd 管理 Gateway，并不自动等于 Windows 登录时会启动 WSL。宿主机的启动触发与 Linux 内的进程恢复，是两个不同问题。

<span id="第一层-windows-startup-启动"></span>

### Windows 登录时启动 WSL

一种启动方式是在 Windows Startup 目录放 VBS，隐藏窗口后启动 PowerShell。要注意：**用户启动目录通常在该用户登录后运行，不是机器通电就必然执行。** 隐藏窗口也只影响显示，不保证进程健康。

如果使用这条路线，至少应验证脚本路径带空格时的引用、登录用户是否正确、目标 WSL 发行版是否正确，以及脚本能否把失败写到日志。`ExecutionPolicy Bypass` 不应成为不加判断的默认选项；按自己的脚本信任与管理策略处理。

对于新部署，优先只保留一个明确的登录入口。官方 [WSL 长期运行说明](https://hermes-agent.nousresearch.com/docs/user-guide/windows-wsl-quickstart#making-wsl-itself-start-on-windows-login)给出了使用 Task Scheduler 启动 WSL 常驻进程的方式。这种方式用于在登录后启动 WSL，并让它持续运行；宿主机休眠或断网时，服务仍会受影响。

<span id="第二层-powershell-自循环守护脚本"></span>

### PowerShell 看门狗与心跳检查

watchdog（看门狗）是定期检查并尝试恢复服务的程序，心跳是它周期性写下的“我还在运行”的记录。退避则是在失败后拉长重试间隔，避免故障期间反复重启。

本案例的自建看门狗每 5 分钟检查一次 WSL 侧 Gateway，并将等待拆成 60 秒小段写心跳。这里分别记录进程检查结果和看门狗心跳，便于判断是哪一部分停止工作。完整实现还需处理锁、超时与恢复。

锁用于防止两个检查者同时判断“服务没运行”并各启动一个实例；检查与启动要在同一次持锁范围内完成。需要自建看门狗时，可以采用以下流程，并让它与服务管理器分工，避免两者同时启动 Gateway 实例：

```text
取得只允许一个看门狗运行的锁
检查目标 WSL 是否可达，并限制检查耗时
调用服务管理器查询 Gateway 状态
  正常：记录成功检查时间
  退出：请求服务管理器启动一次，再确认结果
  状态不明或超时：记录故障，退避，不盲目再起一个进程
周期性更新看门狗心跳；退出时释放锁
```

心跳只能说明「写心跳的代码执行了」，不能证明网关能收到消息，甚至不能证明主检查流程没有卡住。最好分别记录最近一次循环时间、最近一次成功健康检查时间，以及最近一次重启的时间和原因。

实现时还应注意以下细节：

- `pgrep -f` 可能匹配 shell 包装或其他命令；只检查 `/proc/PID/exe` 是 Python，也不能证明它就是正确实例
- PID 可能被复用，PID 文件必须结合进程身份和生命周期检查
- `D` 是不可中断睡眠状态，不等于进程已退出；把它跳过后立即启动替代进程，可能制造双实例。[Linux 进程状态说明](https://man7.org/linux/man-pages/man5/proc_pid_stat.5.html)
- `flock` 锁与打开的文件描述关联，写一个 PID 不会自动防止锁文件被替换。锁目录权限、稳定的锁对象和持锁范围都要设计，仅使用 flock 和 PID 文件仍可能存在竞态。[flock 手册](https://man7.org/linux/man-pages/man2/flock.2.html)
- 如果将「已恢复」编码为退出码 `1`，很多调度器会把它视作失败。应明确约定输出状态与退出码，成功恢复也要让上层正确识别

另外，不建议在找不到指定可执行文件时自动改用 PATH 里的 `hermes`，再改用另一个 Python 模块。它可能启动不同版本、不同环境的程序。更安全的做法是记录明确的安装路径，失败时报告依赖损坏，再修复该安装。

<span id="第三层-windows-计划任务兜底"></span>

<span id="windows-计划任务与备用健康检查"></span>

### 用 Windows 计划任务补充启动和检查

计划任务可以在登录时启动入口，也可以定期执行健康检查。两者的 trigger 不一样：使用 `Register-ScheduledTask` 时，需要明确配置触发器，仅注册任务不代表它会在开机时运行。[Microsoft 的触发器文档](https://learn.microsoft.com/en-us/powershell/module/scheduledtasks/new-scheduledtasktrigger?view=windowsserver2025-ps)明确区分 `-AtLogOn`、`-AtStartup` 和定时触发。

如果自行实现备用恢复流程，需要检查：

1. 它运行的 Windows 用户是否拥有目标 WSL 发行版与所需环境
2. 主看门狗与备用进程是否使用同一锁、同一实例范围
3. 是否在持锁期间完成「检查 → 决定 → 启动」，而不是先检查、放锁后再启动
4. WSL 卡住时能否超时退出，是否有重试退避和频率限制
5. 故意停止服务维护时，是否会被恢复程序误判为故障而重新启动

本案例的「7 分钟心跳过期」是结合上述检查间隔选的阈值。它需要大于预期检查耗时并考虑调度延迟，不是通用判断标准；Mutex 与心跳都正常，也仍需要独立验证网关功能。

<span id="全链路故障恢复"></span>

<span id="故障恢复测试与时间记录"></span>

### 模拟故障，记录恢复要多久

配置好以后，分别测试进程退出、电脑重启、休眠、断网和凭证失效，记录程序多久发现问题、多久恢复。可以按下表逐项检查：

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

每项测试都记下：什么时候制造故障、什么时候发现、什么时候恢复服务，以及第一条消息什么时候成功往返。保留对应日志。从进程启动到消息恢复往返之间仍可能有延迟，因此两者应分别计时。恢复测试尽量在没有进行中的写操作时做，避免人为中断造成文件或任务状态不一致。

<span id="5-camoufox-反爬浏览器"></span>

## 5. Camoufox 浏览器与 Camofox 控制服务

本案例还使用 Camoufox 相关浏览器服务。这里需要区分两个名称：Camoufox 是基于 Firefox 的浏览器项目；[camofox-browser](https://github.com/jo-inc/camofox-browser) 是面向 Agent 提供 REST 接口的包装服务。只安装 Python 的 `camoufox` 包，不等于已经启动后者的控制 API。

浏览器后端可以影响网站兼容性，但不能保证 Google、百度或某个站点永远可访问。IP、账号、网站策略和访问频率都会改变结果；遇到验证码、拒绝访问或登录要求，应使用网站允许的方式处理，不能把「反检测」当成无限制访问的许可。

### 安装过程

使用自己选定版本的 [camofox-browser 官方安装说明](https://github.com/jo-inc/camofox-browser#quick-start)，记录仓库 commit 和依赖。遇到 Node.js/Playwright 兼容性问题时，应结合具体报错、依赖版本和对应 issue 排查。

采用仓库安装路线时，基本步骤仍是克隆、安装依赖、启动服务；实际部署应固定已验证版本，并检查安装脚本会下载哪些浏览器文件。安装时应使用同一项目、同一版本的说明，避免混入名称相近项目的步骤。

### 启动服务

这里有两段连接：Hermes → 本机浏览器控制服务 → 目标网站。`127.0.0.1` 是本机回环地址，9377 是服务接收请求的端口；`/health` 是用于检查服务是否能响应的 HTTP 路径。健康接口成功只说明第一段中的服务能回复，还没有证明浏览器成功加载了目标网站。

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

再通过 `hermes tools` 的 Browser Automation 选项明确选择 Camofox。当前文档说明该选择写入 `browser.cloud_provider: camofox`；迁移含有 `browser.engine: auto` 的配置时，应重新检查后端选择。

会话是否保留登录状态是另一件事：`browser.camofox.managed_persistence` 控制 Hermes 侧的持久身份使用，而服务端也必须支持相应 profile 持久化。确实需要跨任务保留登录时，再启用持久化并验证效果；凭证与 cookie 都能用于访问账号，应妥善保管。配置和行为可对照[Hermes 浏览器文档](https://hermes-agent.nousresearch.com/docs/user-guide/features/browser/)。

<span id="效果对比"></span>

<span id="浏览器功能与重启恢复测试"></span>

### 检查网页操作和重启后的登录状态

比较默认浏览器与 Camoufox 时，应固定测试网站、账号、网络和操作步骤，并记录测试时间与日志。某个页面一次成功或失败，只能说明当时条件下的结果。

更有用的验证顺序是：

1. 服务健康接口可达
2. Hermes 实际选中了预期浏览器后端
3. 测试页面能打开并读取，点击等动作结果可核对
4. 结束任务后会话按预期关闭或保留
5. 重启后再次验证，确认登录、profile 和服务恢复的行为

`camofoxctl.sh` 是本案例的本地管理脚本，并非官方安装产物；`start/status/stop` 这类入口需要自行实现。自己的脚本可以保留这类统一入口，但要让状态查询真正检查所管理的服务。

<a id="6-灵魂画手-架构总览"></a>

<span id="6-架构总览"></span>

## 6. 一条手机消息怎样经过整套服务

一条请求会按下面的顺序处理：

1. 手机把消息交给平台，平台再交给 Weixin 适配器
2. Gateway 检查入口配置，把请求交给 Agent
3. Agent 调用模型，应用核对工具权限后执行具体操作
4. 浏览器任务通过本机控制 API 交给 Camofox 服务，后者访问目标网站
5. 真实工具结果返回 Agent，最终回复再经 Gateway 发回手机

后台运行分两边管理：Windows 侧负责何时启动 WSL，Linux 服务管理器负责已经运行的 WSL 内的 Gateway。浏览器服务有自己的启动和健康检查。自写看门狗只有在这些机制仍缺少具体恢复能力时才值得补充。

因此，手机没有收到回复时，问题可能发生在上述任一环节。先定位消息有没有进网关、任务有没有开始、模型有没有响应、工具有没有完成，最后检查回复是否成功送达。

## 7. 总结

部署时，可以按下面的顺序检查：

- 先在前台确认消息能正常往返，再配置后台服务和登录启动
- 固定版本、用户、路径和配置来源，避免同一账号启动多个实例
- 区分进程存活、服务健康与消息真正往返
- 设置超时和重试间隔，用锁防止重复启动，维护时允许服务保持停止
- 只给需要的用户和工具开放权限，妥善处理账号与浏览器状态
- 记录故障恢复的测试条件和日志，升级版本或更换环境后重新测试

如果要让服务在个人电脑睡眠时也持续可用，就需要另行选择保持运行的宿主环境，而不是继续叠加 watchdog。应用本身如何设计工具与停止条件，回到[应用开发篇的 Agent 部分](/posts/2024-04-24-lang-chain-series/#4-agent)；本文讨论的是已完成设计的应用怎样部署和恢复。
