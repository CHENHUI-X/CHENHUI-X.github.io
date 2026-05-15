================================================================
 Hermes Gateway 三层看门狗系统 - 压力测试/失效分析报告
================================================================

测试日期: 2026-05-15 19:17 - 19:35 CST
测试范围: Layer 1 (VBS启动) / Layer 2 (PS1循环守护) / Layer 3 (计划任务)
测试方法: 31项系统测试 + 深度手动攻击测试
最终状态: Gateway PID 283167 运行中 | PS1 Watchdog 运行中 | 全部已恢复

================================================================
一、CRITICAL FAILURE MODES（严重失效模式）
================================================================

[F1] pgrep 误匹配 —— 最高风险
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
文件: /root/hermes-watchdog.sh 第16行
代码: GW_PID=$(pgrep -f "hermes gateway run" 2>/dev/null | grep -v grep | head -1)

问题: pgrep -f 匹配命令行中包含 "hermes gateway run" 子串的任意进程。
      在生产环境（Hermes Agent CLI 活跃时），bash 包装进程会继承包含
      该文字的完整命令，导致：
      
      ☆ Gateway 真实死亡 → pgrep 仍能匹配到 shell 包装进程 → 
        watchdog.sh 返回 0 ("gateway OK") → PS1 没有日志输出 →
        用户微信断连且永不恢复（直到系统重启）
      
      ☆ 在纯 WSL 环境（无 Hermes Agent CLI）中风险较低，
        但仍可通过监控脚本或 ps/w 命令残留触发

演示（测试中出现）: 
  pgrep -f "hermes gateway run" 返回两个进程:
  PID 283167 - 真实 gateway (python3 ... hermes gateway run)
  PID 283467 - shell 包装进程 (bash -c ... hermes gateway run ...)
  
  kill -9 真实 gateway 后，pgrep 仍返回 1 个匹配 → 误报 OK

修复建议:
  1. 使用 pgrep -x ... 精确匹配（但 -f 和 -x 互斥）
  2. 增加 grep python 或 grep -v bash 过滤
  3. 检查 /proc/$PID/exe 是否为 python3 解释器
  4. 对比 PID 与 /root/.hermes/gateway.lock 记录的 PID

[F2] PS1 进程无声消亡 —— 高风险
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
测试发现: PS1 Watchdog 进程在测试过程中意外终止（原 PID 281197 消失）。
          日志停留在 19:22:47，后续无任何记录。
          系统因此失去主动守护能力长达 ~10 分钟。

原因分析（推测）:
  a) Windows 因资源限制终止了长时间运行的后台 PowerShell 进程
  b) PowerShell 作业（Job）堆积导致内存/句柄泄漏后被 OOM killer 干掉
  c) wsl.exe 调用超时导致 PowerShell Runspace 崩溃

当前防护:
  - Layer 3 计划任务 (HermesWatchdogCheck 每5分钟运行)
  - 但若 PS1 和 Gateway 同时死亡，需等最多 5 分钟才能触发

修复建议:
  1. PS1 增加心跳文件或 Windows Event Log 记录
  2. check-watchdog.ps1 不仅检查 Mutex，还检查进程表中
     是否有 powershell.exe 运行 watchdog 脚本
  3. 减少 Job 创建频率或复用 Runspace

[F3] TOCTOU 竞态 —— 中等风险
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
文件: check-watchdog.ps1 第21-25行
问题: check-watchdog 先检查 Mutex，若不持有则释放，然后运行 wsl.exe
      检查 Gateway。在 Mutex.ReleaseMutex() 和 wsl.exe 调用之间，
      存在时间窗口：
      
      1. PS1 可能在这几毫秒内崩溃 → check-watchdog 看不到
      2. 另一个 check-watchdog 实例可能同时触发
      
      此外，如果 wsl.exe 调用挂起（WSL 卡死），check-watchdog 的 
      1 分钟执行时间限制会终止任务，但不会重试。

修复建议:
  1. check-watchdog 在调用 wsl.exe 期间应持有 Mutex
  2. 或使用单独的 Mutex 来表示"正在检查中"

================================================================
二、MEDIUM ISSUES（中等级别问题）
================================================================

[M1] 失效的 fd 继承 —— 防御特性但需关注
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
watchdog.sh 使用 nohup 启动 Gateway，Gateway 进程继承 fd 200 
（指向 /var/run/hermes-watchdog.lock 的 flock 锁）。
当 watchdog.sh 退出后，Gateway 保持锁持有。

效果: 如果 Gateway 被 kill -9，所有 fd 立即释放 → 锁释放 →
      下次 watchdog.sh 运行时可获取锁并检测到 Gateway 死亡

副作用: 
  - Gateway 的 /proc/PID/fd/200 显示锁文件
  - 锁的生命周期 == Gateway 生命周期
  - 如果锁文件被人删除，Gateway 仍持有原 inode 的 fd
    （内核层面的引用），新创建的锁文件可以被另一个 watchdog.sh 获取

[M2] hermes 命令是 Shell 脚本
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
/usr/local/bin/hermes 内容:
  #!/usr/bin/env bash
  unset PYTHONPATH
  unset PYTHONHOME
  exec "/usr/local/lib/hermes-agent/venv/bin/hermes" "$@"

问题: 
  - 如果 venv/bin/hermes 被删除或 venv 损坏，shell 脚本无错误处理
  - 如果 $PATH 中有一个 homonymous "hermes" 命令优先级更高
  - watchdog.sh 的双尝试机制（command -v hermes 失败后尝试绝对路径）
    能缓解部分问题，但如果绝对路径也失败则返回 2

[M3] VBS 无错误处理
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
start-hermes-watchdog.vbs 不使用 On Error Resume Next，
但如果 PowerShell 执行失败（如 PS1 文件缺失、权限不足），
VBS 不会产生任何错误提示，静默失败。

恢复: 依赖 Layer 3 计划任务（HermesWatchdogStartup）兜底。

[M4] Zombie Gateway 检测失效
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
watchdog.sh 使用 kill -0 $GW_PID 检测 Gateway 是否存活。
但 kill -0 对僵尸进程也返回 0（因为僵尸进程的 PID 仍然存在）。
如果 Gateway 变成僵尸（Z 状态），watchdog 会认为它正常，
但僵尸进程无法处理任何网络请求。

风险评估: 需要特定条件（父进程不 wait()）才能产生僵尸，
Python 进程正常退出时会清理子进程。

[M5] 编码问题
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PS1 使用 Out-File -Encoding utf8 写入日志（带 BOM）。
当通过 WSL 9p 协议读写时：
  - cat 显示正常（Linux 终端忽略 BOM）
  - Windows Notepad 显示正常
  - 但若在 WSL 中用 head/tail 查看，中文日志可能乱码

这不是功能性问题，但影响故障排查体验。

================================================================
三、LOW ISSUES / 已正确处理的场景
================================================================

[L1] ✅ Mutex 防止多实例 - 测试通过
[L2] ✅ 30s 超时保护 - 测试通过（实际 30.01s 触发超时）
[L3] ✅ Test-WslAlive 预检 - 测试通过
[L4] ✅ flock 防止重复 watchdog.sh - 测试通过
[L5] ✅ 快速 kill 循环（5次连续 kill -9） - 测试通过，成功恢复
[L6] ✅ SIGTERM 处理 - Gateway 被终止，watchdog 可重启
[L7] ✅ Gateway 自身锁文件防重复启动 - 测试通过
[L8] ✅ PS1 Job 不泄漏 - 测试时 Job 数量为 0
[L9] ✅ 日志目录可写 - 测试通过
[L10] ✅ .env 文件存在 - 23253 字节
[L11] ✅ 计划任务已配置 - HermesWatchdogCheck (每5分钟) 和 
         HermesWatchdogStartup (登录时) 均处于 Ready/Enabled 状态
[L12] ✅ WSL 不可达处理 - wsl.exe 调用失败被 try/catch 捕获
[L13] ✅ Mutex 正确处理丢弃状态 - Windows 内核自动释放
[L14] ✅ 双错误路径 - watchdog.sh 先尝试 command -v hermes，
         失败后尝试绝对路径 /usr/local/bin/hermes
[L15] ✅ PS1 兜底 - exit=2 时调用 "hermes gateway run &"
         在 WSL 后台启动

================================================================
四、无法测试的场景（受限于安全约束）
================================================================

1. wsl --terminate Ubuntu ❌ 会终止我们自身的会话
2. 重启 Windows ❌ 无法恢复
3. 删除关键文件 ❌ 无法恢复
4. 填满磁盘 ❌ 破坏性操作
5. 网络断开 ❌ 无法控制 Windows 网络

================================================================
五、总结与建议
================================================================

总测试数: 31 + 深度手动测试
通过: 22 (71%)
失败: 2 (6%) - [调度任务配置检测为假阳性，实际任务配置正常]
警告: 7 (23%) - 部分属理论风险，部分需代码修复

CRITICAL 优先修复:
  1. pgrep 误匹配 → 建议增加进程类型过滤
  2. PS1 无声崩溃 → 建议添加心跳监控

MEDIUM 建议修复:
  3. TOCTOU 竞态 → 调整 check-watchdog 的 Mutex 持有策略
  4. Zombie 检测 → 考虑使用 /proc/PID/status 的 State 字段

系统总体评价:
  三层看门狗系统在正常场景下工作良好，能从 Gateway 进程死亡、
  WSL 短暂不可达、快速故障循环等场景中恢复。
  主要风险点在于进程检测的精确性（pgrep 误匹配）和
  PS1 守护进程自身的稳定性。

================================================================
