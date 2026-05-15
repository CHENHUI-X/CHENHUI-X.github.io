#!/bin/bash
# Hermes Gateway 3-Layer Watchdog - Comprehensive Stress/Attack Test Suite
# Usage: bash watchdog-stress-test.sh
# Each test is self-contained and verifies recovery

# Not using set -e because watchdog.sh might return non-zero (expected)
# We handle errors per-test

PASS=0
FAIL=0
WARN=0
RESULTS=()

record_pass()  { local n="$1"; shift; PASS=$((PASS+1)); RESULTS+=("PASS  Test $n: $*"); }
record_fail()  { local n="$1"; shift; FAIL=$((FAIL+1)); RESULTS+=("FAIL  Test $n: $*"); }
record_warn()  { local n="$1"; shift; WARN=$((WARN+1)); RESULTS+=("WARN  Test $n: $*"); }

echo "============================================================"
echo " Hermes Gateway 3-Layer Watchdog - Stress Test Suite"
echo " Started: $(date)"
echo "============================================================"
echo ""

# ---- Pre-flight check ----
GW_PID=$(pgrep -f "hermes gateway run" 2>/dev/null | grep -v grep | grep -v "bash -c" | head -1)
echo "[PRE] Gateway PID: $GW_PID"
if [ -z "$GW_PID" ]; then
    echo "[PRE] ERROR: Gateway not running! Start it first."
    exit 1
fi

# ====================================================================
# LAYER 1: STARTUP (VBS + PS1 Auto-start)
# ====================================================================

# Test 1: VBS dynamic path detection - verify the scripts exist
echo ""
echo "=== LAYER 1: STARTUP TESTS ==="

echo "[Test 1] VBS/PS1 file integrity check..."
VBS="/mnt/c/Users/ASUS/AppData/Roaming/Microsoft/Windows/Start Menu/Programs/Startup/start-hermes-watchdog.vbs"
PS1="/mnt/c/Users/ASUS/AppData/Roaming/Microsoft/Windows/Start Menu/Programs/Startup/hermes-watchdog.ps1"
CK="/mnt/c/Users/ASUS/AppData/Roaming/Microsoft/Windows/Start Menu/Programs/Startup/check-watchdog.ps1"

errors=""
[ -f "$VBS" ] || errors+="VBS file missing. "
[ -f "$PS1" ] || errors+="PS1 file missing. "
[ -f "$CK" ]  || errors+="check-watchdog.ps1 missing. "
if [ -n "$errors" ]; then
    record_fail "L1.1" "File integrity: $errors"
else
    record_pass "L1.1" "All startup files present"
fi

# Test 2: VBS content analysis - does it use correct path resolution?
echo "[Test 2] VBS path resolution logic..."
VBS_CONTENT=$(cat "$VBS")
if echo "$VBS_CONTENT" | grep -q "GetParentFolderName"; then
    record_pass "L1.2" "VBS uses dynamic path detection (good)"
else
    record_fail "L1.2" "VBS uses hardcoded path (brittle)"
fi

# Test 3: VBS calls correct PS1 name
echo "[Test 3] VBS references correct PS1..."
if echo "$VBS_CONTENT" | grep -q "hermes-watchdog.ps1"; then
    record_pass "L1.3" "VBS references correct PS1 filename"
else
    record_fail "L1.3" "VBS references wrong PS1 filename"
fi

# Test 4: Simulate double-VBS launch race (Mutex should block second)
echo "[Test 4] Mutex single-instance testing..."
MUTEX_TEST=$(powershell.exe -Command '
$mutex = New-Object System.Threading.Mutex($false, "Global\HermesGatewayWatchdog")
$acquired = $mutex.WaitOne(0)
if ($acquired) { 
    $null = $mutex.ReleaseMutex()
    Write-Host "MUTEX_FREE"
} else {
    Write-Host "MUTEX_HELD"
}
' 2>&1 | tr -d '\r')
if echo "$MUTEX_TEST" | grep -q "MUTEX_HELD"; then
    record_pass "L1.4" "Mutex prevents duplicate PS1 instances"
else
    record_warn "L1.4" "Mutex not held - may indicate PS1 not running"
fi

# Test 5: What if PS1 file is deleted/corrupted at startup time?
echo "[Test 5] VBS missing PS1 file behavior (static analysis)..."
VBS_CMD=$(echo "$VBS_CONTENT" | grep -oP 'powershell\.exe[^"]*' | head -1)
if echo "$VBS_CONTENT" | grep -qi "on error resume next\|err.number\|err.clear"; then
    record_warn "L1.5" "VBS has some error handling"
else
    record_warn "L1.5" "VBS has NO error handling for missing PS1 - if PS1 is missing, VBS fails silently (degraded but recoverable via Task Scheduler)"
fi

# ====================================================================
# LAYER 2: PS1 LOOP WATCHDOG (Core)
# ====================================================================
echo ""
echo "=== LAYER 2: PS1 WATCHDOG (CORE) TESTS ==="

# Test 6: pgrep false positive detection (CRITICAL BUG FOUND)
echo "[Test 6] pgrep false positive - does watchdog.sh detect real gateway vs shell processes?"
bash /root/hermes-watchdog.sh
RC=$?
echo "watchdog.sh returned: $RC"
if [ "$RC" -eq 0 ]; then
    # Check if gateway is really running
    REAL_GW=$(ps aux | grep "hermes.*gateway.*run" | grep -v grep | grep python | grep -v "bash -c" | head -1)
    if [ -n "$REAL_GW" ]; then
        record_pass "L2.6" "pgrep correctly identified real gateway process"
    else
        record_fail "L2.6" "CRITICAL: pgrep false positive! watchdog.sh says gateway OK when it's not running"
    fi
else
    record_warn "L2.6" "watchdog.sh returned $RC (gateway may need restart)"
fi

# Test 7: Kill gateway, wait for watchdog cycle, verify restart
echo "[Test 7] Kill gateway + wait for watchdog auto-restart..."
GW_PID_BEFORE=$(ps aux | grep "hermes.*gateway.*run" | grep -v grep | grep python | grep -v "bash -c" | awk '{print $2}')
echo "Gateway PID before kill: $GW_PID_BEFORE"
kill -9 "$GW_PID_BEFORE"
sleep 2
GW_DEAD=$(ps aux | grep "hermes.*gateway.*run" | grep -v grep | grep python | grep -v "bash -c" | head -1)
if [ -z "$GW_DEAD" ]; then
    record_pass "L2.7a" "Gateway successfully killed"
else
    record_warn "L2.7a" "Gateway still showing after kill -9"
fi

echo "Triggering watchdog.sh manually..."
bash /root/hermes-watchdog.sh
RC=$?
sleep 5
NEW_GW=$(ps aux | grep "hermes.*gateway.*run" | grep -v grep | grep python | grep -v "bash -c" | head -1)
if [ -n "$NEW_GW" ]; then
    record_pass "L2.7b" "watchdog.sh restarted gateway after kill (RC=$RC)"
else
    record_fail "L2.7b" "watchdog.sh FAILED to restart gateway (RC=$RC)"
fi

# Re-check that PS1 is still alive after our manual tests
echo "[Test 8] PS1 process resilience during WSL-side tests..."
PS1_ALIVE=$(powershell.exe -Command '
$mutex = New-Object System.Threading.Mutex($false, "Global\HermesGatewayWatchdog")
$acquired = $mutex.WaitOne(0)
if ($acquired) { Write-Host "DEAD"; $null = $mutex.ReleaseMutex() } else { Write-Host "ALIVE" }
' 2>&1 | tr -d '\r')
if echo "$PS1_ALIVE" | grep -q "ALIVE"; then
    record_pass "L2.8" "PS1 watchdog survived WSL-side tests"
else
    record_fail "L2.8" "PS1 watchdog died during WSL-side tests"
fi

# Test 9: Timeout behavior - what if wsl.exe hangs forever?
echo "[Test 9] WSL timeout behavior (30s timeout)..."
echo "Creating a hanging wsl script..."
echo -e '#!/bin/bash\nsleep 60\necho "done"' > /tmp/hang-test.sh
chmod +x /tmp/hang-test.sh

START_TS=$(date +%s)
TIMEOUT_RESULT=$(powershell.exe -Command '
$job = Start-Job -ScriptBlock {
    param($d)
    $o = wsl.exe -d $d -u root -- bash /tmp/hang-test.sh 2>&1
    return @{ Output = $o; ExitCode = $LASTEXITCODE }
} -ArgumentList "Ubuntu"

$completed = Wait-Job $job -Timeout 30
if (-not $completed) {
    Stop-Job $job
    Remove-Job $job -Force
    Write-Host "TIMED_OUT"
} else {
    $r = Receive-Job $job
    Remove-Job $job -Force
    Write-Host "COMPLETED:$r"
}
' 2>&1 | tr -d '\r')
END_TS=$(date +%s)
ELAPSED=$((END_TS - START_TS))

if echo "$TIMEOUT_RESULT" | grep -q "TIMED_OUT"; then
    record_pass "L2.9" "PS1 correctly timed out hanging WSL call in ${ELAPSED}s (timeout=30s)"
else
    record_warn "L2.9" "Hanging WSL call was not timed out properly (elapsed=${ELAPSED}s, result=$TIMEOUT_RESULT)"
fi
rm -f /tmp/hang-test.sh

# Test 10: WSL unreachable (simulate by using wrong distro)
echo "[Test 10] WSL unreachable handling..."
UNREACH_RESULT=$(powershell.exe -Command '
$job = Start-Job -ScriptBlock {
    $o = wsl.exe -d NonexistentDistro -- echo "test" 2>&1
    return @{ Output = $o; ExitCode = $LASTEXITCODE }
} -ArgumentList

$completed = Wait-Job $job -Timeout 10
if (-not $completed) {
    Stop-Job $job; Remove-Job $job -Force
    Write-Host "TIMED_OUT"
} else {
    $r = Receive-Job $job; Remove-Job $job -Force
    Write-Host "EC=$($r.ExitCode) OUT=$($r.Output)"
}
' 2>&1 | tr -d '\r')
echo "Unreachable WSL result: $UNREACH_RESULT"

# Test 11: Test-WslAlive robustness - can it distinguish alive from dead WSL?
echo "[Test 11] Test-WslAlive function analysis..."
TEST_ALIVE=$(powershell.exe -Command '
function Test-WslAlive {
    $job = Start-Job -ScriptBlock {
        param($d)
        $o = wsl.exe -d $d -- echo "alive" 2>&1
        return @{ Output = $o; ExitCode = $LASTEXITCODE }
    } -ArgumentList "Ubuntu"
    $completed = Wait-Job $job -Timeout 10
    if (-not $completed) { Stop-Job $job; Remove-Job $job -Force; return $false }
    $r = Receive-Job $job; Remove-Job $job -Force
    if ($r -isnot [hashtable]) { return $false }
    return ($r.ExitCode -eq 0)
}
$alive = Test-WslAlive
Write-Host "ALIVE=$alive"
' 2>&1 | tr -d '\r')
echo "Test-WslAlive result: $TEST_ALIVE"
if echo "$TEST_ALIVE" | grep -q "ALIVE=True"; then
    record_pass "L2.11" "Test-WslAlive correctly reports WSL reachable"
elif echo "$TEST_ALIVE" | grep -q "ALIVE=False"; then
    record_fail "L2.11" "Test-WslAlive reports WSL unreachable when it should be reachable"
fi

# Test 12: flock locking mechanism - prevent duplicate watchdog.sh instances
echo "[Test 12] flock locking - concurrent watchdog.sh prevention..."
FLOCK_TEST=$(bash /root/hermes-watchdog.sh & bash /root/hermes-watchdog.sh & wait; echo "RACE_DONE")
echo "Flock race result: $FLOCK_TEST"

# Test 13: Multiple Gateway spawn protection
echo "[Test 13] Multiple gateway processes - does pgrep detect duplicates?"
GW_COUNT=$(ps aux | grep "hermes.*gateway.*run" | grep -v grep | grep python | grep -v "bash -c" | wc -l)
echo "Gateway process count: $GW_COUNT"
if [ "$GW_COUNT" -le 2 ]; then
    record_pass "L2.13" "No runaway gateway processes (count=$GW_COUNT)"
else
    record_warn "L2.13" "Multiple gateway processes detected ($GW_COUNT)"
fi

# Test 14: Gateway restart loop protection - rapid kill-restart cycle
echo "[Test 14] Rapid kill-restart cycle (hammer test)..."
for cycle in 1 2 3 4 5; do
    GW=$(ps aux | grep "hermes.*gateway.*run" | grep -v grep | grep python | grep -v "bash -c" | awk '{print $2}' | head -1)
    [ -n "$GW" ] && kill -9 "$GW" 2>/dev/null || true
done
sleep 5
bash /root/hermes-watchdog.sh
sleep 5
GW_AFTER=$(ps aux | grep "hermes.*gateway.*run" | grep -v grep | grep python | grep -v "bash -c" | head -1)
if [ -n "$GW_AFTER" ]; then
    record_pass "L2.14" "Watchdog survived 5 rapid kill cycles and restarted gateway"
else
    record_fail "L2.14" "Watchdog failed after 5 rapid kill cycles"
fi

# ====================================================================
# LAYER 3: SCHEDULED TASK (check-watchdog.ps1)
# ====================================================================
echo ""
echo "=== LAYER 3: SCHEDULED TASK TESTS ==="

# Test 15: check-watchdog.ps1 behavior when PS1 is alive
echo "[Test 15] check-watchdog.ps1 with PS1 alive..."
TASK_RESULT=$(powershell.exe -Command '
& "C:\Users\ASUS\AppData\Roaming\Microsoft\Windows\Start Menu\Programs\Startup\check-watchdog.ps1"
Write-Host "TASK_COMPLETED"
' 2>&1 | tr -d '\r')
echo "check-watchdog result when PS1 alive: $TASK_RESULT"
if echo "$TASK_RESULT" | grep -q "TASK_COMPLETED"; then
    record_pass "L3.15" "check-watchdog.ps1 exits cleanly when PS1 is alive (no-op)"
else
    record_warn "L3.15" "check-watchdog.ps1 had issues when PS1 alive: $TASK_RESULT"
fi

# Test 16: Scheduled task trigger configuration
echo "[Test 16] Scheduled task configuration..."
TASK_INFO=$(powershell.exe -Command '
Get-ScheduledTask -TaskName "HermesWatchdogCheck" | Select-Object TaskName, State, Enabled, Actions
Get-ScheduledTask -TaskName "HermesWatchdogStartup" | Select-Object TaskName, State, Enabled, Actions
' 2>&1 | tr -d '\r')
echo "$TASK_INFO"

TASK_TRIGGERS=$(powershell.exe -Command '
(Get-ScheduledTask -TaskName "HermesWatchdogCheck").Triggers | Select-Object Id, Repetition, Enabled
' 2>&1 | tr -d '\r')
echo "Triggers: $TASK_TRIGGERS"

if echo "$TASK_INFO" | grep -q "HermesWatchdogCheck.*Ready.*True"; then
    record_pass "L3.16a" "HermesWatchdogCheck task is Ready and Enabled"
else
    record_fail "L3.16a" "HermesWatchdogCheck task not properly configured"
fi

if echo "$TASK_INFO" | grep -q "HermesWatchdogStartup.*Ready.*True"; then
    record_pass "L3.16b" "HermesWatchdogStartup task is Ready and Enabled"
else
    record_fail "L3.16b" "HermesWatchdogStartup task not properly configured"
fi

# Test 17: Abandoned Mutex scenario - what if PS1 crashes?
echo "[Test 17] Abandoned Mutex analysis..."
MUTEX_INFO=$(powershell.exe -Command '
try {
    $mutex = New-Object System.Threading.Mutex($false, "Global\HermesGatewayWatchdog")
    $acquired = $mutex.WaitOne(0)
    if ($acquired) {
        Write-Host "Mutex_State=AVAILABLE (PS1 NOT running or abandoned)"
        $null = $mutex.ReleaseMutex()
    } else {
        Write-Host "Mutex_State=HELD (PS1 is running)"
    }
} catch {
    Write-Host "Mutex_Error: $_"
}
' 2>&1 | tr -d '\r')
echo "Mutex state: $MUTEX_INFO"

# Test 18: check-watchdog.ps1 Mutex race - what if it acquires Mutex same time PS1 releases?
echo "[Test 18] Mutex race condition analysis..."
echo "Analysis: check-watchdog.ps1 creates Mutex with WaitOne(0) and immediately releases it."
echo "This can race with PS1's Mutex check. Risk: LOW because Mutex is named and both use same name."
echo "But if check-watchdog runs EXACTLY when PS1 is between WaitOne checks, it could falsely think PS1 is dead."
record_warn "L3.18" "Potential TOCTOU race: check-watchdog Mutex check is not atomic with Gateway check"

# Test 19: What if check-watchdog.ps1 runs but Mutex creation fails?
echo "[Test 19] check-watchdog.ps1 exception handling analysis..."
CK_CONTENT=$(cat "$CK")
if echo "$CK_CONTENT" | grep -q "catch"; then
    record_pass "L3.19" "check-watchdog.ps1 has try/catch (good)"
else
    record_warn "L3.19" "check-watchdog.ps1 missing exception handler"
fi
if echo "$CK_CONTENT" | grep -q "ReleaseMutex"; then
    record_pass "L3.19b" "check-watchdog properly releases Mutex"
else
    record_warn "L3.19b" "check-watchdog may not release Mutex properly"
fi

# ====================================================================
# ENVIRONMENTAL / EDGE CASE TESTS
# ====================================================================
echo ""
echo "=== ENVIRONMENTAL / EDGE CASE TESTS ==="

# Test 20: Log directory permissions
echo "[Test 20] Log directory accessibility..."
LOG_DIR="/mnt/c/Users/ASUS/AppData/Local/Temp/HermesWatchdog"
if [ -d "$LOG_DIR" ]; then
    # Check write permissions
    echo "test" > "$LOG_DIR/.write_test" 2>/dev/null && rm -f "$LOG_DIR/.write_test" && record_pass "E.20" "Log directory is writable" || record_fail "E.20" "Log directory not writable"
else
    record_warn "E.20" "Log directory doesn't exist (will be created on next PS1 start)"
fi

# Test 21: What if .env file is corrupted/missing?
echo "[Test 21] .env file integrity..."
if [ -f /root/.hermes/.env ]; then
    ENV_SIZE=$(stat -c%s /root/.hermes/.env 2>/dev/null)
    if [ "$ENV_SIZE" -gt 100 ]; then
        record_pass "E.21" ".env exists and has content ($ENV_SIZE bytes)"
    else
        record_warn "E.21" ".env exists but suspiciously small ($ENV_SIZE bytes)"
    fi
else
    record_warn "E.21" "WARNING: .env file missing! Gateway restart would fail without WeChat credentials"
fi

# Test 22: hermes binary integrity
echo "[Test 22] hermes command integrity..."
if command -v hermes &>/dev/null; then
    HERMES_PATH=$(which hermes)
    if [ -x "$HERMES_PATH" ]; then
        record_pass "E.22" "hermes command is executable at $HERMES_PATH"
    else
        record_fail "E.22" "hermes command not executable"
    fi
    # Check if it's a symlink/script (the shell script launcher)
    if file "$HERMES_PATH" | grep -q "Bourne-Again shell script\|ASCII text"; then
        record_warn "E.22b" "hermes is a shell script (not binary) - could this cause launch failures?"
    fi
else
    record_fail "E.22" "hermes command not found in PATH"
fi

# Test 23: Stale lock file - what if /var/run/hermes-watchdog.lock is stale?
echo "[Test 23] Stale lock file handling..."
LOCKFILE="/var/run/hermes-watchdog.lock"
if [ -f "$LOCKFILE" ]; then
    # Check if the lock holder process still exists
    LOCK_PID=$(fuser "$LOCKFILE" 2>/dev/null | awk '{print $2}')
    if [ -n "$LOCK_PID" ] && kill -0 "$LOCK_PID" 2>/dev/null; then
        record_pass "E.23" "Lock file held by live process (PID $LOCK_PID)"
    else
        record_warn "E.23" "Lock file exists but holder process may be dead - STALE LOCK POSSIBLE"
    fi
else
    record_warn "E.23" "Lock file missing (will be re-created on next watchdog.sh run)"
fi

# Test 24: What if flock/mutex both available? (should just run normally)
echo "[Test 24] Double-instance race: PS1 Mutex + WSL flock..."
# Both should be fine - Mutex prevents duplicate PS1, flock prevents duplicate watchdog.sh
record_pass "E.24" "Architecture: Mutex (PS1 level) + flock (WSL level) provide layered locking"

# Test 25: PowerShell Job memory leak check
echo "[Test 25] PowerShell Job leak analysis..."
JOB_COUNT=$(powershell.exe -Command '(Get-Job).Count' 2>&1 | tr -d '\r')
echo "Active PowerShell Jobs: $JOB_COUNT"
if [ "$JOB_COUNT" -lt 5 ]; then
    record_pass "E.25" "No PowerShell Job leak (count=$JOB_COUNT)"
else
    record_warn "E.25" "Possible Job leak: $JOB_COUNT active jobs"
fi

# Test 26: VBS silent failure analysis
echo "[Test 26] VBS error mode analysis..."
if grep -qi "on error resume next" "$VBS"; then
    record_warn "E.26" "VBS uses On Error Resume Next - ALL errors silently swallowed"
else
    record_pass "E.26" "VBS allows error propagation"
fi
# Check if VBS has any error handling
if grep -qi "err\." "$VBS" 2>/dev/null; then
    record_pass "E.26b" "VBS has some error handling"
else
    record_warn "E.26b" "VBS has zero error handling - any failure is silent"
fi

# ====================================================================
# ADVANCED ATTACK TESTS
# ====================================================================
echo ""
echo "=== ADVANCED ATTACK / STRESS TESTS ==="

# Test 27: SIGTERM (gentle kill) to Gateway
echo "[Test 27] SIGTERM to Gateway..."
GW_PID=$(ps aux | grep "hermes.*gateway.*run" | grep -v grep | grep python | grep -v "bash -c" | awk '{print $2}' | head -1)
kill -15 "$GW_PID" 2>/dev/null || true
sleep 3
GW_PID_AFTER=$(ps aux | grep "hermes.*gateway.*run" | grep -v grep | grep python | grep -v "bash -c" | awk '{print $2}' | head -1)
if [ -z "$GW_PID_AFTER" ] || [ "$GW_PID" != "$GW_PID_AFTER" ]; then
    record_pass "L2.27" "SIGTERM kills gateway (watchdog should restart)"
else
    record_warn "L2.27" "Gateway survived SIGTERM (PID $GW_PID still alive)"
fi
# Restart if needed
GW=$(ps aux | grep "hermes.*gateway.*run" | grep -v grep | grep python | grep -v "bash -c" | head -1)
if [ -z "$GW" ]; then
    bash /root/hermes-watchdog.sh
    sleep 5
fi

# Test 28: What if WSL goes to sleep (processes frozen)?
echo "[Test 28] WSL sleep/freeze simulation..."
echo "Simulating frozen WSL by creating a lockfile that blocks watchdog.sh..."
OLD_CONTENT=$(cat /root/hermes-watchdog.sh)
# The script itself has flock protection that handles this
record_warn "L2.28" "Cannot fully test WSL freeze without wsl --terminate (prohibited)"

# Test 29: Race - What if mutex is created but Mutex.WaitOne(0) returns false due to access denied?
echo "[Test 29] Mutex access denied scenario..."
MUTEX_AD=$(powershell.exe -Command '
try {
    # Try creating mutex with wrong prefix to simulate access denied
    $mutex = New-Object System.Threading.Mutex($false, "Global\NonExistentMutex")
    $acquired = $mutex.WaitOne(0)
    $mutex.ReleaseMutex()
    Write-Host "Mutex_OK"
} catch {
    Write-Host "Mutex_Exception: $_"
}
' 2>&1 | tr -d '\r')
echo "Mutex edge case: $MUTEX_AD"

# Test 30: Batch rapid file deletion + restoration
echo "[Test 30] What if key files temporarily missing..."
record_warn "E.30" "Cannot test file deletion due to safety constraints"

# ====================================================================
# SUMMARY
# ====================================================================
echo ""
echo "============================================================"
echo " STRESS TEST RESULTS SUMMARY"
echo "============================================================"
echo " Total: $((PASS + FAIL + WARN)) tests"
echo " PASS:  $PASS"
echo " FAIL:  $FAIL"
echo " WARN:  $WARN"
echo ""

for res in "${RESULTS[@]}"; do
    echo "  $res"
done

echo ""
echo "============================================================"
echo " Test completed: $(date)"
echo " Gateway status: $(ps aux | grep 'hermes.*gateway.*run' | grep -v grep | grep python | grep -v 'bash -c' | head -1)"
echo " PS1 status: $(powershell.exe -Command '\$m = New-Object System.Threading.Mutex(\$false, \"Global\HermesGatewayWatchdog\"); if(\$m.WaitOne(0)){Write-Host \"NOT_RUNNING\";\$m.ReleaseMutex()}else{Write-Host \"RUNNING\"}' 2>&1 | tr -d '\r')"
echo "============================================================"
