# 整晚測試時保護電腦：拔掉電源、或插著電卻一直掉電（這台筆電長時間跑 GPU 時充電器供電不夠）到 40% 以下，就停掉 WSL 裡的測試。
# 由 run_overnight.bat 啟動；reports\.overnight_running 不見（測試結束）時自己結束。
$flag = Join-Path $PSScriptRoot "..\reports\.overnight_running"
$log = Join-Path $PSScriptRoot "..\reports\power_guard.log"
Start-Sleep 30
while (Test-Path $flag) {
    $b = Get-CimInstance Win32_Battery
    $w = Get-CimInstance -Namespace root\wmi -ClassName BatteryStatus -ErrorAction SilentlyContinue | Select-Object -First 1
    $draining = $w -and $w.Discharging
    if ($b.BatteryStatus -ne 2 -or ($draining -and $b.EstimatedChargeRemaining -lt 40)) {
        "$(Get-Date -Format 'MM-dd HH:mm') stopped: battery $($b.EstimatedChargeRemaining)%, status $($b.BatteryStatus), discharging $draining" | Add-Content $log
        wsl.exe -e pkill -f "tools/explore.py|tools/soak.py|band_mrt2.py|groove_server.py|band.guard"
        break
    }
    Start-Sleep 10
}
