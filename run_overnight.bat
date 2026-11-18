@echo off
REM Overnight run: audition new instruments, then a long soak test of the band until morning (default 07:00).
REM Reports go to reports\. Optional end time: run_overnight.bat 06:30
chcp 65001 >nul
if not exist reports mkdir reports
start "AI Duet power guard" /min powershell -NoProfile -ExecutionPolicy Bypass -File tools\power_guard.ps1
wsl.exe -e bash -lc "cd /mnt/d/Projects/ai_duet && bash tools/overnight.sh %1"
pause
