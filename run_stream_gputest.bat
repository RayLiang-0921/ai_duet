@echo off
REM Run the MRT2 streaming test (test_mrt2.py) on the GPU via WSL2
chcp 65001 >nul
wsl.exe -e bash -lc "cd /mnt/d/Projects/ai_duet && bash run_stream_gputest.sh"
pause
