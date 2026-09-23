@echo off
REM Run MRT2 endless streaming playback (stream_mrt2.py) on the GPU via WSL2
chcp 65001 >nul
wsl.exe -e bash -lc "cd /mnt/d/Projects/ai_duet && bash run_stream_gpu.sh"
pause
