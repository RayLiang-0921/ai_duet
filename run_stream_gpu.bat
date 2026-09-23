@echo off
REM 以 WSL2 GPU 執行 MRT2 無限串流播放（stream_mrt2.py）
wsl.exe -e bash -lc "cd /mnt/d/Projects/ai_duet && bash run_stream_gpu.sh"
pause
