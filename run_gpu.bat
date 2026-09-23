@echo off
REM Start the AI Duet web app on the GPU via WSL2, then open http://localhost:8000
chcp 65001 >nul
wsl.exe -e bash -lc "cd /mnt/d/Projects/ai_duet && bash run_gpu.sh"
pause
