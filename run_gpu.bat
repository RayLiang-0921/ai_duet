@echo off
REM 從 Windows 以 WSL2 GPU 模式啟動 AI Duet 網頁版，完成後開啟 http://localhost:8000
wsl.exe -e bash -lc "cd /mnt/d/Projects/ai_duet && bash run_gpu.sh"
pause
