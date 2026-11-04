@echo off
REM Run the four-agent MRT2 band (band_mrt2.py) on the GPU via WSL2
chcp 65001 >nul
wsl.exe -e bash -lc "cd /mnt/d/Projects/ai_duet && bash run_band.sh"
pause
