@echo off
REM Run the MRT2 band (band_mrt2.py) on the GPU via WSL2; the visuals (vj_tk.py) open on Windows
chcp 65001 >nul
REM Make Windows run the visuals on the integrated GPU (Graphics settings: python = Power saving), keeping the RTX 3050 for the music
for /f "delims=" %%p in ('venv\Scripts\python.exe -c "import sys;print(sys._base_executable)"') do reg add "HKCU\Software\Microsoft\DirectX\UserGpuPreferences" /v "%%p" /t REG_SZ /d "GpuPreference=1;" /f >nul
wsl.exe -e bash -lc "cd /mnt/d/Projects/ai_duet && bash run_band.sh"
pause
