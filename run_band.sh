#!/usr/bin/env bash
# 在 WSL2 中以 GPU 執行 band_mrt2.py（四個 MRT2 agent 加 GrooVAE 鼓手），聲音經 WSLg 的 PulseAudio 送到 Windows 喇叭。
set -euo pipefail

cd "$(dirname "$0")"
source "${AI_DUET_VENV:-$HOME/ai_duet_gpu_venv}/bin/activate"

export MAGENTA_HOME="${MAGENTA_HOME:-/mnt/c/Users/ray/Documents/Magenta}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform

# libpulse 以 apt-get download 解壓到使用者目錄（免 sudo），見 README「GPU 加速 (WSL2)」
PULSE_LIB="$HOME/pulse_libs/root/usr/lib/x86_64-linux-gnu"
export LD_LIBRARY_PATH="$PULSE_LIB:$PULSE_LIB/pulseaudio:${LD_LIBRARY_PATH:-}"

echo "[AI Duet] 載入模型中（約 1 分鐘）..."
exec python band_mrt2.py
