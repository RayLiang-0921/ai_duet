#!/usr/bin/env bash
# 在 WSL2 中以 GPU (CUDA) 啟動 AI Duet 網頁版。
# 用法（WSL 內）：  bash run_gpu.sh
#      （Windows）：雙擊 run_gpu.bat
set -euo pipefail

cd "$(dirname "$0")"

VENV="${AI_DUET_VENV:-$HOME/ai_duet_gpu_venv}"
if [ ! -f "$VENV/bin/activate" ]; then
    echo "[錯誤] 找不到 GPU venv：$VENV（見 README「GPU 加速 (WSL2)」安裝步驟）" >&2
    exit 1
fi
source "$VENV/bin/activate"

# 共用 Windows 端已下載的 MRT2 checkpoint，避免重新下載 2.8GB
export MAGENTA_HOME="${MAGENTA_HOME:-/mnt/c/Users/ray/Documents/Magenta}"
# RTX 3050 只有 4GB VRAM：關閉 JAX 預先配置 75% 顯存的預設行為，否則會 OOM
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform

echo "[AI Duet] 檢查 JAX 裝置..."
python -c "import jax; print('[AI Duet] JAX backend =', jax.default_backend(), jax.devices())"
if [ "$(python -c 'import jax; print(jax.default_backend())')" != "gpu" ]; then
    echo "[警告] JAX 沒有偵測到 GPU，將以 CPU 執行（約慢 9 倍）" >&2
fi

echo "[AI Duet] 載入模型中（首次約 1–2 分鐘），完成後開啟 http://localhost:8000"
exec python ai_duet_web.py
