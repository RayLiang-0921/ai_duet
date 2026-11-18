#!/usr/bin/env bash
# 整晚跑：先找新樂器（tools/explore.py，約 1.5 小時），再長時間測試樂團到早上（tools/soak.py）。報告在 reports/。
# 用法：bash tools/overnight.sh [結束時間，預設 07:00]；Windows 上雙擊 run_overnight.bat
set -uo pipefail
cd "$(dirname "$0")/.."
source "${AI_DUET_VENV:-$HOME/ai_duet_gpu_venv}/bin/activate"
export MAGENTA_HOME="${MAGENTA_HOME:-/mnt/c/Users/ray/Documents/Magenta}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_ALLOCATOR=platform TF_CPP_MIN_LOG_LEVEL=2
PULSE_LIB="$HOME/pulse_libs/root/usr/lib/x86_64-linux-gnu"
export LD_LIBRARY_PATH="$PULSE_LIB:$PULSE_LIB/pulseaudio:${LD_LIBRARY_PATH:-}"

if ss -ltn | grep -qE ":4760[0-2] "; then echo "樂團正在執行，請先結束它再跑整晚測試"; exit 1; fi
mkdir -p reports
touch reports/.overnight_running   # tools/power_guard.ps1 看到這個檔案才會繼續保護
trap 'rm -f reports/.overnight_running' EXIT
echo "[整晚] $(date +%H:%M) 1/2 找新樂器"
python -W ignore tools/explore.py 2>>reports/overnight-stderr.log
echo "[整晚] $(date +%H:%M) 2/2 長時間測試到 ${1:-07:00}"
python -W ignore tools/soak.py "${1:-07:00}" 2>>reports/overnight-stderr.log
echo "[整晚] $(date +%H:%M) 完成，報告在 reports/"
