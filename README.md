# AI Duet — 即時人機互彈系統 (Magenta RealTime 2)

> 一個由神經網路音訊模型驅動的即時 AI 樂團：多個 AI Agent 用 **Magenta RealTime 2 (MRT2)**
> 互相接續彈奏，人類可以在任何時間點「亂入」彈奏，系統會立即偵測、清空 AI 待發音符，
> 並將 AI 導向跟隨人類的新旋律方向繼續演奏。

---

## 專案結構

```
ai_duet/
├── midi_core.py         # 多軌 MIDI 路由、虛擬埠、人類亂入攔截器、Preset Bass 背景軌
├── duet_system.py        # MRT2 AI 互彈引擎（CLI 版本，輸出 duet_output.mid）
├── ai_duet_web.py         # FastAPI + WebSocket 互動網頁版（主要展示介面，含可彈奏鍵盤 UI）
├── run_gpu.sh / .bat   # 以 WSL2 + GPU 啟動網頁版（推薦）
├── stream_mrt2.py      # MRT2 無限串流播放（run_stream_gpu.sh / .bat 以 GPU 執行）
├── tools/ws_latency_check.py  # 量測網頁版每小節是否即時送達
├── train_melody.py        # 旋律模型訓練腳本
├── ai_duet_prototype.py    # 最早期原型（Markov Chain call-and-response，僅供參考）
├── tests/                  # pytest 測試（MIDI 路由、MRT2 生成邏輯，含 mock 與真實模型測試）
├── samples/                # 音色樣本（bass/guitar/kick/snare）
├── pyrightconfig.json      # Pyright / Pylance 型別檢查設定
├── .vscode/settings.json   # VS Code Python 直譯器設定
├── venv/                   # Python 虛擬環境
└── README.md
```

---

## 環境需求

| 需求 | 版本 |
|------|------|
| Python | 3.11+ |
| OS | Windows 10/11（macOS/Linux 亦可） |
| 硬體 | **建議 NVIDIA GPU（已驗證 RTX 3050 Laptop 4GB，經 WSL2）**；純 CPU 可執行但約慢 9 倍，見「即時效能」 |

### 主要套件（已於 `venv` 驗證安裝成功，Windows 原生環境可行）

```powershell
pip install magenta-rt fastapi uvicorn mido note-seq torch scipy numpy resampy
```

| 套件 | 版本 |
|------|------|
| magenta-rt | 2.0.3 |
| torch | 2.12.0 |
| jax / jaxlib | 0.10.2 |
| fastapi | 0.136.1 |
| uvicorn | 0.46.0 |
| mido | 1.3.3 |
| note-seq | 0.0.5 |
| scipy | 1.17.1 |
| numpy | 2.3.5 |
| resampy | 0.4.3 |

> **注意**：與舊版說明不同，`magenta-rt`（MRT2）**已可在 Windows 原生環境安裝並成功運行**，
> 不需要 Linux/WSL。若 `magenta-rt` 未安裝或載入失敗，系統會自動降級為音階隨機游走 fallback，
> 不會直接崩潰。

---

## 快速開始

### 1. 建立並啟動虛擬環境

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 2. 安裝依賴

```powershell
pip install magenta-rt fastapi uvicorn mido note-seq torch scipy numpy resampy
```

### 3a. 執行方式 A — 互動網頁版（推薦，主要展示介面）

**GPU（推薦，可即時演奏）**：雙擊 `run_gpu.bat`，或在 WSL 內執行 `bash run_gpu.sh`。
第一次需先完成下方「GPU 加速 (WSL2)」的安裝。啟動畫面會顯示 `JAX backend = gpu`。

**CPU（無 GPU 時）**：

```powershell
python ai_duet_web.py
```

啟動後於瀏覽器開啟顯示的網址（預設 `http://localhost:8000`）。網頁提供：

- 可彈奏虛擬鍵盤，即時聽到 AI Agent 的回應
- 多種調性 / Hip-Hop 和弦進行選擇
- **Text-to-Synth**：輸入文字描述生成對應風格 embedding
- **Audio Cloning**：上傳音訊片段，萃取其音色風格
- **Prompt Mixing**：在兩個風格之間插值混合

> ⚠️ 已知限制：Audio Cloning 目前僅支援原始 headerless float32 PCM 資料；
> 直接上傳一般音樂檔（.mp3/.wav）會因為前端未先解碼而失敗，見「已知問題」章節。

### 3b. 執行方式 B — CLI 主程式（60 秒，生成 MIDI）

```powershell
python duet_system.py
```

- 前 15 秒：Agent A（MRT2_Primo, C Major）與 Agent B（MRT2_Secondo, G Mixolydian）互彈，
  Preset Bass 持續循環
- T+15s：模擬人類亂入（彈 6 個音），攔截器觸發
- T+15s~60s：AI 順著人類新旋律改變走向
- 結束後輸出 **`duet_output.mid`**

### 3c. 執行方式 C — 核心路由測試（8 秒，純 log 驗證，不含 MRT2）

```powershell
python midi_core.py
```

---

## 自訂參數

### 調整亂入時間與音符數（`duet_system.py` / `ai_duet_web.py`）

```python
kwargs={"delay_sec": 15.0, "note_count": 6}
#        ↑ 幾秒後亂入       ↑ 連彈幾個音
```

### 調整 Agent 風格（style prompt）

```python
agent_a = MRT2Generator(
    name="MRT2_Primo",
    scale=C_MAJOR_PITCHES,
    style_prompt="jazz piano melody improvisation",  # 修改此處文字即可改變音色/風格
    mrt2_system=_mrt2_system,
)
```

### 接入真實 MIDI 鍵盤

在 `midi_core.py` 的 `simulate_human_input()` 裡，把模擬邏輯替換為 `mido.open_input()`：

```python
import mido

with mido.open_input() as port:
    for msg in port:
        if msg.type == "note_on" and msg.velocity > 0:
            note = MidiNote(channel=msg.channel, pitch=msg.note,
                            velocity=msg.velocity, source="Human")
            loop.call_soon_threadsafe(human_in.send, note)
```

---

## 系統架構

```
┌─────────────────────────────────────────────────────────────┐
│                      ai_duet_web.py                          │
│   FastAPI + WebSocket + 可彈奏鍵盤 UI（主要展示介面）           │
│   Text-to-Synth / Audio Cloning / Prompt Mixing               │
└───────────────┬───────────────────────────────────────────────┘
                │ import
┌───────────────▼───────────────────────────────────────────────┐
│                      duet_system.py                            │
│                                                                 │
│  MRT2Generator (Agent A — Primo)   style: "jazz piano melody"  │
│  MRT2Generator (Agent B — Secondo) style: "ambient pad chords" │
│  MRT2Generator (Agent Bass)        style: "deep sub bass"      │
│    → 直接生成 48kHz 立體聲音訊波形（非傳統 MIDI 音符序列生成）   │
│    → streaming state 跨呼叫延續，人類亂入時重設                 │
│                                                                 │
│  Sliding Window (deque) — Bass hint + Melody history → primer  │
│  MidiRecorder → duet_output.mid                                 │
└───────────────┬───────────────────────────────────────────────┘
                │ import
┌───────────────▼───────────────────────────────────────────────┐
│                      midi_core.py                               │
│                                                                 │
│  VirtualMidiPort  AI_Melody_Out / AI_Harmony_Out / Preset_Bass │
│  VirtualMidiPort  Human_In                                     │
│                                                                 │
│  on_human_interrupt()  ← 人類亂入攔截器                         │
│    1. 切換 SystemState → HUMAN_MODE                            │
│    2. 清空 ai_pending_buffer                                    │
│    3. 寫入 human_buffer                                         │
│    4. [duet_system 擴充] reset_primer(Agent A & B)               │
└─────────────────────────────────────────────────────────────┘
```

---

## Thread Safety

| Lock | 保護目標 |
|------|---------|
| `MidiBuffer._lock` | ai_pending_buffer / human_buffer |
| `SystemState._lock` | AI ↔ 人類模式切換 |
| `MRT2Generator._lock` (RLock) | MRT2 streaming state、temperature、style embedding |
| `MidiRecorder._lock` | 錄音事件列表並發寫入 |
| `_sliding_lock` | melody_window、harmony_window、bass_pitches |

---

## GPU 加速 (WSL2)

JAX 在 PyPI 上只提供 Linux 的 CUDA 版本，因此 Windows 上需透過 **WSL2** 使用 GPU。
WSL2 會直接使用 Windows 的 NVIDIA 驅動，不需另外安裝 CUDA toolkit，也不需要 sudo。

```bash
# 在 WSL (Ubuntu) 內執行，一次性安裝
curl -LsSf https://astral.sh/uv/install.sh | sh     # 安裝 uv（使用者目錄，免 sudo）
uv python install 3.11
uv venv ~/ai_duet_gpu_venv --python 3.11
source ~/ai_duet_gpu_venv/bin/activate
uv pip install "jax[cuda12]" magenta-rt fastapi "uvicorn[standard]" mido note-seq scipy numpy resampy
python -c "import jax; print(jax.devices())"         # 應顯示 CudaDevice(id=0)
```

`run_gpu.sh` 會自動設定以下環境變數：

| 變數 | 值 | 原因 |
|------|----|------|
| `MAGENTA_HOME` | `/mnt/c/Users/ray/Documents/Magenta` | 共用 Windows 已下載的 checkpoint，免重新下載 2.8GB |
| `XLA_PYTHON_CLIENT_PREALLOCATE` | `false` | JAX 預設預先配置 75% 顯存，4GB 顯卡會 OOM |
| `XLA_PYTHON_CLIENT_ALLOCATOR` | `platform` | 同上，依需求配置顯存 |

### GPU 串流播放腳本（`stream_mrt2.py`）

雙擊 `run_stream_gpu.bat`：MRT2 在 GPU 上邊生成邊播放，可隨時輸入文字切換風格。
WSL 的聲音經 WSLg 內建 PulseAudio 送到 Windows 喇叭，需先（免 sudo）取得 libpulse：

```bash
mkdir -p ~/pulse_libs/deb && cd ~/pulse_libs/deb
apt-get download libpulse0 libsndfile1 libx11-xcb1 libasyncns0 libflac8 libogg0 libopus0     libvorbis0a libvorbisenc2 libx11-6 libxcb1 libxau6 libxdmcp6 libbsd0 libmd0
for f in *.deb; do dpkg -x "$f" ~/pulse_libs/root; done
uv pip install pasimple
```

> 必須安裝 `uvicorn[standard]`（含 `websockets`）；只裝 `uvicorn` 時 `/ws` 會回傳 404，網頁無法連線。

---

## 即時效能 / 已知限制

`mrt2_small` 實測（real-time factor，>1x 代表比即時播放慢）：

| 測試 | CPU（Windows 原生） | GPU（RTX 3050, WSL2） |
|------|------|------|
| `generate()` 原始速度 | 8.5–9.2x | **0.58–0.87x** |
| 一整小節（2 秒）音訊，單一串流 | — | **0.59x** |
| 一整小節，Bass + Melody 兩條串流 | — | 1.18x（跟不上） |
| 網頁版 60 秒實際演奏（`tools/ws_latency_check.py`） | 跟不上 | **30/30 小節準時、間隔 2.00s、旋律覆蓋 100% 小節** |

單張 GPU 上多條 MRT2 串流會排隊執行，不會真正並行；增加顯存配置也不會變快
（`platform` / BFC allocator 皆為 0.59–0.60x，預先配置 85% 顯存則 OOM）。因此網頁版預設：

- **Melody（Agent A/B 輪流）**：MRT2 生成**整個小節**的音訊
- **Bass**：音符生成器 + 瀏覽器合成器（設定 `AI_DUET_MRT2_BASS=1` 可改回 MRT2，需要更快的 GPU）
- 小節以絕對時間排程送出，避免累積漂移造成片段之間的空隙

可用 `python tools/gpu_measure_bench.py`（WSL GPU venv）重現上述 GPU 數據。
注意事項：4GB 顯存很緊，載入時會出現 XLA `ran out of memory ... retrying` 警告但會自動恢復；
演奏時請避免同時執行其他吃 GPU 的程式。模型首次載入約 1–2 分鐘。

其他已知問題：
- Audio Cloning 功能目前上傳檔案未經解碼即直接送入模型，僅支援 headerless
  float32 PCM 原始資料。

---

## 常見問題

**Q: `ModuleNotFoundError: No module named 'torch'` / `'magenta_rt'`**
A: 執行 `pip install torch magenta-rt` 重新安裝。

**Q: Pylance 顯示 import 錯誤**
A: 確認 VS Code 使用的是 `venv` 裡的 Python（右下角直譯器應為 `venv`），並重新載入視窗
（`Ctrl+Shift+P` → `Reload Window`）。

**Q: 沒有安裝 magenta-rt 會怎樣？**
A: 系統會自動降級為音階隨機游走 fallback（`MRT2Generator.generate()` 內建），
仍可完整執行，但不會有神經網路生成的音樂性。

**Q: 想讓系統跑更久**
A: 修改 `duet_system.py` 的 `await asyncio.sleep(60.0)` 為更大的值，或按 `Ctrl+C`
手動停止（系統會優雅關閉並儲存 MIDI）。
