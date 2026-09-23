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
| 硬體 | 目前僅驗證 **CPU** 推論（無 GPU/TPU 加速），見下方「即時效能」章節 |

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

## 即時效能 / 已知限制

本機環境僅有 CPU（無 GPU/TPU 加速）。實測 `mrt2_small` 在此環境下的即時因子
（real-time factor，>1x 代表比即時播放慢）約為 **9–10 倍**：生成 1 秒音訊約需
9–10 秒運算時間。這代表**目前在純 CPU 環境下無法達到真正零延遲的即時演奏**，
網頁版採用 prefetch/double-buffer 機制隱藏部分延遲，但底層生成速度本身仍是瓶頸。
若要達到真正即時，需要 GPU 加速、更小的模型、或調整生成策略。

其他已知問題：
- `duet_system.py` 的 `magenta_duet_task` 目前對每個 Agent 呼叫了兩次獨立的 MRT2
  推論（一次 `generate_audio()` 結果被丟棄，只用於 log；一次 `generate()` 才是實際
  使用的音符來源），造成不必要的運算浪費。
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
