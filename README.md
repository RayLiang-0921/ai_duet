# AI Duet — 跟觀眾一起即興的 AI 樂團

觀眾在 MIDI 鍵盤上彈一個和弦，AI 樂團就以那個調開始演奏；按住音時樂團跟著你彈，按空白鍵換風格。
畫面由視覺程式即時反應各樂器的音量、拍子和觀眾彈的音。

- **五個 MRT2 agent**（Rhodes、吉他、貝斯、弦樂、digital bell）：用 Google 的 **Magenta RealTime 2** 在 RTX 3050 上即時生成音訊。
- **一個鼓手**：鼓機照規則排 pattern → **GrooVAE**（Magenta 的真人鼓手模型）補 hi-hat、力度、微小的提前延後 → FL Studio 的 one-shot 取樣發聲；memphis 另有跟著大鼓的 808 貝斯。
- **兩種風格**：neo soul、memphis rap（一律小調、較暗）。切換前一小節提示詞逐拍混合過去；每段（4 小節）樂器會隨機退出或加入，鼓每段結尾有過門。

## 架構

```
Windows                                   WSL2 (Ubuntu)
┌──────────────────────┐   MIDI 音符 TCP   ┌───────────────────────────────────────┐
│ MIDI 鍵盤 ──► 橋接    │ ───────────────► │ band_mrt2.py（樂團）                     │
│   （同時發鋼琴聲）     │                  │   band/ 套件：和聲 → MRT2 批次生成（GPU）  │
│                      │   音量/拍子 TCP   │             鼓手 ──► groove_server.py    │
│ vj_tk.py（視覺，內顯） │ ◄─────────────── │                       （GrooVAE，CPU）   │
│   空白鍵 ──► 切換風格  │ ───────────────► │   混音 ──► PulseAudio ──► Windows 喇叭    │
└──────────────────────┘                  └───────────────────────────────────────┘
```

## 檔案

| 檔案 | 用途 |
|------|------|
| `run_band.bat` / `run_band.sh` | **啟動**：設定 Windows python 用內顯，再在 WSL 裡啟動樂團 |
| `band_mrt2.py` | 入口：WSL 端啟動樂團；Windows 端 `--midi-bridge` 讀鍵盤（由樂團自動啟動） |
| `band/settings.py` | **所有可調設定**：BPM、風格與提示詞、音量、曲式機率、鼓取樣對應 |
| `band/band.py` | 主流程：等和弦 → 每批 5 幀決定和弦進行、風格、編制、鼓 → 生成、混音、播放 |
| `band/harmony.py` | 判斷觀眾的和弦；每個 agent 每一幀要彈什麼（MRT2 的 conditioning） |
| `band/mrt2.py` | 把五個 agent 合成一批在 GPU 上生成；風格提示詞編碼與混合 |
| `band/drummer.py` | 鼓機規則、送 GrooVAE、取樣發聲、808 |
| `band/mixer.py` | 平移、限制器、算給視覺的音量 |
| `band/links.py` | 視覺連線（含空白鍵）、觀眾音符、Windows 端 MIDI 橋接與鋼琴聲 |
| `groove_server.py` | 鼓手的人味：收鼓機的 2 小節 pattern，用 GrooVAE 回傳演奏（獨立程序、自己的環境） |
| `vj_tk.py` | 視覺（Windows 端，tkinter + OpenGL）；`--demo` 可不接樂團單獨測試 |
| `samples/<風格>/` | 鼓的 one-shot：kick、snare、hihat、openhat、rim（+ memphis 的 808.wav）。換檔案就換音色 |
| `test_mrt2.py` | 共用工具（時間計算、MIDI 接收、模型載入）；也可單獨跑一個 agent 的串流 |
| `tools/analyze_rec.py` | 分析錄音：調性、跟拍、有沒有偷打鼓 |

## 環境安裝

三個 Python 環境：

| 環境 | 放什麼 | 為什麼分開 |
|------|--------|-----------|
| WSL2 `~/ai_duet_gpu_venv` | MRT2（JAX CUDA）、樂團 | JAX 的 CUDA 版只有 Linux 有；GPU 直接用 Windows 的驅動，不需 CUDA toolkit |
| WSL2 `~/groove_venv` | GrooVAE（舊版 TensorFlow） | 和 MRT2 的套件衝突；只用 CPU |
| Windows `venv` | MIDI 橋接、視覺 | WSL 讀不到 USB MIDI、也不開視窗 |

### WSL2：樂團

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv python install 3.11
uv venv ~/ai_duet_gpu_venv --python 3.11
source ~/ai_duet_gpu_venv/bin/activate
uv pip install "jax[cuda12]" magenta-rt mido scipy numpy librosa soundfile pasimple
python -c "import jax; print(jax.devices())"   # 應顯示 CudaDevice(id=0)
```

聲音經 WSLg 內建的 PulseAudio 送到 Windows 喇叭，需先（免 sudo）取得 libpulse：

```bash
mkdir -p ~/pulse_libs/deb && cd ~/pulse_libs/deb
apt-get download libpulse0 libsndfile1 libx11-xcb1 libasyncns0 libflac8 libogg0 libopus0 \
    libvorbis0a libvorbisenc2 libx11-6 libxcb1 libxau6 libxdmcp6 libbsd0 libmd0
for f in *.deb; do dpkg -x "$f" ~/pulse_libs/root; done
```

### WSL2：鼓手（GrooVAE）

```bash
uv venv ~/groove_venv --python 3.10
source ~/groove_venv/bin/activate
uv pip install tensorflow==2.9.1 tensorflow-probability==0.17.0 tensorflow-datasets==4.6.0 tensorflow-metadata==1.9.0 \
    protobuf==3.20.3 note-seq==0.0.3 "numpy<1.24" tf-slim
uv pip install --no-deps magenta==2.1.4   # 直接裝會因 llvmlite 建置失敗
mkdir -p ~/groovae && cd ~/groovae
curl -O https://storage.googleapis.com/magentadata/models/music_vae/checkpoints/groovae_2bar_add_closed_hh.tar
mkdir groovae_2bar_add_closed_hh && tar xf groovae_2bar_add_closed_hh.tar -C groovae_2bar_add_closed_hh
```

### Windows

```powershell
python -m venv venv
.\venv\Scripts\pip install mido python-rtmidi numpy PyOpenGL pyopengltk pillow
```

`run_band.sh` 會設定 `MAGENTA_HOME`（MRT2 模型檔，第一次執行自動下載到 `%USERPROFILE%\Documents\Magenta`）、
`XLA_PYTHON_CLIENT_PREALLOCATE=false` 與 `XLA_PYTHON_CLIENT_ALLOCATOR=platform`（4 GB 顯卡不預先佔顯存）、`LD_LIBRARY_PATH`（libpulse）。

## 執行與演奏

1. 接上 MIDI 鍵盤，雙擊 `run_band.bat`。第一次啟動要編譯模型約 4 分鐘，之後有快取約 1 分半；視覺視窗會先出現等待畫面。
2. **彈一個和弦**：根音和大小調決定調性，那一刻就是第一拍，音樂與光束開始。彈的音由 Windows 內建合成器立刻發出鋼琴聲，樂團的回應約 0.6 秒後跟上。
3. 之後和弦進行會自己走（每小節從上一個和弦的常見去處挑一個）；**彈一個和弦**可以把進行導向那個和弦；**按住音**時所有樂器在自己的音域彈你按的音級、808 跟你最低的音，退出的樂器也會回來。
4. 在視覺視窗**按空白鍵**切換風格，在下一段（4 小節）開頭生效（畫面會先寫「下一段開始」，切換時寫「風格切換」）；開始前按則是選擇起始風格。
5. 畫面上五道光柱由左到右是：**鼓**（鼓手實際打下的大鼓、小鼓）、**貝斯**（memphis 時是 808）、**觀眾**（中間，你彈的音）、**Rhodes**、**旋律**（吉他＋鐘）。弦樂和 Rhodes 合起來控制背景亮度，hi-hat 控制背景的閃爍。
6. 視覺視窗的按鍵（移動滑鼠會顯示提示）：

   | 按鍵 | 作用 |
   |------|------|
   | 空白鍵 | 下一段切換風格 |
   | `1`–`6` | 切換 Rhodes、吉他、貝斯、弦樂、鐘、鼓的靜音（優先於自動編制；再按一次恢復） |
   | `P` | 暫停／繼續：樂團在背後照常跟著拍子走，只是不出聲，繼續時馬上接上 |
   | `Q` | 結束整個樂團（不用回終端機） |
   | `F` / `Esc` | 全螢幕／離開全螢幕 |

   結束時終端機會印出每幀生成時間和斷音次數；終端機按 Enter 也能結束。已有樂團在跑時再開一個，會提示先結束舊的。

RTX 3050 Laptop 上五個 agent 約 37 ms/幀（預算 40 ms）、0 次斷音、延遲約 0.6 秒；鼓的取樣發聲每幀只要 0.1 ms。

## 怎麼改

- **設定**都在 `band/settings.py`，每一項都有註解：BPM、`STYLES`（提示詞、風格描述、強制調式、hi-hat 密度）、`GAINS` / `PAN`、`DROP_P` / `REJOIN_P`、`NEXT_CHORD`、`SAMPLES`、`DRUM_LAG`。
- **鼓的音色**：換 `samples/<風格>/` 裡同名的 wav。放一個 `808.wav` 就會有 808 貝斯（自動量基頻、調到和弦根音）。
- **鼓的規則**：`band/drummer.py` 的 `Drummer.pattern`（大鼓 / 小鼓 / open hat / 過門）。
- **各樂器怎麼跟和弦**：`band/harmony.py` 的 `agent_conditioning`（音域、起音頻率、人類按鍵時的行為）。
- **加一個樂器**：`AGENTS` 加名字，`STYLES` 兩種風格各加提示詞，`GAINS` / `PAN` 給值，`agent_conditioning` 給音域；每多一個 agent 每幀約多 1 ms。

## 測試

- 不接視覺、把分軌錄下來：`python band_mrt2.py --no-visual --record /tmp/rec.wav`（12 聲道，順序同 `AGENTS` 再加鼓；要寫在 WSL 本機硬碟，寫到 `/mnt/c` 會斷音）。沒有鍵盤時可以用 TCP 送 `"1 60\n"` 之類的行到 127.0.0.1:47600 模擬按鍵；視覺視窗的指令是送到 47601 的一行文字：`s`（切換風格）、`m 樂器名`（例如 `m strings`，切換靜音）、`p`（暫停）、`q`（結束）。
- 分析錄音：`python tools/analyze_rec.py /tmp/rec.wav C major 25.6`（調名、調式、切到 memphis 的秒數）。目前的數字：貝斯 95–97% 在調內、各樂器相對鼓 ±40 ms 內、沒有 MRT2 樂器鎖在拍點上打鼓。
- 單獨測一個 agent：在 WSL 裡用 `run_band.sh` 相同的環境變數執行 `python test_mrt2.py`；視覺單獨測：`venv\Scripts\python vj_tk.py --demo`。

## 注意事項

- **視覺必須用內顯**：和 RTX 3050 搶 GPU 會造成大量斷音。`run_band.bat` 會把 `venv` 的 python 設成「省電」GPU；若終端機出現「正在用 RTX 3050 畫圖」的警告，到 Windows 圖形設定手動改。
- **電源**：GPU 全速時筆電若供電不足會直接斷電。請用原廠 180 W 變壓器、電池先充飽；建議「高效能」計畫把 AC 下處理器最大狀態設為 99%（關 Turbo）。
- **斷電後 WSL 會卡住**（GPU 時脈卡在 210 MHz、Windows 連不到 WSL 的 port）：執行 `wsl --shutdown` 再重開。
- JAX 的編譯結果快取在 WSL 的 `~/.cache/jax`；JAX 或模型升級後若啟動變慢或出錯，刪掉它重新編譯。
- MRT2 的樂器出聲一律比鼓點條件晚約 2 幀（70 ms），所以鼓延後 2 幀（`DRUM_LAG`）才對齊；提示詞裡寫「rap」會讓 MRT2 的樂器偷偷帶鼓，所以 memphis 的風格描述只寫 dark and moody。
