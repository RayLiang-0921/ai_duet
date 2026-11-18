# AI Duet — 跟觀眾一起即興的 AI 樂團

觀眾在 MIDI 鍵盤上彈一個和弦，AI 樂團就以那個調開始演奏；按住音時樂團跟著你彈，按空白鍵換風格。
畫面由視覺程式即時反應各樂器的音量、拍子和觀眾彈的音。

- **五個 MRT2 位置**：和弦、節奏、貝斯、鋪底、主旋律，用 Google 的 **Magenta RealTime 2** 在 RTX 3050 上即時生成音訊。每個位置有一個樂器池（例如和弦：Rhodes、Wurlitzer、管風琴），位置休息時會換成池裡的另一種樂器。
- **一個鼓手**：鼓機照規則排 pattern → **GrooVAE**（Magenta 的真人鼓手模型）補 hi-hat、力度、微小的提前延後 → FL Studio 的 one-shot 取樣發聲；memphis 另有跟著大鼓的 808 貝斯。
- **風格**：neo soul、memphis rap（一律小調、較暗），也可以在視覺視窗**打字新增**（例如 rainy night lo-fi）。切換前一小節提示詞逐拍混合過去；每段（4 小節）樂器會隨機退出或加入，鼓每段結尾有過門。

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
| `band/settings.py` | **所有可調設定**：BPM、風格、各位置的樂器池、音量、曲式機率、鼓取樣對應、打字時要過濾的字 |
| `band/band.py` | 主流程：等和弦 → 每批 5 幀決定和弦進行、風格、編制、鼓 → 生成、混音、播放 |
| `band/harmony.py` | 判斷觀眾的和弦；每個位置每一幀要彈什麼（MRT2 的 conditioning） |
| `band/mrt2.py` | 把五個位置合成一批在 GPU 上生成；文字轉風格 token（存在硬碟快取）與混合 |
| `band/drummer.py` | 鼓機規則、送 GrooVAE、取樣發聲、808 |
| `band/mixer.py` | 平移、限制器、算給視覺的音量 |
| `band/links.py` | 視覺連線（含空白鍵）、觀眾音符、Windows 端 MIDI 橋接與鋼琴聲 |
| `groove_server.py` | 鼓手的人味：收鼓機的 2 小節 pattern，用 GrooVAE 回傳演奏（獨立程序、自己的環境） |
| `vj_tk.py` | 視覺（Windows 端，tkinter + OpenGL）；`--demo` 可不接樂團單獨測試 |
| `custom_styles.txt` | 自訂風格清單：一行一個「風格描述 \| 根據的風格」，開機就準備好，空白鍵輪換得到；演出中打的風格會自動加進來 |
| `samples/<鼓組>/` | 鼓組。`neo soul`、`memphis rap` 是挑好的（固定檔名 kick.wav…）；其他資料夾是整包音色包，鼓手照子資料夾名稱自動找大鼓、小鼓、hi-hat、open hat、rim、808，每次隨機挑一個 |
| `test_mrt2.py` | 共用工具（時間計算、MIDI 接收、模型載入）；也可單獨跑一個 agent 的串流 |
| `tools/analyze_rec.py` | 分析錄音：調性、跟拍、有沒有偷打鼓 |
| `tools/audition.py` | 試音：候選樂器在兩種風格各生成 20 秒，量走音、跟拍、偷帶鼓聲或低頻、音量；也量文字轉風格的時間 |

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
5. 畫面上五道光柱由左到右是：**鼓**（鼓手實際打下的大鼓、小鼓）、**貝斯**（memphis 時是 808）、**觀眾**（中間，你彈的音）、**和弦**、**旋律**（節奏＋主旋律）。和弦與鋪底合起來控制背景亮度，hi-hat 控制背景的閃爍。按 `S` 會在光柱底下顯示當下的樂器名稱。
6. 視覺視窗的按鍵（移動滑鼠會顯示提示）：

   | 按鍵 | 作用 |
   |------|------|
   | 空白鍵 | 下一段切換風格（依序輪換，包含打字新增的） |
   | `Enter` | 輸入一個風格描述，再按 Enter 送出、Esc 取消。英文效果最好；會帶出鼓聲的字（rap、trap、drums、cinematic…）會自動拿掉。約 7 秒準備好，在下一段接上，沿用當時風格的鼓和樂器，並記進 `custom_styles.txt`，下次開機空白鍵就切得到 |
   | `1`–`6` | 切換 Rhodes、吉他、貝斯、鼓、弦樂、鐘的靜音（優先於自動編制；再按一次恢復） |
   | `S` | 顯示／隱藏光柱底下的名稱（鼓會寫出目前的鼓組，memphis 時貝斯寫 808） |
   | `P` | 暫停／繼續：樂團在背後照常跟著拍子走，只是不出聲，繼續時馬上接上 |
   | `Q` | 結束整個樂團（不用回終端機） |
   | `F` / `Esc` | 全螢幕／離開全螢幕 |

   結束時終端機會印出每幀生成時間和斷音次數；終端機按 Enter 也能結束。已有樂團在跑時再開一個，會提示先結束舊的。

RTX 3050 Laptop 上五個 agent 約 37 ms/幀（預算 40 ms）、0 次斷音、延遲約 0.6 秒；鼓的取樣發聲每幀只要 0.1 ms。

## 怎麼改

- **設定**都在 `band/settings.py`，每一項都有註解：BPM、`STYLES`（風格描述、強制調式、hi-hat 密度、鼓組、808）、`INSTRUMENTS`（各位置的樂器池）、`GAINS` / `PAN`、`DROP_P` / `REJOIN_P`、`NEXT_CHORD`、`SAMPLES`、`DRUM_LAG`、`DRUM_WORDS`。
- **鼓組**：每種風格可用哪些鼓組寫在 `band/settings.py` 的 `STYLES[...]["kits"]`。換風格時一定換鼓組，同一風格每 16 小節（`KIT_SEGMENTS`）也會換一次，每次輪到時取樣都重新隨機挑。新增鼓組只要把音色包放進 `samples/`、名字加進 `kits`；子資料夾的命名規則在 `band/drummer.py` 的 `KIT_ROLES`。808 一樣自動找、自動量基頻調到和弦根音。
- **鼓的規則**：`band/drummer.py` 的 `Drummer.pattern`（大鼓 / 小鼓 / open hat / 過門）。
- **各位置怎麼跟和弦**：`band/harmony.py` 的 `agent_conditioning`（音域、起音頻率、人類按鍵時的行為）。
- **加一種樂器**：先把提示詞放進 `tools/audition.py` 的 `CANDIDATES` 試音；兩種風格都沒走音、沒偷帶鼓聲（hat 鎖 > 0.4 且 hf% 高）、非貝斯位置的低頻（lf%）不高的，再加進 `INSTRUMENTS`，第三個數字填試音量到的音量。
- **加一個位置**：GPU 已經用到每幀約 36 ms／40 ms，再加會斷音；要更多種聲音請用樂器池。

## 測試

- 不接視覺、把分軌錄下來：`python band_mrt2.py --no-visual --record /tmp/rec.wav`（12 聲道，順序同 `AGENTS` 再加鼓；要寫在 WSL 本機硬碟，寫到 `/mnt/c` 會斷音）。沒有鍵盤時可以用 TCP 送 `"1 60\n"` 之類的行到 127.0.0.1:47600 模擬按鍵；視覺視窗的指令是送到 47601 的一行文字：`s`（切換風格）、`m1`–`m6`（靜音）、`p`（暫停）、`q`（結束）。
- 分析錄音：`python tools/analyze_rec.py /tmp/rec.wav C major 25.6`（調名、調式、切到 memphis 的秒數）。目前的數字：貝斯 95–97% 在調內、各樂器相對鼓 ±40 ms 內、沒有 MRT2 樂器鎖在拍點上打鼓。
- 單獨測一個 agent：在 WSL 裡用 `run_band.sh` 相同的環境變數執行 `python test_mrt2.py`；視覺單獨測：`venv\Scripts\python vj_tk.py --demo`。

## 注意事項

- **視覺必須用內顯**：和 RTX 3050 搶 GPU 會造成大量斷音。`run_band.bat` 會把 `venv` 的 python 設成「省電」GPU；若終端機出現「正在用 RTX 3050 畫圖」的警告，到 Windows 圖形設定手動改。
- **電源**：GPU 全速時筆電若供電不足會直接斷電。請用原廠 180 W 變壓器、電池先充飽；建議「高效能」計畫把 AC 下處理器最大狀態設為 99%（關 Turbo）。
- **斷電後 WSL 會卡住**（GPU 時脈卡在 210 MHz、Windows 連不到 WSL 的 port）：執行 `wsl --shutdown` 再重開。
- JAX 的編譯結果快取在 WSL 的 `~/.cache/jax`，風格 token 在 `~/.cache/ai_duet/style`；JAX 或模型升級後若啟動變慢或出錯，刪掉它們重新算。新增樂器後第一次開機會多花約 30 秒算風格。
- MRT2 的樂器出聲一律比鼓點條件晚約 2 幀（70 ms），所以鼓延後 2 幀（`DRUM_LAG`）才對齊；提示詞裡寫「rap」會讓 MRT2 的樂器偷偷帶鼓，所以 memphis 的風格描述只寫 dark and moody。
