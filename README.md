# AI Duet：跟觀眾一起即興的 AI 樂團

觀眾在 MIDI 鍵盤上彈一個和弦，AI 樂團就以那個調開始演奏。按住音時，樂團會跟著你彈；在畫面上點一下，可以從選單換風格。
投影畫面會即時反應每個樂器的音量、拍子，以及觀眾彈的音。

- 五個 AI 樂手：和弦、節奏、貝斯、鋪底、主旋律。用 Google 的 Magenta RealTime 2（MRT2）在筆電的 RTX 3050 上即時生成音訊。每個位置有一個樂器池，例如和弦有 Rhodes、Wurlitzer、管風琴；某個位置休息時，會換成池裡的另一種樂器。
- 一個鼓手：鼓機照規則排節奏，再交給 GrooVAE（Magenta 的真人鼓手模型）補上 hi-hat、力度和微小的前後差，最後用 one-shot 取樣發聲。memphis 另外有跟著大鼓走的 808 貝斯。
- 只有鼓手打鼓，也沒有人聲。MRT2 的提示詞都加了 instrumental，「不要鼓」的強度調到 8。另外有監聽程式：演奏中輪流聽每個 AI 樂器，聽起來像鼓聲或人聲就馬上關掉，下一段換別的樂器。
- 風格：
  - 內建 neo soul 和 memphis rap（一律小調、較暗）。
  - 開始演奏前，可以在畫面上打字新增風格，例如 rainy night lo-fi。系統會自動用描述上比較像的那種風格的鼓組和樂器。
  - 切換風格前的最後一小節，提示詞會逐拍混合過去。
  - 每段（4 小節）樂器會隨機退出或加入，鼓在每段結尾有過門。

## 目前進度（2026-11）

已完成，可以現場演出：
- 五個 AI 樂手加一個鼓手同時即時生成。RTX 3050 Laptop 上每幀約 34–37 ms（預算 40 ms），測試中 0 次斷音；按鍵到聽到樂團回應約 0.6 秒，觀眾自己彈的鋼琴聲是立刻發出的。
- 調性跟著觀眾的和弦；和弦進行會自己走，也可以被觀眾的和弦帶走；按住音時所有樂器都跟著彈。
- 兩種內建風格加自訂風格（`custom_styles.txt`），用選單在演出中切換，下一段生效。
- 鼓組在 `samples/` 的音色包之間輪換；memphis 的 808 會自動調到和弦根音。
- 監聽（`band/guard.py`）會擋掉 AI 樂器偶爾冒出來的鼓聲和人聲。
- 視覺：每個聲音一道光柱；自訂風格有自己的顏色和圖案；可以調亮度、暫停、靜音任一樂器、結束整個樂團。

還在進行中：
- 用整晚跑的試音（`run_overnight.bat`）測大約 50 種新樂器，例如薩克斯風、小號、口琴、小提琴、手風琴、合成鋪底，把能用的加進樂器池。
- 用長時間測試（同一個整晚跑）確認連續演出幾個小時的穩定度，並統計監聽最常抓到哪些樂器。

已知限制：
- 監聽要先累積 10 秒聲音才能判斷，所以有問題的樂器最多可能出聲十幾秒才被換掉。
- 打字新增風格只能在開始演奏前：文字轉成風格要約 7 秒。
- 筆電長時間跑 GPU 時，原廠變壓器供電可能不夠，見「注意事項」。

## 架構

```
Windows                                   WSL2 (Ubuntu)
┌──────────────────────┐   MIDI 音符 TCP   ┌───────────────────────────────────────┐
│ MIDI 鍵盤 ──► 橋接    │ ───────────────► │ band_mrt2.py（樂團）                     │
│   （同時發鋼琴聲）     │                  │   band/ 套件：和聲 → MRT2 批次生成（GPU）  │
│                      │   音量/拍子 TCP   │             鼓手 ──► groove_server.py    │
│ vj_tk.py（視覺，內顯） │ ◄─────────────── │                       （GrooVAE，CPU）   │
│   選單 ──► 切換風格    │ ───────────────► │             監聽 ──► band/guard.py       │
│                      │                  │                       （MusicCoCa，CPU）  │
│                      │                  │   混音 ──► PulseAudio ──► Windows 喇叭    │
└──────────────────────┘                  └───────────────────────────────────────┘
```

五個 AI 樂手合成一批，在 GPU 上一次生成。每批 5 幀，每幀 40 ms；4 GB 顯卡一個人一個人生成會來不及。
每一幀告訴 MRT2 三件事：要彈的音級和音域（跟著和弦或觀眾）、風格提示詞（樂器 + 風格 + instrumental）、不要打鼓。

## 檔案

| 檔案 | 用途 |
|------|------|
| `run_band.bat` / `run_band.sh` | 啟動：設定 Windows 的 python 用內顯，再在 WSL 裡啟動樂團 |
| `band_mrt2.py` | 入口：WSL 端啟動樂團；Windows 端 `--midi-bridge` 讀鍵盤（由樂團自動啟動） |
| `band/settings.py` | 所有可調設定：BPM、風格、各位置的樂器池和音量、曲式機率、鼓取樣對應、要過濾的字、監聽門檻 |
| `band/band.py` | 主流程：等和弦 → 每批 5 幀決定和弦進行、風格、編制、樂器、鼓 → 生成、監聽、混音、播放 |
| `band/harmony.py` | 判斷觀眾的和弦；每個位置每一幀要彈什麼（MRT2 的 conditioning） |
| `band/mrt2.py` | 把五個位置合成一批在 GPU 上生成；文字轉風格 token（存在硬碟快取）與混合 |
| `band/drummer.py` | 鼓機規則、送 GrooVAE、取樣發聲、鼓組輪換、808 |
| `band/guard.py` | 監聽：另一個低優先權程序用 MusicCoCa 聽 AI 樂器有沒有鼓聲或人聲 |
| `band/mixer.py` | 平移、限制器、算給視覺的音量 |
| `band/links.py` | 視覺連線（選單、按鍵送回的指令）、觀眾音符、Windows 端 MIDI 橋接與鋼琴聲 |
| `groove_server.py` | 鼓手的人味：收鼓機的 2 小節 pattern，用 GrooVAE 回傳演奏（獨立程序、自己的環境） |
| `vj_tk.py` | 視覺（Windows 端，tkinter + OpenGL）；`--demo` 可以不接樂團單獨測試 |
| `custom_styles.txt` | 自訂風格：一行一個「風格描述 \| 根據的風格」，開機就準備好、選單裡選得到；開始演奏前打的風格會自動加進來 |
| `samples/<鼓組>/` | 鼓組取樣，不包含在 repo 裡（見下方「鼓組」和 `samples/README.md`） |
| `test_mrt2.py` | 共用工具（時間計算、MIDI 接收、模型載入）；也可以單獨跑一個樂手 |
| `tools/audition.py` | 試音：候選樂器在兩種風格各生成 20 秒，量走音、跟拍、鼓聲、人聲、低頻、音量 |
| `tools/explore.py` | 大量試音（整晚跑用）：約 50 種新樂器在每種風格試 2 次，寫成報告 |
| `tools/soak.py` | 長時間測試：假觀眾陪樂團一直演，記錄斷音、錯誤、記憶體、監聽抓到的樂器 |
| `tools/analyze_rec.py` | 分析錄音：調性、跟拍、有沒有偷打鼓 |
| `run_overnight.bat`、`tools/overnight.sh`、`tools/power_guard.ps1` | 整晚跑 explore → soak；拔電或掉電時自動停止 |

## 硬體與環境

- Windows 11 筆電，有 NVIDIA 顯卡（開發用 RTX 3050 Laptop 4 GB）和內顯（畫視覺用）
- WSL2（Ubuntu）跑樂團；MIDI 鍵盤（開發用 AKAI MPK mini 3）

三個 Python 環境：

| 環境 | 放什麼 | 為什麼分開 |
|------|--------|-----------|
| WSL2 `~/ai_duet_gpu_venv` | MRT2（JAX CUDA）、樂團、監聽 | JAX 的 CUDA 版只有 Linux 有；GPU 直接用 Windows 的驅動，不用裝 CUDA toolkit |
| WSL2 `~/groove_venv` | GrooVAE（舊版 TensorFlow） | 和 MRT2 的套件衝突；只用 CPU |
| Windows `venv` | MIDI 橋接、視覺 | WSL 讀不到 USB MIDI，也不開視窗 |

### WSL2：樂團

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv python install 3.11
uv venv ~/ai_duet_gpu_venv --python 3.11
source ~/ai_duet_gpu_venv/bin/activate
uv pip install "jax[cuda12]" magenta-rt mido scipy numpy librosa soundfile pasimple
python -c "import jax; print(jax.devices())"   # 應該顯示 CudaDevice(id=0)
```

聲音經 WSLg 內建的 PulseAudio 送到 Windows 喇叭，要先取得 libpulse（不用 sudo）：

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

`run_band.sh` 會設定這些環境變數：
- `MAGENTA_HOME`：MRT2 模型檔的位置，第一次執行會自動下載到 `%USERPROFILE%\Documents\Magenta`。
- `XLA_PYTHON_CLIENT_PREALLOCATE=false` 和 `XLA_PYTHON_CLIENT_ALLOCATOR=platform`：4 GB 顯卡不預先佔用顯存。
- `LD_LIBRARY_PATH`：指向上面取得的 libpulse。

### 鼓組

鼓聲取樣有授權限制，所以不包含在這個 repo 裡，要自己放進 `samples/`。做法見 [samples/README.md](samples/README.md)。

最簡單的做法：每種風格一個資料夾，例如 `samples/neo soul/`，裡面放固定檔名的 `kick.wav`、`snare.wav`、`hihat.wav`、`openhat.wav`、`rim.wav`；memphis 另外放 `808.wav`。
也可以直接放整包音色包，鼓手會照子資料夾名稱自動找出各個角色。

每種風格可以用哪些鼓組，寫在 `band/settings.py` 的 `STYLES[...]["kits"]`；沒有的資料夾會自動跳過。某種風格一個鼓組都沒有時，鼓會安靜，其他樂器照常演奏。

## 執行與演奏

1. 接上 MIDI 鍵盤，雙擊 `run_band.bat`。
   - 第一次啟動要編譯模型，約 4 分鐘；之後約 1 分 45 秒。終端機和畫面都會顯示 1/4 到 4/4 的載入進度。
   - 開機時會先檢查 WSL 的音效伺服器。送不出聲音時，會直接說明要執行 `wsl --shutdown` 再重開。
2. 彈一個和弦。根音和大小調決定調性，那一刻就是第一拍，音樂和光柱開始。你彈的音會由 Windows 內建合成器立刻發出鋼琴聲，樂團的回應約 0.6 秒後跟上。
3. 之後和弦進行會自己走。彈一個和弦，可以把進行導向那個和弦。按住音時，所有樂器在自己的音域彈你按的音級，808 跟著你最低的音，休息中的樂器也會回來。
4. 在畫面上點一下（或按 `M`）打開風格選單：
   - 選一個風格，會在下一段（4 小節）開頭生效。
   - 開始演奏前，選單最後一項是「輸入新風格…」。
   - 自訂風格依名稱有自己固定的顏色和圖案。
5. 畫面上有七道光柱，每個聲音一道，由左到右是：鼓、貝斯（memphis 時是 808）、和弦、你（中間）、節奏、鋪底、主旋律。按 `S` 會在光柱下面顯示正在演奏的樂器名稱。
6. 畫面的按鍵（移動滑鼠會顯示提示）：

   | 按鍵 | 作用 |
   |------|------|
   | 點一下畫面 / `M` | 風格選單（● 是現在的風格；選回現在的風格 = 取消待切換） |
   | `Enter` | 只有開始演奏前可以用（見下方說明） |
   | `1`–`6` | 依序切換和弦、節奏、貝斯、鼓、鋪底、主旋律的靜音（畫面會寫出樂器名稱；再按一次恢復） |
   | `S` | 顯示／隱藏光柱下面的樂器名稱 |
   | `-` / `=` | 畫面變暗／變亮，每次 10%（20%–150%） |
   | `P` | 暫停／繼續：樂團在背後照常跟著拍子走，只是不出聲，繼續時馬上接上 |
   | `Q` | 結束整個樂團 |
   | `F` / `Esc` | 全螢幕／離開全螢幕 |

   按 `Enter` 會開始輸入一個風格描述，再按 Enter 送出、Esc 取消：
   - 英文效果最好。
   - 會帶出鼓聲或人聲的字（rap、trap、drums、cinematic、song、vocal、choir…）會自動拿掉。
   - 約 7 秒準備好，會當作起始風格。
   - 系統自動選比較像的 neo soul 或 memphis rap，決定鼓組、調式和樂器。
   - 會記進 `custom_styles.txt`，下次開機選單裡就有。

   結束時，終端機會印出每幀生成時間和斷音次數；在終端機按 Enter 也能結束。已經有一個樂團在跑時再開一個，會提示先結束舊的。

## 怎麼改

- 設定都在 `band/settings.py`，每一項都有註解：
  - `BPM`
  - `STYLES`：風格描述、強制調式、hi-hat 密度、鼓組、808
  - `INSTRUMENTS`：各位置的樂器池和音量
  - `GAINS` / `PAN`
  - `DROP_P` / `REJOIN_P`
  - `NEXT_CHORD`
  - `SAMPLES`
  - `DRUM_LAG`
  - `DRUM_WORDS` / `VOICE_WORDS`
  - `LEAK_LIMIT`
- 鼓的規則：`band/drummer.py` 的 `Drummer.pattern`（大鼓、小鼓、open hat、過門）。換風格時一定換鼓組；同一風格每 16 小節（`KIT_SEGMENTS`）也會換一次。
- 各位置怎麼跟和弦：`band/harmony.py` 的 `agent_conditioning`（音域、起音頻率、觀眾按鍵時的行為、「不要鼓」的強度）。
- 加一種樂器：
  1. 先把提示詞放進 `tools/audition.py` 的 `CANDIDATES` 試音，或用整晚跑的 `tools/explore.py`。
  2. 只收這樣的樂器：兩種風格都沒走音、鼓聲和人聲都在 0.3 以下、低頻不高（貝斯位置除外）。
  3. 加進 `INSTRUMENTS`，第三個數字填量到的音量。
- 加一個位置：GPU 已經用到每幀約 36 ms／40 ms，再加會斷音。要更多種聲音，請用樂器池。

## 測試

- 不接視覺、把各樂器分軌錄下來：`python band_mrt2.py --no-visual --record /tmp/rec.wav`。
  - 錄成 12 聲道，順序同 `AGENTS`，最後加鼓。
  - 要寫在 WSL 本機硬碟；寫到 `/mnt/c` 會斷音。
- 沒有鍵盤時：
  - 用 TCP 送 `"1 60\n"`（按下）、`"0 60\n"`（放開）這類的行到 127.0.0.1:47600。
  - 視覺的指令是送到 47601 的一行文字：`g <風格>`（選風格）、`t <描述>`（新增風格，只有開始前）、`m1`–`m6`（靜音）、`p`（暫停）、`q`（結束）。
- 分析錄音：`python tools/analyze_rec.py /tmp/rec.wav C major 25.6`（參數是調名、調式、換風格的秒數）。目前的數字：貝斯 95–97% 在調內、各樂器相對鼓 ±40 ms 內。
- 單獨測一個樂手：在 WSL 裡用 `run_band.sh` 相同的環境變數執行 `python test_mrt2.py`。
- 單獨測視覺：`venv\Scripts\python vj_tk.py --demo`。
- 整晚跑：睡前插著電雙擊 `run_overnight.bat`。
  - 可以加結束時間，例如 `run_overnight.bat 06:30`，預設 07:00。
  - 早上看 `reports/explore-日期.md`（哪些新樂器能用、音量多少）和 `reports/soak-日期.md`（斷音、錯誤、記憶體、監聽抓到哪些樂器）。
  - 跑的時候不能開樂團。

## 注意事項

- 視覺必須用內顯：和 RTX 3050 搶 GPU 會造成大量斷音。`run_band.bat` 會把 `venv` 的 python 設成「省電」GPU；如果終端機出現「正在用 RTX 3050 畫圖」的警告，到 Windows 圖形設定手動改。
- 電源：GPU 全速時，筆電供電不足可能會直接斷電。
  - 請用原廠 180 W 變壓器，電池先充飽。
  - 建議在「高效能」計畫裡，把 AC 下的處理器最大狀態設為 99%（關掉 Turbo）。
- 斷電後 WSL 可能會卡住：GPU 時脈卡在 210 MHz，Windows 也連不到 WSL 的 port。執行 `wsl --shutdown` 再重開。
- 快取：JAX 的編譯結果在 WSL 的 `~/.cache/jax`，風格 token 在 `~/.cache/ai_duet/style`。
  - JAX 或模型升級後，如果啟動變慢或出錯，刪掉這兩個資料夾重新算。
  - 改了樂器或提示詞後，第一次開機會多花約 1 分鐘算風格。
- MRT2 的樂器出聲一律比條件晚約 2 幀（70 ms），所以鼓延後 2 幀（`DRUM_LAG`）才對齊。
- 風格描述裡寫「rap」會讓 MRT2 的樂器偷偷帶鼓，所以 memphis 的風格描述只寫 dark and moody。

## 使用的模型

- [Magenta RealTime 2](https://github.com/magenta/magenta-realtime)（Google）：即時音訊生成；MusicCoCa 文字／聲音 embedding
- [GrooVAE](https://magenta.tensorflow.org/groovae)（Magenta）：鼓的力度與微小時間差
