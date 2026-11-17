# AI Duet — 即時 AI 樂團（Magenta RealTime 2 + GrooVAE）

五個 AI agent（Rhodes、吉他、貝斯、弦樂、digital bell）用 **Magenta RealTime 2 (MRT2)** 即時生成音訊，
鼓手由鼓機排 pattern、**GrooVAE**（Magenta 的真人鼓手模型）補 hi-hat 和人味、one-shot 取樣發聲，一起 jam；
觀眾用 MIDI 鍵盤彈和弦決定調性並啟動音樂（彈的音立刻有鋼琴聲），按住音時樂團跟著走，按最低鍵在下一段切換風格（neo soul ↔ memphis rap）。
畫面由 tkinter + OpenGL 的視覺程式即時反應各樂器的音量、拍子和觀眾彈的音。

## 檔案

| 檔案 | 用途 |
|------|------|
| `band_mrt2.py` | 樂團主程式（WSL GPU 端）：五個 agent 批次生成、鼓機與鼓的取樣發聲、和弦進行、風格切換、混音、送資料給視覺；`--midi-bridge` 是 Windows 端的鍵盤橋接（同時發鋼琴聲） |
| `groove_server.py` | 鼓手的人味（WSL CPU 端）：由樂團自動啟動，收鼓機的 2 小節 pattern，用 GrooVAE 回傳真人般的演奏（hi-hat、力度、微小的提前延後） |
| `samples/<風格>/` | 每種風格的鼓 one-shot（kick、snare、hihat、openhat、rim）：neo soul 用 FL 的「@PRODBY.XERO 90's R&B」kit，memphis rap 用「Trap Essentials」 |
| `vj_tk.py` | 視覺（Windows 端）：由樂團自動啟動，用內顯畫，`--demo` 可不接模型單獨測試 |
| `test_mrt2.py` | 共用工具（時間計算、MIDI 接收）；也可單獨跑一個 agent 的串流 |
| `run_band.bat` / `run_band.sh` | 啟動：設定 Windows python 用內顯，再在 WSL 裡啟動樂團 |

## 環境

三個 Python 環境：

- **WSL2（Ubuntu）`~/ai_duet_gpu_venv`**：JAX CUDA 版只有 Linux 有，所以模型在 WSL 跑，GPU 直接用 Windows 的 NVIDIA 驅動（不需 CUDA toolkit、不需 sudo）。
- **WSL2 `~/groove_venv`**：GrooVAE 需要舊版 TensorFlow，和 MRT2 的套件衝突，所以另開一個環境、只用 CPU。
- **Windows `venv`**：WSL 讀不到 USB MIDI、也不開視窗，所以 MIDI 橋接和視覺在 Windows 端跑，由 `band_mrt2.py` 自動啟動。

### WSL2 一次性安裝

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

`run_band.sh` 會設定：`MAGENTA_HOME`（共用 Windows 已下載的 checkpoint）、
`XLA_PYTHON_CLIENT_PREALLOCATE=false` 與 `XLA_PYTHON_CLIENT_ALLOCATOR=platform`（4 GB 顯卡不預先佔顯存）、`LD_LIBRARY_PATH`（libpulse）。

### WSL2 GrooVAE 環境（鼓手）

```bash
uv venv ~/groove_venv --python 3.10
source ~/groove_venv/bin/activate
uv pip install tensorflow==2.9.1 tensorflow-probability==0.17.0 tensorflow-datasets==4.6.0 tensorflow-metadata==1.9.0     protobuf==3.20.3 note-seq==0.0.3 "numpy<1.24" tf-slim
uv pip install --no-deps magenta==2.1.4   # 直接裝會因 llvmlite 建置失敗
mkdir -p ~/groovae && cd ~/groovae
curl -O https://storage.googleapis.com/magentadata/models/music_vae/checkpoints/groovae_2bar_add_closed_hh.tar
mkdir groovae_2bar_add_closed_hh && tar xf groovae_2bar_add_closed_hh.tar -C groovae_2bar_add_closed_hh
```

### Windows venv

```powershell
python -m venv venv
.\venv\Scripts\pip install mido python-rtmidi numpy PyOpenGL pyopengltk pillow
```

## 執行與演奏

1. 接上 MIDI 鍵盤，雙擊 `run_band.bat`（第一次啟動要編譯模型約 4 分鐘，之後有快取約 1 分半；視覺視窗會先出現等待畫面）。
2. **彈一個和弦**：根音和大小調決定調性，那一刻就是第一拍，音樂與光束開始。彈的音由 Windows 內建合成器立刻發出鋼琴聲（`PIANO_PORT`），樂團的回應約 0.6 秒後跟上。
3. 之後和弦進行會自己走（每小節從上一個和弦的常見去處挑一個）；**彈一個和弦**可以把進行導向那個和弦；**按住單音**時樂團跟著你彈的音。
4. **按最低鍵（C2）** 切換風格，在下一段（4 小節）開頭生效；開始前按則是選擇起始風格。
5. 視覺視窗：`Esc` 離開全螢幕、`F` 全螢幕、`Q` 結束；移動滑鼠會顯示切換風格的鍵。

結束時終端機會印出每幀生成時間（中位數 / 95% / 最慢）和斷音次數；RTX 3050 Laptop 上五個 MRT2 agent 約 37 ms/幀（預算 40 ms）、0 次斷音、延遲約 0.6 秒；鼓的取樣發聲每幀只要 0.1 ms。

主要設定都在 `band_mrt2.py` 開頭：`BPM`、`SWITCH_KEY`、`PIANO_PORT`、`STYLES`（各 agent 提示詞、hi-hat 密度）、`GAINS` / `PAN`、`NEXT_CHORD`（和弦進行）、`SAMPLES`（鼓音高對應的取樣）、`DRUM_LAG`（MRT2 的樂器比鼓點條件晚約 70 ms 出聲，鼓延後 2 幀對齊）。
鼓的規則（小鼓在 2、4 拍；大鼓比小鼓多一點、通常不跟小鼓同時；open hat 每 4 小節一次或偶爾跟大鼓一起）在 `Drummer.pattern`，
每 2 小節提前送給 `groove_server.py` 加人味；GrooVAE 沒回應時就直接打鼓機的 pattern。

## 注意事項

- **視覺必須用內顯**：和 RTX 3050 搶 GPU 會造成大量斷音。`run_band.bat` 會把 `venv` 的 python 設成「省電」GPU；若終端機出現「正在用 RTX 3050 畫圖」的警告，到 Windows 圖形設定手動改。
- **電源**：GPU 全速時筆電若供電不足會直接斷電。請用原廠 180 W 變壓器、電池先充飽；`run_band.bat` 不會改電源計畫，建議「高效能」計畫把 AC 下處理器最大狀態設為 99%（關 Turbo）。
- **斷電後 WSL 會卡住**（GPU 時脈卡在 210 MHz、Windows 連不到 WSL 的 port）：執行 `wsl --shutdown` 再重開。
- 模型檔放在 `%USERPROFILE%\Documents\Magenta\magenta-rt-v2`（`MAGENTA_HOME`），第一次執行會自動下載。
- JAX 的編譯結果快取在 WSL 的 `~/.cache/jax`；JAX 或模型升級後若啟動變慢或出錯，刪掉它重新編譯。
- 單獨測一個 agent：在 WSL 裡用 `run_band.sh` 相同的環境變數執行 `python test_mrt2.py`。
- 測跟拍與調性：`python band_mrt2.py --no-visual --record /tmp/rec.wav` 把各樂器分軌存成 12 聲道 wav（順序同 `AGENTS` 再加鼓）；檔案要寫在 WSL 本機硬碟，寫到 `/mnt/c` 會斷音。
