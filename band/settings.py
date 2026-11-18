"""所有可調的設定都在這裡；其他模組只讀不寫。"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 專案資料夾

# ===== 常改的設定 =====
BPM = 75                    # 兩種風格共用；75 BPM 的 16 分音符剛好 5 幀（40 ms）
MIDI_PORT = None            # MIDI 鍵盤名稱；None = 第一個找到的
PIANO_PORT = "Microsoft GS Wavetable Synth"  # Windows 端發出鋼琴聲的合成器（名稱開頭）；None = 彈的人不聽自己的琴聲
GROOVE_PYTHON = "~/groove_venv/bin/python"  # 鼓手 groove_server.py（GrooVAE）的環境
DRUM_LAG = 2                # 鼓延後幾幀：MRT2 的樂器比鼓點條件晚約 70 ms 出聲，鼓往後挪才對齊

AGENTS = ["rhodes", "guitar", "bass", "strings", "bell"]  # MRT2 生成的五個樂器；鼓另外由 Drummer 負責
STYLES = {  # 每種風格：hi-hat 每拍幾個、各 agent 的提示詞；prompt = 接在提示詞後的風格描述、mode = 強制的調式；
            # kits = 可用的鼓組（samples/ 底下的資料夾）；bass808 = 貝斯改由跟著大鼓的 808 取樣打
    "neo soul": {"hat_div": 2, "kits": ["neo soul", "rnb_2000s_soul_jazz", "band sound"],
                 "prompts": {"rhodes": "solo rhodes electric piano", "guitar": "solo electric guitar",
                             "bass": "solo electric bass", "strings": "cinematic strings", "bell": "digital bell"}},
    "memphis rap": {"hat_div": 4, "prompt": "dark and moody", "mode": "minor",  # 提示詞不寫 rap（會帶出鼓），一律小調
                    "kits": ["memphis rap", "trap", "modern trap"], "bass808": True,
                    "prompts": {"rhodes": "solo rhodes electric piano", "guitar": "solo electric guitar",
                                "bass": "deep sub bass",  # bass808 時這個 agent 被 808 取代
                                "strings": "cinematic strings", "bell": "digital bell"}},
}
GAINS = {"rhodes": 0.8, "guitar": 0.8, "bass": 1.0, "drums": 0.7, "strings": 0.8, "bell": 0.5}
PAN = {"rhodes": -0.4, "guitar": 0.4, "bass": 0.0, "drums": 0.0, "strings": 0.0, "bell": 0.2}  # −1 = 最左、+1 = 最右
MASTER = 0.85               # 整體音量；配合限制器，混音的失真約 −49 dB

# ===== 曲式 =====
PHRASE_BARS = 4             # 風格與編制在每段（4 小節）的開頭變，不在半途變
DROP_P, REJOIN_P = 0.3, 0.6  # 每段開頭：在場的樂器退出的機率、退出的樂器回來的機率（至少留 3 個）
BLENDS = (0.25, 0.5, 0.75)  # 切換風格前的最後一小節，提示詞逐拍混合過去，接得比較順
NEXT_CHORD = {  # 和弦進行自己走：每小節從上一個和弦的常見去處隨機挑一個（級數，0 = 主和弦）
    "minor": {0: [3, 5, 6, 2], 2: [5, 3, 0], 3: [4, 6, 0, 5], 4: [0, 5, 3], 5: [3, 6, 2, 0], 6: [0, 2, 5]},
    "major": {0: [5, 3, 1, 4], 1: [4, 6, 3], 3: [4, 0, 1, 5], 4: [0, 5, 3], 5: [3, 1, 4, 0], 6: [0, 4]},
}
CHORDS = {"major": (0, 4, 7), "minor": (0, 3, 7)}  # 判斷觀眾彈的和弦用

# ===== 鼓的取樣 =====
SAMPLES = {36: ("kick", 1.0), 38: ("snare", 1.0), 42: ("hihat", 0.7), 46: ("openhat", 0.7),  # GrooVAE 的鼓音高 → (取樣, 音量)
           45: ("rim", 0.8), 48: ("rim", 0.8), 50: ("rim", 0.8), 49: ("openhat", 0.6), 51: ("openhat", 0.5)}
# 鼓組 = samples/ 底下的一個資料夾：頂層有 kick.wav、snare.wav… 就直接用；否則照子資料夾名稱（Kicks、Snare、Hi hats…）
# 每次隨機挑一個取樣（規則在 band/drummer.py 的 KIT_ROLES）。808 一樣照資料夾找，自動量基頻、調到和弦根音
KIT_SEGMENTS = 8            # 同一風格每幾段（2 小節一段）換一次鼓組；換風格時一定換

# ===== 效能與連線 =====
TEMPERATURE = 1.1           # 取樣隨機度（MRT2 預設 1.3）：吉他更穩，貝斯不變
FRAMES_PER_CALL = 5         # 一次在 GPU 上算幾幀：5 個 agent 時 5 幀約 37 ms/幀（預算 40 ms）
QUEUE_FRAMES = 8            # 預先生成的幀數，太少會斷音、太多會增加延遲
CHORD_SETTLE = 0.08         # 第一個音按下後等多久再判斷和弦（和弦的音不會剛好同時按下）
VISUAL_TCP = ("127.0.0.1", 47601)  # 視覺程式 vj_tk.py（Windows 端）連到這裡接收資料
GROOVE_TCP = ("127.0.0.1", 47602)  # 鼓手 groove_server.py 聽這裡
